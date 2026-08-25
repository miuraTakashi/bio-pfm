"""
2D Fisher–KPP 反応–拡散（周期境界・拡散の半陰式スペクトル法）。

支配方程式:
  ∂u/∂t = D ∇²u + u(1-u)

時間積分: 反応項は陽的 Euler、拡散は `fhn_excitable_media_2d.py` と同型の
  半陰的 5 点ラプラシアン（rfft2 / irfft2, norm="ortho", ku = 1/RFFT2(kernel)/n）。

初期条件: 領域中心の円盤内で u=1、それ以外 u=0（中心刺激）。

積分ステップ数は環境変数 `FISHER_2D_N_STEPS`（既定 500）で変更可。GIF 用フレーム間隔は
`FISHER_2D_GIF_EVERY`（未設定時は約 25 フレームになるよう自動）。
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# --- atlas heatmap helpers ---
import sys
from pathlib import Path as _Path

for _d in _Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap


def build_diffusion_multiplier_rfft2(n: int, coeff: float) -> np.ndarray:
    """ku = 1 / RFFT2(kernel) / n （2D 半陰的拡散の乗数、周期境界）。"""
    kern = np.zeros((n, n))
    kern[0, 0] = 1.0 + 4.0 * coeff
    kern[[1, -1, 0, 0], [0, 0, 1, -1]] = -coeff
    return 1.0 / np.fft.rfft2(kern, norm="ortho") / n


def initial_center_disk(n: int, radius_cells: float) -> np.ndarray:
    """中心に円形刺激 u=1、外側 u=0。"""
    ii = np.arange(n, dtype=float)[:, None]
    jj = np.arange(n, dtype=float)[None, :]
    cx = 0.5 * (n - 1)
    cy = 0.5 * (n - 1)
    dist = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)
    u = np.where(dist <= radius_cells, 1.0, 0.0)
    return u.astype(np.float64)


def simulate_fisher_2d(
    n: int = 256,
    D: float = 2.0,
    dx: float = 1.0,
    dt: float = 0.08,
    n_steps: int = 500,
    stimulus_radius: float = 6.0,
    n_snapshots: int = 6,
    gif_every: int | None = None,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray], list[float]]:
    """半陰式拡散 + 陽的反応。スナップショット列と対応する時刻を返す。"""
    u = initial_center_disk(n, stimulus_radius)
    coeff = D * dt / (dx * dx)
    ku = build_diffusion_multiplier_rfft2(n, coeff)

    if n_snapshots < 2:
        save_steps = {0, n_steps}
    else:
        save_steps = {
            int(round(i * n_steps / (n_snapshots - 1))) for i in range(n_snapshots)
        }
    save_steps_sorted = sorted(save_steps)

    snaps: list[np.ndarray] = []
    times: list[float] = []
    gif_snaps: list[np.ndarray] = []
    gif_times: list[float] = []

    if gif_every is None:
        gif_every = max(1, n_steps // 25)
    gif_snaps.append(u.copy())
    gif_times.append(0.0)

    # t=0
    snaps.append(u.copy())
    times.append(0.0)

    for step in range(1, n_steps + 1):
        fu = u + dt * (u * (1.0 - u))
        fu = np.clip(fu, 0.0, 1.0)
        u = np.fft.irfft2(
            np.fft.rfft2(fu, norm="ortho") * ku,
            s=(n, n),
            norm="ortho",
        )
        u = np.clip(u, 0.0, 1.0)

        if step in save_steps_sorted:
            snaps.append(u.copy())
            times.append(step * dt)

        if step % gif_every == 0 or step == n_steps:
            gif_snaps.append(u.copy())
            gif_times.append(float(step) * dt)

    # 丸めで重複したフレームを除去（時刻の非減少を維持）
    out_snaps: list[np.ndarray] = []
    out_times: list[float] = []
    last_t = -1.0
    for s, t in zip(snaps, times):
        if not out_times or t > last_t + 1e-12:
            out_snaps.append(s)
            out_times.append(t)
            last_t = t
        else:
            out_snaps[-1] = s
            out_times[-1] = t
            last_t = t

    return out_snaps, out_times, gif_snaps, gif_times


def save_fisher_2d_gif(
    gif_snaps: list[np.ndarray],
    gif_times: list[float],
    out_path: Path,
    fps: int = 12,
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ap.atlas_imshow(
        ax,
        gif_snaps[0],
        heatmap="scalar",
        origin="lower",
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_axis_off()
    title = ax.set_title(f"t = {gif_times[0]:.2f}")

    def _update(i: int):
        im.set_data(gif_snaps[i])
        t = gif_times[i] if i < len(gif_times) else 0.0
        title.set_text(f"t = {t:.2f}")
        return im, title

    ani = FuncAnimation(
        fig, _update, frames=len(gif_snaps), interval=1000 // max(1, fps), blit=False
    )
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    n_steps = int(os.environ.get("FISHER_2D_N_STEPS", "500"))
    n_grid = int(os.environ.get("FISHER_2D_N", "256"))
    gif_every = os.environ.get("FISHER_2D_GIF_EVERY")
    gif_every_i = int(gif_every) if gif_every else None
    snaps, times, gif_snaps, gif_times = simulate_fisher_2d(
        n=n_grid, n_steps=n_steps, gif_every=gif_every_i
    )

    n_plot = len(snaps)
    fig, axes = plt.subplots(1, n_plot, figsize=(2.8 * n_plot, 3.2), squeeze=False)
    axes = axes[0]
    c_last = None
    for ax, u, t in zip(axes, snaps, times):
        c_last = ap.atlas_imshow(
            ax, u, heatmap="scalar", origin="lower", interpolation="bilinear", vmin=0, vmax=1
        )
        ax.set_title(f"t = {t:.2f}", fontsize=9)
        ax.axis("off")
    if c_last is not None:
        fig.colorbar(c_last, ax=axes[-1], fraction=0.046, pad=0.02, label="u")
    fig.suptitle(
        "2D Fisher–KPP: radial front from center stimulus "
        r"($\partial_t u = D\nabla^2 u + u(1-u)$, semi-implicit diffusion)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "results" / "fisher_2d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    gif_path = (Path(__file__).resolve().parent / "results" / "fisher_2d.gif")
    save_fisher_2d_gif(gif_snaps, gif_times, gif_path, fps=12)
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
