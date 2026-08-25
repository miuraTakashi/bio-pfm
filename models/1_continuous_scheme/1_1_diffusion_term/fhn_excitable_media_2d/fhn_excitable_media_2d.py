"""
興奮性媒質と螺旋波パターン — 2D FitzHugh-Nagumo + 拡散
（Mathematica/1_continuous_scheme/1_1_diffusion_term/fhn_excitable_media_1d/1.1.3.ExcitableMediaRungeKutta.nb から移植）。

方程式:
  ∂u/∂t = Du * ∇²u + u - u³/3 - v
  ∂v/∂t = ε * (u + β - γ*v)

時間積分: 反応項は陽的 Euler、u の拡散は phase_field_branching.py と同型の
  半陰的スペクトル法（rfft2 / irfft2, norm="ortho", ku = 1/RFFT2(kernel)/n）。

螺旋波を多数出すための調整:
  - 格子を広く (n=384)、十分長く積分
  - 初期条件: ホワイトノイズをガウシアンで帯域制限（相関長数格子）。ピクセル独立の
    厳密ホワイトノイズは拡散で一瞬で均一化されるため、ランダム性は保ちつつ局部の興奮シードが残る
  - ε≈0.1、Du をやや上げて大領域でも波同士が早く干渉しコアが増えるように

GIF: 環境変数 `FHN_EXCITABLE_MEDIA_2D_GIF_EVERY` でフレーム間隔（ステップ）を指定可。
  未設定時は n_steps//50 刻み（dt=0.06 の既定では終了時刻 600 付近まで約 50 フレーム）。
  `FHN_EXCITABLE_MEDIA_2D_GIF_FPS` で fps（既定 10）。
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
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

from pathlib import Path


def compute_rest_state(beta, gamma):
    """FHN 定常状態を Newton 法で求める。

    拡散なし平衡: v = u - u³/3 かつ v = (u + β)/γ より
      u³ - 3u(1 - 1/γ) + 3β/γ = 0
    """
    u = -1.0
    for _ in range(30):
        c = 1.0 - 1.0 / gamma
        f = u**3 - 3.0 * u * c + 3.0 * beta / gamma
        fp = 3.0 * u**2 - 3.0 * c
        if abs(fp) < 1e-14:
            break
        u -= f / fp
    v = (u + beta) / gamma
    return u, v


def build_diffusion_multiplier_rfft2(n: int, coeff: float) -> np.ndarray:
    """ku = 1 / RFFT2(kernel) / n  （分枝モデルと同じ陰的拡散の乗数）。"""
    kern = np.zeros((n, n))
    kern[0, 0] = 1.0 + 4.0 * coeff
    kern[[1, -1, 0, 0], [0, 0, 1, -1]] = -coeff
    return 1.0 / np.fft.rfft2(kern, norm="ortho") / n


def initial_bandlimited_white_noise(
    n: int,
    u_rest: float,
    v_rest: float,
    rng: np.random.Generator,
    *,
    blur_sigma: float,
    u_amplitude: float,
    v_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """標準正規ホワイトノイズを生成し、周期境界でガウシアン平滑化して相関長を与える。"""
    zu = rng.standard_normal((n, n))
    zv = rng.standard_normal((n, n))
    zu = gaussian_filter(zu, sigma=blur_sigma, mode="wrap")
    zv = gaussian_filter(zv, sigma=blur_sigma, mode="wrap")
    zu /= float(np.std(zu)) + 1e-12
    zv /= float(np.std(zv)) + 1e-12
    u = u_rest + u_amplitude * zu
    v = v_rest + v_amplitude * zv
    return u, v


def simulate_fhn_2d(
    n: int = 384,
    Du: float = 1.35,
    dx: float = 1.0,
    dt: float = 0.06,
    n_steps: int = 10000,
    eps: float = 0.1,
    beta: float = 0.7,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
    noise_blur_sigma: float = 4.5,
    noise_u_amplitude: float = 0.52,
    noise_v_amplitude: float = 0.34,
    n_snapshots: int = 5,
    gif_every: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray], list[float]]:
    """FitzHugh-Nagumo 2D（半陰的拡散 + 陽的 v）。

    戻り値: (PNG 用の (u,v) スナップショット列, GIF 用の u の列, GIF 用の時刻列)
    """
    if rng is None:
        rng = np.random.default_rng(7)
    u_rest, v_rest = compute_rest_state(beta, gamma)
    u, v = initial_bandlimited_white_noise(
        n,
        u_rest,
        v_rest,
        rng,
        blur_sigma=noise_blur_sigma,
        u_amplitude=noise_u_amplitude,
        v_amplitude=noise_v_amplitude,
    )

    coeff = Du * dt / (dx * dx)
    ku = build_diffusion_multiplier_rfft2(n, coeff)

    save_every = max(1, n_steps // max(1, n_snapshots))
    if gif_every is None:
        # n_steps//50 ステップごと（例: n=10000 → 200 ステップ、dt=0.06 で Δt≈12）
        gif_every = max(1, n_steps // 50)

    snapshots: list[tuple[np.ndarray, np.ndarray]] = [(u.copy(), v.copy())]
    gif_us: list[np.ndarray] = [u.copy()]
    gif_times: list[float] = [0.0]

    for step in range(1, n_steps + 1):
        fu = u + dt * (u - u ** 3 / 3.0 - v)
        v = v + dt * eps * (u + beta - gamma * v)
        u = np.fft.irfft2(
            np.fft.rfft2(fu, norm="ortho") * ku,
            s=(n, n),
            norm="ortho",
        )

        if step % save_every == 0:
            snapshots.append((u.copy(), v.copy()))

        if step % gif_every == 0 or step == n_steps:
            gif_us.append(u.copy())
            gif_times.append(float(step) * dt)

    return snapshots, gif_us, gif_times


def save_fhn_excitable_media_2d_gif(
    gif_us: list[np.ndarray],
    gif_times: list[float],
    out_path: Path,
    *,
    fps: int = 10,
) -> None:
    """螺旋が分かるよう u のみをアニメーション GIF に保存。"""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ap.atlas_imshow(
        ax,
        gif_us[0],
        heatmap="scalar",
        origin="lower",
        interpolation="bilinear",
        vmin=-2.0,
        vmax=2.0,
    )
    ax.set_axis_off()
    title = ax.set_title(f"u, t = {gif_times[0]:.1f}")

    def _update(i: int) -> tuple:
        im.set_data(gif_us[i])
        t = gif_times[i] if i < len(gif_times) else 0.0
        title.set_text(f"u, t = {t:.1f}")
        return (im, title)

    ani = FuncAnimation(
        fig,
        _update,
        frames=len(gif_us),
        interval=1000 // max(1, fps),
        blit=False,
    )
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    dt = 0.06
    n_steps = 10000
    save_every = n_steps // 5

    gif_every_env = os.environ.get("FHN_EXCITABLE_MEDIA_2D_GIF_EVERY")
    gif_every_i = int(gif_every_env) if gif_every_env else None
    gif_fps = int(os.environ.get("FHN_EXCITABLE_MEDIA_2D_GIF_FPS", "10"))

    print("Simulating FHN multi-spiral from band-limited white noise (rfft2)...")
    snapshots, gif_us, gif_times = simulate_fhn_2d(
        dt=dt, n_steps=n_steps, gif_every=gif_every_i
    )
    print(f"Done. {len(snapshots)} PNG snapshots, {len(gif_us)} GIF frames.")

    n_snap = len(snapshots)
    fig, axes = plt.subplots(2, n_snap, figsize=(3.2 * n_snap, 6.8))

    for i, (u, v) in enumerate(snapshots):
        t = i * save_every * dt
        im1 = ap.atlas_imshow(
            axes[0, i],
            u,
            heatmap="scalar",
            origin="lower",
            interpolation="bilinear",
            vmin=-2,
            vmax=2,
        )
        axes[0, i].set_title(f"u (t={t:.0f})", fontsize=9)
        axes[0, i].axis("off")

        im2 = ap.atlas_imshow(
            axes[1, i], v, heatmap="scalar", origin="lower", interpolation="bilinear"
        )
        axes[1, i].set_title(f"v (t={t:.0f})", fontsize=9)
        axes[1, i].axis("off")

    plt.colorbar(im1, ax=axes[0, -1], fraction=0.046, pad=0.02)
    plt.colorbar(im2, ax=axes[1, -1], fraction=0.046, pad=0.02)

    fig.suptitle(
        "2D Excitable Medium — Multi-spiral (FHN, n=384, band-limited white noise IC)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "results" / "fhn_excitable_media_2d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    gif_path = (Path(__file__).resolve().parent / "results" / "fhn_excitable_media_2d.gif")
    save_fhn_excitable_media_2d_gif(gif_us, gif_times, gif_path, fps=gif_fps)
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
