"""
Fisher 波の速度測定（Mathematica/1_continuous_scheme/1_1_diffusion_term/fisher_1d/FisherSpeed.nb から移植）。

Fisher-KPP 方程式:
  ∂u/∂t = D * ∂²u/∂x² + u*(1-u)

理論波速度: c* = 2*sqrt(D*f'(0)) = 2*sqrt(D)  (f'(0) = 1)

数値シミュレーションから波面位置を追跡して速度を計算し、
理論値と比較する。

Parameters: D=0.0001, dx=0.01, domain=1
"""
import numpy as np
import matplotlib.pyplot as plt
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

from matplotlib.colors import Normalize
from pathlib import Path


def simulate_fisher(
    D: float = 0.0001,
    dx: float = 0.01,
    n_steps_per_record: int = 25,
    n_records: int = 80,
    seed: int = 42,
) -> tuple:
    """Fisher 波シミュレーション。時系列スナップショットと波速を返す。"""
    n = int(round(1.0 / dx))
    # 安定 dt
    dt = 0.25 * dx**2 / D

    # 初期条件: 左端から 10 セルが u=1、それ以外 0
    u = np.zeros(n)
    u[:10] = 1.0
    u[10] = 0.5

    def diffusion(l: np.ndarray) -> np.ndarray:
        # ノイマン境界
        r = np.empty_like(l)
        r[1:-1] = l[2:] + l[:-2] - 2 * l[1:-1]
        r[0] = l[1] - l[0]
        r[-1] = l[-2] - l[-1]
        return r

    snapshots = []
    times = []
    wave_positions = []

    x = np.arange(n) * dx

    for rec in range(n_records):
        for _ in range(n_steps_per_record):
            u = u + dt * (D / dx**2 * diffusion(u) + u * (1 - u))
            u = np.clip(u, 0, 1)

        t = (rec + 1) * n_steps_per_record * dt
        times.append(t)
        snapshots.append(u.copy())

        # 波面位置: u = 0.5 の x 座標
        crossing = np.where((u[:-1] >= 0.5) & (u[1:] < 0.5))[0]
        if len(crossing) > 0:
            idx = crossing[0]
            pos = x[idx] + (0.5 - u[idx]) / (u[idx + 1] - u[idx]) * dx
            wave_positions.append((t, pos))

    return x, np.array(times), snapshots, wave_positions


def main():
    D = 0.0001
    x, times, snapshots, wave_pos = simulate_fisher(D=D)
    c_theory = 2 * np.sqrt(D)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # スナップショット（記録本数を増やし、全時刻を色で重ね描き）
    times = np.asarray(times)
    t_min, t_max = float(times.min()), float(times.max())
    norm_t = Normalize(vmin=t_min, vmax=t_max if t_max > t_min else t_min + 1.0)
    cmap = plt.get_cmap(ap.SCALAR_CMAP)
    for i, snap in enumerate(snapshots):
        axes[0].plot(
            x,
            snap,
            color=cmap(norm_t(times[i])),
            alpha=0.75,
            linewidth=1.0,
        )
    smap = ap.atlas_scalar_mappable(norm_t, heatmap="scalar")
    smap.set_array([])
    fig.colorbar(smap, ax=axes[0], fraction=0.046, pad=0.02, label="t")
    axes[0].set_title("Fisher Wave Time Evolution (dense samples)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].grid(alpha=0.3)

    # 波速推定
    if len(wave_pos) >= 4:
        t_arr = np.array([p[0] for p in wave_pos])
        x_arr = np.array([p[1] for p in wave_pos])
        # 線形フィット
        coeffs = np.polyfit(t_arr, x_arr, 1)
        c_num = coeffs[0]
        axes[1].scatter(t_arr, x_arr, s=12, label="wave front position")
        axes[1].plot(t_arr, np.polyval(coeffs, t_arr), "r-",
                     label=f"fitted speed c={c_num:.4f}")
        axes[1].axhline(y=0, color="k", linewidth=0.5)
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("wave front x position")
        axes[1].set_title(f"Wave speed: numerical={c_num:.4f}, theory={c_theory:.4f}")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        print(f"Numerical wave speed: {c_num:.5f}, theory 2*sqrt(D): {c_theory:.5f}")
    else:
        axes[1].text(0.5, 0.5, "insufficient wave front data", ha="center", va="center",
                     transform=axes[1].transAxes)

    fig.suptitle("Fisher-KPP Traveling Wave and Speed Measurement (D=0.0001)")
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "results" / "fisher_1d.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
