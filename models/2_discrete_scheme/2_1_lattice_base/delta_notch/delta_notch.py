"""
Delta-Notch 側方抑制モデル（Mathematica/2_discrete_scheme/2_1_lattice_base/delta_notch/Delta-Notchによる側抑制.nb から移植）。

2D 格子上の細胞間 Delta-Notch シグナリング。各細胞は Delta と Notch を
発現し、隣接細胞の Delta がその細胞の Notch を活性化する（側方抑制）。

方程式（陽的オイラー）:
  dDelta/dt = a*delta + b*notch - delta^3
  dNotch/dt = c*delta + d*notch - notch^3 + α * Σ_{neighbors} delta_j

パラメータ: a=-2, b=-1, c=-1, d=-3, α=5, dt=0.01
初期条件: Δ=±0.1 一様乱数
格子サイズ: 10×10
"""
from pathlib import Path

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



def simulate_delta_notch(n: int = 10,
                          a: float = -2, b: float = -1,
                          c: float = -1, d: float = -3, alpha: float = 5.0,
                          dt: float = 0.01, n_steps: int = 1000, seed: int = 42) -> tuple:
    """Delta-Notch 側方抑制シミュレーション。最終の delta, notch 配列を返す。"""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-0.1, 0.1, (n, n))
    notch = rng.uniform(-0.1, 0.1, (n, n))

    for _ in range(n_steps):
        # 4 近傍の Delta の合計（周期境界）
        neighbor_delta = (np.roll(delta, 1, 0) + np.roll(delta, -1, 0) +
                          np.roll(delta, 1, 1) + np.roll(delta, -1, 1))
        dd = a * delta + b * notch - delta**3
        dn = c * delta + d * notch - notch**3 + alpha * neighbor_delta
        delta = delta + dt * dd
        notch = notch + dt * dn

    return delta, notch


def main():
    delta, notch = simulate_delta_notch()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ap.atlas_imshow(
        axes[0],
        delta,
        heatmap="scalar",
        origin="lower",
        interpolation="nearest",
        vmin=-1.5,
        vmax=1.5,
    )
    axes[0].set_title("Delta (final state)")
    plt.colorbar(im1, ax=axes[0])

    im2 = ap.atlas_imshow(
        axes[1],
        notch,
        heatmap="scalar",
        origin="lower",
        interpolation="nearest",
        vmin=-1.5,
        vmax=1.5,
    )
    axes[1].set_title("Notch (final state)")
    plt.colorbar(im2, ax=axes[1])

    fig.suptitle("Delta-Notch Lateral Inhibition Model (10×10 lattice)")
    plt.tight_layout()
    out = (Path(__file__).resolve().parent / "results" / "delta_notch.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
