"""
モルフォゲン勾配モデル — 1D 拡散＋分解（SDD 境界点源版）。

拡散のみの source–sink（両端ディリクレ $u(0)=u_s$, $u(L)=0$）は morphogen_source_sink.py を参照。
Mathematica: 1.1.1.MorphogenGradient.nb から移植。

方程式:
  ∂u/∂t = D * ∂²u/∂x² - k*u
境界条件: u(0,t) = 1 (点源), u(L,t) = 0 (吸収端)
定常解: u(x) = sinh(k*(L-x)/sqrt(D)) / sinh(k*L/sqrt(D))
     ≈ exp(-sqrt(k/D) * x)  (長い系での近似)

陽的オイラー法で時間発展させ、定常状態への収束を確認する。
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



def steady_state(x: np.ndarray, D: float, k: float, L: float) -> np.ndarray:
    """解析的定常解 u(x) = sinh(sqrt(k/D)*(L-x)) / sinh(sqrt(k/D)*L)."""
    alpha = np.sqrt(k / D)
    return np.sinh(alpha * (L - x)) / np.sinh(alpha * L)


def simulate_morphogen(D: float = 1.0, k: float = 1.0, L: float = 10.0,
                       n: int = 100, dt: float = 0.001, n_steps: int = 5000):
    """陽的オイラーで拡散＋分解方程式を時間発展させる。"""
    dx = L / n
    x = np.linspace(0, L, n + 1)

    # 安定性チェック: dt <= dx^2 / (2D)
    assert dt <= dx**2 / (2 * D), "CFL violation: dt is too large"

    u = np.zeros(n + 1)
    u[0] = 1.0  # 点源

    snapshots = []
    save_at = {0, n_steps // 4, n_steps // 2, n_steps}

    for step in range(n_steps + 1):
        if step in save_at:
            snapshots.append((step * dt, u.copy()))

        if step == n_steps:
            break

        lap = np.zeros(n + 1)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        u = u + dt * (D * lap - k * u)
        u[0] = 1.0   # ディリクレ境界
        u[-1] = 0.0

    return x, snapshots


def main():
    D, k, L = 1.0, 1.0, 10.0
    x, snapshots = simulate_morphogen(D=D, k=k, L=L)
    x_exact = np.linspace(0, L, 300)
    u_exact = steady_state(x_exact, D, k, L)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ap.atlas_line_colors(len(snapshots), heatmap="scalar")
    for (t, u), col in zip(snapshots, colors):
        ax.plot(x, u, color=col, label=f"t = {t:.2f}")
    ax.plot(x_exact, u_exact, "r--", linewidth=2, label="steady state (analytical)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Morphogen Gradient (D=1, k=1, L=10)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = (Path(__file__).resolve().parent / "results" / "morphogen_gradient.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
