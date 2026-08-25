"""
1D source–sink モルフォゲン勾配（両端ディリクレ、拡散のみ・分解なし）

  ∂u/∂t = D u_xx,   x ∈ (0, L)

境界: u(0,t) = u_s  （ソース・固定濃度）,  u(L,t) = 0  （シンク）

定常解:
  u(x) = u_s (L - x) / L
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap


@dataclass(frozen=True)
class SourceSinkParams:
    D: float = 1.0
    L: float = 10.0
    u_source: float = 1.0
    n: int = 100
    dt: float = 0.001
    n_steps: int = 120_000


def steady_state(x: np.ndarray, L: float, u_source: float) -> np.ndarray:
    """u(0)=u_s, u(L)=0 の定常解（純拡散）。"""
    return u_source * (L - x) / L


def simulate_source_sink(
    params: SourceSinkParams,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    """陽的オイラー: 拡散のみ、両端ディリクレ。"""
    p = params
    dx = p.L / p.n
    x = np.linspace(0.0, p.L, p.n + 1)

    assert p.dt <= dx**2 / (2.0 * p.D), "CFL violation: dt is too large"

    u = np.zeros(p.n + 1)
    u[0] = p.u_source
    snapshots: list[tuple[float, np.ndarray]] = []
    save_at = {0, p.n_steps // 4, p.n_steps // 2, p.n_steps}

    for step in range(p.n_steps + 1):
        if step in save_at:
            snapshots.append((step * p.dt, u.copy()))
        if step == p.n_steps:
            break

        lap = np.zeros(p.n + 1)
        lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        u = u + p.dt * (p.D * lap)
        u[0] = p.u_source
        u[-1] = 0.0

    return x, snapshots


def main() -> None:
    p = SourceSinkParams()
    if os.environ.get("MORPHOGEN_SS_QUICK", "").strip().lower() in ("1", "true", "yes"):
        p = SourceSinkParams(n_steps=5_000)

    out_dir = Path(__file__).resolve().parent
    x, snapshots = simulate_source_sink(p)
    x_fine = np.linspace(0.0, p.L, 300)
    u_exact = steady_state(x_fine, p.L, p.u_source)
    u_steady_num = snapshots[-1][1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    colors = ap.atlas_line_colors(len(snapshots), heatmap="scalar")
    for (t, u), col in zip(snapshots, colors):
        ax.plot(x, u, color=col, label=f"t = {t:.2f}")
    ax.plot(x_fine, u_exact, "r--", linewidth=2, label="steady (analytical)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(r"Source–sink: $u(0)=u_s$, $u(L)=0$, diffusion only")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, u_steady_num, "C0", linewidth=1.8, label="numerical steady")
    ax.plot(x_fine, u_exact, "r--", linewidth=2, label=r"$u_s(L-x)/L$")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Steady profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Source–sink ($D={p.D}$, $L={p.L}$, $u_s={p.u_source}$, no degradation)",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "results/morphogen_source_sink.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    err = float(np.max(np.abs(steady_state(x, p.L, p.u_source) - u_steady_num)))
    print(f"max |u_num - u_analytic| at grid = {err:.3e}")


if __name__ == "__main__":
    main()
