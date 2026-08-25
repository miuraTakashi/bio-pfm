"""
Source–sink と SDD の定常勾配を L と 2L で比較し、
source–sink（純拡散・両端ディリクレ）のスケール不変性 u(x)/u_s = (L-x)/L を可視化する。

出力: morphogen_scale_invariance.png
"""
from __future__ import annotations

import os
import sys
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

from morphogen_gradient import steady_state as steady_state_sdd, simulate_morphogen
from morphogen_source_sink import SourceSinkParams, simulate_source_sink, steady_state as steady_state_ss


def numerical_steady_source_sink(D: float, L: float, u_source: float, n: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    p = SourceSinkParams(D=D, L=L, u_source=u_source, n=n, n_steps=n_steps)
    x, snaps = simulate_source_sink(p)
    return x, snaps[-1][1]


def numerical_steady_sdd(D: float, k: float, L: float, n: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    x, snaps = simulate_morphogen(D=D, k=k, L=L, n=n, dt=0.001, n_steps=n_steps)
    return x, snaps[-1][1]


def main() -> None:
    D, k, u_s = 1.0, 1.0, 1.0
    L0 = 10.0
    lengths = (L0, 2.0 * L0)
    n_grid = 120
    quick = os.environ.get("MORPHOGEN_SCALE_QUICK", "").strip().lower() in ("1", "true", "yes")
    n_steps_base = 8_000 if quick else 120_000

    use_numerical = not quick

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    row_titles = ("Source–sink (diffusion only)", "SDD ($-ku$ degradation)")
    col_titles = (rf"$L={L0:g}$", rf"$L={2*L0:g}$", r"Rescaled $\xi=x/L$")

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, fontsize=9)

    colors = ap.atlas_line_colors(2, heatmap="scalar")
    xi_ref = np.linspace(0.0, 1.0, 400)

    for col, L in enumerate(lengths):
        n_steps = int(n_steps_base * (L / L0) ** 2)
        x_fine = np.linspace(0.0, L, 400)
        u_ss = steady_state_ss(x_fine, L, u_s)
        u_sdd = steady_state_sdd(x_fine, D, k, L)

        if use_numerical:
            x_num, u_ss_num = numerical_steady_source_sink(D, L, u_s, n_grid, n_steps)
            _, u_sdd_num = numerical_steady_sdd(D, k, L, n_grid, n_steps)
        else:
            x_num, u_ss_num, u_sdd_num = None, None, None

        # --- source–sink: physical x ---
        ax = axes[0, col]
        ax.plot(x_fine, u_ss, color=colors[col], lw=2, label="analytical")
        if use_numerical:
            ax.plot(x_num, u_ss_num, "o", ms=2.5, color=colors[col], alpha=0.45, label="numerical")
        ax.set_xlim(0.0, L)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

        # --- SDD: physical x ---
        ax = axes[1, col]
        al = np.sqrt(k / D)
        ax.plot(x_fine, u_sdd, color=colors[col], lw=2, label=rf"analytical ($\alpha L={al*L:.1f}$)")
        if use_numerical:
            ax.plot(x_num, u_sdd_num, "o", ms=2.5, color=colors[col], alpha=0.45, label="numerical")
        ax.set_xlim(0.0, L)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

        # --- rescaled overlay (column 2 only, accumulate both L) ---
        xi = x_fine / L
        axes[0, 2].plot(
            xi,
            u_ss / u_s,
            color=colors[col],
            lw=2,
            label=rf"$L={L:g}$",
        )
        axes[1, 2].plot(
            xi,
            u_sdd,
            color=colors[col],
            lw=2,
            label=rf"$L={L:g}$, $\alpha L={al*L:.1f}$",
        )

    # Universal source–sink curve on ξ
    axes[0, 2].plot(xi_ref, 1.0 - xi_ref, "k--", lw=1.2, alpha=0.55, label=r"$1-\xi$ (universal)")
    axes[0, 2].set_xlim(0.0, 1.0)
    axes[0, 2].set_xlabel(r"$\xi = x/L$")
    axes[0, 2].set_ylabel(r"$u/u_s$")
    axes[0, 2].set_title(r"Rescaled $\xi=x/L$", fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=7)

    axes[1, 2].set_xlim(0.0, 1.0)
    axes[1, 2].set_xlabel(r"$\xi = x/L$")
    axes[1, 2].set_ylabel(r"$u$ (normalized at source)")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].text(
        0.55,
        0.55,
        r"$\alpha L$ changes with $L$" + "\n(no scale invariance)",
        transform=axes[1, 2].transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    fig.suptitle(
        "Scale invariance: source–sink vs SDD\n"
        rf"($D={D}$, $k={k}$, $u_s=1$; same $(D,k)$ when $L$ is doubled)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "results/morphogen_scale_invariance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Collapse metric for source–sink on common ξ grid
    u1 = steady_state_ss(lengths[0] * xi_ref, lengths[0], u_s) / u_s
    u2 = steady_state_ss(lengths[1] * xi_ref, lengths[1], u_s) / u_s
    collapse_err = float(np.max(np.abs(u1 - u2)))
    print(f"source–sink max |u(L1,xi)/us - u(L2,xi)/us| on xi grid = {collapse_err:.3e}")


if __name__ == "__main__":
    main()
