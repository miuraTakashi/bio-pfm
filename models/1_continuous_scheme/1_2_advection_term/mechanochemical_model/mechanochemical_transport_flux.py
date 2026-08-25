"""
保存則 ∂_t n + ∂_x(n ∂_t u) = 0 を、3 格子間の物質フラックスで説明する模式図。

出力: explanatory_figures/mechanochemical_transport_flux.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

_SCRIPT = Path(__file__)
_OUT_PNG = _SCRIPT.parent / "explanatory_figures" / "mechanochemical_transport_flux.png"


def _setup_fonts() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Hiragino Sans",
                "Yu Gothic",
                "Meiryo",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def _flux_arrow_at_interface(
    ax, x_interface, y, half_width, color, label, label_offset=(0, 0.14)
):
    x0 = x_interface - half_width
    x1 = x_interface + half_width
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.2,
            color=color,
        )
    )
    ax.text(
        x_interface + label_offset[0],
        y + label_offset[1],
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        color=color,
    )


def main() -> None:
    _setup_fonts()
    _OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # 格子中心の密度（説明用の具体値）
    n = np.array([1.2, 1.5, 0.9])
    u_t = 0.4  # 右向きの一定流れ（模式図用の ∂_t u）
    dx = 1.0
    flux_left = 0.5 * (n[0] + n[1]) * u_t   # 界面 i-1/2
    flux_right = 0.5 * (n[1] + n[2]) * u_t  # 界面 i+1/2
    dn_dt_center = -(flux_right - flux_left) / dx

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(-0.3, 3.35)
    ax.set_ylim(-0.55, 2.65)
    ax.axis("off")

    cell_w = 1.0
    base_y = 0.55
    colors = ["#D6EAF8", "#AED6F1", "#D6EAF8"]
    edge_colors = ["#5DADE2", "#2E86C1", "#5DADE2"]
    labels = [r"$i-1$", r"$i$", r"$i+1$"]
    n_labels = [r"$n_{i-1}$", r"$n_i$", r"$n_{i+1}$"]

    for k in range(3):
        x0 = k * cell_w
        height = 0.35 + 0.55 * n[k]
        lw = 2.8 if k == 1 else 1.6
        ax.add_patch(
            Rectangle(
                (x0, base_y),
                cell_w,
                height,
                linewidth=lw,
                edgecolor=edge_colors[k],
                facecolor=colors[k],
            )
        )
        top_y = base_y + height
        ax.text(
            x0 + 0.5 * cell_w,
            top_y + 0.12,
            n_labels[k],
            ha="center",
            va="bottom",
            fontsize=13,
        )
        ax.text(
            x0 + 0.5 * cell_w,
            base_y - 0.12,
            labels[k],
            ha="center",
            va="top",
            fontsize=12,
            color="#555555",
        )
        ax.text(
            x0 + 0.5 * cell_w,
            base_y + 0.32 * height,
            f"{n[k]:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            color="#1B4F72",
        )

    y_flux = 2.05
    arrow_half = 0.15
    # 格子境界 x = 1.0（界面 i-1/2）と x = 2.0（界面 i+1/2）の中心に矢印
    _flux_arrow_at_interface(
        ax,
        x_interface=1.0 * cell_w,
        y=y_flux,
        half_width=arrow_half,
        color="#27AE60",
        label=r"流入 $+(n\,\partial_t u)_{i-\frac{1}{2}}$",
    )
    _flux_arrow_at_interface(
        ax,
        x_interface=2.0 * cell_w,
        y=y_flux,
        half_width=arrow_half,
        color="#C0392B",
        label=r"流出 $-(n\,\partial_t u)_{i+\frac{1}{2}}$",
    )

    ax.text(
        1.0,
        y_flux + 0.42,
        "中心格子 $i$ を固定して見る",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#1B4F72",
    )

    sign = "+" if dn_dt_center >= 0 else ""
    formula = (
        rf"$\partial_t n \approx "
        rf"\dfrac{{(n\,\partial_t u)_{{i-\frac{{1}}{{2}}}}-(n\,\partial_t u)_{{i+\frac{{1}}{{2}}}}}}{{\Delta x}}"
        rf"= \dfrac{{{flux_left:.2f}-{flux_right:.2f}}}{{{dx:.1f}}}"
        rf"= {sign}{dn_dt_center:.2f}$"
    )
    ax.text(
        1.5,
        -0.28,
        formula,
        ha="center",
        va="center",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFF8E7", ec="#C49A00", lw=1.2),
    )

    fig.suptitle(
        r"保存則 $\partial_t n + \partial_x(n\,\partial_t u) = 0$：3 格子間の物質フラックス",
        fontsize=14,
        y=0.98,
    )
    ax.text(
        0.5,
        -0.48,
        r"格子 $i$ に入る量 $(n\,\partial_t u)_{i-\frac{1}{2}}$ − 出る量 $(n\,\partial_t u)_{i+\frac{1}{2}}$ = 0 なら $n_i$ は変化しない（$\partial_t n=0$）。差があれば $\partial_t n\neq 0$。",
        ha="center",
        va="top",
        fontsize=10.5,
        transform=ax.transAxes,
        color="#333333",
    )

    fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_OUT_PNG}")


if __name__ == "__main__":
    main()
