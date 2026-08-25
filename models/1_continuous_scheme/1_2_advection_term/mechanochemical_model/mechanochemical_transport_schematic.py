"""
メカノケミカルモデル — 輸送方程式が現れる理由を示す模式図。

出力: explanatory_figures/mechanochemical_transport_schematic.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

_SCRIPT = Path(__file__)
_OUT_PNG = _SCRIPT.parent / "explanatory_figures" / "mechanochemical_transport_schematic.png"


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


def _box(ax, xy, text, fc="#E8F4FD", ec="#2E6DA4", width=0.22, height=0.28):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, color="#444444", style="-|>", lw=1.8, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def draw_coupling_loop(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(A) なぜ輸送方程式が必要か — メカノケミカル連成ループ", fontsize=12, pad=8)

    boxes = [
        (0.04, 0.58, "密度場\n$n,\\;\\rho$"),
        (0.30, 0.58, "活性応力\n$\\sigma=n\\rho+\\gamma\\rho_{xx}$"),
        (0.56, 0.58, "力学方程式\n$\\mu u_{xxt}+u_{xx}+\\cdots=0$"),
        (0.78, 0.18, "移流速度\n$u_t=\\partial u/\\partial t$"),
    ]
    for x, y, txt in boxes:
        _box(ax, (x, y), txt)

    _arrow(ax, (0.26, 0.72), (0.30, 0.72))
    _arrow(ax, (0.52, 0.72), (0.56, 0.72))
    _arrow(ax, (0.78, 0.58), (0.86, 0.46), rad=-0.15)
    _arrow(ax, (0.86, 0.32), (0.26, 0.32), rad=0.0)
    _arrow(ax, (0.04, 0.46), (0.04, 0.58), rad=0.0)

    ax.text(
        0.50,
        0.32,
        "Poisson 求解\n$(u_t)_{xx}=\\mathrm{rhs}$",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF8E7", ec="#C49A00", lw=1.2),
    )

    highlight = FancyBboxPatch(
        (0.01, 0.08),
        0.30,
        0.38,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=2.0,
        edgecolor="#C0392B",
        facecolor="#FDEDEC",
        linestyle="--",
    )
    ax.add_patch(highlight)
    ax.text(
        0.16,
        0.27,
        "輸送方程式\n$n_t+(nu_t)_x=0$\n$\\rho_t+(\\rho u_t)_x=0$",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#922B21",
        fontweight="bold",
    )
    ax.annotate(
        "密度は $u_t$ と\n一体で動く",
        xy=(0.16, 0.46),
        xytext=(0.16, 0.50),
        ha="center",
        fontsize=9,
        color="#922B21",
        arrowprops=dict(arrowstyle="->", color="#922B21", lw=1.2),
    )

    ax.text(
        0.50,
        0.04,
        "力学で求まった $u_t$ が、$n,\\rho$ を空間上で運ぶ → 輸送方程式が連成系の一部として現れる",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )


def draw_advection_demo(ax) -> None:
    x = np.linspace(0, 4 * np.pi, 400)
    x0 = 1.2 * np.pi
    sigma = 0.45
    n0 = 1.0 + 0.55 * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    u_t = 0.35 + 0.08 * np.sin(x)
    dt = 0.55
    # 単純な Lagrangian シフト（説明用）
    x_shifted = x - u_t * dt
    n1 = 1.0 + 0.55 * np.exp(-0.5 * ((x_shifted - x0) / sigma) ** 2)

    ax.fill_between(x, 1.0, n0, color="#3498DB", alpha=0.35, label="$n(x,t)$")
    ax.plot(x, n0, color="#1F618D", lw=2.0)
    ax.plot(x, n1, color="#E74C3C", lw=2.0, ls="--", label="$n(x,t+\\Delta t)$（$u_t$ で移流）")
    ax.axhline(1.0, color="#777777", lw=0.8, ls=":")

    sample_x = np.array([1.4, 2.2, 3.0, 3.8, 4.6, 5.4]) * np.pi / 2.5
    for xs in sample_x:
        v = 0.35 + 0.08 * np.sin(xs)
        ax.annotate(
            "",
            xy=(xs + 0.35 * v, 1.02),
            xytext=(xs, 1.02),
            arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=1.6, mutation_scale=12),
        )
    ax.text(5.0, 1.06, "流れ", color="#1E8449", fontsize=10, family="sans-serif")
    ax.text(5.55, 1.06, r"$u_t(x)$", color="#1E8449", fontsize=10)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.85, 1.75)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$n$")
    ax.set_title("(B) 1D 移流のイメージ — 密度の塊が $u_t$ に随伴して移動", fontsize=12, pad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)


def draw_control_volume(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(C) 固定位置 $x$ における $n_t$ — フラックスの釣り合い", fontsize=12, pad=8)

    cv_x, cv_w = 4.2, 2.0
    cv_y, cv_h = 1.8, 2.4
    ax.add_patch(
        Rectangle(
            (cv_x, cv_y),
            cv_w,
            cv_h,
            linewidth=2,
            edgecolor="#2E6DA4",
            facecolor="#EBF5FB",
        )
    )
    ax.text(cv_x + cv_w / 2, cv_y + cv_h / 2, "微小\n体積", ha="center", va="center", fontsize=11)

    _arrow(ax, (1.2, 3.0), (3.95, 3.0), color="#27AE60", lw=2.2)
    ax.text(
        2.0,
        3.45,
        "流入",
        ha="center",
        fontsize=10,
        color="#1E8449",
        family="sans-serif",
    )
    ax.text(2.0, 2.65, r"$+(n u_t)$", ha="center", fontsize=10, color="#1E8449")

    _arrow(ax, (6.45, 3.0), (8.8, 3.0), color="#C0392B", lw=2.2)
    ax.text(
        7.6,
        3.45,
        "流出",
        ha="center",
        fontsize=10,
        color="#922B21",
        family="sans-serif",
    )
    ax.text(7.6, 2.65, r"$-(n u_t)$", ha="center", fontsize=10, color="#922B21")

    ax.axvline(cv_x + cv_w / 2, ymin=0.05, ymax=0.95, color="#555555", ls="--", lw=1.2)
    ax.text(cv_x + cv_w / 2, 0.55, "観測点 $x$（固定）", ha="center", fontsize=10)

    ax.text(
        5.0,
        0.9,
        r"$n_t = -(n u_t)_x \;\approx\; \dfrac{\Phi_{\mathrm{in}}-\Phi_{\mathrm{out}}}{\Delta x}$",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E7", ec="#C49A00"),
    )
    ax.text(
        5.0,
        0.35,
        r"$\Phi_{\mathrm{in}},\;\Phi_{\mathrm{out}}$：左端からの流入・右端への流出",
        ha="center",
        fontsize=9,
        family="sans-serif",
    )
    ax.text(
        5.0,
        5.2,
        "右辺が 0 の保存則 $\\;n_t+(nu_t)_x=0$ → 領域全体の $\\int n\\,dx$ は保存",
        ha="center",
        fontsize=10,
        color="#333333",
    )


def main() -> None:
    _setup_fonts()
    _OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 0.95], hspace=0.42)

    draw_coupling_loop(fig.add_subplot(gs[0]))

    gs_mid = gs[1].subgridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.25)
    draw_advection_demo(fig.add_subplot(gs_mid[0, 0]))
    draw_control_volume(fig.add_subplot(gs_mid[0, 1]))

    fig.suptitle(
        "メカノケミカルモデル — 輸送方程式 $n_t+(nu_t)_x=0$ が現れる理由",
        fontsize=14,
        y=0.98,
    )
    fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_OUT_PNG}")


if __name__ == "__main__":
    main()
