"""
Gray-Scott モデル 2D — Pearson (1993) Science 図3の 12 分類（α–μ）を 4×3 で表示。
（Mathematica/1_continuous_scheme/1_1_diffusion_term/gray_scott_1d/1.1.11.GrayScott.nb から移植・拡張）。

u' = -u*v^2 + F*(1-u) + Du*Laplacian(u)
v' =  u*v^2 - (F+k)*v + Dv*Laplacian(v)

周期境界。各 (F, k) は Pearson のギリシャ文字クラスの代表点（主に Munafo, xmorphia の例）。
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Pearson (1993) Fig. 3 の 12 パターン（α–μ）を 4 行×3 列の並び（行優先）。
# (F, k) は Munafo https://mrob.com/pub/comp/xmorphia/pearson-classes.html の各型の代表例。
# η, ι は同文献で単独例が無い型のため、近傍クラスの代表値を使用。
PEARSON_1993_FIG3: list[tuple[str, float, float]] = [
    ("α", 0.010, 0.047),
    ("β", 0.014, 0.039),
    ("γ", 0.022, 0.051),
    ("δ", 0.030, 0.055),
    ("ε", 0.018, 0.055),
    ("ζ", 0.022, 0.061),
    ("η", 0.032, 0.058),
    ("θ", 0.030, 0.057),
    ("ι", 0.038, 0.061),
    ("κ", 0.050, 0.063),
    ("λ", 0.026, 0.061),
    ("μ", 0.046, 0.065),
]


def laplacian_2d(u: np.ndarray) -> np.ndarray:
    """周期境界での 2D 離散ラプラシアン（dx=1 前提、規格化なし）。"""
    return (
        np.roll(u, 1, 0)
        + np.roll(u, -1, 0)
        + np.roll(u, 1, 1)
        + np.roll(u, -1, 1)
        - 4 * u
    )


def simulate_gray_scott_2d(
    n: int = 128,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    k: float = 0.065,
    dx: float = 1.0,
    dt: float = 1.0,
    n_steps: int = 8000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gray-Scott 2D シミュレーション。最終 u, v を返す。"""
    rng = np.random.default_rng(seed)
    u = np.ones((n, n))
    v = np.zeros((n, n))

    cx, cy = n // 2, n // 2
    sq = n // 8
    u[cx - sq : cx + sq, cy - sq : cy + sq] = 0.5
    v[cx - sq : cx + sq, cy - sq : cy + sq] = 0.25
    u += rng.uniform(0, 0.02, (n, n))
    v += rng.uniform(0, 0.02, (n, n))

    inv_dx2 = 1.0 / (dx * dx)

    for _ in range(n_steps):
        lap_u = laplacian_2d(u) * inv_dx2
        lap_v = laplacian_2d(v) * inv_dx2
        uvv = u * v * v
        u_new = u + dt * (-uvv + F * (1 - u) + Du * lap_u)
        v_new = v + dt * (uvv - (F + k) * v + Dv * lap_v)
        u, v = u_new, v_new

    return u, v


def main() -> None:
    n = int(os.environ.get("GRAY_SCOTT_2D_N", "112"))
    n_steps = int(os.environ.get("GRAY_SCOTT_2D_N_STEPS", "6000"))

    fig, axes = plt.subplots(4, 3, figsize=(12.0, 14.0), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (name, F, k) in zip(axes_flat, PEARSON_1993_FIG3):
        u, _ = simulate_gray_scott_2d(n=n, F=F, k=k, n_steps=n_steps)
        im = ax.imshow(
            u,
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"{name}  (F={F:g}, k={k:g})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        "Gray–Scott 2D — Pearson (1993) Fig. 3: twelve classes (α–μ)",
        fontsize=12,
    )
    out_path = (Path(__file__).resolve().parent / "results" / "gray_scott_2d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
