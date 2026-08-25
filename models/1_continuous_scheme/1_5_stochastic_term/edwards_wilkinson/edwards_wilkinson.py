"""
Edwards-Wilkinson 確率的表面成長（Mathematica/1_continuous_scheme/1_5_stochastic_term/edwards_wilkinson/Edwards-Wilkinson.nb から移植）。

方程式（連続 1+1 次元）:
  ∂h/∂t = ν ∂²h/∂x² + η(x, t),
  ⟨η(x,t) η(x',t')⟩ = σ² δ(x − x') δ(t − t').

周期境界 [0, L) 上の陽オイラー + 空間離散ラプラシアン。空間白色ノイズの離散化は
  Δh = ν (Δh) Δt + σ √(Δt / Δx) ξ_i^n,   ξ_i^n ∼ N(0,1) i.i.d.
（各格子・各ステップで独立）。これで連続極限の強度が σ² に一致する。

粗さ（ゼロモード除去） W(t) = √⟨(h − ⟨h⟩)²⟩ は無限系では W ∝ t^{1/4}（β=1/4）。
周期境界では t ≳ τ_sat ∼ L²/ν 程度で飽和し log-log で曲がるため、L を十分
取り、早期区間に参照線 ∝ t^{1/4} を重ねる。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_edwards_wilkinson(
    nu: float = 0.01,
    sigma: float = 0.08,
    L: float = 12.0,
    dx: float = 0.03,
    n_steps: int = 60_000,
    record_every: int = 400,
    n_realizations: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float, int]:
    """
    Returns
    -------
    widths : (n_realizations, n_snapshots)
    profiles : (n_realizations, n)
    dt, n, dx, L, record_every
    """
    rng = np.random.default_rng(seed)
    n = int(round(L / dx))
    if abs(n * dx - L) > 1e-9 * max(L, 1.0):
        raise ValueError("L must be approximately divisible by dx")

    # 陽式熱方程式の安定性: dt ≤ dx² / (2ν)（1D 中心差分）
    dt = dx**2 / (4.0 * nu)

    all_widths: list[list[float]] = []
    final_profiles: list[np.ndarray] = []

    for _ in range(n_realizations):
        h = np.zeros(n, dtype=float)
        widths: list[float] = [0.0]

        for step in range(n_steps):
            lap = (np.roll(h, 1) + np.roll(h, -1) - 2.0 * h) / (dx * dx)
            # 連続の空間白色ノイズに対応する離散強度（σ² δ δ）
            noise = sigma * np.sqrt(dt / dx) * rng.standard_normal(n)
            h = h + dt * nu * lap + noise

            if (step + 1) % record_every == 0:
                w = float(np.sqrt(np.mean((h - h.mean()) ** 2)))
                widths.append(w)

        all_widths.append(widths)
        final_profiles.append(h.copy())

    return (
        np.asarray(all_widths, dtype=float),
        np.asarray(final_profiles, dtype=float),
        float(dt),
        int(n),
        float(dx),
        float(L),
        int(record_every),
    )


def _fit_t_quarter_reference(t: np.ndarray, w: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    """早期区間 [i0:i1] で log W ≈ (1/4) log t + const を最小二乗し参照曲線を返す。"""
    tseg = t[i0:i1]
    wseg = w[i0:i1]
    m = (tseg > 0.0) & (wseg > 0.0)
    lt = np.log(tseg[m])
    lw = np.log(wseg[m])
    if lt.size < 2:
        c = np.log(w[i1]) - 0.25 * np.log(t[i1]) if i1 < len(w) else 0.0
    else:
        # lw ≈ 0.25 * lt + c
        c = float(np.mean(lw - 0.25 * lt))
    t_fine = np.geomspace(float(t[i0]), float(t[min(i1, len(t) - 1)]), num=40)
    return t_fine, np.exp(c) * (t_fine**0.25)


def main() -> None:
    nu, sigma = 0.01, 0.08
    widths, profiles, dt, n, dx, L, record_every = simulate_edwards_wilkinson(
        nu=nu, sigma=sigma
    )
    n_realizations = widths.shape[0]
    x = (np.arange(n, dtype=float) + 0.5) * dx

    t_arr = np.arange(widths.shape[1], dtype=float) * float(record_every) * dt
    mean_w = widths.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for profile in profiles[:3]:
        axes[0].plot(x, profile - profile.mean(), alpha=0.75, linewidth=0.8)
    axes[0].set_title("EW surface (final, mean-subtracted)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$h - \langle h\rangle$")
    axes[0].set_xlim(0.0, L)
    axes[0].grid(alpha=0.3)

    for w in widths:
        axes[1].plot(t_arr, w, alpha=0.25, color="steelblue", linewidth=0.8)
    axes[1].plot(t_arr, mean_w, "r-", linewidth=2.0, label="mean width")
    axes[1].set_title(r"Interface width $W(t)$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("W")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # log-log: t^{1/4} 参照（早期窓で振幅を合わせる）
    mask = (t_arr > 0.0) & (mean_w > 0.0)
    if np.sum(mask) >= 4:
        idx = np.flatnonzero(mask)
        i0 = int(idx[0])
        i1 = min(i0 + 6, int(idx[-1]))
        if i1 <= i0:
            i1 = int(idx[-1])
        t_fine, w_ref = _fit_t_quarter_reference(t_arr, mean_w, i0, i1)
        axes[2].loglog(t_arr[mask], mean_w[mask], "o-", color="steelblue", markersize=3, label="mean W")
        axes[2].loglog(t_fine, w_ref, "k--", linewidth=1.5, label=r"$\propto t^{1/4}$ (EW, fit early window)")
        axes[2].legend(fontsize=8)
    axes[2].set_title(r"$W(t)$ log-log — slope $1/4$ before finite-$L$ saturation")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("W")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        rf"Edwards–Wilkinson: $\partial_t h = \nu \partial_x^2 h + \eta$  "
        rf"($\nu={nu}$, $\sigma={sigma}$, $L={L}$, $\Delta x={dx}$, "
        rf"$N_{{\mathrm{{avg}}}}={n_realizations}$)"
    )
    plt.tight_layout()
    out_path = Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
