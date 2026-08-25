#!/usr/bin/env python3
"""
Delta–Notch 1D 周期鎖：相互作用項 alpha の線形安定性予測と数値検証。

線形解析（alpha_c = 2.5, k_* = pi）: delta_notch.md

出力:
  - delta_notch_1d_verification.png  … 臨界前後のプロファイル比較
  - delta_notch_1d_analysis.png      … sigma_max(k) と検証指標
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
import atlas_plotting as ap  # noqa: E402

# 局所係数（delta_notch.py と同一）
A, B, C, D = -2.0, -1.0, -1.0, -3.0
ALPHA_CRITICAL = 2.5


def jacobian_1d(k: float, alpha: float) -> np.ndarray:
    """2 近傍結合のヤコビアン J(k)。"""
    return np.array(
        [[A, B], [C + 2.0 * alpha * np.cos(k), D]],
        dtype=np.float64,
    )


def max_real_growth_rate(alpha: float, n_samples: int = 400) -> tuple[float, float]:
    """max_k Re lambda_max(k) とその k を返す。"""
    k_grid = np.linspace(0.0, np.pi, n_samples)
    sigma_max = -1e30
    k_star = 0.0
    for k in k_grid:
        vals = np.real(np.linalg.eigvals(jacobian_1d(k, alpha)))
        s = float(np.max(vals))
        if s > sigma_max:
            sigma_max = s
            k_star = float(k)
    return sigma_max, k_star


def sigma_max_curve(
    alpha: float,
    n_samples: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    k_grid = np.linspace(0.0, np.pi, n_samples)
    sigma = np.empty_like(k_grid)
    for i, k in enumerate(k_grid):
        sigma[i] = float(np.max(np.real(np.linalg.eigvals(jacobian_1d(k, alpha)))))
    return k_grid, sigma


def simulate_delta_notch_1d(
    n: int = 128,
    alpha: float = 5.0,
    dt: float = 0.01,
    n_steps: int = 3000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """1D 周期鎖（2 近傍 Delta 結合）を陽的オイラーで積分。"""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-0.1, 0.1, n)
    notch = rng.uniform(-0.1, 0.1, n)

    for _ in range(n_steps):
        neighbor_delta = np.roll(delta, 1) + np.roll(delta, -1)
        dd = A * delta + B * notch - delta**3
        dn = C * delta + D * notch - notch**3 + alpha * neighbor_delta
        delta = delta + dt * dd
        notch = notch + dt * dn

    return delta, notch


def pattern_metrics(delta: np.ndarray) -> dict[str, float]:
    """空間パターンの有無を判定する簡易指標。"""
    centered = delta - np.mean(delta)
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    if power.size > 1:
        power[0] = 0.0
        k_idx = int(np.argmax(power))
        k_dom = 2.0 * np.pi * float(np.fft.rfftfreq(delta.size, d=1.0)[k_idx])
    else:
        k_dom = 0.0

    staggered = float(np.mean((delta - np.roll(delta, -1)) ** 2))
    return {
        "std": float(np.std(delta)),
        "staggered_energy": staggered,
        "k_dom": k_dom,
    }


def run_verification() -> None:
    out_dir = Path(__file__).resolve().parent
    alphas_test = [2.0, 2.5, 3.5, 5.0]
    n = 128
    n_steps = 3000

    results: list[dict] = []
    profiles: dict[float, np.ndarray] = {}

    for alpha in alphas_test:
        delta, _ = simulate_delta_notch_1d(n=n, alpha=alpha, n_steps=n_steps, seed=42)
        metrics = pattern_metrics(delta)
        sigma_lin, k_star = max_real_growth_rate(alpha)
        results.append(
            {
                "alpha": alpha,
                "sigma_lin": sigma_lin,
                "k_star_lin": k_star,
                **metrics,
            }
        )
        profiles[alpha] = delta

    # --- verification figure: final profiles (common y-scale) ---
    fig_v, axes_v = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    x = np.arange(n)
    ymax = max(float(np.max(np.abs(profiles[a]))) for a in alphas_test)
    ylim = (-ymax * 1.05, ymax * 1.05)
    for ax, alpha in zip(axes_v.ravel(), alphas_test):
        d = profiles[alpha]
        r = next(r for r in results if r["alpha"] == alpha)
        pred = "pattern" if alpha > ALPHA_CRITICAL else (
            "critical" if alpha == ALPHA_CRITICAL else "uniform"
        )
        ax.plot(x, d, color="C0", lw=0.9)
        ax.set_ylim(ylim)
        ax.set_title(
            rf"$\alpha={alpha}$  ({pred});  "
            rf"$\sigma_{{\max}}^{{\mathrm{{lin}}}}={r['sigma_lin']:.3f}$,  "
            rf"std={r['std']:.3f}"
        )
        ax.set_ylabel(r"$\Delta$")
        ax.grid(alpha=0.3)
    for ax in axes_v[-1]:
        ax.set_xlabel("cell index")
    fig_v.suptitle(
        rf"Delta–Notch 1D verification ($\alpha_c={ALPHA_CRITICAL}$, $N={n}$, "
        rf"{n_steps} steps)"
    )
    fig_v.tight_layout()
    path_v = out_dir / "results/delta_notch_1d_verification.png"
    fig_v.savefig(path_v, dpi=150)
    plt.close(fig_v)

    # --- analysis figure: dispersion + bar chart ---
    fig_a, axes_a = plt.subplots(1, 2, figsize=(10, 4))

    ax_disp = axes_a[0]
    for alpha, ls in zip([2.0, 3.5, 5.0], ["-", "--", ":"]):
        k_g, sig = sigma_max_curve(alpha)
        ax_disp.plot(k_g, sig, ls=ls, label=rf"$\alpha={alpha}$")
    ax_disp.axhline(0.0, color="k", lw=0.6)
    ax_disp.axvline(np.pi, color="gray", ls=":", lw=0.8, label=r"$k=\pi$")
    ax_disp.set_xlabel(r"$k$")
    ax_disp.set_ylabel(r"$\max\mathrm{Re}\,\lambda$")
    ax_disp.set_title(r"Linear growth rate $\sigma_{\max}(k)$")
    ax_disp.legend(fontsize=8)
    ax_disp.grid(alpha=0.3)

    ax_bar = axes_a[1]
    labels = [rf"{r['alpha']}" for r in results]
    stds = [r["std"] for r in results]
    colors = [
        "C2" if r["alpha"] < ALPHA_CRITICAL else "C3" if r["alpha"] == ALPHA_CRITICAL else "C1"
        for r in results
    ]
    ax_bar.bar(labels, stds, color=colors)
    ax_bar.set_xlabel(r"$\alpha$")
    ax_bar.set_ylabel(r"std$(\Delta)$ at final time")
    ax_bar.set_title("Numerical pattern amplitude vs $\\alpha$")
    ax_bar.grid(axis="y", alpha=0.3)

    fig_a.suptitle(
        rf"1D linear prediction: instability for $\alpha>{ALPHA_CRITICAL}$ "
        rf"($k_*=\pi$, $\lambda_*=2$)"
    )
    fig_a.tight_layout()
    path_a = out_dir / "results/delta_notch_1d_analysis.png"
    fig_a.savefig(path_a, dpi=150)
    plt.close(fig_a)

    print(f"Saved {path_v}")
    print(f"Saved {path_a}")
    print(f"Analytical critical coupling: alpha_c = {ALPHA_CRITICAL}")
    print("-" * 72)
    print(f"{'alpha':>6} {'sigma_lin':>10} {'std_num':>10} {'k_dom':>8}  prediction")
    for r in results:
        pred = (
            "PATTERN" if r["alpha"] > ALPHA_CRITICAL
            else "CRITICAL" if r["alpha"] == ALPHA_CRITICAL
            else "UNIFORM"
        )
        ok = (
            (r["alpha"] < ALPHA_CRITICAL and r["std"] < 0.15)
            or (r["alpha"] > ALPHA_CRITICAL and r["std"] > 0.3)
            or (r["alpha"] == ALPHA_CRITICAL)
        )
        flag = "OK" if ok else "CHECK"
        print(
            f"{r['alpha']:6.1f} {r['sigma_lin']:10.4f} {r['std']:10.4f} "
            f"{r['k_dom']:8.3f}  {pred:8}  [{flag}]"
        )


def main() -> None:
    run_verification()


if __name__ == "__main__":
    main()
