#!/usr/bin/env python3
"""
1D 反応–拡散 + 領域成長（格子間隔 dx が時間とともに増加）。

Mathematica: `1.1.2.軟骨形成とTuringパターン.nb` の領域成長部分に相当。
  Neumann 型離散拡散、
  f(p,q) = 0.6p - q + 1.2p^2 - p^3,  g(p,q) = 1.5p - 2q

外側ループ長 `simulation_length` は既定 **600**（旧 200 の 3 倍相当の実行量）。
短縮は環境変数 `TURING_DOMAIN_GROWTH_LENGTH`。

領域スケール（実装では `dx` の総増分に効く）の終端は `final_domain_size`。
既定では `original_domain_size=1` に対し `final_domain_size=8` とし、**格子間隔 `dx` が約 8 倍**になるようにする（旧 4 倍相当は `final=4`）。

拡散係数 `dp`, `dq`（$D_p$, $D_q$）は既定で旧値の **4 倍**（`0.0008`, `0.04`）。
上書きは `TURING_DOMAIN_GROWTH_DP` / `TURING_DOMAIN_GROWTH_DQ`。

陽解法の安定性のため、既定の時間刻み `dt` は **0.01 → 0.0025**（1/4）に合わせている。
上書きは `TURING_DOMAIN_GROWTH_DT`。

出力:
  - `results/turing_domain_growth_numerical.png`（p のカイモグラフ + 代表時刻のプロファイル）
  - `results/turing_domain_growth_analysis.png`（線形安定性と波長比較）
"""
import os
from pathlib import Path

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

import numpy as np


def jacobian_origin() -> np.ndarray:
    """反応項の (p,q)=(0,0) でのヤコビアン。"""
    return np.array([[0.6, -1.0], [1.5, -2.0]], dtype=np.float64)


def max_real_growth_rate_curve(
    dp: float,
    dq: float,
    *,
    k_max: float = 60.0,
    n_samples: int = 600,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    線形化 A(k)=J-k^2D の最大実部成長率 sigma_max(k) を返す。
    戻り値: k_grid, sigma_grid, k_star, lambda_star
    """
    j = jacobian_origin()
    dmat = np.diag([dp, dq]).astype(np.float64)
    k_grid = np.linspace(0.0, k_max, n_samples)
    sigma_grid = np.empty_like(k_grid)

    for i, k in enumerate(k_grid):
        a = j - (k * k) * dmat
        sigma_grid[i] = float(np.max(np.real(np.linalg.eigvals(a))))

    i_star = int(np.argmax(sigma_grid))
    k_star = float(k_grid[i_star])
    lambda_star = float((2.0 * np.pi / k_star) if k_star > 1e-12 else np.inf)
    return k_grid, sigma_grid, k_star, lambda_star


def estimate_dominant_wavelength_1d(field: np.ndarray, dx: float) -> tuple[float, float]:
    """1D FFT から支配的波数 k_dom と波長 lambda_dom を推定。"""
    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    if power.size <= 1:
        return 0.0, np.inf
    power[0] = 0.0
    idx = int(np.argmax(power))
    freq = float(np.fft.rfftfreq(centered.size, d=dx)[idx])  # cycles per physical length
    k_dom = 2.0 * np.pi * freq
    lambda_dom = float((1.0 / freq) if freq > 1e-12 else np.inf)
    return k_dom, lambda_dom


def f_reaction_growth(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.6 * p - q + 1.2 * p**2 - p**3


def g_reaction(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 1.5 * p - 2.0 * q


def diffusion_neumann(l: np.ndarray) -> np.ndarray:
    fwd = np.roll(l, -1) - l
    fwd[-1] = 0.0
    bwd = np.roll(l, 1) - l
    bwd[0] = 0.0
    return fwd + bwd


def simulate_domain_growth(
    dx0: float = 0.02,
    dt: float = 0.0025,
    dp: float = 0.0008,
    dq: float = 0.04,
    simulation_length: int = 600,
    original_domain_size: float = 1.0,
    final_domain_size: float = 8.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(original_domain_size / dx0))
    steps_per_unit = int(round(1.0 / dt))

    dx_change_per_dt = (
        (final_domain_size - original_domain_size) / simulation_length * dt / n
    )

    rng = np.random.default_rng(seed)
    p = rng.random(n) * 0.1
    q = rng.random(n) * 0.1

    dx = dx0
    history_p = [p.copy()]
    dx_history = [dx]

    for _ in range(simulation_length):
        for _ in range(steps_per_unit):
            dx += dx_change_per_dt
            diff_p = dp * diffusion_neumann(p) / (dx * dx)
            diff_q = dq * diffusion_neumann(q) / (dx * dx)
            p = p + dt * (f_reaction_growth(p, q) + diff_p)
            q = q + dt * (g_reaction(p, q) + diff_q)
        history_p.append(p.copy())
        dx_history.append(dx)

    return np.array(history_p), np.array(dx_history)


def main() -> None:
    sim_len = int(os.environ.get("TURING_DOMAIN_GROWTH_LENGTH", "600"))
    final_sz = float(os.environ.get("TURING_DOMAIN_GROWTH_FINAL_SIZE", "8.0"))
    dp = float(os.environ.get("TURING_DOMAIN_GROWTH_DP", "0.0008"))
    dq = float(os.environ.get("TURING_DOMAIN_GROWTH_DQ", "0.04"))
    dt = float(os.environ.get("TURING_DOMAIN_GROWTH_DT", "0.0025"))
    diff_scale_env = os.environ.get("TURING_DOMAIN_GROWTH_DIFFUSION_SCALES", "0.5,1.0,2.0")
    diff_scales = []
    for token in diff_scale_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = float(token)
        except ValueError:
            continue
        if val > 0.0:
            diff_scales.append(val)
    if not diff_scales:
        diff_scales = [1.0]
    if not any(np.isclose(s, 1.0) for s in diff_scales):
        diff_scales.append(1.0)
    diff_scales = sorted(set(diff_scales))
    hist_growth, dx_hist = simulate_domain_growth(
        simulation_length=sim_len,
        final_domain_size=final_sz,
        dt=dt,
        dp=dp,
        dq=dq,
    )
    n_growth = hist_growth.shape[1]
    t_max = hist_growth.shape[0] - 1
    k_grid, sigma_grid, k_star, lambda_star = max_real_growth_rate_curve(dp=dp, dq=dq)

    k_num_series = np.empty(t_max + 1, dtype=np.float64)
    lambda_num_series = np.empty(t_max + 1, dtype=np.float64)
    for tidx in range(t_max + 1):
        k_num_series[tidx], lambda_num_series[tidx] = estimate_dominant_wavelength_1d(
            hist_growth[tidx], dx=float(dx_hist[tidx])
        )

    fig_num, axes_num = plt.subplots(1, 4, figsize=(14.0, 3.8))
    ax_kymo = axes_num[0]
    axes_profile = list(axes_num[1:])

    im_g = ap.atlas_imshow(
        ax_kymo,
        hist_growth,
        heatmap="scalar",
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, float(t_max)],
    )
    ax_kymo.set_xlabel("grid index (normalised)")
    ax_kymo.set_ylabel("t")
    ax_kymo.set_title("Domain growth — p kymograph")
    plt.colorbar(im_g, ax=ax_kymo, fraction=0.046)

    for idx, tidx in enumerate((0, t_max // 2, t_max)):
        ax = axes_profile[idx]
        domain_len = dx_hist[tidx] * n_growth
        xg = np.linspace(0, domain_len, n_growth)
        ax.plot(xg, hist_growth[tidx], color="C0")
        ax.set_xlabel("x")
        lam_num = lambda_num_series[tidx]
        lam_txt = f"{lam_num:.3f}" if np.isfinite(lam_num) else "inf"
        ax.set_title(f"Growth t={tidx}  (L={domain_len:.1f}, λ_num={lam_txt})")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3)

    fig_num.suptitle(
        f"Turing pattern — 1D domain growth (Neumann, numerical), T_max={t_max}",
        fontsize=12,
    )
    fig_num.tight_layout()
    out_num = (Path(__file__).resolve().parent / "results" / "turing_domain_growth_numerical.png")
    fig_num.savefig(out_num, dpi=150, bbox_inches="tight")
    plt.close(fig_num)

    fig_ana, ax_sigma = plt.subplots(1, 1, figsize=(6.6, 4.2))
    for s in diff_scales:
        dp_s = dp * s
        dq_s = dq * s
        k_s, sigma_s, k_star_s, _ = max_real_growth_rate_curve(dp=dp_s, dq=dq_s)
        ax_sigma.plot(
            k_s,
            sigma_s,
            lw=1.4,
            label=rf"scale={s:g} ($D_p$={dp_s:.4g}, $D_q$={dq_s:.4g}, $k_*$={k_star_s:.3f})",
        )
    ax_sigma.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_sigma.set_xlabel("k")
    ax_sigma.set_ylabel(r"max Re $\lambda(k)$")
    ax_sigma.set_title("Dispersion relation vs diffusion coefficients")
    ax_sigma.set_xlim(0.0, 60.0)
    ax_sigma.grid(alpha=0.3)
    ax_sigma.legend(fontsize=8, loc="best")

    fig_ana.suptitle(
        f"Turing pattern — 1D domain growth (analysis), baseline λ*={lambda_star:.3f}",
        fontsize=12,
    )
    fig_ana.tight_layout()
    out_ana = (Path(__file__).resolve().parent / "results" / "turing_domain_growth_analysis.png")
    fig_ana.savefig(out_ana, dpi=150, bbox_inches="tight")
    plt.close(fig_ana)

    print(
        f"Saved {out_num}  (simulation_length={sim_len}, "
        f"final_domain_size={final_sz}, dt={dt}, dp={dp}, dq={dq}, "
        f"dx_end/dx0≈{dx_hist[-1] / dx_hist[0]:.2f})"
    )
    print(f"Saved {out_ana}")


if __name__ == "__main__":
    main()
