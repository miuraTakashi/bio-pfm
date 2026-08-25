#!/usr/bin/env python3
"""
1D Turing 型反応–拡散（周期境界）。

Mathematica: `1.1.2.軟骨形成とTuringパターン.nb` の 1D 静領域部分に相当。
  f(p,q) = 0.6p - q - p^3,  g(p,q) = 1.5p - 2q
  拡散: 離散ラプラシアン + 陽的オイラー。

出力:
  - `results/turing_1d_numerical.png`（終端プロファイル、p/q のカイモグラフ）
  - `results/turing_1d_analysis.png`（線形安定性解析パネル）
  - `results/turing_1d_numerical_GiererMeinhardt.png` / `results/turing_1d_analysis_GiererMeinhardt.png`
    （Gierer–Meinhardt 型反応項・均一解まわりの線形解析）
"""
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
    jacobian: np.ndarray | None = None,
    k_max: float = 90.0,
    n_samples: int = 600,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    線形化 A(k)=J-k^2 D の最大実部成長率 sigma_max(k) を返す。
    戻り値: k_grid, sigma_grid, k_star, lambda_star
    """
    j = jacobian_origin() if jacobian is None else np.asarray(jacobian, dtype=np.float64)
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
    """
    最終プロファイルの 1D FFT から支配的波数と波長を推定。
    戻り値: k_dom, lambda_dom
    """
    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    if power.size <= 1:
        return 0.0, np.inf
    power[0] = 0.0
    idx = int(np.argmax(power))
    freq = float(np.fft.rfftfreq(centered.size, d=dx)[idx])  # cycles / unit-length
    k_dom = 2.0 * np.pi * freq
    lambda_dom = float((1.0 / freq) if freq > 1e-12 else np.inf)
    return k_dom, lambda_dom


def f_reaction(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.6 * p - q - p**3


def g_reaction(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 1.5 * p - 2.0 * q


# ---------------------------------------------------------------------------
# Gierer–Meinhardt（活性 p、抑制 q）— 均一正の定常解まわりの Turing 帯
# ---------------------------------------------------------------------------
#  f = rho0 + rho * p^2 / (q + eps) - mu * p,   g = sig * p^2 - nu * q
GM_RHO0 = 0.025
GM_RHO = 1.0
GM_MU = 1.0
GM_SIG = 1.0
GM_NU = 1.0
GM_EPS = 0.01
GM_DP = 0.0002
GM_DQ = 0.015
GM_DT = 0.005


def gm_homogeneous_ss() -> tuple[float, float]:
    """空間一様正の定常点 (p*, q*)（解析式）。"""
    p_star = (GM_RHO0 + GM_RHO * GM_NU / GM_SIG) / GM_MU
    q_star = GM_SIG * p_star * p_star / GM_NU
    return float(p_star), float(q_star)


def gm_jacobian_at_ss() -> np.ndarray:
    p_star, q_star = gm_homogeneous_ss()
    qe = q_star + GM_EPS
    fp = 2.0 * GM_RHO * p_star / qe - GM_MU
    fq = -GM_RHO * p_star * p_star / (qe * qe)
    gp = 2.0 * GM_SIG * p_star
    gq = -GM_NU
    return np.array([[fp, fq], [gp, gq]], dtype=np.float64)


def f_gm(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return GM_RHO0 + GM_RHO * p * p / (q + GM_EPS) - GM_MU * p


def g_gm(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return GM_SIG * p * p - GM_NU * q


def diffusion_1d(l: np.ndarray) -> np.ndarray:
    return np.roll(l, 1) + np.roll(l, -1) - 2.0 * l


def pq_after_dt_1d(
    p: np.ndarray,
    q: np.ndarray,
    dt: float,
    dp: float,
    dq: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    diff_p = dp * diffusion_1d(p) / (dx * dx)
    diff_q = dq * diffusion_1d(q) / (dx * dx)
    return (
        p + dt * (f_reaction(p, q) + diff_p),
        q + dt * (g_reaction(p, q) + diff_q),
    )


def simulate_1d(
    dx: float = 0.02,
    dt: float = 0.01,
    dp: float = 0.0002,
    dq: float = 0.01,
    simulation_length: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(1.0 / dx))
    steps_per_unit = int(round(1.0 / dt))

    rng = np.random.default_rng(seed)
    p = rng.random(n) * 0.01
    q = rng.random(n) * 0.01

    history_p = [p.copy()]
    history_q = [q.copy()]

    for _ in range(simulation_length):
        for _ in range(steps_per_unit):
            p, q = pq_after_dt_1d(p, q, dt, dp, dq, dx)
        history_p.append(p.copy())
        history_q.append(q.copy())

    return np.array(history_p), np.array(history_q)


def pq_after_dt_gm(
    p: np.ndarray,
    q: np.ndarray,
    dt: float,
    dp: float,
    dq: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    diff_p = dp * diffusion_1d(p) / (dx * dx)
    diff_q = dq * diffusion_1d(q) / (dx * dx)
    p_new = p + dt * (f_gm(p, q) + diff_p)
    q_new = q + dt * (g_gm(p, q) + diff_q)
    q_new = np.maximum(q_new, 1e-8)
    return p_new, q_new


def simulate_gm_1d(
    dx: float = 0.02,
    dt: float = GM_DT,
    dp: float = GM_DP,
    dq: float = GM_DQ,
    simulation_length: int = 80,
    seed: int = 43,
) -> tuple[np.ndarray, np.ndarray]:
    """均一解 + 小乱数から Gierer–Meinhardt を陽的オイラーで積分。"""
    n = int(round(1.0 / dx))
    steps_per_unit = int(round(1.0 / dt))

    p_star, q_star = gm_homogeneous_ss()
    rng = np.random.default_rng(seed)
    p = p_star + rng.normal(0.0, 0.02, n)
    q = q_star + rng.normal(0.0, 0.02, n)
    q = np.maximum(q, 1e-6)

    history_p = [p.copy()]
    history_q = [q.copy()]

    for _ in range(simulation_length):
        for _ in range(steps_per_unit):
            p, q = pq_after_dt_gm(p, q, dt, dp, dq, dx)
        history_p.append(p.copy())
        history_q.append(q.copy())

    return np.array(history_p), np.array(history_q)


def run_gierer_meinhardt_1d() -> None:
    dx = 0.02
    dt = GM_DT
    dp, dq = GM_DP, GM_DQ
    simulation_length = 80

    history_p, history_q = simulate_gm_1d(
        dx=dx,
        dt=dt,
        dp=dp,
        dq=dq,
        simulation_length=simulation_length,
    )
    n1d = history_p.shape[1]
    x1d = np.arange(n1d) * dx

    j_gm = gm_jacobian_at_ss()
    k_grid, sigma_grid, k_star, lambda_star = max_real_growth_rate_curve(
        dp=dp, dq=dq, jacobian=j_gm
    )
    k_dom, lambda_dom = estimate_dominant_wavelength_1d(history_p[-1], dx=dx)

    fig_num, axes_num = plt.subplots(1, 3, figsize=(12, 3.8))

    ax_profile = axes_num[0]
    ax_profile.plot(x1d, history_p[-1], label="p (activator)", color="C0")
    ax_profile.plot(x1d, history_q[-1], label="q (inhibitor)", color="C1")
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel("field value")
    ax_profile.set_title(
        f"GM 1D final (t={simulation_length})  "
        f"$\\lambda_*$={lambda_star:.3f},  "
        f"$\\lambda_{{num}}$={lambda_dom:.3f}"
    )
    ax_profile.legend(fontsize=8)
    ax_profile.grid(alpha=0.3)

    im_p = ap.atlas_imshow(
        axes_num[1],
        history_p,
        heatmap="scalar",
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, simulation_length],
    )
    axes_num[1].set_xlabel("x")
    axes_num[1].set_ylabel("t")
    axes_num[1].set_title("p kymograph (GM)")
    plt.colorbar(im_p, ax=axes_num[1], fraction=0.046)

    im_q = ap.atlas_imshow(
        axes_num[2],
        history_q,
        heatmap="scalar",
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, simulation_length],
    )
    axes_num[2].set_xlabel("x")
    axes_num[2].set_ylabel("t")
    axes_num[2].set_title("q kymograph (GM)")
    plt.colorbar(im_q, ax=axes_num[2], fraction=0.046)

    fig_num.suptitle(
        "Gierer–Meinhardt — 1D periodic domain (numerical)",
        fontsize=12,
    )
    fig_num.tight_layout()
    out_num = (Path(__file__).resolve().parent / "results" / "turing_1d_numerical_GiererMeinhardt.png")
    fig_num.savefig(out_num, dpi=150, bbox_inches="tight")
    plt.close(fig_num)

    fig_ana, ax_sigma = plt.subplots(1, 1, figsize=(6.8, 4.2))
    ax_sigma.plot(k_grid, sigma_grid, color="C2", lw=1.5, label=r"$\sigma_{\max}(k)$")
    ax_sigma.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_sigma.axvline(k_star, color="C3", lw=1.1, ls="--", label=rf"$k_*={k_star:.3f}$")
    ax_sigma.axvline(
        k_dom,
        color="C0",
        lw=1.0,
        ls="-.",
        label=rf"$k_{{num}}={k_dom:.3f}$",
    )
    ax_sigma.set_xlabel("k")
    ax_sigma.set_ylabel(r"max Re $\lambda(k)$")
    p_star, q_star = gm_homogeneous_ss()
    ax_sigma.set_title(
        rf"Linear stability at $(p^*,q^*)=({p_star:.3f},{q_star:.3f})$: "
        r"$A(k)=J-k^2D$"
    )
    ax_sigma.grid(alpha=0.3)
    ax_sigma.legend(fontsize=8, loc="best")

    fig_ana.suptitle(
        "Gierer–Meinhardt — 1D linear stability analysis",
        fontsize=12,
    )
    fig_ana.tight_layout()
    out_ana = (Path(__file__).resolve().parent / "results" / "turing_1d_analysis_GiererMeinhardt.png")
    fig_ana.savefig(out_ana, dpi=150, bbox_inches="tight")
    plt.close(fig_ana)
    print(f"Saved {out_num}")
    print(f"Saved {out_ana}")


def main() -> None:
    dx = 0.02
    dt = 0.01
    dp = 0.0002
    dq = 0.01
    simulation_length = 50

    history_p, history_q = simulate_1d(
        dx=dx,
        dt=dt,
        dp=dp,
        dq=dq,
        simulation_length=simulation_length,
    )
    n1d = history_p.shape[1]
    x1d = np.arange(n1d) * dx

    k_grid, sigma_grid, k_star, lambda_star = max_real_growth_rate_curve(dp=dp, dq=dq)
    k_dom, lambda_dom = estimate_dominant_wavelength_1d(history_p[-1], dx=dx)

    fig_num, axes_num = plt.subplots(1, 3, figsize=(12, 3.8))

    ax_profile = axes_num[0]
    ax_profile.plot(x1d, history_p[-1], label="p (activator)", color="C0")
    ax_profile.plot(x1d, history_q[-1], label="q (inhibitor)", color="C1")
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel("field value")
    ax_profile.set_title(
        f"1D final (t={simulation_length})  "
        f"$\\lambda_*$={lambda_star:.3f},  "
        f"$\\lambda_{{num}}$={lambda_dom:.3f}"
    )
    ax_profile.legend(fontsize=8)
    ax_profile.grid(alpha=0.3)

    im_p = ap.atlas_imshow(
        axes_num[1],
        history_p,
        heatmap="scalar",
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, simulation_length],
    )
    axes_num[1].set_xlabel("x")
    axes_num[1].set_ylabel("t")
    axes_num[1].set_title("p kymograph")
    plt.colorbar(im_p, ax=axes_num[1], fraction=0.046)

    im_q = ap.atlas_imshow(
        axes_num[2],
        history_q,
        heatmap="scalar",
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, simulation_length],
    )
    axes_num[2].set_xlabel("x")
    axes_num[2].set_ylabel("t")
    axes_num[2].set_title("q kymograph")
    plt.colorbar(im_q, ax=axes_num[2], fraction=0.046)

    fig_num.suptitle(
        "Turing pattern — 1D periodic domain (numerical)",
        fontsize=12,
    )
    fig_num.tight_layout()
    out_num = (Path(__file__).resolve().parent / "results" / "turing_1d_numerical.png")
    fig_num.savefig(out_num, dpi=150, bbox_inches="tight")
    plt.close(fig_num)

    fig_ana, ax_sigma = plt.subplots(1, 1, figsize=(6.8, 4.2))
    ax_sigma.plot(k_grid, sigma_grid, color="C2", lw=1.5, label=r"$\sigma_{\max}(k)$")
    ax_sigma.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_sigma.axvline(k_star, color="C3", lw=1.1, ls="--", label=rf"$k_*={k_star:.3f}$")
    ax_sigma.axvline(
        k_dom,
        color="C0",
        lw=1.0,
        ls="-.",
        label=rf"$k_{{num}}={k_dom:.3f}$",
    )
    ax_sigma.set_xlabel("k")
    ax_sigma.set_ylabel(r"max Re $\lambda(k)$")
    ax_sigma.set_title("Linear stability (A(k)=J-k²D)")
    ax_sigma.grid(alpha=0.3)
    ax_sigma.legend(fontsize=8, loc="best")

    fig_ana.suptitle(
        "Turing pattern — 1D linear stability analysis",
        fontsize=12,
    )
    fig_ana.tight_layout()
    out_ana = (Path(__file__).resolve().parent / "results" / "turing_1d_analysis.png")
    fig_ana.savefig(out_ana, dpi=150, bbox_inches="tight")
    plt.close(fig_ana)
    print(f"Saved {out_num}")
    print(f"Saved {out_ana}")

    run_gierer_meinhardt_1d()


if __name__ == "__main__":
    main()
