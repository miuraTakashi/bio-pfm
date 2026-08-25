"""Phase-field interdigitation: 2-field suture model in standard Allen–Cahn form.

（Mathematica/1_continuous_scheme/1_1_diffusion_term/phase_field_interdigitation/
1.1.8.SuturePhaseField13.nb から移植）

支配方程式（周期境界）:

  ∂u/∂t = (c₀²/τ) Δu
          + (1/(ε²τ)) u(1-u)(u - 1/2 + (ε/(√2 c₀)) F(v))

  ∂v/∂t = (1 - u) - v + d_v Δv

u は phase_field_standard_model と同型の Allen–Cahn 標準形。
v は拡散するシグナル場（骨形成因子相当）。駆動項:

  F(v) = g (v - v*),   既定 g = 0.1, v* = 0.44

係数対応（旧 Mathematica 記法 σ との関係）:

  σ = du = c₀²/τ     物理量（u の拡散係数；sharp interface limit → σ→0）
  ε, τ, c₀           standard_model 書き換えのフリーパラメータ（記法上のみ）
  数値計算・線形安定性は **物理式** を直接積分（ε に依存しない）:
    ∂u/∂t = du Δu + u(1-u)(u - 1/2 + (1/√(2du)) F(v)),  F(v)=g(v-v*)
  標準形 ε=1, τ=1, c₀=√σ と同値（ε を変えても物理は不変）

旧形:
  ∂u/∂t = u(1-u)(u - 1/2 - g(v* - v)/√(2σ)) + σ Δu

既定パラメータ: domain_size=10, dx=0.1, σ=0.0045 (c₀=√σ, τ=1),
d_v=0.1, dt=0.5, 初期条件は中央のランダム幅ギャップ (u=0) + 周囲骨 (u=1)。

Sections in this file (top to bottom):
  - defaults / standard-model u-equation helpers
  - 1D linear stability (flat central band; legacy optimal lamella via --lamella-stability)
  - 2D periodic solver + initial condition
  - GIF export, final-frame (u,v) scatter, and ``run_all_simulations`` CLI entrypoint
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from scipy import optimize as _opt_ls  # type: ignore[import]
except Exception:  # noqa: BLE001
    _opt_ls = None

# ---------------------------------------------------------------------------
# Defaults (keep in sync manually across solvers and export)
# ---------------------------------------------------------------------------

DEFAULT_SIGMA = 0.0045
DEFAULT_C0 = float(np.sqrt(DEFAULT_SIGMA))
DEFAULT_TAU = 1.0
DEFAULT_DV = 0.1
DEFAULT_DX = 0.1
DEFAULT_DY = 0.1
DEFAULT_DT = 0.5
DEFAULT_DOMAIN_SIZE = 10.0
DEFAULT_BAND_HALF_WIDTH = 1.0
DEFAULT_TOTAL_STEPS = 10_000
DEFAULT_V_STAR = 0.44
DEFAULT_FORCING_GAIN = 0.1
DEFAULT_SEED = 42
EXPORT_DPI = 160
GIF_FPS = 12

OUT_GIF_2D_U = "results/PhaseFieldInterdigitation_2D_u.gif"
OUT_GIF_2D_V = "results/PhaseFieldInterdigitation_2D_v.gif"
OUT_PNG_LINEAR_STABILITY = "results/PhaseFieldInterdigitation_linear_stability.png"
OUT_PNG_UV_SCATTER = "results/PhaseFieldInterdigitation_uv_scatter.png"
OUT_PNG_FINAL_FRAME = "results/PhaseFieldInterdigitation_final_frame.png"
OUT_PNG_DELTA_U = "results/PhaseFieldInterdigitation_delta_u.png"


# ---------------------------------------------------------------------------
# Standard Allen–Cahn helpers (same as phase_field_standard_model)
# ---------------------------------------------------------------------------


def _interdigitation_physical_u_params(c0: float, tau: float) -> tuple[float, float, float]:
    """Physical u-equation coefficients (Mathematica σ form; independent of standard-form ε).

    ∂u/∂t = du Δu + alpha·u(1-u)(u - 1/2 + beta·F(v)),
    with du=c₀²/τ, alpha=1, beta=1/√(2 du), F(v)=gain·(v-v*).
    """
    du = c0**2 / tau
    alpha = 1.0
    beta = 1.0 / np.sqrt(2.0 * du)
    return du, alpha, beta


def _standard_form_u_params(c0: float, tau: float, epsilon: float) -> tuple[float, float, float]:
    """Allen–Cahn standard-form coefficients (notation only; not used in this module's dynamics)."""
    du = c0**2 / tau
    alpha = 1.0 / (epsilon**2 * tau)
    beta = epsilon / (np.sqrt(2.0) * c0)
    return du, alpha, beta


def _allen_cahn_reaction(u: np.ndarray, alpha: float, beta: float, F: float | np.ndarray) -> np.ndarray:
    """Explicit reaction term (same functional form in 1D and 2D)."""
    return alpha * (u * (1.0 - u) * (u - 0.5 + beta * F))


def _evaluate_forcing(
    F: float | np.ndarray | Callable,
    u: np.ndarray,
    t: float,
    coords: np.ndarray | tuple[np.ndarray, np.ndarray],
):
    """Resolve forcing from scalar/array/callable and match shape with ``u``."""
    if callable(F):
        try:
            raw = F(u=u, t=t, coords=coords)
        except TypeError:
            try:
                raw = F(u, t, coords)
            except TypeError:
                raw = F(u, t)
    else:
        raw = F
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape != u.shape:
        arr = np.broadcast_to(arr, u.shape)
    return np.asarray(arr, dtype=float)


def _interdigitation_forcing_from_v(
    v: np.ndarray,
    *,
    c0: float,
    tau: float = DEFAULT_TAU,
    v_star: float = DEFAULT_V_STAR,
    gain: float = DEFAULT_FORCING_GAIN,
) -> np.ndarray:
    """F(v) for physical u-equation: beta·F = gain·(v-v*)/√(2 du) with beta=1/√(2 du)."""
    return gain * (v - v_star)


def _u_reaction_jacobian(
    u: np.ndarray,
    v: np.ndarray,
    *,
    alpha: float,
    beta: float,
    c0: float,
    tau: float,
    v_star: float = DEFAULT_V_STAR,
    gain: float = DEFAULT_FORCING_GAIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial derivatives of alpha*u*(1-u)*(u-0.5+beta*F(v)) w.r.t. u and v."""
    F = _interdigitation_forcing_from_v(v, c0=c0, tau=tau, v_star=v_star, gain=gain)
    h = u - 0.5 + beta * F
    dF_dv = gain
    ru = alpha * ((1.0 - 2.0 * u) * h + u * (1.0 - u))
    rv = alpha * u * (1.0 - u) * beta * dF_dv
    return ru, rv


def _cell_centred_extent(nx: int, ny: int, dx: float, dy: float):
    """Physical (left, right, bottom, top) for ``imshow(..., origin='lower'``."""
    x_coords = (np.arange(nx) - nx / 2.0) * dx
    y_coords = (np.arange(ny) - ny / 2.0) * dy
    return (
        x_coords[0] - dx / 2,
        x_coords[-1] + dx / 2,
        y_coords[0] - dy / 2,
        y_coords[-1] + dy / 2,
    )


def _save_every_for_steps(n_steps: int, target_frames: int = 300) -> int:
    return max(1, n_steps // max(1, target_frames))


def _pseudotime_dt_default_sigma(du: float, *, du_ref: float = DEFAULT_SIGMA, dt_ref: float = 0.05) -> float:
    """Pseudotime step for band/lamella relaxation; scale with physical diffusion du=σ."""
    if du <= 0.0:
        raise ValueError("du must be positive.")
    if du < 0.002:
        return 0.02
    ratio = du / du_ref
    return dt_ref * max(ratio, 0.2)


def _default_sigma_sweep_values(*, quick: bool = False) -> tuple[float, ...]:
    """Physical σ values for sharp-interface linear-stability sweeps (σ→0)."""
    if quick:
        return (0.0045, 0.002)
    return (0.0045, 0.003, 0.002, 0.001)


def _resolve_c0_tau(
    *,
    c0: float | None,
    tau: float,
    sigma: float | None,
) -> tuple[float, float]:
    """Resolve (c0, tau); ``sigma`` is physical du with tau=1 (sets c0=sqrt(sigma))."""
    if sigma is not None:
        if c0 is not None:
            raise ValueError("Specify either --c0 or --sigma, not both.")
        return float(np.sqrt(sigma)), 1.0
    if c0 is None:
        c0 = DEFAULT_C0
    return float(c0), float(tau)


# ---------------------------------------------------------------------------
# 1D: linear stability helpers
# ---------------------------------------------------------------------------


def _laplacian_periodic_matrix(n: int, dx: float) -> np.ndarray:
    h2 = dx * dx
    main = (-2.0 / h2) * np.ones(n, dtype=float)
    off = (1.0 / h2) * np.ones(n - 1, dtype=float)
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    lap[0, -1] = lap[-1, 0] = 1.0 / h2
    return lap


def _v_interfaces_binary_stripe(
    W: float, G: float, dv: float, *, n_min: int = 500
) -> tuple[float, float]:
    if W <= 0.0 or G <= 0.0:
        return float("nan"), float("nan")
    period = W + G
    n = max(n_min, int(np.ceil(period / 0.02)))
    dx = period / n
    x = np.arange(n, dtype=float) * dx
    u_bin = (x < W).astype(float)
    source = 1.0 - u_bin
    main = (-2.0 * dv / (dx * dx) - 1.0) * np.ones(n, dtype=float)
    off = (dv / (dx * dx)) * np.ones(n - 1, dtype=float)
    a = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    a[0, -1] = dv / (dx * dx)
    a[-1, 0] = dv / (dx * dx)
    v = np.linalg.solve(a, -source)
    i1 = int(np.floor(W / dx)) % n
    i2 = (i1 + 1) % n
    alpha_interp = (W - i1 * dx) / dx
    v_if1 = float((1.0 - alpha_interp) * v[i1] + alpha_interp * v[i2])
    v_if0 = float(v[0])
    return v_if0, v_if1


def optimal_lamella_widths_v_balance(
    dv: float,
    v_star: float = DEFAULT_V_STAR,
    *,
    w_bounds: tuple[float, float] = (0.3, 8.0),
    g_bounds: tuple[float, float] = (0.3, 8.0),
) -> tuple[float, float, bool]:
    """Bone width W and gap G with v=v* at both interfaces (binary stripe)."""
    if _opt_ls is None:
        return 2.0, 1.5, False

    def eq(z: np.ndarray) -> np.ndarray:
        w1, w0 = float(z[0]), float(z[1])
        if w1 <= 0 or w0 <= 0:
            return np.array([1e3, 1e3], dtype=float)
        v0, v1 = _v_interfaces_binary_stripe(w1, w0, dv)
        return np.array([v0 - v_star, v1 - v_star], dtype=float)

    best = None
    for g in [(1.5, 1.0), (2.0, 1.2), (2.5, 1.0), (3.0, 1.5), (1.0, 2.0), (2.0, 2.0)]:
        sol = _opt_ls.root(eq, np.array(g, dtype=float), method="hybr")
        if sol.success and w_bounds[0] < sol.x[0] < w_bounds[1] and g_bounds[0] < sol.x[1] < g_bounds[1]:
            res = float(np.linalg.norm(eq(sol.x)))
            if best is None or res < best[0]:
                best = (res, float(sol.x[0]), float(sol.x[1]))
    if best is None:
        return 2.0, 1.5, False
    return best[1], best[2], True


def pseudotime_relax_periodic_1d(
    c0: float,
    tau: float,
    dv: float,
    W: float,
    G: float,
    n: int,
    *,
    dt: float | None = None,
    max_steps: int = 500_000,
    tol: float = 1e-9,
    v_star: float = DEFAULT_V_STAR,
    gain: float = DEFAULT_FORCING_GAIN,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Relax (u,v) to steady state on periodic domain P=W+G (fftfreq semi-implicit)."""
    p = W + G
    dx = p / n
    x = np.arange(n, dtype=float) * dx
    u = (x < W).astype(float)
    v = np.zeros(n, dtype=float)
    du, alpha, beta = _interdigitation_physical_u_params(c0, tau)
    if dt is None:
        dt = _pseudotime_dt_default_sigma(du)

    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    denom_u = 1.0 + dt * du * (k**2)
    denom_v = 1.0 + dt * dv * (k**2)

    for _ in range(max_steps):
        F_eval = _interdigitation_forcing_from_v(v, c0=c0, tau=tau, v_star=v_star, gain=gain)
        fu = u + dt * _allen_cahn_reaction(u, alpha, beta, F_eval)
        gv = v + dt * (1.0 - u - v)
        u_new = np.real(np.fft.ifft(np.fft.fft(fu) / denom_u))
        v_new = np.real(np.fft.ifft(np.fft.fft(gv) / denom_v))
        if max(float(np.max(np.abs(u_new - u))), float(np.max(np.abs(v_new - v)))) < tol:
            return u_new, v_new, dx, True
        u, v = u_new, v_new
    return u, v, dx, False


def _build_band_initial_1d(
    n: int,
    dx: float,
    *,
    domain_size: float = DEFAULT_DOMAIN_SIZE,
    band_half_width: float = DEFAULT_BAND_HALF_WIDTH,
) -> np.ndarray:
    """Flat central band u=0, |y - L/2| < w; bone u=1 elsewhere (matches 2D limit without noise)."""
    y = np.arange(n, dtype=float) * dx
    u = np.ones(n, dtype=float)
    u[np.abs(y - domain_size / 2.0) < band_half_width] = 0.0
    return u


def pseudotime_relax_band_1d(
    c0: float,
    tau: float,
    dv: float,
    n: int,
    *,
    domain_size: float = DEFAULT_DOMAIN_SIZE,
    band_half_width: float = DEFAULT_BAND_HALF_WIDTH,
    dt: float | None = None,
    max_steps: int = 500_000,
    tol: float = 1e-9,
    v_star: float = DEFAULT_V_STAR,
    gain: float = DEFAULT_FORCING_GAIN,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Relax (u,v) to steady state on periodic domain L (flat central band IC)."""
    dx = domain_size / n
    u = _build_band_initial_1d(n, dx, domain_size=domain_size, band_half_width=band_half_width)
    v = np.zeros(n, dtype=float)
    du, alpha, beta = _interdigitation_physical_u_params(c0, tau)
    if dt is None:
        dt = _pseudotime_dt_default_sigma(du)

    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    denom_u = 1.0 + dt * du * (k**2)
    denom_v = 1.0 + dt * dv * (k**2)

    for _ in range(max_steps):
        F_eval = _interdigitation_forcing_from_v(v, c0=c0, tau=tau, v_star=v_star, gain=gain)
        fu = u + dt * _allen_cahn_reaction(u, alpha, beta, F_eval)
        gv = v + dt * (1.0 - u - v)
        u_new = np.real(np.fft.ifft(np.fft.fft(fu) / denom_u))
        v_new = np.real(np.fft.ifft(np.fft.fft(gv) / denom_v))
        if max(float(np.max(np.abs(u_new - u))), float(np.max(np.abs(v_new - v)))) < tol:
            return u_new, v_new, dx, True
        u, v = u_new, v_new
    return u, v, dx, False


def _reflection_perm_center(n: int, domain_size: float, dx: float) -> np.ndarray:
    """Mirror permutation about domain centre y = L/2 (band symmetry axis)."""
    ic = int(round((domain_size / 2.0) / dx)) % n
    idx = np.arange(n, dtype=int)
    return (2 * ic - idx) % n


def _reflection_perm_gap_mid(n: int, W: float, G: float, dx: float) -> np.ndarray:
    xc = W + 0.5 * G
    ic = int(round(xc / dx)) % n
    idx = np.arange(n, dtype=int)
    return (2 * ic - idx) % n


def _block_reflection_matrix(n: int, perm: np.ndarray) -> np.ndarray:
    p = np.zeros((n, n), dtype=float)
    p[np.arange(n), perm] = 1.0
    return np.block([[p, np.zeros((n, n))], [np.zeros((n, n)), p]])


def max_Re_lambda_inphase_for_k(
    u0: np.ndarray,
    v0: np.ndarray,
    c0: float,
    tau: float,
    dv: float,
    dx: float,
    k: float,
    reflection_perm: np.ndarray,
    *,
    even_tol: float = 0.18,
    v_star: float = DEFAULT_V_STAR,
    gain: float = DEFAULT_FORCING_GAIN,
) -> tuple[float, float]:
    """Max Re lambda for in-phase (mirror-even) modes and for all modes."""
    n = u0.size
    du, alpha, beta = _interdigitation_physical_u_params(c0, tau)
    lap = _laplacian_periodic_matrix(n, dx)
    ru0, rv0 = _u_reaction_jacobian(
        u0, v0, alpha=alpha, beta=beta, c0=c0, tau=tau, v_star=v_star, gain=gain
    )
    r_big = _block_reflection_matrix(n, reflection_perm)
    k2 = float(k * k)
    i_n = np.eye(n, dtype=float)
    a_uu = du * (lap - k2 * i_n) + np.diag(ru0)
    a_uv = np.diag(rv0)
    a_vu = -i_n
    a_vv = dv * (lap - k2 * i_n) - i_n
    big = np.block([[a_uu, a_uv], [a_vu, a_vv]])
    lam, vecs = np.linalg.eig(big)
    order = np.argsort(-lam.real)
    lam = lam[order]
    vecs = vecs[:, order]
    lam_all_max = float(np.max(lam.real))
    best_even = -1e100
    for j in range(min(big.shape[0], 80)):
        w = vecs[:, j]
        if float(np.linalg.norm(w)) < 1e-14:
            continue
        wn = w / np.linalg.norm(w)
        if float(np.linalg.norm(wn - r_big @ wn)) < even_tol:
            best_even = max(best_even, float(lam[j].real))
    if best_even < -1e90:
        best_even = lam_all_max
    return best_even, lam_all_max


def dispersion_inphase_branch(
    u0: np.ndarray,
    v0: np.ndarray,
    c0: float,
    tau: float,
    dv: float,
    dx: float,
    reflection_perm: np.ndarray,
    k_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inc = np.empty_like(k_vals, dtype=float)
    allm = np.empty_like(k_vals, dtype=float)
    for i, k in enumerate(k_vals):
        ie, ia = max_Re_lambda_inphase_for_k(
            u0, v0, c0, tau, dv, dx, float(k), reflection_perm
        )
        inc[i], allm[i] = ie, ia
    return inc, allm


def plot_interdigitation_linear_stability(
    out_path: Path,
    *,
    c0: float = DEFAULT_C0,
    tau: float = DEFAULT_TAU,
    dv: float = DEFAULT_DV,
    sigma_list: tuple[float, ...] | None = None,
    domain_size: float = DEFAULT_DOMAIN_SIZE,
    band_half_width: float = DEFAULT_BAND_HALF_WIDTH,
    lamella_stability: bool = False,
    k_max: float = 10.0,
    nk: int = 91,
    n_grid: int | None = None,
) -> None:
    """Plot in-phase max Re lambda(k) for flat central band (default) or optimal lamella.

    Uses the physical u-equation (σ only); standard-form ε does not enter stability.
    """
    if lamella_stability:
        plot_lamella_linear_stability(
            out_path,
            c0=c0,
            tau=tau,
            dv=dv,
            sigma_list=sigma_list,
            k_max=k_max,
            nk=nk,
            n_grid=n_grid or 160,
        )
        return

    if sigma_list is None:
        sigma_list = (c0**2 / tau,)
    if n_grid is None:
        n_grid = max(100, int(round(domain_size / DEFAULT_DX)))

    k_vals = np.linspace(0.0, k_max, nk, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    summary: list[str] = []
    for sig in sigma_list:
        c0_run = float(np.sqrt(sig * tau))
        dt_use = _pseudotime_dt_default_sigma(sig)
        u0, v0, dx, ok_r = pseudotime_relax_band_1d(
            c0_run,
            tau,
            dv,
            n_grid,
            domain_size=domain_size,
            band_half_width=band_half_width,
            dt=dt_use,
        )
        if not ok_r:
            summary.append(rf"$\sigma$={sig:g}: relax failed")
            continue
        perm = _reflection_perm_center(n_grid, domain_size, dx)
        inc, _allm = dispersion_inphase_branch(
            u0, v0, c0_run, tau, dv, dx, perm, k_vals
        )
        i_star = int(np.nanargmax(inc))
        k_star = float(k_vals[i_star])
        lam_star = float(inc[i_star])
        wave = float(2.0 * np.pi / k_star) if k_star > 1e-8 else float("inf")
        ax.plot(
            k_vals,
            inc,
            lw=1.4,
            label=rf"$\sigma$={sig:g}, max$\,\mathrm{{Re}}\,\lambda$={lam_star:.3g}",
        )
        stab = "unstable" if lam_star > 0 else "stable"
        wave_str = f"{wave:.3g}" if np.isfinite(wave) else "inf"
        summary.append(
            rf"$\sigma$={sig:g}: in-phase {stab}; $k^*\approx{k_star:.3f}$, "
            rf"$2\pi/k^*\approx {wave_str}$"
        )

    ax.axhline(0.0, color="0.35", ls=":", lw=0.9)
    ax.set_xlabel(r"wavenumber $k$ along band (x direction)")
    ax.set_ylabel(r"in-phase $\max \mathrm{Re}\,\lambda(k)$")
    ax.set_title(
        "Interdigitation — linear stability (flat central band)\n"
        rf"$\delta u,\delta v \propto e^{{\lambda t+ikx}}$, even about $y=L/2$; "
        rf"$L$={domain_size:g}, gap $2w$={2 * band_half_width:g}; sharp limit $\sigma\to 0$"
    )
    print(
        "linear_stability_summary:",
        f"L={domain_size:g}, w={band_half_width:g}, gap=2w={2 * band_half_width:g};",
        " | ".join(summary),
    )
    ax.grid(alpha=0.35)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_lamella_linear_stability(
    out_path: Path,
    *,
    c0: float = DEFAULT_C0,
    tau: float = DEFAULT_TAU,
    dv: float = DEFAULT_DV,
    sigma_list: tuple[float, ...] | None = None,
    k_max: float = 10.0,
    nk: int = 91,
    n_grid: int = 160,
) -> None:
    """Legacy: in-phase stability of optimal 1D periodic lamella (W, G); sweep sigma for sharp limit."""
    if sigma_list is None:
        sigma_list = (c0**2 / tau,)

    W, G, ok_w = optimal_lamella_widths_v_balance(dv)
    k_vals = np.linspace(0.0, k_max, nk, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    summary: list[str] = []
    for sig in sigma_list:
        c0_run = float(np.sqrt(sig * tau))
        dt_use = _pseudotime_dt_default_sigma(sig)
        u0, v0, dx, ok_r = pseudotime_relax_periodic_1d(
            c0_run, tau, dv, W, G, n_grid, dt=dt_use
        )
        if not ok_r:
            summary.append(rf"$\sigma$={sig:g}: relax failed")
            continue
        perm = _reflection_perm_gap_mid(n_grid, W, G, dx)
        inc, _allm = dispersion_inphase_branch(
            u0, v0, c0_run, tau, dv, dx, perm, k_vals
        )
        i_star = int(np.nanargmax(inc))
        k_star = float(k_vals[i_star])
        lam_star = float(inc[i_star])
        wave = float(2.0 * np.pi / k_star) if k_star > 1e-8 else float("inf")
        ax.plot(
            k_vals,
            inc,
            lw=1.4,
            label=rf"$\sigma$={sig:g}, max$\,\mathrm{{Re}}\,\lambda$={lam_star:.3g}",
        )
        stab = "unstable" if lam_star > 0 else "stable"
        wave_str = f"{wave:.3g}" if np.isfinite(wave) else "inf"
        summary.append(
            rf"$\sigma$={sig:g}: in-phase {stab}; $k^*\approx{k_star:.3f}$, "
            rf"$2\pi/k^*\approx {wave_str}$"
        )

    ax.axhline(0.0, color="0.35", ls=":", lw=0.9)
    ax.set_xlabel(r"transverse wavenumber $k$")
    ax.set_ylabel(r"in-phase $\max \mathrm{Re}\,\lambda(k)$")
    ax.set_title(
        "Interdigitation — linear stability (optimal lamella, two interfaces)\n"
        rf"in-phase corrugation: $\delta u,\delta v \propto e^{{\lambda t+iky}}$; sharp limit $\sigma\to 0$"
    )
    print(
        "linear_stability_summary:",
        f"W={W:.3f}, G={G:.3f}, ok={ok_w};",
        " | ".join(summary),
    )
    ax.grid(alpha=0.35)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D: initial data, solver
# ---------------------------------------------------------------------------


def _build_interdigitation_initial_2d(
    nx: int,
    ny: int,
    *,
    dx: float,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Central noisy band gap (u=0) surrounded by bone (u=1)."""
    rng = np.random.default_rng(seed)
    half = ny // 2
    band_half_width = 1.0 / dx
    u = np.ones((nx, ny), dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            noise = rng.uniform(-1.0, 1.0)
            if (iy - half) ** 2 < (band_half_width + noise) ** 2:
                u[ix, iy] = 0.0
    return u


def interdigitation_2d_periodic(
    domain_size: float = DEFAULT_DOMAIN_SIZE,
    dx: float = DEFAULT_DX,
    dy: float = DEFAULT_DY,
    dt: float = DEFAULT_DT,
    n_steps: int = DEFAULT_TOTAL_STEPS,
    save_every: int | None = None,
    c0: float = DEFAULT_C0,
    tau: float = DEFAULT_TAU,
    dv: float = DEFAULT_DV,
    seed: int = DEFAULT_SEED,
    v_star: float = DEFAULT_V_STAR,
    gain: float = DEFAULT_FORCING_GAIN,
):
    """2D periodic interdigitation model (semi-implicit fftfreq diffusion).

    Returns ``(t_hist, u_hist, v_hist, dx, dy, u_prev, v_prev)`` where
    ``(u_prev, v_prev)`` is the consecutive PDE state one step before the final
    ``(u_hist[-1], v_hist[-1])``.
    """
    nx = int(round(domain_size / dx))
    ny = int(round(domain_size / dy))
    u = _build_interdigitation_initial_2d(nx, ny, dx=dx, seed=seed)
    v = np.zeros((nx, ny), dtype=float)

    du, alpha, beta = _interdigitation_physical_u_params(c0, tau)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kxx, kyy = np.meshgrid(kx, ky)
    denom_u = 1.0 + dt * du * (kxx**2 + kyy**2)
    denom_v = 1.0 + dt * dv * (kxx**2 + kyy**2)

    if save_every is None:
        save_every = _save_every_for_steps(n_steps)

    u_hist = [u.copy()]
    v_hist = [v.copy()]
    t_hist = [0.0]
    u_prev = u.copy()
    v_prev = v.copy()

    for step in range(1, n_steps + 1):
        u_prev = u.copy()
        v_prev = v.copy()
        F_eval = _interdigitation_forcing_from_v(v, c0=c0, tau=tau, v_star=v_star, gain=gain)
        rhs_u = u + dt * _allen_cahn_reaction(u, alpha, beta, F_eval)
        rhs_v = v + dt * (1.0 - u - v)
        u = np.real(np.fft.ifft2(np.fft.fft2(rhs_u) / denom_u))
        v = np.real(np.fft.ifft2(np.fft.fft2(rhs_v) / denom_v))
        if step % save_every == 0:
            u_hist.append(u.copy())
            v_hist.append(v.copy())
            t_hist.append(step * dt)

    return np.array(t_hist), np.array(u_hist), np.array(v_hist), dx, dy, u_prev, v_prev


def verify_one_step_kernel_vs_fftfreq(
    *,
    n: int = 32,
    dx: float = DEFAULT_DX,
    dt: float = DEFAULT_DT,
    c0: float = DEFAULT_C0,
    tau: float = DEFAULT_TAU,
    dv: float = DEFAULT_DV,
    seed: int = 0,
) -> dict[str, float]:
    """Compare one u-step: legacy kernel FFT vs fftfreq semi-implicit."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, n))
    v = rng.random((n, n))
    du, alpha, beta = _interdigitation_physical_u_params(c0, tau)
    F_eval = _interdigitation_forcing_from_v(v, c0=c0, tau=tau)
    fu = u + dt * _allen_cahn_reaction(u, alpha, beta, F_eval)

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kxx, kyy = np.meshgrid(kx, ky)
    denom_u = 1.0 + dt * du * (kxx**2 + kyy**2)
    u_fftfreq = np.real(np.fft.ifft2(np.fft.fft2(fu) / denom_u))

    cu = dt * du / (dx * dx)
    kern_u = np.zeros((n, n))
    kern_u[0, 0] = 1 + 4 * cu
    kern_u[[1, -1, 0, 0], [0, 0, 1, -1]] = -cu
    ku = np.fft.fft2(kern_u)
    u_kernel = np.real(np.fft.ifft2(np.fft.fft2(fu) / ku))

    diff = u_fftfreq - u_kernel
    l2 = float(np.sqrt(np.mean(diff**2)))
    linf = float(np.max(np.abs(diff)))
    print("=== one-step diffusion: fftfreq vs legacy kernel ===")
    print(f"L2   error = {l2:.6g}")
    print(f"Linf error = {linf:.6g}")
    return {"l2_error": l2, "linf_error": linf}


# ---------------------------------------------------------------------------
# GIF / PNG export
# ---------------------------------------------------------------------------


def save_final_frame_uv(
    u: np.ndarray,
    v: np.ndarray,
    out_path: Path | str,
    *,
    t: float | None = None,
    dx: float = DEFAULT_DX,
    dy: float = DEFAULT_DY,
    sigma: float | None = None,
    cmap: str = "viridis",
) -> Path:
    """Save side-by-side spatial maps of the final (u, v) fields."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u2 = np.asarray(u, dtype=float)
    v2 = np.asarray(v, dtype=float)
    nx, ny = u2.shape
    extent = _cell_centred_extent(nx, ny, dx, dy)
    sig = DEFAULT_SIGMA if sigma is None else float(sigma)
    t_note = "" if t is None else rf", $t={t:g}$"

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    for ax, field, name, vmax in (
        (axes[0], u2, r"$u$", 1.0),
        (axes[1], v2, r"$v$", max(1.0, float(np.nanmax(v2)))),
    ):
        im = ax.imshow(
            field.T,
            vmin=0.0,
            vmax=vmax,
            cmap=cmap,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax.set_title(name)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(rf"Final frame ($\sigma={sig:g}${t_note})", fontsize=12)
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_final_frame: {out_path}")
    return out_path


def save_delta_u_spatial(
    u: np.ndarray,
    u_prev: np.ndarray,
    out_path: Path | str,
    *,
    dt_effective: float,
    t: float | None = None,
    dx: float = DEFAULT_DX,
    dy: float = DEFAULT_DY,
    sigma: float | None = None,
) -> Path:
    """Save the spatial distribution of Δu between the two UV-flow frames."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u2 = np.asarray(u, dtype=float)
    u0 = np.asarray(u_prev, dtype=float)
    if u2.ndim != 2 or u0.shape != u2.shape:
        raise ValueError("u and u_prev must be matching 2D arrays.")

    delta_u = u2 - u0
    nx, ny = delta_u.shape
    extent = _cell_centred_extent(nx, ny, dx, dy)
    color_limit = float(np.max(np.abs(delta_u)))
    if not np.isfinite(color_limit) or color_limit < 1e-16:
        color_limit = 1.0
    n_decreasing = int(np.count_nonzero(delta_u < 0))
    n_increasing = int(np.count_nonzero(delta_u > 0))
    sig = DEFAULT_SIGMA if sigma is None else float(sigma)
    t_note = "" if t is None else rf", $t={t:g}$"

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    im_delta = axes[0].imshow(
        delta_u.T,
        vmin=-color_limit,
        vmax=color_limit,
        cmap="coolwarm",
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[0].set_title(r"$\Delta u$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    cbar_delta = fig.colorbar(im_delta, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_delta.set_label(r"$\Delta u=u(t)-u(t-\Delta t_{\mathrm{eff}})$")

    im_u = axes[1].imshow(
        u2.T,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[1].set_title(r"$u(t)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    cbar_u = fig.colorbar(im_u, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_u.set_label(r"$u$")

    fig.suptitle(
        rf"Spatial distribution of $\Delta u$ "
        rf"($\sigma={sig:g}${t_note}, $\Delta t_{{\mathrm{{eff}}}}={dt_effective:g}$)"
        "\n"
        rf"$\Delta u<0$: {n_decreasing}/{delta_u.size}, "
        rf"$\Delta u>0$: {n_increasing}/{delta_u.size}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_delta_u: {out_path}")
    return out_path


def save_uv_scatter_final_frame(
    u: np.ndarray,
    v: np.ndarray,
    out_path: Path | str,
    *,
    u_prev: np.ndarray | None = None,
    v_prev: np.ndarray | None = None,
    dt: float = DEFAULT_DT,
    v_star: float = DEFAULT_V_STAR,
    sigma: float | None = None,
    title: str | None = None,
    grid_stride: int = 5,
    quiver_ref_length: float = 0.06,
) -> Path:
    """Scatter final-frame (u, v) and optional transition vectors from previous time.

    The spatial grid is sampled uniformly in both axes with ``grid_stride``.
    When ``u_prev`` / ``v_prev`` are given, each sampled grid point draws a
    quiver from ``(u_prev, v_prev)``. Arrow direction and relative length represent
    ``(Δu, Δv)`` and ``|Δ(u,v)|``. Color is signed ``Δu`` (red increase, blue
    decrease).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u2 = np.asarray(u, dtype=float)
    v2 = np.asarray(v, dtype=float)
    if u2.ndim != 2 or v2.shape != u2.shape:
        raise ValueError("u and v must be matching 2D arrays.")
    uu = u2.ravel()
    vv = v2.ravel()
    stride = max(1, int(grid_stride))
    # Build indices on the original 2D grid. In contrast, ravel()[::step]
    # aliases with the row width and may sample only a few y coordinates.
    sample_idx = np.arange(u2.size).reshape(u2.shape)[::stride, ::stride].ravel()

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(
        uu[sample_idx],
        vv[sample_idx],
        s=8,
        alpha=0.25,
        c="0.55",
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    ax.axhline(v_star, color="0.35", ls="--", lw=1.0, label=rf"$v_*={v_star:g}$")

    if u_prev is not None and v_prev is not None:
        u0 = np.asarray(u_prev, dtype=float).ravel()
        v0 = np.asarray(v_prev, dtype=float).ravel()
        if u0.shape != uu.shape or v0.shape != vv.shape:
            raise ValueError("u_prev/v_prev must match the final-frame shape.")
        du = uu - u0
        dv = vv - v0
        speed = np.hypot(du, dv)
        du_sl = du[sample_idx]
        dv_sl = dv[sample_idx]
        speed_sl = speed[sample_idx]
        speed_ref = float(np.percentile(speed_sl, 95))
        if not np.isfinite(speed_ref) or speed_ref < 1e-16:
            speed_ref = float(np.max(speed_sl)) if np.max(speed_sl) > 0 else 1.0
        # Preserve relative displacement lengths; map the 95th percentile to
        # quiver_ref_length UV-axis units so outliers do not dominate.
        quiver_scale = speed_ref / max(quiver_ref_length, 1e-12)
        # Color by sign only: continuous Δu maps tiny negatives to white
        # under clim dominated by large positive interface updates.
        du_sign = np.sign(du_sl)
        q = ax.quiver(
            u0[sample_idx],
            v0[sample_idx],
            du_sl,
            dv_sl,
            du_sign,
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            cmap="coolwarm",
            width=0.004,
            alpha=0.9,
            zorder=3,
            pivot="tail",
        )
        q.set_clim(-1.0, 1.0)
        cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels([r"$\Delta u<0$", r"$0$", r"$\Delta u>0$"])
        cbar.set_label(r"sign$(\Delta u)$")
        ax.quiverkey(
            q,
            0.18,
            0.08,
            speed_ref,
            rf"$|\Delta|=${speed_ref:.2e}",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 8},
        )
        n_show = int(sample_idx.size)
        n_dec = int(np.count_nonzero(du_sl < 0))
        n_inc = int(np.count_nonzero(du_sl > 0))
        mean_speed = float(np.mean(speed_sl))
        # True |Δu| scales differ by orders of magnitude.
        med_dec = float(np.median(-du_sl[du_sl < 0])) if n_dec else 0.0
        med_inc = float(np.median(du_sl[du_sl > 0])) if n_inc else 0.0
        vec_note = (
            rf", $\langle|\Delta|\rangle$={mean_speed:.2e}, "
            rf"$\Delta u{{<}}0$: {n_dec}/{n_show} (med$|\Delta u|$={med_dec:.1e}), "
            rf"$\Delta u{{>}}0$: {n_inc}/{n_show} (med$|\Delta u|$={med_inc:.1e})"
        )
    else:
        n_show = int(sample_idx.size)
        vec_note = ""

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$v$")
    if title is None:
        sig = DEFAULT_SIGMA if sigma is None else float(sigma)
        title = (
            rf"UV flow ($\sigma={sig:g}$, $N={n_show}$, "
            rf"$\Delta t_{{\mathrm{{eff}}}}={dt:g}${vec_note})"
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ymin = float(min(vv.min(), 0.0)) - 0.03
    ymax = float(max(vv.max(), v_star)) + 0.05
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_uv_scatter: {out_path}")
    return out_path


def save_u_history_gif_2d(
    u_hist,
    out_gif_path,
    *,
    title: str = "Interdigitation u",
    vmin=0.0,
    vmax=1.0,
    fps=GIF_FPS,
    every=1,
    cmap="viridis",
    dx=DEFAULT_DX,
    dy=DEFAULT_DY,
):
    """2D u history GIF."""
    out_gif_path = str(out_gif_path)
    frame_ids = list(range(0, len(u_hist), max(1, every)))
    frame0 = np.asarray(u_hist[frame_ids[0]]).T
    ny, nx = frame0.shape
    extent = _cell_centred_extent(nx, ny, dx, dy)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        frame0,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        animated=True,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"position $y$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(k):
        idx = frame_ids[k]
        im.set_array(np.asarray(u_hist[idx]).T)
        ax.set_title(f"{title}, frame={idx}")
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(frame_ids), blit=True, interval=1000 / fps)
    ani.save(out_gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved_gif: {out_gif_path}")


# ---------------------------------------------------------------------------
# Batch driver + CLI
# ---------------------------------------------------------------------------


def run_all_simulations(
    output_dir=None,
    *,
    c0: float = DEFAULT_C0,
    tau: float = DEFAULT_TAU,
    dv: float = DEFAULT_DV,
    dx: float = DEFAULT_DX,
    dy: float = DEFAULT_DY,
    dt: float = DEFAULT_DT,
    n_steps: int = DEFAULT_TOTAL_STEPS,
    seed: int = DEFAULT_SEED,
    domain_size: float = DEFAULT_DOMAIN_SIZE,
    band_half_width: float = DEFAULT_BAND_HALF_WIDTH,
    lamella_stability: bool = False,
    sigma_list: tuple[float, ...] | None = None,
    run_linear_stability: bool = True,
    run_gifs: bool = True,
    run_uv_scatter: bool = True,
    run_final_frame: bool = True,
    run_delta_u: bool = True,
) -> dict:
    """Run 2D simulation plots/GIFs and optional linear stability analysis."""
    out = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    t_hist, u_hist, v_hist, dx, dy, u_prev, v_prev = interdigitation_2d_periodic(
        domain_size=domain_size,
        dx=dx,
        dy=dy,
        dt=dt,
        n_steps=n_steps,
        c0=c0,
        tau=tau,
        dv=dv,
        seed=seed,
    )
    result = {
        "t": t_hist,
        "u_hist": u_hist,
        "v_hist": v_hist,
        "u_prev": u_prev,
        "v_prev": v_prev,
    }

    if run_gifs:
        gif_u = out / OUT_GIF_2D_U
        gif_v = out / OUT_GIF_2D_V
        save_u_history_gif_2d(u_hist, gif_u, title="Interdigitation u", dx=dx, dy=dy)
        save_u_history_gif_2d(v_hist, gif_v, title="Interdigitation v", dx=dx, dy=dy)
        result["gif_u"] = gif_u
        result["gif_v"] = gif_v

    if run_final_frame:
        png_final = out / OUT_PNG_FINAL_FRAME
        save_final_frame_uv(
            u_hist[-1],
            v_hist[-1],
            png_final,
            t=float(t_hist[-1]),
            dx=dx,
            dy=dy,
            sigma=c0**2 / tau,
        )
        result["final_frame_png"] = png_final

    if run_uv_scatter or run_delta_u:
        # Use the same frame pair for the UV vectors and spatial Δu plot.
        if len(u_hist) >= 2:
            u_from, v_from = u_hist[-2], v_hist[-2]
            dt_eff = float(t_hist[-1] - t_hist[-2])
        else:
            u_from, v_from = u_prev, v_prev
            dt_eff = float(dt)

    if run_uv_scatter:
        png_uv = out / OUT_PNG_UV_SCATTER
        save_uv_scatter_final_frame(
            u_hist[-1],
            v_hist[-1],
            png_uv,
            u_prev=u_from,
            v_prev=v_from,
            dt=dt_eff,
            sigma=c0**2 / tau,
        )
        result["uv_scatter_png"] = png_uv

    if run_delta_u:
        png_delta_u = out / OUT_PNG_DELTA_U
        save_delta_u_spatial(
            u_hist[-1],
            u_from,
            png_delta_u,
            dt_effective=dt_eff,
            t=float(t_hist[-1]),
            dx=dx,
            dy=dy,
            sigma=c0**2 / tau,
        )
        result["delta_u_png"] = png_delta_u

    if run_linear_stability:
        png_ls = out / OUT_PNG_LINEAR_STABILITY
        plot_interdigitation_linear_stability(
            png_ls,
            c0=c0,
            tau=tau,
            dv=dv,
            sigma_list=sigma_list,
            domain_size=domain_size,
            band_half_width=band_half_width,
            lamella_stability=lamella_stability,
        )
        result["linear_stability_png"] = png_ls

    print("=== exported files ===")
    for key in (
        "gif_u",
        "gif_v",
        "final_frame_png",
        "uv_scatter_png",
        "delta_u_png",
        "linear_stability_png",
    ):
        if key in result:
            print(result[key])
    return result


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Run PhaseField interdigitation simulations.")
    p.add_argument("--c0", type=float, default=None, help="Mobility / speed scale c0.")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Time scale tau.")
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Physical u diffusivity du=c0^2/tau with tau=1 (sets c0=sqrt(sigma)).",
    )
    p.add_argument("--dv", type=float, default=DEFAULT_DV, help="Diffusivity for v.")
    p.add_argument("--dx", type=float, default=DEFAULT_DX, help="Grid spacing in x.")
    p.add_argument("--dy", type=float, default=DEFAULT_DY, help="Grid spacing in y.")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="Time step.")
    p.add_argument("--n-steps", type=int, default=DEFAULT_TOTAL_STEPS, help="Number of time steps.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for initial band noise.")
    p.add_argument("--domain-size", type=float, default=DEFAULT_DOMAIN_SIZE, help="Domain length L.")
    p.add_argument(
        "--band-half-width",
        type=float,
        default=DEFAULT_BAND_HALF_WIDTH,
        help="Half-width w of flat central band (gap width 2w).",
    )
    p.add_argument(
        "--lamella-stability",
        action="store_true",
        help="Use legacy optimal-lamella linear stability instead of flat band.",
    )
    p.add_argument("--output-dir", type=str, default=None, help="Output directory for GIF/PNG.")
    p.add_argument(
        "--linear-stability-only",
        action="store_true",
        help="Skip 2D simulation; plot linear stability only.",
    )
    p.add_argument(
        "--uv-scatter-only",
        action="store_true",
        help="Run 2D simulation and export final-frame (u,v) scatter only (no GIF / stability).",
    )
    p.add_argument(
        "--final-frame-only",
        action="store_true",
        help="Run 2D simulation and export final-frame spatial u/v maps only.",
    )
    p.add_argument(
        "--delta-u-only",
        action="store_true",
        help="Run 2D simulation and export the final saved-frame Δu spatial map only.",
    )
    p.add_argument(
        "--verify-one-step",
        action="store_true",
        help="Compare fftfreq vs legacy kernel one-step update and exit.",
    )
    p.add_argument(
        "--sigma-sweep",
        action="store_true",
        help="Linear stability: sweep physical sigma (sharp-interface limit sigma→0).",
    )
    p.add_argument(
        "--sigma-sweep-quick",
        action="store_true",
        help="Short sigma sweep (2 values) for linear stability smoke tests.",
    )
    p.add_argument(
        "--sigmas",
        type=str,
        default=None,
        help="Comma-separated sigma list for --sigma-sweep (default: 0.0045,0.003,0.002,0.001).",
    )
    return p


def _resolve_sigma_list_for_stability(
    args: argparse.Namespace,
    c0: float,
    tau: float,
) -> tuple[float, ...] | None:
    """Return sigma sweep list, or None to use single c0²/tau."""
    if args.sigma_sweep or args.sigma_sweep_quick or args.sigmas:
        if args.sigmas:
            return tuple(float(x.strip()) for x in args.sigmas.split(","))
        return _default_sigma_sweep_values(quick=args.sigma_sweep_quick)
    return None


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    c0, tau = _resolve_c0_tau(c0=args.c0, tau=args.tau, sigma=args.sigma)
    sigma_list = _resolve_sigma_list_for_stability(args, c0, tau)

    if args.verify_one_step:
        verify_one_step_kernel_vs_fftfreq(c0=c0, tau=tau, dv=args.dv, dt=args.dt)
    elif args.linear_stability_only:
        out = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
        out.mkdir(parents=True, exist_ok=True)
        plot_interdigitation_linear_stability(
            out / OUT_PNG_LINEAR_STABILITY,
            c0=c0,
            tau=tau,
            dv=args.dv,
            sigma_list=sigma_list,
            domain_size=args.domain_size,
            band_half_width=args.band_half_width,
            lamella_stability=args.lamella_stability,
        )
    elif args.uv_scatter_only:
        run_all_simulations(
            output_dir=args.output_dir,
            c0=c0,
            tau=tau,
            dv=args.dv,
            dx=args.dx,
            dy=args.dy,
            dt=args.dt,
            n_steps=args.n_steps,
            seed=args.seed,
            domain_size=args.domain_size,
            band_half_width=args.band_half_width,
            lamella_stability=args.lamella_stability,
            sigma_list=sigma_list,
            run_linear_stability=False,
            run_gifs=False,
            run_uv_scatter=True,
            run_final_frame=False,
            run_delta_u=False,
        )
    elif args.final_frame_only:
        run_all_simulations(
            output_dir=args.output_dir,
            c0=c0,
            tau=tau,
            dv=args.dv,
            dx=args.dx,
            dy=args.dy,
            dt=args.dt,
            n_steps=args.n_steps,
            seed=args.seed,
            domain_size=args.domain_size,
            band_half_width=args.band_half_width,
            lamella_stability=args.lamella_stability,
            sigma_list=sigma_list,
            run_linear_stability=False,
            run_gifs=False,
            run_uv_scatter=False,
            run_final_frame=True,
            run_delta_u=False,
        )
    elif args.delta_u_only:
        run_all_simulations(
            output_dir=args.output_dir,
            c0=c0,
            tau=tau,
            dv=args.dv,
            dx=args.dx,
            dy=args.dy,
            dt=args.dt,
            n_steps=args.n_steps,
            seed=args.seed,
            domain_size=args.domain_size,
            band_half_width=args.band_half_width,
            lamella_stability=args.lamella_stability,
            sigma_list=sigma_list,
            run_linear_stability=False,
            run_gifs=False,
            run_uv_scatter=False,
            run_final_frame=False,
            run_delta_u=True,
        )
    else:
        run_all_simulations(
            output_dir=args.output_dir,
            c0=c0,
            tau=tau,
            dv=args.dv,
            dx=args.dx,
            dy=args.dy,
            dt=args.dt,
            n_steps=args.n_steps,
            seed=args.seed,
            domain_size=args.domain_size,
            band_half_width=args.band_half_width,
            lamella_stability=args.lamella_stability,
            sigma_list=sigma_list,
        )
