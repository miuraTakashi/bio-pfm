"""
Amari (1977) two-layer neural field — stationary travelling wave (Fig. 10).

Reference: Amari, Biol. Cybernetics 27, 77–87 (1977), §8.2 and numerical example p. 86.

Model (s1 = s2 = 0):
  τ ∂u1/∂t = -u1 + ∫ w1(x-x') f[u1(x')] dx' - ∫ w2(x-x') f[u2(x')] dx' + h1
  τ ∂u2/∂t = -u2 + w3 f[u1(x)] + h2

with f(u) = 1 if u > 0 else 0.  In the co-moving frame y = x - v t, a stationary
wave satisfies (18)–(19) in the paper; Fig. 10 shows g1(y), g2(y).

Kernels (numerical example):
  w_i(x) = A_i / (sqrt(2π) σ_i) * exp(-x² / (2 σ_i²)),  ∫ w_i dx = A_i.

Parameters: A1=2, A2=4, σ1=1, σ2=1.5, w3=2, h1=-0.1, h2=-1, τ=1.
Paper reports a ≈ 7.6, v ≈ 7.3.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))


@dataclass(frozen=True)
class AmariParams:
    A1: float = 2.0
    A2: float = 4.0
    sigma1: float = 1.0
    sigma2: float = 1.5
    w3: float = 2.0
    h1: float = -0.1
    h2: float = -1.0
    tau: float = 1.0


def w_kernel(x: np.ndarray, A: float, sigma: float) -> np.ndarray:
    return (A / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)


def step_f(u: np.ndarray) -> np.ndarray:
    return (u > 0.0).astype(np.float64)


def excited_bounds(a: float, v: float, p: AmariParams) -> tuple[float, float]:
    """Inhibitory excited interval (y1, y2) from g2(y_i) = 0 (paper §8.2)."""
    tv = p.tau * v
    y1 = tv * np.log(p.h2 / (p.w3 * (np.exp(-a / tv) - 1.0)))
    y2 = a + tv * np.log(1.0 + p.h2 / p.w3)
    return float(y1), float(y2)


def _excitatory_conv(y: np.ndarray, a: float, p: AmariParams) -> np.ndarray:
    """∫_0^a w1(y-y') dy' for scalar or 1D y."""
    y = np.atleast_1d(y).astype(np.float64)
    n = max(48, int(60 * a))
    ys = np.linspace(0.0, a, n)
    return np.trapezoid(w_kernel(y[:, None] - ys[None, :], p.A1, p.sigma1), ys, axis=1)


def _inhibitory_conv(y: np.ndarray, y1: float, y2: float, p: AmariParams) -> np.ndarray:
    y = np.atleast_1d(y).astype(np.float64)
    if y2 <= y1:
        return np.zeros_like(y, dtype=np.float64)
    n = max(48, int(30 * (y2 - y1)))
    ys = np.linspace(y1, y2, n)
    return np.trapezoid(w_kernel(y[:, None] - ys[None, :], p.A2, p.sigma2), ys, axis=1)


def kernel_K(y: np.ndarray | float, a: float, y1: float, y2: float, p: AmariParams) -> np.ndarray | float:
    """K(y) = ∫_0^a w1(y-y') dy' - ∫_{y1}^{y2} w2(y-y') dy'."""
    y_arr = np.atleast_1d(y).astype(np.float64)
    out = _excitatory_conv(y_arr, a, p) - _inhibitory_conv(y_arr, y1, y2, p)
    return float(out[0]) if np.ndim(y) == 0 else out


def g1_profile(y: float, a: float, v: float, p: AmariParams) -> float:
    """g1(y) from paper: (1/(vτ)) ∫_y^∞ exp((y-y')/(vτ)) K(y') dy' + h1."""
    if y < 0.0:
        return p.h1
    tv = p.tau * v
    y1, y2 = excited_bounds(a, v, p)
    y_end = max(a, y2) + 30.0
    yp = np.linspace(y, y_end, 1200)
    kern = np.exp((y - yp) / tv) * kernel_K(yp, a, y1, y2, p)
    return float(np.trapezoid(kern, yp) / tv + p.h1)


def g2_profile(y: float, a: float, v: float, p: AmariParams) -> float:
    """Piecewise explicit g2(y) in §8.2."""
    tv = p.tau * v
    if y > a:
        return p.h2
    if 0.0 < y < a:
        return p.w3 * (1.0 - np.exp((y - a) / tv)) + p.h2
    if y < 0.0:
        amp = p.w3 * (1.0 - np.exp(-a / tv))
        return amp * np.exp(y / tv) + p.h2
    if y == 0.0:
        return g2_profile(1e-12, a, v, p)
    return p.h2


def solve_traveling_wave(
    p: AmariParams | None = None,
    *,
    a0: float = 7.6,
    v0: float = 7.3,
) -> dict[str, float | AmariParams]:
    """Find (a, v) with g1(0) = g1(a) = 0 (numerical root of §8.2)."""

    p = p or AmariParams()

    def residuals(x: np.ndarray) -> np.ndarray:
        a, v = float(x[0]), float(x[1])
        if a <= 0.0 or v <= 0.0:
            return np.array([1e3, 1e3])
        return np.array([g1_profile(0.0, a, v, p), g1_profile(a, a, v, p)])

    sol = root(residuals, np.array([a0, v0], dtype=np.float64), method="hybr")
    if not sol.success:
        raise RuntimeError(f"traveling wave root failed: {sol.message}")
    a, v = float(sol.x[0]), float(sol.x[1])
    y1, y2 = excited_bounds(a, v, p)
    return {
        "params": p,
        "a": a,
        "v": v,
        "y1": y1,
        "y2": y2,
    }


def sample_profiles(
    wave: dict[str, float | AmariParams],
    *,
    y_min: float = -12.0,
    y_max: float = 22.0,
    n: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = wave["params"]
    assert isinstance(p, AmariParams)
    a, v = float(wave["a"]), float(wave["v"])
    y = np.linspace(y_min, y_max, n)
    g2 = np.vectorize(lambda yi: g2_profile(float(yi), a, v, p))(y)
    g1 = np.empty_like(y)
    for i, yi in enumerate(y):
        g1[i] = g1_profile(float(yi), a, v, p)
    return y, g1, g2


def make_spatial_kernels(dx: float, p: AmariParams, half_width: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    n = int(half_width / dx)
    offsets = np.arange(-n, n + 1, dtype=np.float64) * dx
    return w_kernel(offsets, p.A1, p.sigma1), w_kernel(offsets, p.A2, p.sigma2)


def conv1d_zero(arr: np.ndarray, kernel: np.ndarray, dx: float) -> np.ndarray:
    """∫ w(x-y) f(y) dy with zero padding outside the domain."""
    raw = np.convolve(arr, kernel, mode="same") * dx
    return raw


def simulate_pde_traveling_wave(
    wave: dict[str, float | AmariParams],
    *,
    L: float = 120.0,
    dx: float = 0.05,
    dt: float = 0.002,
    t_end: float = 8.0,
    y0: float = 15.0,
) -> dict[str, np.ndarray | float]:
    """
    Time-step (14) from a stationary profile at x ≈ y0 to verify speed v.
    """
    p = wave["params"]
    assert isinstance(p, AmariParams)
    a, v = float(wave["a"]), float(wave["v"])

    nx = int(L / dx) + 1
    x = np.arange(nx, dtype=np.float64) * dx
    w1k, w2k = make_spatial_kernels(dx, p)

    y_shift = x - y0
    u1 = np.empty(nx, dtype=np.float64)
    u2 = np.empty(nx, dtype=np.float64)
    for i, yi in enumerate(y_shift):
        u1[i] = g1_profile(float(yi), a, v, p)
        u2[i] = g2_profile(float(yi), a, v, p)

    n_steps = int(round(t_end / dt))
    record_every = max(1, n_steps // 80)
    times: list[float] = []
    u1_hist: list[np.ndarray] = []
    fronts: list[float] = []

    def front_position(u: np.ndarray) -> float:
        fu = step_f(u)
        idx = np.where(fu > 0.5)[0]
        if idx.size < 2:
            return np.nan
        return float(x[idx[-1]])

    for step in range(n_steps + 1):
        if step % record_every == 0:
            times.append(step * dt)
            u1_hist.append(u1.copy())
            fronts.append(front_position(u1))
        if step == n_steps:
            break
        fu1 = step_f(u1)
        fu2 = step_f(u2)
        c1 = conv1d_zero(fu1, w1k, dx) - conv1d_zero(fu2, w2k, dx)
        u1 = u1 + (dt / p.tau) * (-u1 + c1 + p.h1)
        u2 = u2 + (dt / p.tau) * (-u2 + p.w3 * fu1 + p.h2)

    t_arr = np.array(times, dtype=np.float64)
    fronts_arr = np.array(fronts, dtype=np.float64)
    mask = np.isfinite(fronts_arr)
    speed_fit = float(np.polyfit(t_arr[mask], fronts_arr[mask], 1)[0]) if mask.sum() >= 4 else np.nan

    return {
        "x": x,
        "times": t_arr,
        "u1_hist": np.array(u1_hist, dtype=np.float64),
        "fronts": fronts_arr,
        "speed_fit": speed_fit,
        "v_theory": v,
    }


def simulate_pde_from_perturbation(
    wave: dict[str, float | AmariParams],
    *,
    L: float = 140.0,
    dx: float = 0.05,
    dt: float = 0.002,
    t_end: float = 8.0,
    x0: float = 20.0,
    sigma0: float = 1.2,
    amp_u1: float = 2.0,
) -> dict[str, np.ndarray | float]:
    """
    PDE test from local perturbation around resting state.

    Resting state is u1=h1, u2=h2 (because f(u1)=f(u2)=0). A local Gaussian bump is
    added to u1 to check whether an active pulse is nucleated and propagates.
    """
    p = wave["params"]
    assert isinstance(p, AmariParams)
    v = float(wave["v"])

    nx = int(L / dx) + 1
    x = np.arange(nx, dtype=np.float64) * dx
    w1k, w2k = make_spatial_kernels(dx, p)

    u1 = np.full(nx, p.h1, dtype=np.float64)
    u2 = np.full(nx, p.h2, dtype=np.float64)
    u1 += amp_u1 * np.exp(-0.5 * ((x - x0) / sigma0) ** 2)

    n_steps = int(round(t_end / dt))
    record_every = max(1, n_steps // 100)
    times: list[float] = []
    u1_hist: list[np.ndarray] = []
    right_edges: list[float] = []
    left_edges: list[float] = []

    def active_edges(u: np.ndarray) -> tuple[float, float]:
        idx = np.where(step_f(u) > 0.5)[0]
        if idx.size < 2:
            return np.nan, np.nan
        return float(x[idx[0]]), float(x[idx[-1]])

    for step in range(n_steps + 1):
        if step % record_every == 0:
            times.append(step * dt)
            u1_hist.append(u1.copy())
            left, right = active_edges(u1)
            left_edges.append(left)
            right_edges.append(right)
        if step == n_steps:
            break
        fu1 = step_f(u1)
        fu2 = step_f(u2)
        c1 = conv1d_zero(fu1, w1k, dx) - conv1d_zero(fu2, w2k, dx)
        u1 = u1 + (dt / p.tau) * (-u1 + c1 + p.h1)
        u2 = u2 + (dt / p.tau) * (-u2 + p.w3 * fu1 + p.h2)

    t_arr = np.array(times, dtype=np.float64)
    left_arr = np.array(left_edges, dtype=np.float64)
    right_arr = np.array(right_edges, dtype=np.float64)

    mask_right = np.isfinite(right_arr)
    speed_fit = float(np.polyfit(t_arr[mask_right], right_arr[mask_right], 1)[0]) if mask_right.sum() >= 4 else np.nan
    nucleated = bool(np.isfinite(right_arr).any())
    return {
        "x": x,
        "times": t_arr,
        "u1_hist": np.array(u1_hist, dtype=np.float64),
        "left_edges": left_arr,
        "right_edges": right_arr,
        "speed_fit": speed_fit,
        "v_theory": v,
        "nucleated": float(nucleated),
    }


def plot_fig10(
    wave: dict[str, float | AmariParams],
    out_png: Path,
) -> None:
    """Reproduce Fig. 10 layout (excitatory / inhibitory waveforms)."""
    p = wave["params"]
    assert isinstance(p, AmariParams)
    a, v = float(wave["a"]), float(wave["v"])
    y1, y2 = float(wave["y1"]), float(wave["y2"])

    y, g1, g2 = sample_profiles(wave, y_min=-10.0, y_max=20.0, n=400)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)

    ax0, ax1 = axes
    ax0.plot(y, g1, "k-", lw=1.5)
    ax0.axhline(0.0, color="0.5", lw=0.6)
    ax0.axvspan(0.0, a, color="0.85", alpha=0.6)
    ax0.text(0.5 * a, 0.92 * g1.max(), "exc.", ha="center", fontsize=9)
    ax0.set_ylabel(r"$u_1$")
    ax0.set_ylim(-0.25, max(0.75, g1.max() * 1.05))
    ax0.grid(alpha=0.25)

    ax1.plot(y, g2, "k-", lw=1.5)
    ax1.axhline(0.0, color="0.5", lw=0.6)
    ax1.axvspan(y1, y2, color="0.85", alpha=0.6)
    ax1.text(0.5 * (y1 + y2), 0.55, "inh.", ha="center", fontsize=9)
    ax1.set_ylabel(r"$u_2$")
    ax1.set_xlabel(r"$y$")
    ax1.set_ylim(-1.15, max(0.65, g2.max() * 1.1))
    ax1.grid(alpha=0.25)

    fig.suptitle(
        f"Amari (1977) travelling wave — $a={a:.2f}$, $v={v:.2f}$ "
        f"($A_1={p.A1}$, $A_2={p.A2}$, $\\sigma_1={p.sigma1}$, $\\sigma_2={p.sigma2}$)"
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pde_check(
    wave: dict[str, float | AmariParams],
    pde: dict[str, np.ndarray | float],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    x = pde["x"]
    times = pde["times"]
    u1_hist = pde["u1_hist"]
    for k in range(0, u1_hist.shape[0], max(1, u1_hist.shape[0] // 10)):
        axes[0].plot(x, u1_hist[k], alpha=0.65, lw=0.9)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$u_1$")
    axes[0].set_title("PDE snapshots ($u_1$)")
    axes[0].grid(alpha=0.25)

    fronts = pde["fronts"]
    mask = np.isfinite(fronts)
    axes[1].plot(times[mask], fronts[mask], "o", ms=3, label="right edge ($f(u_1)>0.5$)")
    if mask.sum() >= 2:
        coef = np.polyfit(times[mask], fronts[mask], 1)
        axes[1].plot(times[mask], np.polyval(coef, times[mask]), "r--", label=f"fit $c\\approx{coef[0]:.2f}$")
    axes[1].axhline(pde["v_theory"], color="green", ls=":", label=f"theory $v={pde['v_theory']:.2f}$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("front position")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("PDE verification of travelling wave speed")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pulse_nucleation_check(
    pde: dict[str, np.ndarray | float],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    x = pde["x"]
    times = pde["times"]
    u1_hist = pde["u1_hist"]
    for k in range(0, u1_hist.shape[0], max(1, u1_hist.shape[0] // 10)):
        axes[0].plot(x, u1_hist[k], alpha=0.7, lw=0.9)
    axes[0].axhline(0.0, color="0.5", lw=0.7)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$u_1$")
    axes[0].set_title("From local perturbation")
    axes[0].grid(alpha=0.25)

    left = pde["left_edges"]
    right = pde["right_edges"]
    mask_left = np.isfinite(left)
    mask_right = np.isfinite(right)
    if mask_left.any():
        axes[1].plot(times[mask_left], left[mask_left], "o", ms=3, label="left edge")
    if mask_right.any():
        axes[1].plot(times[mask_right], right[mask_right], "o", ms=3, label="right edge")
    if mask_right.sum() >= 2:
        coef = np.polyfit(times[mask_right], right[mask_right], 1)
        axes[1].plot(times[mask_right], np.polyval(coef, times[mask_right]), "r--", label=f"right fit $c\\approx{coef[0]:.2f}$")
    axes[1].axhline(pde["v_theory"], color="green", ls=":", label=f"theory $v={pde['v_theory']:.2f}$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("edge position")
    axes[1].set_title("Pulse nucleation / propagation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("PDE check from perturbed initial condition")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    wave = solve_traveling_wave()
    p = wave["params"]
    assert isinstance(p, AmariParams)

    print("Travelling wave (numerical root of g1(0)=g1(a)=0):")
    print(f"  a = {wave['a']:.4f}  (paper 7.6)")
    print(f"  v = {wave['v']:.4f}  (paper 7.3)")
    print(f"  inhibitory excited interval: ({wave['y1']:.3f}, {wave['y2']:.3f})")

    fig10_path = out_dir / "amari_neural_field_fig10.png"
    plot_fig10(wave, fig10_path)
    print(f"Saved {fig10_path}")

    pde = simulate_pde_traveling_wave(wave, t_end=6.0, y0=20.0, L=140.0)
    print(f"PDE front speed (linear fit): {pde['speed_fit']:.4f}  (theory v = {pde['v_theory']:.4f})")

    pde_path = out_dir / "amari_neural_field_pde_check.png"
    plot_pde_check(wave, pde, pde_path)
    print(f"Saved {pde_path}")

    pde_pert = simulate_pde_from_perturbation(wave, t_end=8.0, x0=70.0, sigma0=1.0, amp_u1=2.2)
    print(
        "PDE perturbation test:"
        f" nucleated={bool(pde_pert['nucleated'])},"
        f" right-edge speed={pde_pert['speed_fit']:.4f},"
        f" theory v={pde_pert['v_theory']:.4f}"
    )
    pde_pert_path = out_dir / "amari_neural_field_pulse_nucleation_check.png"
    plot_pulse_nucleation_check(pde_pert, pde_pert_path)
    print(f"Saved {pde_pert_path}")


if __name__ == "__main__":
    main()
