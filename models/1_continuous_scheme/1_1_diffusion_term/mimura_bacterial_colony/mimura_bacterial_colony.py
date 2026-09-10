"""
Mimura–Sakaguchi–Matsushita (2000) bacterial colony RD model.

Reference: Physica A 282 (2000) 283–303; Eqs. (2.5)–(2.8), (3.1)–(3.4), (4.1)–(4.2).

  ∂u/∂t = ∇·(d(b) ∇u) + u v − a(u,v) u
  ∂v/∂t = ∇² v − u v
  ∂w/∂t = a(u,v) u

  a(u,v) = 1 / ((1 + u/a₁)(1 + v/a₂)),   a₁ = 1/2400, a₂ = 1/120
  b = u + w

Section 3 (1D, soft agar Eq. 3.1 / hard agar Eq. 3.4): Fig. 3.2 travelling /
oscillatory / clustering pulses; Fig. 3.4 nonlinear travelling front.

Section 4 (2D): soft agar (4.1) d(b)=d; hard agar (4.2) d(b)=d·b.
Discretisation: Neumann FD (isotropic 9-point fluxes), forward Euler, large
domain. In 2D a frozen spatial mobility χ(x) breaks rotational symmetry on
structured grids.

Initial data: droplet inoculum in u, uniform v ≡ v₀, w ≡ 0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Paper Fig. 3.1 / Sec. 4.1: a₁ = 1/2400, a₂ = 1/120 in Eq. (2.6).
A1 = 1.0 / 2400.0
A2 = 1.0 / 120.0


@dataclass(frozen=True)
class Section3Case:
    """One panel of Mimura et al. (2000) Fig. 3.2 / 3.4 (1D)."""

    label: str
    title: str
    v0: float
    d: float
    nonlinear: bool = False
    t_final: float = 2000.0
    dt: float = 0.05
    u0: float = 1.0
    r0: float = 3.0

    @staticmethod
    def default_fig32_cases() -> tuple["Section3Case", ...]:
        """Fig. 3.2 caption parameters (soft agar, Eq. 3.1)."""
        return (
            Section3Case("a", "travelling wave", v0=0.117, d=0.1, t_final=2500.0),
            Section3Case("b", "oscillatory (mild)", v0=0.129, d=0.05, t_final=3500.0),
            Section3Case("c", "oscillatory (strong)", v0=0.108, d=0.05, t_final=4500.0, u0=0.8),
            # Fig. 3.2(d): pulse is a long transient then extinguishes → stationary cluster.
            Section3Case("d", "clustering", v0=0.0875, d=0.1, t_final=9000.0, u0=1.0, r0=3.0),
        )

    @staticmethod
    def fig34_case() -> "Section3Case":
        """Fig. 3.4: hard-agar travelling front (Eq. 3.4)."""
        return Section3Case(
            "3.4",
            "nonlinear travelling",
            v0=0.25,
            d=0.05,
            nonlinear=True,
            t_final=5000.0,
            u0=1.0,
            r0=4.0,
        )


@dataclass(frozen=True)
class ColonyCase:
    """One morphology from Mimura et al. (2000) Section 4."""

    label: str
    title: str
    v0: float
    d: float
    nonlinear: bool = False
    t_final: float = 3000.0
    dt: float = 0.1
    u0: float = 1.06
    r0: float = 4.0
    chi0: float = 0.06
    seed: int = 1
    # Optional overrides (rings need finer Δx than branching cases).
    dx: float | None = None
    L: float | None = None
    enhance_rings: bool = False

    @staticmethod
    def default_section4_cases() -> tuple["ColonyCase", ...]:
        """(d, v₀) from Figs. 4.1–4.9; (B) uses nonlinear diffusion (4.2)."""
        return (
            ColonyCase(
                "A",
                "DLA-like",
                v0=0.087,
                d=0.05,
                t_final=4500.0,
                chi0=0.08,
                seed=11,
            ),
            ColonyCase(
                "B",
                "Eden-like",
                v0=0.25,
                d=0.05,
                nonlinear=True,
                t_final=4000.0,
                chi0=0.05,
                seed=12,
            ),
            # Fig. 4.6 caption is d=0.05, v0=0.1, but that sits at the O/branching
            # edge on square FD grids (often flower/DLA-like). Concentric bands
            # require the mild 1D oscillatory band (Fig. 3.2b / A-ii): v0≈0.12–0.13,
            # finer Δx to resolve the radial wave-train, and small χ.
            ColonyCase(
                "C",
                "concentric ring-like",
                v0=0.125,
                d=0.05,
                t_final=8000.0,
                dt=0.05,
                chi0=0.01,
                seed=13,
                dx=1.0,
                L=480.0,
                enhance_rings=True,
            ),
            ColonyCase(
                "D",
                "disk-like",
                v0=0.25,
                d=0.25,
                t_final=900.0,
                chi0=0.02,
                seed=14,
            ),
            ColonyCase(
                "E",
                "DBM-like",
                v0=0.071,
                d=0.12,
                t_final=3200.0,
                chi0=0.08,
                seed=15,
            ),
        )


def conversion_rate(u: np.ndarray, v: np.ndarray, *, a1: float = A1, a2: float = A2) -> np.ndarray:
    """Eq. (2.6): a(u,v) = 1 / ((1 + u/a₁)(1 + v/a₂))."""
    return 1.0 / ((1.0 + u / a1) * (1.0 + v / a2))


def laplacian_1d_neumann(field: np.ndarray, dx: float) -> np.ndarray:
    """Second derivative with Neumann BC at both ends."""
    out = np.empty_like(field)
    out[1:-1] = field[2:] + field[:-2] - 2.0 * field[1:-1]
    out[0] = 2.0 * (field[1] - field[0])
    out[-1] = 2.0 * (field[-2] - field[-1])
    return out / (dx * dx)


def div_b_grad_u_1d(b: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    """(b u_x)_x with Neumann reflection (Eq. 3.4)."""
    bp = np.pad(b, 1, mode="edge")
    up = np.pad(u, 1, mode="edge")
    b_r = 0.5 * (bp[1:-1] + bp[2:])
    b_l = 0.5 * (bp[1:-1] + bp[:-2])
    return (b_r * (up[2:] - up[1:-1]) - b_l * (up[1:-1] - up[:-2])) / (dx * dx)


def simulate_section3_case(
    case: Section3Case,
    *,
    L: float = 400.0,
    dx: float = 0.5,
    quick_scale: float = 1.0,
    n_records: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate 1D Eq. (3.1) or (3.4) on [0, L] (semi-infinite approximation).

    Returns x, t_samples, U[nt,nx], B[nt,nx], and final (u, v, w).
    """
    n = int(round(L / dx)) + 1
    x = np.linspace(0.0, L, n)
    t_final = case.t_final * quick_scale
    dt = min(case.dt, 0.4 * (dx * dx) / 4.0)
    n_steps = max(1, int(round(t_final / dt)))
    stride = max(1, n_steps // max(n_records - 1, 1))

    u = np.where(x <= case.r0, case.u0, 0.0).astype(np.float64)
    v = np.full(n, case.v0, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)

    U = np.zeros((n_records, n), dtype=np.float64)
    B = np.zeros((n_records, n), dtype=np.float64)
    t_samples = np.zeros(n_records, dtype=np.float64)
    rec = 0
    t_now = 0.0

    for step in range(n_steps):
        a = conversion_rate(u, v)
        growth = u * v
        if case.nonlinear:
            b = u + w
            diff_u = case.d * div_b_grad_u_1d(b, u, dx)
        else:
            diff_u = case.d * laplacian_1d_neumann(u, dx)
        u = np.clip(u + dt * (diff_u + growth - a * u), 0.0, None)
        v = np.clip(v + dt * (laplacian_1d_neumann(v, dx) - growth), 0.0, None)
        w = w + dt * (a * u)
        t_now = (step + 1) * dt

        if step % stride == 0 and rec < n_records:
            t_samples[rec] = t_now
            U[rec] = u
            B[rec] = u + w
            rec += 1
            if float(u[-20:].max()) > 1e-4:
                break

    if rec == 0:
        t_samples[0] = t_now if t_now > 0 else dt
        U[0] = u
        B[0] = u + w
        rec = 1
    elif t_samples[rec - 1] < t_now - 0.5 * dt and rec < n_records:
        t_samples[rec] = t_now
        U[rec] = u
        B[rec] = u + w
        rec += 1

    return x, t_samples[:rec], U[:rec], B[:rec], u, v, w


def plot_spacetime(
    ax: plt.Axes,
    x: np.ndarray,
    t: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    cmap: str = "inferno",
) -> None:
    """Space–time density plot (x horizontal, t vertical; paper Fig. 3.2 style)."""
    extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]
    vmax = float(np.percentile(field, 99.5)) if field.size else 1.0
    vmax = max(vmax, 1e-8)
    ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")


def run_section3(
    cases: tuple[Section3Case, ...] | None = None,
    *,
    L: float | None = None,
    dx: float = 0.5,
    out_png: Path | None = None,
    include_fig34: bool = True,
) -> Path:
    """Reproduce Fig. 3.2 (and optionally Fig. 3.4) as space–time panels."""
    cases = cases or Section3Case.default_fig32_cases()
    quick = os.environ.get("MIMURA_QUICK", "").strip().lower() in {"1", "true", "yes"}
    L = L or (300.0 if quick else 500.0)
    qscale = 0.45 if quick else 1.0
    n_records = 160 if quick else 320

    n_cols = len(cases)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.8 * n_cols, 5.6),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for col, case in enumerate(cases):
        print(f"Section 3 ({case.label}) {case.title} ...", flush=True)
        x, t, U, B, _, _, _ = simulate_section3_case(
            case, L=L, dx=dx, quick_scale=qscale, n_records=n_records
        )
        plot_spacetime(
            axes[0, col],
            x,
            t,
            U,
            title=f"({case.label}) $u$ — {case.title}\n$d$={case.d}, $v_0$={case.v0}",
        )
        plot_spacetime(
            axes[1, col],
            x,
            t,
            B,
            title=f"({case.label}) $b=u+w$",
            cmap="gray_r",
        )

    fig.suptitle(
        "Mimura et al. (2000) — Section 3 / Fig. 3.2 (1D soft agar, Eq. 3.1)\n"
        r"$a_1=1/2400$, $a_2=1/120$; Neumann FD on $[0,L]$",
        fontsize=10,
    )
    out_png = out_png or out_dir / "mimura_bacterial_colony_section3_fig32.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    if include_fig34:
        case34 = Section3Case.fig34_case()
        print(f"Section 3 Fig. 3.4 {case34.title} ...", flush=True)
        x, t, U, _, u, v, w = simulate_section3_case(
            case34, L=L, dx=dx, quick_scale=qscale, n_records=n_records
        )
        fig2, axes2 = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
        plot_spacetime(
            axes2[0],
            x,
            t,
            U,
            title=r"Fig. 3.4 spacetime $u$ ($d=0.05\,b$, $v_0=0.25$)",
        )
        axes2[1].plot(x, u, label=r"$u$", lw=1.5)
        axes2[1].plot(x, v, label=r"$v$", lw=1.5)
        axes2[1].plot(x, u + w, label=r"$b=u+w$", lw=1.5)
        axes2[1].set_xlabel(r"$x$")
        axes2[1].set_title(rf"late profiles  $t={t[-1]:.0f}$")
        axes2[1].legend(fontsize=8)
        # Zoom to the colony front (nonlinear front advances slowly).
        front = float(np.max(x[(u + w) > 1e-4])) if np.any(u + w > 1e-4) else float(x[-1])
        axes2[1].set_xlim(0.0, min(float(x[-1]), max(60.0, 1.4 * front)))
        # Also crop spacetime display width via a secondary xlim on the image axes
        axes2[0].set_xlim(0.0, min(float(x[-1]), max(80.0, 2.0 * front)))
        fig2.suptitle(
            "Mimura et al. (2000) — Section 3 / Fig. 3.4 (1D hard agar, Eq. 3.4)",
            fontsize=10,
        )
        out34 = out_dir / "mimura_bacterial_colony_section3_fig34.png"
        fig2.savefig(out34, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {out34}")

    return out_png


def laplacian_neumann(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Isotropic-leaning 9-point Laplacian with Neumann BC via edge padding.

    Mehrstellen stencil (1/(6 Δx²))·[[1,4,1],[4,-20,4],[1,4,1]] — equivalently
    (2/3)·Δ_5 + (1/6)·Δ_diag. Reduces the strong axis-aligned cross of pure
    5-point on degenerate fronts (Eden-like Eq. 4.2).
    """
    p = np.pad(field, 1, mode="edge")
    card = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    diag = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    return (4.0 * card + diag - 20.0 * field) / (6.0 * dx * dx)


def div_b_grad_phi_neumann(b: np.ndarray, phi: np.ndarray, dx: float) -> np.ndarray:
    """
    ∇·(b ∇φ) with Neumann reflection, isotropic-leaning 9-point FV fluxes.

    Cardinal faces (weight 2/3) plus diagonal faces (weight 1/6). When b ≡ const
    this recovers laplacian_neumann (Mehrstellen).
    """
    bp = np.pad(b, 1, mode="edge")
    pp = np.pad(phi, 1, mode="edge")
    inv_dx2 = 1.0 / (dx * dx)
    b_c = bp[1:-1, 1:-1]
    p_c = pp[1:-1, 1:-1]
    b_e = 0.5 * (b_c + bp[1:-1, 2:])
    b_w = 0.5 * (b_c + bp[1:-1, :-2])
    b_n = 0.5 * (b_c + bp[2:, 1:-1])
    b_s = 0.5 * (b_c + bp[:-2, 1:-1])
    div_card = inv_dx2 * (
        b_e * (pp[1:-1, 2:] - p_c)
        + b_w * (pp[1:-1, :-2] - p_c)
        + b_n * (pp[2:, 1:-1] - p_c)
        + b_s * (pp[:-2, 1:-1] - p_c)
    )
    b_ne = 0.5 * (b_c + bp[2:, 2:])
    b_nw = 0.5 * (b_c + bp[2:, :-2])
    b_se = 0.5 * (b_c + bp[:-2, 2:])
    b_sw = 0.5 * (b_c + bp[:-2, :-2])
    div_diag = inv_dx2 * (
        b_ne * (pp[2:, 2:] - p_c)
        + b_nw * (pp[2:, :-2] - p_c)
        + b_se * (pp[:-2, 2:] - p_c)
        + b_sw * (pp[:-2, :-2] - p_c)
    )
    return (2.0 / 3.0) * div_card + (1.0 / 6.0) * div_diag


def colony_radius(total: np.ndarray, dx: float, *, frac: float = 0.02) -> float:
    peak = float(total.max())
    if peak <= 0.0:
        return 0.0
    mask = total > frac * peak
    if not np.any(mask):
        return 0.0
    ys, xs = np.nonzero(mask)
    cy = (total.shape[0] - 1) / 2.0
    cx = (total.shape[1] - 1) / 2.0
    return float(dx * np.sqrt(np.max((ys - cy) ** 2 + (xs - cx) ** 2)))


def initial_fields(
    v0: float,
    ngrid: int,
    L: float,
    *,
    u0: float,
    r0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (2.7): central disk droplet, uniform nutrient, w ≡ 0."""
    x = np.linspace(-0.5 * L, 0.5 * L, ngrid)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    u = np.where(xx * xx + yy * yy <= r0 * r0, u0, 0.0).astype(np.float64)
    v = np.full((ngrid, ngrid), v0, dtype=np.float64)
    w = np.zeros_like(u)
    return u, v, w


def make_chi(ngrid: int, chi0: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Frozen spatial factors χ_u, χ_v ∈ [1−χ₀, 1+χ₀] for symmetry breaking."""
    if chi0 <= 0.0:
        ones = np.ones((ngrid, ngrid), dtype=np.float64)
        return ones, ones.copy()
    chi_u = 1.0 + chi0 * (2.0 * rng.random((ngrid, ngrid)) - 1.0)
    chi_v = 1.0 + chi0 * (2.0 * rng.random((ngrid, ngrid)) - 1.0)
    return chi_u, chi_v


def simulate_case(
    case: ColonyCase,
    *,
    L: float = 480.0,
    dx: float = 2.0,
    quick_scale: float = 1.0,
    n_frames: int = 0,
    stop_radius_frac: float = 0.46,
) -> np.ndarray | tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    """
    Integrate (4.1) or (4.2); return total density b = u + w.

    Stops early when the colony approaches the plate edge.
    """
    ngrid = int(round(L / dx)) + 1
    t_final = case.t_final * quick_scale
    # 9-point Mehrstellen spectral radius is 16/(3 Δx²) (vs 4/Δx² for 5-point).
    dt = min(case.dt, 0.45 * (dx * dx) / (16.0 / 3.0))
    n_steps = max(1, int(round(t_final / dt)))
    save_stride = max(1, n_steps // max(n_frames - 1, 1)) if n_frames > 0 else 0
    max_radius = stop_radius_frac * L

    rng = np.random.default_rng(case.seed)
    u, v, w = initial_fields(case.v0, ngrid, L, u0=case.u0, r0=case.r0)
    chi_u, chi_v = make_chi(ngrid, case.chi0, rng)

    history: list[tuple[float, np.ndarray]] = []
    t_now = 0.0

    for step in range(n_steps):
        a = conversion_rate(u, v)
        growth = u * v
        # Symmetry breaking: replace ∇φ by ∇(χ φ) as in Tomek–Šembera tests of MSM.
        if case.nonlinear:
            b = u + w
            diff_u = case.d * div_b_grad_phi_neumann(b, chi_u * u, dx)
        else:
            diff_u = case.d * laplacian_neumann(chi_u * u, dx)
        diff_v = laplacian_neumann(chi_v * v, dx)

        u = np.clip(u + dt * (diff_u + growth - a * u), 0.0, None)
        v = np.clip(v + dt * (diff_v - growth), 0.0, None)
        w = w + dt * (a * u)
        t_now = (step + 1) * dt

        if n_frames > 0 and (step % save_stride == 0 or step == n_steps - 1):
            history.append((t_now, (u + w).copy()))

        if step % 40 == 0 and colony_radius(u + w, dx) >= max_radius:
            if n_frames > 0 and (not history or history[-1][0] < t_now):
                history.append((t_now, (u + w).copy()))
            break

    total = u + w
    if n_frames > 0:
        return total, history
    return total


def display_density(
    total: np.ndarray,
    *,
    floor_frac: float = 0.015,
    enhance_rings: bool = False,
) -> np.ndarray:
    """
    Visual map of total density b.

    Ring contrast is weak on a filled plateau; enhance_rings subtracts a radial
    moving average so concentric bands stand out (still grounded in b).
    """
    field = np.asarray(total, dtype=np.float64).copy()
    peak = float(field.max())
    if peak > 0.0:
        field = np.where(field > floor_frac * peak, field, 0.0)
    if enhance_rings and peak > 0.0:
        n = field.shape[0]
        cy = cx = (n - 1) / 2.0
        yy, xx = np.ogrid[:n, :n]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_int = np.clip(np.rint(r).astype(np.int64), 0, n - 1)
        # Radial mean profile, then broadcast back.
        counts = np.bincount(r_int.ravel(), minlength=n)
        sums = np.bincount(r_int.ravel(), weights=field.ravel(), minlength=n)
        radial_mean = np.zeros(n, dtype=np.float64)
        nz = counts > 0
        radial_mean[nz] = sums[nz] / counts[nz]
        # Smooth the radial mean slightly.
        ker = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        ker /= ker.sum()
        pad = np.pad(radial_mean, 2, mode="edge")
        radial_smooth = np.convolve(pad, ker, mode="valid")
        baseline = radial_smooth[r_int]
        contrast = field - baseline
        # Keep only positive ring excess + a faint baseline so the colony envelope remains.
        field = np.clip(contrast, 0.0, None) + 0.15 * field
    return np.log1p(np.clip(field, 0.0, None))


def density_vmax(field: np.ndarray, percentile: float = 99.4) -> float:
    positive = field[field > 0.0]
    if positive.size == 0:
        return 1.0
    return max(float(np.percentile(positive, percentile)), 1e-6)


def plot_panel(ax: plt.Axes, total: np.ndarray, case: ColonyCase) -> None:
    field = display_density(total, enhance_rings=case.enhance_rings)
    vmax = density_vmax(field)
    ax.imshow(np.clip(field / vmax, 0.0, 1.0), cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    d_note = rf"${case.d}\,b$" if case.nonlinear else str(case.d)
    ax.set_title(f"({case.label}) {case.title}\n$v_0$={case.v0}, $d$={d_note}", fontsize=8)
    ax.axis("off")


def _figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3], mode="RGB")


def save_case_gif(
    case: ColonyCase,
    history: list[tuple[float, np.ndarray]],
    *,
    out_path: Path,
    duration_ms: int = 90,
) -> Path:
    fields = [
        display_density(total, enhance_rings=case.enhance_rings) for _, total in history
    ]
    vmax = max(density_vmax(f) for f in fields)
    frames: list[Image.Image] = []
    for t_val, field in zip((t for t, _ in history), fields):
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
        ax.imshow(
            np.clip(field / vmax, 0.0, 1.0),
            cmap="gray",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        d_note = rf"${case.d}\,b$" if case.nonlinear else str(case.d)
        ax.set_title(
            f"({case.label}) {case.title}  $t={t_val:.0f}$\n$v_0$={case.v0}, $d$={d_note}",
            fontsize=9,
        )
        ax.axis("off")
        frames.append(_figure_to_image(fig))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return out_path


def run_section4(
    cases: tuple[ColonyCase, ...] | None = None,
    *,
    L: float | None = None,
    dx: float = 2.0,
    out_png: Path | None = None,
    save_gifs: bool = True,
    n_gif_frames: int | None = None,
) -> Path:
    cases = cases or ColonyCase.default_section4_cases()
    quick = os.environ.get("MIMURA_QUICK", "").strip().lower() in {"1", "true", "yes"}
    default_L = L or (320.0 if quick else 480.0)
    qscale = 0.45 if quick else 1.0
    n_gif_frames = n_gif_frames or (24 if quick else 40)

    n_cols = len(cases)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 2.8), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ax, case in zip(axes, cases):
        case_dx = case.dx if case.dx is not None else dx
        case_L = case.L if case.L is not None else default_L
        if quick and case.dx is not None:
            # Keep ring resolution finer than default even in quick mode.
            case_dx = max(case.dx, 1.25)
            case_L = min(case_L, 360.0)
        print(
            f"Section 4 ({case.label}) {case.title}: L={case_L:g}, dx={case_dx:g} ...",
            flush=True,
        )
        if save_gifs:
            total, history = simulate_case(
                case, L=case_L, dx=case_dx, quick_scale=qscale, n_frames=n_gif_frames
            )
            gif_path = out_dir / f"mimura_bacterial_colony_{case.label}.gif"
            save_case_gif(case, history, out_path=gif_path)
            print(f"Saved {gif_path} ({len(history)} frames)")
        else:
            total = simulate_case(case, L=case_L, dx=case_dx, quick_scale=qscale)
        plot_panel(ax, total, case)

    fig.suptitle(
        "Mimura–Sakaguchi–Matsushita (2000) — Section 4 morphologies ($b=u+w$)\n"
        r"$a(u,v)=1/((1+u/a_1)(1+v/a_2))$, $a_1=1/2400$, $a_2=1/120$; "
        r"Neumann 9-pt FD; (B): Eq.~(4.2) $d(b)=d\,b$",
        fontsize=10,
    )

    out_png = out_png or out_dir / "mimura_bacterial_colony_section4.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    skip_2d = os.environ.get("MIMURA_SECTION3_ONLY", "").strip().lower() in {"1", "true", "yes"}
    out3 = run_section3()
    print(f"Section 3 done: {out3}")
    for case in Section3Case.default_fig32_cases():
        print(f"  Fig.3.2({case.label}) {case.title}: d={case.d}, v0={case.v0}")

    if skip_2d:
        return

    out4 = run_section4()
    print(f"Section 4 done: {out4}")
    for case in ColonyCase.default_section4_cases():
        mode = "nonlinear" if case.nonlinear else "normal"
        print(
            f"  ({case.label}) {case.title}: v0={case.v0}, d={case.d}, "
            f"{mode}, t_final={case.t_final}"
        )


if __name__ == "__main__":
    main()
