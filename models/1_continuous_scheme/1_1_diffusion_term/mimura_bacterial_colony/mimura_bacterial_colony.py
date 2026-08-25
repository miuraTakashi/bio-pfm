"""
Matsushita et al. (1997) unified bacterial colony model — Fig. 5 morphologies.

Reference: Physica A 249 (1998) 517–524; Eqs. (1)–(5), non-dimensional form.

  ∂b/∂t = ∇·(D(b,n)∇b) + g(n)b − a(b,n)b
  ∂s/∂t = a(b,n)b
  ∂n/∂t = ∇²n − g(n)b,   g(n) = n

Active density b and nutrient n are advanced with a semi-implicit FFT scheme
(same pattern as `turing_2d.py`: reaction explicit, diffusion implicit on a
periodic grid). A fixed exterior buffer enforces n = n₀ and b = s = 0 there,
approximating growth on an infinite nutrient bath (periodic FFT alone depletes
nutrient globally). Inactive cells s are integrated explicitly; Fig. 5 shows b+s.

Fig. 5b uses density-dependent motility D = d·b (Eden-like). With
∇·(d b ∇b) = d b ∇²b + d|∇b|², the Laplacian term is advanced semi-implicitly
(FFT, coefficient d·bⁿ frozen per step) and d|∇bⁿ|² is treated explicitly.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))


@dataclass(frozen=True)
class Fig5Case:
    """One panel of Fig. 5 (paper caption parameters)."""

    label: str
    title: str
    n0: float
    d: float
    eden_diffusion: bool = False
    t_final: float = 2000.0
    dt: float = 0.03
    a0: float = 0.4
    a1: float = 2.0
    a2: float = 0.55
    seed: int = 1
    d_floor: float = 0.2
    b_peak: float = 0.9

    @staticmethod
    def default_fig5_cases() -> tuple["Fig5Case", ...]:
        """Paper Fig. 5 caption: (n0, d); panel (b) uses d = 0.05·b (Physica A 249, p.523)."""
        return (
            Fig5Case("a", "DLA-like", 0.98, 0.05, t_final=5500.0, a0=0.28, a1=1.4, a2=0.4, seed=11),
            Fig5Case(
                "b",
                "Eden-like",
                1.2,
                0.05,
                eden_diffusion=True,
                t_final=1400.0,
                a0=0.46,
                a1=2.2,
                a2=0.65,
                d_floor=0.12,
                seed=12,
            ),
            Fig5Case(
                "c", "concentric ring-like", 1.2, 0.05, t_final=3600.0, a0=0.40, a1=3.8, a2=0.85, seed=13
            ),
            Fig5Case("d", "disk-like", 1.5, 0.12, t_final=320.0, a0=0.44, a1=2.5, a2=0.45, seed=14),
            Fig5Case("e", "DBM-like", 0.855, 0.12, t_final=5000.0, a0=0.27, a1=1.8, a2=0.35, seed=15),
        )


def build_implicit_kernel_2d(grid_size: int, dt: float, d_coeff: float, dx: float) -> np.ndarray:
    """(I − dt D ∇²)⁻¹ on a periodic 2D grid (5-point Laplacian), as in turing_2d."""
    kernel = np.zeros((grid_size, grid_size), dtype=np.float64)
    c = dt * d_coeff / (dx * dx)
    kernel[0, 0] = 1.0 + 4.0 * c
    kernel[1, 0] = -c
    kernel[-1, 0] = -c
    kernel[0, 1] = -c
    kernel[0, -1] = -c
    return 1.0 / np.fft.fft2(kernel)


def diffuse_semi_implicit_2d(rhs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """One semi-implicit diffusion step: (I − dt D ∇²)⁻¹ rhs (periodic FFT)."""
    return np.real(np.fft.ifft2(kernel * np.fft.fft2(rhs)))


def eden_grad_squared_term(b: np.ndarray, d: float, dx: float) -> np.ndarray:
    """Explicit d|∇b|² from ∇·(d b ∇b) = d b ∇²b + d|∇b|²."""
    bx = (np.roll(b, -1, 1) - np.roll(b, 1, 1)) / (2.0 * dx)
    by = (np.roll(b, -1, 0) - np.roll(b, 1, 0)) / (2.0 * dx)
    return d * (bx * bx + by * by)


def eden_diffusion_coefficient(case: Fig5Case, b: np.ndarray, pad: int) -> float:
    """
    Effective D for semi-implicit FFT step when D = d·b (frozen at bⁿ).

    Uses the mean of d·max(b, d_floor) over the active colony interior.
    """
    core = np.maximum(b[pad:-pad, pad:-pad], case.d_floor)
    active = core > case.d_floor
    if np.any(active):
        return case.d * float(np.mean(core[active]))
    return case.d * case.d_floor


def deactivation_rate(
    b: np.ndarray,
    n: np.ndarray,
    *,
    a0: float,
    a1: float,
    a2: float,
    n_ref: float,
) -> np.ndarray:
    """
    a(b, n) decreasing in b and n (Physica A 282, 2000: a0/((1+a1 b)(1+a2 n)) form).

    a = a0 / ((1 + a1 b) (1 + a2 max(0, 1 − n/n_ref)))
    """
    factor_b = 1.0 + a1 * np.clip(b, 0.0, None)
    factor_n = 1.0 + a2 * np.clip(1.0 - n / n_ref, 0.0, 1.0)
    return a0 / (factor_b * factor_n)


def exterior_buffer_pad(ngrid: int) -> int:
    return max(12, ngrid // 12)


def apply_exterior_buffer(
    b: np.ndarray,
    n: np.ndarray,
    s: np.ndarray,
    *,
    n0: float,
    pad: int,
) -> None:
    """Fixed nutrient bath and zero cells in a rim (mitigates periodic wrap-around)."""
    b[:pad, :] = b[-pad:, :] = b[:, :pad] = b[:, -pad:] = 0.0
    s[:pad, :] = s[-pad:, :] = s[:, :pad] = s[:, -pad:] = 0.0
    n[:pad, :] = n[-pad:, :] = n[:, :pad] = n[:, -pad:] = n0


def initial_fields(
    n0: float,
    ngrid: int,
    *,
    b_peak: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper Eq. (5): point-like inoculum b₀(r̃), uniform nutrient n₀, s = 0.
    """
    b = np.zeros((ngrid, ngrid), dtype=np.float64)
    b[ngrid // 2, ngrid // 2] = b_peak
    n = np.full((ngrid, ngrid), n0, dtype=np.float64)
    s = np.zeros_like(b)
    return b, n, s


def _interface_noise(b: np.ndarray, rng: np.random.Generator, dt: float, amp: float) -> np.ndarray:
    """Isotropic perturbation on the growing front (avoids grid-aligned artefacts)."""
    active = b > 0.04 * float(b.max())
    if not np.any(active):
        return b
    noise = rng.standard_normal(b.shape)
    lap = (
        np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1) - 4.0 * b
    )
    front = active & (np.abs(lap) > 0.02 * float(b.max()))
    return b + dt * amp * noise * front.astype(np.float64)


def simulate_fig5_case(
    case: Fig5Case,
    *,
    ngrid: int = 200,
    dx: float = 1.0,
    quick_scale: float = 1.0,
    n_frames: int = 0,
) -> np.ndarray | tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    """
    Integrate (1)–(3) and return total cell density b + s.

    If n_frames > 0, also return [(t, total), ...] snapshots along the trajectory.
    """
    t_final = case.t_final * quick_scale
    dt = case.dt
    n_steps = max(1, int(round(t_final / dt)))
    pad = exterior_buffer_pad(ngrid)
    save_stride = max(1, n_steps // max(n_frames - 1, 1)) if n_frames > 0 else 0

    b, n, s = initial_fields(case.n0, ngrid, b_peak=case.b_peak)
    apply_exterior_buffer(b, n, s, n0=case.n0, pad=pad)

    kb_const: np.ndarray | None = None
    if not case.eden_diffusion:
        kb_const = build_implicit_kernel_2d(ngrid, dt, case.d, dx)
    kn = build_implicit_kernel_2d(ngrid, dt, 1.0, dx)

    b_max = 12.0
    s_max = 12.0
    rng = np.random.default_rng(case.seed + 1000)
    history: list[tuple[float, np.ndarray]] = []
    branch_noise = case.label in {"a", "e"}

    for step in range(n_steps):
        a = deactivation_rate(b, n, a0=case.a0, a1=case.a1, a2=case.a2, n_ref=case.n0)
        growth = n * b
        reaction_b = growth - a * b
        if case.eden_diffusion:
            reaction_b = reaction_b + eden_grad_squared_term(b, case.d, dx)
            kb = build_implicit_kernel_2d(
                ngrid, dt, eden_diffusion_coefficient(case, b, pad), dx
            )
        else:
            kb = kb_const
        fb = b + dt * reaction_b
        b = np.clip(diffuse_semi_implicit_2d(fb, kb), 0.0, b_max)

        fn = n + dt * (-growth)
        n = diffuse_semi_implicit_2d(fn, kn)
        s = s + dt * a * b

        n = np.clip(n, 0.0, None)
        s = np.clip(s, 0.0, s_max)
        apply_exterior_buffer(b, n, s, n0=case.n0, pad=pad)
        if branch_noise:
            b = np.clip(_interface_noise(b, rng, dt, amp=0.35), 0.0, b_max)

        if n_frames > 0 and (step % save_stride == 0 or step == n_steps - 1):
            history.append((step * dt, (b + s).copy()))

    total = b + s
    if n_frames > 0:
        return total, history
    return total


def crop_interior(field: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return field
    return field[pad:-pad, pad:-pad]


def display_density(total: np.ndarray, pad: int, *, rim: int = 5) -> np.ndarray:
    """Log-compressed interior field for visualization (paper: bright colony on dark)."""
    inner = crop_interior(total, pad).copy()
    inner[:rim, :] = inner[-rim:, :] = inner[:, :rim] = inner[:, -rim:] = 0.0
    peak = float(inner.max())
    if peak > 0.0:
        inner = np.where(inner > 0.03 * peak, inner, 0.0)
    return np.log1p(np.clip(inner, 0.0, None))


def density_vmax(field: np.ndarray, percentile: float = 99.2) -> float:
    positive = field[field > 0.0]
    if positive.size == 0:
        return 1.0
    return max(float(np.percentile(positive, percentile)), 1e-6)


def plot_fig5_panel(ax: plt.Axes, total: np.ndarray, case: Fig5Case, *, pad: int) -> None:
    """Paper Fig. 5 style: bright colony (b+s) on dark background."""
    field = display_density(total, pad)
    vmax = density_vmax(field)
    ax.imshow(np.clip(field / vmax, 0.0, 1.0), cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    d_note = r"$0.05\,b$" if case.eden_diffusion else str(case.d)
    ax.set_title(f"({case.label}) {case.title}\n$n_0$={case.n0}, $d$={d_note}", fontsize=8)
    ax.axis("off")


def _figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3], mode="RGB")


def save_case_gif(
    case: Fig5Case,
    history: list[tuple[float, np.ndarray]],
    *,
    pad: int,
    out_path: Path,
    duration_ms: int = 90,
) -> Path:
    """Write formation process GIF for one Fig. 5 panel."""
    fields = [display_density(total, pad) for _, total in history]
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
        d_note = r"$0.05\,b$" if case.eden_diffusion else str(case.d)
        ax.set_title(
            f"({case.label}) {case.title}  $t={t_val:.0f}$\n$n_0$={case.n0}, $d$={d_note}",
            fontsize=9,
        )
        ax.axis("off")
        frames.append(_figure_to_image(fig))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return out_path


def run_fig5(
    cases: tuple[Fig5Case, ...] | None = None,
    *,
    ngrid: int | None = None,
    out_png: Path | None = None,
    save_gifs: bool = True,
    n_gif_frames: int | None = None,
) -> Path:
    cases = cases or Fig5Case.default_fig5_cases()
    quick = os.environ.get("MIMURA_FIG5_QUICK", "").strip().lower() in {"1", "true", "yes"}
    ngrid = ngrid or (160 if quick else 220)
    qscale = 0.55 if quick else 1.0
    n_gif_frames = n_gif_frames or (48 if quick else 72)

    n_cols = len(cases)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 2.8), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = exterior_buffer_pad(ngrid)

    for ax, case in zip(axes, cases):
        if save_gifs:
            total, history = simulate_fig5_case(
                case, ngrid=ngrid, quick_scale=qscale, n_frames=n_gif_frames
            )
            gif_path = out_dir / f"mimura_bacterial_colony_fig5_{case.label}.gif"
            save_case_gif(case, history, pad=pad, out_path=gif_path)
            print(f"Saved {gif_path} ({len(history)} frames)")
        else:
            total = simulate_fig5_case(case, ngrid=ngrid, quick_scale=qscale)
        plot_fig5_panel(ax, total, case, pad=pad)

    fig.suptitle(
        "Mimura unified colony model — Fig. 5 (total cells $b+s$)\n"
        "semi-implicit diffusion (FFT); exterior nutrient buffer; panel (b): $D=0.05\\,b$",
        fontsize=10,
    )

    out_png = out_png or out_dir / "mimura_bacterial_colony_fig5.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    out = run_fig5()
    print(f"Saved {out}")
    for case in Fig5Case.default_fig5_cases():
        print(
            f"  ({case.label}) n0={case.n0}, d={case.d}, a0={case.a0}, "
            f"eden={case.eden_diffusion}"
        )


if __name__ == "__main__":
    main()
