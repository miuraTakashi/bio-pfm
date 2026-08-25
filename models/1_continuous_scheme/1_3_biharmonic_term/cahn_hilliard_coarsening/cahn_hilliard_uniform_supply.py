"""
Cahn-Hilliard + uniform source term (2D, periodic).

Equation:
    dc/dt = M * Laplacian(mu) + S0
    mu = f'(c) - eps^2 * Laplacian(c)

with a quartic double-well bulk energy on c in [0, 1]:
    f(c) = 0.5 * c^2 * (1 - c)^2
    f'(c) = c - 3 c^2 + 2 c^3

Spinodal region for this choice is:
    f''(c) = 1 - 6c + 6c^2 < 0
    <=> c in ((3 - sqrt(3))/6, (3 + sqrt(3))/6) ≈ (0.211, 0.789)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap


SPINODAL_C_LOW = (3.0 - np.sqrt(3.0)) / 6.0
SPINODAL_C_HIGH = (3.0 + np.sqrt(3.0)) / 6.0


def f_prime(c: np.ndarray) -> np.ndarray:
    """Derivative of 0.5*c^2*(1-c)^2."""
    return c - 3.0 * c**2 + 2.0 * c**3


def k_squared_rfft_2d(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    """k^2 array for rfft2 layout."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing="ij")
    return kx2 + ky2


def step_ch_uniform_supply_2d(
    c: np.ndarray,
    dt: float,
    M: float,
    eps: float,
    k2: np.ndarray,
    source_rate: float,
    thermal_noise: np.ndarray | None = None,
) -> np.ndarray:
    """One semi-implicit step with spatially uniform source S0 and optional thermal noise."""
    n0, n1 = c.shape
    # Add thermal noise before CH step so it seeds spinodal growth naturally.
    c_in = c if thermal_noise is None else c + thermal_noise
    cp_hat = np.fft.rfft2(f_prime(c_in), norm="ortho")
    c_hat = np.fft.rfft2(c_in, norm="ortho")
    s_hat = np.fft.rfft2(np.full_like(c_in, source_rate), norm="ortho")

    denom = 1.0 + dt * M * (eps**2) * (k2**2)
    numer = c_hat + dt * (-M * k2 * cp_hat + s_hat)
    return np.fft.irfft2(numer / denom, s=(n0, n1), norm="ortho")


def simulate_uniform_supply(
    n: int = 128,
    eps: float = 0.6,
    M: float = 1.0,
    dx: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 90_000,
    save_every: int = 1_500,
    c_mean0: float = 0.0,
    noise_amp: float = 0.02,
    thermal_noise_amp: float = 5e-4,
    source_rate: float = 4.0e-4,
    seed: int = 7,
) -> tuple[list[np.ndarray], list[float], np.ndarray]:
    """2D CH simulation where total mass increases slowly by constant source.

    thermal_noise_amp is added as white noise each step to continuously seed
    spinodal decomposition once the mean enters the unstable band.
    Without it, the initial noise decays to ~0 before spinodal is reached.
    """
    rng = np.random.default_rng(seed)
    c = c_mean0 + noise_amp * rng.standard_normal(size=(n, n))
    k2 = k_squared_rfft_2d(n, n, dx, dx)

    snaps = [c.copy()]
    times = [0.0]
    means = [float(np.mean(c))]

    for step in range(n_steps):
        # Per-step thermal noise seeds spinodal growth (mimics thermal fluctuations).
        th = thermal_noise_amp * rng.standard_normal(size=(n, n))
        c = step_ch_uniform_supply_2d(c, dt, M, eps, k2, source_rate, th)
        if (step + 1) % save_every == 0:
            snaps.append(c.copy())
            times.append((step + 1) * dt)
            means.append(float(np.mean(c)))

    return snaps, times, np.array(means, dtype=float)


def save_outputs(
    snaps: list[np.ndarray],
    times: list[float],
    means: np.ndarray,
    out_dir: Path,
) -> None:
    """Save PNG (summary) and GIF (2D time series)."""
    out_png = out_dir / "results/cahn_hilliard_uniform_supply.png"
    out_gif = out_dir / "results/cahn_hilliard_uniform_supply.gif"

    pick = np.linspace(0, len(times) - 1, 5, dtype=int)
    pick = np.unique(pick)
    fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.3), constrained_layout=True)

    im = None
    for col, idx in enumerate(pick):
        field = snaps[idx]
        # Show fluctuation around instantaneous mean for better pattern visibility.
        fluct = field - float(np.mean(field))
        im = ap.atlas_imshow(
            axes[0, col],
            fluct.T,
            heatmap="scalar",
            origin="lower",
            interpolation="nearest",
            vmin=-0.18,
            vmax=0.18,
        )
        axes[0, col].set_title(f"c (t={times[idx]:.0f})", fontsize=9)
        axes[0, col].axis("off")

        t_series = np.array(times[: idx + 1], dtype=float)
        m_series = means[: idx + 1]
        axes[1, col].plot(t_series, m_series, color="C1", lw=1.3)
        axes[1, col].axhline(SPINODAL_C_LOW, color="k", ls="--", lw=0.8, alpha=0.7)
        axes[1, col].axhline(SPINODAL_C_HIGH, color="k", ls="--", lw=0.8, alpha=0.7)
        axes[1, col].set_ylim(0.0, 1.0)
        axes[1, col].set_xlabel("t")
        if col == 0:
            axes[1, col].set_ylabel("mean(c)")
        axes[1, col].grid(alpha=0.25)
        axes[1, col].set_title("mean concentration", fontsize=9)

    if im is not None:
        fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.02, label="c - mean(c)")
    fig.suptitle(
        "Cahn-Hilliard with uniform source: total amount increases from c_mean~0",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.4, 4.8))
    im2 = ap.atlas_imshow(
        ax2,
        (snaps[0] - float(np.mean(snaps[0]))).T,
        heatmap="scalar",
        origin="lower",
        interpolation="nearest",
        vmin=-0.18,
        vmax=0.18,
    )
    ax2.set_title(f"c (t={times[0]:.0f}, mean={means[0]:.3f})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    cbar = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("c - mean(c)")
    fig2.tight_layout()

    def _update(i: int):
        im2.set_data((snaps[i] - float(np.mean(snaps[i]))).T)
        ax2.set_title(f"c (t={times[i]:.0f}, mean={means[i]:.3f})")
        return (im2,)

    ani = FuncAnimation(fig2, _update, frames=len(snaps), interval=120, blit=False, repeat=True)
    ani.save(out_gif, writer=PillowWriter(fps=10))
    plt.close(fig2)

    print(f"Saved {out_png}")
    print(f"Saved {out_gif}")
    print(
        "Spinodal band for this f(c): "
        f"{SPINODAL_C_LOW:.3f} < c < {SPINODAL_C_HIGH:.3f}, "
        f"mean(c): {means[0]:.3f} -> {means[-1]:.3f}"
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    snaps, times, means = simulate_uniform_supply()
    save_outputs(snaps, times, means, out_dir)


if __name__ == "__main__":
    main()

