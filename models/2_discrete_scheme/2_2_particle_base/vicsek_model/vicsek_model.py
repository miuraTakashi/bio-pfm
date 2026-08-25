"""
Vicsek model (self-propelled particles with local alignment, periodic 2D box).

Discrete-time update (Vicsek et al., 1995):
  theta_i(t+1) = Arg( sum_{j in N_i} exp(i theta_j(t)) ) + xi_i(t)
  r_i(t+1)     = r_i(t) + v0 * [cos(theta_i), sin(theta_i)] * dt

where N_i is the metric neighborhood (distance <= r0) on a periodic domain, and
xi_i is uniform angular noise in [-eta, eta].

Outputs:
  vicsek_model.png — comparison of four regimes vs noise η
  vicsek_model_<slug>.gif — animation per regime (ordered, transition, partial, disordered)
  vicsek_model.gif — alias of ordered-regime GIF (backward compatible)
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize


@dataclass(frozen=True)
class VicsekCase:
    label: str
    eta: float
    slug: str


@dataclass
class VicsekRun:
    case: VicsekCase
    wrapped_hist: np.ndarray
    order_hist: np.ndarray
    theta_hist: np.ndarray


# Same N, L, seed; varying η shows order → disorder transition (Vicsek et al. 1995).
VICSEK_REGIMES: list[VicsekCase] = [
    VicsekCase(r"ordered ($\eta=0$)", 0.0, "ordered"),
    VicsekCase(r"near-critical ($\eta=0.5$)", 0.5, "transition"),
    VicsekCase(r"partial ($\eta=1.0$)", 1.0, "partial"),
    VicsekCase(r"disordered ($\eta=2.0$)", 2.0, "disordered"),
]


def _minimal_image(delta: np.ndarray, L: float) -> np.ndarray:
    """Periodic minimum-image displacement."""
    return (delta + 0.5 * L) % L - 0.5 * L


def simulate_vicsek(
    n_particles: int = 300,
    L: float = 20.0,
    v0: float = 0.2,
    r0: float = 1.0,
    eta: float = 0.5,
    dt: float = 1.0,
    n_steps: int = 600,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    wrapped_history : (n_steps+1, n_particles, 2)
    unwrapped_history : (n_steps+1, n_particles, 2)
    order_history : (n_steps+1,)
    theta_history : (n_steps+1, n_particles)
    """
    rng = np.random.default_rng(seed)

    pos = rng.uniform(0.0, L, size=(n_particles, 2))
    pos_unwrapped = pos.copy()
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_particles)

    wrapped_hist = np.empty((n_steps + 1, n_particles, 2), dtype=float)
    unwrapped_hist = np.empty((n_steps + 1, n_particles, 2), dtype=float)
    theta_hist = np.empty((n_steps + 1, n_particles), dtype=float)
    order_hist = np.empty(n_steps + 1, dtype=float)
    wrapped_hist[0] = pos
    unwrapped_hist[0] = pos_unwrapped
    theta_hist[0] = theta
    order_hist[0] = np.abs(np.mean(np.exp(1j * theta)))

    r0_sq = r0 * r0
    for step in range(1, n_steps + 1):
        dr = pos[:, None, :] - pos[None, :, :]
        dr = _minimal_image(dr, L)
        dist_sq = np.sum(dr * dr, axis=2)
        neighbors = dist_sq <= r0_sq

        sin_local = neighbors @ np.sin(theta)
        cos_local = neighbors @ np.cos(theta)
        theta = np.arctan2(sin_local, cos_local) + rng.uniform(-eta, eta, size=n_particles)

        vel = np.column_stack((np.cos(theta), np.sin(theta)))
        delta = v0 * dt * vel
        pos_unwrapped = pos_unwrapped + delta
        pos = (pos + delta) % L

        wrapped_hist[step] = pos
        unwrapped_hist[step] = pos_unwrapped
        theta_hist[step] = theta
        order_hist[step] = np.abs(np.mean(np.exp(1j * theta)))

    return wrapped_hist, unwrapped_hist, order_hist, theta_hist


def resolve_regimes(quick: bool) -> list[VicsekCase]:
    if quick:
        return [
            VicsekCase(r"ordered ($\eta=0$)", 0.0, "ordered"),
            VicsekCase(r"disordered ($\eta=2$)", 2.0, "disordered"),
        ]
    return list(VICSEK_REGIMES)


def run_regimes(
    cases: list[VicsekCase],
    *,
    n_particles: int = 300,
    L: float = 20.0,
    n_steps: int = 600,
    seed: int = 42,
) -> list[VicsekRun]:
    runs: list[VicsekRun] = []
    for case in cases:
        print(f"Vicsek: {case.label}  (eta={case.eta:g}, steps={n_steps})")
        wrapped, _unwrapped, order_hist, theta_hist = simulate_vicsek(
            n_particles=n_particles,
            L=L,
            eta=case.eta,
            n_steps=n_steps,
            seed=seed,
        )
        runs.append(VicsekRun(case, wrapped, order_hist, theta_hist))
    return runs


def save_regime_comparison(
    runs: list[VicsekRun],
    out_png: Path,
    *,
    L: float = 20.0,
    n_particles: int = 300,
) -> None:
    ncols = len(runs)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.0), constrained_layout=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    norm = Normalize(vmin=0.0, vmax=2.0 * np.pi)
    cmap = plt.cm.hsv

    for col, run in enumerate(runs):
        case = run.case
        pos = run.wrapped_hist[-1]
        theta = run.theta_hist[-1]
        phi_final = float(run.order_hist[-1])

        ax_snap = axes[0, col]
        ax_snap.scatter(
            pos[:, 0],
            pos[:, 1],
            c=theta,
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.85,
            linewidths=0,
        )
        stride = max(1, n_particles // 35)
        ax_snap.quiver(
            pos[::stride, 0],
            pos[::stride, 1],
            np.cos(theta[::stride]),
            np.sin(theta[::stride]),
            color="0.15",
            alpha=0.55,
            width=0.003,
            scale=4.5,
            headwidth=3.5,
            headlength=4,
        )
        ax_snap.set_xlim(0.0, L)
        ax_snap.set_ylim(0.0, L)
        ax_snap.set_aspect("equal")
        ax_snap.set_box_aspect(1)
        ax_snap.set_title(f"{case.label}\n$\\Phi_{{final}}$={phi_final:.3f}", fontsize=9)
        ax_snap.set_xlabel("x")
        if col == 0:
            ax_snap.set_ylabel("y")
        ax_snap.grid(alpha=0.25)

        ax_ord = axes[1, col]
        t = np.arange(run.order_hist.size, dtype=float)
        ax_ord.plot(t, run.order_hist, color="tab:red", lw=1.5)
        ax_ord.axhline(phi_final, color="0.4", ls=":", lw=0.8)
        ax_ord.set_ylim(0.0, 1.05)
        ax_ord.set_xlabel("step")
        if col == 0:
            ax_ord.set_ylabel(r"$\Phi(t)$")
        ax_ord.set_title(r"Polar order $\Phi=|\langle e^{i\theta}\rangle|$", fontsize=9)
        ax_ord.grid(alpha=0.3)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[0, :],
        fraction=0.025,
        pad=0.02,
        label=r"heading $\theta$",
    )
    fig.suptitle(
        rf"Vicsek model — noise-induced order/disorder ($N={n_particles}$, $L={L:g}$, same seed)",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def save_regime_gif(
    run: VicsekRun,
    out_gif: Path,
    *,
    L: float = 20.0,
    quick: bool = False,
) -> None:
    wrapped_hist = run.wrapped_hist
    theta_hist = run.theta_hist
    order_hist = run.order_hist
    case = run.case

    frame_stride = 4 if quick else 3
    frame_ids = list(range(0, wrapped_hist.shape[0], frame_stride))
    if frame_ids[-1] != wrapped_hist.shape[0] - 1:
        frame_ids.append(wrapped_hist.shape[0] - 1)

    norm = Normalize(vmin=0.0, vmax=2.0 * np.pi)
    cmap = plt.cm.hsv

    fig, ax = plt.subplots(figsize=(5, 5))
    scat = ax.scatter([], [], s=12, c=[], cmap=cmap, norm=norm, alpha=0.85)
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    title = ax.set_title(case.label, fontsize=10)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label=r"$\theta$",
    )

    def _update(k: int):
        idx = frame_ids[k]
        xy = wrapped_hist[idx]
        th = theta_hist[idx]
        scat.set_offsets(xy)
        scat.set_array(th)
        phi = float(order_hist[idx])
        title.set_text(f"{case.label}\nstep={idx}  Φ={phi:.3f}")
        return scat, title

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=50,
        blit=False,
        repeat=True,
    )
    anim.save(str(out_gif), writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved {out_gif}")


def main() -> None:
    quick = os.environ.get("VICSEK_QUICK", "").strip().lower() in ("1", "true", "yes")
    out_dir = Path(__file__).resolve().parent
    n_steps = 300 if quick else 600
    cases = resolve_regimes(quick)
    L = 20.0

    runs = run_regimes(cases, n_steps=n_steps)
    save_regime_comparison(runs, out_dir / "results/vicsek_model.png", L=L)

    for run in runs:
        gif_path = out_dir / f"vicsek_model_{run.case.slug}.gif"
        save_regime_gif(run, gif_path, L=L, quick=quick)
        if run.case.slug == "ordered":
            legacy = out_dir / "results/vicsek_model.gif"
            shutil.copy2(gif_path, legacy)
            print(f"Saved {legacy}  (copy of ordered regime)")


if __name__ == "__main__":
    main()
