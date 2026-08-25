"""
群れ振動子モデル（Mathematica/2_discrete_scheme/2_2_particle_base/swarm_oscillators/SwarmOscillators8.nb から移植）。

位置 r_i と位相 ψ_i を持つ振動子が相互作用する:
  dψ_i/dt = Σ_j exp(-|r_j-r_i|) sin(ψ_j-ψ_i + α|r_j-r_i| - c1)
  dr_i/dt = c3 * Σ_j normalize(r_j-r_i) exp(-|r_j-r_i|) sin(ψ_j-ψ_i + α|r_j-r_i| - c2)

既定の単一デモ: n=50, L=50, c1=1.5, c2=0.5, c3=2.0, α=0.5, dt=0.05

相 A–M（13 種）の GIF は Iwasa et al., Phys. Lett. A 376 (2012) 2117–2121
（Fig.1–5 の (c1, c2)；対照用 PDF は手元の References/SwarmOscillatorsPatterns.pdf 等）
と本文の L=10, N=50, c3=α=1.0 に合わせて生成する。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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

from matplotlib.animation import FuncAnimation, PillowWriter

# Iwasa et al., Phys. Lett. A 376 (2012) 2117–2121, Figs. 1–5 (phases A–M)
IWASA_ET_AL_PHASES: list[tuple[str, float, float, str]] = [
    ("A", 5.5, 5.5, "aggregation, synchronized"),
    ("B", 4.0, 1.0, "aggregation, not synchronized, static"),
    ("C", 2.0, 2.0, "static triangular lattice, synchronized"),
    ("D", 2.0, 3.0, "static square lattice, synchronized"),
    ("E", 1.0, 3.0, "lattice, steady flow, wave of internal state"),
    ("F", 1.0, 1.0, "2D cluster(s), synchronized, steady"),
    ("G", 2.0, 1.0, "1D / quasi-1D cluster(s), synchronized"),
    ("H", 3.0, 5.0, "waving curve"),
    ("I", 3.5, 4.5, "rotating polygon"),
    ("J", 3.5, 3.5, "rotating circle, wave of internal state"),
    ("K", 2.0, 5.0, "scattering (small phase diff → approach)"),
    ("L", 6.0, 3.0, "scattering (large phase diff → approach)"),
    ("M", 5.0, 3.0, "cluster, swarming"),
]


def simulate_swarm(
    n: int = 50,
    L: float = 50.0,
    c1: float = 1.5,
    c2: float = 0.5,
    c3: float = 2.0,
    alpha: float = 0.5,
    dt: float = 0.05,
    n_steps: int = 5000,
    snapshot_interval: int = 1,
    seed: int = 42,
) -> list:
    rng = np.random.default_rng(seed)
    psi = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0, L, (n, 2))

    snapshots = [(r.copy(), psi.copy())]

    for step in range(n_steps):
        dr = r[np.newaxis, :, :] - r[:, np.newaxis, :]
        dpsi = psi[np.newaxis, :] - psi[:, np.newaxis]
        dist = np.linalg.norm(dr, axis=2)
        np.fill_diagonal(dist, 1.0)

        w = np.exp(-dist)
        np.fill_diagonal(w, 0.0)

        d_psi = np.sum(w * np.sin(dpsi + alpha * dist - c1), axis=1)
        psi = (psi + dt * d_psi) % (2 * np.pi)

        norm_dr = dr / dist[:, :, np.newaxis]
        d_r = c3 * np.sum(
            w[:, :, np.newaxis]
            * np.sin((dpsi + alpha * dist - c2)[:, :, np.newaxis])
            * norm_dr,
            axis=1)
        r = (r + dt * d_r) % L

        if (step + 1) % snapshot_interval == 0:
            snapshots.append((r.copy(), psi.copy()))

    return snapshots


def make_gif(
    snapshots,
    L: float = 8.0,
    dt: float = 0.05,
    snapshot_interval: int = 10,
    filename: Path | str | None = None,
    fps: int = 20,
    max_frames: int = 500,
    title_prefix: str = "Swarm Oscillators",
):
    if filename is None:
        filename = Path(__file__).with_name("swarm_oscillators.gif")
    fig, ax = plt.subplots(figsize=(6, 6))

    r0, psi0 = snapshots[0]
    sc = ap.atlas_scatter(
        ax,
        r0[:, 0],
        r0[:, 1],
        c=psi0 % (2 * np.pi),
        phase=True,
        vmin=0,
        vmax=2 * np.pi,
        s=60,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_title(f"{title_prefix}  t=0.0")
    plt.colorbar(sc, ax=ax, label="phase (rad)")

    total_frames = len(snapshots)
    skip = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, skip))

    def update(idx):
        i = frame_indices[idx]
        r, psi = snapshots[i]
        sc.set_offsets(np.column_stack([r[:, 0], r[:, 1]]))
        sc.set_array(psi % (2 * np.pi))
        t = i * snapshot_interval * dt
        ax.set_title(f"{title_prefix}  t={t:.1f}")
        return (sc,)

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=True, interval=1000 // fps)
    anim.save(str(filename), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {filename}  ({len(frame_indices)} frames)")


def generate_iwasa_et_al_phase_gifs(
    out_dir: Path | None = None,
    *,
    n: int = 50,
    L: float = 10.0,
    c3: float = 1.0,
    alpha: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 6000,
    snapshot_interval: int = 20,
    seed: int = 42,
    fps: int = 16,
    max_frames: int = 450,
    phases: list[tuple[str, float, float, str]] | None = None,
) -> None:
    """Generate one GIF per phase (A–M) matching Iwasa et al. (2012) Fig.1–5."""
    out_dir = out_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_list = phases if phases is not None else IWASA_ET_AL_PHASES
    finals: list[tuple[str, float, float, str, np.ndarray, np.ndarray]] = []

    for letter, c1, c2, desc in phase_list:
        snapshots = simulate_swarm(
            n=n,
            L=L,
            c1=c1,
            c2=c2,
            c3=c3,
            alpha=alpha,
            dt=dt,
            n_steps=n_steps,
            snapshot_interval=snapshot_interval,
            seed=seed,
        )
        gif_path = out_dir / f"swarm_oscillators_phase_{letter}.gif"
        title = f"Phase {letter} (c1={c1}, c2={c2})"
        make_gif(
            snapshots,
            L=L,
            dt=dt,
            snapshot_interval=snapshot_interval,
            filename=gif_path,
            fps=fps,
            max_frames=max_frames,
            title_prefix=title,
        )
        r, psi = snapshots[-1]
        finals.append((letter, c1, c2, desc, r, psi))

    n_phases = len(finals)
    if n_phases == 13:
        grid_fig, grid_axes = plt.subplots(4, 4, figsize=(14, 13))
        axes_flat = grid_axes.ravel()
    else:
        ncols = min(4, n_phases)
        nrows = int(np.ceil(n_phases / ncols))
        grid_fig, axes_flat = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        axes_flat = np.atleast_1d(axes_flat).ravel()

    for idx, (letter, c1, c2, desc, r, psi) in enumerate(finals):
        gax = axes_flat[idx]
        ap.atlas_scatter(
            gax,
            r[:, 0],
            r[:, 1],
            c=psi % (2 * np.pi),
            phase=True,
            vmin=0,
            vmax=2 * np.pi,
            s=18,
            edgecolors="k",
            linewidths=0.2,
        )
        gax.set_xlim(0, L)
        gax.set_ylim(0, L)
        gax.set_aspect("equal")
        gax.set_title(f"Phase {letter}\n({desc})", fontsize=7)
        gax.set_xticks([])
        gax.set_yticks([])

    for j in range(len(finals), len(axes_flat)):
        axes_flat[j].set_visible(False)

    grid_fig.suptitle(
        "Iwasa et al. (2012) Phys. Lett. A — final frame "
        f"(N={n}, L={L}, c3=α={alpha})",
        fontsize=11,
    )
    plt.tight_layout()
    grid_png = out_dir / "results/swarm_oscillators_phases_A_M_grid.png"
    grid_fig.savefig(grid_png, dpi=150)
    plt.close(grid_fig)
    print(f"Saved {grid_png}")


def main():
    quick = os.environ.get("SWARM_OSCILLATORS_QUICK", "").strip() not in ("", "0", "false", "False")
    if quick:
        generate_iwasa_et_al_phase_gifs(
            n_steps=800,
            snapshot_interval=40,
            phases=IWASA_ET_AL_PHASES[:1],
            max_frames=120,
            fps=12,
        )
        return

    generate_iwasa_et_al_phase_gifs()


if __name__ == "__main__":
    main()
