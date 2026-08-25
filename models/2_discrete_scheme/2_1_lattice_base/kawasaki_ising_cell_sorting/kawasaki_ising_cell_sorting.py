#!/usr/bin/env python3
"""
Mochizuki, Iwasa & Takeda (1996) J. Theor. Biol. — Fig. 3 cell-sorting patterns.

Spin-exchange Ising (Kawasaki) dynamics on a 2D square lattice (Neumann, z=4):
  - B/W cells on N×N lattice, periodic boundaries, no vacancies.
  - Random adjacent pair chosen each micro-attempt; B–W swaps accepted with
    Pr = min(1, 2m / (1 + exp(−ΔE/m))),  ΔE = A (n_W(B) − n_B(B)).
  - One **time unit** = N² micro-attempts (standard Kawasaki sweep on N² sites).

Fig. 3 settings: 100×100, r_B=0.5, m=0.5, 10 000 time units, random IC,
A/m ∈ {−2, −1, 0, 0.6, 1.2, 2, 4, 6}; display central 40×40 crop.

Outputs (this directory):
  kawasaki_ising_cell_sorting_fig3.png
  kawasaki_ising_cell_sorting_Am*.gif  — one time series per A/m (Fig. 3 conditions)
"""
from __future__ import annotations

import os
import sys
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
import atlas_plotting as ap

# --- paper Fig. 3 ---
LATTICE_SIZE = 100
FRACTION_BLACK = 0.5
MOTILITY = 0.5
TIME_UNITS = 10_000
A_OVER_M_VALUES = (-2.0, -1.0, 0.0, 0.6, 1.2, 2.0, 4.0, 6.0)
CROP_SIZE = 40
GIF_FPS = 4
RNG_SEED = 42
CHUNK_ATTEMPTS = 2_000_000

BLACK = 1
WHITE = 0
DIRS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)


def random_initial_lattice(n: int, n_black: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.full(n * n, WHITE, dtype=np.int8)
    flat[:n_black] = BLACK
    rng.shuffle(flat)
    return flat.reshape(n, n)


def neighbor_black_counts(lattice: np.ndarray) -> np.ndarray:
    """For each site, number of black neighbors (4-neighbor, periodic)."""
    return (
        np.roll(lattice, 1, axis=0)
        + np.roll(lattice, -1, axis=0)
        + np.roll(lattice, 1, axis=1)
        + np.roll(lattice, -1, axis=1)
    ).astype(np.int16)


def q_black_on_black(lattice: np.ndarray) -> float:
    """q_B/B: fraction of black among neighbors of a random black cell."""
    nb = neighbor_black_counts(lattice)
    mask = lattice == BLACK
    n_black = int(mask.sum())
    if n_black == 0:
        return 0.0
    return float(nb[mask].sum()) / (4.0 * n_black)


def isolated_black_count(lattice: np.ndarray) -> int:
    """IBC: black cells with no black neighbors."""
    nb = neighbor_black_counts(lattice)
    return int(((lattice == BLACK) & (nb == 0)).sum())


def _count_black_excluding_partner(
    lattice: np.ndarray, bi: int, bj: int, wi: int, wj: int, n: int
) -> int:
    nb = 0
    for di, dj in DIRS:
        ni = (bi + di) % n
        nj = (bj + dj) % n
        if ni == wi and nj == wj:
            continue
        nb += int(lattice[ni, nj] == BLACK)
    return nb


def run_kawasaki(
    lattice: np.ndarray,
    adhesion: float,
    motility: float,
    n_attempts: int,
    rng: np.random.Generator,
) -> None:
    """In-place Kawasaki updates (hot loop, no per-attempt Python calls)."""
    n = lattice.shape[0]
    remaining = n_attempts
    while remaining > 0:
        batch = min(CHUNK_ATTEMPTS, remaining)
        ri = rng.integers(0, n, batch)
        rj = rng.integers(0, n, batch)
        rd = rng.integers(0, 4, batch)
        ru = rng.random(batch)
        for k in range(batch):
            i = int(ri[k])
            j = int(rj[k])
            di, dj = DIRS[int(rd[k])]
            ni = (i + di) % n
            nj = (j + dj) % n
            if lattice[i, j] == lattice[ni, nj]:
                continue
            if lattice[i, j] == BLACK:
                bi, bj, wi, wj = i, j, ni, nj
            else:
                bi, bj, wi, wj = ni, nj, i, j
            nb = _count_black_excluding_partner(lattice, bi, bj, wi, wj, n)
            nw = 3 - nb
            delta_e = adhesion * (nw - nb)
            prob = 2.0 * motility / (1.0 + np.exp(-delta_e / motility))
            if ru[k] < prob:
                lattice[bi, bj] = WHITE
                lattice[wi, wj] = BLACK
        remaining -= batch


def simulate(
    adhesion: float,
    motility: float,
    *,
    n: int,
    n_black: int,
    time_units: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lattice = random_initial_lattice(n, n_black, rng)
    run_kawasaki(lattice, adhesion, motility, time_units * n * n, rng)
    return lattice


def snapshot_times(time_units: int) -> tuple[int, ...]:
    """Log-spaced snapshot times up to ``time_units`` (always includes 0 and final)."""
    if time_units <= 0:
        return (0,)
    candidates = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 7500, 10_000]
    times = sorted({t for t in candidates if t <= time_units})
    if times[-1] != time_units:
        times.append(time_units)
    return tuple(times)


def simulate_timeseries(
    adhesion: float,
    motility: float,
    *,
    n: int,
    n_black: int,
    time_units: int,
    times: tuple[int, ...],
    crop: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int], list[float], list[float]]:
    """Run Kawasaki dynamics and record cropped snapshots at selected time units."""
    lattice = random_initial_lattice(n, n_black, rng)
    snap = set(times)
    frames: list[np.ndarray] = []
    labels: list[int] = []
    q_vals: list[float] = []
    ibc_vals: list[float] = []
    attempts_per_unit = n * n

    for t in range(time_units + 1):
        if t in snap:
            frames.append(central_crop(lattice, crop))
            labels.append(t)
            q_vals.append(q_black_on_black(lattice))
            ibc_vals.append(float(isolated_black_count(lattice)))
        if t < time_units:
            run_kawasaki(lattice, adhesion, motility, attempts_per_unit, rng)

    return frames, labels, q_vals, ibc_vals


def central_crop(lattice: np.ndarray, crop: int) -> np.ndarray:
    n = lattice.shape[0]
    i0 = (n - crop) // 2
    j0 = (n - crop) // 2
    return lattice[i0 : i0 + crop, j0 : j0 + crop].copy()


def am_gif_filename(a_over_m: float) -> str:
    """Filesystem-safe GIF name, e.g. Am_m2 (A/m=-2), Am0p6 (A/m=0.6)."""
    if a_over_m < 0:
        body = f"{abs(a_over_m):g}".replace(".", "p")
        return f"kawasaki_ising_cell_sorting_Am_m{body}.gif"
    body = f"{a_over_m:g}".replace(".", "p")
    return f"kawasaki_ising_cell_sorting_Am{body}.gif"


def _gif_am_values() -> tuple[float, ...]:
    raw = os.environ.get("KAWASAKI_GIF_AM", "").strip()
    if not raw:
        return A_OVER_M_VALUES
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def run_fig3(
    *,
    n: int = LATTICE_SIZE,
    fraction_black: float = FRACTION_BLACK,
    motility: float = MOTILITY,
    time_units: int = TIME_UNITS,
    a_over_m_values: tuple[float, ...] = A_OVER_M_VALUES,
    crop: int = CROP_SIZE,
    seed: int = RNG_SEED,
    out_dir: Path | None = None,
    gif_fps: int = GIF_FPS,
    gif_am_values: tuple[float, ...] | None = None,
) -> tuple[list[np.ndarray], list[float], list[float], list[float], list[Path]]:
    n_black = int(round(fraction_black * n * n))
    times = snapshot_times(time_units)

    def _write_gif(am: float) -> bool:
        if gif_am_values is None:
            return True
        return any(abs(am - g) < 1e-9 for g in gif_am_values)
    panels: list[np.ndarray] = []
    am_ratios: list[float] = []
    q_vals: list[float] = []
    ibc_vals: list[float] = []
    gif_paths: list[Path] = []

    for i, am in enumerate(a_over_m_values):
        adhesion = am * motility
        run_rng = np.random.default_rng(seed + i * 10_007)
        frames, labels, series_q, series_ibc = simulate_timeseries(
            adhesion,
            motility,
            n=n,
            n_black=n_black,
            time_units=time_units,
            times=times,
            crop=crop,
            rng=run_rng,
        )
        panels.append(frames[-1])
        am_ratios.append(am)
        q_vals.append(series_q[-1])
        ibc_vals.append(series_ibc[-1])
        print(
            f"  A/m={am:4.1f}  q_B/B={q_vals[-1]:.3f}  IBC={ibc_vals[-1]:.0f}",
            flush=True,
        )
        if out_dir is not None and _write_gif(am):
            out_gif = out_dir / am_gif_filename(am)
            save_timeseries_gif(
                frames,
                labels,
                series_q,
                a_over_m=am,
                motility=motility,
                out_gif=out_gif,
                fps=gif_fps,
            )
            gif_paths.append(out_gif)
            print(f"    → {out_gif.name}  ({len(frames)} frames)", flush=True)

    return panels, am_ratios, q_vals, ibc_vals, gif_paths


def render_fig3(
    panels: list[np.ndarray],
    am_ratios: list[float],
    q_vals: list[float],
    out_png: Path,
) -> None:
    n_panels = len(panels)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    letters = "abcdefgh"

    for k, (ax, panel, am, q) in enumerate(zip(axes_flat, panels, am_ratios, q_vals)):
        ap.atlas_imshow(ax, panel, heatmap="binary", interpolation="nearest", vmin=0, vmax=1)
        letter = letters[k] if k < len(letters) else str(k)
        ax.set_title(f"({letter})  A/m = {am:g}\n$q_{{B/B}}$ = {q:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Mochizuki et al. (1996) Fig. 3 — Kawasaki–Ising cell sorting\n"
        f"$N={LATTICE_SIZE}^2$, $r_B={FRACTION_BLACK}$, $m={MOTILITY}$, "
        f"{TIME_UNITS} time units, random IC (crop {CROP_SIZE}×{CROP_SIZE})",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_timeseries_gif(
    frames: list[np.ndarray],
    labels: list[int],
    q_vals: list[float],
    *,
    a_over_m: float,
    motility: float,
    out_gif: Path,
    fps: int = GIF_FPS,
) -> None:
    """Save GIF via Pillow (avoids FuncAnimation single-frame bug)."""
    if not frames:
        return
    duration_ms = max(int(1000 / max(fps, 1)), 50)
    pil_frames: list[Image.Image] = []
    for frame, label, q in zip(frames, labels, q_vals):
        fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
        ap.atlas_imshow(ax, frame, heatmap="binary", interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"$t={label}$,  $A/m={a_over_m:g}$,  $q_{{B/B}}={q:.3f}$",
            fontsize=10,
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))
        plt.close(fig)

    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _quick_settings() -> dict:
    if os.environ.get("KAWASAKI_FIG3_QUICK", "").strip() in ("1", "true", "yes"):
        return dict(n=40, time_units=500, crop=24)
    return dict(
        n=int(os.environ.get("KAWASAKI_N", str(LATTICE_SIZE))),
        time_units=int(os.environ.get("KAWASAKI_TIME", str(TIME_UNITS))),
        crop=int(os.environ.get("KAWASAKI_CROP", str(CROP_SIZE))),
    )


def main() -> None:
    cfg = _quick_settings()
    seed = int(os.environ.get("KAWASAKI_SEED", str(RNG_SEED)))
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "results/kawasaki_ising_cell_sorting_fig3.png"
    gif_fps = int(os.environ.get("KAWASAKI_GIF_FPS", str(GIF_FPS)))
    gif_am_values = _gif_am_values()

    print(
        f"Kawasaki–Ising Fig. 3  (N={cfg['n']}, T={cfg['time_units']}, m={MOTILITY})",
        flush=True,
    )
    panels, am_ratios, q_vals, ibc_vals, gif_paths = run_fig3(
        seed=seed,
        out_dir=out_dir,
        gif_fps=gif_fps,
        gif_am_values=gif_am_values,
        **cfg,
    )
    render_fig3(panels, am_ratios, q_vals, out_png)
    print(f"Saved {out_png.name}", flush=True)
    if gif_paths:
        print(f"Saved {len(gif_paths)} GIF(s): {', '.join(p.name for p in gif_paths)}", flush=True)


if __name__ == "__main__":
    main()
