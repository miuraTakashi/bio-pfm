#!/usr/bin/env python3
"""
Graner & Glazier (1992) PRL — Fig. 1 cell-sorting time series (pure Python CPM).

Extended Potts Hamiltonian (differential adhesion + area constraint):
  E = Σ_{<i,j>, σ_i≠σ_j} J(τ(σ_i), τ(σ_j))
      + λ Σ_{cells c} (A_c − A_{τ(c)})²  · θ(A_{τ(c)})

Types: medium (M), dark (d, high adhesivity), light (l, low adhesivity).
Paper values: J(d,d)=2, J(d,l)=11, J(l,l)=14, J(d,M)=J(l,M)=16, λ=1, T=10,
target area A≈40. One MCS = 16·(Lx·Ly) copy attempts (overridable).

Pipeline (Fig. 1):
  1. Rectangular homotypic aggregate → equilibrate 400 MCS (rounding).
  2. Random light/dark assignment → initial panel (a).
  3. Sort to 10000 MCS; snapshots at 0, 1, 100, 1000, 4000, 10000 MCS.
  4. Two T=0 annealing MCS before each displayed frame (paper).

Outputs:
  results/cellular_potts_cell_sorting_fig1.png
  results/cellular_potts_cell_sorting_fig1.gif
"""
from __future__ import annotations

import os
from collections import deque
from functools import lru_cache
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

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

from cpm_boundary_draw import draw_lattice_snapshot

# --- cell types τ (paper: M, dark, light) ---
TYPE_MEDIUM = 0
TYPE_DARK = 1
TYPE_LIGHT = 2

# Graner & Glazier (1992) contact energies J(τ, τ')
J_MATRIX = np.array(
    [
        [0.0, 16.0, 16.0],
        [16.0, 2.0, 11.0],
        [16.0, 11.0, 14.0],
    ],
    dtype=np.float64,
)

# Defaults (paper); CELL_POTTS_FIG1_QUICK shrinks grid / MCS for CI
NX = NY = 100
N_ROWS = 5
N_COLS = 8
TARGET_AREA = 40.0
LAMBDA_AREA = 1.0
TEMPERATURE = 10.0
EQUIL_MCS = 400
SORT_MCS = 10_000
FIG1_SNAPSHOT_MCS = (0, 1, 100, 1000, 4000, 10_000)
ANNEAL_MCS = 2
MCS_MULTIPLIER = 16  # paper: 16 × number of lattice sites per MCS
RNG_SEED = 42

_OFFSETS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFFSETS8 = _OFFSETS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

_DI4 = np.array([-1, 1, 0, 0], dtype=np.int32)
_DJ4 = np.array([0, 0, -1, 1], dtype=np.int32)
_DI8 = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
_DJ8 = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)


if _HAS_NUMBA:

    @njit(cache=True)
    def _cell_connected_without_site_nb(
        lattice: np.ndarray,
        cell_id: int,
        ex_i: int,
        ex_j: int,
        n_remaining: int,
    ) -> bool:
        nx, ny = lattice.shape
        if n_remaining <= 0:
            return False

        seed_i = np.empty(4, dtype=np.int32)
        seed_j = np.empty(4, dtype=np.int32)
        n_seeds = 0
        for k in range(4):
            ni = ex_i + _DI4[k]
            nj = ex_j + _DJ4[k]
            if 0 <= ni < nx and 0 <= nj < ny and lattice[ni, nj] == cell_id:
                seed_i[n_seeds] = ni
                seed_j[n_seeds] = nj
                n_seeds += 1
        if n_seeds == 0:
            return False

        visited = np.zeros((nx, ny), dtype=np.uint8)
        max_q = nx * ny
        qi = np.empty(max_q, dtype=np.int32)
        qj = np.empty(max_q, dtype=np.int32)
        head = 0
        tail = 0
        for s in range(n_seeds):
            si = seed_i[s]
            sj = seed_j[s]
            if visited[si, sj] == 0:
                visited[si, sj] = 1
                qi[tail] = si
                qj[tail] = sj
                tail += 1

        n_seen = 0
        while head < tail:
            ci = qi[head]
            cj = qj[head]
            head += 1
            n_seen += 1
            for k in range(4):
                ni = ci + _DI4[k]
                nj = cj + _DJ4[k]
                if ni == ex_i and nj == ex_j:
                    continue
                if 0 <= ni < nx and 0 <= nj < ny:
                    if lattice[ni, nj] == cell_id and visited[ni, nj] == 0:
                        visited[ni, nj] = 1
                        qi[tail] = ni
                        qj[tail] = nj
                        tail += 1
        return n_seen == n_remaining

    @njit(cache=True)
    def _boundary_delta_nb(
        lattice: np.ndarray,
        kind: np.ndarray,
        J: np.ndarray,
        i: int,
        j: int,
        old_id: int,
        new_id: int,
    ) -> float:
        if old_id == new_id:
            return 0.0
        nx, ny = lattice.shape
        d = 0.0
        for k in range(8):
            ni = i + _DI8[k]
            nj = j + _DJ8[k]
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            nid = int(lattice[ni, nj])
            if old_id != nid:
                d -= J[kind[old_id], kind[nid]]
            if new_id != nid:
                d += J[kind[new_id], kind[nid]]
        return d

    @njit(cache=True)
    def _run_mcs_nb(
        lattice: np.ndarray,
        kind: np.ndarray,
        n_mcs: int,
        temperature: float,
        target_area: float,
        lam_area: float,
        J: np.ndarray,
        mcs_multiplier: int,
        seed: int,
        enforce_connectivity: int,
    ) -> None:
        np.random.seed(seed)
        nx, ny = lattice.shape
        n_ids = int(lattice.max())
        areas = np.zeros(n_ids + 1, dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                areas[lattice[i, j]] += 1.0
        n_attempts = max(1, mcs_multiplier) * nx * ny
        cand_i = np.empty(4, dtype=np.int32)
        cand_j = np.empty(4, dtype=np.int32)

        for _ in range(n_mcs):
            for _ in range(n_attempts):
                i = np.random.randint(0, nx)
                j = np.random.randint(0, ny)
                n_valid = 0
                for k in range(4):
                    ni = i + _DI4[k]
                    nj = j + _DJ4[k]
                    if 0 <= ni < nx and 0 <= nj < ny:
                        cand_i[n_valid] = ni
                        cand_j[n_valid] = nj
                        n_valid += 1
                if n_valid == 0:
                    continue
                pick = np.random.randint(0, n_valid)
                ni = cand_i[pick]
                nj = cand_j[pick]
                old_id = int(lattice[i, j])
                new_id = int(lattice[ni, nj])
                if new_id == old_id:
                    continue
                if old_id > 0 and areas[old_id] <= 1.0:
                    continue
                if old_id > 0 and enforce_connectivity != 0:
                    rem = int(areas[old_id]) - 1
                    if rem > 0 and not _cell_connected_without_site_nb(
                        lattice, old_id, i, j, rem
                    ):
                        continue

                d_e = _boundary_delta_nb(lattice, kind, J, i, j, old_id, new_id)
                if old_id > 0 and target_area > 0.0:
                    a_old = areas[old_id]
                    d_e += lam_area * (
                        (a_old - 1.0 - target_area) ** 2 - (a_old - target_area) ** 2
                    )
                if new_id > 0 and target_area > 0.0:
                    a_new = areas[new_id]
                    d_e += lam_area * (
                        (a_new + 1.0 - target_area) ** 2 - (a_new - target_area) ** 2
                    )

                accept = False
                if d_e <= 0.0:
                    accept = True
                elif temperature > 0.0:
                    if np.random.random() < np.exp(-d_e / temperature):
                        accept = True

                if accept:
                    lattice[i, j] = new_id
                    if old_id > 0:
                        areas[old_id] -= 1.0
                    if new_id > 0:
                        areas[new_id] += 1.0


def _neighbors(
    i: int, j: int, nx: int, ny: int, offsets: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    table = _neighbor_coord_table(nx, ny, len(offsets) == 8)
    return list(table[i * ny + j])


@lru_cache(maxsize=32)
def _neighbor_index_table(nx: int, ny: int, use8: bool) -> tuple[tuple[int, ...], ...]:
    offsets = _OFFSETS8 if use8 else _OFFSETS4
    out: list[tuple[int, ...]] = []
    for i in range(nx):
        for j in range(ny):
            nbrs: list[int] = []
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    nbrs.append(ni * ny + nj)
            out.append(tuple(nbrs))
    return tuple(out)


@lru_cache(maxsize=32)
def _neighbor_coord_table(
    nx: int, ny: int, use8: bool
) -> tuple[tuple[tuple[int, int], ...], ...]:
    idx_table = _neighbor_index_table(nx, ny, use8)
    out: list[tuple[tuple[int, int], ...]] = []
    for nbrs in idx_table:
        out.append(tuple((idx // ny, idx % ny) for idx in nbrs))
    return tuple(out)


def cell_connected_without_site(
    lattice: np.ndarray,
    cell_id: int,
    ex_i: int,
    ex_j: int,
    nx: int,
    ny: int,
    n_remaining: int,
) -> bool:
    if n_remaining <= 0:
        return False
    nyi = ny
    ex_idx = ex_i * nyi + ex_j
    nbr4 = _neighbor_index_table(nx, ny, False)
    lat_flat = lattice.ravel()
    seeds = [n_idx for n_idx in nbr4[ex_idx] if int(lat_flat[n_idx]) == cell_id]
    if not seeds:
        return False
    seen = bytearray(nx * ny)
    q: deque[int] = deque()
    for s in seeds:
        if not seen[s]:
            seen[s] = 1
            q.append(s)
    n_seen = 0
    while q:
        c_idx = q.popleft()
        n_seen += 1
        for n_idx in nbr4[c_idx]:
            if n_idx == ex_idx:
                continue
            if int(lat_flat[n_idx]) != cell_id or seen[n_idx]:
                continue
            seen[n_idx] = 1
            q.append(n_idx)
    return n_seen == n_remaining


def build_rectangular_aggregate(
    nx: int,
    ny: int,
    n_rows: int,
    n_cols: int,
    cell_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Place n_rows×n_cols rectangular cells in a central square block.
  Returns lattice (cell id per site) and kind[cell_id] = τ.
  """
    n_cells = n_rows * n_cols
    lattice = np.zeros((nx, ny), dtype=np.int32)
    margin = 8
    x0, x1 = margin, nx - margin
    y0, y1 = margin, ny - margin
    xs = np.linspace(x0, x1, n_cols + 1, dtype=int)
    ys = np.linspace(y0, y1, n_rows + 1, dtype=int)

    cid = 1
    for r in range(n_rows):
        for c in range(n_cols):
            lattice[xs[c] : xs[c + 1], ys[r] : ys[r + 1]] = cid
            cid += 1

    kind = np.zeros(n_cells + 1, dtype=np.int32)
    kind[1:] = cell_type
    return lattice, kind


def assign_random_types(
    kind: np.ndarray, n_cells: int, rng: np.random.Generator
) -> None:
    for cid in range(1, n_cells + 1):
        kind[cid] = TYPE_DARK if rng.random() < 0.5 else TYPE_LIGHT


def boundary_delta(
    lattice: np.ndarray,
    kind: np.ndarray,
    J: np.ndarray,
    i: int,
    j: int,
    old_id: int,
    new_id: int,
    nx: int,
    ny: int,
) -> float:
    """Δ contact energy for copying site (i,j) from old_id to new_id (8-neighbor bonds)."""
    if old_id == new_id:
        return 0.0
    idx = i * ny + j
    nbr8 = _neighbor_index_table(nx, ny, True)
    lat_flat = lattice.ravel()
    d = 0.0
    for n_idx in nbr8[idx]:
        nid = int(lat_flat[n_idx])
        if old_id != nid:
            d -= float(J[kind[old_id], kind[nid]])
        if new_id != nid:
            d += float(J[kind[new_id], kind[nid]])
    return d


def volume_delta(
    areas: np.ndarray,
    kind: np.ndarray,
    old_id: int,
    new_id: int,
    target: float,
    lam: float,
) -> float:
    d = 0.0
    if old_id > 0 and target > 0.0:
        a = float(areas[old_id])
        d += lam * ((a - 1.0 - target) ** 2 - (a - target) ** 2)
    if new_id > 0 and target > 0.0:
        a = float(areas[new_id])
        d += lam * ((a + 1.0 - target) ** 2 - (a - target) ** 2)
    return d


def mcs_step(
    lattice: np.ndarray,
    kind: np.ndarray,
    areas: np.ndarray,
    rng: np.random.Generator,
    *,
    nx: int,
    ny: int,
    n_attempts: int,
    temperature: float,
    target_area: float,
    lam_area: float,
    J: np.ndarray,
    enforce_connectivity: bool = True,
) -> None:
    nbr4 = _neighbor_index_table(nx, ny, False)
    lat_flat = lattice.ravel()
    for _ in range(n_attempts):
        i = int(rng.integers(0, nx))
        j = int(rng.integers(0, ny))
        idx = i * ny + j
        nbrs = nbr4[idx]
        if not nbrs:
            continue
        n_idx = int(nbrs[int(rng.integers(0, len(nbrs)))])
        old_id = int(lat_flat[idx])
        new_id = int(lat_flat[n_idx])
        if new_id == old_id:
            continue
        if old_id > 0 and areas[old_id] <= 1.0:
            continue
        if old_id > 0 and enforce_connectivity:
            rem = int(areas[old_id]) - 1
            if rem > 0 and not cell_connected_without_site(
                lattice, old_id, i, j, nx, ny, rem
            ):
                continue

        d_e = boundary_delta(lattice, kind, J, i, j, old_id, new_id, nx, ny)
        d_e += volume_delta(areas, kind, old_id, new_id, target_area, lam_area)

        accept = False
        if d_e <= 0.0:
            accept = True
        elif temperature > 0.0 and rng.random() < np.exp(-d_e / temperature):
            accept = True

        if accept:
            lat_flat[idx] = new_id
            if old_id > 0:
                areas[old_id] -= 1.0
            if new_id > 0:
                areas[new_id] += 1.0


def run_mcs(
    lattice: np.ndarray,
    kind: np.ndarray,
    n_mcs: int,
    rng: np.random.Generator,
    *,
    temperature: float,
    target_area: float = TARGET_AREA,
    lam_area: float = LAMBDA_AREA,
    J: np.ndarray = J_MATRIX,
    mcs_multiplier: int = MCS_MULTIPLIER,
    enforce_connectivity: bool = (
        os.environ.get("CELL_POTTS_ENFORCE_CONNECTIVITY", "0").strip().lower()
        in ("1", "true", "yes")
    ),
) -> None:
    use_numba = (
        _HAS_NUMBA
        and os.environ.get("CELL_POTTS_USE_NUMBA", "0").strip().lower()
        not in ("0", "false", "no")
    )
    if use_numba:
        seed = int(rng.integers(0, 2**31 - 1))
        _run_mcs_nb(
            lattice,
            kind,
            int(n_mcs),
            float(temperature),
            float(target_area),
            float(lam_area),
            J,
            int(mcs_multiplier),
            seed,
            int(bool(enforce_connectivity)),
        )
        return

    nx, ny = lattice.shape
    n_ids = int(lattice.max())
    areas = np.bincount(lattice.ravel(), minlength=n_ids + 1).astype(np.float64)
    n_attempts = max(1, mcs_multiplier) * nx * ny
    for _ in range(n_mcs):
        mcs_step(
            lattice,
            kind,
            areas,
            rng,
            nx=nx,
            ny=ny,
            n_attempts=n_attempts,
            temperature=temperature,
            target_area=target_area,
            lam_area=lam_area,
            J=J,
            enforce_connectivity=enforce_connectivity,
        )


def anneal_zero(lattice: np.ndarray, kind: np.ndarray, rng: np.random.Generator) -> None:
    run_mcs(lattice, kind, ANNEAL_MCS, rng, temperature=0.0)


def snapshot_rgb(lattice: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Light / dark / medium fill only (boundaries via cpm_boundary_draw)."""
    t = kind[np.clip(lattice, 0, kind.size - 1)]
    vis = np.zeros((*lattice.shape, 3), dtype=np.float32)
    vis[t == TYPE_MEDIUM] = (0.97, 0.97, 0.98)
    vis[t == TYPE_DARK] = (0.18, 0.28, 0.55)
    vis[t == TYPE_LIGHT] = (0.98, 0.90, 0.55)
    return vis


def draw_snapshot(
    ax: plt.Axes,
    lattice: np.ndarray,
    kind: np.ndarray,
    *,
    show_boundaries: bool = True,
) -> None:
    draw_lattice_snapshot(
        ax,
        snapshot_rgb(lattice, kind),
        lattice,
        show_boundaries=show_boundaries,
        bond_filter="all",
    )


def run_fig1_pipeline(
    rng: np.random.Generator,
    *,
    nx: int,
    ny: int,
    n_rows: int,
    n_cols: int,
    equil_mcs: int,
    sort_mcs: int,
    snapshot_mcs: tuple[int, ...],
    mcs_multiplier: int,
    apply_anneal: bool = True,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    n_cells = n_rows * n_cols
    lattice, kind = build_rectangular_aggregate(nx, ny, n_rows, n_cols, TYPE_DARK)
    run_mcs(
        lattice,
        kind,
        equil_mcs,
        rng,
        temperature=TEMPERATURE,
        mcs_multiplier=mcs_multiplier,
    )
    assign_random_types(kind, n_cells, rng)

    targets = sorted(set(snapshot_mcs))
    frames: list[np.ndarray] = []
    labels: list[int] = []
    mcs_done = 0

    for target in targets:
        if target > sort_mcs:
            break
        delta = target - mcs_done
        if delta > 0:
            run_mcs(
                lattice,
                kind,
                delta,
                rng,
                temperature=TEMPERATURE,
                mcs_multiplier=mcs_multiplier,
            )
            mcs_done = target
        snap = lattice.copy()
        if apply_anneal:
            anneal_zero(snap, kind, rng)
        frames.append(snap)
        labels.append(target)

    if labels[-1] != sort_mcs and sort_mcs not in labels:
        run_mcs(
            lattice,
            kind,
            sort_mcs - mcs_done,
            rng,
            temperature=TEMPERATURE,
            mcs_multiplier=mcs_multiplier,
        )
        snap = lattice.copy()
        if apply_anneal:
            anneal_zero(snap, kind, rng)
        frames.append(snap)
        labels.append(sort_mcs)

    return frames, labels, kind


def render_fig1_panels(
    frames: list[np.ndarray],
    labels: list[int],
    kind: np.ndarray,
    out_png: Path,
) -> None:
    n = len(frames)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.8))
    axes_flat = np.atleast_1d(axes).ravel()
    panel_letters = "abcdef"
    show_edges = os.environ.get("CELL_POTTS_NO_BOUNDARIES", "").strip() not in ("1", "true", "yes")
    for k, (ax, snap, mcs) in enumerate(zip(axes_flat, frames, labels)):
        draw_snapshot(ax, snap, kind, show_boundaries=show_edges)
        letter = panel_letters[k] if k < len(panel_letters) else str(k)
        ax.set_title(f"({letter})  MCS = {mcs}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(
        "Graner & Glazier (1992) Fig. 1 — differential-adhesion cell sorting",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gif(
    frames: list[np.ndarray],
    labels: list[int],
    kind: np.ndarray,
    out_gif: Path,
    fps: int = 4,
) -> None:
    show_edges = os.environ.get("CELL_POTTS_NO_BOUNDARIES", "").strip() not in ("1", "true", "yes")
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    draw_snapshot(ax, frames[0], kind, show_boundaries=show_edges)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"MCS = {labels[0]}", fontsize=10)

    def _update(k: int):
        ax.clear()
        draw_snapshot(ax, frames[k], kind, show_boundaries=show_edges)
        ax.set_xticks([])
        ax.set_yticks([])
        # Re-create title after clear(); this also keeps each GIF frame visually distinct.
        title_k = ax.set_title(f"MCS = {labels[k]}", fontsize=10)
        return (title_k,)

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // max(fps, 1), blit=False
    )
    ani.save(out_gif, writer=animation.PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def _quick_settings() -> dict:
    if os.environ.get("CELL_POTTS_FIG1_QUICK", "").strip() in ("1", "true", "yes"):
        return dict(
            nx=56,
            ny=56,
            n_rows=4,
            n_cols=5,
            equil_mcs=80,
            sort_mcs=400,
            snapshot_mcs=(0, 1, 40, 120, 250, 400),
            mcs_multiplier=4,
        )
    sort_mcs = int(os.environ.get("CELL_POTTS_MCS", str(SORT_MCS)))
    snap = os.environ.get("CELL_POTTS_FIG1_SNAPSHOTS", "")
    if snap.strip():
        snapshot_mcs = tuple(int(x) for x in snap.split(","))
    else:
        snapshot_mcs = tuple(s for s in FIG1_SNAPSHOT_MCS if s <= sort_mcs)
        if sort_mcs not in snapshot_mcs:
            snapshot_mcs = snapshot_mcs + (sort_mcs,)
    return dict(
        nx=int(os.environ.get("CELL_POTTS_NX", str(NX))),
        ny=int(os.environ.get("CELL_POTTS_NY", str(NY))),
        n_rows=int(os.environ.get("CELL_POTTS_ROWS", str(N_ROWS))),
        n_cols=int(os.environ.get("CELL_POTTS_COLS", str(N_COLS))),
        equil_mcs=int(os.environ.get("CELL_POTTS_EQUIL_MCS", str(EQUIL_MCS))),
        sort_mcs=sort_mcs,
        snapshot_mcs=snapshot_mcs,
        mcs_multiplier=int(os.environ.get("CELL_POTTS_MCS_MULT", str(MCS_MULTIPLIER))),
    )


def _gif_snapshot_schedule(sort_mcs: int) -> tuple[int, ...]:
    raw = os.environ.get("CELL_POTTS_GIF_SNAPSHOTS", "").strip()
    if raw:
        vals = sorted(set(int(x) for x in raw.split(",") if x.strip()))
        vals = [v for v in vals if 0 <= v <= sort_mcs]
        if 0 not in vals:
            vals.insert(0, 0)
        if sort_mcs not in vals:
            vals.append(sort_mcs)
        return tuple(vals)

    # Default: denser early-time sampling + coarse late-time sampling.
    base = (
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        400,
        700,
        1000,
        1500,
        2000,
        3000,
        4000,
        6000,
        8000,
        sort_mcs,
    )
    vals = [v for v in base if v <= sort_mcs]
    if vals[-1] != sort_mcs:
        vals.append(sort_mcs)
    return tuple(vals)


def main() -> None:
    rng = np.random.default_rng(int(os.environ.get("CELL_POTTS_SEED", str(RNG_SEED))))
    cfg = _quick_settings()
    panel_mcs = tuple(sorted(set(cfg["snapshot_mcs"])))
    gif_mcs = _gif_snapshot_schedule(int(cfg["sort_mcs"]))
    all_mcs = tuple(sorted(set(panel_mcs) | set(gif_mcs)))
    cfg_all = dict(cfg)
    cfg_all["snapshot_mcs"] = all_mcs
    # Keep simulation snapshots un-annealed for GIF motion visibility.
    frames_all, labels_all, kind = run_fig1_pipeline(rng, apply_anneal=False, **cfg_all)

    frame_by_mcs = {m: f for m, f in zip(labels_all, frames_all)}
    panel_frames = []
    for m in panel_mcs:
        if m not in frame_by_mcs:
            continue
        snap = frame_by_mcs[m].copy()
        # Paper Fig.1 display includes short T=0 anneal.
        anneal_zero(snap, kind, rng)
        panel_frames.append(snap)
    panel_labels = [m for m in panel_mcs if m in frame_by_mcs]
    gif_frames = [frame_by_mcs[m] for m in gif_mcs if m in frame_by_mcs]
    gif_labels = [m for m in gif_mcs if m in frame_by_mcs]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cellular_potts_cell_sorting_fig1.png"
    out_gif = out_dir / "cellular_potts_cell_sorting_fig1.gif"
    render_fig1_panels(panel_frames, panel_labels, kind, out_png)
    save_gif(
        gif_frames,
        gif_labels,
        kind,
        out_gif,
        fps=int(os.environ.get("CELL_POTTS_GIF_FPS", "4")),
    )
    print(
        f"Saved {out_png.name} and {out_gif.name}  "
        f"(sort MCS={cfg['sort_mcs']}, panel snapshots={panel_labels}, gif snapshots={gif_labels})"
    )


if __name__ == "__main__":
    main()
