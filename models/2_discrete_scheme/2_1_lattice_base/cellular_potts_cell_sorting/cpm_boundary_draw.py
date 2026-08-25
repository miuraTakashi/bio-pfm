"""
Lattice-bond cell outlines for Cellular Potts visualisation.

Lines are drawn on **edges between sites** (x = j±0.5, y = i±0.5), not on site pixels.
Each bond is emitted once (right/down scan) so segments meet at corners without overlap.

Use together with imshow(..., extent=lattice_extent(h, w)) so vector geometry matches pixels.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

BondFilter = Literal["all", "cell_cell", "cell_medium"]


def lattice_extent(height: int, width: int) -> tuple[float, float, float, float]:
    """Matplotlib extent (left, right, bottom, top) for an (height, width) lattice field."""
    return (-0.5, float(width) - 0.5, -0.5, float(height) - 0.5)


def _bond_visible(a: int, b: int, bond_filter: BondFilter) -> bool:
    if a == b:
        return False
    if bond_filter == "cell_cell":
        return a > 0 and b > 0
    if bond_filter == "cell_medium":
        return (a > 0) != (b > 0)
    # "all": any interface touching at least one cell (includes aggregate outline)
    return a > 0 or b > 0


def collect_bond_segments(
    lattice: np.ndarray,
    *,
    bond_filter: BondFilter = "all",
) -> np.ndarray:
    """
    Segment endpoints in data coordinates (x, y).

    Returns shape (N, 2, 2): each row is ((x0, y0), (x1, y1)) along one lattice bond.
    """
    h, w = lattice.shape
    segments: list[list[tuple[float, float]]] = []

    for i in range(h):
        for j in range(w):
            a = int(lattice[i, j])
            if j + 1 < w:
                b = int(lattice[i, j + 1])
                if _bond_visible(a, b, bond_filter):
                    x = j + 0.5
                    segments.append([(x, i - 0.5), (x, i + 0.5)])
            if i + 1 < h:
                b = int(lattice[i + 1, j])
                if _bond_visible(a, b, bond_filter):
                    y = i + 0.5
                    segments.append([(j - 0.5, y), (j + 0.5, y)])

    if not segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.asarray(segments, dtype=np.float64)


def draw_bond_boundaries(
    ax: Axes,
    lattice: np.ndarray,
    *,
    bond_filter: BondFilter = "all",
    color: str = "#1e1e1e",
    linewidth: float = 0.55,
    alpha: float = 0.9,
    zorder: int = 5,
) -> LineCollection | None:
    """Overlay line segments on lattice bonds (call after imshow with matching extent)."""
    segments = collect_bond_segments(lattice, bond_filter=bond_filter)
    if segments.size == 0:
        return None
    lc = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        capstyle="butt",
        joinstyle="round",
        antialiaseds=True,
        zorder=zorder,
    )
    ax.add_collection(lc)
    return lc


def draw_lattice_snapshot(
    ax: Axes,
    rgb: np.ndarray,
    lattice: np.ndarray,
    *,
    show_boundaries: bool = True,
    bond_filter: BondFilter = "all",
    boundary_color: str = "#1e1e1e",
    boundary_linewidth: float = 0.55,
) -> None:
    """
    Show an RGB lattice field and optional bond-aligned cell outlines.

    ``rgb`` and ``lattice`` must share the same (height, width) shape.
    """
    if rgb.shape[0] != lattice.shape[0] or rgb.shape[1] != lattice.shape[1]:
        raise ValueError("rgb and lattice must have the same (height, width)")

    h, w = lattice.shape
    extent = lattice_extent(h, w)
    ax.imshow(
        rgb,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        aspect="equal",
        zorder=1,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    if show_boundaries:
        draw_bond_boundaries(
            ax,
            lattice,
            bond_filter=bond_filter,
            color=boundary_color,
            linewidth=boundary_linewidth,
        )
