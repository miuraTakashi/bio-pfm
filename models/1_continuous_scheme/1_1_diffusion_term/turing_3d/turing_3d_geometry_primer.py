#!/usr/bin/env python3
"""
3D Turing 形態の幾何プライマ — 周期関数 φ(x,y,z) と φ=0 等値面。

Shoji et al. (2007) や本フォルダの turing_3d.py で名付けられる
ラメラ / gyroid / BCC / FCC / diamond / Fddd などを、
反応拡散の数値解なしに **フーリエ型の典型式** だけで可視化する。

出力:
  - turing_3d_geometry_primer.png — 全形態の z 断面 + 等値面一覧
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class GeometryCase:
    slug: str
    title: str
    formula: str
    note: str
    builder: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


def coord_grid(n: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z


def _k0(L: float) -> float:
    return 2.0 * np.pi / L


def field_lamella(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """層状（ラメラ）: z に垂直な平行面 φ=0。"""
    return np.cos(_k0(L) * Z)


def field_perforated_lamella(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """穿孔ラメラ: 層 + xy モジュレーションで層内に穴。"""
    k = _k0(L)
    return np.cos(k * Z) + 0.38 * np.cos(k * X) * np.cos(k * Y)


def field_bcc(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """BCC / 立方斑点: 3 軸コサインの和（体心立方型モジュレーション）。"""
    k = _k0(L)
    return np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z)


def field_fcc(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """FCC: 4 本の〈111〉族平面波の和（面心立方型の対称性）。"""
    k = _k0(L)
    dirs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    dirs /= np.sqrt(3.0)
    u = np.zeros_like(X)
    for d in dirs:
        u += np.cos(k * (d[0] * X + d[1] * Y + d[2] * Z))
    return u / 2.0


def field_single_gyroid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """シングル gyroid (G 型 TPMS): sin cos の三重和。"""
    k = _k0(L)
    return (
        np.sin(k * X) * np.cos(k * Y)
        + np.sin(k * Y) * np.cos(k * Z)
        + np.sin(k * Z) * np.cos(k * X)
    )


def field_double_gyroid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """ダブル gyroid 型: cos sin 三重和（別相の二重ネットワークの低モード近似）。"""
    k = _k0(L)
    return (
        np.cos(k * X) * np.sin(k * Y)
        + np.cos(k * Y) * np.sin(k * Z)
        + np.cos(k * Z) * np.sin(k * X)
    )


def field_diamond(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """ダイヤモンド (Schwarz D 型 TPMS)。"""
    k = _k0(L)
    return np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z) + np.cos(k * X) * np.cos(k * Y) * np.cos(k * Z)


def field_fddd(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """Fddd 型: 直交だが非等方な 3 波数（直方晶・Fddd 対称のモチーフ）。"""
    k = _k0(L)
    return np.cos(k * X) + np.cos(1.31 * k * Y) + np.cos(0.87 * k * Z)


GEOMETRIES: list[GeometryCase] = [
    GeometryCase(
        "lamella",
        "Lamella",
        r"$\phi=\cos(2\pi z/L)$",
        "活性相が $z$ 方向の層に分かれる。",
        field_lamella,
    ),
    GeometryCase(
        "perforated_lamella",
        "Perforated lamella",
        r"$\phi=\cos kz + \alpha\cos kx\cos ky$",
        "層内に穴が開く穿孔層。",
        field_perforated_lamella,
    ),
    GeometryCase(
        "bcc",
        "BCC (cubic spots)",
        r"$\phi=\sum_i \cos(k x_i)$",
        "立方格子状の斑点／ドメイン。",
        field_bcc,
    ),
    GeometryCase(
        "fcc",
        "FCC",
        r"$\phi=\sum_{\langle111\rangle}\cos(\mathbf{k}\!\cdot\!\mathbf{x})$",
        "4 本の〈111〉波の和。",
        field_fcc,
    ),
    GeometryCase(
        "single_gyroid",
        "Single gyroid",
        r"$\phi=\sum_{\mathrm{cyc}} \sin kx_i\cos kx_{i+1}$",
        "三周期極小曲面（TPMS）gyroid に近い。",
        field_single_gyroid,
    ),
    GeometryCase(
        "double_gyroid",
        "Double gyroid",
        r"$\phi=\sum_{\mathrm{cyc}} \cos kx_i\sin kx_{i+1}$",
        "二重ネットワーク gyroid の低モード近似。",
        field_double_gyroid,
    ),
    GeometryCase(
        "diamond",
        "Diamond (Schwarz D)",
        r"$\phi=\prod\sin kx_i + \prod\cos kx_i$",
        "ダイヤモンド型 TPMS。",
        field_diamond,
    ),
    GeometryCase(
        "fddd",
        "Fddd",
        r"$\phi=\cos kx + \cos(1.31ky) + \cos(0.87kz)$",
        "直方晶（非等方）の 3 波モジュレーション。",
        field_fddd,
    ),
]


def isosurface_mesh(
    phi: np.ndarray,
    L: float,
    level: float = 0.0,
    *,
    sigma: float = 0.35,
    max_n: int = 56,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from skimage.measure import marching_cubes as mc
    except ImportError:
        mc = None

    u_s = gaussian_filter(phi, sigma=sigma, mode="wrap")
    stride = 1
    while u_s.shape[0] > max_n and stride < 8:
        stride += 1
        u_s = u_s[::stride, ::stride, ::stride]

    if mc is None:
        raise RuntimeError("scikit-image is required for isosurface meshes")

    spacing = (L / u_s.shape[0],) * 3
    try:
        verts, faces, normals, _ = mc(
            u_s, level=level, spacing=spacing, method="lewiner", allow_degenerate=False
        )
    except (TypeError, ValueError):
        verts, faces, normals, _ = mc(u_s, level=level, spacing=spacing)
    return verts, faces.astype(int), normals


def _face_colors(normals: np.ndarray) -> np.ndarray:
    light = np.array([0.35, 0.25, 1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = 0.32 + 0.68 * np.clip(normals @ light, 0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    rgba = cmap(np.clip(shade, 0.0, 1.0))
    return rgba[:, :3]


# 図中フォント（プライマ PNG 用）
FS_SUPTITLE = 14
FS_PANEL_TITLE = 9
FS_3D_TITLE = 10
FS_AXIS = 10
FS_FOOTER = 10
FS_CBAR = 10


def plot_panel_3d(ax, phi: np.ndarray, L: float, title: str) -> None:
    try:
        verts, faces, normals = isosurface_mesh(phi, L)
    except (RuntimeError, ValueError) as exc:
        ax.text(0.5, 0.5, 0.5, str(exc), ha="center", va="center", fontsize=FS_PANEL_TITLE)
        ax.set_title(title, fontsize=FS_3D_TITLE)
        return

    colors = _face_colors(normals)
    mesh = Poly3DCollection(
        verts[faces],
        facecolors=colors,
        edgecolor=(0.1, 0.1, 0.15, 0.05),
        linewidths=0.05,
        alpha=0.92,
    )
    ax.add_collection3d(mesh)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    ax.set_title(title, fontsize=FS_3D_TITLE, pad=2)
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.margins(0)


def _align_3d_panel_size(ax_sl: plt.Axes, ax_3d: plt.Axes) -> None:
    """3D 軸の描画枠を、直上の断面軸と同じ幅・高さにする（縦位置は下段のまま）。"""
    pos_sl = ax_sl.get_position()
    pos_3d = ax_3d.get_position()
    ax_3d.set_position([pos_sl.x0, pos_3d.y0, pos_sl.width, pos_sl.height])
    try:
        ax_3d.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def save_gallery(out_path: Path, *, n: int = 72, L: float = 2.0 * np.pi) -> None:
    X, Y, Z = coord_grid(n, L)
    n_cases = len(GEOMETRIES)
    ncols = 4
    nrows = int(np.ceil(n_cases / ncols))
    row_pairs = 2 * nrows

    fig = plt.figure(figsize=(15.0, 5.0 * nrows + 0.8), facecolor="white")
    fig.suptitle(
        r"3D pattern geometry primer: $\phi(\mathbf{x})=0$ isosurfaces "
        r"(Fourier / TPMS prototypes; $L=2\pi$)",
        fontsize=FS_SUPTITLE,
        y=0.995,
    )
    gs = gridspec.GridSpec(
        row_pairs,
        ncols,
        figure=fig,
        height_ratios=[1.0, 1.0] * nrows,
        hspace=0.38,
        wspace=0.28,
    )

    slice_axes: list[plt.Axes] = []
    surf_axes: list[plt.Axes] = []
    fields = [case.builder(X, Y, Z, L) for case in GEOMETRIES]
    vmax_global = float(max(np.max(np.abs(phi)) for phi in fields))

    for idx, case in enumerate(GEOMETRIES):
        phi = fields[idx]
        row, col = divmod(idx, ncols)

        ax_sl = fig.add_subplot(gs[2 * row, col])
        mid = n // 2
        im = ax_sl.imshow(
            phi[:, :, mid].T,
            origin="lower",
            cmap="viridis",
            vmin=-vmax_global,
            vmax=vmax_global,
            extent=(0, L, 0, L),
            aspect="equal",
        )
        ax_sl.set_title(f"{case.title}\n{case.formula}", fontsize=FS_PANEL_TITLE)
        ax_sl.set_xlabel("x", fontsize=FS_AXIS)
        ax_sl.set_ylabel("y", fontsize=FS_AXIS)
        ax_sl.tick_params(labelsize=FS_AXIS)
        slice_axes.append(ax_sl)

        ax_3d = fig.add_subplot(gs[2 * row + 1, col], projection="3d")
        plot_panel_3d(ax_3d, phi, L, rf"$\phi=0$: {case.slug}")
        surf_axes.append(ax_3d)

    fig.text(
        0.5,
        0.01,
        "Top: $\\phi$ at $z=L/2$; bottom: $\\phi=0$ isosurface. "
        "Prototypes only — actual RD steady states need not match exactly.",
        ha="center",
        fontsize=FS_FOOTER,
    )
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.94, hspace=0.42, wspace=0.22)
    cbar = fig.colorbar(im, ax=slice_axes, location="right", shrink=0.52, pad=0.02)
    cbar.ax.tick_params(labelsize=FS_CBAR)
    cbar.set_label(r"$\phi$", fontsize=FS_CBAR)
    # 3D は枠内で縮小表示されやすい → 断面と同サイズの bbox（y は下段を維持）
    fig.canvas.draw()
    for ax_sl, ax_3d in zip(slice_axes, surf_axes):
        _align_3d_panel_size(ax_sl, ax_3d)
        ax_sl.set_zorder(2)
        ax_3d.set_zorder(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    out = (Path(__file__).resolve().parent / "results" / "turing_3d_geometry_primer.png")
    save_gallery(out)


if __name__ == "__main__":
    main()
