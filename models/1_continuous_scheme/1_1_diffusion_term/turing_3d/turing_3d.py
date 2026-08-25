#!/usr/bin/env python3
"""
3D Turing patterns — Shoji, Yamada, Ueyama & Ohta (2007), Phys. Rev. E 75, 046212.

Models (periodic box, semi-implicit diffusion via 3D FFT — same IMEX pattern as turing_2d):
  • FitzHugh–Nagumo (Sec. II):  u_t = D_u ∇²u + u − u³ − v,
                                v_t = D_v ∇²v + γ(u − αv − β)
  • Brusselator (Sec. III):     u_t = D_u ∇²u + A − (B+1)u + u²v,
                                v_t = D_v ∇²v + Bu − u²v
  • Gray–Scott (Sec. IV):       u_t = D_u ∇²u − uv² + F(1−u),
                                v_t = D_v ∇²v + uv² − (F+K)v

Paper: δt = 0.2 with an Eyre-type fully implicit FFT solver; here diffusion is treated
implicitly (7-point Laplacian kernel) and reaction explicitly, with δt chosen for stability.

Outputs:
  - turing_3d_fhn_gallery.png / turing_3d_fhn_analysis.png
  - turing_3d_brusselator_gallery.png / turing_3d_brusselator_analysis.png
  - turing_3d_gray_scott_gallery.png / turing_3d_gray_scott_analysis.png
  - turing_3d_*_surface_<slug>.gif / .obj — u isosurface (swift_hohenberg_3d と同型)

Use TURING_3D_QUICK=1 for CI-scale runs (smaller grid and fewer steps).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap


# ---------------------------------------------------------------------------
# Shared 3D semi-implicit FFT infrastructure
# ---------------------------------------------------------------------------


def build_implicit_kernel_3d(n: int, dt: float, d_coeff: float, dx: float) -> np.ndarray:
    """(I - dt D ∇²)^{-1} on a periodic 3D grid (standard 6-neighbor Laplacian)."""
    kernel = np.zeros((n, n, n), dtype=np.float64)
    c = dt * d_coeff / (dx * dx)
    kernel[0, 0, 0] = 1.0 + 6.0 * c
    kernel[1, 0, 0] = -c
    kernel[-1, 0, 0] = -c
    kernel[0, 1, 0] = -c
    kernel[0, -1, 0] = -c
    kernel[0, 0, 1] = -c
    kernel[0, 0, -1] = -c
    return 1.0 / np.fft.fftn(kernel)


def simulate_rd3d(
    u: np.ndarray,
    v: np.ndarray,
    *,
    dt: float,
    dx: float,
    du: float,
    dv: float,
    n_steps: int,
    react_u: Callable[[np.ndarray, np.ndarray], np.ndarray],
    react_v: Callable[[np.ndarray, np.ndarray], np.ndarray],
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    v_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IMEX Euler: explicit reaction, implicit diffusion (both components)."""
    n = u.shape[0]
    ku = build_implicit_kernel_3d(n, dt, du, dx)
    kv = build_implicit_kernel_3d(n, dt, dv, dx)
    if rng is None:
        rng = np.random.default_rng(0)

    for _ in range(n_steps):
        fu = u + dt * react_u(u, v)
        fv = v + dt * react_v(u, v)
        u = np.real(np.fft.ifftn(ku * np.fft.fftn(fu)))
        v = np.real(np.fft.ifftn(kv * np.fft.fftn(fv)))
        if v_floor is not None:
            v = np.maximum(v, v_floor)
        if noise_sigma > 0.0:
            u = u + noise_sigma * (rng.random(u.shape) - 0.5)
            v = v + noise_sigma * (rng.random(v.shape) - 0.5)
    return u, v


def max_real_growth_curve(
    jacobian: np.ndarray,
    du: float,
    dv: float,
    *,
    k_max: float = 80.0,
    n_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """σ_max(k) = max Re λ(J − k²D) for isotropic 3D modulation."""
    dmat = np.diag([du, dv]).astype(np.float64)
    k_grid = np.linspace(0.0, k_max, n_samples)
    sigma = np.empty_like(k_grid)
    for i, k in enumerate(k_grid):
        a = jacobian - (k * k) * dmat
        sigma[i] = float(np.max(np.real(np.linalg.eigvals(a))))
    i_star = int(np.argmax(sigma))
    k_star = float(k_grid[i_star])
    lam_star = float((2.0 * np.pi / k_star) if k_star > 1e-12 else np.inf)
    return k_grid, sigma, k_star, lam_star


def radial_spectrum_peak(field: np.ndarray, dx: float) -> tuple[float, float]:
    """Isotropic radial average of |FFT(field)|² → dominant k and λ."""
    arr = np.asarray(field, dtype=np.float64)
    centered = arr - float(np.mean(arr))
    fhat = np.fft.fftn(centered)
    power = np.abs(fhat) ** 2
    n = arr.shape[0]
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    kmax = float(kmag.max())
    n_bins = max(48, n // 2)
    bins = np.linspace(0.0, kmax, n_bins + 1)
    which = np.digitize(kmag.ravel(), bins) - 1
    which = np.clip(which, 0, n_bins - 1)
    radial = np.bincount(which, weights=power.ravel(), minlength=n_bins)
    count = np.bincount(which, minlength=n_bins)
    radial = radial / np.maximum(count, 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    if radial.size <= 1:
        return 0.0, np.inf
    radial[0] = 0.0
    idx = int(np.argmax(radial))
    k_peak = float(k_centers[idx])
    lam_peak = float((2.0 * np.pi / k_peak) if k_peak > 1e-12 else np.inf)
    return k_peak, lam_peak


def coord_grid(n: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, L, n, endpoint=False)
    return np.meshgrid(x, x, x, indexing="ij")


# ---------------------------------------------------------------------------
# FitzHugh–Nagumo (Shoji et al. Sec. II)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FHN3DParams:
    n: int = 32
    L: float = 0.17
    Du: float = 5.0e-5
    Dv: float = 5.0e-3
    alpha: float = 0.5
    gamma: float = 26.0
    beta: float = 0.04
    dt: float = 0.1
    n_steps: int = 12000
    noise_sigma: float = 0.0
    seed: int = 42


def fhn_homogeneous_ss(alpha: float, beta: float) -> tuple[float, float]:
    """Uniform steady state (ū, v̄) from u − u³ − v = 0 and u − αv − β = 0."""
    u_grid = np.linspace(-0.5, 1.5, 4000)
    best_u, best_res = 0.0, np.inf
    for u in u_grid:
        v = (u - beta) / alpha
        res = abs(u - u * u * u - v)
        if res < best_res:
            best_res, best_u = res, float(u)
    u_bar = best_u
    v_bar = u_bar - u_bar**3
    return u_bar, v_bar


def fhn_jacobian(u_bar: float, v_bar: float, alpha: float, gamma: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 3.0 * u_bar * u_bar, -1.0],
            [gamma, -gamma * alpha],
        ],
        dtype=np.float64,
    )


def fhn_turing_kc(u_bar: float, alpha: float, gamma: float, du: float, dv: float) -> float:
    """kc from Eq. (4): k_c² = (1 − 3ū²)/(2Du) − αγ/(2Dv)."""
    return float(np.sqrt(max((1.0 - 3.0 * u_bar * u_bar) / (2.0 * du) - alpha * gamma / (2.0 * dv), 0.0)))


def initial_fhn(params: FHN3DParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u_bar, v_bar = fhn_homogeneous_ss(params.alpha, params.beta)
    amp = 0.02
    u = u_bar + amp * rng.standard_normal((params.n, params.n, params.n))
    v = v_bar + amp * rng.standard_normal((params.n, params.n, params.n))
    return u, v


def simulate_fhn3d(params: FHN3DParams) -> tuple[np.ndarray, np.ndarray]:
    dx = params.L / params.n
    rng = np.random.default_rng(params.seed)
    u, v = initial_fhn(params, rng)

    def ru(u_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        return u_ - u_**3 - v_

    def rv(u_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        return params.gamma * (u_ - params.alpha * v_ - params.beta)

    return simulate_rd3d(
        u,
        v,
        dt=params.dt,
        dx=dx,
        du=params.Du,
        dv=params.Dv,
        n_steps=params.n_steps,
        react_u=ru,
        react_v=rv,
        noise_sigma=params.noise_sigma,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Brusselator (Sec. III — near-threshold parameters)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Brusselator3DParams:
    n: int = 32
    L: float = 0.124
    Du: float = 2.0e-4
    Dv: float = 1.6e-3
    A: float = 2.7
    B: float = 4.0
    dt: float = 0.1
    n_steps: int = 12000
    noise_sigma: float = 0.0
    seed: int = 44


def brusselator_homogeneous(A: float, B: float) -> tuple[float, float]:
    return float(A), float(B / A)


def brusselator_jacobian(A: float, B: float) -> np.ndarray:
    u0, v0 = A, B / A
    return np.array(
        [
            [-(B + 1.0) + 2.0 * u0 * v0, u0 * u0],
            [B - 2.0 * u0 * v0, -u0 * u0],
        ],
        dtype=np.float64,
    )


def brusselator_Bc(A: float, du: float, dv: float) -> float:
    return float((1.0 + A * np.sqrt(du / dv)) ** 2)


def brusselator_kc(A: float, du: float, dv: float) -> float:
    return float(np.sqrt(A / np.sqrt(du * dv)))


def initial_brusselator(params: Brusselator3DParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u0, v0 = brusselator_homogeneous(params.A, params.B)
    amp = 0.02
    u = u0 + amp * rng.standard_normal((params.n, params.n, params.n))
    v = v0 + amp * rng.standard_normal((params.n, params.n, params.n))
    v = np.maximum(v, 1e-8)
    return u, v


def simulate_brusselator3d(params: Brusselator3DParams) -> tuple[np.ndarray, np.ndarray]:
    dx = params.L / params.n
    rng = np.random.default_rng(params.seed)
    u, v = initial_brusselator(params, rng)
    a, b = params.A, params.B

    def ru(u_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        return a - (b + 1.0) * u_ + u_ * u_ * v_

    def rv(u_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        return b * u_ - u_ * u_ * v_

    return simulate_rd3d(
        u,
        v,
        dt=params.dt,
        dx=dx,
        du=params.Du,
        dv=params.Dv,
        n_steps=params.n_steps,
        react_u=ru,
        react_v=rv,
        noise_sigma=params.noise_sigma,
        rng=rng,
        v_floor=1e-8,
    )


# ---------------------------------------------------------------------------
# Gray–Scott (Sec. IV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrayScott3DParams:
    n: int = 32
    L: float = 1.10
    Du: float = 2.0e-4
    Dv: float = 1.0e-4
    F: float = 0.03
    K: float = 0.062
    dt: float = 0.1
    n_steps: int = 12000
    noise_sigma: float = 0.0
    seed: int = 55
    seed_box_cells: int = 10


def gray_scott_jacobian(F: float, K: float) -> np.ndarray:
    return np.array([[-F, 0.0], [0.0, -(F + K)]], dtype=np.float64)


def initial_gray_scott(params: GrayScott3DParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """u=1, v=0 with a cubic seed (10 cells) at the box centre (paper Sec. IV)."""
    n = params.n
    u = np.ones((n, n, n), dtype=np.float64)
    v = np.zeros((n, n, n), dtype=np.float64)
    half = max(1, params.seed_box_cells // 2)
    c = n // 2
    sl = slice(c - half, c + half)
    u[sl, sl, sl] = 0.5 + 0.01 * rng.standard_normal((2 * half, 2 * half, 2 * half))
    v[sl, sl, sl] = 0.25 + 0.01 * rng.standard_normal((2 * half, 2 * half, 2 * half))
    v = np.maximum(v, 0.0)
    return u, v


def simulate_gray_scott3d(params: GrayScott3DParams) -> tuple[np.ndarray, np.ndarray]:
    dx = params.L / params.n
    rng = np.random.default_rng(params.seed)
    u, v = initial_gray_scott(params, rng)

    def ru(u_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        return -u_ * v_ * v_ + params.F * (1.0 - u_)

    def rv(u_: np.ndarray, v_: np.ndarray) -> np.ndarray:
        return u_ * v_ * v_ - (params.F + params.K) * v_

    return simulate_rd3d(
        u,
        v,
        dt=params.dt,
        dx=dx,
        du=params.Du,
        dv=params.Dv,
        n_steps=params.n_steps,
        react_u=ru,
        react_v=rv,
        noise_sigma=params.noise_sigma,
        rng=rng,
        v_floor=0.0,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GalleryCase:
    label: str
    params: object
    surface_slug: str | None = None
    isosurface_level: float | None = None


def _mid_z_slice(field: np.ndarray) -> np.ndarray:
    return field[:, :, field.shape[2] // 2]


def simulate_gallery_case(case: GalleryCase) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one gallery case; return (u, v, L)."""
    p = case.params
    if isinstance(p, FHN3DParams):
        u, v = simulate_fhn3d(p)
        return u, v, p.L
    if isinstance(p, Brusselator3DParams):
        u, v = simulate_brusselator3d(p)
        return u, v, p.L
    if isinstance(p, GrayScott3DParams):
        u, v = simulate_gray_scott3d(p)
        return u, v, p.L
    raise TypeError(f"unknown params type: {type(p)}")


def default_isosurface_level(u: np.ndarray, case: GalleryCase) -> float:
    if case.isosurface_level is not None:
        return case.isosurface_level
    p = case.params
    if isinstance(p, FHN3DParams):
        return 0.05
    if isinstance(p, Brusselator3DParams):
        return 2.7
    if isinstance(p, GrayScott3DParams):
        return 0.5
    return float(np.median(u))


# ---------------------------------------------------------------------------
# Isosurface mesh / GIF / OBJ (swift_hohenberg_3d と同型)
# ---------------------------------------------------------------------------


def _downsample_field(u: np.ndarray, max_n: int = 48) -> tuple[np.ndarray, float]:
    stride = 1
    while u.shape[0] > max_n and stride < 8:
        stride += 1
    if stride > 1:
        return u[::stride, ::stride, ::stride], float(stride)
    return u, 1.0


def isosurface_mesh_periodic(
    u: np.ndarray,
    L: float,
    level: float = 0.0,
    *,
    sigma: float = 0.4,
    max_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        from skimage.measure import marching_cubes as mc
    except ImportError:
        mc = None

    mesh_n = 72 if mc is not None else 48
    if max_n is not None:
        mesh_n = max_n

    u_s = gaussian_filter(u, sigma=sigma, mode="wrap")
    if u_s.shape[0] > mesh_n:
        u_s, _stride = _downsample_field(u_s, max_n=mesh_n)

    if mc is not None:
        spacing = (L / u_s.shape[0],) * 3
        try:
            verts, faces, normals, _values = mc(
                u_s, level=level, spacing=spacing, method="lewiner", allow_degenerate=False
            )
        except (TypeError, ValueError):
            verts, faces, normals, _values = mc(u_s, level=level, spacing=spacing)
        return verts, faces.astype(int), normals, "marching_cubes (scikit-image)"

    pos = u_s > level
    if pos.mean() < 0.02 or pos.mean() > 0.98:
        raise ValueError(f"level set trivial (occupancy={pos.mean():.3f})")

    n = pos.shape[0]
    h = L / n
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []

    def add_quad(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        d: tuple[float, float, float],
    ) -> None:
        base = len(verts)
        verts.extend([a, b, c, d])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

    for i in range(n):
        ip = (i + 1) % n
        for j in range(n):
            jp = (j + 1) % n
            for k in range(n):
                kp = (k + 1) % n
                if not pos[i, j, k]:
                    continue
                xi, yi, zi = i * h, j * h, k * h
                xip, yjp, zkp = (i + 1) * h, (j + 1) * h, (k + 1) * h
                if not pos[(i - 1) % n, j, k]:
                    add_quad((xi, yi, zi), (xi, yi, zkp), (xi, yjp, zkp), (xi, yjp, zi))
                if not pos[ip, j, k]:
                    add_quad((xip, yi, zi), (xip, yjp, zi), (xip, yjp, zkp), (xip, yi, zkp))
                if not pos[i, (j - 1) % n, k]:
                    add_quad((xi, yi, zi), (xip, yi, zi), (xip, yi, zkp), (xi, yi, zkp))
                if not pos[i, jp, k]:
                    add_quad((xi, yjp, zi), (xi, yjp, zkp), (xip, yjp, zkp), (xip, yjp, zi))
                if not pos[i, j, (k - 1) % n]:
                    add_quad((xi, yi, zi), (xi, yjp, zi), (xip, yjp, zi), (xip, yi, zi))
                if not pos[i, j, kp]:
                    add_quad((xi, yi, zkp), (xip, yi, zkp), (xip, yjp, zkp), (xi, yjp, zkp))

    verts_arr = np.array(verts, dtype=np.float64)
    faces_arr = np.array(faces, dtype=int)
    v0 = verts_arr[faces_arr[:, 0]]
    v1 = verts_arr[faces_arr[:, 1]]
    v2 = verts_arr[faces_arr[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    return verts_arr, faces_arr, normals, "voxel boundary (fallback)"


def _face_colors_from_normals(normals: np.ndarray, u_sample: float) -> np.ndarray:
    light = np.array([0.35, 0.25, 1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = 0.32 + 0.68 * np.clip(normals @ light, 0.0, 1.0)
    base = np.array([0.82, 0.28, 0.22]) if u_sample >= 0 else np.array([0.22, 0.42, 0.88])
    return np.clip(base * shade[:, None], 0.0, 1.0)


def save_obj_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    out_obj: Path,
    *,
    comment: str = "",
) -> None:
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_obj.open("w", encoding="utf-8") as f:
        if comment:
            for line in comment.splitlines():
                f.write(f"# {line}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for fi, (a, b, c) in enumerate(faces, start=1):
            a1, b1, c1 = int(a) + 1, int(b) + 1, int(c) + 1
            f.write(f"f {a1}//{fi} {b1}//{fi} {c1}//{fi}\n")
    print(f"Saved {out_obj}  ({len(verts)} vertices, {len(faces)} triangles)")


def save_rotating_isosurface_gif(
    u: np.ndarray,
    L: float,
    title: str,
    out_gif: Path,
    *,
    level: float,
    n_frames: int = 90,
    fps: int = 12,
    elev: float = 22.0,
    sigma: float = 0.4,
    quick_mesh: bool = False,
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray, str] | None = None,
) -> None:
    if mesh is None:
        mesh_max_n = 56 if quick_mesh else 72
        verts, faces, normals, mesh_method = isosurface_mesh_periodic(
            u, L, level=level, sigma=sigma, max_n=mesh_max_n
        )
    else:
        verts, faces, normals, mesh_method = mesh

    colors = _face_colors_from_normals(normals, float(np.mean(u) - level))
    tri_verts = verts[faces]
    collection = Poly3DCollection(
        tri_verts,
        facecolors=colors,
        edgecolor=(0.12, 0.12, 0.18, 0.04),
        linewidths=0.08,
        alpha=0.94,
    )

    fig = plt.figure(figsize=(7.0, 6.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(collection)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8)
    pad = 0.08 * L
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, L + pad)
    ax.set_zlim(-pad, L + pad)

    def update(frame: int):
        ax.view_init(elev=elev, azim=frame * 360.0 / n_frames)
        return (collection,)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        blit=False,
        interval=max(1, 1000 // max(fps, 1)),
    )
    anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    print(f"Saved {out_gif}  ({n_frames} frames, {len(faces)} triangles, {mesh_method})")


def save_case_surfaces(
    case: GalleryCase,
    u: np.ndarray,
    L: float,
    out_dir: Path,
    *,
    model_prefix: str,
    quick: bool,
) -> None:
    slug = case.surface_slug or case.label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug)
    level = default_isosurface_level(u, case)
    title = f"{case.label}\n$u={level:g}$ isosurface"
    mesh_max_n = 56 if quick else 72
    mesh = isosurface_mesh_periodic(u, L, level=level, sigma=0.4, max_n=mesh_max_n)
    n_frames = 48 if quick else 96
    fps = 10 if quick else 12

    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    base = meshes_dir / f"turing_3d_{model_prefix}_surface_{slug}"
    if os.environ.get("TURING_3D_SKIP_OBJ", "").strip().lower() not in ("1", "true", "yes"):
        save_obj_mesh(
            mesh[0],
            mesh[1],
            mesh[2],
            base.with_suffix(".obj"),
            comment=f"{title}  ({mesh[3]})",
        )
    save_rotating_isosurface_gif(
        u,
        L,
        title,
        base.with_suffix(".gif"),
        level=level,
        n_frames=n_frames,
        fps=fps,
        quick_mesh=quick,
        mesh=mesh,
    )


def surface_cases_from_gallery(cases: list[GalleryCase]) -> list[GalleryCase]:
    """Cases that have an explicit surface_slug (representative morphologies)."""
    return [c for c in cases if c.surface_slug is not None]


def save_fhn_gallery(
    cases: list[GalleryCase], out_png: Path
) -> dict[str, tuple[np.ndarray, float]]:
    ncols = min(3, len(cases))
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    results: list[tuple[GalleryCase, np.ndarray, float, float]] = []
    surface_cache: dict[str, tuple[np.ndarray, float]] = {}
    vmax = 0.0
    for case in cases:
        p = case.params
        assert isinstance(p, FHN3DParams)
        print(f"FHN 3D: {case.label}  n={p.n}, L={p.L:g}, β={p.beta:g}, steps={p.n_steps}")
        u, _v, L = simulate_gallery_case(case)
        if case.surface_slug:
            surface_cache[case.surface_slug] = (u, L)
        vmax = max(vmax, float(np.max(np.abs(u))))
        k_spec, lam_spec = radial_spectrum_peak(u, p.L / p.n)
        results.append((case, u, k_spec, lam_spec))

    for ax, (case, u, k_spec, lam_spec) in zip(axes_flat, results):
        p = case.params
        assert isinstance(p, FHN3DParams)
        im = ap.atlas_imshow(
            ax,
            _mid_z_slice(u),
            heatmap="scalar",
            origin="lower",
            vmin=-vmax,
            vmax=vmax,
        )
        u_bar, _ = fhn_homogeneous_ss(p.alpha, p.beta)
        ax.set_title(
            f"{case.label}\nβ={p.beta:g}, L={p.L:g}\n"
            f"ū={u_bar:.3f}, $k_{{spec}}$={k_spec:.1f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(results) :]:
        ax.axis("off")

    fig.suptitle(
        "Shoji et al. (2007) — 3D FHN Turing patterns (mid-$z$ slice of $u$)\n"
        r"semi-implicit FFT; $D_u=5\times10^{-5}$, $D_v=5\times10^{-3}$, "
        r"$\alpha=0.5$, $\gamma=26$",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return surface_cache


def save_fhn_analysis(out_png: Path, *, beta: float = 0.04) -> None:
    alpha, gamma = 0.5, 26.0
    du, dv = 5.0e-5, 5.0e-3
    u_bar, v_bar = fhn_homogeneous_ss(alpha, beta)
    j = fhn_jacobian(u_bar, v_bar, alpha, gamma)
    k_grid, sigma, k_star, lam_star = max_real_growth_curve(j, du, dv, k_max=120.0)
    k_c = fhn_turing_kc(u_bar, alpha, gamma, du, dv)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(k_grid, sigma, "C0-", lw=1.5, label=r"$\sigma_{\max}(k)$")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(k_c, color="C2", ls="--", lw=1.0, label=rf"$k_c$ (linear) = {k_c:.2f}")
    ax.axvline(k_star, color="C1", ls=":", lw=1.0, label=rf"$k_*$ (max $\sigma$) = {k_star:.2f}")
    ax.set_xlabel("wave number $k$")
    ax.set_ylabel(r"$\max\mathrm{Re}\,\lambda$")
    ax.set_title(
        rf"FHN linear stability at $(\bar u,\bar v)=({u_bar:.4f},{v_bar:.4f})$, $\beta={beta:g}$\n"
        rf"predicted $\lambda_* \approx {lam_star:.3f}$",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def save_brusselator_gallery(
    cases: list[GalleryCase], out_png: Path
) -> dict[str, tuple[np.ndarray, float]]:
    ncols = min(3, len(cases))
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    results: list[tuple[GalleryCase, np.ndarray]] = []
    surface_cache: dict[str, tuple[np.ndarray, float]] = {}
    vmax = 0.0
    for case in cases:
        p = case.params
        assert isinstance(p, Brusselator3DParams)
        print(f"Brusselator 3D: {case.label}  B={p.B:g}, L={p.L:g}, steps={p.n_steps}")
        u, _v, L = simulate_gallery_case(case)
        if case.surface_slug:
            surface_cache[case.surface_slug] = (u, L)
        vmax = max(vmax, float(np.max(u)))
        results.append((case, u))

    for ax, (case, u) in zip(axes_flat, results):
        p = case.params
        assert isinstance(p, Brusselator3DParams)
        ap.atlas_imshow(ax, _mid_z_slice(u), heatmap="scalar", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(f"{case.label}\nB={p.B:g}, L={p.L:g}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(results) :]:
        ax.axis("off")

    fig.suptitle(
        "Shoji et al. (2007) — 3D Brusselator (mid-$z$ slice of $u$)\n"
        r"$D_u=2\times10^{-4}$, $D_v=1.6\times10^{-3}$, $A=2.7$ (near $B_c$)",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return surface_cache


def save_brusselator_analysis(out_png: Path) -> None:
    A, B = 2.7, 4.0
    du, dv = 2.0e-4, 1.6e-3
    j = brusselator_jacobian(A, B)
    k_grid, sigma, k_star, lam_star = max_real_growth_curve(j, du, dv, k_max=80.0)
    bc = brusselator_Bc(A, du, dv)
    kc = brusselator_kc(A, du, dv)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(k_grid, sigma, "C0-", lw=1.5)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(kc, color="C2", ls="--", lw=1.0, label=rf"$k_c$ = {kc:.2f}")
    ax.axvline(k_star, color="C1", ls=":", lw=1.0, label=rf"$k_*$ = {k_star:.2f}")
    ax.set_xlabel("wave number $k$")
    ax.set_ylabel(r"$\max\mathrm{Re}\,\lambda$")
    ax.set_title(
        rf"Brusselator linear stability at $(A,B)=({A},{B})$, $B_c={bc:.3f}$\n"
        rf"$\lambda_* \approx {lam_star:.3f}$",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def save_gray_scott_gallery(
    cases: list[GalleryCase], out_png: Path
) -> dict[str, tuple[np.ndarray, float]]:
    ncols = min(3, len(cases))
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    results: list[tuple[GalleryCase, np.ndarray]] = []
    surface_cache: dict[str, tuple[np.ndarray, float]] = {}
    vmax = 0.0
    for case in cases:
        p = case.params
        assert isinstance(p, GrayScott3DParams)
        print(f"Gray–Scott 3D: {case.label}  F={p.F:g}, L={p.L:g}, steps={p.n_steps}")
        u, _v, L = simulate_gallery_case(case)
        if case.surface_slug:
            surface_cache[case.surface_slug] = (u, L)
        vmax = max(vmax, float(np.max(u)))
        results.append((case, u))

    for ax, (case, u) in zip(axes_flat, results):
        p = case.params
        assert isinstance(p, GrayScott3DParams)
        ap.atlas_imshow(ax, _mid_z_slice(u), heatmap="scalar", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(f"{case.label}\nF={p.F:g}, L={p.L:g}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(results) :]:
        ax.axis("off")

    fig.suptitle(
        "Shoji et al. (2007) — 3D Gray–Scott (mid-$z$ slice of $u$)\n"
        r"$D_u=2\times10^{-4}$, $D_v=10^{-4}$, $K=0.062$; cubic seed at centre",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return surface_cache


def save_gray_scott_analysis(out_png: Path, *, F: float = 0.03) -> None:
    K = 0.062
    du, dv = 2.0e-4, 1.0e-4
    j = gray_scott_jacobian(F, K)
    k_grid, sigma, k_star, lam_star = max_real_growth_curve(j, du, dv, k_max=60.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(k_grid, sigma, "C0-", lw=1.5)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(k_star, color="C1", ls=":", lw=1.0, label=rf"$k_*$ = {k_star:.2f}")
    ax.set_xlabel("wave number $k$")
    ax.set_ylabel(r"$\max\mathrm{Re}\,\lambda$")
    ax.set_title(
        rf"Gray–Scott linear stability at $(u,v)=(1,0)$, $F={F:g}$, $K={K:g}$\n"
        rf"$\lambda_* \approx {lam_star:.3f}$",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def resolve_fhn_cases(quick: bool, *, noise_sigma: float = 0.0) -> list[GalleryCase]:
    steps = 2500 if quick else 12000
    n = 24 if quick else 32
    ns = dict(noise_sigma=noise_sigma)
    if quick:
        return [
            GalleryCase(
                "lamella (β≈0.08)",
                FHN3DParams(n=n, L=0.18, beta=0.08, n_steps=steps, seed=11, **ns),
            ),
            GalleryCase(
                "double-gyroid (β=0.04)",
                FHN3DParams(n=n, L=0.17, beta=0.04, n_steps=steps, seed=22, **ns),
                surface_slug="double_gyroid",
                isosurface_level=0.05,
            ),
            GalleryCase(
                "perforated lamella (β=0.045)",
                FHN3DParams(n=n, L=0.20, beta=0.045, n_steps=steps, seed=33, **ns),
                surface_slug="perforated_lamella",
                isosurface_level=0.05,
            ),
        ]
    return [
        GalleryCase("lamella (β=0.08)", FHN3DParams(n=32, L=0.18, beta=0.08, n_steps=steps, seed=11, **ns)),
        GalleryCase(
            "double-gyroid (β=0.04)",
            FHN3DParams(n=32, L=0.17, beta=0.04, n_steps=steps, seed=22, **ns),
            surface_slug="double_gyroid",
            isosurface_level=0.05,
        ),
        GalleryCase(
            "perforated lamella (β=0.045)",
            FHN3DParams(n=32, L=0.20, beta=0.045, n_steps=steps, seed=33, **ns),
            surface_slug="perforated_lamella",
            isosurface_level=0.05,
        ),
        GalleryCase(
            "Fddd (β=0.04)",
            FHN3DParams(n=32, L=0.176, beta=0.04, n_steps=steps, seed=44, **ns),
            surface_slug="fddd",
            isosurface_level=0.05,
        ),
        GalleryCase(
            "bcc (β=0.09)",
            FHN3DParams(n=32, L=0.198, beta=0.09, n_steps=steps, seed=55, **ns),
            surface_slug="bcc",
            isosurface_level=0.05,
        ),
        GalleryCase(
            "fcc (β=0.09)",
            FHN3DParams(n=32, L=0.128, beta=0.09, n_steps=steps, seed=66, **ns),
            surface_slug="fcc",
            isosurface_level=0.05,
        ),
    ]


def resolve_brusselator_cases(quick: bool) -> list[GalleryCase]:
    steps = 2500 if quick else 12000
    n = 24 if quick else 32
    base = dict(n=n, n_steps=steps, A=2.7, Du=2.0e-4, Dv=1.6e-3)
    if quick:
        return [
            GalleryCase(
                "single-gyroid (B=4)",
                Brusselator3DParams(**base, B=4.0, L=0.124, seed=12),
                surface_slug="single_gyroid",
                isosurface_level=2.7,
            ),
            GalleryCase(
                "single-diamond (B=4)",
                Brusselator3DParams(**base, B=4.0, L=0.160, seed=13),
                surface_slug="single_diamond",
                isosurface_level=2.7,
            ),
        ]
    return [
        GalleryCase("lamella", Brusselator3DParams(**base, B=3.90, L=0.208, seed=10)),
        GalleryCase(
            "single-gyroid",
            Brusselator3DParams(**base, B=4.0, L=0.124, seed=12),
            surface_slug="single_gyroid",
            isosurface_level=2.7,
        ),
        GalleryCase(
            "single-diamond",
            Brusselator3DParams(**base, B=4.0, L=0.160, seed=13),
            surface_slug="single_diamond",
            isosurface_level=2.7,
        ),
        GalleryCase(
            "double-gyroid",
            Brusselator3DParams(**base, B=4.0, L=0.128, seed=14),
            surface_slug="double_gyroid",
            isosurface_level=2.7,
        ),
        GalleryCase(
            "Fddd",
            Brusselator3DParams(**base, B=4.0, L=0.124, seed=15),
            surface_slug="fddd",
            isosurface_level=2.7,
        ),
    ]


def resolve_gray_scott_cases(quick: bool) -> list[GalleryCase]:
    steps = 2500 if quick else 12000
    n = 24 if quick else 32
    base = dict(n=n, n_steps=steps, Du=2.0e-4, Dv=1.0e-4, K=0.062)
    if quick:
        return [
            GalleryCase(
                "double-gyroid (F=0.03)",
                GrayScott3DParams(**base, F=0.03, L=1.10, seed=20),
                surface_slug="double_gyroid",
                isosurface_level=0.5,
            ),
            GalleryCase(
                "Fddd (F=0.036)",
                GrayScott3DParams(**base, F=0.036, L=1.05, seed=21),
                surface_slug="fddd",
                isosurface_level=0.5,
            ),
        ]
    return [
        GalleryCase(
            "double-gyroid",
            GrayScott3DParams(**base, F=0.03, L=1.10, seed=20),
            surface_slug="double_gyroid",
            isosurface_level=0.5,
        ),
        GalleryCase(
            "Fddd",
            GrayScott3DParams(**base, F=0.036, L=1.05, seed=21),
            surface_slug="fddd",
            isosurface_level=0.5,
        ),
        GalleryCase(
            "perforated lamella",
            GrayScott3DParams(**base, F=0.038, L=1.08, seed=22),
            surface_slug="perforated_lamella",
            isosurface_level=0.5,
        ),
        GalleryCase(
            "hexagonal",
            GrayScott3DParams(**base, F=0.040, L=1.12, seed=23),
            surface_slug="hexagonal",
            isosurface_level=0.5,
        ),
    ]


def save_surface_showcase(
    cases: list[GalleryCase],
    out_dir: Path,
    *,
    model_prefix: str,
    quick: bool,
    cache: dict[str, tuple[np.ndarray, float]] | None = None,
) -> None:
    """Rotating isosurface GIF + OBJ for cases marked with surface_slug."""
    surface_cases = surface_cases_from_gallery(cases)
    if not surface_cases:
        return
    if cache is None:
        cache = {}
    for case in surface_cases:
        key = case.surface_slug or case.label
        if key not in cache:
            print(f"Surface mesh: {model_prefix} / {case.label}")
            u, _v, L = simulate_gallery_case(case)
            cache[key] = (u, L)
        u, L = cache[key]
        try:
            save_case_surfaces(case, u, L, out_dir, model_prefix=model_prefix, quick=quick)
        except ValueError as exc:
            print(f"  skip surface {key}: {exc}")


def main() -> None:
    quick = os.environ.get("TURING_3D_QUICK", "").strip().lower() in ("1", "true", "yes")
    out_dir = Path(__file__).resolve().parent
    noise = float(os.environ.get("TURING_3D_NOISE", "0"))
    skip_surface = os.environ.get("TURING_3D_SKIP_ISOSURFACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if quick:
        print("Turing 3D (Shoji et al. 2007): QUICK mode")
    else:
        print("Turing 3D (Shoji et al. 2007): full run")

    if noise > 0.0:
        print(f"  stochastic noise σ={noise:g} per step")

    fhn_cases = resolve_fhn_cases(quick, noise_sigma=noise)
    fhn_cache = save_fhn_gallery(fhn_cases, out_dir / "results/turing_3d_fhn_gallery.png")
    save_fhn_analysis(out_dir / "results/turing_3d_fhn_analysis.png", beta=0.04)

    br_cases = resolve_brusselator_cases(quick)
    br_cache = save_brusselator_gallery(br_cases, out_dir / "results/turing_3d_brusselator_gallery.png")
    save_brusselator_analysis(out_dir / "results/turing_3d_brusselator_analysis.png")

    gs_cases = resolve_gray_scott_cases(quick)
    gs_cache = save_gray_scott_gallery(gs_cases, out_dir / "results/turing_3d_gray_scott_gallery.png")
    save_gray_scott_analysis(out_dir / "results/turing_3d_gray_scott_analysis.png", F=0.03)

    if not skip_surface:
        save_surface_showcase(fhn_cases, out_dir, model_prefix="fhn", quick=quick, cache=fhn_cache)
        save_surface_showcase(
            br_cases, out_dir, model_prefix="brusselator", quick=quick, cache=br_cache
        )
        save_surface_showcase(
            gs_cases, out_dir, model_prefix="gray_scott", quick=quick, cache=gs_cache
        )


if __name__ == "__main__":
    main()
