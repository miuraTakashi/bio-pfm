"""
3D Swift–Hohenberg / Landau–Brazovskii 型（半陰式 FFT）

    ∂u/∂t = r u + q u² - (1 + ∇²)² u - u³

線形項 (1+∇²)² をスペクトル空間で陰的、非線形 q u² - u³ を陽的 IMEX Euler。
周期境界、3D FFT（fftn / ifftn）。

出力:
  - swift_hohenberg_3d_gallery.png — 複数 (r,q, 初期条件) の z 中央断面
  - swift_hohenberg_3d_lamella.gif — ラメラ例の z 断面アニメ（フル run）
  - swift_hohenberg_3d_zero_surface_*.gif — u=0 等値面の回転ムービー（3D 構造）
  - swift_hohenberg_3d_zero_surface_*.obj — 同上メッシュ（VS Code / Cursor 3D Viewer 用）
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class SH3DParams:
    n: int = 72
    L: float = 6.0 * np.pi
    r: float = 0.2
    q: float = 0.0
    dt: float = 0.45
    n_steps: int = 1400
    seed: int = 42


def build_k2_grid(n: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = L / n
    k = np.fft.fftfreq(n, d=dx / (2.0 * np.pi))
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    return KX, KY, KZ, K2


def coord_grid(n: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, L, n, endpoint=False)
    return np.meshgrid(x, x, x, indexing="ij")


def initial_noise(n: int, L: float, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal((n, n, n))


def initial_lamella(n: int, L: float, amp: float, axis: str = "z") -> np.ndarray:
    X, Y, Z = coord_grid(n, L)
    if axis == "x":
        return amp * np.cos(2.0 * np.pi * X / L)
    if axis == "y":
        return amp * np.cos(2.0 * np.pi * Y / L)
    return amp * np.cos(2.0 * np.pi * Z / L)


def initial_bcc(n: int, L: float, amp: float) -> np.ndarray:
    """|k|≈1 の単純立方モジュレーション（BCC / 斑点系の種）。"""
    X, Y, Z = coord_grid(n, L)
    k0 = 2.0 * np.pi / L
    return amp * (np.cos(k0 * X) + np.cos(k0 * Y) + np.cos(k0 * Z))


def initial_gyroid_seed(n: int, L: float, amp: float) -> np.ndarray:
    """gyroid 型 triply periodic の低モード近似."""
    X, Y, Z = coord_grid(n, L)
    k0 = 2.0 * np.pi / L
    return amp * (
        np.cos(k0 * X) * np.sin(k0 * Y)
        + np.cos(k0 * Y) * np.sin(k0 * Z)
        + np.cos(k0 * Z) * np.sin(k0 * X)
    )


def initial_hcp_seed(n: int, L: float, amp: float) -> np.ndarray:
    """basal 六角（120° 3 波）+ c 軸モジュレーション（hcp / 六角密積の種）。"""
    X, Y, Z = coord_grid(n, L)
    k0 = 2.0 * np.pi / L
    hex_xy = (
        np.cos(k0 * X)
        + np.cos(k0 * (-0.5 * X + (np.sqrt(3.0) / 2.0) * Y))
        + np.cos(k0 * (-0.5 * X - (np.sqrt(3.0) / 2.0) * Y))
    )
    u = hex_xy + 0.28 * np.cos(k0 * Z)
    u /= float(np.sqrt(np.mean(u**2)) + 1e-12)
    return amp * u


def initial_fcc_seed(n: int, L: float, amp: float) -> np.ndarray:
    """4 本の〈111〉族波数（FCC / 面心立方の種）。"""
    X, Y, Z = coord_grid(n, L)
    k0 = 2.0 * np.pi / L
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
        u += np.cos(k0 * (d[0] * X + d[1] * Y + d[2] * Z))
    u /= float(np.sqrt(np.mean(u**2)) + 1e-12)
    return amp * u


def initial_stripe_tubes(n: int, L: float, amp: float) -> np.ndarray:
    """xy 平面の縞（円筒配向）。"""
    X, Y, _Z = coord_grid(n, L)
    return amp * np.cos(2.0 * np.pi * X / L)


def simulate_sh3d(
    params: SH3DParams,
    u0: np.ndarray | None = None,
    *,
    record_every: int | None = None,
) -> dict[str, np.ndarray]:
    n, L = params.n, params.L
    if u0 is None:
        u = initial_noise(n, L, 0.04, params.seed)
    else:
        u = np.asarray(u0, dtype=float).copy()

    _KX, _KY, _KZ, K2 = build_k2_grid(n, L)
    L_op = params.r - (1.0 - K2) ** 2
    denom = 1.0 - params.dt * L_op

    if record_every is None:
        record_every = max(1, params.n_steps // 80)

    frames: list[np.ndarray] = []
    times: list[float] = []

    def record(t: float) -> None:
        frames.append(u.copy())
        times.append(t)

    record(0.0)
    for step in range(1, params.n_steps + 1):
        nl = params.q * (u**2) - (u**3)
        u_hat = np.fft.fftn(u)
        nl_hat = np.fft.fftn(nl)
        u_hat = (u_hat + params.dt * nl_hat) / denom
        u = np.real(np.fft.ifftn(u_hat))
        if step % record_every == 0:
            record(step * params.dt)

    return {
        "u": u,
        "u_hist": np.array(frames, dtype=np.float64),
        "times": np.array(times, dtype=np.float64),
        "K2": K2,
    }


def pattern_label(u: np.ndarray, K2: np.ndarray) -> str:
    """|k|≈1 シェル上のパワー集中度から簡易ラベル。"""
    uh = np.fft.fftn(u)
    power = np.abs(np.fft.fftshift(uh)) ** 2
    K2s = np.fft.fftshift(K2)
    ring = (K2s > 0.72) & (K2s < 1.28)
    if not np.any(ring):
        return "mixed"
    p = power[ring].astype(float)
    s = float(np.sum(p))
    if s <= 0.0:
        return "mixed"
    p /= s
    mx = float(np.max(p))
    ent = float(-np.sum(p * np.log(p + 1e-30)))
    if ent < 2.2 and mx > 0.32:
        return "lamella/stripe"
    if mx < 0.14:
        return "isotropic"
    return "bcc/spot"


@dataclass(frozen=True)
class GalleryCase:
    name: str
    r: float
    q: float
    ic: str
    axis: str = "z"


def resolve_cases(quick: bool) -> list[GalleryCase]:
    if quick:
        return [
            GalleryCase("lamella (z)", 0.22, 0.0, "lamella", "z"),
            GalleryCase("lamella (x)", 0.22, 0.0, "lamella", "x"),
            GalleryCase("BCC spots (q>0)", 0.2, 0.55, "bcc"),
            GalleryCase("noise → spots", 0.25, 0.4, "noise"),
            GalleryCase("gyroid seed", 0.18, 0.0, "gyroid"),
            GalleryCase("hcp seed", 0.2, 0.35, "hcp"),
        ]
    return [
        GalleryCase("lamella (z layers)", 0.22, 0.0, "lamella", "z"),
        GalleryCase("lamella (x layers)", 0.22, 0.0, "lamella", "x"),
        GalleryCase("stripe tubes", 0.2, -0.25, "stripe"),
        GalleryCase("BCC / cubic (q>0)", 0.2, 0.6, "bcc"),
        GalleryCase("noise isotropic", 0.28, 0.0, "noise"),
        GalleryCase("noise + q>0", 0.24, 0.5, "noise"),
        GalleryCase("gyroid-type seed", 0.17, 0.0, "gyroid"),
        GalleryCase("hcp-type seed", 0.19, 0.4, "hcp"),
        GalleryCase("BCC weak q", 0.18, 0.2, "bcc"),
    ]


def make_initial(case: GalleryCase, n: int, L: float, seed: int) -> np.ndarray:
    amp = 0.1
    if case.ic == "lamella":
        return initial_lamella(n, L, amp, axis=case.axis)
    if case.ic == "bcc":
        return initial_bcc(n, L, amp)
    if case.ic == "gyroid":
        return initial_gyroid_seed(n, L, amp)
    if case.ic == "hcp":
        return initial_hcp_seed(n, L, amp)
    if case.ic == "fcc":
        return initial_fcc_seed(n, L, amp)
    if case.ic == "stripe":
        return initial_stripe_tubes(n, L, amp)
    return initial_noise(n, L, 0.05, seed)


def save_gallery(
    cases: list[GalleryCase],
    params_base: SH3DParams,
    out_png: Path,
) -> None:
    ncols = 3 if len(cases) <= 6 else 3
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    vmax = 0.0
    results: list[tuple[GalleryCase, np.ndarray, str]] = []
    for i, case in enumerate(cases):
        p = SH3DParams(
            n=params_base.n,
            L=params_base.L,
            r=case.r,
            q=case.q,
            dt=params_base.dt,
            n_steps=params_base.n_steps,
            seed=params_base.seed + i,
        )
        u0 = make_initial(case, p.n, p.L, p.seed)
        sim = simulate_sh3d(p, u0)
        u = sim["u"]
        label = pattern_label(u, sim["K2"])
        results.append((case, u, label))
        vmax = max(vmax, float(np.max(np.abs(u))))

    for ax, (case, u, label) in zip(axes_flat, results):
        mid = u.shape[2] // 2
        im = ax.imshow(
            u[:, :, mid],
            origin="lower",
            interpolation="bilinear",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"{case.name}\nr={case.r:g}, q={case.q:g}\n→ {label}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(results) :]:
        ax.axis("off")

    fig.suptitle(
        "Swift–Hohenberg 3D — semi-implicit FFT gallery (mid-$z$ slice)\n"
        rf"$L={params_base.L / np.pi:.1f}\pi$, $n={params_base.n}$, "
        rf"$\Delta t={params_base.dt:g}$, steps={params_base.n_steps}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def _downsample_field(u: np.ndarray, max_n: int = 48) -> tuple[np.ndarray, float]:
    """Return field and L-scale factor (h multiplier) after striding."""
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
    """
    u=level の等値面メッシュ。scikit-image の marching_cubes（Lewiner）を優先。
    """
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
    """Lambert shading + red/blue tint by field sign."""
    light = np.array([0.35, 0.25, 1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = 0.32 + 0.68 * np.clip(normals @ light, 0.0, 1.0)
    if u_sample >= 0:
        base = np.array([0.82, 0.28, 0.22])
    else:
        base = np.array([0.22, 0.42, 0.88])
    return np.clip(base * shade[:, None], 0.0, 1.0)


def save_obj_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    out_obj: Path,
    *,
    comment: str = "",
) -> None:
    """Wavefront OBJ (1-based indices, per-face normals for flat shading)."""
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


def save_rotating_zero_surface_gif(
    u: np.ndarray,
    L: float,
    title: str,
    out_gif: Path,
    *,
    level: float = 0.0,
    n_frames: int = 90,
    fps: int = 12,
    elev: float = 22.0,
    sigma: float = 0.4,
    quick_mesh: bool = False,
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if mesh is None:
        mesh_max_n = 56 if quick_mesh else 72
        verts, faces, normals, mesh_method = isosurface_mesh_periodic(
            u, L, level=level, sigma=sigma, max_n=mesh_max_n
        )
    else:
        verts, faces, normals, mesh_method = mesh
    u_mean = float(np.mean(u))
    colors = _face_colors_from_normals(normals, u_mean)

    tri_verts = verts[faces]
    mesh = Poly3DCollection(
        tri_verts,
        facecolors=colors,
        edgecolor=(0.12, 0.12, 0.18, 0.04),
        linewidths=0.08,
        alpha=0.94,
    )

    fig = plt.figure(figsize=(7.0, 6.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(mesh)
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_zlim(0.0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8)
    pad = 0.08 * L
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, L + pad)
    ax.set_zlim(-pad, L + pad)

    def update(frame: int):
        azim = frame * 360.0 / n_frames
        ax.view_init(elev=elev, azim=azim)
        return (mesh,)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        blit=False,
        interval=max(1, 1000 // max(fps, 1)),
    )
    anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    print(
        f"Saved {out_gif}  ({n_frames} frames, {len(faces)} triangles, {mesh_method})"
    )
    return verts, faces, normals, mesh_method


def save_isosurface_showcase(out_dir: Path, quick: bool) -> None:
    """Representative morphologies: rotating u=0 isosurface movies."""
    n = 48 if quick else 60
    n_steps = 700 if quick else 1200
    n_frames = 48 if quick else 96
    base_dt = 0.45

    demos: list[tuple[str, SH3DParams, str, str]] = [
        (
            "lamella_z",
            SH3DParams(n=n, L=5.0 * np.pi, r=0.22, q=0.0, dt=base_dt, n_steps=n_steps, seed=11),
            "lamella",
            "z",
        ),
        (
            "bcc_spots",
            SH3DParams(n=n, L=5.0 * np.pi, r=0.2, q=0.58, dt=base_dt, n_steps=n_steps, seed=22),
            "bcc",
            "z",
        ),
        (
            "gyroid",
            SH3DParams(n=n, L=5.0 * np.pi, r=0.17, q=0.0, dt=base_dt, n_steps=n_steps, seed=33),
            "gyroid",
            "z",
        ),
        (
            "hcp",
            SH3DParams(n=n, L=5.0 * np.pi, r=0.19, q=0.42, dt=base_dt, n_steps=n_steps, seed=44),
            "hcp",
            "z",
        ),
        (
            "fcc",
            SH3DParams(n=n, L=5.0 * np.pi, r=0.2, q=0.48, dt=base_dt, n_steps=n_steps, seed=55),
            "fcc",
            "z",
        ),
    ]

    for slug, params, ic, axis in demos:
        case = GalleryCase(slug, params.r, params.q, ic, axis)
        u0 = make_initial(case, params.n, params.L, params.seed)
        print(f"Isosurface movie: {slug} (n={params.n}, steps={params.n_steps})")
        sim = simulate_sh3d(params, u0)
        u = sim["u"]
        title = f"SH 3D: {slug}  ($u=0$ surface)\n$r={params.r:g}$, $q={params.q:g}$"
        mesh_max_n = 56 if quick else 72
        mesh = isosurface_mesh_periodic(u, params.L, level=0.0, sigma=0.4, max_n=mesh_max_n)
        verts, faces, normals, mesh_method = mesh
        if os.environ.get("SH3D_SKIP_OBJ", "").strip().lower() not in ("1", "true", "yes"):
            save_obj_mesh(
                verts,
                faces,
                normals,
                out_dir / "meshes" / f"swift_hohenberg_3d_zero_surface_{slug}.obj",
                comment=f"{title.replace(chr(10), ' ')}  ({mesh_method})",
            )
        save_rotating_zero_surface_gif(
            u,
            params.L,
            title,
            out_dir / f"swift_hohenberg_3d_zero_surface_{slug}.gif",
            n_frames=n_frames,
            fps=10 if quick else 12,
            quick_mesh=quick,
            mesh=mesh,
        )


def save_lamella_gif(params: SH3DParams, out_gif: Path, fps: int = 10) -> None:
    u0 = initial_lamella(params.n, params.L, 0.12, axis="z")
    sim = simulate_sh3d(params, u0, record_every=max(1, params.n_steps // 60))
    u_hist = sim["u_hist"]
    times = sim["times"]
    mid = u_hist.shape[2] // 2
    vmax = float(np.max(np.abs(u_hist)))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ap.atlas_imshow(
        ax,
        u_hist[0, :, :, mid],
        heatmap="scalar",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_axis_off()
    title = ax.set_title(f"lamella (z), t={times[0]:.1f}", fontsize=9)

    def update(i: int):
        im.set_data(u_hist[i, :, :, mid])
        title.set_text(f"lamella (z), t={times[i]:.1f}")
        return (im, title)

    anim = FuncAnimation(fig, update, frames=len(times), blit=False, interval=1000 // max(fps, 1))
    anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"Saved {out_gif}  ({len(times)} frames)")


def main() -> None:
    quick = os.environ.get("SH3D_QUICK", "").strip().lower() in ("1", "true", "yes")
    out_dir = Path(__file__).resolve().parent
    (out_dir / "meshes").mkdir(exist_ok=True)
    if quick:
        base = SH3DParams(n=48, L=4.0 * np.pi, dt=0.5, n_steps=600)
    else:
        base = SH3DParams(n=72, L=6.0 * np.pi, dt=0.45, n_steps=1400)

    cases = resolve_cases(quick)
    print(
        f"Swift–Hohenberg 3D: n={base.n}, L={base.L:.2f}, dt={base.dt}, "
        f"steps={base.n_steps}, {len(cases)} gallery cases"
    )
    save_gallery(cases, base, out_dir / "results/swift_hohenberg_3d_gallery.png")

    if os.environ.get("SH3D_SKIP_ISOSURFACE", "").strip().lower() not in ("1", "true", "yes"):
        save_isosurface_showcase(out_dir, quick=quick)

    if not quick and os.environ.get("SH3D_SKIP_SLICE_GIF", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        gif_params = SH3DParams(n=64, L=5.0 * np.pi, r=0.22, q=0.0, dt=0.45, n_steps=1200, seed=7)
        save_lamella_gif(gif_params, out_dir / "swift_hohenberg_3d_lamella.gif")


if __name__ == "__main__":
    main()
