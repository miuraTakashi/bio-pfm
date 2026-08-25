"""
2D 非圧縮流（MAC スタガード格子）を masks/maskS.raw 障害物付きで解く。

要求仕様に合わせた実装:
  - 格子: 128x128 の等間隔直交格子（256x256 の 1/4 セル数）
  - 物理: 粘性優位（既定で advection を無効化）、SI 単位
  - 流体: 水（20 °C 付近）、流路長 1 mm、圧力差 10 mmHg
  - 境界: 上下壁は無滑り（u=v=0）
  - 左右: 定圧境界（左 p_left, 右 p_right）
  - 障害物: masks/*.raw（ImageJ raw, uint8）の領域を固体として no-flow
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---- 物理定数（水 @ 20 °C 付近） ----
MMHG_TO_PA = 133.322387415  # 1 mmHg [Pa]
WATER_RHO = 998.0  # 密度 [kg/m^3]
WATER_MU = 1.002e-3  # 粘度 [Pa·s]
WATER_NU = WATER_MU / WATER_RHO  # 動粘性係数 [m^2/s] ≈ 1.004e-6

DEFAULT_LX = 1.0e-3  # 流路長 [m] = 1 mm
DEFAULT_LY = 1.0e-3  # 流路高さ [m] = 1 mm
DEFAULT_DP = 10.0 * MMHG_TO_PA  # 圧力差 [Pa]


@dataclass(frozen=True)
class FlowParams:
    """SI 単位でのシミュレーション条件。"""

    nx: int = 128
    ny: int = 128
    lx: float = DEFAULT_LX
    ly: float = DEFAULT_LY
    rho: float = WATER_RHO
    mu: float = WATER_MU
    p_left: float = DEFAULT_DP
    p_right: float = 0.0
    n_steps: int = 50000
    poisson_iter: int = 350
    print_every: int = 25
    mask_name: str = "masks/maskS.raw"

    @property
    def nu(self) -> float:
        return self.mu / self.rho

    @property
    def dp(self) -> float:
        return self.p_left - self.p_right


def load_flow_params() -> FlowParams:
    """環境変数からパラメータを読み込む（未指定時は水 / 1 mm / 10 mmHg）。"""
    dp = float(os.environ.get("FLOW_DP", str(DEFAULT_DP)))
    p_right = float(os.environ.get("FLOW_P_RIGHT", "0.0"))
    p_left = float(os.environ.get("FLOW_P_LEFT", str(p_right + dp)))
    return FlowParams(
        nx=int(os.environ.get("FLOW_NX", "128")),
        ny=int(os.environ.get("FLOW_NY", "128")),
        lx=float(os.environ.get("FLOW_LX", str(DEFAULT_LX))),
        ly=float(os.environ.get("FLOW_LY", str(DEFAULT_LY))),
        rho=float(os.environ.get("FLOW_RHO", str(WATER_RHO))),
        mu=float(os.environ.get("FLOW_MU", str(WATER_MU))),
        p_left=p_left,
        p_right=p_right,
        n_steps=int(os.environ.get("FLOW_N_STEPS", "50000")),
        poisson_iter=int(os.environ.get("FLOW_POISSON_ITER", "350")),
        print_every=int(os.environ.get("FLOW_PRINT_EVERY", "25")),
        mask_name=os.environ.get("FLOW_MASK_NAME", "masks/maskS.raw"),
    )


def _load_raw_mask_array(mask_path: Path, nx: int, ny: int) -> np.ndarray:
    """ImageJ raw (uint8) を読み込んで 2D 配列 (ny, nx) を返す。"""
    buf = np.fromfile(mask_path, dtype=np.uint8)
    n_expected = nx * ny
    if buf.size == n_expected:
        return buf.reshape((ny, nx))

    side = int(round(np.sqrt(buf.size)))
    if side * side != buf.size:
        raise ValueError(
            f"Unsupported raw size for {mask_path.name}: {buf.size} bytes "
            f"(expected nx*ny={n_expected} or square)."
        )
    arr0 = buf.reshape((side, side))
    img = Image.fromarray(arr0, mode="L")
    img = img.resize((nx, ny), resample=Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def load_solid_mask(mask_path: Path, nx: int, ny: int, solid_is_dark: bool = False) -> np.ndarray:
    """マスクから固体領域 True=solid を作る（raw/tif/png 対応）。"""
    if mask_path.suffix.lower() == ".raw":
        arr = _load_raw_mask_array(mask_path, nx, ny)
    else:
        img = Image.open(mask_path).convert("L")
        if img.size != (nx, ny):
            img = img.resize((nx, ny), resample=Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.uint8)
    if solid_is_dark:
        solid = arr < 128
    else:
        solid = arr > 127
    return solid.T  # x-first (nx, ny)


def build_face_masks(fluid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """速度面の有効領域（True=流体面）。"""
    nx, ny = fluid.shape
    u_active = np.zeros((nx + 1, ny), dtype=bool)  # x-face
    v_active = np.zeros((nx, ny + 1), dtype=bool)  # y-face

    # 内部面
    u_active[1:nx, :] = fluid[:-1, :] & fluid[1:, :]
    v_active[:, 1:ny] = fluid[:, :-1] & fluid[:, 1:]
    # 左右境界面（圧力境界）
    u_active[0, :] = fluid[0, :]
    u_active[nx, :] = fluid[nx - 1, :]
    # 上下壁は no-slip
    v_active[:, 0] = False
    v_active[:, ny] = False
    return u_active, v_active


def apply_velocity_bc(
    u: np.ndarray,
    v: np.ndarray,
    u_active: np.ndarray,
    v_active: np.ndarray,
) -> None:
    """壁 no-slip と障害物 no-flow を適用。"""
    # 上下壁
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    # 左右境界の接線速度も 0（境界流速 0 の要件）
    v[0, :] = 0.0
    v[-1, :] = 0.0

    u[~u_active] = 0.0
    v[~v_active] = 0.0


def laplacian_u(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
        + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    )
    return out


def laplacian_v(v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(v)
    out[1:-1, 1:-1] = (
        (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / (dx * dx)
        + (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) / (dy * dy)
    )
    return out


def divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy


def pressure_poisson(
    rhs: np.ndarray,
    fluid: np.ndarray,
    p_left: float,
    p_right: float,
    dx: float,
    dy: float,
    n_iter: int,
    p_init: np.ndarray | None = None,
    tol: float = 1e-6,
) -> np.ndarray:
    """固体マスク対応 Poisson: ∇²p = rhs, 左右 Dirichlet / 上下 Neumann."""
    p = np.zeros_like(rhs) if p_init is None else p_init.copy()
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    nx, ny = fluid.shape

    west_fluid = np.zeros_like(fluid, dtype=bool)
    east_fluid = np.zeros_like(fluid, dtype=bool)
    south_fluid = np.zeros_like(fluid, dtype=bool)
    north_fluid = np.zeros_like(fluid, dtype=bool)
    west_fluid[1:, :] = fluid[:-1, :]
    east_fluid[:-1, :] = fluid[1:, :]
    south_fluid[:, 1:] = fluid[:, :-1]
    north_fluid[:, :-1] = fluid[:, 1:]

    left_bc = np.zeros_like(fluid, dtype=bool)
    right_bc = np.zeros_like(fluid, dtype=bool)
    left_bc[0, :] = fluid[0, :]
    right_bc[-1, :] = fluid[-1, :]

    coef = np.zeros_like(p)
    coef += west_fluid * idx2
    coef += east_fluid * idx2
    coef += south_fluid * idy2
    coef += north_fluid * idy2
    coef += left_bc * idx2
    coef += right_bc * idx2
    valid = fluid & (coef > 1e-12)

    sum_const = np.zeros_like(p)
    sum_const += left_bc * p_left * idx2
    sum_const += right_bc * p_right * idx2

    for _ in range(n_iter):
        p_old = p.copy()
        summ = sum_const.copy()

        # 隣接セル寄与（都度の大配列生成を避ける）
        summ[1:nx, :] += west_fluid[1:nx, :] * p[0 : nx - 1, :] * idx2
        summ[0 : nx - 1, :] += east_fluid[0 : nx - 1, :] * p[1:nx, :] * idx2
        summ[:, 1:ny] += south_fluid[:, 1:ny] * p[:, 0 : ny - 1] * idy2
        summ[:, 0 : ny - 1] += north_fluid[:, 0 : ny - 1] * p[:, 1:ny] * idy2

        p[valid] = (summ[valid] - rhs[valid]) / coef[valid]
        p[~fluid] = 0.0

        if float(np.max(np.abs(p - p_old))) < tol:
            break

    return p


def project_velocity(
    u_star: np.ndarray,
    v_star: np.ndarray,
    p: np.ndarray,
    fluid: np.ndarray,
    u_active: np.ndarray,
    v_active: np.ndarray,
    p_left: float,
    p_right: float,
    rho: float,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """u* を発散ゼロに投影。"""
    nx, ny = fluid.shape
    u = u_star.copy()
    v = v_star.copy()

    # 内部 u-face
    dpdx_u = (p[1:, :] - p[:-1, :]) / dx
    u[1:nx, :] -= (dt / rho) * dpdx_u
    # 左右境界 u-face（境界圧との片側差分）
    u[0, :] -= (dt / rho) * (p[0, :] - p_left) / (0.5 * dx)
    u[nx, :] -= (dt / rho) * (p_right - p[nx - 1, :]) / (0.5 * dx)

    # 内部 v-face
    dpdy_v = (p[:, 1:] - p[:, :-1]) / dy
    v[:, 1:ny] -= (dt / rho) * dpdy_v

    apply_velocity_bc(u, v, u_active, v_active)
    return u, v


def cell_center_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc


def run_simulation(
    nx: int = 128,
    ny: int = 128,
    lx: float = DEFAULT_LX,
    ly: float = DEFAULT_LY,
    nu: float = WATER_NU,
    rho: float = WATER_RHO,
    p_left: float = DEFAULT_DP,
    p_right: float = 0.0,
    n_steps: int = 50000,
    poisson_iter: int = 350,
    print_every: int = 25,
    mask_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = lx / nx
    dy = ly / ny
    dt = 0.2 * min(dx, dy) ** 2 / max(nu, 1e-12)  # 粘性制約

    if mask_name is None:
        mask_name = os.environ.get("FLOW_MASK_NAME", "masks/maskS.raw")
    mask_path = Path(__file__).parent / mask_name
    solid_is_dark = os.environ.get("FLOW_MASK_SOLID_IS_DARK", "0").lower() in ("1", "true", "yes")
    solid = load_solid_mask(mask_path, nx, ny, solid_is_dark=solid_is_dark)
    fluid = ~solid
    u_active, v_active = build_face_masks(fluid)

    u = np.zeros((nx + 1, ny), dtype=float)
    v = np.zeros((nx, ny + 1), dtype=float)
    p = np.zeros((nx, ny), dtype=float)

    t0 = time.perf_counter()
    for step in range(n_steps):
        # 粘性優位: 既定では移流を入れず Stokes 極限に近づける
        u_star = u + dt * nu * laplacian_u(u, dx, dy)
        v_star = v + dt * nu * laplacian_v(v, dx, dy)
        apply_velocity_bc(u_star, v_star, u_active, v_active)

        rhs = (rho / dt) * divergence(u_star, v_star, dx, dy)
        rhs[~fluid] = 0.0
        p = pressure_poisson(
            rhs,
            fluid,
            p_left,
            p_right,
            dx,
            dy,
            poisson_iter,
            p_init=p,
            tol=1e-7,
        )
        u, v = project_velocity(
            u_star, v_star, p, fluid, u_active, v_active,
            p_left, p_right, rho, dt, dx, dy
        )

        if (step + 1) % print_every == 0 or (step + 1) == n_steps:
            div_inf = float(np.max(np.abs(divergence(u, v, dx, dy)[fluid])))
            done = step + 1
            elapsed = time.perf_counter() - t0
            frac = done / max(n_steps, 1)
            eta = elapsed * (1.0 - frac) / max(frac, 1e-12)
            print(
                f"[flow] {done:5d}/{n_steps} ({100.0*frac:5.1f}%) "
                f"dt={dt:.3e}  ||div||inf={div_inf:.3e}  "
                f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s",
                flush=True,
            )

    return u, v, p, fluid, solid


def save_figure(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    fluid: np.ndarray,
    solid: np.ndarray,
    out_path: Path,
    lx: float = DEFAULT_LX,
    ly: float = DEFAULT_LY,
    dp_mmhg: float = DEFAULT_DP / MMHG_TO_PA,
) -> None:
    uc, vc = cell_center_velocity(u, v)
    speed = np.sqrt(uc**2 + vc**2)
    speed = np.where(fluid, speed, np.nan)
    p_plot = np.where(fluid, p, np.nan)

    nx, ny = p.shape
    lx_mm = lx * 1.0e3
    ly_mm = ly * 1.0e3
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    ax00 = axes[0, 0]
    ax01 = axes[0, 1]
    ax10 = axes[1, 0]
    ax11 = axes[1, 1]

    im0 = ax00.imshow(
        speed.T,
        origin="lower",
        extent=[0.0, lx_mm, 0.0, ly_mm],
        interpolation="nearest",
    )
    ax00.set_title("Speed |u|")
    ax00.set_xlabel("x [mm]")
    ax00.set_ylabel("y [mm]")
    fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04, label="|u| [m/s]")

    im1 = ax01.imshow(
        p_plot.T,
        origin="lower",
        extent=[0.0, lx_mm, 0.0, ly_mm],
        interpolation="nearest",
    )
    ax01.set_title("Pressure p")
    ax01.set_xlabel("x [mm]")
    ax01.set_ylabel("y [mm]")
    fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04, label="p [Pa]")

    stride = 8
    finite_speed = speed[np.isfinite(speed)]
    vmax = float(np.max(finite_speed)) if finite_speed.size else 0.0
    # 速度が小さいケースでもベクトルを視認できるよう、描画時のみ自動スケーリング。
    vis_gain = 1.0 if vmax < 1e-14 else min(80.0, 0.08 / vmax)
    ax10.imshow(
        solid.T.astype(float),
        origin="lower",
        extent=[0.0, lx_mm, 0.0, ly_mm],
        interpolation="nearest",
        alpha=0.35,
    )
    qv = ax10.quiver(
        (X * 1.0e3)[::stride, ::stride],
        (Y * 1.0e3)[::stride, ::stride],
        (uc * vis_gain)[::stride, ::stride],
        (vc * vis_gain)[::stride, ::stride],
        speed[::stride, ::stride],
        cmap="viridis",
        pivot="mid",
        scale_units="xy",
        scale=1.0,
        width=0.0022,
    )
    ax10.set_title(f"Velocity vectors (display x{vis_gain:.1f}, max|u|={vmax:.3e} m/s)")
    ax10.set_xlabel("x [mm]")
    ax10.set_ylabel("y [mm]")
    fig.colorbar(qv, ax=ax10, fraction=0.046, pad=0.04, label="|u| [m/s]")

    # 流線場（障害物内はマスク）
    uc_stream = np.ma.masked_where(~fluid.T, uc.T)
    vc_stream = np.ma.masked_where(~fluid.T, vc.T)
    sp_stream = np.ma.masked_where(~fluid.T, speed.T)
    ax11.imshow(
        solid.T.astype(float),
        origin="lower",
        extent=[0.0, lx_mm, 0.0, ly_mm],
        interpolation="nearest",
        alpha=0.25,
    )
    sp = ax11.streamplot(
        x * 1.0e3,
        y * 1.0e3,
        uc_stream,
        vc_stream,
        density=1.35,
        color=sp_stream,
        cmap="viridis",
        linewidth=1.0,
        arrowsize=0.85,
    )
    ax11.set_title("Streamlines")
    ax11.set_xlabel("x [mm]")
    ax11.set_ylabel("y [mm]")
    fig.colorbar(sp.lines, ax=ax11, fraction=0.046, pad=0.04, label="|u| [m/s]")

    for ax in axes.ravel():
        ax.set_xlim(0.0, lx_mm)
        ax.set_ylim(0.0, ly_mm)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Water flow, Δp={dp_mmhg:g} mmHg, L={lx_mm:.2f} mm ({nx}x{ny} MAC grid)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    params = load_flow_params()
    dx = params.lx / params.nx
    dt = 0.2 * dx * dx / max(params.nu, 1e-12)

    print(
        f"[flow] water: rho={params.rho:g} kg/m^3, mu={params.mu:g} Pa·s, "
        f"nu={params.nu:.3e} m^2/s",
        flush=True,
    )
    print(
        f"[flow] domain={params.lx * 1e3:g}x{params.ly * 1e3:g} mm, "
        f"dp={params.dp / MMHG_TO_PA:g} mmHg ({params.dp:g} Pa), "
        f"grid={params.nx}x{params.ny}, steps={params.n_steps}, dt={dt:.3e} s",
        flush=True,
    )
    print(
        f"[flow] poisson_iter={params.poisson_iter}, mask={params.mask_name}",
        flush=True,
    )

    u, v, p, fluid, solid = run_simulation(
        nx=params.nx,
        ny=params.ny,
        lx=params.lx,
        ly=params.ly,
        nu=params.nu,
        rho=params.rho,
        p_left=params.p_left,
        p_right=params.p_right,
        n_steps=params.n_steps,
        poisson_iter=params.poisson_iter,
        print_every=params.print_every,
        mask_name=params.mask_name,
    )
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        u, v, p, fluid, solid,
        out_dir / "flow.png",
        lx=params.lx,
        ly=params.ly,
        dp_mmhg=params.dp / MMHG_TO_PA,
    )


if __name__ == "__main__":
    main()
