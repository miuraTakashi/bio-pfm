"""
pipe.raw マスクでの定圧流れを平面 Poiseuille（ハーゲン–ポワズイユ流の 2D 版）と比較する。

出力: results/pipe_poiseuille_verification.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import flow
from flow import MMHG_TO_PA, load_flow_params

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
OUT_PNG = OUT_DIR / "pipe_poiseuille_verification.png"
PIPE_MASK = "masks/pipe.raw"


def poiseuille_profile(
    y: np.ndarray,
    h: float,
    dp_dx: float,
    nu: float,
) -> np.ndarray:
    """平行平板間 Poiseuille: u(y) = -(dp/dx)/(2 nu) y (h - y)."""
    return -(dp_dx) / (2.0 * nu) * y * (h - y)


def run_verification() -> None:
    params = load_flow_params()
    n_steps = int(os.environ.get("FLOW_N_STEPS", "50000"))
    poisson_iter = int(os.environ.get("FLOW_POISSON_ITER", "800"))

    print(
        f"[verify] water nu={params.nu:.3e} m^2/s, "
        f"L={params.lx * 1e3:g} mm, Δp={params.dp / MMHG_TO_PA:g} mmHg",
        flush=True,
    )

    u, v, p, fluid, solid = flow.run_simulation(
        nx=params.nx,
        ny=params.ny,
        lx=params.lx,
        ly=params.ly,
        nu=params.nu,
        rho=params.rho,
        p_left=params.p_left,
        p_right=params.p_right,
        n_steps=n_steps,
        poisson_iter=poisson_iter,
        print_every=max(n_steps // 10, 1),
        mask_name=PIPE_MASK,
    )

    uc, vc = flow.cell_center_velocity(u, v)
    dx = params.lx / params.nx
    dy = params.ly / params.ny
    lx_mm = params.lx * 1.0e3
    ly_mm = params.ly * 1.0e3

    i_mid = params.nx // 2
    y_centers = (np.arange(params.ny) + 0.5) * dy

    u_num = uc[i_mid, :].copy()
    u_num[~fluid[i_mid, :]] = np.nan

    fluid_cols = np.where(fluid[i_mid, :])[0]
    if fluid_cols.size < 3:
        raise RuntimeError("Insufficient fluid cells for profile extraction.")
    j_lo, j_hi = int(fluid_cols[0]), int(fluid_cols[-1])
    y_wall_lo = j_lo * dy if j_lo > 0 else 0.0
    y_wall_hi = (j_hi + 1) * dy if j_hi < params.ny - 1 else params.ly
    h_eff = y_wall_hi - y_wall_lo

    y_fluid = y_centers[fluid_cols]
    u_fluid = uc[i_mid, fluid_cols]

    dp_dx = (params.p_right - params.p_left) / params.lx
    y_rel = y_fluid - y_wall_lo
    u_ref = poiseuille_profile(y_rel, h_eff, dp_dx, params.nu)

    abs_err = np.abs(u_fluid - u_ref)
    rel_err = abs_err / np.max(u_ref)
    l2_abs = float(np.linalg.norm(u_fluid - u_ref) / np.linalg.norm(u_ref))
    max_rel = float(np.max(rel_err))
    amp_ratio = float(np.max(u_fluid) / np.max(u_ref))

    u_norm = u_fluid / np.max(u_fluid)
    u_ref_norm = u_ref / np.max(u_ref)
    shape_err = float(np.linalg.norm(u_norm - u_ref_norm) / np.linalg.norm(u_ref_norm))
    corr = float(np.corrcoef(u_norm, u_ref_norm)[0, 1])

    print(f"[verify] Effective channel height h = {h_eff * 1e3:.4f} mm")
    print(f"[verify] u_max numerical = {np.max(u_fluid):.6e} m/s")
    print(f"[verify] u_max analytical  = {np.max(u_ref):.6e} m/s")
    print(f"[verify] Amplitude ratio (num/ref) = {amp_ratio:.4f}")
    print(f"[verify] L2 absolute error         = {l2_abs:.4e}")
    print(f"[verify] L2 shape error (normalized) = {shape_err:.4e}")
    print(f"[verify] Profile correlation       = {corr:.6f}")
    print(f"[verify] Max pointwise rel. error  = {max_rel:.4e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    speed = np.sqrt(uc**2 + vc**2)
    speed = np.where(fluid, speed, np.nan)
    im = axes[0].imshow(
        speed.T,
        origin="lower",
        extent=[0.0, lx_mm, 0.0, ly_mm],
        aspect="auto",
        interpolation="nearest",
    )
    axes[0].axvline(x=i_mid * dx * 1e3, color="w", ls="--", lw=0.8, alpha=0.8)
    axes[0].set_title(f"Speed |u| ({PIPE_MASK})")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="|u| [m/s]")

    axes[1].plot(u_fluid * 1e3, y_fluid * 1e3, "C0", lw=2, label="numerical")
    axes[1].plot(u_ref * 1e3, y_fluid * 1e3, "C1--", lw=2, label="Poiseuille")
    axes[1].set_xlabel("u_x [mm/s]")
    axes[1].set_ylabel("y [mm]")
    axes[1].set_title(f"Velocity profile at x={i_mid * dx * 1e3:.3f} mm")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    shape_err_pts = np.abs(u_norm - u_ref_norm)
    axes[2].plot(shape_err_pts * 100.0, y_fluid * 1e3, "C3", lw=2)
    axes[2].set_xlabel("normalized shape error [%]")
    axes[2].set_ylabel("y [mm]")
    axes[2].set_title(
        f"Shape error (L2={100 * shape_err:.2f}%, r={corr:.4f}, amp={amp_ratio:.3f})"
    )
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Plane Poiseuille (water, Δp={params.dp / MMHG_TO_PA:g} mmHg, "
        f"h={h_eff * 1e3:.3f} mm, steps={n_steps})",
        fontsize=11,
    )
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[verify] Saved {OUT_PNG}")


if __name__ == "__main__":
    run_verification()
