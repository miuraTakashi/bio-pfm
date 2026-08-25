"""
Direct Python translation of:
  Mathematica/1_continuous_scheme/1_4_nonlocal_term/armstrong_1d/ArmstrongCellSortingEngulfment.nb

This script intentionally follows notebook choices:
  dx=0.1, n=200, dt=1e-4
  uI=vI=0.2+0.01*RandomReal[]
  m=1
  sensingRadius=1  -> sensingRadiusLattice=Round[sensingRadius/dx]=10
  kernel = {1,...,1,0,-1,...,-1} (length 21)
  su=25, sv=2.5, cuv=5
  duPer1 = Nest[duDt, {u,v}, Round[(1/dt)/10]]
  result = NestList[duPer1, {uI,vI}, 200]
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def list_convolve_mathematica(kernel: np.ndarray, arr: np.ndarray, center: int) -> np.ndarray:
    """
    Mathematica ListConvolve[kernel, arr, center] compatible (1D).
    - center is 1-based index in kernel (nb uses center=sr+1)
    - out-of-range samples are treated as 0 (no periodic wrap in ListConvolve)
    """
    n = arr.size
    m = kernel.size
    out = np.zeros_like(arr)
    ii = np.arange(n)
    for j in range(m):
        lag = j - (center - 1)  # Mathematica index shift
        idx = ii - lag
        mask = (idx >= 0) & (idx < n)
        out[mask] += kernel[j] * arr[idx[mask]]
    return out


def simulate_notebook_engulfment(
    *,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # --- notebook constants ---
    dx = 0.1
    n = 200
    dt = 0.0001
    m = 1.0
    sensing_radius = 1.0
    sr = int(round(sensing_radius / dx))  # 10
    su = 25.0
    sv = 2.5
    cuv = 5.0

    # --- initial condition ---
    rng = np.random.default_rng(seed)
    u = 0.2 + 0.01 * rng.random(n)
    v = 0.2 + 0.01 * rng.random(n)
    x = (np.arange(n, dtype=float) + 0.5) * dx

    # kernel = {1..1,0,-1..-1}, length 2*sr+1
    kernel = np.ones(2 * sr + 1, dtype=float)
    kernel[sr] = 0.0
    kernel[sr + 1 :] = -1.0
    center = sr + 1  # Mathematica 1-based center index

    def guu(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return np.where(uu + vv < m, uu * (1.0 - (uu + vv) / m), 0.0)

    def guv(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return np.where(uu + vv < m, vv * (1.0 - (uu + vv) / m), 0.0)

    def ku(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return su * list_convolve_mathematica(kernel, guu(uu, vv), center) + cuv * list_convolve_mathematica(
            kernel, guv(uu, vv), center
        )

    def kv(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return sv * list_convolve_mathematica(kernel, guv(uu, vv), center) + cuv * list_convolve_mathematica(
            kernel, guu(uu, vv), center
        )

    def du_dt(uu: np.ndarray, vv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kuu = ku(uu, vv)
        kvv = kv(uu, vv)
        du = (
            (np.roll(uu, -1) + np.roll(uu, 1) - 2.0 * uu) / (dx * dx)
            - (np.roll(uu * kuu, -1) - np.roll(uu * kuu, 1)) / (2.0 * dx)
        )
        dv = (
            (np.roll(vv, -1) + np.roll(vv, 1) - 2.0 * vv) / (dx * dx)
            - (np.roll(vv * kvv, -1) - np.roll(vv * kvv, 1)) / (2.0 * dx)
        )
        return uu + dt * du, vv + dt * dv

    # duPer1 = Nest[duDt, {u,v}, Round[(1/dt)/10]]
    n_inner = int(round((1.0 / dt) / 10.0))  # 1000
    # result = NestList[duPer1, {uI,vI}, 200]
    n_outer = 200

    traj_u = [u.copy()]
    traj_v = [v.copy()]
    for _ in range(n_outer):
        for _ in range(n_inner):
            u, v = du_dt(u, v)
        traj_u.append(u.copy())
        traj_v.append(v.copy())

    dt_out = dt * n_inner  # 0.1
    return x, np.array(traj_u), np.array(traj_v), dt_out


def main() -> None:
    x, tu, tv, dt_out = simulate_notebook_engulfment(seed=7)

    # notebook-compatible snapshots for t=0,5,10,20 (dt_out=0.1)
    snap_idx = (0, 50, 100, 200)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    for i, (ax, k) in enumerate(zip(axes.ravel(), snap_idx)):
        ax.plot(x, tu[k], color="C0", lw=1.2, label="u")
        ax.plot(x, tv[k], color="C1", lw=1.2, label="v")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"t = {k * dt_out:.1f}")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("density")

    axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(
        "ArmstrongCellSortingEngulfment.nb -> Python (direct translation)\n"
        "dx=0.1, n=200, dt=1e-4, su=25, sv=2.5, cuv=5, logistic g",
        fontsize=11,
    )

    out_path = Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # GIF movie over all saved outer steps (dt_out = 0.1)
    gif_path = Path(__file__).with_suffix(".gif")
    frames: list[Image.Image] = []
    for k in range(tu.shape[0]):
        fig_f, ax_f = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
        ax_f.plot(x, tu[k], color="C0", lw=1.2, label="u")
        ax_f.plot(x, tv[k], color="C1", lw=1.2, label="v")
        ax_f.set_xlim(float(x[0]), float(x[-1]))
        ax_f.set_ylim(0.0, 1.0)
        ax_f.set_xlabel("x")
        ax_f.set_ylabel("density")
        ax_f.set_title(f"t = {k * dt_out:.1f}")
        ax_f.grid(alpha=0.2)
        ax_f.legend(loc="upper right", fontsize=8, frameon=False)

        fig_f.canvas.draw()
        buf = np.asarray(fig_f.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf[:, :, :3], mode="RGB"))
        plt.close(fig_f)

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
            optimize=False,
        )
        print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
