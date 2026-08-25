"""
Lorentz attractor (Lorenz equations) — Mathematica/0_ode/lorentz_attractor/ChaosLorentzAttractor.nb から移植。
dx/dt = sigma*(y-x),  dy/dt = rho*x - y - x*z,  dz/dt = x*y - beta*z
オイラー法で oneStep + NestList 相当。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def one_step(xyz: np.ndarray, sigma: float, rho: float, beta: float, dt: float) -> np.ndarray:
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = rho * x - y - x * z
    dz = x * y - beta * z
    return xyz + dt * np.array([dx, dy, dz])


def run_euler(xyz0, sigma: float, rho: float, beta: float, dt: float, n_steps: int):
    """オイラー法で時系列を計算。"""
    xyz = np.array(xyz0, dtype=float)
    history = [xyz.copy()]
    for _ in range(n_steps - 1):
        xyz = one_step(xyz, sigma, rho, beta, dt)
        history.append(xyz.copy())
    return np.array(history)


def main():
    rho = 28.0 * 2   # 56
    sigma = 10.0
    beta = 8.0 / 3.0
    dt = 0.01
    n_steps = 100_000
    xyz0 = np.array([0.1, 0.0, 0.0])

    result = run_euler(xyz0, sigma, rho, beta, dt, n_steps)
    t = np.arange(len(result)) * dt
    x_series = result[:, 0]
    y_series = result[:, 1]
    z_series = result[:, 2]

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(t, x_series)
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    ax1.set_title("Lorentz attractor — x(t)")
    ax1.grid(True, alpha=0.3)

    ax3d = fig.add_subplot(122, projection="3d")
    ax3d.plot(result[:, 0], result[:, 1], result[:, 2], lw=0.3, alpha=0.8)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Phase space (x,y,z)")
    plt.tight_layout()
    out = (Path(__file__).resolve().parent / "results" / "lorentz_attractor.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
