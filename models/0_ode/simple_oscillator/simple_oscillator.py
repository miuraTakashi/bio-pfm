"""
単振動子 (Simple harmonic oscillator) — Mathematica/0_ode/simple_oscillator/単振動子.nb から移植。
du/dt = -v,  dv/dt = u  =>  d²u/dt² = -u

時間積分:
  - Explicit Euler（陽的オイラー）
  - 4 次 Runge–Kutta (RK4)

同一 dt・同一初期条件で両者を比較（Euler はエネルギーが増大しやすく、
RK4 は位相軌道が真円に近い）。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def f(u: float, v: float):
    """右辺: (du/dt, dv/dt) = (-v, u)."""
    return np.array([-v, u], dtype=float)


def run_euler(u0, dt: float, n_steps: int):
    """Explicit Euler で時系列を計算。"""
    state = np.array(u0, dtype=float)
    history = [state.copy()]
    for _ in range(n_steps - 1):
        du, dv = f(state[0], state[1])
        state = state + dt * np.array([du, dv])
        history.append(state.copy())
    return np.array(history)


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """1 ステップの RK4。state = [u, v]."""
    u, v = state[0], state[1]
    k1 = f(u, v)
    k2 = f(u + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
    k3 = f(u + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
    k4 = f(u + dt * k3[0], v + dt * k3[1])
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def run_rk4(u0, dt: float, n_steps: int):
    """4 次 Runge–Kutta で時系列を計算。"""
    state = np.array(u0, dtype=float)
    history = [state.copy()]
    for _ in range(n_steps - 1):
        state = rk4_step(state, dt)
        history.append(state.copy())
    return np.array(history)


def main():
    simulation_length = 50.0
    u0 = np.array([0.2, 0.0])
    dt = 0.1
    n_steps = int(round(simulation_length / dt))

    euler = run_euler(u0, dt, n_steps)
    rk4 = run_rk4(u0, dt, n_steps)
    t = np.arange(len(euler)) * dt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(t, euler[:, 0], label="u (Euler)", ls="-", alpha=0.85)
    axes[0].plot(t, euler[:, 1], label="v (Euler)", ls="--", alpha=0.85)
    axes[0].plot(t, rk4[:, 0], label="u (RK4)", ls="-", alpha=0.85)
    axes[0].plot(t, rk4[:, 1], label="v (RK4)", ls=":", alpha=0.85)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u, v")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Time series")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(euler[:, 0], euler[:, 1], label="Euler", alpha=0.85)
    axes[1].plot(rk4[:, 0], rk4[:, 1], label="RK4", alpha=0.85)
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("v")
    axes[1].set_aspect("equal")
    axes[1].legend()
    axes[1].set_title("Phase plane (u, v)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Simple harmonic oscillator: Euler vs RK4")
    plt.tight_layout()
    out = (Path(__file__).resolve().parent / "results" / "simple_oscillator.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
