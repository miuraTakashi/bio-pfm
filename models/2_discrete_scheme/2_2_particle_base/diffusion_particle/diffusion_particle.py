"""
ブラウン運動の可視化（Mathematica/2_discrete_scheme/2_2_particle_base/diffusion_particle/DiffusionVisual.nb から移植）。

200 個の粒子を (0.5, 0.5) から出発させ、各ステップで
  p_{t+1} = p_t + 0.02 * Uniform[-1, 1]^2
の更新を 500 ステップ繰り返す。

出力図は 2 パネル:
  1) 全粒子の軌跡（x-y 平面）
  2) 平均 MSD = <|p_t - p_0|^2> の時間変化（理論直線 ~ t も併記）
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_brownian(n_particles: int = 200, n_steps: int = 500, step_size: float = 0.02,
                      seed: int = 42) -> np.ndarray:
    """ブラウン運動シミュレーション。shape: (n_steps+1, n_particles, 2)"""
    rng = np.random.default_rng(seed)
    positions = np.full((n_particles, 2), 0.5)
    history = [positions.copy()]
    for _ in range(n_steps):
        positions = positions + step_size * rng.uniform(-1, 1, size=(n_particles, 2))
        history.append(positions.copy())
    return np.array(history)


def main() -> None:
    n_particles, n_steps, step_size = 200, 500, 0.02
    history = simulate_brownian(
        n_particles=n_particles,
        n_steps=n_steps,
        step_size=step_size,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel 1: trajectories of all particles
    ax_traj = axes[0]
    for i in range(history.shape[1]):
        ax_traj.plot(
            history[:, i, 0],
            history[:, i, 1],
            linewidth=0.7,
            alpha=0.35,
            color="steelblue",
        )
    ax_traj.scatter(history[0, :, 0], history[0, :, 1], s=8, color="black", alpha=0.6, label="start")
    ax_traj.scatter(history[-1, :, 0], history[-1, :, 1], s=8, color="tab:red", alpha=0.6, label="end")
    ax_traj.set_aspect("equal")
    ax_traj.set_title("Particle trajectories")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.legend(fontsize=8)
    ax_traj.grid(alpha=0.25)

    # Panel 2: ensemble-averaged mean squared displacement
    ax_msd = axes[1]
    disp = history - history[0:1, :, :]
    msd_each = np.sum(disp * disp, axis=2)  # shape: (n_steps+1, n_particles)
    msd_mean = msd_each.mean(axis=1)
    t = np.arange(history.shape[0], dtype=float)
    ax_msd.plot(t, msd_mean, color="tab:blue", linewidth=2.0, label="mean MSD")

    # E[MSD_t] = t * 2 * step_size^2 * Var(U), Var(U[-1,1]) = 1/3
    msd_ref = t * (2.0 * step_size * step_size / 3.0)
    ax_msd.plot(t, msd_ref, "k--", linewidth=1.6, label=r"reference $\propto t$")
    ax_msd.set_title("Ensemble-averaged MSD")
    ax_msd.set_xlabel("step")
    ax_msd.set_ylabel(r"$\langle |p_t - p_0|^2 \rangle$")
    ax_msd.grid(alpha=0.3)
    ax_msd.legend(fontsize=8)

    fig.suptitle(f"Brownian random walk ({n_particles} particles, step_size={step_size})")
    plt.tight_layout()
    out_path = Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
