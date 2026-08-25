"""
1D Swift-Hohenberg 方程式（Mathematica/1_continuous_scheme/1_3_biharmonic_term/swift_hohenberg_1d/1.1.2.SHH1D.nb から移植）。

方程式:
  ∂u/∂t = r*u - (1 + ∂²/∂x²)²u - u³

スペクトル空間で線形項は陰的、非線形項 -u³ は陽の IMEX Euler（剛性に耐える）。
RK4 + 大きい dt は線形の (1+∂²)² により容易に不安定化するため不採用。

周期境界、角波数 κ = 2π m / L。フーリエでは (1+∂²)² → (1-κ²)²。
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_sh1d(
    n: int = 256,
    L: float = 8 * np.pi,
    r: float = 0.35,
    dt: float = 0.15,
    n_steps: int = 15000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IMEX Euler（擬スペクトル）。戻り値: x, times, traj (n_snap × n)。
    """
    rng = np.random.default_rng(seed)
    dx = L / n
    x = np.linspace(0.0, L, n, endpoint=False)
    kappa = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    L_hat = r - (1.0 - kappa**2) ** 2
    denom = 1.0 - dt * L_hat

    u = 0.04 * rng.standard_normal(n)
    u_hat = np.fft.rfft(u)

    save_every = max(1, n_steps // 600)
    traj_list = [u.copy()]
    times_list = [0.0]

    for step in range(1, n_steps + 1):
        u = np.fft.irfft(u_hat, n=n)
        nl_hat = np.fft.rfft(u**3)
        u_hat = (u_hat - dt * nl_hat) / denom

        if step % save_every == 0:
            traj_list.append(np.fft.irfft(u_hat, n=n).copy())
            times_list.append(step * dt)

    return x, np.array(times_list), np.array(traj_list)


def main():
    r_sh = 0.35
    x, times, traj = simulate_sh1d(r=r_sh)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[1.15, 1.0])
    t_max = float(times[-1])
    dx = float(x[1] - x[0])
    im = axes[0].imshow(
        traj,
        origin="upper",
        aspect="auto",
        extent=[0.0, float(x[-1] + dx), t_max, 0.0],
        interpolation="bilinear",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(f"u(x, t) heatmap  (r={r_sh}, IMEX Euler / spectral)")
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02, label="u")

    n_show = min(6, len(traj))
    idxs = np.linspace(0, len(traj) - 1, n_show, dtype=int)
    for j in idxs:
        axes[1].plot(x, traj[j], lw=1.2, label=f"t = {times[j]:.1f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].set_title("Snapshots")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Swift–Hohenberg 1D — stripe pattern (supercritical r)", fontsize=12)
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "results" / "swift_hohenberg_1d.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}  |u|_max final = {np.max(np.abs(traj[-1])):.4f}")


if __name__ == "__main__":
    main()
