"""
Cahn-Hilliard 相分離（1D / 2D）
（Mathematica/1_continuous_scheme/1_3_biharmonic_term/cahn_hilliard_coarsening/1.1.4.Cahn-Hilliard.nb を参考に、phase_field_branching.py と同系の
 FFT 半陰式スキームで実装）。

方程式（周期境界）:
  ∂u/∂t = M ∇² μ,   μ = -ε² ∇²u - f(u),   f(u) = u - u³

半陰式（スペクトル・オイラー、線形項を陰的）:
  û^{n+1} = ( û^n + dt M k² f̂(u^n) ) / (1 + dt M ε² k⁴)
  （∂û/∂t = -M ε² k⁴ û + M k² f̂ より。以前の実装は f 項の符号が逆だった。）

ここで k² は角波数の二乗（1D: 2π*rfftfreq、2D: kx=2π*fftfreq, ky=2π*rfftfreq）。
入力は実数のみのため rfft / irfft、rfft2 / irfft2 に統一。norm="ortho"。

スピノーダル分解のため平均濃度 ⟨u⟩ は不安定帯 |⟨u⟩| < 1/√3 に置く。
初期値: u = u_mean + noise_amp * N(0,1)（小さなランダム摂動でモードを励起）。

パラメータ例: M=1, ε=0.5, dx=1, dt=0.005, n_steps=120000（T=600）,
              u_mean=0.35, noise_amp=0.05
"""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

SPINODAL_THRESHOLD = 1.0 / np.sqrt(3.0)


def f_double_well(u: np.ndarray) -> np.ndarray:
    """化学ポテンシャルに入る非線形項 f(u) = u - u³。"""
    return u - u ** 3


def k_squared_rfft_1d(n: int, dx: float) -> np.ndarray:
    """rfft 用の k²（長さ n//2+1）。"""
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    return k * k


def k_squared_rfft_2d(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    """rfft2 用の k²（形状 (nx, ny//2+1)）。"""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    kx2, ky2 = np.meshgrid(kx ** 2, ky ** 2, indexing="ij")
    return kx2 + ky2


def step_ch_semi_implicit_rfft_1d(
    u: np.ndarray, dt: float, M: float, eps: float, k2: np.ndarray
) -> np.ndarray:
    """1D: 半陰式スペクトル更新（rfft / irfft）。"""
    f_hat = np.fft.rfft(f_double_well(u), norm="ortho")
    u_hat = np.fft.rfft(u, norm="ortho")
    denom = 1.0 + dt * M * (eps ** 2) * (k2 ** 2)
    numer = u_hat + dt * M * k2 * f_hat
    return np.fft.irfft(numer / denom, n=u.shape[0], norm="ortho")


def step_ch_semi_implicit_rfft_2d(
    u: np.ndarray, dt: float, M: float, eps: float, k2: np.ndarray
) -> np.ndarray:
    """2D: 半陰式スペクトル更新（rfft2 / irfft2）。"""
    n0, n1 = u.shape
    f_hat = np.fft.rfft2(f_double_well(u), norm="ortho")
    u_hat = np.fft.rfft2(u, norm="ortho")
    denom = 1.0 + dt * M * (eps ** 2) * (k2 ** 2)
    numer = u_hat + dt * M * k2 * f_hat
    return np.fft.irfft2(numer / denom, s=(n0, n1), norm="ortho")


def simulate_cahn_hilliard_1d(
    n: int = 100,
    e: float = 0.5,
    M: float = 1.0,
    dx: float = 1.0,
    dt: float = 0.005,
    n_steps: int = 120000,
    seed: int = 42,
    u_mean: float = 0.35,
    noise_amp: float = 0.05,
    save_every: int | None = None,
) -> tuple:
    rng = np.random.default_rng(seed)
    u = u_mean + noise_amp * rng.standard_normal(n)
    k2 = k_squared_rfft_1d(n, dx)
    snapshots = [u.copy()]
    times = [0.0]
    if save_every is None:
        save_every = max(1, n_steps // 4)
    for step in range(n_steps):
        u = step_ch_semi_implicit_rfft_1d(u, dt, M, e, k2)
        if (step + 1) % save_every == 0:
            snapshots.append(u.copy())
            times.append((step + 1) * dt)
    x = np.arange(n, dtype=float) * dx
    return x, snapshots, times


def simulate_cahn_hilliard_2d(
    n: int = 100,
    e: float = 0.5,
    M: float = 1.0,
    dx: float = 1.0,
    dt: float = 0.005,
    n_steps: int = 120000,
    seed: int = 42,
    u_mean: float = 0.35,
    noise_amp: float = 0.05,
    save_every: int | None = None,
) -> tuple:
    rng = np.random.default_rng(seed + 1)
    u = u_mean + noise_amp * rng.standard_normal(size=(n, n))
    k2 = k_squared_rfft_2d(n, n, dx, dx)
    snapshots = [u.copy()]
    times = [0.0]
    if save_every is None:
        save_every = max(1, n_steps // 4)
    for step in range(n_steps):
        u = step_ch_semi_implicit_rfft_2d(u, dt, M, e, k2)
        if (step + 1) % save_every == 0:
            snapshots.append(u.copy())
            times.append((step + 1) * dt)
    return snapshots, times


def save_cahn_hilliard_2d_gif(
    snaps_2d: list[np.ndarray], times: list[float], out_path: Path
) -> None:
    """2D 時系列スナップショットを GIF として保存。"""
    fig, ax = plt.subplots(figsize=(5.3, 4.8))
    im = ax.imshow(
        snaps_2d[0].T,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_title(f"2D  u (t={times[0]:.0f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("u")
    fig.tight_layout()

    def _update(i: int):
        im.set_data(snaps_2d[i].T)
        ax.set_title(f"2D  u (t={times[i]:.0f})")
        return (im,)

    ani = FuncAnimation(
        fig,
        _update,
        frames=len(snaps_2d),
        interval=120,
        blit=False,
        repeat=True,
    )
    ani.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def run_case(case_name: str, params: dict, require_spinodal: bool) -> None:
    """Run one Cahn-Hilliard case and export PNG/GIF."""
    in_spinodal = abs(params["u_mean"]) < SPINODAL_THRESHOLD
    if require_spinodal and (not in_spinodal):
        raise ValueError(
            f"{case_name}: u_mean is outside spinodal regime. "
            f"|u_mean|={abs(params['u_mean']):.3f}, 1/sqrt(3)={SPINODAL_THRESHOLD:.3f}"
        )
    if (not require_spinodal) and in_spinodal:
        raise ValueError(
            f"{case_name}: u_mean is inside spinodal regime. "
            f"|u_mean|={abs(params['u_mean']):.3f}, 1/sqrt(3)={SPINODAL_THRESHOLD:.3f}"
        )

    x, snaps_1d, times = simulate_cahn_hilliard_1d(**params)
    snaps_2d, _ = simulate_cahn_hilliard_2d(**params)

    # PNG は代表時刻のみを表示（可読性のため 5 枚に間引き）
    n_total = len(times)
    pick = np.linspace(0, n_total - 1, 5, dtype=int)
    pick = np.unique(pick)
    n_snap = len(pick)
    fig, axes = plt.subplots(
        2, n_snap, figsize=(3.2 * n_snap, 6.5), constrained_layout=True
    )

    for col, idx in enumerate(pick):
        t = times[idx]
        axes[0, col].plot(x, snaps_1d[idx], color="C0", lw=0.9)
        axes[0, col].axhline(1.0, color="k", ls="--", lw=0.4, alpha=0.45)
        axes[0, col].axhline(-1.0, color="k", ls="--", lw=0.4, alpha=0.45)
        axes[0, col].set_ylim(-1.5, 1.5)
        axes[0, col].set_title(f"1D  u (t={t:.0f})", fontsize=9)
        axes[0, col].grid(alpha=0.25)
        if col == 0:
            axes[0, col].set_ylabel("u")
        axes[0, col].set_xlabel("x")

        im = axes[1, col].imshow(
            snaps_2d[idx].T,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=-1.0,
            vmax=1.0,
        )
        axes[1, col].set_title(f"2D  u (t={t:.0f})", fontsize=9)
        axes[1, col].axis("off")

    fig.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.02, label="u")
    regime = "spinodal" if in_spinodal else "non-spinodal"
    cond = "|⟨u⟩₀|<1/sqrt(3)" if in_spinodal else "|⟨u⟩₀|>1/sqrt(3)"
    fig.suptitle(
        f"Cahn–Hilliard {regime} case (semi-implicit RFFT): 1D + 2D  "
        f"(ε={params['e']}, M={params['M']}, dt={params['dt']}, "
        f"T≈{params['n_steps'] * params['dt']:.0f}, "
        f"⟨u⟩₀≈{params['u_mean']}, {cond}, 1/sqrt(3)≈{SPINODAL_THRESHOLD:.3f})",
        fontsize=12,
    )
    out_png = Path(__file__).with_name(f"cahn_hilliard_{case_name}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")

    out_gif = Path(__file__).with_name(f"cahn_hilliard_{case_name}.gif")
    save_cahn_hilliard_2d_gif(snaps_2d, times, out_gif)
    print(f"Saved {out_gif}")


def main():
    # スピノーダル分解を狙った代表設定（|u_mean| < 1/sqrt(3)）
    spinodal_params = {
        "n": 128,
        "e": 0.5,
        "M": 1.0,
        "dx": 1.0,
        "dt": 0.005,
        "n_steps": 120_000,
        "seed": 42,
        "u_mean": 0.20,
        "noise_amp": 0.03,
        "save_every": 2_000,
    }
    # 非スピノーダル例（|u_mean| > 1/sqrt(3)）
    non_spinodal_params = dict(spinodal_params)
    non_spinodal_params["u_mean"] = 0.70

    run_case("spinodal", spinodal_params, require_spinodal=True)
    run_case("non_spinodal", non_spinodal_params, require_spinodal=False)

    # Backward-compatible alias: keep legacy filename in sync with spinodal output.
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / "results/cahn_hilliard_spinodal.png", here / "results/cahn_hilliard_coarsening.png")
    shutil.copyfile(here / "results/cahn_hilliard_spinodal.gif", here / "results/cahn_hilliard_coarsening.gif")
    print(f"Saved {here / 'results/cahn_hilliard_coarsening.png'}")
    print(f"Saved {here / 'results/cahn_hilliard_coarsening.gif'}")


if __name__ == "__main__":
    main()
