#!/usr/bin/env python3
"""
2D Turing 型反応–拡散（周期境界・拡散の半陰式 FFT）および対称性破れ項 **q u^2** の比較。

Mathematica: `1.1.2.軟骨形成とTuringパターン.nb` の 2D 静領域に相当する部分を拡張。

反応項（活性 u、抑制 v）:
  f(u, v) = 0.6 u - v + q u^2 - u^3,   g(u, v) = 1.5 u - 2 v

ここで **q** は二乗項の係数（スカラーパラメータ）。抑制因子の場を表す変数は **v**
（パラメータ q と混同しない）。

数値積分の実時間は **T = 200** に統一（`simulate_2d` と `q` スイープとも **dt = 0.1**、n_steps = 2000）。
上段は **t = 0, 100, 200** の 3 枚、下段は **q = 0, 0.8, −0.8** の 3 列（いずれも T=200 で対応）。

**数理（線形化）**: 空間一様な原点付近では u^2 は二次以上なので、ヤコビアンは q に依らず

  J(0,0) = [[∂f/∂u, ∂f/∂v], [∂g/∂u, ∂g/∂v]] = [[0.6, -1], [1.5, -2]]

固有値はコードで計算し、図下に表示する。

出力:
  - `results/turing_2d_numerical.png`（2D 数値パネル）
  - `results/turing_2d_analysis.png`（線形安定性解析パネル）
  - `results/turing_2d_numerical_GiererMeinhardt.png` / `results/turing_2d_analysis_GiererMeinhardt.png`
"""
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np


def f_stripe(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """従来の stripe: q=0 に相当（u^2 項なし）。"""
    return 0.6 * u - v - u**3


def f_with_quadratic(u: np.ndarray, v: np.ndarray, q_quad: float) -> np.ndarray:
    """f = 0.6 u - v + q_quad * u^2 - u^3"""
    return 0.6 * u - v + q_quad * u**2 - u**3


def g_reaction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 1.5 * u - 2.0 * v


def jacobian_origin() -> tuple[np.ndarray, np.ndarray]:
    """反応項のみ、空間一様 (u,v)=(0,0) におけるヤコビアン（q は出ない）。"""
    j = np.array([[0.6, -1.0], [1.5, -2.0]], dtype=np.float64)
    return j, np.linalg.eigvals(j)


# Gierer–Meinhardt（均一正の定常解 (u*,v*) まわり）— turing_1d と同型の係数
GM_RHO0 = 0.025
GM_RHO = 1.0
GM_MU = 1.0
GM_SIG = 1.0
GM_NU = 1.0
GM_EPS = 0.01
GM_DU = 0.0002
GM_DV = 0.015


def gm_homogeneous_ss() -> tuple[float, float]:
    u_star = (GM_RHO0 + GM_RHO * GM_NU / GM_SIG) / GM_MU
    v_star = GM_SIG * u_star * u_star / GM_NU
    return float(u_star), float(v_star)


def gm_jacobian_at_ss() -> np.ndarray:
    u_star, v_star = gm_homogeneous_ss()
    ve = v_star + GM_EPS
    fu = 2.0 * GM_RHO * u_star / ve - GM_MU
    fv = -GM_RHO * u_star * u_star / (ve * ve)
    gu = 2.0 * GM_SIG * u_star
    gv = -GM_NU
    return np.array([[fu, fv], [gu, gv]], dtype=np.float64)


def f_gm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return GM_RHO0 + GM_RHO * u * u / (v + GM_EPS) - GM_MU * u


def g_gm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return GM_SIG * u * u - GM_NU * v


def max_real_growth_rate_curve(
    dp: float,
    dq: float,
    *,
    jacobian: np.ndarray | None = None,
    k_max: float = 90.0,
    n_samples: int = 600,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    A(k)=J-k^2 D の最大実部成長率 sigma_max(k) と最不安定波数を返す。
    戻り値: k_grid, sigma_grid, k_star, lambda_star
    """
    if jacobian is None:
        j, _ = jacobian_origin()
    else:
        j = np.asarray(jacobian, dtype=np.float64)
    dmat = np.diag([dp, dq]).astype(np.float64)
    k_grid = np.linspace(0.0, k_max, n_samples)
    sigma_grid = np.empty_like(k_grid)

    for i, k in enumerate(k_grid):
        a = j - (k * k) * dmat
        sigma_grid[i] = float(np.max(np.real(np.linalg.eigvals(a))))

    i_star = int(np.argmax(sigma_grid))
    k_star = float(k_grid[i_star])
    lambda_star = float((2.0 * np.pi / k_star) if k_star > 1e-12 else np.inf)
    return k_grid, sigma_grid, k_star, lambda_star


def isotropic_spectrum_peak_k(field: np.ndarray, dx: float) -> tuple[float, float]:
    """
    2D 場の等方平均スペクトルから支配的波数と波長を推定。
    戻り値: k_peak, lambda_peak
    """
    arr = np.asarray(field, dtype=np.float64)
    n, m = arr.shape
    centered = arr - float(np.mean(arr))
    fhat = np.fft.fft2(centered)
    power = np.abs(fhat) ** 2

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(m, d=dx)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.sqrt(kxg * kxg + kyg * kyg)

    kmax = float(kmag.max())
    n_bins = max(64, n // 2)
    bins = np.linspace(0.0, kmax, n_bins + 1)
    which = np.digitize(kmag.ravel(), bins) - 1
    which = np.clip(which, 0, n_bins - 1)
    radial_power = np.bincount(which, weights=power.ravel(), minlength=n_bins)
    radial_count = np.bincount(which, minlength=n_bins)
    radial_mean = radial_power / np.maximum(radial_count, 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    if radial_mean.size <= 1:
        return 0.0, np.inf
    radial_mean[0] = 0.0
    idx = int(np.argmax(radial_mean))
    k_peak = float(k_centers[idx])
    lambda_peak = float((2.0 * np.pi / k_peak) if k_peak > 1e-12 else np.inf)
    return k_peak, lambda_peak


def build_implicit_kernel_2d(grid_size: int, dt: float, d_coeff: float, dx: float):
    kernel = np.zeros((grid_size, grid_size))
    c = dt * d_coeff / (dx * dx)
    kernel[0, 0] = 1.0 + 4.0 * c
    kernel[1, 0] = -c
    kernel[-1, 0] = -c
    kernel[0, 1] = -c
    kernel[0, -1] = -c
    return 1.0 / np.fft.fft2(kernel)


def simulate_2d(
    grid_size: int = 64,
    dt: float = 0.1,
    dp: float = 0.0002,
    dq: float = 0.01,
    T_final: float = 200.0,
    snap_times: tuple[float, ...] = (0.0, 100.0, 200.0),
    seed: int = 42,
):
    """2D 静領域（stripe 型 f）。実時間 T_final；u のスナップショットは snap_times に対応（最大 3 点想定）。"""
    dx = 1.0 / grid_size
    n_steps = int(round(T_final / dt))

    ku = build_implicit_kernel_2d(grid_size, dt, dp, dx)
    kv = build_implicit_kernel_2d(grid_size, dt, dq, dx)

    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.1, 0.1, (grid_size, grid_size))
    v = rng.uniform(-0.1, 0.1, (grid_size, grid_size))

    snap_times_sorted = tuple(sorted(set(snap_times)))
    target_step_counts = {
        int(round(t / dt)) for t in snap_times_sorted if t > 0.0 and int(round(t / dt)) <= n_steps
    }

    snapshots_u: list[np.ndarray] = []
    snap_times_out: list[float] = []
    if any(t <= 0.0 for t in snap_times_sorted):
        snapshots_u.append(u.copy())
        snap_times_out.append(0.0)

    for step in range(n_steps):
        fu = u + dt * f_stripe(u, v)
        gv = v + dt * g_reaction(u, v)
        u = np.real(np.fft.ifft2(ku * np.fft.fft2(fu)))
        v = np.real(np.fft.ifft2(kv * np.fft.fft2(gv)))
        sc = step + 1
        if sc in target_step_counts:
            snapshots_u.append(u.copy())
            snap_times_out.append(float(sc * dt))

    return u, v, snapshots_u, snap_times_out


def simulate_2d_gm(
    grid_size: int = 64,
    dt: float = 0.1,
    du: float = GM_DU,
    dv: float = GM_DV,
    T_final: float = 200.0,
    snap_times: tuple[float, ...] = (0.0, 100.0, 200.0),
    seed: int = 44,
):
    """Gierer–Meinhardt 反応項の 2D 半陰式 FFT 積分（周期境界）。"""
    dx = 1.0 / grid_size
    n_steps = int(round(T_final / dt))

    ku = build_implicit_kernel_2d(grid_size, dt, du, dx)
    kv = build_implicit_kernel_2d(grid_size, dt, dv, dx)

    u_star, v_star = gm_homogeneous_ss()
    rng = np.random.default_rng(seed)
    u = u_star + rng.normal(0.0, 0.02, (grid_size, grid_size))
    v = v_star + rng.normal(0.0, 0.02, (grid_size, grid_size))
    v = np.maximum(v, 1e-6)

    snap_times_sorted = tuple(sorted(set(snap_times)))
    target_step_counts = {
        int(round(t / dt)) for t in snap_times_sorted if t > 0.0 and int(round(t / dt)) <= n_steps
    }

    snapshots_u: list[np.ndarray] = []
    snap_times_out: list[float] = []
    if any(t <= 0.0 for t in snap_times_sorted):
        snapshots_u.append(u.copy())
        snap_times_out.append(0.0)

    for step in range(n_steps):
        fv = v + dt * g_gm(u, v)
        fu = u + dt * f_gm(u, v)
        u = np.real(np.fft.ifft2(ku * np.fft.fft2(fu)))
        v = np.real(np.fft.ifft2(kv * np.fft.fft2(fv)))
        v = np.maximum(v, 1e-8)
        sc = step + 1
        if sc in target_step_counts:
            snapshots_u.append(u.copy())
            snap_times_out.append(float(sc * dt))

    return u, v, snapshots_u, snap_times_out


def simulate_with_q_quad(
    q_quad: float,
    grid_size: int = 64,
    dt: float = 0.1,
    dp: float = 0.0002,
    dq: float = 0.01,
    T_final: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """f = 0.6 u - v + q_quad * u^2 - u^3 の半陰式 FFT 積分。T_final まで進めた u を返す。"""
    dx = 1.0 / grid_size
    n_steps = int(round(T_final / dt))
    ku = build_implicit_kernel_2d(grid_size, dt, dp, dx)
    kv = build_implicit_kernel_2d(grid_size, dt, dq, dx)

    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.1, 0.1, (grid_size, grid_size))
    v = rng.uniform(-0.1, 0.1, (grid_size, grid_size))

    for _ in range(n_steps):
        fu = u + dt * f_with_quadratic(u, v, q_quad)
        gv = v + dt * g_reaction(u, v)
        u = np.real(np.fft.ifft2(ku * np.fft.fft2(fu)))
        v = np.real(np.fft.ifft2(kv * np.fft.fft2(gv)))

    return u


def main() -> None:
    j0, eig0 = jacobian_origin()
    print("Jacobian J(0,0) (reaction kinetics, q-independent):\n", j0)
    print("Eigenvalues:", eig0)

    dt = 0.1
    dp = 0.0002
    dq = 0.01
    grid_size = 64
    t_end = 200.0

    _, _, snapshots, snap_times = simulate_2d(grid_size=grid_size, dt=dt, dp=dp, dq=dq, T_final=t_end)
    q_values = (0.0, 0.8, -0.8)
    results_q = {
        q: simulate_with_q_quad(
            q,
            grid_size=grid_size,
            dt=dt,
            dp=dp,
            dq=dq,
            T_final=t_end,
        )
        for q in q_values
    }

    k_grid, sigma_grid, k_star, lambda_star = max_real_growth_rate_curve(dp=dp, dq=dq)
    dx = 1.0 / grid_size
    k_spec, lambda_spec = isotropic_spectrum_peak_k(results_q[0.0], dx=dx)

    fig_num, axes_num = plt.subplots(2, 3, figsize=(10.2, 7.4))
    axes_top = list(axes_num[0, :])
    axes_bottom = list(axes_num[1, :])

    for i, (snap, t_phys) in enumerate(zip(snapshots, snap_times)):
        ax = axes_top[i]
        im = ax.imshow(
            snap,
            origin="lower",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(f"2D stripe  u  ($t={t_phys:.0f}$)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    for j, qcoef in enumerate(q_values):
        ax = axes_bottom[j]
        im = ax.imshow(
            results_q[qcoef],
            origin="lower",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(
            f"q = {qcoef:g}\n"
            f"$f = 0.6u - v + ({qcoef:g})u^2 - u^3$,  "
            f"$t={t_end:.0f}$",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    eig_str = ", ".join(f"{z:.4f}" for z in sorted(eig0, key=lambda z: z.real))
    fig_num.suptitle(
        "Turing 2D — stripe +  q·u²  symmetry-breaking (q=0, 0.8, −0.8)\n"
        rf"Linearization at $(0,0)$: $\lambda \in \{{{eig_str}\}}$ (independent of $q$),  "
        rf"$k_*={k_star:.3f}$",
        fontsize=11,
    )
    fig_num.tight_layout()
    out_num = (Path(__file__).resolve().parent / "results" / "turing_2d_numerical.png")
    fig_num.savefig(out_num, dpi=150, bbox_inches="tight")
    plt.close(fig_num)

    fig_ana, ax_theory = plt.subplots(1, 1, figsize=(7.0, 4.4))
    ax_theory.plot(k_grid, sigma_grid, color="C2", lw=1.5, label=r"$\sigma_{\max}(k)$")
    ax_theory.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_theory.axvline(k_star, color="C3", lw=1.1, ls="--", label=rf"$k_*={k_star:.3f}$")
    ax_theory.axvline(
        k_spec,
        color="C0",
        lw=1.1,
        ls="-.",
        label=rf"$k_{{spec}}={k_spec:.3f}$",
    )
    ax_theory.set_xlabel("k")
    ax_theory.set_ylabel(r"max Re $\lambda(k)$")
    ax_theory.set_title(
        rf"Linear stability: $\lambda_*={lambda_star:.3f}$,  "
        rf"$\lambda_{{spec}}={lambda_spec:.3f}$"
    )
    ax_theory.grid(alpha=0.3)
    ax_theory.legend(fontsize=8, loc="best")
    fig_ana.suptitle("Turing 2D — linear stability analysis", fontsize=12)
    fig_ana.tight_layout()
    out_ana = (Path(__file__).resolve().parent / "results" / "turing_2d_analysis.png")
    fig_ana.savefig(out_ana, dpi=150, bbox_inches="tight")
    plt.close(fig_ana)

    # --- Gierer–Meinhardt 2D（同一グリッド・実時間 T） ---
    _, _, gm_snaps, gm_t_out = simulate_2d_gm(
        grid_size=grid_size,
        dt=dt,
        du=GM_DU,
        dv=GM_DV,
        T_final=t_end,
        snap_times=(0.0, 100.0, 200.0),
        seed=44,
    )
    fig_gm, axes_gm = plt.subplots(1, 3, figsize=(10.2, 3.7))
    for ax, snap, t_phys in zip(np.ravel(axes_gm), gm_snaps, gm_t_out):
        im = ax.imshow(
            snap,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"GM 2D  $u$  ($t={t_phys:.0f}$)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    u_s, v_s = gm_homogeneous_ss()
    fig_gm.suptitle(
        rf"Gierer–Meinhardt 2D — $(u^*,v^*)=({u_s:.3f},{v_s:.3f})$,  "
        rf"$D_u={GM_DU:g}$, $D_v={GM_DV:g}$",
        fontsize=11,
    )
    fig_gm.tight_layout()
    out_gm_num = (Path(__file__).resolve().parent / "results" / "turing_2d_numerical_GiererMeinhardt.png")
    fig_gm.savefig(out_gm_num, dpi=150, bbox_inches="tight")
    plt.close(fig_gm)

    k_grid_gm, sigma_gm, k_star_gm, lambda_star_gm = max_real_growth_rate_curve(
        GM_DU, GM_DV, jacobian=gm_jacobian_at_ss()
    )
    k_spec_gm, lambda_spec_gm = isotropic_spectrum_peak_k(gm_snaps[-1], dx=dx)
    fig_gm_ana, ax_gm = plt.subplots(1, 1, figsize=(7.0, 4.4))
    ax_gm.plot(k_grid_gm, sigma_gm, color="C2", lw=1.5, label=r"$\sigma_{\max}(k)$")
    ax_gm.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_gm.axvline(k_star_gm, color="C3", lw=1.1, ls="--", label=rf"$k_*={k_star_gm:.3f}$")
    ax_gm.axvline(
        k_spec_gm,
        color="C0",
        lw=1.1,
        ls="-.",
        label=rf"$k_{{spec}}={k_spec_gm:.3f}$",
    )
    ax_gm.set_xlabel("k")
    ax_gm.set_ylabel(r"max Re $\lambda(k)$")
    ax_gm.set_title(
        rf"GM linear stability: $\lambda_*={lambda_star_gm:.3f}$,  "
        rf"$\lambda_{{spec}}={lambda_spec_gm:.3f}$"
    )
    ax_gm.grid(alpha=0.3)
    ax_gm.legend(fontsize=8, loc="best")
    fig_gm_ana.suptitle("Gierer–Meinhardt 2D — linear stability analysis", fontsize=12)
    fig_gm_ana.tight_layout()
    out_gm_ana = (Path(__file__).resolve().parent / "results" / "turing_2d_analysis_GiererMeinhardt.png")
    fig_gm_ana.savefig(out_gm_ana, dpi=150, bbox_inches="tight")
    plt.close(fig_gm_ana)

    print(f"Saved {out_num}")
    print(f"Saved {out_ana}")
    print(f"Saved {out_gm_num}")
    print(f"Saved {out_gm_ana}")


if __name__ == "__main__":
    main()
