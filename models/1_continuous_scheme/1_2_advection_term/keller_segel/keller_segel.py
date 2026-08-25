"""
修正 Keller–Segel（1D）— 血管網形成と走化性
移植元: Mathematica/1_continuous_scheme/1_2_advection_term/keller_segel/血管網形成と走化性：modifiedKellerSegel.nb
「数値計算」小節（In[1]–In[11]）と同じ離散式。

支配方程式（連続極限のイメージ）:
  u_t = f(u,v) + du u_xx + c0 * ∂x( χ(u) ∂x v ),
      χ(u) = u (us - u)   （界面通量に u の平均を使用した離散版）
  v_t = g(u,v) + dv v_xx,
      f = fu*u + fv*v = 0,   g = gu*u + gv*v = u - v

離散（周期境界, RotateLeft = インデックス +1 側へシフト = np.roll(..., -1)）:
  Lap(u)_j = (u_{j+1} + u_{j-1} - 2u_j) / dx^2
  走化項: c0/dx * ( F_{j+1/2} - F_{j-1/2} ),
    F_{j+1/2} = ((u_j+u_{j+1})/2) * (us - (u_j+u_{j+1})/2) * (v_{j+1}-v_j)/dx
    F_{j-1/2} = ((u_j+u_{j-1})/2) * (us - (u_j+u_{j-1})/2) * (v_j - v_{j-1})/dx

パラメータ（ノートブック In[1], In[2]）:
  noiseAmp=0.01, dx=0.1, domainSize=6.3, du=0.05, dv=1, c0=-1, u0=1, us=2, dt=0.001,
  fu=fv=0, gu=1, gv=-1。
初期: u, v は各格子で u0 + Uniform[0,1]*noiseAmp（ノートブックと同型）。
時間積分: 陽的オイラー（NestList と同じ）, 20000 ステップ。
"""
import numpy as np
import matplotlib.pyplot as plt
# --- atlas heatmap helpers ---
import sys
from pathlib import Path as _Path

for _d in _Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap

from pathlib import Path


def dispersion_matrix(
    k: float,
    du: float,
    dv: float,
    c0: float,
    u0: float,
    us: float,
) -> np.ndarray:
    """
    u,v の線形化に対する 2×2 行列 A(k)。

    連続極限:
      u_t = du u_xx + c0 ∂x(χ(u0) v_x),  χ(u)=u(us-u)
      v_t = (u - v) + dv v_xx
    Fourier モード ~ exp(ikx) に対し u_xx, v_xx → -k^2。
    """
    chi0 = u0 * (us - u0)
    a11 = -du * k * k
    a12 = c0 * chi0 * (-k * k)
    a21 = 1.0
    a22 = -dv * k * k - 1.0
    return np.array([[a11, a12], [a21, a22]], dtype=float)


def dispersion_max_growth_curve(
    du: float,
    dv: float,
    c0: float,
    u0: float,
    us: float,
    dx: float,
    n_samples: int = 400,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    max Re σ(k) を k∈[0, k_Nyq]（k_Nyq=π/dx）上で評価。
    戻り値: k_grid, sigma_max_re, k_star, lambda_star
    """
    k_nyq = np.pi / dx
    k_max = min(10.0, float(k_nyq))
    k_grid = np.linspace(0.0, k_max, n_samples)
    sigma_max = np.empty_like(k_grid)
    for i, k in enumerate(k_grid):
        evals = np.linalg.eigvals(dispersion_matrix(float(k), du, dv, c0, u0, us))
        sigma_max[i] = float(np.max(np.real(evals)))
    idx_star = int(np.argmax(sigma_max))
    k_star = float(k_grid[idx_star])
    lambda_star = float((2.0 * np.pi / k_star) if k_star > 1e-12 else np.inf)
    return k_grid, sigma_max, k_star, lambda_star

def laplacian_1d_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) / (dx * dx)

def chemotaxis_term(u: np.ndarray, v: np.ndarray, dx: float, c0: float, us: float) -> np.ndarray:
    """ノートブック oneStep の c0 * (...)/dx に相当する項（あとで du 項と足す）。"""
    u_r = 0.5 * (u + np.roll(u, -1))
    chi_r = u_r * (us - u_r)
    flux_r = chi_r * (np.roll(v, -1) - v) / dx

    u_l = 0.5 * (u + np.roll(u, 1))
    chi_l = u_l * (us - u_l)
    flux_l = chi_l * (v - np.roll(v, 1)) / dx

    return c0 * (flux_r - flux_l) / dx


def simulate_modified_keller_segel_1d(
    domain_size: float = 6.3,
    dx: float = 0.1,
    du: float = 0.05,
    dv: float = 1.0,
    c0: float = -1.0,
    u0: float = 1.0,
    us: float = 2.0,
    dt: float = 0.001,
    n_steps: int = 20000,
    noise_amp: float = 0.01,
    record_every: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(domain_size / dx))
    x = (np.arange(n, dtype=float) + 0.5) * dx

    u = u0 + rng.random(n) * noise_amp
    v = u0 + rng.random(n) * noise_amp

    traj_u = [u.copy()]
    times = [0.0]

    for step in range(1, n_steps + 1):
        lap_u = laplacian_1d_periodic(u, dx)
        lap_v = laplacian_1d_periodic(v, dx)
        chemo = chemotaxis_term(u, v, dx, c0, us)
        du_dt = du * lap_u + chemo
        dv_dt = (u - v) + dv * lap_v
        u = u + dt * du_dt
        v = v + dt * dv_dt
        if step % record_every == 0:
            traj_u.append(u.copy())
            times.append(step * dt)

    return x, np.array(times), np.array(traj_u), v


def main():
    # シミュレーションと既存プロット
    x, times, traj_u, v_final = simulate_modified_keller_segel_1d()
    t_max = float(times[-1])
    dx = float(x[1] - x[0])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1.2, 1.0])

    im = ap.atlas_imshow(
        axes[0],
        traj_u,
        heatmap="scalar",
        origin="upper",
        aspect="auto",
        extent=[float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), t_max, 0.0],
        interpolation="bilinear",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(r"$u(x,t)$ — modified Keller–Segel (1D, notebook parameters)")
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02, label="u")

    axes[1].plot(x, traj_u[-1], "b-", lw=1.2, label="u (final)")
    axes[1].plot(x, v_final, "r-", lw=1.0, alpha=0.85, label="v (final)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density / signal")
    axes[1].set_title("Final profiles (aggregation / pattern from chemotaxis)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Modified Keller–Segel 1D — vascular chemotaxis (from modifiedKellerSegel.nb)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "results" / "keller_segel.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}  u range final: [{traj_u[-1].min():.3f}, {traj_u[-1].max():.3f}]")

    # --- 線形安定性解析（分散関係） ---
    du = 0.05
    dv = 1.0
    c0 = -1.0
    u0 = 1.0
    us = 2.0

    k_grid, sigma_max, k_star, lambda_star = dispersion_max_growth_curve(
        du=du,
        dv=dv,
        c0=c0,
        u0=u0,
        us=us,
        dx=dx,
    )

    fig2, ax = plt.subplots(1, 1, figsize=(7.0, 4.4))
    ax.plot(k_grid, sigma_max, color="C2", lw=1.5, label=r"max Re $\sigma(k)$")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    if np.isfinite(lambda_star):
        ax.axvline(k_star, color="C3", lw=1.1, ls="--", label=rf"$k_*={k_star:.3f}$")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel(r"max Re $\sigma(k)$")
    ax.set_title("Linear stability (dispersion relation at homogeneous state)")
    ax.grid(alpha=0.3)
    txt = (
        rf"$u_0={u0:.2f}, v_0={u0:.2f}$" + "\n"
        rf"$\lambda_* = 2\pi/k_* \approx {lambda_star:.3f}$"
        if np.isfinite(lambda_star)
        else rf"$u_0={u0:.2f}, v_0={u0:.2f}$"
    )
    ax.text(
        0.03,
        0.15,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    ax.legend(fontsize=8, loc="best")
    fig2.tight_layout()
    out_disp = (Path(__file__).resolve().parent / "results" / "keller_segel_dispersion.png")
    fig2.savefig(out_disp, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out_disp}")


if __name__ == "__main__":
    main()
