"""
Euler–Bernoulli 軸方向座屈（1D、周期境界、スペクトル半陰式 FFT）。

論文の式 (Euler 4) 相当を実装する:

  μ ∂h/∂t = -P̄(t) ∂²h/∂x² - EI ∂⁴h/∂x⁴ - k_s h
            + (ES/(2L)) (∫₀ᴸ h_x² dx) ∂²h/∂x²,
  P̄(t) = ES ΔL(t) / L.

物理的一貫した半陰式（抜け距離を明示）では、ステップ n で

  Pⁿ = (ES/L) ( ΔL(t_n) - ½ ∫₀ᴸ (h_xⁿ)² dx ),
  μ (h^{n+1} - hⁿ)/Δt = -Pⁿ h_xx^{n+1} - EI h_xxxx^{n+1} - k_s h^{n+1}.

各モード k（角波数、∂/∂x → ik）について

  ĥ^{n+1} = ĥⁿ / ( 1 + (Δt/μ) (EI k⁴ + k_s - Pⁿ k²) ).

単位: 長さ [mm], 時間 [year]。E [N/mm²], S [mm²], I [mm⁴] → EI [N·mm²]。
μ [N·year/mm²], k_s [N/mm³], ΔL(t)=ν t で ν [mm/year]。

物理・数値の既定値は論文 Table
「Parameters for buckling simulation with growth-driven axial loading」
（\\label{tab:simulation_parameters}）および M1 の order-of-magnitude 表に準拠。
I = π(r_o⁴−r_i⁴)/4, S = π(r_o²−r_i²) をコードで計算（表中の S の指数は
文献の誤記の可能性あり; 幾何式を優先）。

既定: L₀=100 mm, N=1000, t_max=60 year, Δt=0.05 year, ν=0.612 mm/year,
σ_h(0)=0.001 mm, seed=1, 出力間隔 20 step。短縮は環境変数 EULER_BUCKLING_STEPS。

注（他プロジェクトとの関係）:
  式・パラメータは本文中で参照する公開論文の表記に基づく再実装である。
  他ソフトウェアリポジトリのソースコードの転載・引用は行っていない。
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- 論文 tab:simulation_parameters / M1 representative parameters ----
R_I_MM = 1.5  # 内半径 r_i [mm]
T_W_MM = 0.1  # 壁厚 t_w [mm]
R_O_MM = R_I_MM + T_W_MM  # r_o = r_i + t_w [mm]

L_MM = 100.0  # 計算領域長 L_0 [mm]
E_YOUNG = 1.0  # E [N/mm²]
K_S = 0.001  # k_s [N/mm³]（foundation stiffness）

# 幾何から導出: I [mm⁴], S [mm²]（表中 I≈1.17, E=1 なら EI≈1.17 N·mm²）
I_SECOND = (np.pi / 4.0) * (R_O_MM**4 - R_I_MM**4)
S_AREA = np.pi * (R_O_MM**2 - R_I_MM**2)
EI = E_YOUNG * I_SECOND

MU = 0.016  # μ [N·year/mm²]（シミュレーションで最適化と記載）
NU_ELONGATION_MM_PER_YR = 0.612  # ν: ΔL(t) = ν t [mm/year]

N_GRID = 1000
DT = 0.05  # Δt [year]
T_MAX = 60.0  # t_max [year]
IC_STD = 0.001  # 初期摂動の標準偏差 [mm]（論文 0.01 より小さく、カイモグラムのコントラスト確保）
RNG_SEED = 1
RECORD_STRIDE = 20  # 出力間隔 [steps]


def _integrate_dx(y: np.ndarray, dx: float) -> float:
    """∫ y dx（周期格子の等間隔）。NumPy 1.x は trapz、2.x は trapezoid。"""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    return float(np.trapz(y, dx=dx))


def periodic_derivative(h: np.ndarray, dx: float, order: int) -> np.ndarray:
    """周期境界での空間微分（スペクトル）。"""
    n = h.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    h_hat = np.fft.fft(h)
    factor = (1j * k) ** order
    return np.real(np.fft.ifft(factor * h_hat))


def semi_implicit_step(
    h: np.ndarray,
    dx: float,
    dt: float,
    mu: float,
    ei: float,
    k_s: float,
    p_n: float,
) -> np.ndarray:
    """ĥ^{n+1} = ĥ^n / (1 + (dt/μ)(EI k^4 + k_s - P^n k^2))."""
    n = h.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    h_hat = np.fft.fft(h)
    denom = 1.0 + (dt / mu) * (ei * k**4 + k_s - p_n * k**2)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return np.real(np.fft.ifft(h_hat / denom))


def simulate(
    *,
    l_mm: float = L_MM,
    n_grid: int = N_GRID,
    dt: float = DT,
    n_steps: int | None = None,
    ei: float = EI,
    mu: float = MU,
    k_s: float = K_S,
    e_young: float = E_YOUNG,
    s_area: float = S_AREA,
    nu_mm_per_yr: float = NU_ELONGATION_MM_PER_YR,
    ic_std: float = IC_STD,
    seed: int = RNG_SEED,
    record_stride: int = RECORD_STRIDE,
):
    """
    周期格子 x ∈ [0, L)（endpoint なし）、半陰式 FFT で時間発展。
    戻り値: x, t_rec, h_hist, p_hist, p_bar_hist, times_full
    """
    if n_steps is None:
        n_steps = int(round(T_MAX / dt))
    dx = l_mm / n_grid
    x = np.linspace(0.0, l_mm, n_grid, endpoint=False)
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, ic_std, size=n_grid)

    es_over_l = e_young * s_area / l_mm
    times_full = np.arange(n_steps + 1, dtype=np.float64) * dt

    p_hist = np.zeros(n_steps + 1)
    p_bar_hist = np.zeros(n_steps + 1)
    hx = periodic_derivative(h, dx, 1)
    q = _integrate_dx(hx**2, dx)
    t0 = 0.0
    delta_l0 = nu_mm_per_yr * t0
    p_bar_hist[0] = es_over_l * delta_l0
    p_hist[0] = es_over_l * (delta_l0 - 0.5 * q)

    rows = (n_steps // record_stride) + 2
    h_hist = np.zeros((rows, n_grid))
    t_rec = np.zeros(rows)
    ri = 0
    h_hist[ri] = h.copy()
    t_rec[ri] = 0.0
    ri += 1

    for step in range(n_steps):
        t_n = step * dt
        delta_l = nu_mm_per_yr * t_n
        p_bar = es_over_l * delta_l
        hx = periodic_derivative(h, dx, 1)
        q = _integrate_dx(hx**2, dx)
        p_n = es_over_l * (delta_l - 0.5 * q)
        h = semi_implicit_step(h, dx, dt, mu, ei, k_s, p_n)
        t_np1 = (step + 1) * dt
        delta_l_np1 = nu_mm_per_yr * t_np1
        p_bar_hist[step + 1] = es_over_l * delta_l_np1
        hx_np1 = periodic_derivative(h, dx, 1)
        q_np1 = _integrate_dx(hx_np1**2, dx)
        p_hist[step + 1] = es_over_l * (delta_l_np1 - 0.5 * q_np1)
        if (step + 1) % record_stride == 0 or step + 1 == n_steps:
            if ri < rows:
                h_hist[ri] = h.copy()
                t_rec[ri] = t_np1
                ri += 1

    return x, t_rec[:ri], h_hist[:ri], p_hist, p_bar_hist, times_full


def main() -> None:
    out = Path(__file__).with_suffix(".png")
    env_steps = os.environ.get("EULER_BUCKLING_STEPS")
    n_steps = int(env_steps) if env_steps else None

    x, t_rec, h_hist, p_hist, p_bar_hist, times_full = simulate(n_steps=n_steps)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2.2, 1]}
    )
    extent = [0.0, L_MM, t_rec[-1], t_rec[0]]
    im = ax0.imshow(
        h_hist,
        aspect="auto",
        extent=extent,
        interpolation="bilinear",
        origin="lower",
    )
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("t [year]")
    ax0.set_title(
        "Euler (4) buckling — periodic BC, semi-implicit FFT "
        r"($P^n$ explicit, $h^{n+1}$ implicit)"
    )
    fig.colorbar(im, ax=ax0, label="h", fraction=0.046, pad=0.04)

    ax1.plot(times_full, p_hist, label=r"$P(t)$ (effective)")
    ax1.plot(times_full, p_bar_hist, "--", label=r"$\bar P(t)=ES\Delta L/L$")
    ax1.set_xlabel("t [year]")
    ax1.set_ylabel("load scale [N]")
    ax1.legend(loc="best", fontsize=9)
    ax1.set_title(r"$P$ from $\Delta L(t) - \frac{1}{2}\int h_x^2\,dx$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
