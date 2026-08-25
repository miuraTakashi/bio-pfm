"""
Ornstein–Uhlenbeck (OU) 過程 — 数理計算と統計的性質の可視化。

dX_t = θ(μ - X_t) dt + σ dW_t

解析結果:
  - 平均:     E[X_t] = μ + (x_0 - μ) e^{-θt}
  - 分散:     Var(X_t) = (σ²/2θ)(1 - e^{-2θt})
  - 定常分布: N(μ, σ²/2θ)
  - 自己相関: ρ(τ) = e^{-θ|τ|}  (定常状態)
  - パワースペクトル (定常): S(ω) = σ²/(θ² + ω²)  [Lorentz 型]

数値シミュレーションは OU のガウス遷移に従う exact 更新（Euler–Maruyama より正確）。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def ou_mean(t: np.ndarray, x0: float, mu: float, theta: float) -> np.ndarray:
    """E[X_t] = μ + (x_0 - μ) exp(-θ t)."""
    return mu + (x0 - mu) * np.exp(-theta * t)


def ou_variance(t: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    """Var(X_t) = (σ²/2θ)(1 - exp(-2θ t), 初期値 x_0 確定的)."""
    return (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t))


def ou_std(t: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    return np.sqrt(ou_variance(t, sigma, theta))


def ou_stationary_variance(sigma: float, theta: float) -> float:
    """定常分散 σ²/(2θ)."""
    return sigma**2 / (2.0 * theta)


def ou_autocorrelation(tau: np.ndarray, theta: float) -> np.ndarray:
    """定常 OU 過程の自己相関 ρ(τ) = exp(-θ|τ|)."""
    return np.exp(-theta * np.abs(tau))


def ou_autocovariance(tau: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    """定常 OU 過程の自己共分散 R(τ) = (σ²/2θ) exp(-θ|τ|)."""
    return ou_stationary_variance(sigma, theta) * np.exp(-theta * np.abs(tau))


def ou_power_spectral_density(omega: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    """定常 OU の両側角周波数パワースペクトル密度 S(ω) = σ²/(θ² + ω²)."""
    return sigma**2 / (theta**2 + omega**2)


def ou_power_spectral_density_hz(freq: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    """片側 Hz 表記 P(f) = 2σ²/(θ² + (2πf)²). ∫_0^∞ P(f) df = Var(X)."""
    omega = 2.0 * np.pi * freq
    return 2.0 * sigma**2 / (theta**2 + omega**2)


def ou_transition_step(
    x: np.ndarray, mu: float, theta: float, sigma: float, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """1 ステップの exact ガウス更新: X_{t+dt} | X_t."""
    decay = np.exp(-theta * dt)
    mean = mu + (x - mu) * decay
    var = (sigma**2 / (2.0 * theta)) * (1.0 - decay**2)
    return mean + rng.normal(0.0, np.sqrt(var), size=x.shape)


def simulate_ou(
    x0: float,
    mu: float,
    theta: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """OU 過程のサンプルパスを生成。戻り値 shape = (n_steps, n_paths)."""
    paths = np.empty((n_steps, n_paths), dtype=float)
    paths[0] = x0
    for n in range(n_steps - 1):
        paths[n + 1] = ou_transition_step(paths[n], mu, theta, sigma, dt, rng)
    return paths


def empirical_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """1 本の時系列から自己相関（lag 0 … max_lag）を推定。"""
    x = series - np.mean(series)
    n = len(x)
    acf = np.empty(max_lag + 1, dtype=float)
    denom = np.dot(x, x)
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / denom if denom > 0 else np.nan
    return acf


def main():
    # --- パラメータ ---
    mu = 0.0
    theta = 1.0
    sigma = 1.0
    x0 = 2.0
    t_end = 10.0
    dt = 0.01
    n_steps = int(round(t_end / dt)) + 1
    n_paths = 200
    burn_in_steps = int(round(5.0 / dt))  # 定常分布用に十分な burn-in
    rng = np.random.default_rng(42)

    t = np.arange(n_steps) * dt
    paths = simulate_ou(x0, mu, theta, sigma, dt, n_steps, n_paths, rng)

    mean_theory = ou_mean(t, x0, mu, theta)
    var_theory = ou_variance(t, sigma, theta)
    std_theory = np.sqrt(var_theory)
    var_stationary = ou_stationary_variance(sigma, theta)

    # --- 図 1: 統計的性質の概要 ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) サンプルパスと理論平均 ± 1σ
    ax = axes[0, 0]
    n_show = min(30, n_paths)
    ax.plot(t, paths[:, :n_show], color="C0", alpha=0.25, lw=0.8)
    ax.plot(t, mean_theory, color="C3", lw=2, label=r"$\mathbb{E}[X_t]$")
    ax.fill_between(
        t,
        mean_theory - std_theory,
        mean_theory + std_theory,
        color="C3",
        alpha=0.2,
        label=r"$\mathbb{E}[X_t] \pm \mathrm{Var}(X_t)^{1/2}$",
    )
    ax.axhline(mu, color="k", ls="--", lw=1, alpha=0.6, label=r"$\mu$ (stationary mean)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$X_t$")
    ax.set_title("(a) Sample paths & theoretical mean ± std")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) 経験分散 vs 理論分散
    ax = axes[0, 1]
    empirical_var = np.var(paths, axis=1, ddof=1)
    ax.plot(t, empirical_var, color="C0", alpha=0.85, label="empirical Var")
    ax.plot(t, var_theory, color="C3", ls="--", lw=2, label=r"$(\sigma^2/2\theta)(1-e^{-2\theta t})$")
    ax.axhline(var_stationary, color="k", ls=":", lw=1.5, label=r"$\sigma^2/2\theta$ (stationary)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathrm{Var}(X_t)$")
    ax.set_title(f"(b) Variance ({n_paths} paths)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) 自己相関（定常区間の平均）
    ax = axes[1, 0]
    max_lag = min(200, n_steps // 4)
    acf_emp = np.mean(
        [empirical_autocorrelation(paths[burn_in_steps:, p], max_lag) for p in range(n_paths)],
        axis=0,
    )
    lags = np.arange(max_lag + 1) * dt
    acf_theory = ou_autocorrelation(lags, theta)
    ax.plot(lags, acf_emp, "o", ms=3, color="C0", alpha=0.7, label="empirical (burn-in avg)")
    ax.plot(lags, acf_theory, color="C3", lw=2, label=r"$e^{-\theta|\tau|}$")
    ax.set_xlabel(r"lag $\tau$")
    ax.set_ylabel(r"$\rho(\tau)$")
    ax.set_title("(c) Autocorrelation (stationary regime)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) 定常分布 vs 理論ガウス N(μ, σ²/2θ)
    ax = axes[1, 1]
    stationary_samples = paths[burn_in_steps:, :].ravel()
    bins = np.linspace(mu - 3.5 * np.sqrt(var_stationary), mu + 3.5 * np.sqrt(var_stationary), 40)
    ax.hist(stationary_samples, bins=bins, density=True, color="C0", alpha=0.55, label="empirical")
    x_grid = np.linspace(bins[0], bins[-1], 300)
    pdf = np.exp(-0.5 * ((x_grid - mu) ** 2) / var_stationary) / np.sqrt(2.0 * np.pi * var_stationary)
    ax.plot(x_grid, pdf, color="C3", lw=2, label=r"$\mathcal{N}(\mu,\,\sigma^2/2\theta)$")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel("density")
    ax.set_title("(d) Stationary distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    param_text = (
        rf"$\theta={theta}$, $\mu={mu}$, $\sigma={sigma}$, $x_0={x0}$, "
        rf"$\Delta t={dt}$, {n_paths} paths"
    )
    fig.suptitle("Ornstein–Uhlenbeck process: statistical properties\n" + param_text, fontsize=11)
    plt.tight_layout()
    out_summary = (Path(__file__).resolve().parent / "results" / "ornstein_uhlenbeck_process.png")
    plt.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close()

    # --- 図 2: 遷移密度（固定時刻でのヒストグラム vs 理論正規分布）---
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.5))
    check_times = [0.5, 2.0, 10.0]
    for ax, t_check in zip(axes2, check_times):
        idx = int(round(t_check / dt))
        samples = paths[idx, :]
        m = ou_mean(np.array([t_check]), x0, mu, theta)[0]
        v = ou_variance(np.array([t_check]), sigma, theta)[0]
        bins = np.linspace(m - 3.5 * np.sqrt(v), m + 3.5 * np.sqrt(v), 35)
        ax.hist(samples, bins=bins, density=True, color="C0", alpha=0.55, label="empirical")
        xg = np.linspace(bins[0], bins[-1], 300)
        pdf_t = np.exp(-0.5 * ((xg - m) ** 2) / v) / np.sqrt(2.0 * np.pi * v)
        ax.plot(xg, pdf_t, color="C3", lw=2, label="theory")
        ax.axvline(m, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel(r"$X$")
        ax.set_ylabel("density")
        ax.set_title(rf"$t={t_check}$: $\mathrm{{Var}}={v:.3f}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Marginal distribution at fixed times (OU is Gaussian at each t)")
    plt.tight_layout()
    out_marginal = (Path(__file__).resolve().parent / "results" / "ornstein_uhlenbeck_marginals.png")
    plt.savefig(out_marginal, dpi=150, bbox_inches="tight")
    plt.close()

    # --- 図 3: パワースペクトル（定常区間・Welch 平均 vs 理論 Lorentz 型）---
    psd_t_end = 200.0
    psd_n_steps = int(round(psd_t_end / dt)) + 1
    psd_burn_in = int(round(20.0 / dt))
    psd_paths = simulate_ou(mu, mu, theta, sigma, dt, psd_n_steps, n_paths, rng)

    nperseg = 1024
    psd_accum = None
    freqs = None
    for p in range(n_paths):
        segment = psd_paths[psd_burn_in:, p] - mu
        freqs_p, psd_p = welch(segment, fs=1.0 / dt, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
        if psd_accum is None:
            freqs = freqs_p
            psd_accum = np.zeros_like(psd_p)
        psd_accum += psd_p
    psd_emp = psd_accum / n_paths
    psd_theory_hz = ou_power_spectral_density_hz(freqs, sigma, theta)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes3[0]
    ax.loglog(freqs[1:], psd_emp[1:], color="C0", alpha=0.85, label="empirical (Welch avg)")
    ax.loglog(freqs[1:], psd_theory_hz[1:], color="C3", lw=2, ls="--", label=r"$2\sigma^2/(\theta^2+(2\pi f)^2)$")
    ax.axvline(theta / (2.0 * np.pi), color="k", ls=":", lw=1.2, alpha=0.7, label=r"$f_c=\theta/2\pi$")
    ax.set_xlabel("frequency $f$ (Hz)")
    ax.set_ylabel(r"$S(f)$ (one-sided)")
    ax.set_title("(a) Power spectral density (log–log)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes3[1]
    omega_plot = np.linspace(0.0, 5.0 * theta, 400)
    psd_omega = ou_power_spectral_density(omega_plot, sigma, theta)
    ax.plot(omega_plot, psd_omega, color="C3", lw=2, label=r"$S(\omega)=\sigma^2/(\theta^2+\omega^2)$")
    ax.axhline(sigma**2 / theta**2, color="k", ls=":", lw=1, alpha=0.6, label=r"$S(0)=\sigma^2/\theta^2$")
    ax.axvline(theta, color="k", ls="--", lw=1, alpha=0.6, label=r"$\omega_c=\theta$")
    ax.set_xlabel(r"angular frequency $\omega$")
    ax.set_ylabel(r"$S(\omega)$")
    ax.set_title("(b) Lorentzian PSD (linear scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig3.suptitle(
        "Ornstein–Uhlenbeck process: power spectrum (stationary regime after burn-in)",
        fontsize=11,
    )
    plt.tight_layout()
    out_psd = (Path(__file__).resolve().parent / "results" / "ornstein_uhlenbeck_psd.png")
    plt.savefig(out_psd, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_summary}")
    print(f"Saved {out_marginal}")
    print(f"Saved {out_psd}")
    print()
    print("Theoretical summary:")
    print(f"  E[X_t]     = mu + (x0 - mu) exp(-theta t)")
    print(f"  Var(X_t)   = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))")
    print(f"  stationary = N(mu, sigma^2/(2*theta)) = N({mu}, {var_stationary:.4f})")
    print(f"  rho(tau)   = exp(-theta |tau|)")
    print(f"  S(omega)   = sigma^2 / (theta^2 + omega^2)  [Lorentzian]")


if __name__ == "__main__":
    main()
