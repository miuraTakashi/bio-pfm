"""
KPZ 方程式（Kardar-Parisi-Zhang）— 非線形確率的表面成長
（Mathematica/1_continuous_scheme/1_5_stochastic_term/kpz/KPZのコピー.nb から移植）。

方程式:
  ∂h/∂t = ν * ∂²h/∂x² + (λ/2) * (∂h/∂x)² + η(x,t)

パラメータ: dx=0.01, ν=du=0.1, dt=dx²/ν/4, noiseAmp=0.000002, λ=2, domain=10
周期境界条件。分散（Variance）の時間発展で KPZ スケーリング則を確認する。

出力:
  kpz.png — カイモグラフ・時刻別プロファイル・平均高さの進行、分散・スケーリング
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap


def simulate_kpz(
    domain_size: float = 10.0,
    D: float = 0.1,
    lam: float = 2.0,
    noise_amp: float = 0.000002,
    n_steps: int = 10000,
    n_saves: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    KPZ 方程式シミュレーション。

    Returns
    -------
    times, variances, h_final, history, dx, L
    history : (n_records, n) 記録された h(x)（t=0 含む）
    """
    rng = np.random.default_rng(seed)
    dx = 0.01
    dt = dx**2 / D / 4.0
    n = int(round(domain_size / dx))
    L = n * dx

    h = np.zeros(n)
    variances: list[float] = [0.0]
    history: list[np.ndarray] = [h.copy()]
    times: list[float] = [0.0]
    save_every = max(1, n_steps // n_saves)

    for step in range(n_steps):
        lap = (np.roll(h, 1) + np.roll(h, -1) - 2.0 * h) / (dx * dx)
        grad = (np.roll(h, -1) - np.roll(h, 1)) / (2.0 * dx)
        nonlinear = (lam / 2.0) * grad**2
        eta = noise_amp / np.sqrt(dt) * rng.uniform(-1, 1, n)
        h = h + dt * (D * lap + nonlinear + eta)

        if (step + 1) % save_every == 0:
            variances.append(float(np.var(h)))
            times.append((step + 1) * dt)
            history.append(h.copy())

    return (
        np.array(times, dtype=np.float64),
        np.array(variances, dtype=np.float64),
        h,
        np.array(history, dtype=np.float64),
        dx,
        L,
    )


def _mean_growth_velocity(times: np.ndarray, mean_h: np.ndarray) -> float:
    """⟨h⟩(t) の後半窓での線形フィット傾き（界面の平均進行速度）。"""
    mask = times > 0.0
    if np.sum(mask) < 4:
        return float("nan")
    t = times[mask]
    h = mean_h[mask]
    i0 = len(t) // 3
    slope, _ = np.polyfit(t[i0:], h[i0:], 1)
    return float(slope)


def main() -> None:
    quick = os.environ.get("KPZ_QUICK", "").strip().lower() in ("1", "true", "yes")
    n_steps = 3000 if quick else 10000
    n_saves = 30 if quick else 50

    times, variances, h_final, history, dx, L = simulate_kpz(n_steps=n_steps, n_saves=n_saves)
    x = np.arange(history.shape[1], dtype=np.float64) * dx
    t_max = float(times[-1]) if times.size else 0.0
    mean_h = history.mean(axis=1)
    v_mean = _mean_growth_velocity(times, mean_h)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)

    # (0,0) Space–time kymograph
    ax_k = axes[0, 0]
    extent = [0.0, L, 0.0, t_max]
    im = ap.atlas_imshow(
        ax_k,
        history,
        heatmap="scalar",
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    ax_k.set_xlabel("x")
    ax_k.set_ylabel("t")
    ax_k.set_title(r"$h(x,t)$ kymograph (interface evolution)")
    fig.colorbar(im, ax=ax_k, fraction=0.046, pad=0.02, label="h")

    # (0,1) Profiles at selected times (growth / roughening)
    ax_prof = axes[0, 1]
    n_hist = history.shape[0]
    idxs = np.unique(np.linspace(0, n_hist - 1, num=5, dtype=int))
    colors = ap.atlas_line_colors(len(idxs), heatmap="scalar")
    for idx, color in zip(idxs, colors):
        tk = times[idx] if idx < len(times) else 0.0
        h_line = history[idx] - history[0].mean()
        ax_prof.plot(x, h_line, color=color, lw=1.0, label=f"t={tk:.2f}")
    ax_prof.set_xlabel("x")
    ax_prof.set_ylabel(r"$h - \langle h\rangle_{t=0}$")
    ax_prof.set_title("Surface profiles (mean-subtracted from $t=0$)")
    ax_prof.legend(fontsize=7, loc="upper left")
    ax_prof.grid(alpha=0.3)

    # (0,2) Mean height vs time — upward propagation of the interface
    ax_mean = axes[0, 2]
    ax_mean.plot(times, mean_h, "C0-o", ms=3, lw=1.2, label=r"$\langle h\rangle(t)$")
    if np.isfinite(v_mean):
        h_fit = mean_h[0] + v_mean * times
        ax_mean.plot(times, h_fit, "C3--", lw=1.2, label=rf"fit: $v={v_mean:.4g}$")
    ax_mean.set_xlabel("t")
    ax_mean.set_ylabel(r"$\langle h\rangle$")
    ax_mean.set_title("Mean interface height (propagation)")
    ax_mean.legend(fontsize=8)
    ax_mean.grid(alpha=0.3)

    # (1,0) Final profile
    axes[1, 0].plot(x, h_final, color="steelblue", linewidth=0.8)
    axes[1, 0].set_title("Final surface profile $h(x)$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("h")
    axes[1, 0].grid(alpha=0.3)

    # (1,1) Variance
    mask = times > 0
    axes[1, 1].semilogy(times[mask], variances[mask], "o-", color="steelblue", markersize=3)
    axes[1, 1].set_title(r"Variance $\mathrm{Var}(h)$")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("Var(h)")
    axes[1, 1].grid(alpha=0.3)

    # (1,2) log-log scaling
    ax_log = axes[1, 2]
    if np.any(variances[mask] > 0):
        ax_log.loglog(times[mask], variances[mask], "o-", color="tomato", markersize=3)
        t_fit = times[mask]
        v_fit = variances[mask]
        if len(t_fit) > 5:
            log_t = np.log(t_fit[t_fit > 0])
            log_v = np.log(v_fit[t_fit > 0])
            slope, _ = np.polyfit(log_t[-len(log_t) // 2 :], log_v[-len(log_v) // 2 :], 1)
            ax_log.set_title(rf"log-log: slope $\approx {slope:.3f}$ (KPZ: $2/3$)")
        else:
            ax_log.set_title("log-log scaling")
        ax_log.set_xlabel("t")
        ax_log.set_ylabel("Var(h)")
        ax_log.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        rf"KPZ equation ($\nu=0.1$, $\lambda=2$, noiseAmp$=2\times10^{{-6}}$, $L={L:g}$)",
        fontsize=12,
    )
    out_path = (Path(__file__).resolve().parent / "results" / "kpz.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
