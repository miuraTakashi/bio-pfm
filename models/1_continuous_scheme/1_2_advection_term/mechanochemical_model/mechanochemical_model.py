"""
メカノケミカル（機械的）1D モデル — ノートブック由来の輸送 + Poisson 連成。

他の atlas スクリプトと同様に、既定の図表は `Path(__file__).with_suffix(".png")`
および同じベース名の補助 PNG に保存する。
"""
from pathlib import Path

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

import numpy as np

_SCRIPT = Path(__file__)
# メイン図: 時空間 + 最終プロファイル
_OUT_MAIN_PNG = _SCRIPT.with_suffix(".png")


def _artifact(suffix: str) -> Path:
    """例: suffix='_linear_compare.png' → '<stem>_linear_compare.png'"""
    return _SCRIPT.parent / f"{_SCRIPT.stem}{suffix}"


def first_derivative_central_periodic(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * dx)


def first_derivative_spectral(values: np.ndarray, dx: float) -> np.ndarray:
    n = values.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    v_hat = np.fft.fft(values)
    d_hat = 1j * k * v_hat
    return np.fft.ifft(d_hat).real


def dispersion_sigma(k: np.ndarray, s: float, tau: float, mu: float, gamma: float) -> np.ndarray:
    """
    Growth rate from notebook linear analysis:
      sigma(k) = (-k^2 - s + 2*k^2*tau - k^4*gamma*tau) / (k^2*mu)
    """
    k2 = k * k
    return (-k2 - s + 2.0 * k2 * tau - (k2 * k2) * gamma * tau) / (k2 * mu)


def first_derivative_upwind(values: np.ndarray, vel: np.ndarray, dx: float) -> np.ndarray:
    """
    First derivative using local upwind direction.
    If vel[i] >= 0 -> backward difference.
    If vel[i] < 0  -> forward difference.
    """
    backward = (values - np.roll(values, 1)) / dx
    forward = (np.roll(values, -1) - values) / dx
    return np.where(vel >= 0.0, backward, forward)


def second_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1) - 2.0 * values + np.roll(values, 1)) / (dx * dx)


def solve_poisson_periodic(rhs: np.ndarray, dx: float) -> np.ndarray:
    """
    Solve phi_xx = rhs with periodic boundary conditions.
    The mean mode is set to zero (gauge fixing).
    """
    n = rhs.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    rhs_hat = np.fft.fft(rhs - np.mean(rhs))
    denom = -(k * k)
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    mask = np.abs(denom) > 0.0
    phi_hat[mask] = rhs_hat[mask] / denom[mask]
    phi_hat[0] = 0.0
    return np.fft.ifft(phi_hat).real


def upwind_step(q: np.ndarray, vel: np.ndarray, dt: float, dx: float) -> np.ndarray:
    # Conservative flux form: q_t + (q*vel)_x = 0
    q_r = np.roll(q, -1)
    vel_face = 0.5 * (vel + np.roll(vel, -1))
    flux = np.where(vel_face >= 0.0, vel_face * q, vel_face * q_r)
    q_new = q - (dt / dx) * (flux - np.roll(flux, 1))
    return q_new


def run_simulation(
    L=8.0 * np.pi,
    T=12.0,
    nx=256,
    dt=1.0e-3,
    s=1.0,
    tau=3.0,
    mu=1.0,
    gamma=1.0,
    noise_amp=2.0e-5,
    seed=0,
    cfl=0.3,
    symmetric_init=True,
    stress_derivative_method="spectral",
    initial_condition="unbiased_noise",
):
    """
    Extracted model from notebook:
      n_t + (n u_t)_x = 0
      rho_t + (rho u_t)_x = 0
      mu u_xxt + u_xx + tau (n rho + gamma rho_xx)_x - s rho u = 0

    Periodic boundary condition for all fields.
    """
    x = np.linspace(0.0, L, nx, endpoint=False)
    dx = x[1] - x[0]
    rng = np.random.default_rng(seed)
    phase = 2.0 * np.pi * x / L
    if initial_condition == "unbiased_noise":
        noise_n = rng.uniform(-0.5, 0.5, size=nx)
        noise_rho = rng.uniform(-0.5, 0.5, size=nx)
        if symmetric_init:
            noise_n = 0.5 * (noise_n + noise_n[::-1])
            noise_rho = 0.5 * (noise_rho + noise_rho[::-1])
        n = 1.0 + noise_amp * noise_n
        rho = 1.0 + noise_amp * noise_rho
    elif initial_condition == "biased_34":
        if symmetric_init:
            mode_3_4 = 0.6 * np.cos(3.0 * phase) + 0.4 * np.cos(4.0 * phase)
            noise = rng.uniform(-0.5, 0.5, size=nx)
            noise = 0.5 * (noise + noise[::-1])
            n = 1.0 + noise_amp * (mode_3_4 + 0.2 * noise)
            rho = 1.0 + noise_amp * (
                0.8 * mode_3_4 + 0.2 * np.cos(2.0 * phase) + 0.2 * noise
            )
        else:
            mode_3_4 = 0.6 * np.sin(3.0 * phase) + 0.4 * np.sin(4.0 * phase + 0.35)
            n = 1.0 + noise_amp * (
                mode_3_4 + 0.2 * rng.uniform(-0.5, 0.5, size=nx)
            )
            rho = 1.0 + noise_amp * (
                0.7 * mode_3_4
                + 0.3 * np.cos(3.0 * phase + 0.2)
                + 0.2 * rng.uniform(-0.5, 0.5, size=nx)
            )
    else:
        raise ValueError("initial_condition must be 'unbiased_noise' or 'biased_34'")
    u = np.zeros(nx)

    n_hist = [n.copy()]
    rho_hist = [rho.copy()]
    u_hist = [u.copy()]
    t_hist = [0.0]
    current_time = 0.0

    target_frames = 200
    save_interval = T / target_frames
    next_save_time = save_interval
    step = 0
    while current_time < T:
        step += 1
        rho_xx = second_derivative(rho, dx)
        stress = n * rho + gamma * rho_xx
        if stress_derivative_method == "spectral":
            stress_x = first_derivative_spectral(stress, dx)
        elif stress_derivative_method == "central":
            stress_x = first_derivative_central_periodic(stress, dx)
        else:
            raise ValueError("stress_derivative_method must be 'spectral' or 'central'")
        u_xx = second_derivative(u, dx)

        rhs = (-u_xx - tau * stress_x + s * rho * u) / mu
        u_t = solve_poisson_periodic(rhs, dx)

        umax = np.max(np.abs(u_t))
        dt_eff = min(dt, cfl * dx / (umax + 1.0e-12), T - current_time)

        n = upwind_step(n, u_t, dt_eff, dx)
        rho = upwind_step(rho, u_t, dt_eff, dx)
        u = u + dt_eff * u_t
        n = np.maximum(n, 1.0e-8)
        rho = np.maximum(rho, 1.0e-8)
        current_time += dt_eff

        if current_time >= next_save_time or np.isclose(current_time, T):
            n_hist.append(n.copy())
            rho_hist.append(rho.copy())
            u_hist.append(u.copy())
            t_hist.append(current_time)
            next_save_time += save_interval

    return x, np.array(t_hist), np.array(n_hist), np.array(rho_hist), np.array(u_hist)


def linear_regime_compare(
    L=8.0 * np.pi,
    T=0.8,
    nx=256,
    dt=2.0e-4,
    s=1.0,
    tau=3.0,
    mu=1.0,
    gamma=1.0,
    noise_amp=2.0e-8,
    seed=0,
    max_mode=8,
    stress_derivative_method="spectral",
    initial_condition="unbiased_noise",
    output_dir: Path | None = None,
    save_outputs: bool = True,
):
    """
    Run a short-time simulation and compare numerical growth rates
    with analytical dispersion relation in the linear regime.
    """
    use_dir = output_dir is not None
    out = output_dir if use_dir else _SCRIPT.parent
    if save_outputs and use_dir:
        out.mkdir(parents=True, exist_ok=True)

    x, t, n_hist, rho_hist, u_hist = run_simulation(
        L=L,
        T=T,
        nx=nx,
        dt=dt,
        s=s,
        tau=tau,
        mu=mu,
        gamma=gamma,
        noise_amp=noise_amp,
        seed=seed,
        cfl=0.2,
        symmetric_init=False,
        stress_derivative_method=stress_derivative_method,
        initial_condition=initial_condition,
    )

    dx = x[1] - x[0]
    n0 = n_hist - np.mean(n_hist, axis=1, keepdims=True)
    n_hat = np.fft.rfft(n0, axis=1)
    amp = np.abs(n_hat)

    mode_ids = np.arange(1, min(max_mode, amp.shape[1] - 1) + 1)
    k = 2.0 * np.pi * mode_ids / L
    sigma_ana = dispersion_sigma(k, s=s, tau=tau, mu=mu, gamma=gamma)

    sigma_num = []
    eps = 1.0e-30
    for m in mode_ids:
        y = np.log(amp[:, m] + eps)
        p = np.polyfit(t, y, 1)
        sigma_num.append(p[0])
    sigma_num = np.array(sigma_num)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(mode_ids, sigma_ana, "o-", label="analysis")
    axes[0].plot(mode_ids, sigma_num, "s--", label="numerical fit")
    axes[0].set_xlabel("mode number m")
    axes[0].set_ylabel("growth rate sigma")
    axes[0].set_title("Linear growth rate comparison")
    axes[0].legend()

    for m in mode_ids[:4]:
        a0 = max(amp[0, m], eps)
        axes[1].plot(t, np.log(amp[:, m] + eps), label=f"log|A_{m}| (num)")
        axes[1].plot(t, np.log(a0) + sigma_ana[m - 1] * t, "--", label=f"log|A_{m}| (ana)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("log amplitude")
    axes[1].set_title("Early-time Fourier amplitude growth")
    axes[1].legend(ncol=2, fontsize=8)

    fig.tight_layout()
    if save_outputs:
        p = (out / "linear_compare.png") if use_dir else _artifact("_linear_compare.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mode_ids, sigma_ana, sigma_num


def compare_stress_derivative_methods(output_dir: Path | None = None):
    use_dir = output_dir is not None
    out = output_dir if use_dir else _SCRIPT.parent
    if use_dir:
        out.mkdir(parents=True, exist_ok=True)
    methods = ["central", "spectral"]
    results = {}
    lin_out = out if use_dir else None
    for m in methods:
        mode, sigma_ana, sigma_num = linear_regime_compare(
            stress_derivative_method=m,
            output_dir=lin_out,
            save_outputs=False,
        )
        results[m] = (mode, sigma_ana, sigma_num)

    mode = results["spectral"][0]
    sigma_ana = results["spectral"][1]
    sigma_num_c = results["central"][2]
    sigma_num_s = results["spectral"][2]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(mode, sigma_ana, "o-", label="analysis")
    ax.plot(mode, sigma_num_c, "s--", label="numerical (central)")
    ax.plot(mode, sigma_num_s, "d--", label="numerical (spectral)")
    ax.set_xlabel("mode number m")
    ax.set_ylabel("growth rate sigma")
    ax.set_title("Linear growth comparison: central vs spectral stress_x")
    ax.legend()
    fig.tight_layout()
    path_fig = (out / "linear_compare_methods.png") if use_dir else _artifact("_stress_methods.png")
    fig.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mode, sigma_ana, sigma_num_c, sigma_num_s


def save_visualizations(
    x: np.ndarray,
    t: np.ndarray,
    n_hist: np.ndarray,
    rho_hist: np.ndarray,
    u_hist: np.ndarray,
    output_path: Path | None = None,
):
    """
    メイン図: 上段 n, rho, u の x–t、下段最終時刻のプロファイル（1 ファイルに統合）。
    """
    out_png = output_path if output_path is not None else _OUT_MAIN_PNG
    dx = float(x[1] - x[0])
    t_max = float(t[-1])
    extent = [float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), t_max, 0.0]

    fig = plt.figure(figsize=(14, 8))
    fields = [
        ("n(x,t)", n_hist),
        ("rho(x,t)", rho_hist),
        ("u(x,t)", u_hist),
    ]
    for col, (title, field) in enumerate(fields):
        ax = plt.subplot2grid((2, 3), (0, col))
        im = ap.atlas_imshow(
            ax,
            field,
            heatmap="scalar",
            origin="upper",
            aspect="auto",
            extent=extent,
            interpolation="bilinear",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax_bottom = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    ax_bottom.plot(x, n_hist[-1], label="n (final)")
    ax_bottom.plot(x, rho_hist[-1], label="rho (final)")
    ax_bottom.plot(x, u_hist[-1], label="u (final)")
    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel("value")
    ax_bottom.set_title("Final spatial profiles")
    ax_bottom.legend()
    ax_bottom.grid(alpha=0.3)

    fig.suptitle("Mechanochemical model (1D) — spacetime + final profiles", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dominant_mode_number(field: np.ndarray, dx: float, L: float) -> int:
    """rFFT bin index m>=1 with largest amplitude (mean removed)."""
    a = field - np.mean(field)
    ah = np.abs(np.fft.rfft(a))
    m = int(np.argmax(ah[1:]) + 1)
    return m


def main():
    # ホワイトノイズ（バイアスなし・左右対称化なし）。
    # t≳8 では粗いモードへ遷移し支配モードが m=1 になりやすいため、
    # 波数 3 が優勢な時間帯の終端付近で打ち切る。
    x, t, n_hist, rho_hist, u_hist = run_simulation(
        T=7.5,
        seed=0,
        symmetric_init=False,
        initial_condition="unbiased_noise",
    )
    save_visualizations(x, t, n_hist, rho_hist, u_hist)
    print(f"Saved {_OUT_MAIN_PNG}")

    mode, sigma_ana, sigma_num = linear_regime_compare()
    print(f"Saved {_artifact('_linear_compare.png')}")

    L = float(x[-1] - x[0] + (x[1] - x[0]))
    dx = float(x[1] - x[0])
    m_n = dominant_mode_number(n_hist[-1], dx, L)
    print(f"Dominant Fourier mode in n at t_final: m={m_n} (target: 3)")

    print("Linear-mode growth rates (mode, analysis, numerical):")
    for m, sa, sn in zip(mode, sigma_ana, sigma_num):
        print(f"  m={int(m)}: ana={sa:.6e}, num={sn:.6e}")
    print(f"Grid points: {x.size}, saved frames: {t.size}, final time: {t[-1]:.4f}")


if __name__ == "__main__":
    main()
