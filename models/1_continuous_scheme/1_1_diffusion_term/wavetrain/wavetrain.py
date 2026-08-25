"""
精子形成と波列（Wavetrain）— Mathematica/1_continuous_scheme/1_1_diffusion_term/wavetrain/1.1.2.精子形成とWavetrain.nb の 3 変数マップ。

周期境界（RotateLeft / RotateRight と同型）の離散ラプラシアン:

  diffusion[l] = (RotateLeft[l] + RotateRight[l] - 2 l) / dx^2

離散時間マップ f[{u,v,w}]（ノートブック In[30] と同じ）:

  u <- u + dt*(0.6*u - v + 0.01*diffusion[u] - w) - 0.2*u^3
  v <- v + dt*(1.5*u - 2*v + diffusion[v] - w)
  w <- w + dt*(u + v)

パラメータ（ノートブック In[22], In[26]）:
  du=0.01（u の拡散係数として式中 0.01 と一致）, dv=1（dt 安定化用）,
  domainSize = 6.28*4, dx = 0.1, dt = dx^2/dv/4,
  初期: u,v,w を [-amplitude, amplitude] の一様乱数, amplitude=0.01。

追加解析出力（2 ファイルに分割）:
  - wavetrain_analysis_dispersion.png … 支配的モード周りの固有値 Re/Im vs k、最不安定 k* と数値 k_num
  - wavetrain_analysis_wave.png … 支配的モード位相・ω・位相速度、および k_num での固有値一覧
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

from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


def diffusion(l: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(l, 1) + np.roll(l, -1) - 2.0 * l) / (dx * dx)


def step_f(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dt: float,
    dx: float,
    du_coeff: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 ステップの f（ノートブックと同式）。"""
    lap_u = diffusion(u, dx)
    lap_v = diffusion(v, dx)
    u_n = u + dt * (0.6 * u - v + du_coeff * lap_u - w) - 0.2 * (u**3)
    v_n = v + dt * (1.5 * u - 2.0 * v + lap_v - w)
    w_n = w + dt * (u + v)
    return u_n, v_n, w_n


def simulate_wavetrain_nb(
    domain_size: float = 6.28 * 4.0,
    dx: float = 0.1,
    dv: float = 1.0,
    amplitude: float = 0.01,
    n_steps: int = 20000,
    record_every: int = 40,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    戻り値: x, times, traj_u（n_time × n）。
    """
    rng = np.random.default_rng(seed)
    n = int(round(domain_size / dx))
    dt = (dx * dx / dv) / 4.0
    x = (np.arange(n, dtype=float) + 0.5) * dx

    u = rng.uniform(-amplitude, amplitude, size=n)
    v = rng.uniform(-amplitude, amplitude, size=n)
    w = rng.uniform(-amplitude, amplitude, size=n)

    traj = [u.copy()]
    times = [0.0]

    for step in range(1, n_steps + 1):
        u, v, w = step_f(u, v, w, dt, dx)
        if step % record_every == 0:
            traj.append(u.copy())
            times.append(step * dt)

    return x, np.array(times), np.array(traj)


def linear_mode_matrix(k: float, du_coeff: float = 0.01) -> np.ndarray:
    """Fourier モード k に対する線形化行列（u,v,w around 0）。"""
    return np.array(
        [
            [0.6 - du_coeff * k * k, -1.0, -1.0],
            [1.5, -2.0 - k * k, -1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def leading_eigenvalue_dispersion(
    dx: float,
    du_coeff: float = 0.01,
    n_samples: int = 320,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """leading eigenvalue の Re/Im を k に対して返す。"""
    k_nyq = np.pi / dx
    k_grid = np.linspace(0.0, k_nyq, n_samples)
    lead_re = np.empty_like(k_grid)
    lead_im = np.empty_like(k_grid)
    for i, k in enumerate(k_grid):
        eigs = np.linalg.eigvals(linear_mode_matrix(float(k), du_coeff=du_coeff))
        idx = int(np.argmax(np.real(eigs)))
        lead = eigs[idx]
        lead_re[i] = float(np.real(lead))
        lead_im[i] = float(np.imag(lead))
    k_star = float(k_grid[int(np.argmax(lead_re))])
    return k_grid, lead_re, lead_im, k_star


def estimate_wave_metrics(
    traj_u: np.ndarray,
    times: np.ndarray,
    dx: float,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    数値場から支配的波数・波長・角周波数・位相速度を推定。
    戻り値: k_dom, lambda_dom, omega, c_phase, phase_series, phase_fit
    """
    centered = traj_u - np.mean(traj_u, axis=1, keepdims=True)
    spec_tk = np.fft.rfft(centered, axis=1)
    power_k = np.mean(np.abs(spec_tk) ** 2, axis=0)
    if power_k.size <= 1:
        n_t = len(times)
        return 0.0, np.inf, 0.0, 0.0, np.zeros(n_t), np.zeros(n_t)

    power_k[0] = 0.0
    k_idx = int(np.argmax(power_k))
    freqs = np.fft.rfftfreq(traj_u.shape[1], d=dx)  # cycles / length
    freq_dom = float(freqs[k_idx])
    k_dom = 2.0 * np.pi * freq_dom
    lambda_dom = float((1.0 / freq_dom) if freq_dom > 1e-12 else np.inf)

    phase = np.unwrap(np.angle(spec_tk[:, k_idx]))
    slope, intercept = np.polyfit(times, phase, 1)
    phase_fit = slope * times + intercept
    omega = float(-slope)
    c_phase = float(omega / k_dom) if k_dom > 1e-12 else 0.0
    return k_dom, lambda_dom, omega, c_phase, phase, phase_fit


def _wavetrain_analysis_bundle(
    x: np.ndarray,
    times: np.ndarray,
    traj_u: np.ndarray,
    du_coeff: float = 0.01,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    list[complex],
]:
    dx = float(x[1] - x[0])
    k_grid, lead_re, lead_im, k_star = leading_eigenvalue_dispersion(dx=dx, du_coeff=du_coeff)
    k_dom, lambda_dom, omega, c_phase, phase, phase_fit = estimate_wave_metrics(traj_u, times, dx=dx)
    lambda_star = float((2.0 * np.pi / k_star) if k_star > 1e-12 else np.inf)
    eig_dom = np.linalg.eigvals(linear_mode_matrix(k_dom, du_coeff=du_coeff))
    eig_dom_sorted = sorted(eig_dom, key=lambda z: np.real(z), reverse=True)
    return (
        k_grid,
        lead_re,
        lead_im,
        k_star,
        lambda_star,
        k_dom,
        lambda_dom,
        omega,
        c_phase,
        phase,
        phase_fit,
        eig_dom_sorted,
    )


def save_wavetrain_analysis_split(
    x: np.ndarray,
    times: np.ndarray,
    traj_u: np.ndarray,
    out_dispersion: Path,
    out_wave: Path,
    du_coeff: float = 0.01,
) -> None:
    """解析を dispersion 図と位相・速度図の 2 PNG に分割して保存。"""
    (
        k_grid,
        lead_re,
        lead_im,
        k_star,
        lambda_star,
        k_dom,
        lambda_dom,
        omega,
        c_phase,
        phase,
        phase_fit,
        eig_dom_sorted,
    ) = _wavetrain_analysis_bundle(x, times, traj_u, du_coeff=du_coeff)

    fig_d, ax_d = plt.subplots(1, 1, figsize=(7.2, 4.4))
    ax_d.plot(k_grid, lead_re, color="C3", lw=1.4, label=r"Re $\lambda_{\max}(k)$")
    ax_d.plot(k_grid, lead_im, color="C0", lw=1.2, ls="--", label=r"Im $\lambda_{\max}(k)$")
    ax_d.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_d.axvline(k_star, color="C3", lw=1.0, ls=":", label=rf"$k_*={k_star:.3f}$")
    ax_d.axvline(k_dom, color="C2", lw=1.0, ls="-.", label=rf"$k_{{num}}={k_dom:.3f}$")
    ax_d.set_xlabel("k")
    ax_d.set_ylabel(r"eigenvalue components")
    ax_d.set_title("Linearized leading eigenvalue vs k")
    ax_d.grid(alpha=0.3)
    ax_d.legend(fontsize=8, loc="best")
    ax_d.text(
        0.02,
        0.98,
        f"lambda*={lambda_star:.3f}\nlambda_num={lambda_dom:.3f}",
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    fig_d.suptitle("Wavetrain analysis — dispersion (Re/Im λ)", fontsize=11)
    fig_d.tight_layout()
    fig_d.savefig(out_dispersion, dpi=150)
    plt.close(fig_d)
    print(f"Saved {out_dispersion}")

    fig_w, ax_w = plt.subplots(1, 1, figsize=(7.2, 4.4))
    ax_w.plot(times, phase, color="C0", lw=1.0, label="phase (dominant mode)")
    ax_w.plot(times, phase_fit, color="C3", lw=1.2, ls="--", label="linear fit")
    ax_w.set_xlabel("t")
    ax_w.set_ylabel("phase [rad]")
    ax_w.set_title("Phase slope → ω and phase speed")
    ax_w.grid(alpha=0.3)
    ax_w.legend(fontsize=8, loc="best")
    ax_w.text(
        0.02,
        0.98,
        f"omega={omega:.3f}, c_phase={c_phase:.3f}\n"
        f"k_num={k_dom:.3f}, lambda_num={lambda_dom:.3f}\n"
        f"eig(k_num)={eig_dom_sorted[0].real:.3f}"
        f"{eig_dom_sorted[0].imag:+.3f}i\n"
        f"2nd={eig_dom_sorted[1].real:.3f}{eig_dom_sorted[1].imag:+.3f}i\n"
        f"3rd={eig_dom_sorted[2].real:.3f}{eig_dom_sorted[2].imag:+.3f}i",
        transform=ax_w.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    fig_w.suptitle("Wavetrain analysis — wave metrics from simulation", fontsize=11)
    fig_w.tight_layout()
    fig_w.savefig(out_wave, dpi=150)
    plt.close(fig_w)
    print(f"Saved {out_wave}")


def make_gif(
    x: np.ndarray,
    times: np.ndarray,
    traj_u: np.ndarray,
    out_path: Path,
    fps: int = 12,
    max_frames: int = 200,
    dpi: int = 120,
) -> None:
    """Animate u(x, t) along recorded time slices (subsampled if many frames)."""
    n_t = traj_u.shape[0]
    skip = max(1, int(np.ceil(n_t / max_frames)))
    frame_indices = list(range(0, n_t, skip))

    um = float(np.max(np.abs(traj_u))) * 1.05
    if um <= 0.0:
        um = 1e-6

    fig, ax = plt.subplots(figsize=(10, 4))
    (line,) = ax.plot(x, traj_u[0], color="steelblue", lw=1.5)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-um, um)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Wavetrain u(x,t)  t = {float(times[0]):.2f}")
    ax.grid(alpha=0.3)

    def update(i: int):
        k = frame_indices[i]
        line.set_ydata(traj_u[k])
        ax.set_title(f"Wavetrain u(x,t)  t = {float(times[k]):.2f}")
        return (line,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        blit=False,
        interval=max(1, 1000 // max(fps, 1)),
    )
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}  ({len(frame_indices)} frames)")


def main():
    x, times, traj_u = simulate_wavetrain_nb()
    t_max = float(times[-1])
    dx = float(x[1] - x[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.25, 1.0])

    im = ap.atlas_imshow(
        axes[0],
        traj_u,
        heatmap="scalar",
        origin="upper",
        aspect="auto",
        extent=[float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), t_max, 0.0],
        interpolation="nearest",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title(
        r"$u(x,t)$ — 3-component RD (notebook $f$), wave bifurcation / wavetrain"
    )
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02, label="u")

    idxs = np.linspace(0, len(traj_u) - 1, 6, dtype=int)
    cols = ap.atlas_line_colors(len(idxs), heatmap="scalar")
    for j, c in zip(idxs, cols):
        axes[1].plot(x, traj_u[j], color=c, lw=1.0, label=f"t = {times[j]:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].set_title("u(x) snapshots")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Wavetrain model — 3-field explicit map (notebook 1.1.2 SHH Wavetrain.nb)",
        fontsize=11,
    )
    plt.tight_layout()
    out_png = (Path(__file__).resolve().parent / "results" / "wavetrain.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}  |u|_max last = {np.max(np.abs(traj_u[-1])):.4f}")

    out_disp = (Path(__file__).resolve().parent / "results" / "wavetrain_analysis_dispersion.png")
    out_wave = (Path(__file__).resolve().parent / "results" / "wavetrain_analysis_wave.png")
    save_wavetrain_analysis_split(
        x, times, traj_u, out_dispersion=out_disp, out_wave=out_wave, du_coeff=0.01
    )

    out_gif = Path(__file__).with_suffix(".gif")
    make_gif(x, times, traj_u, out_path=out_gif)


if __name__ == "__main__":
    main()
