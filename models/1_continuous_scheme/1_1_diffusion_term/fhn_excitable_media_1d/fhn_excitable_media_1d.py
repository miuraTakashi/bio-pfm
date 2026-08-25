"""
1D 興奮性媒質 — FitzHugh–Nagumo + 拡散（進行パルス）
`fhn_excitable_media_2d.py` と同型の反応項・平衡の取り方、
u の拡散は 1D 周期格子の半陰的スペクトル法（`fft` / `ifft`；陰式核の周波数応答は `1/fft(kern)`）。

方程式:
  ∂u/∂t = Du * u_xx + u - u³/3 - v
  ∂v/∂t = ε * (u + β - γ*v)
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


def compute_rest_state(beta: float, gamma: float) -> tuple[float, float]:
    """拡散なし平衡（2D 版と同一の Newton 手続き）。"""
    u = -1.0
    for _ in range(30):
        c = 1.0 - 1.0 / gamma
        f = u**3 - 3.0 * u * c + 3.0 * beta / gamma
        fp = 3.0 * u**2 - 3.0 * c
        if abs(fp) < 1e-14:
            break
        u -= f / fp
    v = (u + beta) / gamma
    return float(u), float(v)


def build_diffusion_multiplier_fft1d(n: int, coeff: float) -> np.ndarray:
    """
    陰式ステップ (I - coeff * L)^{-1} の周波数乗数。
    L は周期 3 点ラプラシアン。`norm='ortho'` の rfft ではなく、
    標準 `fft` の固有値 `fft(kern)` で除算する（直接法と一致）。
    """
    kern = np.zeros(n, dtype=np.float64)
    kern[0] = 1.0 + 2.0 * coeff
    kern[1] = -coeff
    kern[-1] = -coeff
    lam = np.fft.fft(kern)
    return 1.0 / lam


def cubic_roots_frozen_v(v_frozen: float) -> tuple[float, float, float]:
    """
    f(u;v_frozen)=u-u^3/3-v_frozen=0 の 3 根（昇順実数）を返す。
    """
    roots = np.roots([1.0, 0.0, -3.0, 3.0 * v_frozen])
    r = np.sort(np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)]))
    if r.size != 3:
        raise ValueError(f"frozen-v cubic does not have 3 real roots: v={v_frozen}, roots={roots}")
    return float(r[0]), float(r[1]), float(r[2])


def analytical_front_speed_frozen_v(Du: float, v_frozen: float) -> float:
    """
    Frozen-v Nagumo front 近似:
      u_t = Du u_xx - (1/3)(u-r1)(u-r2)(u-r3)
      c ≈ sqrt(Du/6) * (r1 + r3 - 2 r2)
    """
    r1, r2, r3 = cubic_roots_frozen_v(v_frozen)
    return float(np.sqrt(max(Du, 0.0) / 6.0) * (r1 + r3 - 2.0 * r2))


def estimate_right_front_positions(
    history: np.ndarray,
    x: np.ndarray,
    dt_rec: float,
    *,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    中央刺激から右向きに進むフロント位置 x_front(t) を閾値交差で推定。
    """
    n_times, nx = history.shape
    center = nx // 2
    times = np.arange(n_times, dtype=np.float64) * dt_rec
    x_front = np.full(n_times, np.nan, dtype=np.float64)

    for k in range(n_times):
        u = history[k]
        seg = u[center:]
        idxs = np.where((seg[:-1] >= threshold) & (seg[1:] < threshold))[0]
        if idxs.size == 0:
            continue
        i0 = center + int(idxs[0])
        u0 = u[i0]
        u1 = u[i0 + 1]
        if abs(u1 - u0) < 1e-12:
            x_front[k] = x[i0]
        else:
            alpha = (threshold - u0) / (u1 - u0)
            x_front[k] = x[i0] + alpha * (x[i0 + 1] - x[i0])
    return times, x_front


def initial_center_stimulus(
    nx: int,
    u_rest: float,
    v_rest: float,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    t=0 で領域中央（格子インデックスの中点）に閾値超えのガウス刺激を重ね、
    周期境界上で左右対称に進行するパルス対を誘起する。
    """
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype=np.float64)
    x0 = 0.5 * (nx - 1)
    width = max(4.0, 0.022 * nx)
    bump = 3.2 * np.exp(-0.5 * ((x - x0) / width) ** 2)
    u = np.full(nx, u_rest, dtype=np.float64) + bump
    v = np.full(nx, v_rest, dtype=np.float64)
    u += 1e-5 * rng.standard_normal(nx)
    v += 1e-5 * rng.standard_normal(nx)
    return u, v


def simulate_fhn_1d(
    nx: int = 640,
    Du: float = 1.2,
    dx: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 2000,
    eps: float = 0.1,
    beta: float = 0.7,
    gamma: float = 0.5,
    record_every: int = 10,
    ic_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    1D FHN。実時間は `T = n_steps * dt`（既定で T=200）。
    `record_every` ステップごとに u を行として積み、カイモグラフ用 (n_times, nx) を返す。
    """
    u_rest, v_rest = compute_rest_state(beta, gamma)
    u, v = initial_center_stimulus(nx, u_rest, v_rest, seed=ic_seed)

    coeff = Du * dt / (dx * dx)
    ku = build_diffusion_multiplier_fft1d(nx, coeff)

    n_times = 1 + n_steps // record_every
    history = np.empty((n_times, nx), dtype=np.float64)
    history[0] = u.copy()

    t_idx = 1
    for step in range(1, n_steps + 1):
        fu = u + dt * (u - u**3 / 3.0 - v)
        v = v + dt * eps * (u + beta - gamma * v)
        u = np.real(np.fft.ifft(np.fft.fft(fu) * ku))

        if step % record_every == 0:
            if t_idx < n_times:
                history[t_idx] = u.copy()
                t_idx += 1

    if t_idx < n_times:
        history = history[:t_idx]
    return history, np.arange(nx) * dx, dt * record_every


def main() -> None:
    nx = 640
    Du = 1.2
    eps = 0.1
    beta = 0.7
    gamma = 0.5
    dt = 0.1
    t_end = 200.0
    n_steps = int(round(t_end / dt))
    record_every = 10

    print("Simulating FHN 1D traveling pulse (semi-implicit u-diffusion, fft)...")
    history, x, dt_rec = simulate_fhn_1d(
        nx=nx,
        Du=Du,
        dt=dt,
        n_steps=n_steps,
        eps=eps,
        beta=beta,
        gamma=gamma,
        record_every=record_every,
    )
    n_times = history.shape[0]
    t_max = (n_times - 1) * dt_rec
    print(f"Done. {n_times} time slices, t in [0, {t_max:.1f}].")

    u_rest, v_rest = compute_rest_state(beta=beta, gamma=gamma)
    c_ana = analytical_front_speed_frozen_v(Du=Du, v_frozen=v_rest)
    times_front, x_front = estimate_right_front_positions(history, x, dt_rec, threshold=0.0)
    valid = np.isfinite(x_front)
    fit_mask = valid & (times_front >= 0.2 * t_max) & (times_front <= 0.9 * t_max)
    c_num = np.nan
    x_fit = np.full_like(x_front, np.nan)
    if np.sum(fit_mask) >= 4:
        a, b = np.polyfit(times_front[fit_mask], x_front[fit_mask], 1)
        c_num = float(a)
        x_fit = a * times_front + b

    fig_num, (ax_kymo, ax_prof) = plt.subplots(2, 1, figsize=(10.0, 7.2), constrained_layout=True)

    u_lo = float(np.percentile(history, 1.0))
    u_hi = float(np.percentile(history, 99.0))
    if u_hi <= u_lo + 1e-6:
        u_lo, u_hi = float(history.min()), float(history.max())
    im = ap.atlas_imshow(
        ax_kymo,
        history,
        origin="lower",
        aspect="auto",
        extent=(float(x[0]), float(x[-1]), 0.0, t_max),
        interpolation="nearest",
        heatmap="scalar",
        vmin=u_lo,
        vmax=u_hi,
    )
    ax_kymo.set_xlabel("x")
    ax_kymo.set_ylabel("t")
    ax_kymo.set_title("u(x, t)  kymograph (center stimulus at t=0)")
    fig_num.colorbar(im, ax=ax_kymo, fraction=0.035, pad=0.02, label="u")

    idx_lines = np.linspace(0, n_times - 1, num=5, dtype=int)
    for k in idx_lines:
        tk = k * dt_rec
        ax_prof.plot(x, history[k], lw=1.2, label=f"t = {tk:.0f}")
    ax_prof.set_xlabel("x")
    ax_prof.set_ylabel("u")
    ax_prof.set_title("u profiles at selected times")
    ax_prof.legend(loc="upper right", fontsize=8)
    ax_prof.set_ylim(-2.0, 2.0)
    ax_prof.grid(True, alpha=0.3)

    fig_num.suptitle(
        "1D Excitable Medium — Traveling pulse (numerical)\n"
        f"(FHN, Du={Du:.1f}, ε={eps:.1f}, T={t_end:.0f}, semi-implicit u-diffusion, fft)",
        fontsize=11,
    )
    out_num = (Path(__file__).resolve().parent / "results" / "fhn_excitable_media_1d_numerical.png")
    fig_num.savefig(out_num, dpi=150, bbox_inches="tight")
    plt.close(fig_num)

    fig_ana, ax_cmp = plt.subplots(1, 1, figsize=(8.2, 5.0), constrained_layout=True)
    if np.any(valid):
        ax_cmp.plot(times_front[valid], x_front[valid], "o", ms=2.8, alpha=0.65, label="front position")
    if np.isfinite(c_num):
        ax_cmp.plot(times_front, x_fit, "r-", lw=1.3, label=f"linear fit: c_num={c_num:.3f}")
    x0 = float(x[nx // 2])
    ax_cmp.plot(times_front, x0 + c_ana * times_front, "k--", lw=1.1, label=f"frozen-v approx: c_ana={c_ana:.3f}")
    ax_cmp.set_xlabel("t")
    ax_cmp.set_ylabel("front position x")
    ax_cmp.set_title("Pulse-front speed: numerical vs analytical")
    ax_cmp.grid(True, alpha=0.3)
    ax_cmp.legend(loc="upper left", fontsize=8)
    ax_cmp.text(
        0.03,
        0.97,
        f"v_rest={v_rest:.3f}, u_rest={u_rest:.3f}\n"
        r"$c_{ana}\approx\sqrt{D_u/6}\,(r_1+r_3-2r_2)$"
        "\n"
        f"(r_i: roots of u-u^3/3-v_rest=0)",
        transform=ax_cmp.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82),
    )
    fig_ana.suptitle(
        "1D Excitable Medium — Front-speed analysis (numerical vs analytical)",
        fontsize=11,
    )
    out_ana = (Path(__file__).resolve().parent / "results" / "fhn_excitable_media_1d_analysis.png")
    fig_ana.savefig(out_ana, dpi=150, bbox_inches="tight")
    plt.close(fig_ana)
    print(f"Saved {out_num}")
    print(f"Saved {out_ana}")


if __name__ == "__main__":
    main()
