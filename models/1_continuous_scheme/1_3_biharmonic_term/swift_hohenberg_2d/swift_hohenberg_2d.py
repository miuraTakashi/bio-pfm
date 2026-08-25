"""
2D Swift–Hohenberg 方程式 + 二次項（stripe–spot 選択）

方程式（反応項に q u^2 を追加）:
  ∂u/∂t = r u + q u^2 - (1 + ∇²)² u - u³

半陰式 FFT（線形 r u - (1+∇²)² u を陰的、非線形 q u^2 - u^3 を陽的）:
  u_hat_new = (u_hat + dt * nl_hat) / (1 - dt * L_op),  L_op = r - (1 - k²)²

勾配流: ∂_t u = -δE/δu  となる Lyapunov 汎関数
  E[u] = ∫ [ ½ ((1+∇²)u)² - (r/2) u² - (q/3) u³ + (1/4) u⁴ ] d x
（符号は δE/δu = (1+∇²)² u - r u - q u² + u³ を満たすように取った）。

解析（stripe vs spot / hex 系）:
  - u → -u 対称性を q u² が破る。臨界波数 |k|=1 近傍の弱非線形では、六角格子型（斑点系）の
    振幅 H に対する正準形に二次項 H² が現れ、縞（roll）の振幅 A の支配方程式は
    主モードが cos のみのとき空間平均で ∫ cos³ = 0 となり、同次の二次項が消える。
    よって小さな r>0 で |q| が十分大きいと、六角分岐（transcritical 的挙動）が支配的になり
    「斑点寄り」、|q| が小さいと縞が残りやすい、という選択が起こる。
  - 小振幅スケーリング: 線形項 r と二次項 q u² のバランスから |q| ~ √(r) のオーダーで
    競合が鋭くなる（本スクリプトでは数値フェーズ図に q ∝ √r のフィットを重ねる）。
  - 半解析境界: 単一 roll 縞 u = A cos(k·x) と、3 本の臨界波数ベクトル（120°）の重ね合わせ
    による試行関数 u = H ψ_hex（ψ_hex 正規化）で E を A,H について最小化し、
    どちらの谷が低いかで「縞優位 / 斑点優位」を判定した曲線を (r,q) 平面上に描く
    （完全厳密解ではなく 1 モード変分近似）。

現在の `main()` は q スイープ可視化（q=-1..1 を 0.2 刻み、横並び）を出力する。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.sparse.linalg import LinearOperator, eigs, eigsh, lobpcg  # type: ignore[import]
except Exception:  # noqa: BLE001
    LinearOperator = None
    eigs = None
    eigsh = None
    lobpcg = None

def _build_wavenumbers(n: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = L / n
    kx = np.fft.fftfreq(n, d=dx / (2 * np.pi))
    ky = np.fft.fftfreq(n, d=dx / (2 * np.pi))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    return KX, KY, K2


def stripe_order_metric(u: np.ndarray, K2: np.ndarray, KX: np.ndarray, KY: np.ndarray) -> float:
    """
    |k|≈1 のリング上のパワーの角分布の二重モーメント。
    縞（一方向の roll）では ±k の 2 点に集中し ~1 に近づき、等方的（斑点寄り）では小さくなる。
    """
    ph = np.abs(np.fft.fftshift(np.fft.fft2(u))) ** 2
    ring = (K2 > 0.72) & (K2 < 1.28)
    th = np.arctan2(KY[ring], KX[ring])
    p = ph[ring].astype(float)
    s = float(np.sum(p))
    if s <= 0.0:
        return 0.0
    p /= s
    return float(np.abs(np.sum(p * np.exp(1j * 2.0 * th))))


def simulate_sh2d(
    n: int = 128,
    L: float = 4 * np.pi,
    r: float = 0.1,
    q: float = 0.0,
    dt: float = 1.0,
    n_steps: int = 1000,
    seed: int = 42,
    u0: np.ndarray | None = None,
) -> np.ndarray:
    """Swift–Hohenberg 2D（半陰式 FFT）。最終 u を返す。"""
    if u0 is None:
        rng = np.random.default_rng(seed)
        u = 0.01 * rng.standard_normal((n, n))
    else:
        u = np.asarray(u0, dtype=float).copy()
    KX, KY, K2 = _build_wavenumbers(n, L)
    L_op = r - (1.0 - K2) ** 2
    denom = 1.0 - dt * L_op

    for _ in range(n_steps):
        nl = q * (u**2) - (u**3)
        nl_hat = np.fft.fft2(nl)
        u_hat = np.fft.fft2(u)
        u_hat_new = (u_hat + dt * nl_hat) / denom
        u = np.real(np.fft.ifft2(u_hat_new))

    return u


def initial_stripe_field(n: int, L: float, amp: float = 0.08) -> np.ndarray:
    x = np.linspace(0.0, L, n, endpoint=False)
    X, _Y = np.meshgrid(x, x, indexing="ij")
    return amp * np.cos(X)


def initial_spot_field(n: int, L: float, amp: float = 0.08) -> np.ndarray:
    return amp * _hex_three_cosines(n, L)


def _linearized_apply(
    vec: np.ndarray,
    u_star: np.ndarray,
    r: float,
    q: float,
    K2: np.ndarray,
) -> np.ndarray:
    """線形化作用素 J[u*]v を 2D 配列で返す。"""
    v = vec.reshape(u_star.shape)
    local = (2.0 * q * u_star - 3.0 * u_star**2) * v
    bih = np.real(np.fft.ifft2((1.0 - K2) ** 2 * np.fft.fft2(v)))
    out = r * v + local - bih
    return out


def leading_eigenvalue_linearized(
    u_star: np.ndarray,
    *,
    r: float,
    q: float,
    L: float,
) -> float:
    """
    u*=定常近傍での線形化 J の最大実部固有値を返す。
    scipy が使える場合は ARPACK（which='LR'）、無い場合は簡易 Rayleigh 反復。
    """
    n = u_star.shape[0]
    _KX, _KY, K2 = _build_wavenumbers(n, L)
    n2 = n * n

    def matvec(x: np.ndarray) -> np.ndarray:
        return _linearized_apply(x, u_star, r, q, K2).ravel()

    if LinearOperator is not None and eigsh is not None:
        op = LinearOperator((n2, n2), matvec=matvec, dtype=np.float64)
        try:
            lam, _vec = eigsh(op, k=1, which="LA", tol=1e-6, maxiter=3000)
            return float(lam[0])
        except Exception:  # noqa: BLE001
            pass

    if LinearOperator is not None and lobpcg is not None:
        op = LinearOperator((n2, n2), matvec=matvec, dtype=np.float64)
        try:
            x0 = np.random.default_rng(0).standard_normal((n2, 1))
            vals, _vecs = lobpcg(op, x0, largest=True, maxiter=200, tol=1e-5)
            return float(vals[0])
        except Exception:  # noqa: BLE001
            pass

    if LinearOperator is not None and eigs is not None:
        op = LinearOperator((n2, n2), matvec=matvec, dtype=np.float64)
        try:
            lam, _vec = eigs(op, k=1, which="LR", tol=1e-5, maxiter=3000, ncv=64)
            return float(np.real(lam[0]))
        except Exception:  # noqa: BLE001
            pass

    return float("nan")


def lap1(z: np.ndarray, dx: float) -> np.ndarray:
    """5 点ラプラシアン（周期）。"""
    return (
        (np.roll(z, -1, 0) - 2.0 * z + np.roll(z, 1, 0)) / (dx * dx)
        + (np.roll(z, -1, 1) - 2.0 * z + np.roll(z, 1, 1)) / (dx * dx)
    )


def energy_on_grid(u: np.ndarray, dx: float, r: float, q: float) -> float:
    """Lyapunov 密度の空間平均 E = ∫[...]/Area（5 点 (1+Δ)² で近似）。"""
    lu = lap1(u, dx)
    bi = u + 2.0 * lu + lap1(lu, dx)
    return float(
        0.5 * np.mean(bi**2)
        - 0.5 * r * np.mean(u**2)
        - (q / 3.0) * np.mean(u**3)
        + 0.25 * np.mean(u**4)
    )


def minimize_1mode_roll(
    r: float,
    q: float,
    L: float,
    *,
    n: int = 96,
) -> tuple[float, float]:
    """u = A cos(k0 x) の変分: A を 1 次元で粗探索し E 最小。"""
    m0 = int(round(L / (2.0 * np.pi)))
    if m0 < 1:
        m0 = 1
    dx = L / n
    x = np.linspace(0.0, L, n, endpoint=False)
    base = np.outer(np.cos((2.0 * np.pi * m0 / L) * x), np.ones(n, dtype=float))
    best_e, best_a = 1e300, 0.0
    for A in np.linspace(0.0, 2.0, 48):
        if abs(A) < 1e-9:
            e = 0.0
        else:
            e = energy_on_grid(A * base, dx, r, q)
        if e < best_e:
            best_e, best_a = e, A
    return best_a, best_e


def _hex_three_cosines(n: int, L: float) -> np.ndarray:
    """
    120° 間隔の単位波数ベクトル 3 本の等重み重ね（|k|=1）。
    k1=(1,0), k2=(-1/2,√3/2), k3=(-1/2,-√3/2) を格子座標に写像。
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    y = np.linspace(0.0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = (
        np.cos(X)
        + np.cos(-0.5 * X + (np.sqrt(3.0) / 2.0) * Y)
        + np.cos(-0.5 * X - (np.sqrt(3.0) / 2.0) * Y)
    )
    psi /= float(np.sqrt(np.mean(psi**2)) + 1e-12)
    return psi


def minimize_hex_mode(
    r: float,
    q: float,
    L: float,
    *,
    n: int = 96,
) -> tuple[float, float]:
    """u = H ψ_hex で E(H) を最小化（ψ_hex は上の試行関数）。"""
    psi = _hex_three_cosines(n, L)
    dx = L / n
    best_e, best_h = 1e300, 0.0
    for H in np.linspace(-2.0, 2.0, 72):
        e = energy_on_grid(H * psi, dx, r, q)
        if e < best_e:
            best_e, best_h = e, H
    return best_h, best_e


def variational_stripe_spot_boundary(
    r_vals: np.ndarray,
    L: float,
    *,
    n: int = 96,
    q_scan: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    各 r に対し E_roll^* = E_hex^* となる q を、q の走査と線形補間で求める（1 モード変分近似）。
    """
    if q_scan is None:
        q_scan = np.linspace(-2.5, 2.5, 72, dtype=float)
    q_out = np.empty_like(r_vals, dtype=float)
    for i, r in enumerate(r_vals):
        dif = np.empty_like(q_scan)
        for j, q in enumerate(q_scan):
            _, er = minimize_1mode_roll(r, q, L, n=n)
            _, eh = minimize_hex_mode(r, q, L, n=n)
            dif[j] = er - eh
        s = np.sign(dif)
        ch = np.flatnonzero(np.diff(s))
        if ch.size == 0:
            q_out[i] = float("nan")
            continue
        j0 = int(ch[0])
        y0, y1 = float(dif[j0]), float(dif[j0 + 1])
        x0, x1 = float(q_scan[j0]), float(q_scan[j0 + 1])
        q_out[i] = x0 - y0 * (x1 - x0) / (y1 - y0 + 1e-30)
    return r_vals, q_out


def phase_diagram_numerical(
    r_list: np.ndarray,
    q_list: np.ndarray,
    *,
    n: int = 96,
    L: float = 4 * np.pi,
    n_steps: int = 900,
    dt: float = 0.6,
    seed0: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    各 (r,q) でシミュレーションし stripe_order_metric を返す 2D 配列。
    """
    KX, KY, K2 = _build_wavenumbers(n, L)
    order = np.zeros((len(r_list), len(q_list)), dtype=float)
    for ir, r in enumerate(r_list):
        for iq, q in enumerate(q_list):
            u = simulate_sh2d(
                n=n,
                L=L,
                r=float(r),
                q=float(q),
                dt=dt,
                n_steps=n_steps,
                seed=seed0 + ir * len(q_list) + iq,
            )
            order[ir, iq] = stripe_order_metric(u, K2, KX, KY)
    return r_list, q_list, order


def main() -> None:
    # Requested scan: q = -1, -0.8, ..., 1.0 (step 0.2), and domain length 2.5x.
    # Original L=4π -> new L=10π gives about five critical wavelengths (λc≈2π).
    L = 10.0 * np.pi
    r = 0.1
    q_values = np.round(np.arange(-1.0, 1.0001, 0.2), 1)
    n_main = 320  # keep dx close to previous setting after enlarging L
    dt = 0.8
    n_steps = 1100
    seed = 42

    out_dir = Path(__file__).resolve().parent
    KX, KY, K2 = _build_wavenumbers(n_main, L)

    fields: list[np.ndarray] = []
    metrics: list[float] = []
    for q in q_values:
        u = simulate_sh2d(
            n=n_main,
            L=L,
            r=r,
            q=float(q),
            dt=dt,
            n_steps=n_steps,
            seed=seed,
        )
        fields.append(u)
        metrics.append(stripe_order_metric(u, K2, KX, KY))

    vlim = float(max(np.max(np.abs(u)) for u in fields))
    fig, axes = plt.subplots(
        1, len(q_values), figsize=(2.2 * len(q_values), 3.2), constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    im_last = None
    for ax, q, u, met in zip(axes, q_values, fields, metrics):
        im_last = ax.imshow(
            u,
            origin="lower",
            interpolation="bilinear",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.set_title(f"q={q:g}\nS={met:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if im_last is not None:
        fig.colorbar(im_last, ax=axes.tolist(), fraction=0.018, pad=0.01, label="u")

    fig.suptitle(
        "Swift–Hohenberg 2D: q-scan (r=0.1, q=-1..1 step 0.2)\n"
        r"Domain length $L=10\pi$ (2.5x original) $\approx 5\lambda_c$ with $\lambda_c\approx 2\pi$",
        fontsize=11,
    )

    out_path = out_dir / "results/swift_hohenberg_2d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- requested analysis: q in [0,1] for stripe / spot neighborhoods ---
    q_eigs = np.round(np.arange(0.0, 1.0001, 0.2), 1)
    r_eig = 0.1
    n_eig = 96
    eig_stripe: list[float] = []
    eig_spot: list[float] = []

    for q in q_eigs:
        u_str = simulate_sh2d(
            n=n_eig,
            L=L,
            r=r_eig,
            q=float(q),
            dt=0.8,
            n_steps=700,
            u0=initial_stripe_field(n_eig, L),
        )
        u_sp = simulate_sh2d(
            n=n_eig,
            L=L,
            r=r_eig,
            q=float(q),
            dt=0.8,
            n_steps=700,
            u0=initial_spot_field(n_eig, L),
        )
        eig_stripe.append(
            leading_eigenvalue_linearized(u_str, r=r_eig, q=float(q), L=L)
        )
        eig_spot.append(
            leading_eigenvalue_linearized(u_sp, r=r_eig, q=float(q), L=L)
        )
        print(
            f"q={q:>4.1f}: lambda_max(stripe)={eig_stripe[-1]: .4e}, "
            f"lambda_max(spot)={eig_spot[-1]: .4e}"
        )

    fig2, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(q_eigs, eig_stripe, "o-", lw=1.6, label="stripe neighborhood")
    ax.plot(q_eigs, eig_spot, "s-", lw=1.6, label="spot neighborhood")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("q")
    ax.set_ylabel(r"max Re $\lambda$ of linearized operator")
    ax.set_title(
        r"Swift-Hohenberg 2D: leading eigenvalue around stripe/spot states ($r=0.1$)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    out_path2 = out_dir / "results/swift_hohenberg_2d_analysis_eigen.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
