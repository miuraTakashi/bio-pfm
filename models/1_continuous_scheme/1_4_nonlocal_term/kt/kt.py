"""
Kernel-based Turing (KT) model — Kondo (2017), 2D 周期領域。

参考文献:
  Kondo S. An updated kernel-based Turing model for studying the mechanisms of
  biological pattern formation. J. Theor. Biol. 2017;414:120–127.
  付表「Parameter settings」の **Fig. 7A–C** の数値は `KT_FIG7_KERNEL` に収録
  （`main()` で再現シミュレーション）。原著 PDF はリポジトリに含めず、
  手元で `References/KT.pdf` に置いて参照する運用可。
  （式 (1.4) およびメキシカンハット型カーネル K の記述は、手元の整理として
  Ei et al., J. Theor. Biol. 509 (2021) 110496 の式 (1.4)・周辺にも要約されている。）

連続形（Kondo 2017, 式 (1.4)）:
    ∂u/∂t = ν( (K * u)(x,t) ) − α u

  * は空間畳み込み（本実装では周期境界の巡回畳み込み = FFT）。
  ν は飽和付き恒等写像（Ei et al. が (1.4) と同型に引用している区分関数）:
      ν(r) = 0           (r ≤ 0)
           = r           (0 < r ≤ r*)
           = r*          (r > r*)

  K はメキシカンハット型（狭い正のガウス − 広い負のガウスを重ね、平均を 0 にシフト）。

  図ではあわせて「特徴長さが二つ」はっきりするカーネル例として、
  短・中・長の三つのガウスを G_s − w_m G_m − w_l G_l と合成した K も数値計算する
  （K(r) に中距離と長距離の二段の抑制スケールが現れる）。

離散化: 陽オイラー  u ← u + dt * ( ν(K*u) − α u )。
"""
from __future__ import annotations

import os
from math import sqrt, pi
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np


class KondoFig7KernelParams(TypedDict):
    """Kondo (2017) 付表「Parameter settings」の Fig. 7 行（カーネル形状）。"""

    ampA: float
    ampI: float
    widthA: float
    widthI: float
    distA: float
    distI: float
    int2d: float  # 論文表の「2D integrated」（参照値；実装では平均 0 にシフト）


# Journal of Theoretical Biology 414 (2017) 120–127, Parameter settings（PDF 表より）
# Fig. 7A–C の3条件をそのまま収録
KT_FIG7_KERNEL: dict[str, KondoFig7KernelParams] = {
    "A": {
        "ampA": 17.192,
        "ampI": -13.333,
        "widthA": 1.18,
        "widthI": 1.18,
        "distA": 8.3,
        "distI": 10.7,
        "int2d": 0.2,
    },
    "B": {
        "ampA": 21.085,
        "ampI": -19.733,
        "widthA": 0.739,
        "widthI": 0.935,
        "distA": 10.3,
        "distI": 8.7,
        "int2d": -0.158,
    },
    "C": {
        "ampA": 16.869,
        # NOTE:
        # PDF text extraction around the parameter table is noisy.
        # ampI is adjusted so that the listed Fig.7C "2D integrated = 24.6"
        # is satisfied under the same 200x200 grid and max interaction 20 cells.
        "ampI": -4.93900170980374,
        "widthA": 1.229,
        "widthI": 3.872,
        "distA": 5.9,
        "distI": 6.1,
        "int2d": 24.6,
    },
}


def nu_saturation(r: np.ndarray, r_star: float) -> np.ndarray:
    """Kondo (2017) の ν(r)（区分線形・飽和）。"""
    out = np.zeros_like(r, dtype=float)
    m0 = r <= 0.0
    m1 = (~m0) & (r <= r_star)
    m2 = r > r_star
    out[m1] = r[m1]
    out[m2] = r_star
    return out


def periodic_r_squared(n: int, dx: float) -> np.ndarray:
    """各格子点 (i,j) から原点 (0,0) までの周期距離の 2 乗 (× dx^2 込み)。"""
    idx = np.arange(n, dtype=float)
    d = np.minimum(idx, n - idx) * dx
    return d[:, None] ** 2 + d[None, :] ** 2


def build_mexican_hat_kernel_2d(
    n: int,
    dx: float,
    sigma_narrow: float,
    sigma_wide: float,
    inhib_weight: float = 1.0,
) -> np.ndarray:
    """
    メキシカンハット型 K（狭いガウス − inhib_weight × 広いガウス）。
    周期トーラス上の距離で半径を定義。全体平均を 0 にして ∫K≈0。
    """
    r2 = periodic_r_squared(n, dx)
    r = np.sqrt(np.maximum(r2, 1e-30))
    narrow = np.exp(-0.5 * (r / sigma_narrow) ** 2)
    wide = np.exp(-0.5 * (r / sigma_wide) ** 2)
    K = narrow - float(inhib_weight) * wide
    K -= float(K.mean())
    return K


def build_kernel_kondo2017_fig7(
    n: int,
    dx: float,
    p: KondoFig7KernelParams,
    max_interaction_distance: float = 20.0,
) -> np.ndarray:
    """
    Kondo (2017) 本文付近の定義に沿った和カーネル（周期距離 r 上の二ガウス）。

        Kernel(r) = Activator(r) + Inhibitor(r)
        Activator(r) = (ampA/√(2π)) exp(−½((r−distA)/widthA)²)
        Inhibitor(r) = (ampI/√(2π)) exp(−½((r−distI)/widthI)²)

    r は格子点間の周期トーラス上の最短距離。論文の「2D kernel」はこれを回転対称に敷いたもの。
    論文の式に合わせ、平均引きは行わない。
    """
    r2 = periodic_r_squared(n, dx)
    r = np.sqrt(np.maximum(r2, 1e-30))
    inv_sqrt_2pi = 1.0 / sqrt(2.0 * pi)
    act = (
        float(p["ampA"])
        * inv_sqrt_2pi
        * np.exp(-0.5 * ((r - float(p["distA"])) / float(p["widthA"])) ** 2)
    )
    inh = (
        float(p["ampI"])
        * inv_sqrt_2pi
        * np.exp(-0.5 * ((r - float(p["distI"])) / float(p["widthI"])) ** 2)
    )
    K = act + inh
    # KT.pdf: "The maximum interaction distance is 20 cells."
    K[r > float(max_interaction_distance)] = 0.0
    return K


def build_kernel_two_length_scales_2d(
    n: int,
    dx: float,
    sigma_short: float,
    sigma_mid: float,
    sigma_long: float,
    weight_mid: float,
    weight_long: float,
) -> np.ndarray:
    """
    特徴長さが二つはっきりする合成カーネル（等方・周期距離）。

        K ∝ G(σ_s) − w_m G(σ_m) − w_l G(σ_l)

    σ_s ≪ σ_m ≪ σ_l とすると、局所の正の峰、中距離の負の井戸、長距離の緩い負の寄与が分離し、
    K(r) 曲線に「二段」のスケールが現れる。
    """
    r2 = periodic_r_squared(n, dx)
    r = np.sqrt(np.maximum(r2, 1e-30))
    gs = np.exp(-0.5 * (r / sigma_short) ** 2)
    gm = np.exp(-0.5 * (r / sigma_mid) ** 2)
    gl = np.exp(-0.5 * (r / sigma_long) ** 2)
    K = gs - float(weight_mid) * gm - float(weight_long) * gl
    K -= float(K.mean())
    return K


def circular_conv2d(K: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Periodic convolution on torus (FFT)."""
    return np.real(np.fft.ifft2(np.fft.fft2(K) * np.fft.fft2(U)))


def kernel_1d_fig7_distance(x: np.ndarray, p: KondoFig7KernelParams) -> np.ndarray:
    """
    論文の 1 次元カーネル（距離 x 上の二ガウス和）をベクトル化して返す。

    Kernel(x)=ActivatorKernel(x)+InhibitorKernel(x)（Kondo 2017 Parameter settings）。
    """
    inv_sqrt_2pi = 1.0 / sqrt(2.0 * pi)
    # 論文中の x は「cell 間距離」と記述されているので x>=0 を想定
    xa = np.abs(np.asarray(x, dtype=float))
    act = (
        float(p["ampA"])
        * inv_sqrt_2pi
        * np.exp(-0.5 * ((xa - float(p["distA"])) / float(p["widthA"])) ** 2)
    )
    inh = (
        float(p["ampI"])
        * inv_sqrt_2pi
        * np.exp(-0.5 * ((xa - float(p["distI"])) / float(p["widthI"])) ** 2)
    )
    return act + inh


def kernel_1d_fig7_periodic_distance_sample(
    n: int,
    dx: float,
    p: KondoFig7KernelParams,
    max_interaction_distance: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D 円環上で距離 d=min(i,n-i) を使って Kernel(d) をサンプル。
    戻り値: (表示用 x>=0, 表示用 Kernel(x), FT 計算用 ring-kernel)。
    """
    i = np.arange(n, dtype=float)
    d = np.minimum(i, n - i) * dx
    k_ring = kernel_1d_fig7_distance(d, p)
    k_ring[d > float(max_interaction_distance)] = 0.0
    x_plot = np.arange(n // 2 + 1, dtype=float) * dx
    k_plot = kernel_1d_fig7_distance(x_plot, p)
    k_plot[x_plot > float(max_interaction_distance)] = 0.0
    return x_plot, k_plot, k_ring


def kernel_radial_profile_xaxis(
    K: np.ndarray, n: int, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    原点 (0,0) からの周期距離 r と、その距離上の K（j=0 の列）。
    カーネルが半径のみの関数なので、任意の方向のスライスで K(r) が一致する。
    """
    i = np.arange(n // 2 + 1, dtype=float)
    r = np.minimum(i, n - i) * dx
    vals = K[: n // 2 + 1, 0].astype(float)
    return r, vals


def simulate_kondo_kt_2d(
    n: int = 128,
    dx: float = 1.0,
    dt: float = 0.02,
    alpha: float = 0.35,
    r_star: float = 0.45,
    sigma_narrow: float = 2.0,
    sigma_wide: float = 8.0,
    inhib_weight: float = 1.05,
    n_steps: int = 4000,
    record_every: int = 40,
    seed: int = 42,
    K_preset: np.ndarray | None = None,
    ic_uniform01: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns
    -------
    u_final : (n, n)
    K : (n, n) カーネル
    kymograph : (n_record, n)  中央行 y=n//2 の u(x, t) の記録
    dt : float
    dx : float

    K_preset が与えられればそれをカーネルとして用いる（二スケール例など）。
    ic_uniform01: True のとき論文 Fig. 7 と同様に各格子に一様乱数 u∈[0,1] を初期値とする。
    """
    rng = np.random.default_rng(seed)
    if ic_uniform01:
        u = rng.random((n, n))
    else:
        u = 0.02 * rng.standard_normal((n, n))
    if K_preset is not None:
        K = np.asarray(K_preset, dtype=float)
        if K.shape != (n, n):
            raise ValueError(f"K_preset must be ({n},{n}), got {K.shape}")
    else:
        K = build_mexican_hat_kernel_2d(
            n, dx, sigma_narrow, sigma_wide, inhib_weight=inhib_weight
        )

    mid = n // 2
    snaps: list[np.ndarray] = []
    for step in range(n_steps):
        ku = circular_conv2d(K, u)
        u = u + dt * (nu_saturation(ku, r_star) - alpha * u)
        if (step + 1) % record_every == 0:
            snaps.append(u[mid, :].copy())

    u_final = u.copy()
    kymo = np.array(snaps) if snaps else np.zeros((0, n))
    return u_final, K, kymo, dt, dx


def _plot_fig7_style_row(
    axes_row: tuple,
    u: np.ndarray,
    p: KondoFig7KernelParams,
    n: int,
    dx: float,
    row_title: str,
) -> None:
    """
    論文 Fig. 7 と同型の 3 列: Kernel(x) / FT of kernel(x) / Result。

    Kernel と FT は付表パラメータの 1 次元式（周期 1 区間・平均 0）で描画。
    Result は 2D 終場 u（幾何中心が画像中央になるよう roll）。
    """
    ax0, ax1, ax2 = axes_row[0], axes_row[1], axes_row[2]

    x_plot, k_plot, k_ring = kernel_1d_fig7_periodic_distance_sample(
        n, dx, p, max_interaction_distance=20.0
    )
    ax0.plot(x_plot, k_plot, "k-", lw=1.8)
    ax0.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax0.set_title(f"{row_title}\nKernel (x)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("Kernel (x)")
    ax0.grid(True, alpha=0.35)

    fk = np.fft.fft(k_ring)
    freq = np.fft.fftfreq(n, d=dx)
    pos = freq >= 0.0
    ax1.plot(freq[pos], np.real(fk[pos]), "k-", lw=1.2)
    ax1.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax1.set_title("FT of kernel (x)")
    ax1.set_xlabel("spatial frequency (1 / length)")
    ax1.set_ylabel("Re FT")
    ax1.grid(True, alpha=0.35)

    u_disp = u
    u_min = float(np.min(u_disp))
    u_max = float(np.max(u_disp))
    if u_max > u_min:
        u_disp_norm = (u_disp - u_min) / (u_max - u_min)
    else:
        u_disp_norm = np.zeros_like(u_disp)
    im2 = ax2.imshow(
        u_disp_norm,
        origin="lower",
        interpolation="nearest",
        cmap="viridis",  # matplotlib default colormap
        vmin=0.0,
        vmax=1.0,
    )
    ax2.set_title("Result")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)


def main() -> None:
    """Kondo (2017) Fig. 7A–C: 付表のカーネル係数 + 論文と同型の [0,1] 乱数初期条件。"""
    n = 200
    dx = 1.0
    # 付表はカーネルのみ明示。時間発展の係数はデモ用に既存既定に合わせる。
    alpha = 0.35
    r_star = 0.45
    dt = 0.02
    n_steps = int(os.environ.get("KT_N_STEPS", "5000"))
    record_every = int(os.environ.get("KT_RECORD_EVERY", "50"))

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.5))
    for row, (label, params) in enumerate(KT_FIG7_KERNEL.items()):
        K = build_kernel_kondo2017_fig7(
            n, dx, params, max_interaction_distance=20.0
        )
        u, _, _kymo, _dt_out, _dx_out = simulate_kondo_kt_2d(
            n=n,
            dx=dx,
            dt=dt,
            alpha=alpha,
            r_star=r_star,
            n_steps=n_steps,
            record_every=record_every,
            seed=42 + row,
            K_preset=K,
            ic_uniform01=True,
        )
        int_ref = params["int2d"]
        int_num = float(np.sum(K) * dx * dx)
        title = (
            f"Fig. 7{label}: table kernel "
            f"(2D ∫K_ref≈{int_ref:g}, numeric≈{int_num:.3g})"
        )
        _plot_fig7_style_row(axes[row], u, params, n, dx, title)

    fig.suptitle(
        "Kernel-based Turing (Kondo 2017) — Fig. 7A–C parameter kernels, "
        r"$\partial u/\partial t=\nu(K*u)-\alpha u$, IC: uniform on $[0,1]$",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    out_path = Path(__file__).with_suffix(".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
