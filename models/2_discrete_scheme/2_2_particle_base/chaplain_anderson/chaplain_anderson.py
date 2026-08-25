"""
Chaplain-Anderson 走化性モデル — バイアス付きランダムウォーク
（Mathematica/2_discrete_scheme/2_2_particle_base/chaplain_anderson/ChaplainAnderson.nb から移植）。

10 本の分枝（血管）が x 正方向への走化性バイアスを受けながら
ランダムウォークし、さらに一定確率で分枝する。

パラメータ:
  stepSize = 0.01 (ランダムステップ振幅)
  chemotaxis = 0.01 (x 方向バイアス)
  p = 0.01 (分枝確率)
  n_steps = 200 (ステップ数)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap


def _copy_branches(branches: list) -> list[list[tuple[float, float]]]:
    return [list(b) for b in branches]


def branches_to_occupancy(
    branches: list[list[tuple[float, float]]],
    grid_n: int = 120,
) -> np.ndarray:
    """枝の軌跡から占有格子（0/1）を作る。"""
    occ = np.zeros((grid_n, grid_n), dtype=np.float64)

    def _mark(xv: float, yv: float) -> None:
        xi = int(np.clip(np.floor(xv * grid_n), 0, grid_n - 1))
        yi = int(np.clip(np.floor(yv * grid_n), 0, grid_n - 1))
        occ[yi, xi] = 1.0  # imshow しやすいよう [y, x]

    for br in branches:
        if len(br) == 0:
            continue
        _mark(br[0][0], br[0][1])
        for k in range(1, len(br)):
            x0, y0 = br[k - 1]
            x1, y1 = br[k]
            seg_len = float(np.hypot(x1 - x0, y1 - y0))
            n_sub = max(1, int(np.ceil(seg_len * grid_n * 2.0)))
            for s in range(1, n_sub + 1):
                a = s / n_sub
                _mark((1.0 - a) * x0 + a * x1, (1.0 - a) * y0 + a * y1)
    return occ


def analytic_presence_probability(
    n_init: int,
    step_size: float,
    chemotaxis: float,
    branch_prob: float,
    n_steps: int,
    grid_n: int,
) -> np.ndarray:
    """拡散移流（+ 反応）近似で存在確率を解析的に評価。

    近似モデル:
      ∂ρ/∂t + v ∂ρ/∂x = D∇²ρ + λρ
    を点源初期条件の Green 関数で表し、時間積分した強度 Φ から
      P_exist = 1 - exp(-c Φ)
    の形で存在確率に変換する（c は呼び出し側で較正可能）。
    """
    # 離散ランダムウォークの連続極限近似
    v = chemotaxis
    D = (step_size**2) / 4.0  # var(cosθ)=var(sinθ)=1/2 より
    lam = float(np.log(1.0 + branch_prob))

    xs = (np.arange(grid_n, dtype=float) + 0.5) / grid_n
    ys = (np.arange(grid_n, dtype=float) + 0.5) / grid_n
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    y0s = np.array([0.1 * i + 0.05 for i in range(1, n_init + 1)], dtype=float)
    # 0 と n_steps 近傍の特異性を避ける
    taus = np.linspace(1e-3, float(n_steps), 320, dtype=float)
    phi = np.zeros_like(X)

    for y0 in y0s:
        acc = np.zeros_like(X)
        for t in taus:
            denom = 4.0 * np.pi * D * t
            expo = -((X - v * t) ** 2 + (Y - y0) ** 2) / (4.0 * D * t)
            rho = np.exp(lam * t) * np.exp(expo) / denom
            acc += rho
        # 台形則（均一刻み）
        phi += acc * (taus[1] - taus[0])
    return phi


def simulate_chaplain_anderson(
    n_init: int = 10,
    step_size: float = 0.01,
    chemotaxis: float = 0.006,
    branch_prob: float = 0.008,
    n_steps: int = 300,
    max_branches: int = 200,
    seed: int = 42,
    *,
    history_stride: int = 4,
    record_history: bool = True,
) -> tuple[list, list[tuple[int, list[list[tuple[float, float]]]]] | None]:
    """
    走化性バイアス付きランダムウォーク + 分枝。

    Returns
    -------
    branches : 最終時刻の各分枝の座標リスト
    history : (step, branches_snapshot) の列（record_history=False なら None）
    """
    rng = np.random.default_rng(seed)

    branches = []
    for y_idx in range(1, n_init + 1):
        y0 = 0.1 * y_idx + 0.1 * rng.random()
        branches.append([(0.0, y0)])

    history: list[tuple[int, list[list[tuple[float, float]]]]] = []
    if record_history:
        history.append((0, _copy_branches(branches)))

    for step in range(n_steps):
        new_branches = []
        for branch in branches:
            x, y = branch[-1]
            theta = rng.uniform(0, 2 * np.pi)
            x_new = x + chemotaxis + step_size * np.cos(theta)
            y_new = y + step_size * np.sin(theta)
            branch.append((x_new, y_new))

            if rng.random() < branch_prob and len(branches) + len(new_branches) < max_branches:
                phi = rng.uniform(0, 2 * np.pi)
                x_b = x_new + step_size * np.cos(phi)
                y_b = y_new + step_size * np.sin(phi)
                new_branches.append([(x_new, y_new), (x_b, y_b)])

        branches.extend(new_branches)

        if record_history and ((step + 1) % history_stride == 0):
            history.append((step + 1, _copy_branches(branches)))

    if record_history:
        last_step, _ = history[-1]
        if last_step != n_steps:
            history.append((n_steps, _copy_branches(branches)))

    return branches, history if record_history else None


def main() -> None:
    n_steps = 300
    history_stride = 5
    branches, history = simulate_chaplain_anderson(
        n_steps=n_steps,
        history_stride=history_stride,
        record_history=True,
    )

    base = Path(__file__).resolve().parent
    png_path = base / "results/chaplain_anderson.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    for branch in branches:
        xs = [p[0] for p in branch]
        ys = [p[1] for p in branch]
        ax.plot(xs, ys, color="red", alpha=0.6, linewidth=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Chaplain-Anderson Chemotaxis Model\n(Angiogenesis: biased random walk + branching)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved {png_path}")

    # --- GIF ---
    assert history is not None
    fig_g, ax_g = plt.subplots(figsize=(6.5, 6.5))

    def _draw_frame(snap: list[list[tuple[float, float]]]) -> None:
        ax_g.clear()
        for branch in snap:
            xs = [p[0] for p in branch]
            ys = [p[1] for p in branch]
            ax_g.plot(xs, ys, color="tab:red", alpha=0.65, linewidth=0.85)
        ax_g.set_xlim(0.0, 1.0)
        ax_g.set_ylim(0.0, 1.0)
        ax_g.set_aspect("equal")
        ax_g.set_xlabel("x")
        ax_g.set_ylabel("y")
        ax_g.grid(True, alpha=0.3)

    def _update(k: int) -> None:
        step, snap = history[k]
        _draw_frame(snap)
        ax_g.set_title(f"Chaplain-Anderson (step {step}/{n_steps}, {len(snap)} branches)")

    anim = FuncAnimation(
        fig_g,
        _update,
        frames=len(history),
        interval=90,
        blit=False,
    )
    gif_path = base / "results/chaplain_anderson.gif"
    anim.save(str(gif_path), writer=PillowWriter(fps=12))
    plt.close(fig_g)
    print(f"Saved {gif_path} ({len(history)} frames)")

    # --- 10サンプルのタイル表示 ---
    n_samples = 10
    grid_n = 120
    sample_branches: list[list[list[tuple[float, float]]]] = []
    occ_list: list[np.ndarray] = []
    for i in range(n_samples):
        br_i, _ = simulate_chaplain_anderson(
            n_steps=n_steps,
            seed=100 + i,
            history_stride=history_stride,
            record_history=False,
        )
        sample_branches.append(br_i)
        occ_list.append(branches_to_occupancy(br_i, grid_n=grid_n))

    fig_t, axes_t = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    axes_flat = axes_t.ravel()
    for i, ax in enumerate(axes_flat):
        if i >= n_samples:
            ax.axis("off")
            continue
        for branch in sample_branches[i]:
            xs_i = [p[0] for p in branch]
            ys_i = [p[1] for p in branch]
            ax.plot(xs_i, ys_i, color="tab:red", alpha=0.65, linewidth=0.65)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"sample {i + 1}", fontsize=9)
    fig_t.suptitle("Chaplain-Anderson: 10 stochastic samples", fontsize=12)
    tile_path = base / "results/chaplain_anderson_samples.png"
    fig_t.savefig(tile_path, dpi=150, bbox_inches="tight")
    plt.close(fig_t)
    print(f"Saved {tile_path}")

    # --- 実測存在確率（サンプル平均） ---
    p_emp = np.mean(np.stack(occ_list, axis=0), axis=0)

    # --- 解析的存在確率（拡散移流 + 分枝反応近似） ---
    phi = analytic_presence_probability(
        n_init=10,
        step_size=0.01,
        chemotaxis=0.006,
        branch_prob=0.008,
        n_steps=n_steps,
        grid_n=grid_n,
    )
    # 平均占有率が一致するよう Poisson 変換係数を二分探索で較正
    target = float(np.mean(p_emp))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        m = float(np.mean(1.0 - np.exp(-mid * phi)))
        if m < target:
            lo = mid
        else:
            hi = mid
    c_fit = 0.5 * (lo + hi)
    p_ana = 1.0 - np.exp(-c_fit * phi)
    p_ana = np.clip(p_ana, 0.0, 1.0)

    # --- 比較図 ---
    xs = (np.arange(grid_n, dtype=float) + 0.5) / grid_n
    x_prof_emp = np.mean(p_emp, axis=0)
    x_prof_ana = np.mean(p_ana, axis=0)

    fig_c, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    im0 = ap.atlas_imshow(
        axs[0, 0],
        p_emp,
        heatmap="scalar",
        origin="lower",
        extent=(0, 1, 0, 1),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axs[0, 0].set_title("Empirical presence probability (10 samples)")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    fig_c.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = ap.atlas_imshow(
        axs[0, 1],
        p_ana,
        heatmap="scalar",
        origin="lower",
        extent=(0, 1, 0, 1),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axs[0, 1].set_title("Analytical advection-diffusion probability")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    fig_c.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = ap.atlas_imshow(
        axs[1, 0],
        p_emp - p_ana,
        heatmap="scalar",
        origin="lower",
        extent=(0, 1, 0, 1),
        vmin=-0.5,
        vmax=0.5,
        interpolation="nearest",
    )
    axs[1, 0].set_title("Difference (empirical - analytical)")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    fig_c.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    axs[1, 1].plot(xs, x_prof_emp, color="k", lw=1.8, label="empirical")
    axs[1, 1].plot(xs, x_prof_ana, color="tab:blue", lw=1.6, ls="--", label="analytical")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("mean presence probability over y")
    axs[1, 1].set_title(f"x-profile comparison (fit c={c_fit:.3e})")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(frameon=False)

    cmp_path = base / "results/chaplain_anderson_probability_comparison.png"
    fig_c.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"Saved {cmp_path}")


if __name__ == "__main__":
    main()
