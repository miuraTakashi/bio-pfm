"""
Diffusion-Limited Aggregation (DLA) — 拡散律速凝集
（Mathematica/2_discrete_scheme/2_1_lattice_base/dla/2.1.4.DLA.nb から移植）。

200×200 格子の中央に 1 つの種を置き、境界からランダムウォーカーを
放出して格子の周期境界で動かす。ウォーカーが種または既存クラスタに
隣接したらその場所に付着させる（クラスタの一部になる）。
この操作を繰り返して樹状構造を形成する。

枝が表示上の境界に達する前に打ち切るため、中心からのチェビシェフ半径が
`grid_size//2 - wall_margin` に達したら付着ループを終了する。
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


def box_counting_dimension(
    binary_grid: np.ndarray,
    box_sizes: list[int] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """ボックスカウント法でフラクタル次元を推定。

    Returns
    -------
    d_est : float
        推定次元（log N(s) vs log(1/s) の最小二乗傾き）。
    s_used : np.ndarray
        使用したボックスサイズ。
    n_boxes : np.ndarray
        各ボックスサイズで 1 以上占有されたボックス数。
    """
    occ = (binary_grid > 0).astype(np.int8)
    n = occ.shape[0]
    if occ.shape[0] != occ.shape[1]:
        raise ValueError("box_counting_dimension expects square grid")

    if box_sizes is None:
        # 2,4,8,... の dyadic 尺度（格子以内）
        sizes = []
        s = 2
        while s <= n // 2:
            sizes.append(s)
            s *= 2
        box_sizes = sizes

    s_used: list[int] = []
    n_boxes: list[int] = []
    for s in box_sizes:
        if s <= 0 or s > n:
            continue
        # 端数は切り捨て（n=200, s in {2,4,8,16,32,64} なら問題なし）
        nn = (n // s) * s
        if nn <= 0:
            continue
        sub = occ[:nn, :nn]
        by = nn // s
        bx = nn // s
        # (by, s, bx, s) に並べ替えて各ボックスの占有有無を計算
        blocks = sub.reshape(by, s, bx, s)
        occupied = blocks.max(axis=(1, 3))
        cnt = int(np.count_nonzero(occupied))
        if cnt > 0:
            s_used.append(int(s))
            n_boxes.append(cnt)

    if len(s_used) < 2:
        return float("nan"), np.asarray(s_used, dtype=float), np.asarray(n_boxes, dtype=float)

    s_arr = np.asarray(s_used, dtype=float)
    n_arr = np.asarray(n_boxes, dtype=float)
    x = np.log(1.0 / s_arr)
    y = np.log(n_arr)
    d_est, _intercept = np.polyfit(x, y, 1)
    return float(d_est), s_arr, n_arr


def run_dla(
    grid_size: int = 200,
    n_particles: int = 8000,
    seed: int = 42,
    wall_margin: int = 22,
    record_every: int = 20,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """DLA シミュレーション。最終格子と時系列スナップショットを返す。

    wall_margin:
        境界（表示上の外周）まで残すセル数の目安。大きいほど早く終了する。
    record_every:
        付着イベント何回ごとにフレームを保存するか。
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    cx, cy = grid_size // 2, grid_size // 2
    grid[cx, cy] = 1
    snapshots: list[np.ndarray] = [grid.copy()]
    snapshot_steps: list[int] = [0]
    attached = 0

    # 4 方向のモーション
    moves = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    cluster_cells = {(cx, cy)}
    extent_limit = max(1, grid_size // 2 - wall_margin)
    max_cheb = 0

    def is_adjacent_to_cluster(x, y):
        for dx, dy in moves:
            nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
            if (nx, ny) in cluster_cells:
                return True
        return False

    for _ in range(n_particles):
        # ランダムな境界点からウォーカーを放出
        side = rng.integers(4)
        if side == 0:
            x, y = 0, rng.integers(grid_size)
        elif side == 1:
            x, y = grid_size - 1, rng.integers(grid_size)
        elif side == 2:
            x, y = rng.integers(grid_size), 0
        else:
            x, y = rng.integers(grid_size), grid_size - 1

        for _ in range(grid_size * grid_size * 4):
            if is_adjacent_to_cluster(x, y):
                grid[x, y] = 1
                cluster_cells.add((x, y))
                attached += 1
                if attached % record_every == 0:
                    snapshots.append(grid.copy())
                    snapshot_steps.append(attached)
                max_cheb = max(max_cheb, max(abs(x - cx), abs(y - cy)))
                if max_cheb >= extent_limit:
                    if snapshot_steps[-1] != attached:
                        snapshots.append(grid.copy())
                        snapshot_steps.append(attached)
                    return grid, snapshots, snapshot_steps
                break
            mv = moves[rng.integers(4)]
            x = (x + mv[0]) % grid_size
            y = (y + mv[1]) % grid_size

    if snapshot_steps[-1] != attached:
        snapshots.append(grid.copy())
        snapshot_steps.append(attached)
    return grid, snapshots, snapshot_steps


def main():
    n_samples = 10
    base_seed = 42

    # 代表 1 例（従来どおり png/gif 出力）
    grid, snapshots, snapshot_steps = run_dla(seed=base_seed)
    fig, ax = plt.subplots(figsize=(6, 6))
    ap.atlas_imshow(
        ax, grid, heatmap="binary", origin="upper", interpolation="nearest"
    )
    ax.set_title("Diffusion-Limited Aggregation (200×200, stop before wall)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/dla.png", dpi=150)
    plt.close()
    print("Saved dla.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ap.atlas_imshow(
        ax,
        snapshots[0],
        heatmap="binary",
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.axis("off")
    title = ax.set_title("DLA growth (attached=0)")
    plt.tight_layout()

    def _update(i: int):
        im.set_data(snapshots[i])
        title.set_text(f"DLA growth (attached={snapshot_steps[i]})")
        return im, title

    ani = FuncAnimation(fig, _update, frames=len(snapshots), interval=80, blit=False, repeat=True)
    ani.save("results/dla.gif", writer=PillowWriter(fps=12))
    plt.close()
    print("Saved dla.gif")

    # 複数サンプルの生成とボックスカウント次元推定
    samples: list[np.ndarray] = []
    dims: list[float] = []
    scales_all: list[np.ndarray] = []
    counts_all: list[np.ndarray] = []
    seeds: list[int] = []
    for i in range(n_samples):
        seed = base_seed + i
        g, _sn, _st = run_dla(seed=seed)
        d, s_used, n_boxes = box_counting_dimension(g)
        samples.append(g)
        dims.append(d)
        scales_all.append(s_used)
        counts_all.append(n_boxes)
        seeds.append(seed)

    # 10サンプル可視化
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n_samples:
            ap.atlas_imshow(
                ax,
                samples[i],
                heatmap="binary",
                origin="upper",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            d_txt = f"{dims[i]:.3f}" if np.isfinite(dims[i]) else "nan"
            ax.set_title(f"seed={seeds[i]}, D={d_txt}", fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle("DLA samples (box-counting fractal dimension)", fontsize=11)
    fig.savefig("results/dla_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved dla_samples.png")

    # ボックスカウント曲線＋次元分布
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i in range(n_samples):
        s_used = scales_all[i]
        n_boxes = counts_all[i]
        if s_used.size < 2:
            continue
        ax0.plot(np.log(1.0 / s_used), np.log(n_boxes), marker="o", ms=3, lw=1, alpha=0.7)
    ax0.set_xlabel("log(1/s)")
    ax0.set_ylabel("log N(s)")
    ax0.set_title("Box-counting curves")
    ax0.grid(alpha=0.3)

    d_arr = np.asarray(dims, dtype=float)
    d_valid = d_arr[np.isfinite(d_arr)]
    ax1.hist(d_valid, bins=min(8, max(3, d_valid.size)), color="0.35", alpha=0.85)
    if d_valid.size > 0:
        d_mean = float(np.mean(d_valid))
        d_std = float(np.std(d_valid, ddof=0))
        ax1.axvline(d_mean, color="tab:red", lw=1.5, ls="--", label=f"mean={d_mean:.3f}")
        ax1.legend(frameon=False, fontsize=9)
        ax1.set_title(f"Estimated D distribution (n={d_valid.size}, sd={d_std:.3f})")
    else:
        ax1.set_title("Estimated D distribution")
    ax1.set_xlabel("fractal dimension D")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.3)

    fig.savefig("results/dla_boxcount.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved dla_boxcount.png")

    # 結果表（CSV）
    with open("data/dla_fractal_dimensions.csv", "w", encoding="utf-8") as f:
        f.write("sample,seed,fractal_dimension\n")
        for i, (seed, d) in enumerate(zip(seeds, dims), start=1):
            d_txt = f"{d:.6f}" if np.isfinite(d) else "nan"
            f.write(f"{i},{seed},{d_txt}\n")
    print("Saved dla_fractal_dimensions.csv")


if __name__ == "__main__":
    main()
