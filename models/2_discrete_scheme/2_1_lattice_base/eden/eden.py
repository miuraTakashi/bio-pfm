"""
Eden モデル — 界面成長（Mathematica/2_discrete_scheme/2_1_lattice_base/eden/2.1.3.Eden.nb から移植）。

64×64 格子の中央から開始し、境界セル（occupied に隣接する空セル）から
ランダムに 1 つを選んで occupied にする操作を 1000 ステップ繰り返す。
最終的な格子パターンを PNG に保存する。
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
    """ボックスカウント法でフラクタル次元を推定。"""
    occ = (binary_grid > 0).astype(np.int8)
    n = occ.shape[0]
    if occ.shape[0] != occ.shape[1]:
        raise ValueError("box_counting_dimension expects square grid")

    if box_sizes is None:
        sizes: list[int] = []
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
        nn = (n // s) * s
        if nn <= 0:
            continue
        sub = occ[:nn, :nn]
        by = nn // s
        bx = nn // s
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


def get_boundary_cells(lattice: np.ndarray) -> list:
    """occupied セル (1) に 4 近傍で隣接する空セル (0) のリストを返す。"""
    n = lattice.shape[0]
    boundary = []
    # 境界を除く内部のみ走査
    for x in range(1, n - 1):
        for y in range(1, n - 1):
            if lattice[x, y] == 0:
                if (lattice[x - 1, y] == 1 or lattice[x + 1, y] == 1 or
                        lattice[x, y - 1] == 1 or lattice[x, y + 1] == 1):
                    boundary.append((x, y))
    return boundary


def run_eden(
    system_size: int = 64,
    n_steps: int = 1000,
    seed: int = 42,
    record_every: int = 10,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Eden 成長シミュレーション。最終格子と時系列スナップショットを返す。"""
    rng = np.random.default_rng(seed)
    lattice = np.zeros((system_size, system_size), dtype=np.int8)
    cx, cy = system_size // 2, system_size // 2
    lattice[cx, cy] = 1
    snapshots: list[np.ndarray] = [lattice.copy()]
    snapshot_steps: list[int] = [0]

    last_step = 0
    for step in range(1, n_steps + 1):
        last_step = step
        boundary = get_boundary_cells(lattice)
        if not boundary:
            break
        idx = rng.integers(len(boundary))
        x, y = boundary[idx]
        lattice[x, y] = 1
        if step % record_every == 0:
            snapshots.append(lattice.copy())
            snapshot_steps.append(step)

    if snapshot_steps[-1] != last_step:
        snapshots.append(lattice.copy())
        snapshot_steps.append(last_step)

    return lattice, snapshots, snapshot_steps


def main():
    n_samples = 10
    base_seed = 42

    lattice, snapshots, snapshot_steps = run_eden(seed=base_seed)
    fig, ax = plt.subplots(figsize=(6, 6))
    ap.atlas_imshow(
        ax, lattice, heatmap="binary", origin="upper", interpolation="nearest"
    )
    ax.set_title("Eden Model (64×64, 1000 steps)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/eden.png", dpi=150)
    plt.close()
    print("Saved eden.png")

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
    title = ax.set_title("Eden growth (step=0)")
    plt.tight_layout()

    def _update(i: int):
        im.set_data(snapshots[i])
        title.set_text(f"Eden growth (step={snapshot_steps[i]})")
        return im, title

    ani = FuncAnimation(fig, _update, frames=len(snapshots), interval=80, blit=False, repeat=True)
    ani.save("results/eden.gif", writer=PillowWriter(fps=12))
    plt.close()
    print("Saved eden.gif")

    # 複数サンプルの生成とボックスカウント次元推定
    samples: list[np.ndarray] = []
    dims: list[float] = []
    scales_all: list[np.ndarray] = []
    counts_all: list[np.ndarray] = []
    seeds: list[int] = []
    for i in range(n_samples):
        seed = base_seed + i
        g, _sn, _st = run_eden(seed=seed)
        d, s_used, n_boxes = box_counting_dimension(g)
        samples.append(g)
        dims.append(d)
        scales_all.append(s_used)
        counts_all.append(n_boxes)
        seeds.append(seed)

    # 10サンプル可視化
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 6.0), constrained_layout=True)
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
    fig.suptitle("Eden samples (box-counting fractal dimension)", fontsize=11)
    fig.savefig("results/eden_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved eden_samples.png")

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

    fig.savefig("results/eden_boxcount.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved eden_boxcount.png")

    # 結果表（CSV）
    with open("data/eden_fractal_dimensions.csv", "w", encoding="utf-8") as f:
        f.write("sample,seed,fractal_dimension\n")
        for i, (seed, d) in enumerate(zip(seeds, dims), start=1):
            d_txt = f"{d:.6f}" if np.isfinite(d) else "nan"
            f.write(f"{i},{seed},{d_txt}\n")
    print("Saved eden_fractal_dimensions.csv")


if __name__ == "__main__":
    main()
