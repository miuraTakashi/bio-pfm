"""
L-system — 再帰的分枝構造（Mathematica/2_discrete_scheme/2_3_edge_base/l_system/L-system.nb から移植）。

各分枝 {p, v, t} (位置, 方向ベクトル, 線幅) から回転・縮小した
子分枝を生成するルールを繰り返し適用して樹木状構造を描く。
パラメータ（角度・長さ比・非対称性）を変えた複数パターンを比較表示する。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def rotation_2d(v: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]]) @ v


def generate_tree(n_gen: int = 8,
                  angle_left: float = -np.pi / 6,
                  angle_right: float = np.pi / 6,
                  scale_left: float = 0.8,
                  scale_right: float = 0.8,
                  thickness_decay: float = 1.5,
                  jitter_ratio: float = 0.05,
                  rng: np.random.Generator | None = None) -> list:
    """分枝木を生成。

    jitter_ratio:
        各分枝で回転角に与える相対揺らぎ（標準偏差）。
        例: 0.05 なら「基準角の約5%」のランダム揺らぎを与える。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    initial = (np.array([0.0, 0.0]), np.array([0.0, 1.0]), 0.1)
    branches = [initial]
    all_segments = []

    for _ in range(n_gen):
        new_branches = []
        for p, v, t in branches:
            all_segments.append((p.copy(), (p + v).copy(), t))
            p_end = p + v
            # 枝分かれ角に約5%の確率的ゆらぎ（ガウス）を入れる
            jl = 1.0 + jitter_ratio * float(rng.standard_normal())
            jr = 1.0 + jitter_ratio * float(rng.standard_normal())
            th_left = angle_left * jl
            th_right = angle_right * jr
            new_branches.append((p_end, scale_left * rotation_2d(v, th_left), t / thickness_decay))
            new_branches.append((p_end, scale_right * rotation_2d(v, th_right), t / thickness_decay))
        branches = new_branches

    for p, v, t in branches:
        all_segments.append((p.copy(), (p + v).copy(), t))

    return all_segments


def draw_tree(ax, segments, color="saddlebrown"):
    lc = LineCollection(
        [[[s[0][0], s[0][1]], [s[1][0], s[1][1]]] for s in segments],
        linewidths=[max(s[2] * 80, 0.2) for s in segments],
        colors=[color] * len(segments),
        alpha=0.85,
    )
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")


def thickness_count_relation(segments: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray]:
    """枝太さと本数の関係を返す（太さごとにカウント）。"""
    tvals = np.asarray([float(s[2]) for s in segments], dtype=float)
    # 浮動小数誤差でレベルが割れないよう丸めて集計
    t_round = np.round(tvals, 12)
    uniq, cnt = np.unique(t_round, return_counts=True)
    order = np.argsort(uniq)
    return uniq[order], cnt[order].astype(float)


TREE_CONFIGS = [
    {
        "label": "Symmetric ±20°",
        "angle_left": -np.pi / 9,
        "angle_right": np.pi / 9,
        "scale_left": 0.8,
        "scale_right": 0.8,
        "color": "saddlebrown",
    },
    {
        "label": "Symmetric ±30° (original)",
        "angle_left": -np.pi / 6,
        "angle_right": np.pi / 6,
        "scale_left": 0.8,
        "scale_right": 0.8,
        "color": "saddlebrown",
    },
    {
        "label": "Symmetric ±45°",
        "angle_left": -np.pi / 4,
        "angle_right": np.pi / 4,
        "scale_left": 0.8,
        "scale_right": 0.8,
        "color": "darkgreen",
    },
    {
        "label": "Wide ±60°",
        "angle_left": -np.pi / 3,
        "angle_right": np.pi / 3,
        "scale_left": 0.8,
        "scale_right": 0.8,
        "color": "darkgreen",
    },
    {
        "label": "Asymmetric −15° / +45°",
        "angle_left": -np.pi / 12,
        "angle_right": np.pi / 4,
        "scale_left": 0.85,
        "scale_right": 0.75,
        "color": "olive",
    },
    {
        "label": "Asymmetric −40° / +20°, scale 0.9/0.65",
        "angle_left": -np.pi * 40 / 180,
        "angle_right": np.pi * 20 / 180,
        "scale_left": 0.9,
        "scale_right": 0.65,
        "color": "olive",
    },
]


def main():
    n_gen = 8
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flat

    rng_master = np.random.default_rng(42)
    rel_data: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for i, (ax, cfg) in enumerate(zip(axes, TREE_CONFIGS)):
        segs = generate_tree(
            n_gen=n_gen,
            angle_left=cfg["angle_left"],
            angle_right=cfg["angle_right"],
            scale_left=cfg["scale_left"],
            scale_right=cfg["scale_right"],
            jitter_ratio=0.05,
            rng=np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)) + i),
        )
        draw_tree(ax, segs, color=cfg["color"])
        ax.set_title(f'{cfg["label"]} + 5% jitter', fontsize=10)
        th, cnt = thickness_count_relation(segs)
        rel_data.append((cfg["label"], th, cnt, cfg["color"]))

    fig.suptitle(f"L-system — Branching Variations ({n_gen} generations)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/l_system.png", dpi=150)
    plt.close()
    print("Saved l_system.png")

    # 太さと本数の関係（両対数）
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for label, th, cnt, color in rel_data:
        valid = (th > 0.0) & (cnt > 0.0)
        ax.loglog(
            th[valid],
            cnt[valid],
            marker="o",
            lw=1.2,
            ms=4,
            alpha=0.85,
            color=color,
            label=label,
        )
    ax.set_xlabel("branch thickness")
    ax.set_ylabel("count")
    ax.set_title("L-system: thickness-count relation (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig("results/l_system_thickness_count_loglog.png", dpi=150)
    plt.close()
    print("Saved l_system_thickness_count_loglog.png")


if __name__ == "__main__":
    main()
