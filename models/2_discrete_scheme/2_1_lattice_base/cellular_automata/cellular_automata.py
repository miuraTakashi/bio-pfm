"""
1D セルオートマトン — 256 ルール全て（Mathematica/2_discrete_scheme/2_1_lattice_base/cellular_automata/CellularAutomata.nb から移植）。

Wolfram の Elementary CA: 2 状態 3 近傍ルール (rule 0〜255)。
各ルールについて 1D 空間-時間図（幅 64 セル × 64 世代）を計算し、
全 256 ルールを 16×16 のグリッドに並べて 1 枚の PNG に保存する。
初期状態: 中央 1 セルのみ ON。
"""
from pathlib import Path

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



def apply_rule(cells: np.ndarray, rule: int) -> np.ndarray:
    """Wolfram elementary CA ルールを 1 ステップ適用（周期境界）。"""
    n = len(cells)
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
    left = np.roll(cells, 1)
    right = np.roll(cells, -1)
    idx = (left * 4 + cells * 2 + right).astype(int)
    return rule_bits[idx]


def spacetime_diagram(rule: int, width: int = 64, generations: int = 64) -> np.ndarray:
    """空間-時間ダイアグラムを返す。shape: (generations, width)"""
    cells = np.zeros(width, dtype=np.int8)
    cells[width // 2] = 1
    diagram = [cells.copy()]
    for _ in range(generations - 1):
        cells = apply_rule(cells, rule)
        diagram.append(cells.copy())
    return np.array(diagram)


def main():
    n_rules = 256
    cols = 16
    rows = n_rules // cols  # 16
    width, gens = 64, 64

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle("Elementary Cellular Automata — Rule 0–255", fontsize=14)

    for rule in range(n_rules):
        r, c = rule // cols, rule % cols
        ax = axes[r, c]
        diagram = spacetime_diagram(rule, width, gens)
        ap.atlas_imshow(ax, diagram, interpolation="nearest", aspect="auto")
        ax.set_title(str(rule), fontsize=5, pad=1)
        ax.axis("off")

    plt.tight_layout()
    out = (Path(__file__).resolve().parent / "results" / "cellular_automata.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
