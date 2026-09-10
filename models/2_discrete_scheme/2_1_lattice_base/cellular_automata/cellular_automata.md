# 初等セルオートマトン（Wolfram 256 規則）

対応スクリプト: [`cellular_automata.py`](cellular_automata.py)。

## モデルの導出（現象論）

**セルラーオートマトン** は、格子状態が局所ルールで離散更新される。

- **仮説**: 空間を格子に離散化；時間も離散；各セルは近傍のみ参照。
- **近似**: 連続極限はモデル依存（Life 等は意図的に離散）。
- 次節の更新則は、局所相互作用からパターンが生じる最小離散モデル。

## 離散モデル

1 次元、2 状態、3 近傍 $(s_{i-1},s_i,s_{i+1})$ から次世代 $s_i'$ を決める **ルール番号 0–255**。境界は **周期**（`np.roll`）。

## 出典（原著・標準文献）

- Wolfram, S. (1983). Statistical mechanics of cellular automata. *Reviews of Modern Physics*, 55(3), 601–644. <https://doi.org/10.1103/RevModPhys.55.601>
- Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media. <https://www.wolframscience.com/nks/>
- **Mathematica（移植元）**: [`CellularAutomata.nb`](../../../../Mathematica/2_discrete_scheme/2_1_lattice_base/cellular_automata/CellularAutomata.nb)

### 公開実装との類似（コード出典の注記）

本スクリプトは上記 Mathematica ノートからの移植（ルール 0–255 を 16×16 に並べ、中央 1 セル初期化）であり、下記サイトのソースを転載したものではない。`np.roll` で左右近傍を取り、ルール番号の 8 bit で更新する型は教育用コードで頻出する。

- Rossant, C. *IPython Cookbook*, 12.2 Simulating an elementary cellular automaton: <https://ipython-books.github.io/122-simulating-an-elementary-cellular-automaton/>

## 数値計算スキーム

- 各規則について空間–時間図を直接生成（代数更新、連続時間なし）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `width` | 64 | 空間セル数 |
| `generations` | 64 | 時間世代数 |
| 初期条件 | 中央 1 セルのみ ON | — |

## 数理解析

- 規則ごとに吸引子・周期軌道・カオス様挙動が知られる（分類は Wolfram 等）。本コードは **一覧可視化のみ**。
