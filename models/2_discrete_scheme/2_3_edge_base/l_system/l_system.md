# L システム

**同梱スクリプト**: [`l_system.py`](l_system.py)

## モデルの導出（現象論）

**L-system** は、 rewriting 規則で **分枝パターン**（植物等）を生成する。

- **仮説**: トurtle 幾何＋文字列 production；連続 PDE ではない。
- **構成**: axiom＋production rules＋幾何解釈（角度・長さ）。
- 次節の grammar が離散的形態生成モデル。

## 離散モデル（モデル族）

文字列書き換え（Lindenmayer system）により分岐・再帰的構造を生成し、幾何 interpretation で植物形態等を表現します。

## 出典（原著・標準文献）

- Lindenmayer, A. (1968). Mathematical models for cellular interactions in development. *Journal of Theoretical Biology*, 18(3), 280–315. <https://doi.org/10.1016/0022-5193(68)90079-9>
- Prusinkiewicz, P., & Lindenmayer, A. (1990). *The Algorithmic Beauty of Plants*. Springer.
- **Mathematica**: [`L-system.nb`](../../../../Mathematica/2_discrete_scheme/2_3_edge_base/l_system/L-system.nb)

## 数値計算スキーム

[`l_system.py`](l_system.py) を参照（文字列反復＋タートル幾何）。

## パラメータ一覧

公理・反復回数・描画設定は [`l_system.py`](l_system.py) を参照。

## 数理解析

形式言語としての L 系の性質（D0L, OL 等）は理論 CS と関連。
