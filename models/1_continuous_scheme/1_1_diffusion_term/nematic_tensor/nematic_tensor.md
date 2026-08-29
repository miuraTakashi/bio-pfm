# nematic_tensor — 2D ネマティック・テンソル場のデモ

対応スクリプト: [`nematic_tensor.py`](nematic_tensor.py)。  
出力は `results/` 以下の **PNG 1 枚**と **GIF 1 枚**のみ（`streamlines_linewidth_timeseries.png`, `streamlines_linewidth.gif`）。

## モデルの導出（現象論）

液晶・細胞配向では **テンソル秩序変数** $Q$ が空間的に変調する。

- **仮説**: 自由エネルギー $F[Q,
abla Q]$（Landau–de Gennes）を最小化しつつ緩和。
- **構成**: 流体力学的耦合を省略し、$\partial_t Q = -L\,\delta F/\delta Q$ 型の梯度流。
- 次節は配向欠陥・テクスチャのフェーズフィールドテンソル方程式。

## 支配方程式（コードと同型）

2D での対称トレースレス秩序テンソル $Q$ を、独立成分 $(Q_{11}, Q_{12})$ のみで書く表現（$Q_{22}=-Q_{11}$）に対し、

$$
\frac{\partial Q_{11}}{\partial t}
= r\,\phi\,Q_{11} - 2u\,(Q_{11}^2+Q_{12}^2)\,Q_{11} + K\,\Delta Q_{11},
$$

$$
\frac{\partial Q_{12}}{\partial t}
= r\,\phi\,Q_{12} - 2u\,(Q_{11}^2+Q_{12}^2)\,Q_{12} + K\,\Delta Q_{12}.
$$

を周期境界（`np.roll` による 5 点ラプラシアン）のもと、陽的オイラーで積分している。

- **線形項** $r\phi\,Q_{ij}$: 有効温度に相当する係数（等方相／秩序相の傾きを与える Landau 型の最低次）。
- **立方飽和項** $-2u|Q|^2 Q_{ij}$（$|Q|^2=Q_{11}^2+Q_{12}^2$）: $|Q|^4$ 型の体積自由エネルギーから来る勾配流の形に対応する、秩序変数の飽和・安定化。
- **拡散項** $K\Delta Q_{ij}$: 一弾性定数近似の勾配エネルギー（Frank エネルギーの等方化した形）。

可視化では $(Q_{11},Q_{12})$ を **streamplot のベクトル場**として描いている（流線の接線方向が秩序の向きに対応；線幅は $|Q|$ に比例）。これは速度場の流線ではない（スクリプト docstring 同様）。

## 数値計算スキーム

- **空間**: 正方形領域 $[0,L]^2$、等間隔格子、周期境界の 5 点ラプラシアン。
- **時間**: 陽的オイラー。`dt` は拡散安定の目安 `dt ≲ dx²/(4K)` に対して選ぶ（実行時に目安を表示）。

## パラメータ一覧（スクリプト先頭の既定）

| 記号 | 既定値 | 意味 |
|------|--------|------|
| `L` | 12.8 | 領域一辺の長さ |
| `r` | −0.1 | 線形係数（$r\phi$ と組み合わせ） |
| `u` | 1.0 | 非線形（飽和）の強さ |
| `phi` | 0.1 | 線形項のスケール |
| `T` | 5.0 | 実時間の終端 $t\in[0,T]$ |
| `K` | 0.1 | 拡散（一弾性に相当）係数 |
| `dx` | 0.1 | 空間刻み |
| `dt` | 0.01 | 時間刻み |
| `nOutput` | 6 | 保存する時間スナップショット数 |

## 出典・根拠文献（Q テンソルと Landau–de Gennes）

### 教科書・総説

- **de Gennes, P. G. & Prost, J.** *The Physics of Liquid Crystals*（2nd ed., Oxford University Press, 1993）. <https://doi.org/10.1093/oso/9780198520245.001.0001>
  ネマティックの秩序変数、弾性（Frank）、揺らぎの章で、ディレクター表示とテンソル表示の物理的背景が整理されている。
- **Mottram, N. J. & Newton, C. J. P.** “Introduction to Q-tensor theory.” *arXiv*:1409.3542 (2014). <https://doi.org/10.48550/arXiv.1409.3542>  
  2D/3D の $Q$ テンソル定式化、境界条件、Landau–de Gennes エネルギーと平衡・欠陥の数学的枠組みへの入門として、本デモの記号と直結しやすい。
- **Virga, E. G.** *Variational Theories for Liquid Crystals*（Chapman & Hall, 1994）.  
  連続体理論の変分構造（弾性＋体積）を厳密に扱う文献。

### Landau–de Gennes モデルとダイナミクス

- **Beris, A. N. & Edwards, B. J.** *Thermodynamics of Flowing Systems: With Internal Microstructure*（Oxford University Press, 1994）. <https://doi.org/10.1093/oso/9780195076943.001.0001>
  配向テンソルに対する熱力学的整合な時間発展方程式（流れと結合した一般形）の古典的参照。
- **Sonnet, A. M., Kilian, A. & Hess, S.** “Alignment tensor versus director: description of defects in nematic liquid crystals.” *Physical Review E* **52**(1), 718–722 (1995). <https://doi.org/10.1103/PhysRevE.52.718>
  テンソル秩序変数の不可逆熱力学とディレクター記述の関係。

### 元ノート

- スクリプト docstring にあるとおり、元は `nematicTensor2D.ipynb` 系のデモ。本リポジトリの `Mathematica/` 配下に同名 `.nb` が無い場合は外部ノート由来の名称の可能性がある。

## 数理解析

- 本実装は完全な Leslie–Ericksen 方程式や流体力学との結合ではなく、**空間一様な係数を持つ Landau 型体積項＋等方拡散**による、秩序テンソル成分の簡略化された時間発展モデルである。
- 厳密な自由エネルギー最小化や Q の固有値拘束（物理的な $Q$ の許容範囲）をコード上で強制していない点に注意（デモ用の最小式）。
