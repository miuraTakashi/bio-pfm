# Gray–Scott 1D

**同梱スクリプト**: [`gray_scott_1d.py`](gray_scott_1d.py)

## モデルの導出（現象論）

**Gray–Scott** は触媒–基質型反応 $U+2V	o 3V$、$V	o P$ を 2 変量 RD で表す。

- **仮説**: $U,V$ の Fick 拡散；$U$ は feed、$V$ は removal で補充。
- **保存則**: 各種の拡散＋反応源項。
- パラメータ $(F,k)$ で斑点・縞・パルスが生じ、次節の 1D 系が Turing 型パターンの標準例。

## 支配方程式（モデル族）

Gray–Scott 反応–拡散系（活性 $u$、抑制 $v$）の 1D 版。

## 出典（原著・標準文献）

- Pearson, J. E. (1993). Complex patterns in a simple system. *Science*, 261(5118), 189–192. <https://doi.org/10.1126/science.261.5118.189>
- **Mathematica**: [`1.1.11.GrayScott.nb`](../../../../Mathematica/1_continuous_scheme/1_1_diffusion_term/gray_scott_1d/1.1.11.GrayScott.nb)（次元・初期条件はノートブックに従う）

## 数値計算スキーム

[`gray_scott_1d.py`](gray_scott_1d.py) を参照。

## パラメータ一覧

[`gray_scott_1d.py`](gray_scott_1d.py) の docstring および関数引数を参照。

## 数理解析

Turing 様・スパイラル等のパラメータ領域が知られる。1D では空間パターンの次元が異なる。
