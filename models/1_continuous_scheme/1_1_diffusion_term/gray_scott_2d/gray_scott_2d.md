# Gray–Scott 2D

**同梱スクリプト**: [`gray_scott_2d.py`](gray_scott_2d.py)

## モデルの導出（現象論）

Gray–Scott 反応拡散は 2D でも **局所反応＋拡散** の同型から導かれる。

- **現象**: Pearson 分類に対応する 2D パターン（spots, stripes, worms）。
- **仮説**: 等方拡散、均質領域。
- 次節の 2D 支配方程式は 1D モデルの自然な拡張。

## 支配方程式（モデル族）

Gray–Scott 反応–拡散系（2 成分、拡散係数比と反応項で多様なパターン）。

## 出典（原著・標準文献）

- Pearson, J. E. (1993). *Science*, 261(5118), 189–192. <https://doi.org/10.1126/science.261.5118.189>（図3の 12 分類 α–μ）
- 各 (F, k) の代表点: Munafo, *Pearson's Classification*（xmorphia） <https://mrob.com/pub/comp/xmorphia/pearson-classes.html>
- **Mathematica**: [`1.1.11.GrayScott.nb`](../../../../Mathematica/1_continuous_scheme/1_1_diffusion_term/gray_scott_1d/1.1.11.GrayScott.nb)

## 数値計算スキーム

[`gray_scott_2d.py`](gray_scott_2d.py) を参照（周期境界・陽的オイラー）。出力は **Pearson (1993) 図3に相当する 12 パラメータ**を **4 行×3 列**で可視化。

短縮実行: 環境変数 `GRAY_SCOTT_2D_N`（格子一辺）、`GRAY_SCOTT_2D_N_STEPS`（時間ステップ数）。

## パラメータ一覧

`simulate_gray_scott_2d` の `n`, `Du`, `Dv`, `F`, `k`, `dx`, `dt`, `n_steps`, `seed` 等。既定は `Du=0.16`, `Dv=0.08`, `dt=1`, 中央種（u,v）=(1/2, 1/4) ＋小乱数（Pearson と同型の初期化）。

## 数理解析

パラメータ空間における斑点・迷路・スパイラル等の分類が知られる（Pearson 分類）。
