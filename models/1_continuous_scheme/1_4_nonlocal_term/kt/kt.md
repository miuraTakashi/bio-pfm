# Kernel-based Turing（Kondo 2017）— `KT.py`

**同梱スクリプト**: [`KT.py`](KT.py)（旧レイアウトでは `kt.py`。**Kuramoto–Tsuzuki の略称 KT とは無関係**）。

## モデルの導出（現象論）

**非局所抑制** 付き反応拡散は、長距離フィードバックでパルス間隔を決める。

- **仮説**: 局所活性化 $f(u)$ に、カーネル $K$ による非局所抑制 $\int K(x-y)u(y)\,dy$ を加える。
- **近似**: 1D 連続媒質；拡散は Fick 型。
- 次節の integro-differential / 実装 equivalent が KT 型モデル（本フォルダの `kt` は Kuramoto–Tsuzuki ではない）。

## 支配方程式（連続形）

Kondo (2017) の式 (1.4) 型:

$$
\frac{\partial u}{\partial t} = \nu\bigl( (K * u)(x,t) \bigr) - \alpha u
$$

$K$ はメキシカンハット型などの非局所カーネル、$\nu$ は飽和付き区分線形写像。実装では周期境界の巡回畳み込み（FFT）と陽的オイラー時間積分。

## Mathematica ノート `KT.nb` について

[`KT.nb`](../../../../Mathematica/1_continuous_scheme/1_4_nonlocal_term/kt/KT.nb) はリポジトリ上ほぼ空の場合があります。複素 Ginzburg–Landau（Kuramoto–Tsuzuki 系）のスタブとして言及されることがありますが、**本フォルダの Python は近藤のカーネル型 Turing モデル**です。

## 出典（原著・標準文献）

- Kondo, S. (2017). An updated kernel-based Turing model for studying the mechanisms of biological pattern formation. *Journal of Theoretical Biology*, 414, 120–127. <https://doi.org/10.1016/j.jtbi.2016.11.013>
- Ei, S.-I., et al. (2021). *Journal of Theoretical Biology*, 509, 110496（式 (1.4) 周辺の要約として参照されている例）。

## 数値計算スキーム

2D 周期領域・FFT 畳み込み・陽オイラー。

- **`main()` 既定**: Kondo (2017) 付表 **Fig. 7A–C** の係数（`ampA`, `ampI`, `widthA`, `widthI`, `distA`, `distI`）で、本文の二ガウス和カーネルを周期距離 $r$ 上に構築。付表の数値はソース内定数 `KT_FIG7_KERNEL` に収録されており、原著 PDF はリポジトリには含めません。突き合わせる場合は各自で論文を入手し、必要ならリポジトリルートに `References/KT.pdf` を置いてください。初期値は論文と同様に **各格子一様 $[0,1]$**（`ic_uniform01=True`）。出力図は論文 Fig. 7 と同型の **Kernel (x) / FT / Result** の 3 列（Kernel・FT は 1 次元式の周期サンプル）。
- **補助関数**: メキシカンハット型 `build_mexican_hat_kernel_2d`、三ガウス合成 `build_kernel_two_length_scales_2d` も利用可能。

環境変数（長い計算の短縮用）: `KT_N_STEPS`（既定 5000）、`KT_RECORD_EVERY`（既定 50）。

## パラメータ一覧

`simulate_kondo_kt_2d` のキーワード引数（`n`, `dx`, `dt`, `alpha`, `r_star`, `n_steps`, `ic_uniform01`, …）および `KT_FIG7_KERNEL`（Fig. 7 行）。詳細は [`KT.py`](KT.py) 先頭 docstring。

## 数理解析

非局所相互作用による Turing 型不安定と斑形成。カーネル形状が線形安定性と定常パターンのスケールに直結する。
