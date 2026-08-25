# turing_domain_growth — 1D 領域成長（Neumann）

対応スクリプト: [`turing_domain_growth.py`](turing_domain_growth.py)。

移植元: [`1.1.2.軟骨形成とTuringパターン.nb`](../../../../Mathematica/1_continuous_scheme/1_1_diffusion_term/turing_1d/1.1.2.軟骨形成とTuringパターン.nb) の **1D 領域成長** 部分。

## モデルの導出（現象論）

胚発生では **領域の成長** と **パターン形成** が同時に起こる。

- **仮説**: 反応拡散は固定則、領域半径 $R(t)$ は所与の成長則（例: 線形）で拡大。
- **近似**: 成長は幾何学的（物質流入の詳細は省略）、パターンは Turing 型 RD。
- 境界条件を動く $R(t)$ に合わせることで、次節の成長付き RD 問題になる。

## 支配方程式

格子間隔 $dx(t)$ を増加させながら、

$$
f(p,q)=0.6p-q+1.2p^2-p^3,\quad g(p,q)=1.5p-2q
$$

と Neumann 型の離散拡散で更新する（実装は元スクリプトと同じ）。

## 数値スキーム

`diffusion_neumann` による離散勾配 + 陽的オイラー。各「単位時間」外側ループ後にスナップショット。

## 線形安定性と波長比較

原点近傍のヤコビアン

$$
J(0,0)=\begin{pmatrix}0.6 & -1\\1.5 & -2\end{pmatrix},
\quad D=\mathrm{diag}(D_p,D_q)
$$

から

$$
A(k)=J-k^2D,\qquad
\sigma_{\max}(k)=\max\operatorname{Re}\lambda(A(k))
$$

を計算し、最不安定波数 $k_*$ と理論波長 $\lambda_*=2\pi/k_*$ を推定する。

さらに各時刻の数値場 $p(x,t)$ に対して 1D FFT から支配的波長 $\lambda_{\mathrm{num}}(t)$ を推定し、成長ドメインでの理論値との比較をプロットする。

## パラメータ

| 名前 | 既定 | 説明 |
|------|------|------|
| `simulation_length` | 600 | 外側ループ長（旧 200 の 3 倍）。実行時間は概ねこれに比例 |
| `TURING_DOMAIN_GROWTH_LENGTH` | （未設定時は上記） | 環境変数で上書き |
| `final_domain_size` | 8.0 | `original_domain_size=1` と組み合わせ、`dx` が約 **8 倍**まで伸びるスケール（実装の `dx_change_per_dt` に入る） |
| `TURING_DOMAIN_GROWTH_FINAL_SIZE` | （未設定時は上記） | 環境変数で `final_domain_size` を上書き |
| `dp`, `dq` | `0.0008`, `0.04` | 拡散係数 $D_p$, $D_q$（旧 `0.0002`, `0.01` の 4 倍） |
| `TURING_DOMAIN_GROWTH_DP`, `TURING_DOMAIN_GROWTH_DQ` | （未設定時は上記） | 環境変数で上書き |
| `dt` | `0.0025` | 時間刻み（拡散 4 倍に合わせ既定を 1/4 にし陽解法を安定化） |
| `TURING_DOMAIN_GROWTH_DT` | （未設定時は上記） | 環境変数で上書き |

## 出力

- `results/turing_domain_growth_numerical.png`
  - `p` のカイモグラフ
  - $t=0,\;T/2,\;T$ のプロファイル（各タイトルに $\lambda_{\mathrm{num}}$）
- `results/turing_domain_growth_analysis.png`
  - 線形安定性曲線 $\sigma_{\max}(k)$ と $k_*$
  - $\lambda_*$ と $\lambda_{\mathrm{num}}(t)$ の時系列比較

## 出典

- Turing, A. M. (1952). *Phil. Trans. R. Soc. B*, 237, 37–72. <https://doi.org/10.1098/rstb.1952.0012>
- Gierer, A., & Meinhardt, H. (1972). *Kybernetik*, 12, 30–39. <https://doi.org/10.1007/BF00289234>
- Kondo, S., & Asai, R. (1995). A reaction-diffusion wave on the skin of the marine angelfish *Pomacanthus*. *Nature*, 376, 765–768. <https://doi.org/10.1038/376765a0>
- Miura, T., & Maini, P. K. (2004). Speed of pattern appearance in reaction-diffusion models: Implications in the pattern formation of limb bud mesenchyme cells. *Bulletin of Mathematical Biology*, 66, 627–649.
