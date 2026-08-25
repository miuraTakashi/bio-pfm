# Delta–Notch 側方抑制（2D 格子 ODE）

対応スクリプト: [`delta_notch.py`](delta_notch.py)（2D デモ）、[`delta_notch_1d.py`](delta_notch_1d.py)（1D 線形安定性の数値検証）。

## モデルの導出（現象論）

**Notch–Delta** 側方抑制は、隣接細胞間シグナルで **チェッカーボード／縞** が生じる。

- **仮説**: 各細胞が Notch 活性 $N_i$、Delta $D_i$ を持ち、近傍から Delta を受け取る。
- **構成**: $N$ は Delta により抑制、$D$ は $N$ により誘導（ lateral inhibition）。
- 離散更新則／ODE 近似が次節の Delta–Notch モデル。

## 支配方程式（細胞ごとの現象論モデル）

隣接細胞の Delta 和を $\sum_{\mathrm{nb}}\Delta_j$（nb = neighbors）とすると

$$
\frac{d\Delta}{dt} = a\Delta + b N - \Delta^3,\qquad
\frac{dN}{dt} = c\Delta + d N - N^3 + \alpha \sum_{\mathrm{nb}}\Delta_j.
$$

（コード変数名 `delta`, `notch`。）

## 出典（原著・標準文献）

- Collier, J. R., Monk, N. A. M., Maini, P. K., & Lewis, J. H. (1996). Pattern formation by lateral inhibition with feedback: a mathematical model of Delta-Notch intercellular signalling. *Journal of Theoretical Biology*, 183(4), 429–446. <https://doi.org/10.1006/jtbi.1996.0233>
- 発生における Notch: Artavanis-Tsakonas, S., Rand, M. D., & Lake, R. J. (1999). Notch signaling: cell fate control and signal integration in development. *Science*, 284(5415), 770–776. <https://doi.org/10.1126/science.284.5415.770>

## 数値計算スキーム

- **時間**: 陽的オイラー。
- **空間**: 周期境界 4 近傍（2D）。1D 検証は 2 近傍（下記）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `n` | 10 | 格子一辺の細胞数 |
| `a`, `b`, `c`, `d` | -2, -1, -1, -3 | 局所ダイナミクス係数 |
| `alpha` | 5.0 | 隣接 Delta 結合 $\alpha$ |
| `dt` | 0.01 | 時間刻み |
| `n_steps` | 1000 | ステップ数 |
| `seed` | 42 | 初期乱数 |

## 数理解析：相互作用項 $\alpha$ の線形安定性（1D）

側方抑制による増幅は Collier 等の線形解析で説明される。本リポジトリの 2D コードは **非線形立方項付きの数値デモ** に加え、1D 還元で $\alpha$ の臨界を予測・検証する（[`delta_notch_1d.py`](delta_notch_1d.py) → `results/delta_notch_1d_verification.png`, `results/delta_notch_1d_analysis.png`）。

### 1D モデル（周期鎖）

細胞 $i=0,\ldots,N-1$ に対し、隣接 2 細胞の Delta のみが Notch を活性化する（2D コードの 4 近傍を 1D に還元）:

$$
\frac{d\Delta_i}{dt} = a\Delta_i + b N_i - \Delta_i^3,
$$

$$
\frac{dN_i}{dt} = c\Delta_i + d N_i - N_i^3 + \alpha\bigl(\Delta_{i-1}+\Delta_{i+1}\bigr).
$$

周期境界 $\Delta_{-1}=\Delta_{N-1}$, $\Delta_N=\Delta_0$ 等。既定係数 $a=-2$, $b=-1$, $c=-1$, $d=-3$。**コントロールパラメータ**は隣接結合強度 $\alpha$。

### 一様定常解と線形化

立方非線形項は原点で消えるため、$(\Delta_i,N_i)=(0,0)$ は定常解の一つ。小摂動 $\delta\Delta_i$, $\delta N_i$ について、離散フーリエモード（$x_i=i$）

$$
\delta\Delta_i = \hat{\delta\Delta}\, e^{ikx_i}, \qquad \delta N_i = \hat{\delta N}\, e^{ikx_i}
$$

を代入すると、フーリエ振幅は

$$
\frac{d\,\hat{\delta\Delta}}{dt} = a\,\hat{\delta\Delta} + b\,\hat{\delta N},
$$

$$
\frac{d\,\hat{\delta N}}{dt}
= \bigl(c + \alpha e^{-ik} + \alpha e^{ik}\bigr)\,\hat{\delta\Delta} + d\,\hat{\delta N}
= \bigl(c + 2\alpha\cos k\bigr)\,\hat{\delta\Delta} + d\,\hat{\delta N}.
$$

（隣接和が $2\cos k$ 倍になる）。すなわち波数 $k$ に対するヤコビアン $J(k)$ の成分は

$$
J_{11}=a,\quad J_{12}=b,\quad J_{21}=c+2\alpha\cos k,\quad J_{22}=d.
$$

### 固有値と安定性条件

$$
\mathrm{tr}\,J(k)=a+d=-5,\qquad
\det J(k)=ad-bc-2\alpha b\cos k
=6+c+2\alpha\cos k
=5+2\alpha\cos k.
$$

（$ad-bc=6$, $b=-1$ を用いた）。2 次特性方程式の判別式は $\mathrm{tr}^2-4\det = 5 - 8\alpha\cos k$。

#### 長波長モード $k=0$

$$
\det J(0)=5+2\alpha>0 \quad (\alpha \gt -2.5),\qquad \mathrm{tr}<0.
$$

よって $k=0$ は **常に（線形意味で）安定**（両固有値の実部は負）。

#### 短波長側の不安定化

$\mathrm{tr} \lt 0$ のもとで **鞍型不安定**（一方の固有値の実部が正）になるのは $\det J(k) \lt 0$ のとき:

$$
5+2\alpha\cos k<0
\quad\Leftrightarrow\quad
\cos k<-\frac{5}{2\alpha}.
$$

$\alpha \gt 0$ では $\det J(k)$ は $\cos k$ の最小値 $k=\pi$ で最小になる。したがって **最初に失稳するモードはチェッカーボード $k=\pi$**（細胞間隔 2 の交互パターン）。

#### 臨界値

$$
\alpha_c=\frac{5}{2}=2.5.
$$

- $\alpha \lt \alpha_c$ … $k=0$, $k=\pi$ とも安定 → パターンなし（一様減衰）
- $\alpha = \alpha_c$ … $k=\pi$ で限界（$\lambda=0$）→ 臨界
- $\alpha \gt \alpha_c$ … $k=\pi$ 不安定 → **パターンあり**（最優先波数 $k_*=\pi$）

最不安定波数・波長:

$$
k_*=\pi,\qquad \lambda_*=\frac{2\pi}{k_*}=2
$$

（細胞インデックス単位）。

$\alpha \gt \alpha_c$ での最大実部成長率は $k=\pi$ で

$$
\sigma_{\max}(\alpha)=\max_{k}\operatorname{Re}\lambda_{\max}(k)
=\frac{5+\sqrt{8\alpha-5}}{2}
\quad (\alpha \ge \alpha_c).
$$

例: $\alpha=5$ → $\sigma_{\max}\approx 0.854$。

### 2D（4 近傍）との対応

[`delta_notch.py`](delta_notch.py) の 2D 4 近傍和は、波数 $(k,\ell)$ に対して

$$
J_{21}=c+4\alpha-4\alpha\left(\sin^2\frac{k}{2}+\sin^2\frac{\ell}{2}\right).
$$

（Mathematica `Delta-Notchによる側抑制.nb` の `Eigenvalues` セルと同型）。最も先に失稳するのは $k=\ell=\pi$ で

$$
\det=5-4\alpha,\qquad \alpha_c^{(2\mathrm{D})}=\frac{5}{4}=1.25.
$$

1D 数値検証は **2 近傍** 還元に合わせる（$\alpha_c=2.5$）。

### 1D 数値検証の設計

[`delta_notch_1d.py`](delta_notch_1d.py) では

1. $\alpha=2.0 \lt \alpha_c$ … 空間変動が減衰することを確認
2. $\alpha=2.5 = \alpha_c$ … 臨界付近
3. $\alpha=3.5$, $5.0 \gt \alpha_c$ … 交互パターン（$k=\pi$）が増幅することを確認

指標: 最終時刻の $\mathrm{std}(\Delta)$、隣接差分エネルギー $\sum_i(\Delta_{i+1}-\Delta_i)^2$、FFT 支配的波数。検証図の縦軸は全 $\alpha$ で共通スケール。
