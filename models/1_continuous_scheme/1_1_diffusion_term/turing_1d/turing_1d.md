# turing_1d — 1D 周期境界の反応–拡散

対応スクリプト: [`turing_1d.py`](turing_1d.py)。

移植元: [`1.1.2.軟骨形成とTuringパターン.nb`](../../../../Mathematica/1_continuous_scheme/1_1_diffusion_term/turing_1d/1.1.2.軟骨形成とTuringパターン.nb) の **1D 静領域** 部分。

## モデルの導出（現象論）

反応拡散系では、**局所反応**（自己活性化・抑制）と **Fick 拡散** を組み合わせる。

- **現象**: 均一定常解から空間パターン（斑点・縞）が自発形成（Turing 不安定性）。
- **仮説**: 可塑素 $u$ と抑制子 $v$ が拡散し、$v$ の拡散が $u$ より速い（$D_u < D_v$）。
- **保存則**: 各化学種の質量保存 → $\partial_t u = f(u,v) + D_u u_{xx}$ 型。
- 次節の Brusselator / Gray–Scott 型はこの骨格の具体例。

## 支配方程式

$$
\frac{\partial p}{\partial t} = D_p \Delta p + f(p,q),\quad
\frac{\partial q}{\partial t} = D_q \Delta q + g(p,q),
$$

周期境界。$f(p,q)=0.6p-q-p^3$, $g(p,q)=1.5p-2q$。

## Gierer–Meinhardt 系（追加の数値例）

スクリプトは **別系**として、活性子 $p$、抑制子 $q$ の古典的 GM 型

$$
f_{\mathrm{GM}}(p,q)=\rho_0+\frac{\rho\,p^2}{q+\varepsilon}-\mu p,\qquad
g_{\mathrm{GM}}(p,q)=\sigma p^2-\nu q
$$

を同じ周期領域で積分する（係数はソース内の `GM_*` 定数）。空間一様な正の定常点 $(p^*,q^*)$ は

$$
p^*=\frac{\rho_0+\rho\,\nu/\sigma}{\mu},\qquad
q^*=\frac{\sigma}{\nu}(p^*)^2
$$

で与えられ、線形安定性の $J$ は $(p^*,q^*)$ で評価する。分母の $\varepsilon$ は数値用の正則化。

## 数値スキーム

離散ラプラシアン（`np.roll`）＋陽的オイラー。

## 線形安定性と初期波長推定

原点まわりのヤコビアン

$$
J(0,0)=\begin{pmatrix}0.6 & -1\\1.5 & -2\end{pmatrix},
\quad
D=\mathrm{diag}(D_p,D_q)
$$

を用いて

$$
A(k)=J-k^2D,\qquad
\sigma_{\max}(k)=\max\operatorname{Re}\lambda(A(k))
$$

を掃引し、最不安定モード $k_*$ と初期波長

$$
\lambda_*=\frac{2\pi}{k_*}
$$

を推定する。コードでは最終 $p(x)$ の FFT から得る支配的波長 $\lambda_{\mathrm{num}}$ も併記する。

**GM 図**では $J$ を $(p^*,q^*)$ におけるヤコビアンに取り替えた同じ $\sigma_{\max}(k)$ を用いる。

## 出力

- `results/turing_1d_numerical.png`
  - 最終プロファイル $p,q$（タイトルに $\lambda_*$, $\lambda_{\mathrm{num}}$）
  - $p,q$ のカイモグラフ
- `results/turing_1d_analysis.png`
  - $\sigma_{\max}(k)$ と $k_*$, $k_{\mathrm{num}}$
- `results/turing_1d_numerical_GiererMeinhardt.png` / `results/turing_1d_analysis_GiererMeinhardt.png` — 上記 GM 反応項・均一解まわりの線形解析

## 出典

- Turing, A. M. (1952). The chemical basis of morphogenesis. *Phil. Trans. R. Soc. B*, 237, 37–72. <https://doi.org/10.1098/rstb.1952.0012>
- Gierer, A., & Meinhardt, H. (1972). A theory of biological pattern formation. *Kybernetik*, 12, 30–39. <https://doi.org/10.1007/BF00289234>
