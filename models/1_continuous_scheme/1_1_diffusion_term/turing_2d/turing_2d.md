# turing_2d — 2D 周期境界（半陰式 FFT）と **q u²** 項

対応スクリプト: [`turing_2d.py`](turing_2d.py)。

移植元: [`1.1.2.軟骨形成とTuringパターン.nb`](../../../../Mathematica/1_continuous_scheme/1_1_diffusion_term/turing_1d/1.1.2.軟骨形成とTuringパターン.nb) の **2D 静領域** を拡張。

## モデルの導出（現象論）

1D と同様、**局所反応＋異方性のない Fick 拡散** から 2 次元反応拡散系が得られる。

- **現象**: 均一場から 2D チューリングパターン（斑点・迷路・縞）。
- **仮説**: 短程活性化・長程抑制を 2 化学種の拡散係数差で表現。
- 質量保存から $\partial_t \mathbf{u} = \mathbf{f}(\mathbf{u}) + D
abla^2 \mathbf{u}$ が次節の形になる。

## 支配方程式

$$
\frac{\partial u}{\partial t} = D_u \Delta u + f(u,v),\qquad
\frac{\partial v}{\partial t} = D_v \Delta v + g(u,v),
$$

$$
f(u,v) = 0.6\,u - v + q\,u^2 - u^3,\qquad
g(u,v) = 1.5\,u - 2\,v.
$$

**q** は二乗項の係数（スカラー）。スクリプトでは **q = 0, 0.8, −0.8** の三場合を同じ乱数初期条件で数値積分する。  
抑制因子の場は **v**（パラメータ q と混同しない）。

上の **2D stripe** 行（1 行目の図）は $q=0$ と同型の $f = 0.6u - v - u^3$ を使用し、**$t=0,100,200$** の 3 枚を並べる（下段の 3 列と対応）。

## Gierer–Meinhardt 系（追加の 2D 数値例）

別系として、活性 $u$、抑制 $v$ に対し

$$
f_{\mathrm{GM}}(u,v)=\rho_0+\frac{\rho\,u^2}{v+\varepsilon}-\mu u,\qquad
g_{\mathrm{GM}}(u,v)=\sigma u^2-\nu v
$$

を **半陰式 FFT**（既存の stripe ケースと同じ離散化）で $T=200$ まで積分する。均一正の定常点 $(u^*,v^*)$ は turing_1d の $(p^*,q^*)$ と同じ解析式で与えられ、$(u^*,v^*)$ における $J$ から $\sigma_{\max}(k)$ を別図に保存する（係数は `turing_2d.py` の `GM_*`）。

## 線形化（数理）

空間一様な原点 $(u,v)=(0,0)$ では $q u^2$ は二次以上なので、反応項のヤコビアンは **q に依らない**:

$$
J(0,0)=\begin{pmatrix}0.6 & -1\\ 1.5 & -2\end{pmatrix}.
$$

固有値はコードで `numpy.linalg.eigvals` により計算し、図のタイトルに表示する。

さらに拡散を含む線形化として

$$
A(k)=J-k^2D,\qquad D=\mathrm{diag}(D_p,D_q)
$$

を用い、

$$
\sigma_{\max}(k)=\max\operatorname{Re}\lambda(A(k))
$$

から最不安定波数 $k_*$ と初期波長 $\lambda_*=2\pi/k_*$ を推定する。実装では最終 2D 場（`q=0` ケース）の等方平均パワースペクトルから支配的波数 $k_{\mathrm{spec}}$ も算出し、理論値と併記する。

## 数値スキーム・実時間

拡散は FFT 上の陰的カーネル、反応は陽的。積分の **実時間は $T=200$** に統一（`simulate_2d` と `q` スイープとも **`dt=0.1`**, `n_steps=2000`）。

## 出力

- `results/turing_2d_numerical.png`
  - 1 行目: `q=0`（stripe 型）の $t=0,100,200$
  - 2 行目: $q=0,0.8,-0.8$ の最終 $u$
- `results/turing_2d_analysis.png`
  - $\sigma_{\max}(k)$ と $k_*$, $k_{\mathrm{spec}}$（理論/数値の初期波長比較）
- `results/turing_2d_numerical_GiererMeinhardt.png` — GM の $u$ を $t=0,100,200$ で表示
- `results/turing_2d_analysis_GiererMeinhardt.png` — $(u^*,v^*)$ における $J$ と $k_{\mathrm{spec}}$

## 出典

- Turing, A. M. (1952). The chemical basis of morphogenesis. *Phil. Trans. R. Soc. B*, 237, 37–72. <https://doi.org/10.1098/rstb.1952.0012>
- Gierer, A., & Meinhardt, H. (1972). A theory of biological pattern formation. *Kybernetik*, 12, 30–39. <https://doi.org/10.1007/BF00289234>
