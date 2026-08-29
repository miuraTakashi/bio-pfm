# 3D Turing patterns — Shoji et al. (2007)

対応スクリプト: [`Turing_3D.py`](Turing_3D.py)  
原著: Shoji, Yamada, Ueyama & Ohta, “Turing patterns in three dimensions.” *Phys. Rev. E* **75**, 046212 (2007).
[DOI: 10.1103/PhysRevE.75.046212](https://doi.org/10.1103/PhysRevE.75.046212)

3 次元周期箱上で **FitzHugh–Nagumo**・**Brusselator**・**Gray–Scott** の 2 成分反応–拡散を数値積分し、ラメラ・gyroid・BCC/FCC などの定常 Turing 構造を再現する（[`turing_2d`](../turing_2d/README.md) と同型の **半陰式 FFT**）。

## 3D パターン名と幾何（プライマ）

Shoji et al. や本デモのギャラリーで使う名称は、**活性成分 $u$（またはその偏差）の空間配置の対称性・トポロジー** を指す。反応拡散の厳密な定常解は下の単純な式そのものではないが、**同名パターンの幾何イメージ**として次の周期関数 $\phi(\mathbf{x})$ の **$\phi=0$ 等値面**（または $\phi$ の極大・極小ドメイン）で理解できる。

![典型 3D パターンの $\phi=0$ 等値面と $z=L/2$ 断面（幾何プライマ）](results/turing_3d_geometry_primer.png)

生成: [`turing_3d_geometry_primer.py`](turing_3d_geometry_primer.py)（数値積分なし・フーリエ / TPMS の原型のみ）。

| 名称（本リポジトリ） | 原型 $\phi(\mathbf{x})$（$k=2\pi/L$） | 見え方 |
|----------------------|----------------------------------------|--------|
| **Lamella（ラメラ）** | $\phi=\cos(kz)$ | $z$ に垂直な層（板） |
| **Perforated lamella** | $\phi=\cos(kz)+\alpha\cos(kx)\cos(ky)$ | 層内に穴の開いた層 |
| **BCC** | $\phi=\cos(kx)+\cos(ky)+\cos(kz)$ | 立方系の斑点／ドメイン（体心立方モチーフ） |
| **FCC** | $\phi=\sum_{\langle111\rangle}\cos(\mathbf{k}\!\cdot\!\mathbf{x})$（4 本） | 面心立方型の対称性 |
| **Single gyroid** | $\phi=\sin kx\cos ky+\sin ky\cos kz+\sin kz\cos kx$ | 1 つの gyroid ネットワーク（TPMS） |
| **Double gyroid** | $\phi=\cos kx\sin ky+\cos ky\sin kz+\cos kz\sin kx$ | 二重ネットワーク gyroid の低モード近似 |
| **Diamond** | $\phi=\sin kx\sin ky\sin kz+\cos kx\cos ky\cos kz$ | Schwarz **D** 型 TPMS |
| **Fddd** | $\phi=\cos(kx)+\cos(1.31\,ky)+\cos(0.87\,kz)$ | 直方晶（非等方）の 3 波和 |

- **TPMS**（triply periodic minimal surface）: 周期境界内で $\phi=0$ が極小曲面に近い形状を与える曲面族。gyroid・diamond はその代表。
- **BCC / FCC**: 結晶構造の名前が付いた **波数ベクトルの対称性** の呼び方。2D の「斑点」と同様、3D では立方・面心立方のモジュレーションに対応する。
- 本 [`turing_3d.py`](turing_3d.py) の定常パターンは、$L,\beta,B,F$ と初期条件で上記のどれか（または変形）に **近い** 形になる（原著 Fig. 1）。

## モデルの導出（現象論）

3 次元領域でも **反応＋ラプラス拡散** の同型式が成り立つ。

- **現象**: 体積内の定常・時間発展パターン（ラメラ、ギロイド等の曲面）。
- **仮説**: 均質な等方拡散、体積反応のみ（流れ・成長なし）。
- 3D ラプラシアン $
abla^2$ を加えた次節の系は、Turing 機構の 3D 拡張。

## 支配方程式

**FHN** (Sec. II)

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - u^3 - v,\qquad
\frac{\partial v}{\partial t} = D_v \nabla^2 v + \gamma(u - \alpha v - \beta).
$$

**Brusselator** (Sec. III, near threshold)

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u + A - (B+1)u + u^2 v,\qquad
\frac{\partial v}{\partial t} = D_v \nabla^2 v + B u - u^2 v.
$$

**Gray–Scott** (Sec. IV)

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u - u v^2 + F(1-u),\qquad
\frac{\partial v}{\partial t} = D_v \nabla^2 v + u v^2 - (F+K)v.
$$

## 数値スキーム

| 項目 | 内容 |
|------|------|
| 拡散 | 6 近傍ラプラシアンを **FFT 半陰式**（`build_implicit_kernel_3d`） |
| 反応 | **陽的**オイラー（IMEX） |
| 境界 | 周期 |
| 初期条件 | FHN/Brusselator: 均一定常解 + 小乱；Gray–Scott: 中央立方体シード（原著 Sec. IV） |
| 時間刻み | 既定 `dt = 0.1`（原著は Eyre 型ソルバで `δt = 0.2`） |

## パラメータ（原著に基づく既定）

| モデル | 主な定数 |
|--------|----------|
| FHN | $D_u=5\times10^{-5}$, $D_v=5\times10^{-3}$, $\alpha=0.5$, $\gamma=26$, $\beta$ 可変, $N=32$, $L$ ケースごと |
| Brusselator | $D_u=2\times10^{-4}$, $D_v=1.6\times10^{-3}$, $A=2.7$, $B\approx 4$ |
| Gray–Scott | $D_u=2\times10^{-4}$, $D_v=10^{-4}$, $K=0.062$, $F$ 可変 |

## 出力

- `Turing_3D_fhn_gallery.png` / `Turing_3D_fhn_analysis.png`
- `Turing_3D_brusselator_gallery.png` / `Turing_3D_brusselator_analysis.png`
- `Turing_3D_gray_scott_gallery.png` / `Turing_3D_gray_scott_analysis.png`
- `Turing_3D_{fhn,brusselator,gray_scott}_surface_<slug>.gif` — $u$ 等値面の 360° 回転（[`Swift_Hohenberg_3d`](../Swift_Hohenberg_3d/README.md) と同型）
- `meshes/turing_3d_*_surface_<slug>.obj` — 同上メッシュ（3D Viewer 用）

等値面レベル（原著の可視化に合わせる）: FHN $u=0.05$, Brusselator $u=2.7$, Gray–Scott $u=0.5$。

ギャラリーは $u$ の **$z$ 中央断面**。解析図は均一定常解での $\sigma_{\max}(k)=\max\mathrm{Re}\,\lambda(J-k^2 D)$。

等値面メッシュは `scikit-image` の `marching_cubes` を推奨（未インストール時はボクセル境界フォールバック）。

## 実行

```bash
cd python/1_continuous_scheme/1_1_diffusion_term/turing_3d
python3 turing_3d.py
```

幾何プライマ図のみ再生成:

```bash
python3 turing_3d_geometry_primer.py
```

| 環境変数 | 意味 |
|----------|------|
| `TURING_3D_QUICK=1` | $N=24$, 2500 ステップ, 各モデル 2–3 ケース |
| `TURING_3D_NOISE` | 各ステップの一様乱数強度（原著の $\sigma$ に相当；既定 0） |
| `TURING_3D_SKIP_ISOSURFACE=1` | 等値面 GIF・OBJ をスキップ |
| `TURING_3D_SKIP_OBJ=1` | OBJ のみスキップ（GIF は生成） |

## 注意

原著は系サイズ $L$ とセル幅 $\delta x$ を系統的にスキャンして最適な周期を選んでいる。本デモは表に示した代表 $(L,\beta)$ または $(L,B,F)$ の一点計算であり、**同じパラメータでも $L$ やシードで別構造（BCC vs FCC など）が出る**ことがある（原著 Fig. 1 参照）。

## 依存

`numpy`, `matplotlib`, **`scikit-image`**（等値面推奨、`python/requirements.txt`）。
