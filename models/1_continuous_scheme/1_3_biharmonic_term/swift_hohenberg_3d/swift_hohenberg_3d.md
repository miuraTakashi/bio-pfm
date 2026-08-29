# Swift–Hohenberg 3D

対応スクリプト: [`swift_hohenberg_3d.py`](swift_hohenberg_3d.py)

3 次元の Swift–Hohenberg / Landau–Brazovskii 型。2D 版と同型の **二次項** $q u^2$ 付きモデルを、周期境界・**半陰式 FFT**（`fftn` / `ifftn`）で積分する。

## モデルの導出（現象論）

3D Swift–Hohenberg は **3D 勾配エネルギー** の Euler–Lagrange 演化として導入される。

- **現象**: 3D 周期構造・局所化。
- **仮説**: 等方 $(
abla^2+k_0^2)^2$ 演算子。
- 次節の 3D 支配方程式は 1D/2D と同型の現象論的拡張。

## 支配方程式

$$
\frac{\partial u}{\partial t} = r\,u + q\,u^2 - (1+\nabla^2)^2 u - u^3
$$

$(1+\nabla^2)^2$ を展開すると 4 階の空間微分項（双調和型）を含む。$q=0$ で立方非線形のみの標準 SH に一致。

## 代表的な 3D パターン（本デモの初期条件）

| 初期条件 | 期待される構造 |
|----------|----------------|
| $\cos(2\pi z/L)$ 等 | **ラメラ**（層状） |
| $\sum_i \cos(k_i\cdot x)$, $q \gt 0$ | **BCC / 斑点**系 |
| gyroid 型 $\sum \cos\sin$ | **gyroid 型** TPMS に近いモチーフ |
| 3 方向の等角波数 | **hcp 型**の種 |
| $\cos(2\pi x/L)$ | **円筒／縞**配向 |
| basal 六角 + $z$ モジュレーション | **HCP** 型の種 |
| 4 本の〈111〉族平面波 | **FCC** 型の種 |
| ノイズ + 適当な $(r,q)$ | 自発的な斑点・ラメラ選択 |

## 数値スキーム

- **線形**: $r u - (1+\nabla^2)^2 u$ をフーリエ空間で陰的（$\hat u^{n+1} = (\hat u^n + \Delta t\,\widehat{NL}) / (1 - \Delta t\,L_{\mathrm{op}})$, $L_{\mathrm{op}} = r - (1-k^2)^2$）
- **非線形**: $q u^2 - u^3$ 陽的
- **境界**: 周期（3D FFT）

## パラメータ（フル `main()`）

| 識別子 | 値 |
|--------|-----|
| `n` | 72 |
| `L` | $6\pi$ |
| `r` | 0.17–0.28（ケースごと） |
| `q` | $-0.25$–$0.6$（ケースごと） |
| `dt` | 0.45 |
| `n_steps` | 1400 |

## 出力

- `results/swift_hohenberg_3d_gallery.png` — 9 条件の $z$ 中央断面ギャラリー
- `results/swift_hohenberg_3d_zero_surface_lamella_z.gif` — **$u=0$ 等値面**の 360° 回転（ラメラ）
- `results/swift_hohenberg_3d_zero_surface_bcc_spots.gif` — 同上（BCC / 斑点系）
- `results/swift_hohenberg_3d_zero_surface_gyroid.gif` — 同上（gyroid 種）
- `results/swift_hohenberg_3d_zero_surface_hcp.gif` — 同上（HCP / 六角密積型の種）
- `results/swift_hohenberg_3d_zero_surface_fcc.gif` — 同上（FCC / 面心立方型の種）
- `meshes/swift_hohenberg_3d_zero_surface_*.obj` — 上記 GIF と同じ $u=0$ 等値面メッシュ（[3D Viewer for VSCode](https://marketplace.visualstudio.com/items?itemName=slevesque.vscode-3dviewer) 等で回転プレビュー）
- `swift_hohenberg_3d_lamella.gif` — ラメラ例の $z$ 断面時間発展（フル run・`SH3D_SKIP_SLICE_GIF=1` で省略可）

等値面は `scipy.ndimage.gaussian_filter` で平滑化したうえで **`scikit-image` の `marching_cubes`（Lewiner）** で三角メッシュ化（未インストール時はボクセル境界のフォールバック）。GIF と OBJ は同一メッシュから生成する。

```bash
pip install scikit-image
# またはリポジトリの venv: python/.venv/bin/pip install -r python/requirements.txt
```

## 実行

```bash
cd python/1_continuous_scheme/1_3_biharmonic_term/Swift_Hohenberg_3d
python3 swift_hohenberg_3d.py
```

| 環境変数 | 意味 |
|----------|------|
| `SH3D_QUICK=1` | $n=48$, 600 ステップ, 6 ケース・等値面 GIF 48 フレーム |
| `SH3D_SKIP_ISOSURFACE=1` | 等値面回転 GIF・OBJ をスキップ |
| `SH3D_SKIP_OBJ=1` | 等値面 OBJ のみスキップ（GIF は生成） |
| `SH3D_SKIP_SLICE_GIF=1` | 断面ラメラ GIF をスキップ |

## 出典

- Swift & Hohenberg (1977). Hydrodynamic fluctuations at the convective instability. *Phys. Rev. A* **15**, 319–328.
- Cross & Hohenberg (1993). Pattern formation outside of equilibrium. *Rev. Mod. Phys.* **65**, 851–1112.
- 2D 実装: [`../Swift_Hohenberg_2d`](../Swift_Hohenberg_2d/README.md)

## 依存

`numpy`, `matplotlib`, **`scikit-image`**（等値面 GIF 推奨、`python/requirements.txt`）。
