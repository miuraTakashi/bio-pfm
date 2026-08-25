# 2D 非圧縮流（MAC スタガード格子、mask 障害物）

対応スクリプト: [`flow.py`](flow.py)。

## モデルの導出（現象論）

- **現象**: 粘性が優勢な低 Reynolds 数流れで、圧力差により流体が移動する。
- **仮説**: 非圧縮（$\nabla\cdot\mathbf{u}=0$）、密度一定、ニュートン流体。
- **境界**: 上下壁は no-slip、左（上流）/右（下流）は定圧境界。
- **障害物**: [`masks/maskS.raw`](masks/maskS.raw)（ImageJ raw, uint8）を固体マスクとして使い、流速 0（no-flow）を課す。

## フォルダ構成

```
flow/
  flow.py                          # 本番シミュレーション
  verify_pipe_poiseuille.py        # Poiseuille 検証
  flow.md
  results/
    flow.png                         # 出力（障害物付き流れ）
    pipe_poiseuille_verification.png # 出力（検証用）
  masks/
    maskS.raw                      # 障害物マスク（既定）
    pipe.raw                       # 上壁のみ（検証用）
```

## 支配方程式

非圧縮 Navier–Stokes を MAC 法（圧力投影法）で離散化:

$$
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u},
\qquad
\nabla\cdot\mathbf{u}=0.
$$

本実装は **粘性優位** を重視し、既定では移流項を省いた Stokes 極限に近い更新:

$$
\mathbf{u}^{*} = \mathbf{u}^{n} + \Delta t\,\nu \nabla^2\mathbf{u}^{n},
\qquad
\nabla^2 p = \frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^{*},
\qquad
\mathbf{u}^{n+1} = \mathbf{u}^{*} - \frac{\Delta t}{\rho}\nabla p.
$$

## 数値計算スキーム

- **格子**: 128x128 セル中心圧力、速度はスタガード配置（`u`: x-face, `v`: y-face）。
- **Poisson**: 左右は Dirichlet（`p_left`, `p_right`）、上下は Neumann（$\partial p/\partial n=0$）。
- **固体境界（mask）**: 面フラックスを無効化し、障害物をまたぐ速度面を 0 に固定。
- **時間刻み**: 粘性安定条件 `dt ~ O(h^2 / nu)` を使用。

## 物理条件（既定）

| 量 | 値 |
|----|-----|
| 流体 | 水（20 °C 付近） |
| 密度 $\rho$ | 998 kg/m³ |
| 粘度 $\mu$ | $1.002\times10^{-3}$ Pa·s |
| 動粘性係数 $\nu=\mu/\rho$ | $\approx 1.004\times10^{-6}$ m²/s |
| 流路長 $L_x$ | 1 mm |
| 流路高さ $L_y$ | 1 mm |
| 圧力差 $\Delta p$ | 10 mmHg ($\approx 1333$ Pa) |

左右境界: $p_\mathrm{left}=\Delta p$, $p_\mathrm{right}=0$（ゲージ圧）。

## パラメータ一覧（環境変数）

| 変数 | 既定値 | 意味 |
|------|--------|------|
| `FLOW_NX`, `FLOW_NY` | `128`, `128` | 格子数 |
| `FLOW_LX`, `FLOW_LY` | `1e-3`, `1e-3` | ドメインサイズ [m] |
| `FLOW_RHO` | `998` | 密度 [kg/m³] |
| `FLOW_MU` | `1.002e-3` | 粘度 [Pa·s] |
| `FLOW_DP` | `1333.22` | 圧力差 [Pa]（10 mmHg） |
| `FLOW_P_LEFT`, `FLOW_P_RIGHT` | `FLOW_DP`, `0` | 左右境界圧力 [Pa] |
| `FLOW_N_STEPS` | `50000` | 時間ステップ数 |
| `FLOW_POISSON_ITER` | `350` | 圧力 Poisson 反復回数 |
| `FLOW_MASK_NAME` | `masks/maskS.raw` | 使用するマスクファイル（`flow/` からの相対パス） |
| `FLOW_MASK_SOLID_IS_DARK` | `0` | `1`: 暗画素を固体扱い |
| `FLOW_PRINT_EVERY` | `25` | 進捗表示間隔 |

## 実行

```bash
cd python/1_continuous_scheme/1_2_advection_term/flow
python3 flow.py
```

出力: [`results/flow.png`](results/flow.png)（速度大きさ、圧力、ベクトル場、流線場）。

Poiseuille 検証:

```bash
python3 verify_pipe_poiseuille.py
```

出力: [`results/pipe_poiseuille_verification.png`](results/pipe_poiseuille_verification.png)。

実行中は `[flow] ...` 行で進捗（% / 残り時間見積り / 発散ノルム）を標準出力に表示する。
