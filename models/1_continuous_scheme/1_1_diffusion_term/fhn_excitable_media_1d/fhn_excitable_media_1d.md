# 1D 興奮性媒質（FitzHugh–Nagumo＋拡散、進行パルス）

対応スクリプト: [`fhn_excitable_media_1d.py`](fhn_excitable_media_1d.py)。

[`fhn_excitable_media_2d`](../fhn_excitable_media_2d/fhn_excitable_media_2d.md) と同型の FHN 反応・平衡の取り方に対し、空間は 1 次元・**t=0 で領域中央にガウス刺激**を与え、周期境界上で左右に分かれて進むパルス対を誘起する。

## モデルの導出（現象論）

心筋・神経など **興奮性媒質** では、閾値超えの活性化変数 $u$ と遅い抑制 $v$ が **進行パルス** を担う。

- **仮説**: 局所ダイナミクスは FitzHugh–Nagumo 型（立方 $u$ 非線形＋線形 $v$）。
- **拡散**: 速い変数 $u$ のみ Fick 拡散 $D_u u_{xx}$（$v$ は局所）。
- **保存則**: $u$ の拡散フラックス $J=-D_u \partial_x u$ を連続の式に代入 → 次節の RD。

## 支配方程式

$$
\frac{\partial u}{\partial t} = D_u \frac{\partial^2 u}{\partial x^2} + u - \frac{u^3}{3} - v,\qquad
\frac{\partial v}{\partial t} = \varepsilon\,(u + \beta - \gamma v).
$$

## 出典（原著・標準文献）

- FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal*, 1(6), 445–466. <https://doi.org/10.1016/S0006-3495(61)86902-6>
- Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse transmission line simulating nerve axon. *Proceedings of the IRE*, 50(10), 2061–2070. <https://doi.org/10.1109/JRPROC.1962.288235>
- 興奮性媒質の進行波: 総説的に Murray, J. D. (2002). *Mathematical Biology I: An Introduction* (3rd ed.). Springer. <https://doi.org/10.1007/b98868>

## 数値計算スキーム

- $u$ の拡散: **半陰的**スペクトル法（周期 3 点ラプラシアンを `fft` / `ifft` で逆算、`build_diffusion_multiplier_fft1d`）。`norm='ortho'` の `rfft` で除算するとスケールが合わず構造が潰れるため、標準 `fft(kern)` の固有値で除算する。
- 反応項・$v$ 方程式: **陽的オイラー**（2D 版と同順序）。
- 初期条件: 一様静止状態に中央のガウス上乗せ（`initial_center_stimulus`）。
- 可視化: $u(x,t)$ のカイモグラフと、複数時刻の $u(x)$ プロファイル。既定の実時間は **$T=200$**（`n_steps * dt`）。

## パラメータ一覧

| 識別子 | 既定（`simulate_fhn_1d` / `main`） | 意味 |
|--------|-----------------------------------|------|
| `nx` | 640 | 格子点数 |
| `Du` | 1.2 | $D_u$ |
| `dx` | 1.0 | 格子間隔（無次元） |
| `dt` | 0.1 | 時間刻み |
| `n_steps` | 2000 | ステップ数（`dt=0.1` で実時間 $T=200$） |
| `record_every` | 10 | カイモグラフ用の記録間隔（ステップ） |
| `eps` | 0.1 | $\varepsilon$ |
| `beta` | 0.7 | $\beta$ |
| `gamma` | 0.5 | $\gamma$ |

## 数理解析

- フロント部を inhibitor 固定（`v ≈ v_rest`）の Nagumo 近似で扱い、
  $$
  u_t = D_u u_{xx} - \frac13 (u-r_1)(u-r_2)(u-r_3)
  $$
  に対する速度近似
  $$
  c_{\mathrm{ana}} \approx \sqrt{D_u/6}\,(r_1+r_3-2r_2)
  $$
  （$r_i$ は $u-u^3/3-v_{rest}=0$ の実根）を用いる。
- 数値側は $u=0$ の右向きフロント交差位置 $x_f(t)$ を追跡し、線形フィット傾き
  $c_{\mathrm{num}}$ を推定。
- 出力を分離:
  - `results/fhn_excitable_media_1d_numerical.png`（カイモグラフ + 複数時刻プロファイル）
  - `results/fhn_excitable_media_1d_analysis.png`（フロント速度の数値 vs 解析比較）

## 実行

```bash
cd python/1_continuous_scheme/1_1_diffusion_term/fhn_excitable_media_1d
python3 fhn_excitable_media_1d.py
```

- 拡散なし平衡は `compute_rest_state`（Newton 法）。進行パルスの速度・形状は **数値実験ベース**。
