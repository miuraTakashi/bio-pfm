# 2D 興奮性媒質（FitzHugh–Nagumo＋拡散、螺旋波）

対応スクリプト: [`fhn_excitable_media_2d.py`](fhn_excitable_media_2d.py)。1D 進行パルスは [`fhn_excitable_media_1d`](../fhn_excitable_media_1d/fhn_excitable_media_1d.md)。

## モデルの導出（現象論）

1D 興奮性媒質と同型の **活性化–抑制子＋拡散** 仮説を 2 次元に拡張する。

- **現象**: スパイラル波・ターゲットパターン。
- **仮説**: $u$ の等方拡散、$v$ は局所のみ；FHN 型局所反応。
- 2D ラプラシアンを加えた次節の系が、興奮性波の最小連続モデル。

## 支配方程式

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - \frac{u^3}{3} - v,\qquad
\frac{\partial v}{\partial t} = \varepsilon\,(u + \beta - \gamma v).
$$

## 出典（原著・標準文献）

- FitzHugh, R. (1961). *Biophysical Journal*, 1(6), 445–466. <https://doi.org/10.1016/S0006-3495(61)86902-6>
- Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse transmission line simulating nerve axon. *Proceedings of the IRE*, 50(10), 2061–2070. <https://doi.org/10.1109/JRPROC.1962.288235>
- 反応–拡散での螺旋波の数学: Winfree, A. T. (1980). *The Geometry of Biological Time*. Springer（総説的）。

## 数値計算スキーム

- $u$ の拡散: **半陰的**スペクトル法（実 FFT、`build_diffusion_multiplier_rfft2`）。
- 反応項・$v$ 方程式: **陽的オイラー**。
- 初期条件: 帯域制限ホワイトノイズ（多数の螺旋シード）。

## パラメータ一覧

| 識別子 | 既定（`simulate_fhn_2d` / `main`） | 意味 |
|--------|-----------------------------------|------|
| `n` | 384 | 格子一辺 |
| `Du` | 1.2 | $D_u$ |
| `dx` | 1.0 | 格子間隔（無次元） |
| `dt` | 0.06（`main`） | 時間刻み |
| `n_steps` | 10000（`main`） | ステップ数 |
| `eps` | 0.1 | $\varepsilon$ |
| `beta` | 0.7 | $\beta$ |
| `gamma` | 0.5 | $\gamma$ |

## 出力

- `results/fhn_excitable_media_2d.png` — 複数時刻の $u,v$ スナップショット
- `results/fhn_excitable_media_2d.gif` — $u$ のアニメーション（`FHN_EXCITABLE_MEDIA_2D_GIF_EVERY` / `FHN_EXCITABLE_MEDIA_2D_GIF_FPS` で調整可）

## 数理解析

- 拡散なし平衡はコード内 Newton 法（`compute_rest_state`）で求める。螺旋波の存在・安定性は **数値実験ベース**。

## 実行

```bash
cd python/1_continuous_scheme/1_1_diffusion_term/fhn_excitable_media_2d
python3 fhn_excitable_media_2d.py
```
