# phase_field_interdigitation — interdigitation のフェーズフィールド（2 場、標準 Allen–Cahn 形）

対応スクリプト: [`phase_field_interdigitation.py`](phase_field_interdigitation.py)。

## モデルの導出（現象論）

2 相の **相互侵入（interdigitation）** は界面エネルギーと拡散のバランスで生じる。

- **仮説**: 2 つの秩序変数または多成分フェーズフィールド；同型接着差を界面張力で表す。
- **保存則**: 各成分の拡散＋界面項から Cahn–Hilliard / 多相 Allen–Cahn 型。
- 次節の式は、指状侵入パターンを再現するフェーズフィールドモデル。

## 支配方程式

**u** は [`phase_field_standard_model`](phase_field_standard_model/phase_field_standard_model.md) と同一の Allen–Cahn 標準形。**v** は拡散するシグナル場（interdigitation 固有）。

$$
\frac{\partial u}{\partial t}
= \frac{c_0^2}{\tau}\Delta u
+ \frac{1}{\epsilon^2\tau}\,u(1-u)\Bigl(u-\tfrac12 + \frac{\epsilon}{\sqrt{2}\,c_0}\,F(v)\Bigr),
$$

$$
\frac{\partial v}{\partial t} = (1-u) - v + d_v \Delta v.
$$

### 駆動項と旧 Mathematica 記法との対応

旧ノートブック形:

$$
\frac{\partial u}{\partial t} = u(1-u)\Bigl(u-\tfrac12 - \tfrac{0.1(0.44-v)}{\sqrt{2\sigma}}\Bigr) + \sigma\,\Delta u
$$

| 記号 | 意味 | 既定値（デモ） |
|------|------|----------------|
| `sigma` | **物理量**: u の拡散係数 `du = c0²/τ` | `0.0045` |
| `c0` | 速度スケール（`c0=√σ` when `τ=1`） | `√σ ≈ 0.0671` |
| `tau` | 時間スケール（フリーパラメータ） | `1.0` |
| `v_star` | 臨界濃度 | `0.44` |
| `F(v)` | 標準駆動 | `0.1\,(v - v_\*)` |

**物理式（数値計算・線形安定性で使用）**:

$$
\frac{\partial u}{\partial t}
= \sigma\,\Delta u
+ u(1-u)\Bigl(u-\tfrac12 + \frac{0.1\,(v-v_*)}{\sqrt{2\sigma}}\Bigr)
$$

**ε について**: 上の Allen–Cahn 標準形への書き換えでは ε は記法上のフリーパラメータ。本実装は物理式を直接積分するため **ε は安定性にも時間発展にも入らない**（標準形で ε=1, τ=1, c₀=√σ と同値）。

検算: `beta = ε/(√2 c0) = 1/(√2 √σ)` かつ `F = 0.1(v-v*)` より  
`beta·F = 0.1(v-v*)/√(2σ)` → 旧反応項と一致。

## phase_field_standard_model との関係

- **u 方程式**: `phase_field_standard_model.py` と同じ `_allen_cahn_reaction` 核を使用。
- **違い**: 本モデルでは駆動 `F` が **v に依存**（骨形成シグナル結合）。v 方程式は standard_model には無い追加場。
- **数値スキーム**: 反応項陽的 + `fftfreq` 半陰式拡散（standard_model と同型）。

## 出典（原著・標準文献）

### 頭蓋縫合の interdigitation（三浦ら）

- **Miura, T. *et al.*** “Mechanism of skull suture maintenance and interdigitation.” *Journal of Anatomy* **215**, 642–655 (2009). <https://doi.org/10.1111/j.1469-7580.2009.01148.x>
- **Yoshimura, K., Kobayashi, R., Ohmura, T., Kajimoto, Y. & Miura, T.** “A new mathematical model for pattern formation by cranial sutures.” *Journal of Theoretical Biology* **408**, 66–74 (2016). <https://doi.org/10.1016/j.jtbi.2016.08.003>
- **Shibusawa, N., Endo, Y., Morimoto, N., Takahashi, I. & Miura, T.** “Mathematical modeling of palatal suture pattern formation…” *Scientific Reports* **11**, 88255 (2021). <https://doi.org/10.1038/s41598-021-88255-y>
- **三浦岳**『発生の数理』第 9 章、京都大学学術出版会、2015 年。ISBN 978-4-87698-887-7。

### 界面の数学（一般）

- **Cahn, J. W. & Hilliard, J. E.** (1958). <https://doi.org/10.1063/1.1744102>
- **Allen, S. M. & Cahn, J. W.** (1979). <https://doi.org/10.1016/0001-6160(79)90196-2>

## 数値計算スキーム

- **空間**: 周期境界、`fftfreq` スペクトル法（`fft2` / `ifft2`）。
- **時間**: 反応項を陽的に足し込み後、拡散を周波数空間で除算（半陰式）。
- **次元**: 2D 本計算 + 1D 線形安定性解析（平坦中央バンド）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `c0` | `√0.0045` | 速度スケール（`du = c0²/τ`） |
| `tau` | `1.0` | 時間スケール |
| `sigma` | `0.0045` | **物理量**: u の拡散係数（**sharp limit: σ→0**） |
| `dv` | `0.1` | v の拡散係数 |
| `domain_size` | `10.0` | 領域一辺 L |
| `band_half_width` | `1.0` | バンド半幅 w（ギャップ幅 2w=2） |
| `dx`, `dy` | `0.1` | 空間刻み |
| `dt` | `0.5` | 時間刻み |
| `n_steps` | `10000` | 総ステップ |
| `seed` | `42` | 2D 初期帯ノイズ用 |

## 実装されている出力

1. **2D 時間発展 GIF**
   - `results/PhaseFieldInterdigitation_2D_u.gif`
   - `results/PhaseFieldInterdigitation_2D_v.gif`
2. **1D 線形安定性（平坦中央バンド、in-phase 枝）**
   - `results/PhaseFieldInterdigitation_linear_stability.png`

## 使い方

このフォルダで実行:

```bash
python3 phase_field_interdigitation.py
```

主なオプション:

- `--c0`, `--tau`, `--sigma`（`c0=√σ`, `tau=1`）
- `--dv`, `--dx`, `--dy`, `--dt`, `--n-steps`, `--seed`
- `--domain-size` 領域長 L（既定 10）
- `--band-half-width` バンド半幅 w（既定 1、ギャップ幅 2w）
- `--lamella-stability` 旧来の最適ラメラ解析（参考用）
- `--linear-stability-only` 2D をスキップし線形安定性 PNG のみ
- `--sigma-sweep` 線形安定性で σ スイープ（sharp limit σ→0）
- `--sigmas`, `--sigma-sweep-quick`
- `--output-dir` 出力先

例:

```bash
python3 phase_field_interdigitation.py --sigma 0.0045
python3 phase_field_interdigitation.py --linear-stability-only
python3 phase_field_interdigitation.py --linear-stability-only --sigma-sweep
python3 phase_field_interdigitation.py --lamella-stability --linear-stability-only
```

## 数理解析

### 平坦中央バンド（既定）

2D 計算と同型の **1D 断面（y 方向）** を基準に、バンドに沿った方向（x）の摂動 $\delta u,\delta v \propto e^{\lambda t + ikx}$ の線形安定性を評価する。

- **基準解**: 領域 $L=10$ 上の平坦中央バンド（$|y-L/2|<w$ で $u=0$、それ以外 $u=1$、$w=1$ → ギャップ幅 2）
- **定常連成**: `pseudotime_relax_band_1d` で $(u_0,v_0)$ に半陰式擬似時間緩和
- **in-phase 摂動**: $y=L/2$ に関する鏡映偶モードの $\max_k \mathrm{Re}\,\lambda(k)$
- **2D との対応**: 解析の $k$ は 2D の **界面蛇行方向（x）** の波数に対応
- **sharp interface limit**: 物理拡散 **σ→0** で in-phase 枝が不安定化しうる

### `PhaseFieldInterdigitation_linear_stability.png` の読み方

- 平坦バンド定常解近傍の **in-phase 分岐** の線形成長率（横軸 $k$ はバンドに沿った x 方向）。
- $\max_k \mathrm{Re}\,\lambda(k) > 0$ なら、その $k^*$ から最不安定波長 $2\pi/k^*$ を読み取れる。
- $\max_k \mathrm{Re}\,\lambda(k) < 0$ でも、2D 非線形過渡・有限振幅・ノイズ付き初期条件では模様が現れうる。

### 安定性解析手順（バンド、再現メモ）

1. `_build_band_initial_1d` で平坦バンド初期条件を設定。
2. `pseudotime_relax_band_1d` で定常 $(u_0,v_0)$ に緩和。
3. `_reflection_perm_center` により $y=L/2$ 偶対称 in-phase モードを抽出。
4. `dispersion_inphase_branch` で $k$ 掃引し PNG を出力。

### 参考: 最適ラメラ解析（`--lamella-stability`）

旧実装の 1D 周期ラメラ（骨幅 $W$、ギャップ $G$）に対する in-phase 安定性。2D バンド計算とは幾何が異なる。

1. `optimal_lamella_widths_v_balance` で $(W,G)$ を求める。
2. `pseudotime_relax_periodic_1d` で定常化。
3. ギャップ中点に関する in-phase 摂動を評価。
