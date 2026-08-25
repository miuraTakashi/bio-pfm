# 群れ振動子（位相＋位置結合、トーラス上）

対応スクリプト: [`swarm_oscillators.py`](swarm_oscillators.py)。

## モデルの導出（現象論）

**群れ振動子** は、位相 $	heta_i$ と位置が耦合し **同期＋空間パターン** を示す。

- **仮説**: Kuramoto 型位相結合＋移動；または位置依存周波数。
- **近似**: 粒子 $N$ 個、近傍または全結合。
- 次節の更新則は swarm oscillators の現象論。

## 支配方程式（コードと同型）

粒子 $i$ の位相 $\psi_i$、位置 $\mathbf{r}_i\in[0,L)^2$（周期）。距離 $d_{ij}=\|\mathbf{r}_i-\mathbf{r}_j\|$、重み $w_{ij}=e^{-d_{ij}}$（対角 0）:

$$
\frac{d\psi_i}{dt} = \sum_j w_{ij}\sin\bigl(\psi_j-\psi_i + \alpha d_{ij} - c_1\bigr),
$$

$$
\frac{d\mathbf{r}_i}{dt} = c_3 \sum_j w_{ij}\sin\bigl(\psi_j-\psi_i + \alpha d_{ij} - c_2\bigr)\,\frac{\mathbf{r}_j-\mathbf{r}_i}{d_{ij}}.
$$

## 出典（原著・標準文献）

- **論文再現バッチ（`main()`）**: Iwasa, M., Iida, K., Tanaka, D. (2012). Various collective behavior in swarm oscillator model. *Physics Letters A*, 376(33), 2117–2121. <https://doi.org/10.1016/j.physleta.2012.05.025>  
  相 A–M の代表点 $(c_1,c_2)$、系サイズ $L=10$、粒子数 $N=50$、$c_3=\alpha=1$ は本文・Fig. 1–5 に合わせています。図表との対照用の PDF は手元で `References/SwarmOscillatorsPatterns.pdf` などのファイル名に保存して参照してください（**リポジトリには同梱されません**）。
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. In *International Symposium on Mathematical Problems in Theoretical Physics* (pp. 420–422). Springer. <https://doi.org/10.1007/BFb0013365>
- 群れ・Swarming の連続極限: Vicsek, T., et al. (1995). Novel type of phase transition in a system of self-driven particles. *Physical Review Letters*, 75(6), 1226–1229. <https://doi.org/10.1103/PhysRevLett.75.1226>
- Mathematica 移植元: [`SwarmOscillators8.nb`](../../../../Mathematica/2_discrete_scheme/2_2_particle_base/swarm_oscillators/SwarmOscillators8.nb)（`simulate_swarm()` のシグネチャ既定はノート側の別パラメータセットに近い値のままです）。

## 実行と出力

```bash
cd python/2_discrete_scheme/2_2_particle_base/swarm_oscillators
python swarm_oscillators.py
```

- **フル実行（既定）**: 相 **A–M** それぞれについて `swarm_oscillators_phase_<A..M>.gif` と、最終フレーム一覧 `results/swarm_oscillators_phases_A_M_grid.png` を書き出します（数分程度かかることがあります）。
- **短いテスト**: 環境変数 `SWARM_OSCILLATORS_QUICK=1`（`python/run_tests.sh` から設定）では相 **A** のみ・ステップ数短縮で終了します。

## 数値計算スキーム

- **時間**: 陽的オイラー。位相は $\bmod 2\pi$、位置は $\bmod L$。

## パラメータ一覧

### `simulate_swarm(...)` のキーワード既定（任意デモ・他スクリプトから呼ぶ場合）

| 識別子 | 既定 | 意味 |
|--------|------|------|
| `n` | 50 | 振動子数 |
| `L` | 50.0 | 領域一辺（周期） |
| `c1`, `c2` | 1.5, 0.5 | 位相・位置結合の位相シフト |
| `c3` | 2.0 | 位置更新のゲイン |
| `alpha` | 0.5 | 距離依存位相 $\alpha d$ |
| `dt` | 0.05 | 時間刻み |
| `n_steps` | 5000 | ステップ数 |
| `snapshot_interval` | 1 | スナップショット間隔 |
| `seed` | 42 | RNG |

### `main()` が呼ぶ論文再現（`generate_iwasa_et_al_phase_gifs` の主な既定）

| 識別子 | 値 | 意味 |
|--------|-----|------|
| `n`, `L` | 50, 10.0 | Iwasa et al. (2012) 本文 |
| `c3`, `alpha` | 1.0, 1.0 | 同上 |
| `c1`, `c2` | 相ごと（ソース内 `IWASA_ET_AL_PHASES`） | Fig. 1–5 の代表点 |
| `n_steps` / `snapshot_interval` | 6000 / 20 | GIF 用（変更可） |

## 数理解析

- 位相同期の秩序変数 $R=|N^{-1}\sum e^{i\psi}|$ 等は本コードでは計算していない。**パターンの数値観察**が主。
