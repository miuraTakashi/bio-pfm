# ブラウン運動（ランダムウォーク粒子デモ）

対応スクリプト: [`diffusion_particle.py`](diffusion_particle.py)。

## モデルの導出（現象論）

**Brown 粒子** のランダムウォークは、Fick 拡散の粒子実装。

- **仮説**: 各粒子が独立にランダム変位（$\Delta x \sim \sqrt{2D\Delta t}$）。
- **保存則**: 粒子数保存；密度の極限で $\partial_t u = D
abla^2 u$。
- 次節は Monte Carlo 拡散の離散モデル。

## 離散モデル

各粒子 $\mathbf{p}$ について

$$
\mathbf{p}_{t+1} = \mathbf{p}_t + s \cdot \mathbf{U},\quad U_i \sim \mathrm{Uniform}(-1,1).
$$

## 出典（原著・標準文献）

- Einstein, A. (1905). Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen. *Annalen der Physik*, 322(8), 549–560. <https://doi.org/10.1002/andp.19053220806>
- 教科書的整理: Risken, H. (1989). *The Fokker-Planck Equation*. Springer.

## 数値計算スキーム

- 離散時間の独立増分（Euler–Maruyama の拡散項なし・離散ノイズ版）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `n_particles` | 200 | 粒子数 |
| `n_steps` | 500 | ステップ数 |
| `step_size` | 0.02 | スケール $s$ |
| `seed` | 42 | RNG シード |

## 数理解析

- 長時間・多粒子で均一なら、均二乗変位は $\langle |\mathbf{p}-\mathbf{p}_0|^2\rangle \propto t$（拡散係数は $s$ と分布に依存）。
- `diffusion_particle.py` の出力図は、**全粒子軌跡**（x-y）と **MSD 平均**（時間）を並べて表示。
