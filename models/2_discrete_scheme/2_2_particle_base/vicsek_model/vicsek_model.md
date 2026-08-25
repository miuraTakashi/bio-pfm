# Vicsek model（自己駆動粒子の配向秩序化）

対応スクリプト: [`vicsek_model.py`](vicsek_model.py)。

## モデルの導出（現象論）

**Vicsek モデル** は、自己推進粒子が **速度整列** で collective motion を示す。

- **仮説**: 各粒子は速さ $v_0$ 一定、方向は近傍平均方向＋ノイズ。
- **保存則**: 粒子数保存；運動量は一般に保存しない。
- 次節の離散更新が active matter の最小モデル。

## 離散モデル

粒子 $i$ の位置 $\mathbf{r}_i$、向き $\theta_i$ を周期境界上で更新:

$$
\theta_i(t+1)=\mathrm{Arg}\left(\sum_{j\in N_i} e^{i\theta_j(t)}\right)+\xi_i(t),\quad
\xi_i\sim \mathrm{Uniform}[-\eta,\eta],
$$
$$
\mathbf{r}_i(t+1)=\mathbf{r}_i(t)+v_0\,(\cos\theta_i,\sin\theta_i)\,\Delta t.
$$

ここで $N_i$ は半径 $r_0$ 以内の近傍（最小像で計算）。

## 出典（原著）

- Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995). Novel type of phase transition in a system of self-driven particles. *Physical Review Letters*, 75(6), 1226–1229. <https://doi.org/10.1103/PhysRevLett.75.1226>

## 数値計算スキーム

- 時間離散（Vicsek の標準更新則）
- 距離判定は $O(N^2)$ の全対全探索（小規模デモ）

## パラメータ一覧（`simulate_vicsek` 既定）

| 識別子 | 値 | 意味 |
|---|---:|---|
| `n_particles` | 300 | 粒子数 |
| `L` | 20.0 | 周期領域サイズ |
| `v0` | 0.2 | 自己推進速度 |
| `r0` | 1.0 | 配向近傍半径 |
| `eta` | 0.5 | 角度ノイズ振幅 |
| `dt` | 1.0 | 時間刻み |
| `n_steps` | 600 | ステップ数 |
| `seed` | 42 | RNG シード |

## 出力

- `results/vicsek_model.png` — **ノイズ $\eta$ を変えた4レジーム比較**（同じ初期乱数・粒子数）
  - $\eta=0$: 秩序相（集団的に同じ向き、$\Phi\to 1$）
  - $\eta=0.5$: 遷移付近（高秩序だがゆらぎあり）
  - $\eta=1.0$: 中間（部分秩序）
  - $\eta=2.0$: 無秩序相（$\Phi$ が低いまま）
  - 上段: 最終配置（色＝向き $\theta$、矢印＝速度方向）
  - 下段: $\Phi(t)=|N^{-1}\sum_j e^{i\theta_j}|$
- `results/vicsek_model_ordered.gif` — $\eta=0$（秩序相）
- `results/vicsek_model_transition.gif` — $\eta=0.5$
- `results/vicsek_model_partial.gif` — $\eta=1.0$
- `results/vicsek_model_disordered.gif` — $\eta=2.0$
- `results/vicsek_model.gif` — `results/vicsek_model_ordered.gif` のコピー（後方互換）

## 実行

```bash
cd python/2_discrete_scheme/2_2_particle_base/vicsek_model
python3 vicsek_model.py
```

| 環境変数 | 意味 |
|----------|------|
| `VICSEK_QUICK=1` | 2 レジーム（$\eta=0,2$）、短縮ステップ |

