# Cahn–Hilliard 相分離・粗化（1D + 2D）

対応スクリプト:

| スクリプト | 内容 |
|------------|------|
| [`cahn_hilliard_coarsening.py`](cahn_hilliard_coarsening.py) | スピノーダル / 非スピノーダル 2 ケース |
| [`cahn_hilliard_uniform_supply.py`](cahn_hilliard_uniform_supply.py) | 均一供給付き変種（あれば） |

移植元: [`1.1.4.Cahn-Hilliard.nb`](../../../../Mathematica/1_continuous_scheme/1_3_biharmonic_term/cahn_hilliard_coarsening/1.1.4.Cahn-Hilliard.nb)

## モデルの導出（現象論）

- **現象**: 二成分流体の **相分離** とドメインの **粗化**（界面エネルギー最小化）。
- **仮説**: 秩序変数 $u$、化学ポテンシャル $\mu=-\varepsilon^2\nabla^2 u - f(u)$、$f(u)=u-u^3$；移動度 $M$。
- **保存則**: $\partial_t u = M\nabla^2\mu$（質量保存）。
- スピノーダル分解は $| \langle u\rangle_0 | < 1/\sqrt{3}$ で不安定帯が開く。

## 支配方程式

$$
\frac{\partial u}{\partial t} = M \nabla^2 \mu,\qquad
\mu = -\varepsilon^2 \nabla^2 u - (u - u^3).
$$

周期境界。半陰式スペクトル法（`rfft` / `rfft2`, `norm="ortho"`）は [`cahn_hilliard_coarsening.py`](cahn_hilliard_coarsening.py) 参照。

## 出典

- Cahn, J. W., & Hilliard, J. E. (1958). Free energy of a nonuniform system. I. Interfacial free energy. *J. Chem. Phys.*, 28(2), 258–267.
- **Mathematica**: 上記 `1.1.4.Cahn-Hilliard.nb`

## 数値計算スキーム

- 線形拡散項 $(\varepsilon^2 k^4)$ を陰的、非線形 $f(u)$ を陽的。
- 1D・2D を同一パラメータで並列比較（`run_case`）。

## パラメータ一覧（代表）

| 識別子 | 既定 | 意味 |
|--------|------|------|
| `e` / $\varepsilon$ | 0.5 | 界面幅 |
| `M` | 1.0 | 移動度 |
| `dt` | 0.005 | 時間刻み |
| `n_steps` | 120000 | ステップ数 |
| `u_mean` | 0.20 / 0.70 | スピノーダル / 非スピノーダル |

## 出力

- `results/cahn_hilliard_spinodal.png`, `results/cahn_hilliard_spinodal.gif`
- `results/cahn_hilliard_non_spinodal.png`, `results/cahn_hilliard_non_spinodal.gif`

## 実行

```bash
cd python/1_continuous_scheme/1_3_biharmonic_term/cahn_hilliard_coarsening
python3 cahn_hilliard_coarsening.py
```
