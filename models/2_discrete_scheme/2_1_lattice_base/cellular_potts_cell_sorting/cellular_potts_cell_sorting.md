# Cellular Potts 細胞選別（Graner & Glazier 1992, Fig. 1）

対応スクリプト:

| スクリプト | 内容 |
|------------|------|
| [`cellular_potts_cell_sorting.py`](cellular_potts_cell_sorting.py) | Fig. 1 型の選別時系列（メイン） |
| [`cpm_boundary_draw.py`](cpm_boundary_draw.py) | 格子スナップショット描画ユーティリティ |

## モデルの導出（現象論）

- **現象**: 異型接着性のある細胞が **homotypic 凝集** により選別される（Steinberg 差次接着の Potts 版）。
- **仮説**: 拡張 Potts 模型 — 格子 $\sigma_i$（細胞 ID）、型 $\tau(\sigma)$、界面エネルギー $J(\tau,\tau')$、面積制約 $\lambda(A_c-A_{\tau})^2$。
- **更新**: ランダムコピー試行 + Metropolis（温度 $T$）；1 MCS = $16 L_x L_y$ 試行。
- 詳細つり合いから平衡分布 $\propto e^{-E/T}$。

## 離散モデル（Hamiltonian）

$$
E = \sum_{\langle i,j\rangle,\,\sigma_i\neq\sigma_j} J(\tau(\sigma_i),\tau(\sigma_j))
+ \lambda \sum_{\text{cells }c} (A_c - A_{\tau(c)})^2\,\theta(A_{\tau(c)}).
$$

型: medium (M), dark (d), light (l)。論文値 $J_{dd}=2$, $J_{dl}=11$, $J_{ll}=14$, $J_{dM}=J_{lM}=16$, $\lambda=1$, $T=10$, $A\approx 40$。

## 出典

- Graner, F., & Glazier, J. A. (1992). Simulation of biological cell sorting using a two-dimensional extended Potts model. *Phys. Rev. Lett.*, 69(13), 2013–2016. <https://doi.org/10.1103/PhysRevLett.69.2013>
- PDF（同フォルダ）: `Graner and Glazier 1992 - Simulation of biological cell sorting using a two-dimensional extended Potts model.pdf`
- **Mathematica**: [`CellularPottsSorting2D.generated.nb`](../../../../Mathematica/2_discrete_scheme/2_1_lattice_base/cellular_potts_cell_sorting/CellularPottsSorting2D.generated.nb)

### 公開実装との類似（コード出典の注記）

本スクリプトは純 Python の再実装であり、下記フレームワークのソースを転載したものではない。$J$ の数値は Graner & Glazier (1992) の論文値で、同じ論文を再現する公開デモとパラメータが一致する。

- Artistoo *Back to the Classics: Cell Sorting*: <https://artistoo.net/explorables/Explorable-CellSorting.html>
- CompuCell3D 細胞選別チュートリアル（XML/Python フロントエンド）: <https://compucell3d.org/>

## 数値計算スキーム

- 純 Python CPM（CompuCell3D 不要）。
- パイプライン: 矩形同型塊 → 400 MCS 平衡化 → ランダム型割当 → 10000 MCS、表示前に $T=0$ で 2 MCS アニール。

## パラメータ一覧

| 環境変数 | 意味 |
|----------|------|
| `CELL_POTTS_MCS` | 総 MCS 数の上書き |
| `CELL_POTTS_FIG1_QUICK` | 短縮実行（CI / `run_tests.sh`） |
| `CELL_POTTS_USE_NUMBA` | `1` で Numba JIT 経路を使用（既定 `0`。未導入環境では自動で pure Python にフォールバック） |

## 出力

- `results/cellular_potts_cell_sorting_fig1.png`
- `results/cellular_potts_cell_sorting_fig1.gif`

## 実行

```bash
cd python/2_discrete_scheme/2_1_lattice_base/cellular_potts_cell_sorting
python3 cellular_potts_cell_sorting.py
```

関連: [Kawasaki–Ising 選別](../kawasaki_ising_cell_sorting/kawasaki_ising_cell_sorting.md)（格子スピン交換型の対照モデル）。
