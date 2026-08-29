# 上皮頂点動力学（vertex model）簡易デモ

対応スクリプト: [`honda_vertex_dynamics.py`](honda_vertex_dynamics.py)。周期境界・T1 転移を含む。**詳細な式とアルゴリズム**は [2_3_edge_base/README.md](../README.md)（索引） を参照。

## モデルの導出（現象論）

**頂点モデル**（Honda 型）は、多角形細胞の **頂点位置** が力学平衡で更新される。

- **仮説**: 各辺に線張力、各頂点に角度・面積制約；準静的力学。
- **保存則**: 細胞数・トポロジー（T1 遷移）を明示。
- 次節の力の釣り合い／更新則が epithelial mechanics の現象論。

## 支配方程式（エネルギーと運動）

細胞 $\alpha$ の面積 $A_\alpha$、周長 $P_\alpha$、辺 $e$ の長さ $\ell_e$:

$$
E = \sum_e \lambda \ell_e
+ \sum_\alpha \Bigl[\frac{K_A}{2}(A_\alpha-A_0)^2 + \frac{K_P}{2}(P_\alpha-P_0)^2\Bigr]
+ \frac{K_{\mathrm{vol}}}{2}\Bigl(\sum_\alpha A_\alpha - L_x L_y\Bigr)^2.
$$

**過阻尼**: $\dot{\mathbf{r}} \propto -\nabla E$（明示的オイラー）。周期境界は最小像。

## 出典（原著・標準文献）

- Honda, H. (1978). Description of cellular patterns by Dirichlet domains: the two-dimensional case. *Journal of Theoretical Biology*, 72(3), 523–543. <https://doi.org/10.1016/0022-5193(78)90315-6>
- 上皮頂点モデルのレビュー的整理: Alt, S., Ganguly, P., & Salbreux, G. (2017). Vertex models: from cell mechanics to tissue morphogenesis. *Philosophical Transactions of the Royal Society B*, 372(1720), 20150520. <https://doi.org/10.1098/rstb.2015.0520>
- Farhadifar, R., et al. (2007). The influence of cell mechanics, cell–cell interactions, and proliferation on epithelial packing. *Current Biology*, 17(24), 2095–2104. <https://doi.org/10.1016/j.cub.2007.11.049>

## 数値計算スキーム

- Voronoi 由来の周期メッシュで初期化（既定 `layout="grid"`）。
- 各ステップで力を計算し頂点移動、短辺で T1 を試行（詳細は親ディレクトリの [README.md](../README.md)）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `HONDA_VERTEX_N_STEPS`（環境変数） | 3600 | ステップ数（`main`） |
| `HONDA_VERTEX_SNAPSHOT_EVERY` | 60 | スナップショット間隔 |
| `dt`（`main`） | 0.002 | 時間刻み |
| `lam` | 1.0 | 線張力 $\lambda$ |
| `K_A` | 6.0 | 面積弾性 |
| `K_vol` | 4.0 | 大域体積項 |
| `K_P` | 0.85 | 周長弾性 |
| `l_min_frac` | 0.038 | T1 辺長閾値の平均辺長に対する比 |
| `n_seeds` | 20 | Voronoi 種点数 |
| `layout` | `"grid"` | 初期配置モード |
| `Lx`, `Ly`（`simulate` 内） | 1.0, 1.0 | 周期領域 |

## 数理解析

- エネルギー最小構造や T1 ネットワークはパラメータとトポロジーに依存。本実装は **デモ用**で、実上皮の定量比較は主目的としない。
