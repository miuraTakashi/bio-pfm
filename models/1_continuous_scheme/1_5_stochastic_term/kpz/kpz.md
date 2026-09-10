# KPZ

**同梱スクリプト**: [`kpz.py`](kpz.py)

## モデルの導出（現象論）

**KPZ** は、非線形表面成長 $ (
abla h)^2 $ による **アノマラス粗さ** の標準モデル。

- **仮説**: 斜面依存沉积速度；拡散平滑化＋白ノイズ。
- **スケーリング**: 1+1D で KPZ  universality（出典参照）。
- $\partial_t h = 
u
abla^2 h + rac{\lambda}{2}|
abla h|^2 + \eta$ が次節の形。

## 支配方程式（モデル族）

Kardar–Parisi–Zhang 界面成長:

$$
\frac{\partial h}{\partial t} = \nu \Delta h + \frac{\lambda}{2}(\nabla h)^2 + \eta(\mathbf{x},t).
$$

## 出典（原著・標準文献）

- Kardar, M., Parisi, G., & Zhang, Y.-C. (1986). Dynamic scaling of growing interfaces. *Physical Review Letters*, 56(9), 889–892. <https://doi.org/10.1103/PhysRevLett.56.889>
- **Mathematica**: [`KPZのコピー.nb`](../../../../Mathematica/1_continuous_scheme/1_5_stochastic_term/kpz/KPZのコピー.nb)（リポジトリ上のファイル名に従う）

### 公開実装との類似（コード出典の注記）

本スクリプトは上記 Mathematica ノートからの移植であり、下記ライブラリのソースを転載したものではない。1+1D KPZ の差分＋加法ノイズという構成は公開ソルバと近い。

- Zwicker, D. *py-pde* `KPZInterfacePDE`: <https://py-pde.readthedocs.io/en/latest/packages/pde.pdes.kpz_interface.html>

## 数値計算スキーム

[`kpz.py`](kpz.py) を参照（空間差分＋Euler–Maruyama 等）。

## 出力

- `results/kpz.png` — $h(x,t)$ カイモグラフ、複数時刻の $h(x)$、平均高さ $\langle h\rangle(t)$（界面の進行）、最終プロファイル、$\mathrm{Var}(h)$ と log-log スケーリング

## 実行

```bash
cd python/1_continuous_scheme/1_5_stochastic_term/KPZ
python3 kpz.py
```

| 環境変数 | 意味 |
|----------|------|
| `KPZ_QUICK=1` | 短縮積分（煙テスト） |

## パラメータ一覧

[`kpz.py`](kpz.py) の docstring および関数引数を参照。

## 数理解析

1+1 次元の動的スケーリング（1/3 指数など）が有名。詳細は KPZ 普遍級のレビューを参照。
