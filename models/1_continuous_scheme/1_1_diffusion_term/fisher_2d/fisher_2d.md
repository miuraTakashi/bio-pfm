# fisher_2d — 2D Fisher–KPP（半陰式スペクトル拡散）

**同梱スクリプト**: [`fisher_2d.py`](fisher_2d.py)

1D の [`fisher_1d`](../fisher_1d/fisher_1d.md) と同型の **Fisher–KPP** 反応項を、正方形領域の **2 次元** に拡張したデモです。

## モデルの導出（現象論）

1D Fisher–KPP と同様、**拡散＋ロジスティック成長** から 2D 進行波が得られる。

- **仮説**: 等方拡散、均質環境（$r,K$ 一定）。
- **現象**: 円形 front の拡大、最小波速度の選択。
- 2D ラプラシアンを加えた次節の式へ。

## 支配方程式

$$
\frac{\partial u}{\partial t} = D \Delta u + u(1-u).
$$

## 初期条件

領域中心の円盤内で $u=1$、それ以外 $u=0$（中心刺激）。等方な設定のため、外周に達するまではほぼ **同心円状の進行波面** が現れます（周期境界のため、波がトーラス上を一周すると干渉します）。

## 数値スキーム

- **境界**: 周期（`rfft2` / `irfft2` による半陰的 5 点ラプラシアン。`fhn_excitable_media_2d` と同型の乗数 `ku`）。
- **時間**: 反応項は陽的 Euler、拡散は陰的（スペクトル空間で解く）。

積分ステップ数は環境変数 **`FISHER_2D_N_STEPS`**（既定 `500`）で変更可能です。実行すると **`results/fisher_2d.png`** に加え **`results/fisher_2d.gif`** も出力されます。格子サイズは **`FISHER_2D_N`**（既定 `256`）、GIF の記録間隔は **`FISHER_2D_GIF_EVERY`**（未設定時は約 25 フレーム相当で自動）で調整できます。

## 出典（原著・標準文献）

[`fisher_1d.md`](../fisher_1d/fisher_1d.md) と同様（Fisher 1937; Kolmogorov–Petrovsky–Piskunov 1937 等）。

## パラメータ一覧

`n`, `D`, `dx`, `dt`, `n_steps`, `stimulus_radius`, `n_snapshots`, `gif_every` は [`fisher_2d.py`](fisher_2d.py) の `simulate_fisher_2d` および `main()` を参照。
