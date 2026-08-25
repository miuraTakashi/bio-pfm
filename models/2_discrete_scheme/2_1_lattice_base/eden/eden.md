# Eden 成長（格子ベース）

**同梱スクリプト**

- [`eden.py`](eden.py) — Eden モデル（近傍占有成長）

## モデルの導出（現象論）

**Eden 成長** は、格子界面のランダム付着でクラスターが拡大する。

- **仮説**: occupied セルに近接する empty セルが等確率で occupied へ。
- **保存則**: 粒子数単調増加（拡散なし）。
- 次節の確率更新は、KPZ/EW とは別 universal class の離散成長。

## 概要・パラメータ

`64×64` 格子の中心種から開始し、occupied に隣接する空セルをランダムに
埋める操作を反復します。詳細は `eden.py` 先頭 docstring と `run_eden()` を参照。
出力 PNG に加え、時系列アニメーション `results/eden.gif` も同じディレクトリに保存されます。

## Mathematica（参考）

`Mathematica/2_discrete_scheme/2_1_lattice_base/eden/2.1.3.Eden.nb`
