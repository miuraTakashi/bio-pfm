# Diffusion-Limited Aggregation（DLA, 格子ベース）

**同梱スクリプト**

- [`dla.py`](dla.py) — 拡散限定凝集（DLA）

## モデルの導出（現象論）

**拡散限定凝集 (DLA)** は、ランダムウォーカーがクラスターに付着すると枝分かれが生じる。

- **仮説**: 粒子拡散＋界面での不可逆付着；Screening 効果。
- **近似**: 格子上の walker；連続極限は複雑。
- 次節のアルゴリズムは fractal クラスターの現象論。

## 概要・パラメータ

`200×200` 格子で中央種から樹状クラスタを形成します。詳細は `dla.py` 先頭
docstring と `run_dla()` を参照。出力 PNG はスクリプトと同じディレクトリに
保存されます。
時系列の成長過程は `results/dla.gif` として保存されます。

## Mathematica（参考）

`Mathematica/2_discrete_scheme/2_1_lattice_base/dla/2.1.4.DLA.nb`
