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

## 出典（原著・標準文献）

- Witten, T. A., & Sander, L. M. (1981). Diffusion-limited aggregation, a kinetic critical phenomenon. *Physical Review Letters*, 47(19), 1400–1403. <https://doi.org/10.1103/PhysRevLett.47.1400>
- **Mathematica（移植元）**: [`2.1.4.DLA.nb`](../../../../Mathematica/2_discrete_scheme/2_1_lattice_base/dla/2.1.4.DLA.nb)

### 公開実装との類似（コード出典の注記）

本スクリプトは上記 Mathematica ノートからの移植（200×200 格子・中央種・周期境界ウォーカー）であり、下記のソースを転載したものではない。格子上ランダムウォーク＋隣接付着は公開 DLA デモとアルゴリズムが近い。

- `ratwolfzero/DLA`（中央種＋格子ウォーカー）: <https://github.com/ratwolfzero/DLA>
