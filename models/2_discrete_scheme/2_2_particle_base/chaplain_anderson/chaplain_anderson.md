# Chaplain–Anderson（腫瘍侵襲）

**同梱スクリプト**: [`chaplain_anderson.py`](chaplain_anderson.py)

## モデルの導出（現象論）

**Chaplain–Anderson** は、腫瘍–免疫／組織の **粒子–場** モデル。

- **仮説**: 細胞を粒子、化学場 $c$ を連続；走化・増殖・死。
- **保存則**: 粒子数＋拡散–反応。
- 次節の連成 ODE/PDE–粒子更新。

## 概要

腫瘍細胞・基底質・尿激酶型プラスミノゲン活性化因子（uPA）の連成モデル（Mathematica ノートからの移植）。支配方程式・パラメータ・スキームは [`chaplain_anderson.py`](chaplain_anderson.py) 先頭 docstring を参照。

## Mathematica（移植元の想定パス）

`Mathematica/2_discrete_scheme/2_2_particle_base/chaplain_anderson/ChaplainAnderson.nb`（完全なアトラス配布ではこのノートと対をなす想定）。
