# 修正 Keller–Segel（1D）

**同梱スクリプト**: [`keller_segel.py`](keller_segel.py)

## モデルの導出（現象論）

**走化性** 細菌集団では、密度 $n$ と attractant $c$ が耦合する。

- **仮説**: $n$ の Fick 拡散＋ chemotactic drift $\chi n 
abla c$；$c$ は拡散・分解。
- **保存則**: $\partial_t n + 
abla\cdot(-D_n
abla n + \chi n
abla c) = 0$。
- Keller–Segel 型が次節の支配方程式（凝集・パターン）。

## 支配方程式・離散化

血管網形成と走化性の修正 Keller–Segel 系（1D・周期境界・陽的オイラー）。式と格子通量の定義は [`keller_segel.py`](keller_segel.py) 先頭 docstring に Mathematica ノート（`modifiedKellerSegel.nb`）との対応付きで記載。

## 出典

### アトラス内の移植元

- **Mathematica（移植元の想定パス）**: `Mathematica/1_continuous_scheme/1_2_advection_term/keller_segel/血管網形成と走化性：modifiedKellerSegel.nb`（完全なアトラス配布ではこのノートと対をなす想定）

### 文献（モデルの数学的根拠）

- **Keller, E. F. & Segel, L. A.** “Initiation of slime mold aggregation viewed as an instability.” *Journal of Theoretical Biology* **26**, 399–415 (1970). <https://doi.org/10.1016/0022-5193(70)90092-5>  
  走化性と拡散を結合した古典的パラドックス系の出発点（不安定性による凝集のイメージ）。
- **Keller, E. F. & Segel, L. A.** “Model for chemotaxis.” *Journal of Theoretical Biology* **30**, 225–234 (1971). <https://doi.org/10.1016/0022-5193(71)90050-6>  
  最小化された Keller–Segel 型の連続モデル。
- **Hillen, T. & Painter, K. J.** “A user’s guide to PDE models for chemotaxis.” *Journal of Mathematical Biology* **58**, 183–217 (2009). <https://doi.org/10.1007/s00285-008-0201-3>  
  Keller–Segel 型の諸変種（体積充填・飽和感度などを含む）を体系的に整理した総説。本スクリプトの \(\chi(u)=u(u_s-u)\) は、定数感度の線形化系とは異なる**非線形走化項**に属し、生物学的モチーフ（飽和・上限）と数値的安定性の両面でよく用いられる枠組みに位置づけられる。
- **Murray, J. D.** *Mathematical Biology II: Spatial Models and Biomedical Applications*（3rd ed., Springer, 2003）. <https://doi.org/10.1007/b98869> — 走化性・パターン形成を含む反応拡散・移流拡散の応用数学の標準的整理。

連続極限での線形安定解析（`dispersion_matrix` 等）は、上記の線形化 Keller–Segel 文献と同型の Fourier モード解析に相当する。

### 公開実装との類似（コード出典の注記）

本スクリプトは Mathematica ノート `modifiedKellerSegel.nb` 由来の **修正** Keller–Segel（走化感度 $\chi(u)=u(u_s-u)$）であり、下記の古典 KS ソルバを転載したものではない。1D・周期・`np.roll` によるフラックス離散化は公開実装と型が近い。

- Liu, Z. `15.KellerSegelModel.py`（定数 $\chi_0$ の古典 1D KS）: <https://github.com/Liu-Zhihang/Nonequilibrium-Field-Theories-and-Stochastic-Dynamics/blob/main/code/15.KellerSegelModel.py>

## パラメータ

`noiseAmp`, `dx`, `domainSize`, `du`, `dv`, `c0`, `u0`, `us`, `dt`, 線形反応係数 `fu`, `fv`, `gu`, `gv`, `n_steps` 等。詳細はソース。
