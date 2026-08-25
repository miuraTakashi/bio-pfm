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
- **Horstmann, D.** “From 1970 until present: the Keller–Segel model in chemotaxis and its consequences.” *Jahresbericht der Deutschen Mathematiker-Vereinigung* **105**, 103–165 (2003). <https://doi.org/10.1365/s13291-003-0021-5>  
  数学的側面（整域・大域時間解・ブローアップ等）の総説。
- **Painter, K. J. & Hillen, T.** “Volume-filling and quorum-sensing in models for chemosensitive movement.” *Canadian Applied Mathematics Quarterly* **10**, 501–543 (2002).  
  感度係数を密度に依存させる（体積充填・飽和）ことで古典的ブローアップを緩和する枠組み。本スクリプトの $\chi(u)=u(u_s-u)$ は、定数感度の線形化系とは異なる**非線形走化項**に属し、生物学的モチーフ（飽和・上限）と数値的安定性の両面でよく用いられる。
- **Murray, J. D.** *Mathematical Biology I: An Introduction*（3rd ed., Springer, 2002）— 走化性・パターン形成を含む反応拡散・移流拡散の応用数学の標準的整理。

連続極限での線形安定解析（`dispersion_matrix` 等）は、上記の線形化 Keller–Segel 文献と同型の Fourier モード解析に相当する。

## パラメータ

`noiseAmp`, `dx`, `domainSize`, `du`, `dv`, `c0`, `u0`, `us`, `dt`, 線形反応係数 `fu`, `fv`, `gu`, `gv`, `n_steps` 等。詳細はソース。
