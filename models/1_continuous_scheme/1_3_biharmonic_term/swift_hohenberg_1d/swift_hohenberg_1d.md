# Swift–Hohenberg 1D

**同梱スクリプト**: [`swift_hohenberg_1d.py`](swift_hohenberg_1d.py)

## モデルの導出（現象論）

**Swift–Hohenberg** は、短波不安定を **勾配エネルギー** から導く 4 階 PDE。

- **仮説**: $F=\int (rac{r}{2}u^2 + rac{1}{2}(\Delta u)^2 + rac{g}{4}u^4)\,dx$ を梯度流 $\partial_t u = -L\delta F/\delta u$。
- **近似**: 長波長極限で Cahn–Hilliard 型から SH へ；Rayleigh–Bénard の amplitude 方程式。
- 双調和項 $(\Delta u)^2$ が次節の $ (
abla^2 + k_0^2)^2 u$ 形へ。

## 支配方程式（モデル族）

典型的には

$$
\frac{\partial u}{\partial t} = r u - (1+\partial_{xx})^2 u + f(u)
$$

などの **Swift–Hohenberg 型**（詳細はノートブック）。

## 出典（原著・標準文献）

- Swift, J., & Hohenberg, P. C. (1977). Hydrodynamic fluctuations at the convective instability. *Physical Review A*, 15(1), 319–328. <https://doi.org/10.1103/PhysRevA.15.319>
- Cross, M. C., & Hohenberg, P. C. (1993). Pattern formation outside of equilibrium. *Rev. Mod. Phys.*, 65, 851–1112. <https://doi.org/10.1103/RevModPhys.65.851>
- **Mathematica**: [`1.1.2.SHH1D.nb`](../../../../Mathematica/1_continuous_scheme/1_3_biharmonic_term/swift_hohenberg_1d/1.1.2.SHH1D.nb)

## 数値計算スキーム

[`swift_hohenberg_1d.py`](swift_hohenberg_1d.py) を参照。

## パラメータ一覧

[`swift_hohenberg_1d.py`](swift_hohenberg_1d.py) の docstring および関数引数を参照。

## 数理解析

分岐から縞・斑点パターンが生じる。`SwiftHohenberg-StripeSpot.nb` は記号分岐中心（[`../../README.md`](../../README.md) のスキップ表参照）。
