# Swift–Hohenberg 2D

**同梱スクリプト**: [`swift_hohenberg_2d.py`](swift_hohenberg_2d.py)

## モデルの導出（現象論）

2D でも **勾配エネルギー＋物質保存（または非保存 SH）** から 4 階 RD 型が得られる。

- **現象**: 縞・六角・局所化パターン。
- **仮説**: 等方双調和；パラメータ $r$ で均一解の安定性が変わる。
- 次節の 2D Swift–Hohenberg 方程式へ。

## 支配方程式（モデル族）

2D Swift–Hohenberg 型（$(1+\nabla^2)^2$ 項を含む 4 階系の変種）。スクリプトでは **二次項** $q u^2$ を付加した形

$$
\frac{\partial u}{\partial t} = r\,u + q\,u^2 - (1+\nabla^2)^2 u - u^3
$$

を半陰式 FFT で積分する（$q=0$ で従来の立方のみの SH に一致）。

## stripe–spot 選択（解析と数値）

- **勾配流**: 上式は Lyapunov 汎関数

  $$
  E[u]=\int\left[\frac{1}{2}\bigl((1+\nabla^2)u\bigr)^2-\frac{r}{2}u^2-\frac{q}{3}u^3+\frac{1}{4}u^4\right]\mathrm{d}x
  $$

  の勾配流 $\partial_t u=-\delta E/\delta u$ として書ける。
- **対称性**: $q\neq 0$ で $u\mapsto -u$ が破れるため、弱非線形で六角（斑点系）振幅に二次項が入りやすく、一方単一 roll 縞の主モード $\cos(kx)$ では空間平均で $\int\cos^3=0$ となり同次の二次結合が弱い、という標準的なパターン選択のメカニズム（Cross–Hohenberg 総説の枠組み）がある。
- **半解析境界**: 単一モード縞 $u=A\cos(k\cdot x)$ と、$120^\circ$ の 3 本の単位波数の重ね合わせ $\psi_{\mathrm{hex}}$ に対する 1 自由度変分で $E$ を最小化し、$E_{\mathrm{roll}}^*=E_{\mathrm{hex}}^*$ となる $(r,q)$ 曲線を走査により求め、数値フェーズ図に重ね描きする（厳密な分岐解析ではなく **試行関数近似**）。
- **数値フェーズ図**: $|k|\approx 1$ のリング上のパワーの角二重モーメントで「縞寄り / 斑点寄り」を可視化し、小振幅理論のスケーリング $|q|\sim\sqrt{r}$ に合わせた最小二乗曲線を参考表示。

## 出典（原著・標準文献）

- Swift & Hohenberg (1977). Hydrodynamic fluctuations at the convective instability. *Phys. Rev. A*, 15(1), 319–328. <https://doi.org/10.1103/PhysRevA.15.319>
- **Mathematica**: [`1.1.2.SHH2D.nb`](../../../../Mathematica/1_continuous_scheme/1_3_biharmonic_term/swift_hohenberg_2d/1.1.2.SHH2D.nb)

### 公開実装との類似（コード出典の注記）

本スクリプトは上記 Mathematica ノートからの移植であり、下記のソースを転載したものではない。線形項 $\varepsilon-(1-k^2)^2$ を陰的にするスペクトル半陰式は公開実装と式が近い。

- `swifthohenberg.py`（FFT 半陰式の一例）: <https://git.chaospott.de/pixel/pixelserver2/src/commit/5b2f7e4b5efdc27cb465e7fbfc84a97e360b0809/apps/swifthohenberg.py>

## 数値計算スキーム

[`swift_hohenberg_2d.py`](swift_hohenberg_2d.py) を参照。

## パラメータ一覧

[`swift_hohenberg_2d.py`](swift_hohenberg_2d.py) の docstring および関数引数を参照。

## 数理解析

上記「stripe–spot 選択」節および Cross–Hohenberg 総説（*Rev. Mod. Phys.* 1993）を参照。
