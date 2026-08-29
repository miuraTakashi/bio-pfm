# Fisher–KPP 進行波（1D）

対応スクリプト: [`fisher_1d.py`](fisher_1d.py)。

左端から $u=1$ の領域が右へ侵食する **Fisher–KPP 型**の 1 次元進行波を数値積分し、波面位置 $x_f(t)$（$u=0.5$ の交差）の線形フィットから得た **数値波速度** $c_{\mathrm{num}}$ を、線形化に基づく **最小進行波速度** $c_*=2\sqrt{D}$ と比較する。

**Mathematica 移植元**: [`FisherSpeed.nb`](../../../../Mathematica/1_continuous_scheme/1_1_diffusion_term/fisher_1d/FisherSpeed.nb)

## モデルの導出（現象論）

**個体群の空間拡散** と **ロジスティック増殖** を組み合わせると、進行波前が生じる。

- **仮説**: 密度 $u$ の Fick 拡散 $D u_{xx}$；局所成長 $ru(1-u/K)$。
- **保存則**: 粒子数のフラックス＋源項 → Fisher–KPP 型。
- 次節の 1D Fisher 方程式は、侵入・拡大 front の最小モデル。

## 支配方程式

本実装では成長率を 1 に正規化した

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + f(u),
\qquad
f(u) = u(1-u)
$$

を $x\in[0,1]$ 上で解く。$f(0)=f(1)=0$ であり、$u=0$（消滅）と $u=1$（飽和）が定常解となる。

**境界条件**（Neumann・ゼロフラックス）:

$$
\frac{\partial u}{\partial x}\bigg|_{x=0} = \frac{\partial u}{\partial x}\bigg|_{x=1} = 0.
$$

**初期条件**（左端からの侵食）:

$$
u(x,0)=
\begin{cases}
1 & (x \lesssim 10\,\Delta x),\\
0.5 & (\text{遷移セル}),\\
0 & (\text{それ以外}).
\end{cases}
$$

一般の Fisher 型 $ru(1-u)$ は $r$ を時間・空間スケールで吸収できるため、本コードでは $r=1$ とみなす（$f'(0)=1$）。

## 出典（原著・標準文献）

- Fisher, R. A. (1937). The wave of advance of advantageous genes. *Annals of Eugenics*, 7(4), 355–369. <https://doi.org/10.1111/j.1469-1809.1937.tb02153.x>
- Kolmogorov, A. N., Petrovskii, I. G., & Piskunov, N. S. (1937). A study of the diffusion equation with increase in the amount of substance, and its application to a biological problem. *Bulletin of Moscow University, Mathematics and Mechanics*, 1(6), 1–26. Reprinted in *Selected Works of A. N. Kolmogorov*, Vol. 1, 242–270 (1991). <https://doi.org/10.1007/978-94-011-3030-1_38>
- 進行波の教科書的整理: Murray, J. D. (2002). *Mathematical Biology I: An Introduction* (3rd ed.). Springer. <https://doi.org/10.1007/b98868>

## 数値計算スキーム

- **空間**: 等間隔格子 $x_i=i\Delta x$、$\Delta x=0.01$、$n=\lfloor 1/\Delta x\rfloor$。
- **拡散**: 内部は 2 階中心差分、端点は Neumann に対応する 1 階差分（`diffusion`）。
- **時間**: 陽的オイラー
  $$
  u^{n+1}_i = u^n_i + \Delta t\left(\frac{D}{\Delta x^2}(Lu)^n_i + f(u^n_i)\right),
  $$
  $u$ は各ステップ後に $[0,1]$ へクリップ。
- **安定条件**: $\Delta t = 0.25\,\Delta x^2/D$（拡散の顕式 CFL）。
- **記録**: `n_steps_per_record=25` ステップごとに $u(x)$ を保存（計 `n_records=80` 本）。
- **波面追跡**: 各記録時刻で $u=0.5$ を挟む最初のセル間を線形補間し $x_f(t)$ を得る。

## パラメータ一覧

| 識別子 | 記号 | 既定値 | 意味 |
|--------|------|--------|------|
| `D` | $D$ | `0.0001` | 拡散係数 |
| `dx` | $\Delta x$ | `0.01` | 格子間隔 |
| `n_steps_per_record` | — | `25` | スナップショット間のステップ数 |
| `n_records` | — | `80` | スナップショット本数 |
| `seed` | — | `42` | 乱数シード（現行 IC は決定的） |

$\Delta t$ はコード内で $\Delta t=0.25\,\Delta x^2/D$ として自動設定される。

## 数理解析

### 定常解と局所安定性

$f(u)=u(1-u)$ より $u^*=0,1$。線形化 $f'(u^*)$ は

$$
f'(0)=1>0 \quad\text{（不安定）},\qquad
f'(1)=-1<0 \quad\text{（安定）}.
$$

小さな正の摂動は $u=0$ から増幅し、飽和状態 $u=1$ へ向かう **侵食（invasion）** が起こる。

### 進行波 ansatz

右向き進行波 $u(x,t)=U(\eta)$、$\eta=x-ct$（$c>0$）を代入すると

$$
D U'' + c U' + f(U) = 0,
\qquad
U(-\infty)=1,\quad U(+\infty)=0.
$$

$U$ は $\eta\to-\infty$ で飽和、$\eta\to+\infty$ で消滅する単調（または単峰）プロファイルを与える。

### 先端（$U\ll 1$）の線形化と最小波速度

先端 $\eta\to+\infty$ で $U\sim e^{-\lambda\eta}$（$\lambda>0$）とすると、$f(U)\approx f'(0)\,U$ より

$$
D\lambda^2 - c\lambda + f'(0) = 0.
$$

実根 $\lambda$ が存在し指数減衰を実現するには判別式 $c^2-4Df'(0)\ge 0$ が必要。これを満たす **最小** 波速度が

$$
c_* = 2\sqrt{D\,f'(0)}.
$$

本モデルでは $f'(0)=1$ なので

$$
c_* = 2\sqrt{D}.
$$

$r\,u(1-u)$ の一般形では $f'(0)=r$ となり $c_*=2\sqrt{rD}$ である（$r$ を 1 に正規化すれば上式と一致）。

### 位相平面（概要）

$\xi=U$、$p=U'$ として

$$
\frac{d\xi}{d\eta}=p,\qquad
\frac{dp}{d\eta}=-\frac{c}{D}p-\frac{f(\xi)}{D}
$$

と書ける。結合点 $(1,0)$（飽和）と $(0,0)$（消滅）を結ぶ heteroclinic orbit が進行波解に対応する。$c<c_*$ では先端の線形減衰が不可能になり、進行波は存在しない（KPP の最小速度定理）。

### 本リポジトリでの数値検証

[`fisher_1d.py`](fisher_1d.py) は $x_f(t)$ に対し最小二乗直線フィットし $c_{\mathrm{num}}=\mathrm{d}x_f/\mathrm{d}t$ を推定する。理論値は $c_*=2\sqrt{D}$。

既定 $D=10^{-4}$ では $c_*=0.02$。有限領域・離散化・初期の有限幅（左端 10 セル）のため、$c_{\mathrm{num}}$ はやや小さめに出ることがある（代表 run: $c_{\mathrm{num}}\approx 0.016$、$c_*\approx 0.020$）。

![Fisher–KPP: 時系列プロファイルと波面速度の比較](results/fisher_1d.png)

左: 複数時刻の $u(x)$（色は時刻）。右: $u=0.5$ フロント位置 $x_f(t)$ の散布と線形フィット。

**本 md では行わないこと**: 厳密な $U(\eta)$ の閉形式、分岐・加速波、2 次元の円形波面（2D は [`fisher_2d`](../fisher_2d/fisher_2d.md)）。

## 出力

| ファイル | 内容 |
|----------|------|
| `results/fisher_1d.png` | 左: $u(x)$ の時系列重ね描き。右: $x_f(t)$ と $c_{\mathrm{num}}$ vs $c_*$ |

## 実行

```bash
cd python/1_continuous_scheme/1_1_diffusion_term/fisher_1d
python3 fisher_1d.py
```

標準出力に `Numerical wave speed` と `theory 2*sqrt(D)` が表示される。
