# 単振動子（調和振動子の積分法比較）

対応スクリプト: [`simple_oscillator.py`](simple_oscillator.py)。

## モデルの導出（現象論）

力学系の入門として、**減衰振動子** $\ddot x + \gamma \dot x + \omega^2 x = 0$ を採用する。

- **仮説**: 線形復元力 $-\omega^2 x$ と速度比例減衰 $-\gamma \dot x$。
- **保存則**: エネルギーは減衰により散逸（$\gamma>0$）。
- 2 次 ODE を $(x,v)$ に分解すれば次節の 1 次系になる。生物物理の特定系ではなく **振動現象の標準デモ**。

## 支配方程式

$$
\frac{du}{dt} = -v,\qquad \frac{dv}{dt} = u
\quad\Rightarrow\quad \frac{d^2 u}{dt^2} + u = 0.
$$

## 出典（原著・標準文献）

- 連続時間調和振動子の厳密解は初等常微分方程式の標準結果（例: Arnold, V. I. *Ordinary Differential Equations*, Springer）。
- **数値積分**: Runge–Kutta 4 次法の古典的記述は Butcher, J. C. (2008). *Numerical Methods for Ordinary Differential Equations* (2nd ed.). Wiley. <https://doi.org/10.1002/9780470753767>

## 数値計算スキーム

状態 $\mathbf{y}=(u,v)^\mathsf{T}$、右辺 $\mathbf{f}(\mathbf{y})=(-v,\,u)^\mathsf{T}$ として、$[t_n,t_{n+1}]$（$t_{n+1}=t_n+\Delta t$）で近似する。本スクリプトは **同一 $\Delta t$・同一初期条件** で 2 法を比較する（`run_euler` / `run_rk4`）。用語: [`GLOSSARY.md`](../../GLOSSARY.md)。

### Explicit Euler（陽的オイラー法）

現在の傾きだけを使って 1 ステップ進める **1 次法**:

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \Delta t\,\mathbf{f}(\mathbf{y}_n).
$$

成分では $u_{n+1} = u_n - \Delta t\,v_n$, $v_{n+1} = v_n + \Delta t\,u_n$（コードの `run_euler`）。

- **局所切り捨て誤差**（1 ステップ）: $O(\Delta t^2)$。
- **大域誤差**（固定時間区間）: $O(\Delta t)$（1 次収束）。
- **調和振動子での性質**: 厳密解は $(u,v)$ 平面上の円（エネルギー $E=\frac12(u^2+v^2)$ 保存）だが、Euler は **シンプレクティックでない**ため、長時間で振幅が増えたり減ったりし、位相平面の軌道が **渦状に外側へ膨らみやすい**（本デモの $\Delta t=0.1$ では特に顕著）。

### 4 次 Runge–Kutta（RK4）

中間点の傾きを 4 回評価して加重平均する **4 次の明示的 Runge–Kutta**（`rk4_step`）:

$$
\mathbf{k}_1 = \mathbf{f}(\mathbf{y}_n),
$$
$$
\mathbf{k}_2 = \mathbf{f}\!\left(\mathbf{y}_n + \tfrac{\Delta t}{2}\mathbf{k}_1\right),\quad
\mathbf{k}_3 = \mathbf{f}\!\left(\mathbf{y}_n + \tfrac{\Delta t}{2}\mathbf{k}_2\right),\quad
\mathbf{k}_4 = \mathbf{f}\!\left(\mathbf{y}_n + \Delta t\,\mathbf{k}_3\right),
$$
$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}\bigl(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\bigr).
$$

- **局所切り捨て誤差**: $O(\Delta t^5)$。
- **大域誤差**（固定時間区間）: $O(\Delta t^4)$（4 次収束）。
- **調和振動子での性質**: 同じ $\Delta t$ でも Euler より位相軌道が **円に近い**。エネルギーは厳密には保存しないが、長時間のドリフトは Euler より小さいことが多い。

### 本デモでの比較の見方

| 法 | 次数 | 典型な位相平面 | 備考 |
|----|------|----------------|------|
| Explicit Euler | 1 | 円から外側へ膨らむ渦 | 実装が最も単純 |
| RK4 | 4 | 円に近い閉曲線 | 1 ステップあたり $\mathbf{f}$ を 4 回評価 |

出力 `results/simple_oscillator.png` は左が $u(t),v(t)$、右が $(u,v)$ 位相平面。Hamilton 系の長期積分には **シンプレクティック積分器**（エネルギー保存型）も選択肢になるが、本アトラスでは入門比較として Euler と RK4 のみを同梱している。

## パラメータ一覧

| 識別子 | 記号 | 既定値 | 単位 | 意味 |
|--------|------|--------|------|------|
| `simulation_length` | — | 50.0 | — | 積分区間長 |
| `u0` | $(u_0,v_0)$ | (0.2, 0.0) | — | 初期状態 |
| `dt` | $\Delta t$ | 0.1 | — | 時間刻み |

## 数理解析

### 厳密解の導出

第 1 式 $\dot u = -v$ を $t$ で微分し、第 2 式 $\dot v = u$ を代入すると

$$
\frac{d^2 u}{dt^2} = -\dot v = -u
\quad\Rightarrow\quad
\frac{d^2 u}{dt^2} + u = 0.
$$

特性方程式 $r^2+1=0$ の根は $r=\pm i$ なので、実解は

$$
u(t) = A\cos t + B\sin t
$$

と書ける。$v = -\dot u$ より

$$
v(t) = A\sin t - B\cos t.
$$

初期条件 $u(0)=u_0$, $v(0)=v_0$ を当てはめると $A=u_0$, $B=v_0$ となり、

$$
u(t) = u_0\cos t + v_0\sin t,\qquad
v(t) = -u_0\sin t + v_0\cos t.
$$

**行列形（回転）**でも同じ結果が得られる。$\dot{\mathbf{y}} = J\mathbf{y}$,

$$
J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix},\qquad
J^2 = -I,\quad e^{Jt} = \cos t\, I + \sin t\, J
= \begin{pmatrix} \cos t & -\sin t \\ \sin t & \cos t \end{pmatrix},
$$

より $\mathbf{y}(t) = e^{Jt}\mathbf{y}(0)$ すなわち上の $(u(t),v(t))$ となる（原点まわりに角度 $t$ だけ回転）。

**エネルギー保存**（検算）: $E(t)=\frac12(u^2+v^2)$ とすると

$$
\frac{dE}{dt} = u\dot u + v\dot v = u(-v) + v(u) = 0,
$$

よって $E(t)=E(0)=\frac12(u_0^2+v_0^2)$。位相平面上の軌道は中心 $(0,0)$・半径 $\sqrt{u_0^2+v_0^2}$ の円。

本デモの既定 $(u_0,v_0)=(0.2,0)$ では $u(t)=0.2\cos t$, $v(t)=-0.2\sin t$（振幅 $0.2$ の円運動）。

### 平衡点と線形化

平衡点は原点 $(0,0)$ のみ。ヤコビアンは上の $J$ で、固有値 $\lambda=\pm i$（中性安定・周波数 $1$ の純虚数）。小振幅では軌道は円に近いが、数値法（とくに Explicit Euler）の誤差がエネルギーを人工的に増減させる（**数値計算スキーム** 節参照）。
