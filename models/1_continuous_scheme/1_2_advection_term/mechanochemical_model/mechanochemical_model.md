# メカノケミカル（機械）1D 連成モデル

対応スクリプト: [`mechanochemical_model.py`](mechanochemical_model.py)。

## モデルの導出（現象論）

**メカノケミカル** モデルは、化学濃度 $c$ と機械変位 $\mathbf{u}$ が双方向に耦合する。

- **仮説**: 走化性・収縮ストレスが細胞／基質変形を駆動；変形が $c$ の輸送を変える。
- **構成**: 力の釣り合い＋質量保存＋Hook 型応力。
- 次節の連成 PDE は、形態形成のメカノケミカル現象論。

## 支配方程式（コード docstring より）

周期境界上の 3 場 $n(x,t)$, $\rho(x,t)$, $u(x,t)$ について、輸送と弾性 Poisson 連成。

**記法** — 時間・空間微分は $\partial_t$, $\partial_x$, $\partial_{xx}$ と書く。下付きの $i$, $i\pm\frac{1}{2}$ は **離散格子のインデックス**（模式図・数値スキーム）にのみ用いる。コード内部では `n_t`, `u_t`, `u_xx` 等の略記がある。

**尺度** — 長さ $\ell_0$、時間 $\tau_0$ を参照スケールとする（本実装の既定パラメータは $\ell_0=\tau_0=1$ の無次元系）。$n,\rho$ は基準値 $n_0,\rho_0$ で除いた **無次元の相対密度・活性度**（平均 $\approx 1$）。$u$ は **変位**、$\partial_t u$ は **速度** として扱う。

| 記号 | 役割 | 単位 |
|------|------|------|
| $x$ | 1 次元空間座標 | $\ell_0$ |
| $t$ | 時間 | $\tau_0$ |
| $n(x,t)$ | 第 1 密度場（例: 細胞密度・ネットワーク密度の相対値）。平均 $n\approx 1$ 付近の小摂動 | —（無次元） |
| $\rho(x,t)$ | 第 2 密度場（例: 収縮性・活性度の相対値）。$n$ と同様 | —（無次元） |
| $u(x,t)$ | 弾性変位（またはひずみに対応するスカラー場） | $\ell_0$ |
| $\partial_t u$ | $u$ の時間変化率。**同時に** $n,\rho$ を移流させる速度場 | $\ell_0/\tau_0$ |
| $\sigma$ | 活性応力 $n\rho+\gamma\,\partial_{xx}\rho$ | —（無次元；弾性モジュラスで除いた応力） |
| $(n\,\partial_t u)_{i\pm 1/2}$ | 格子界面の物質フラックス（模式図） | $\ell_0/\tau_0$ |

3 場は力学（$u$）→ 移流速度（$\partial_t u$）→ 密度再分布（$n,\rho$）→ 活性応力（$n\rho$）→ 力学、という **メカノケミカル（機械–化学）フィードバック** を形成する。

### 輸送方程式（$n$, $\rho$）

密度 $n,\rho$ は力学変位 $u$ と一体の物質として扱われ、力学から得た速度 $\partial_t u$ に **随伴して輸送** される。したがって連成系には、密度の保存則として次の輸送方程式が現れる。

![$\partial_t n+\partial_x(n\,\partial_t u)=0$ の説明：3 格子間の物質の出入り（[`mechanochemical_transport_flux.py`](mechanochemical_transport_flux.py) で生成）](explanatory_figures/mechanochemical_transport_flux.png)

$$
\partial_t n + \partial_x(n\,\partial_t u) = 0,\qquad
\partial_t \rho + \partial_x(\rho\,\partial_t u) = 0
$$

**$\partial_t n$, $\partial_t \rho$** — 固定した位置 $x$ における各密度の **局所時間変化率**（単位 $\tau_0^{-1}$；$\partial_t n>0$ で増加）。

**$\partial_x(n\,\partial_t u)$, $\partial_x(\rho\,\partial_t u)$** — 保存則形式の **移流項**（単位 $\tau_0^{-1}$）。密度が速度 $\partial_t u$ に随伴して輸送される。格子 $i$ では界面フラックス $(n\,\partial_t u)_{i-\frac{1}{2}}$（流入）と $(n\,\partial_t u)_{i+\frac{1}{2}}$（流出）の差で $\partial_t n$ が決まり、移項すると $\partial_t n=-\partial_x(n\,\partial_t u)$ となる。右辺が 0 なので $\int n\,\mathrm{d}x$（無次元密度の空間積分）は保存される。数値実装は風上フラックス形式 [`upwind_step`](mechanochemical_model.py)。

### 力学方程式（$u$）

$$
\mu\,\partial_t \partial_{xx} u + \partial_{xx} u + \tau\,\partial_x\bigl(n\rho + \gamma \partial_{xx}\rho\bigr) - s\,\rho\,u = 0
$$

活性応力 $\sigma = n\rho + \gamma\,\partial_{xx}\rho$ と書くと $\mu\,\partial_t \partial_{xx} u + \partial_{xx} u + \tau\,\partial_x\sigma - s\,\rho\,u = 0$ となる。

| 項 | 記号 | 物理的役割 | 項の単位 | 係数の単位 |
|----|------|------------|----------|------------|
| 粘弾性慣性 | $\mu\,\partial_t \partial_{xx} u$ | $\partial_t u$ の空間曲率の時間変化に比例。Kelvin–Voigt 型 **粘弾性**（$\mu$ が大きいほど応答が遅れる） | $\ell_0^{-1}$ | $\mu$: $\tau_0$ |
| 弾性復元 | $\partial_{xx} u$ | 線形弾性 **復元力**。$u$ の不均一を平滑化し短波長を安定化 | $\ell_0^{-1}$ | — |
| 活性応力勾配 | $\tau\,\partial_x\sigma$ | メカノケミカル **駆動項**。密度の不均一から生じる応力勾配が $u$ を駆動 | $\ell_0^{-1}$ | $\tau$: — |
| 摩擦・減衰 | $-s\,\rho\,u$ | 基質に対する **摩擦**（$\rho$ が大きいほど $u$ の減衰が強い） | $\ell_0^{-1}$ | $s$: $\ell_0^{-2}$ |

力学式の **4 項はいずれも $\ell_0^{-1}$** で釣り合う。粘弾性項だけ見た目に $\partial_t$ が余分だが、$[\partial_t\partial_{xx}u]=\ell_0^{-1}\tau_0^{-1}$ なので係数 $\mu$（$\tau_0$）を掛けると他項と同次元になる。

**活性応力 $\sigma$ の内訳**（$\sigma$ は無次元）

- **$n\rho$** — 2 密度の積。局所 **収縮性・活性応力**（メカノケミカル結合項）。単位 —。
- **$\gamma\,\partial_{xx}\rho$** — $\rho$ の空間 2 階微分。密度勾配・曲率に伴う **追加応力**（界面張力・剛性に相当）。$\gamma$ の単位 $\ell_0^2$。線形解析では $-k^4\gamma\tau$ となり短波長を抑制。

### 補助 Poisson 方程式（$\partial_t u$ の求解）

力学式は $\partial_t u$ について整理した Poisson 問題と等価である。実装では $\partial_t u$ を次の式の解として求める:

$$
\partial_{xx}(\partial_t u) = \frac{-\partial_{xx} u - \tau\,\partial_x\sigma + s\,\rho\,u}{\mu},
\qquad
\sigma = n\rho + \gamma\,\partial_{xx}\rho
$$

（[`solve_poisson_periodic`](mechanochemical_model.py)）

**Poisson 方程式** — スカラー場 $\phi$ のラプラシアンが既知の右辺 $f$ に等しい方程式。1 次元では $\partial_{xx}\phi=f(x)$、2 次元では $\Delta\phi=f(\mathbf{x})$。「右辺 $f$ というソースから、曲がり具合が $f$ になる場 $\phi$ を逆算する」問題である。右辺が 0 のときは **ラプラス方程式**。周期境界では $\phi$ の定数部分（ゼロ波数モード）が定まらないため、本実装では平均を 0 に固定して **ゲージ自由度を除去** する。

| 分野 | $\phi$ の意味 | 方程式 |
|------|--------------|--------|
| 静電学 | 電位 $V$ | $\Delta V = -\rho/\varepsilon_0$ |
| 定常熱伝導 | 温度 $T$ | $\Delta T = -q/k$ |
| 重力 | ポテンシャル $\Phi$ | $\Delta\Phi = 4\pi G\rho$ |

**本モデルでの右辺** — 左辺 $\partial_{xx}(\partial_t u)$ は移流速度 $\partial_t u$ の空間曲率。右辺は弾性項・活性応力・摩擦から決まる駆動。FFT により周波数空間で $-k^2$ 除算して求解する。

得られた $\partial_t u$ で $u\leftarrow u+\Delta t\,\partial_t u$ と更新し、同じ $\partial_t u$ を $n,\rho$ の移流速度として使う。

## 出典（原著・標準文献）

- 形態形成における連続モデルの枠組: Murray, J. D. (2003). *Mathematical Biology II: Spatial Models and Biomedical Applications* (3rd ed.). Springer. <https://doi.org/10.1007/b98869>
- 活性ゲル・メカノケミカル連成の近年総説: Marchetti, M. C., et al. (2013). Hydrodynamics of soft active matter. *Reviews of Modern Physics*, 85(3), 1143–1189. <https://doi.org/10.1103/RevModPhys.85.1143>
- **本実装の式はノートブック由来の特定形**であり、上記はモデル族の文脈付け。

## 数値計算スキーム

- **空間**: 周期境界、中心差分または FFT 微分（`stress_derivative_method`）。
- **$n,\rho$**: 風上フラックス形式の `upwind_step`。
- **$\partial_t u$**: FFT で Poisson 求解後、適応時間刻み（CFL 型）で明示的更新。
- **線形比較**: `linear_regime_compare` で分散関係 $\sigma(k)$ に基づく理論ゲインと数値を比較可能。
- **説明専用図**: 解析・模式図は `explanatory_figures/` に保存（atlas ギャラリー/PPT 収集対象外）。

## パラメータ一覧（`run_simulation` 既定の主なもの）

| 識別子 | 記号 | 既定値 | 単位 | 意味 |
|--------|------|--------|------|------|
| `L` | $L$ | $8\pi$ | $\ell_0$ | 領域長 |
| `T` | $T$ | 12.0 | $\tau_0$ | 終了時刻 |
| `nx` | — | 256 | — | 格子数 |
| `dt` | $\Delta t$ | 1e-3 | $\tau_0$ | 時間刻み上限 |
| `s` | $s$ | 1 | $\ell_0^{-2}$ | 摩擦係数 |
| `tau` | $\tau$ | 3 | — | 活性応力–力学の結合強度 |
| `mu` | $\mu$ | 1 | $\tau_0$ | 粘弾性時間（慣性） |
| `gamma` | $\gamma$ | 1 | $\ell_0^2$ | $\partial_{xx}\rho$ 項の応力係数 |
| `noise_amp` | — | 2e-5 | — | 初期ノイズ振幅（$n,\rho$ の無次元摂動） |
| `cfl` | — | 0.3 | — | CFL 係数 |
| `stress_derivative_method` | — | `"spectral"` | — | `"spectral"` / `"central"` |
| `initial_condition` | — | `"unbiased_noise"` | — | 初期データ種別 |

## 数理解析

### 線形化の設定

均一背景 $n=\rho=1$, $u=0$ まわりに小摂動 $\delta n,\delta\rho,\delta u$ を置き、周期境界 $[0,L)$ 上の Fourier モード

$$
\delta n,\;\delta\rho,\;\delta u \;\propto\; e^{\sigma(k)\,t + i k x},
\qquad
k=\frac{2\pi m}{L}\quad (m=1,2,\ldots)
$$

とする。線形成長率 $\sigma(k)$ の符号で、各波数が指数関数的に増大（$\sigma>0$）するか減衰（$\sigma<0$）するかを判定する（単位 $\tau_0^{-1}$）。

### 分散関係 $\sigma(k)$

ノートブック由来の解析（[`dispersion_sigma`](mechanochemical_model.py)）より、$n$ のモードに対して

$$
\sigma(k)=\frac{-k^2-s+2k^2\tau-k^4\gamma\tau}{k^2\,\mu}
$$

が得られる。分子の各項は力学式の線形化に対応する。

| 分子の項 | 由来（線形化） | 波数依存 | 役割 |
|----------|----------------|----------|------|
| $-k^2$ | $\partial_{xx}u$（弾性復元） | $\propto k^2$ | 短波長を安定化 |
| $-s$ | $-s\rho u$（摩擦；$\rho\approx 1$） | 定数 | 全モードを減衰 |
| $+2k^2\tau$ | $\tau\partial_x(n\rho)$ の活性応力結合 | $\propto k^2$ | **不安定化**（$\tau$ が大きいほど強い） |
| $-k^4\gamma\tau$ | $\gamma\partial_{xx}\rho$ の高次応力 | $\propto k^4$ | 短波長を追加安定化 |

分母 $k^2\mu$ は粘弾性慣性 $\mu\partial_t\partial_{xx}u$ による **応答の遅れ** を表す（$\mu$ が大きいほど成長は遅れる）。

### 最不安定モードと波長

$\sigma(k)$ は $k\to 0$ で $\sigma\sim -s/(k^2\mu)\to -\infty$（長波長は強く減衰）、$k\to\infty$ で $\sigma\sim -k^2\gamma\tau/\mu\to -\infty$（短波長も安定）。中間の $k$ で $\sigma(k)>0$ となれば **線形不安定** が起こる。

既定パラメータ $s=\mu=\gamma=1$, $\tau=3$ では分子 $N(k)=-k^2-s+2k^2\tau-k^4\gamma\tau=5k^2-1-3k^4$ を最大化する波数は

$$
k_*=\sqrt{\frac{5}{6}}\approx 0.91,
\qquad
\sigma_{\max}=\sigma(k_*)\approx 1.3\;\;(\tau_0^{-1})
$$

周期 $L=8\pi$ では最不安定 **モード番号** $m_*=k_*L/(2\pi)=4\sqrt{5/6}\approx 3.7$（$m=3$ または $4$ 付近）。対応 **波長** $\lambda_*=2\pi/k_*\approx 6.9\,\ell_0$。

用語: 線形理論の $k_*$ は **最不安定モード**、非線形数値の FFT ピークは **支配的モード**（[`GLOSSARY.md`](../../../GLOSSARY.md)）。

### 数値による線形検証

[`linear_regime_compare`](mechanochemical_model.py) は小振幅（`noise_amp=2e-8`）・短時間（$T=0.8$）で走らせ、$n$ の rFFT 振幅 $|A_m(t)|$ に

$$
\log|A_m(t)|\approx \log|A_m(0)|+\sigma(k_m)\,t
$$

をフィットした $\sigma_{\mathrm{num}}$ と、上記 $\sigma(k_m)$ を比較する。出力 [`results/mechanochemical_model_linear_compare.png`](results/mechanochemical_model_linear_compare.png)（`main()` 実行時）。

`stress_derivative_method="spectral"`（FFT 微分）の方が `"central"` より線形成長率に近い（[`compare_stress_derivative_methods`](mechanochemical_model.py)）。

### 非線形領域

$\sigma(k)>0$ のモードが成長した後は移流・活性応力の非線形効果が支配し、**粗いモードへのエネルギー移動**（コアスニング）が起こる。`main()` 既定（`T=7.5`, `noise_amp=2e-5`）では $t\gtrsim 8$ 以降に支配的モードが $m=1$ へ移りやすいため、$m=3$ 付近が優勢な時間帯で打ち切っている。非線形パターンの詳細は数値実験に委ねる。
