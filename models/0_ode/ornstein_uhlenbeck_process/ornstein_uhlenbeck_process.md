# Ornstein–Uhlenbeck（OU）過程

対応スクリプト: [`ornstein_uhlenbeck_process.py`](ornstein_uhlenbeck_process.py)。パラメータ表はソースの既定値に準拠（変更時はコードと併せて更新してください）。

## モデルの導出（現象論）

粒子の位置 $X_t$ が **ランダムウォーク＋復元力**（平均回帰）に従うと仮定する。

- **仮説**: フラックス $J = -D
abla u$ に加え、ポテンシャル勾配によるドリフト $-	heta X_t$（Ornstein–Uhlenbeck）。
- **保存則**: 確率密度の Fokker–Planck から SDE $dX_t = -	heta X_t\,dt + \sigma\,dW_t$ が得られる。
- 次節はその離散時間近似・数値実装。

## 支配方程式

**確率微分方程式**（SDE）:

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t,
\qquad X_0 = x_0,
$$

ここで $\theta > 0$ は **平均回帰速度**、$\mu$ は **長期平均**、$\sigma > 0$ は **拡散係数**、$W_t$ は標準 **ウィーナー過程**（Brown 運動）である。ドリフト項 $\theta(\mu - X_t)$ により $X_t$ は $\mu$ 方向へ引き戻される（**平均回帰**、mean reversion）。拡散項 $\sigma\,dW_t$ がランダムな揺らぎを与える。

**等価形**（定常 OU）:

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t \quad (\mu = 0),
$$

または $Y_t = X_t - \mu$ とおけば $dY_t = -\theta Y_t\,dt + \sigma\,dW_t$ となり、$\mu$ は平行移動に相当する。

## 出典（原著・標準文献）

- Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the Theory of the Brownian Motion II. *Physical Review*, 36(5), 823–841. <https://doi.org/10.1103/PhysRev.36.823>
- Gardiner, C. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences* (4th ed.). Springer.（OU 過程・Fokker–Planck 方程式の標準的記述）
- Øksendal, B. (2003). *Stochastic Differential Equations: An Introduction with Applications* (6th ed.). Springer.（Itô 積分・線形 SDE の解法）
- 金融・時系列文脈: Vasicek, O. (1977). An equilibrium characterization of the term structure. *Journal of Financial Economics*, 5(2), 177–188.（金利モデルとしての OU / Vasicek モデル）

## 数値計算スキーム

本スクリプトは **Euler–Maruyama 法は用いない**。OU 過程は有限時間 $\Delta t$ ごとの **条件付き正規分布**（ガウス遷移）が厳密に書けるため、`ou_transition_step` で **exact 更新**を行う:

$$
X_{t+\Delta t}\,\big|\,X_t \sim \mathcal{N}\!\left(
\mu + (X_t - \mu)\,e^{-\theta\Delta t},\;
\frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta\Delta t}\bigr)
\right).
$$

- **利点**: 離散化バイアスが Euler–Maruyama より小さい（$\Delta t$ が同じでも分散・自己相関の一致が良い）。
- **実装**: `simulate_ou` が `n_paths` 本の独立パスを生成。各パネルで経験統計量と上記の解析式を比較する。

## パラメータ一覧

| 識別子 | 記号 | 既定値 | 単位 | 意味 |
|--------|------|--------|------|------|
| `mu` | $\mu$ | 0.0 | — | 長期平均（定常分布の中心） |
| `theta` | $\theta$ | 1.0 | 1/時間 | 平均回帰速度 |
| `sigma` | $\sigma$ | 1.0 | — | 拡散係数（$dW_t$ に対する振幅） |
| `x0` | $x_0$ | 2.0 | — | 初期値（$\mu$ から離して平均回帰を可視化） |
| `t_end` | — | 10.0 | — | シミュレーション終了時刻 |
| `dt` | $\Delta t$ | 0.01 | — | 時間刻み |
| `n_paths` | — | 200 | — | 独立サンプルパスの本数 |
| `burn_in_steps` | — | 500 | — | 定常統計用に捨てる初期ステップ数（$5.0/\Delta t$） |
| `check_times` | — | 0.5, 2.0, 10.0 | — | 周辺分布を比較する時刻（図 2） |
| `psd_t_end` | — | 200.0 | — | PSD 推定用シミュレーション長 |
| `psd_burn_in` | — | 2000 | — | PSD 用 burn-in ステップ数（$20.0/\Delta t$） |

## 数理解析

### 平均の時間発展

SDE 両辺の期待値をとる（$W_t$ のマルティンゲール性より $dW_t$ の期待は 0）:

$$
\frac{d}{dt}\,\mathbb{E}[X_t] = \theta\bigl(\mu - \mathbb{E}[X_t]\bigr).
$$

これは $\mathbb{E}[X_t]$ についての 1 次常微分方程式。初期条件 $\mathbb{E}[X_0]=x_0$ の解は

$$
\mathbb{E}[X_t] = \mu + (x_0 - \mu)\,e^{-\theta t}.
$$

- $x_0 \neq \mu$ なら指数関数的に $\mu$ へ収束（**平均回帰**）。
- 定常状態 ($t\to\infty$): $\mathbb{E}[X_t] \to \mu$。

コード: `ou_mean`。

### 分散の時間発展

**方法 1（分散の ODE）**  
$m(t)=\mathbb{E}[X_t]$、$v(t)=\mathrm{Var}(X_t)$ とする。Itô の公式より

$$
dX_t^2 = 2X_t\,dX_t + (dX_t)^2
= 2X_t\bigl[\theta(\mu-X_t)\,dt + \sigma\,dW_t\bigr] + \sigma^2\,dt.
$$

期待値をとると

$$
\frac{d}{dt}\,\mathbb{E}[X_t^2] = 2\theta\mu\,\mathbb{E}[X_t] - 2\theta\,\mathbb{E}[X_t^2] + \sigma^2.
$$

$v(t) = \mathbb{E}[X_t^2] - m(t)^2$ と $m(t)$ の ODE を組み合わせると

$$
\frac{dv}{dt} = -2\theta\,v + \sigma^2,
\qquad v(0) = 0 \quad (x_0 \text{ 確定的}).
$$

解は

$$
\mathrm{Var}(X_t) = \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta t}\bigr).
$$

**方法 2（中心化過程）**  
$Y_t = X_t - \mathbb{E}[X_t]$ とすると $dY_t = -\theta Y_t\,dt + \sigma\,dW_t$、$Y_0=0$。同様に $\mathrm{Var}(Y_t)=\mathrm{Var}(X_t)$ が上式を満たす。

- $t=0$ で分散 0（確定的初期値）。
- $t\to\infty$ で $\mathrm{Var}(X_t) \to \sigma^2/(2\theta)$（**定常分散**）。

コード: `ou_variance`, `ou_stationary_variance`。

### 定常分布

Fokker–Planck 方程式の定常解（または上式の $t\to\infty$）より、定常分布は

$$
\pi(x) = \mathcal{N}\!\left(\mu,\;\frac{\sigma^2}{2\theta}\right)
= \frac{1}{\sqrt{2\pi\,\sigma^2/(2\theta)}}
\exp\!\left(-\frac{(x-\mu)^2}{\sigma^2/\theta}\right).
$$

本デモの既定値 ($\mu=0$, $\sigma=1$, $\theta=1$) では $\mathcal{N}(0,\,0.5)$。

図 (d) では `burn_in_steps` 以降のサンプルをヒストグラム化し、上記 PDF と重ね合わせる。

### 自己相関（定常状態）

定常 OU 過程（$t\to\infty$、$X_t \sim \mathcal{N}(\mu,\,\sigma^2/(2\theta))$）では、ラグ $\tau$ に対する自己相関関数は

$$
\rho(\tau) = \frac{\mathrm{Cov}(X_t, X_{t+\tau})}{\mathrm{Var}(X_t)}
= e^{-\theta|\tau|}.
$$

- $\tau=0$ で 1、$|\tau|$ が増えると指数減衰。
- **相関時間**の目安: $1/\theta$（$e^{-1}$ に減るラグ）。

図 (c) では burn-in 後の各パスから経験的自己相関を計算し、パス平均と $e^{-\theta|\tau|}$ を比較する（`ou_autocorrelation`, `empirical_autocorrelation`）。

### パワースペクトル（定常状態）

定常 OU 過程の **自己共分散関数**（autocovariance）は

$$
R(\tau) = \mathrm{Cov}(X_t, X_{t+\tau})
= \frac{\sigma^2}{2\theta}\,e^{-\theta|\tau|}.
$$

$\mu$ は定常平均であり、共分散・スペクトルには影響しない（$Y_t = X_t - \mu$ で同じ $R$ になる）。

#### Wiener–Khinchin の定理

広義定常過程では、自己共分散 $R(\tau)$ と **両側**パワースペクトル密度 $S(\omega)$（角周波数 $\omega$）が Fourier 変換で対応する:

$$
S(\omega) = \int_{-\infty}^{\infty} R(\tau)\,e^{-i\omega\tau}\,d\tau,
\qquad
R(\tau) = \frac{1}{2\pi}\int_{-\infty}^{\infty} S(\omega)\,e^{i\omega\tau}\,d\omega.
$$

$R(\tau)$ が偶関数 $e^{-\theta|\tau|}$ なので $S(\omega)$ も偶関数（実数・非負）になる。$\int_{-\infty}^{\infty} e^{-\theta|\tau|} e^{-i\omega\tau}\,d\tau = 2\theta/(\theta^2+\omega^2)$ より

$$
\boxed{
S(\omega) = \frac{\sigma^2}{\theta^2 + \omega^2}
}
\qquad\text{（Lorentz 型 / Cauchy 型）}.
$$

コード: `ou_power_spectral_density`, `ou_autocovariance`。

#### スペクトルの形状と特徴周波数

| 周波数域 | 漸近 | 意味 |
|----------|------|------|
| $\omega \ll \theta$ | $S(\omega) \approx \sigma^2/\theta^2$ | **低周波でほぼ平坦**（白ノイズほど尖らない） |
| $\omega = \theta$ | $S(\theta) = \sigma^2/(2\theta^2)$ | 半値幅（最大値 $S(0)=\sigma^2/\theta^2$ の 1/2） |
| $\omega \gg \theta$ | $S(\omega) \approx \sigma^2/\omega^2$ | **$1/\omega^2$ 減衰**（高周波成分が強く抑えられる） |

- **コーナー角周波数**（特性角周波数）: $\omega_c = \theta$。Hz 表記では $f_c = \theta/(2\pi)$。
- **物理的解釈**: OU 過程は「白ノイズ $ \sigma\,dW_t$」を **1 次ローパスフィルタ**（時定数 $1/\theta$）で平滑化した信号とみなせる。平均回帰速度 $\theta$ が大きいほど高周波カットオフが上がり、時系列はより「白っぽく」揺れる。

#### 片側スペクトル・Hz 表記・分散との関係

実データ解析（Welch 法など）では **片側**・**Hz** の PSD $P(f)$ を使うことが多い。$f = \omega/(2\pi)$ として

$$
P(f) = \frac{2\sigma^2}{\theta^2 + (2\pi f)^2},
\qquad
\int_0^{\infty} P(f)\,df = \mathrm{Var}(X_t) = \frac{\sigma^2}{2\theta}.
$$

コード: `ou_power_spectral_density_hz`。図 3 左パネルは burn-in 後の時系列に Welch 法を適用し、パス平均と上式を log–log で比較する。

#### 他のノイズとの比較

| 過程 | 定常性 | 低周波 | 高周波 | スペクトル形状 |
|------|--------|--------|--------|----------------|
| **白ノイズ** | 定常 | 平坦 | 平坦 | $S(\omega) \sim \mathrm{const}$ |
| **OU 過程** | 定常 | 平坦 ($\to \sigma^2/\theta^2$) | $\sim 1/\omega^2$ | Lorentz 型 |
| **Brown 運動**（$dX = \sigma\,dW$） | 非定常 | $\sim 1/\omega^2$ | $\sim 1/\omega^2$ | スペクトル定義が別（増分は白） |
| **赤ノイズ**（ランダムウォーク型） | 非定常 | 大 | — | 低周波支配 |

OU は **有限分散の定常ガウス過程**であり、白ノイズより低周波が強く、Brown 運動のように分散が発散しない点がスペクトルにも現れる。

#### 離散時間（AR(1)）との対応

刻み $\Delta t$ の exact 更新 $X_{n+1} = \mu + (X_n-\mu)e^{-\theta\Delta t} + \varepsilon_n$（$\varepsilon_n$ は i.i.d. ガウス）は **AR(1)** である。連続極限 $\Delta t \to 0$ で上記 Lorentz 型 $S(\omega)$ を回収する。サンプリング周波数 $f_s = 1/\Delta t$ では **ナイキスト周波数** $f_{\mathrm{Nyq}} = f_s/2$ より上に理論スペクトルは存在せず、$\Delta t$ が大きいと高周波側で離散化による歪みが出る（本デモは $\Delta t=0.01$, $\theta=1$ で $f_c \approx 0.16\,\mathrm{Hz} \ll f_{\mathrm{Nyq}}=50\,\mathrm{Hz}$ と十分離れている）。

#### 非定常初期区間

$t$ が小さい区間（$x_0$ から平均・分散が定常値へ弛豫する間）は、厳密な PSD は時間依存であり、上記 $S(\omega)$ は **定常状態**（本デモでは burn-in 後）での式である。平均回帰の指数減衰 $e^{-\theta t}$ がスペクトルに現れるのは主に **低周波側のエネルギー過剰**としてであり、十分長い burn-in で除去する。

![パワースペクトル（Welch 推定 vs 理論 Lorentz 型）](results/ornstein_uhlenbeck_psd.png)

### 有限時刻の周辺分布（ガウス性）

線形 SDE である OU 過程は **任意の $t$ で $X_t$ がガウス分布**に従う:

$$
X_t \sim \mathcal{N}\!\left(
\mu + (x_0-\mu)\,e^{-\theta t},\;
\frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta t}\bigr)
\right).
$$

図 2 (`results/ornstein_uhlenbeck_marginals.png`) では $t=0.5,\,2,\,10$ におけるヒストグラムと上記理論 PDF を比較する。$t$ が大きいほど平均は $\mu$ に、分散は $\sigma^2/(2\theta)$ に近づく。

### 遷移核（1 ステップ exact 更新の根拠）

条件付き分布 $X_{t+\Delta t}\,|\,X_t=x$ もガウスである。オラクル SDE

$$
dZ_s = \theta(\mu - Z_s)\,ds + \sigma\,dW_s, \quad Z_0 = x,\; s\in[0,\Delta t]
$$

の解は

$$
Z_{\Delta t} = \mu + (x-\mu)\,e^{-\theta\Delta t}
+ \sigma\int_0^{\Delta t} e^{-\theta(\Delta t-s)}\,dW_s.
$$

積分項は平均 0 のガウス乱数で、分散は

$$
\sigma^2\int_0^{\Delta t} e^{-2\theta(\Delta t-s)}\,ds
= \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta\Delta t}\bigr).
$$

これが `ou_transition_step` の更新式の理論的根拠である。

### 本デモでの 4 パネルの読み方

| パネル | 内容 | 理論と照合する量 |
|--------|------|------------------|
| (a) | サンプルパス、$\mathbb{E}[X_t]$、$\pm 1$ 標準偏差帯 | `ou_mean`, `ou_std` |
| (b) | パス間の経験分散 vs 時間 | `ou_variance`, 定常値 $\sigma^2/(2\theta)$ |
| (c) | burn-in 後の自己相関 | $e^{-\theta|\tau|}$ |
| (d) | 定常区間のヒストグラム | $\mathcal{N}(\mu,\,\sigma^2/(2\theta))$ |

![統計的性質の概要（パス・分散・自己相関・定常分布）](results/ornstein_uhlenbeck_process.png)

![固定時刻における周辺分布（$t=0.5,\,2,\,10$）](results/ornstein_uhlenbeck_marginals.png)

### OU 過程の位置づけ（他モデルとの対比）

| 性質 | OU 過程 | 幾何ブラウン運動 $dX=\mu X\,dt+\sigma X\,dW$ | 決定論 ODE（本 `0_ode` の他モデル） |
|------|---------|-----------------------------------------------|--------------------------------------|
| ランダム性 | あり（$dW_t$） | あり | なし |
| 平均 | $\mu$ へ回帰 | ドリフトで発散しうる | 平衡点・周期軌道など |
| 定常分布 | ガウス（有界分散） | 対数正規（境界で 0） | 軌道のみ |
| 解析 | 1 次線形 SDE で閉形式 | 非線形 | 平衡点・線形化など |

パターン形成・反応拡散系では、**外部ノイズ**や **フィルタリングされた乱数源**として OU 過程（またはその離散版）が使われることがある。本アトラスでは **確率過程単体**の統計的性質を可視化する入門デモとして同梱している。

### 本スクリプトが計算しないもの

- **共分散** $\mathrm{Cov}(X_s, X_t)$ の一般形（$s \neq t$、非定常）のプロット
- **Euler–Maruyama** との離散化誤差比較
- **パラメータ推定**（最尤・カルマン・ベイズ）
- **多次元 OU** や **相関ノイズ**
- **時間依存スペクトル**（非定常初期区間の evolutive PSD）

必要なら上記の解析式を拡張して別スクリプトに追加できる。
