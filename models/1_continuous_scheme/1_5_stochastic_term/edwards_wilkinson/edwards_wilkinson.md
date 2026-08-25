# Edwards–Wilkinson

**同梱スクリプト**: [`edwards_wilkinson.py`](edwards_wilkinson.py)

## モデルの導出（現象論）

**Edwards–Wilkinson (EW)** は、界面 $h(\mathbf{x},t)$ の **熱ノイズ付き梯度流**。

- **仮説**: 界面エネルギー $F\propto\int|
abla h|^2$；カップリングなしの純拡散＋白ノイズ。
- **保存則**: 非保存（沉积・エッチングの理想化）；$\partial_t h = 
u
abla^2 h + \eta$。
- 次節は EW  universality class の最小 SPDE。

## 支配方程式（モデル族）

界面高さ $h$ の線形拡散＋ノイズ:

$$
\frac{\partial h}{\partial t} = \nu \Delta h + \eta(\mathbf{x},t)
$$

（空間白色ノイズの相関関数は下記「数理解析」を参照。）

## 出典（原著・標準文献）

- Edwards, S. F., & Wilkinson, D. R. (1982). The surface statistics of a granular aggregate. *Proceedings of the Royal Society of London. A*, 381(1780), 17–31. <https://doi.org/10.1098/rspa.1982.0058>
- Family, F., & Vicsek, T. (1985). Scaling of the interface width in two-dimensional directed percolation. *Journal of Physics A*, 18(2), L75. <https://doi.org/10.1088/0305-4470/18/2/005>
- **Mathematica**: [`Edwards-Wilkinson.nb`](../../../../Mathematica/1_continuous_scheme/1_5_stochastic_term/edwards_wilkinson/Edwards-Wilkinson.nb)

## 数値計算スキーム

[`edwards_wilkinson.py`](edwards_wilkinson.py) を参照。

## パラメータ一覧

[`edwards_wilkinson.py`](edwards_wilkinson.py) の docstring および関数引数を参照。

## 数理解析

### 粗さ（界面幅）の定義

平均高さ $\bar h(t)=\frac{1}{L}\int_0^L h(x,t)\,dx$ を除いた RMS 粗さを

$$
W(t)=\sqrt{\left\langle\bigl(h(x,t)-\bar h(t)\bigr)^2\right\rangle_x}
$$

と定義する（実装 [`edwards_wilkinson.py`](edwards_wilkinson.py) の `np.sqrt(np.mean((h - h.mean())**2))` に対応）。周期境界では $\bar h$ がゼロモード $k=0$ に相当し、$W$ は $k\neq 0$ のフーリエ振幅から決まる。

### 線形性とフーリエ空間での応答

空間 $d$ 次元・周期領域 $[0,L)^d$ 上で、ノイズを

$$
\langle\eta(\mathbf x,t)\,\eta(\mathbf x',t')\rangle=\sigma^2\,\delta^d(\mathbf x-\mathbf x')\,\delta(t-t')
$$

とする（1D 実装では $d=1$）。フーリエモード $h_{\mathbf k}(t)$（$k=0$ 除く）に対して

$$
\frac{\partial h_{\mathbf k}}{\partial t}=-\nu k^2 h_{\mathbf k}+\eta_{\mathbf k}(t),
\qquad
\langle\eta_{\mathbf k}(t)\,\eta_{-\mathbf k}(t')\rangle
=\frac{\sigma^2}{L^d}\,\delta(t-t')
$$

という **独立な OU 型** 確率微分方程式の列に分解される。これがスケーリング解析の出発点である。

初期条件 $h_{\mathbf k}(0)=0$ のとき、各モードの二乗振幅（構造因子）$S_{\mathbf k}(t)=\langle|h_{\mathbf k}(t)|^2\rangle$ は Itô 公式から

$$
S_{\mathbf k}(t)=\frac{\sigma^2}{2\nu k^2 L^d}
\left(1-e^{-2\nu k^2 t}\right)
\qquad (k\neq 0)
$$

を満たす。

- **短期** $t\ll(\nu k^2)^{-1}$：$S_{\mathbf k}(t)\approx\sigma^2 t/L^d$（ノイズが白ノイズのまま蓄積）
- **長期** $t\gg(\nu k^2)^{-1}$：$S_{\mathbf k}(t)\to\sigma^2/(2\nu k^2 L^d)$（拡散がそのモードを平衡化）

粗さはモード和で

$$
W^2(t)=\frac{1}{L^d}\sum_{\mathbf k\neq 0} S_{\mathbf k}(t)
$$

と書ける。

### なぜ $W\sim t^{1/4}$ か（1+1 次元・無限系の成長期）

1 次元無限系（または $t\ll L^2/\nu$）では

$$
W^2(t)=\frac{\sigma^2}{2\pi}\int_{-\infty}^{\infty}\frac{dk}{2\nu k^2}
\left(1-e^{-2\nu k^2 t}\right)
$$

と書ける。被積分関数は $k=0$ で $\propto t$、$k\to\infty$ で $\propto k^{-2}$ と減衰する。**成長期**ではクロスオーバー波数 $k_c\sim(\nu t)^{-1/2}$ 付近で

- 低 $k$ 側：$1-e^{-2\nu k^2 t}\approx 2\nu k^2 t$ より $\int_0^{k_c} t\,dk\sim t\,k_c\sim t^{1/2}$
- 高 $k$ 側：$1-e^{-2\nu k^2 t}\approx 1$ より $\int_{k_c}^{\infty} k^{-2}\,dk\sim k_c^{-1}\sim t^{1/2}$

と両側が同じ $\sim(\nu t)^{1/2}$ のオーダーになる。したがって $W^2(t)\propto t^{1/2}$、

$$
W(t)\propto\left(\frac{\sigma^2}{\nu}\right)^{1/2} t^{1/4}.
$$

**物理的には**、拡散 $\nu\partial_x^2 h$ が短波長のランダム起伏を消し、長波長成分だけが残る。各モードへのノイズ注入（$\propto t$）と拡散による有効カットオフ $k_c\sim t^{-1/2}$ のバランスが $W\sim t^{1/4}$ を決める。非線形項 $(\nabla h)^2$ がないため、KPZ の $t^{1/3}$ ではなく **EW 普遍級** $\beta=1/4$ になる。

### 有限サイズ飽和

周期長 $L$ では最小波数 $k_{\min}=2\pi/L$ 以下しか存在しない。全モードが平衡化するのに要する時間は

$$
\tau_{\mathrm{sat}}\sim\frac{L^2}{\nu}
$$

（拡散時間）。$t\gg\tau_{\mathrm{sat}}$ では $S_k\to\sigma^2/(2\nu k^2 L)$ となり、1D 周期系では $k_n=2\pi n/L$ のモード和から

$$
W_{\mathrm{sat}}^2=\sum_{n=1}^{\infty}\frac{\sigma^2 L}{8\pi^2\nu n^2}
=\frac{\sigma^2 L}{48\nu},
\qquad
W_{\mathrm{sat}}\propto L^{1/2}.
$$

したがって log-log プロットでは $t^{1/4}$ 直線は **$t\lesssim L^2/\nu$ の早期区間** にのみ現れ、のちに水平に飽和する（[`edwards_wilkinson.py`](edwards_wilkinson.py) の参照線フィットもこの区間を想定）。

### ある程度時間が経った後の $k^{-1}$ スケーリング

周期境界 $[0,L)$ 上のフーリエ変換を

$$
\hat h(k,t)=\frac{1}{L}\int_0^L h(x,t)\,e^{-ikx}\,dx,
\qquad
h(x,t)=\sum_{k=2\pi n/L,\,n\in\mathbb Z} \hat h(k,t)\,e^{ikx}
$$

とする（$k=0$ は平均モード）。各モードは前節の $h_{\mathbf k}$ と同一で、$S(k,t)=\langle|\hat h(k,t)|^2\rangle$ は $k\neq 0$ で

$$
S(k,t)=\frac{\sigma^2}{2\nu k^2 L}
\left(1-e^{-2\nu k^2 t}\right)
$$

を満たす。

**モードごとの平衡化。** 波数 $k$ の緩和時定数は $\tau_k\sim 1/(\nu k^2)$ である。したがって **ある時刻 $t$ より十分後**（$t\gg\tau_k$）には、そのモードは平衡値

$$
S(k,\infty)=\frac{\sigma^2}{2\nu k^2 L}
\qquad (k\neq 0)
$$

に達する。振幅の平均（複素ガウス分布では $\langle|\hat h|\rangle\propto\sqrt{\langle|\hat h|^2\rangle}$）は

$$
\langle|\hat h(k,\infty)|\rangle
\propto\sqrt{S(k,\infty)}
\propto k^{-1}
$$

という **$k^{-1}$ スケーリング** を示す。構造因子 $S(k,\infty)\propto k^{-2}$ と同値である。

**なぜ $k^{-1}$ か（フラクチュエーション–散逸平衡）。** モード $k$ の OU 型方程式 $\partial_t\hat h=-\nu k^2\hat h+\hat\eta_k$ において、定常状態ではノイズ注入率と拡散による減衰率が釣り合う。白ノイズ $\langle\hat\eta_k(t)\hat\eta_{-k}(t')\rangle=\sigma^2/L\,\delta(t-t')$ から注入されるパワーは **$k$ に依存しない**（$O(\sigma^2/L)$）。一方、線形減衰 $\nu k^2\hat h$ による散逸は **$k$ が大きいほど強い**。平衡条件 $\nu k^2 S(k,\infty)\sim\sigma^2/L$ より $S(k,\infty)\propto k^{-2}$、すなわち $\langle|\hat h(k,\infty)|\rangle\propto k^{-1}$ となる。

**物理的解釈。** 拡散 $\nu\partial_x^2 h$ は波数 $k$ に対して $\nu k^2$ 倍の減衰を与えるため、短波長（大 $k$）ほどノイズで励起されても振幅が抑えられる。十分時間が経って各モードが平衡に入ると、スペクトルは「長波長優位」の $\propto k^{-2}$（振幅 $\propto k^{-1}$）になる。これが粗さ $W_{\mathrm{sat}}\propto L^{1/2}$ の直接の原因でもあり、

$$
W_{\mathrm{sat}}^2
=\sum_{k\neq 0} S(k,\infty)
\sim\int_{2\pi/L}^{k_{\max}}\frac{dk}{k^2}
\propto L
$$

と $k^{-2}$ スペクトルを $k_{\min}\sim 1/L$ から積分すると $\alpha=1/2$ が得られる。

**時間依存との関係。** 成長期 $t\ll L^2/\nu$ では $k_c\sim(\nu t)^{-1/2}$ より大きい $k$ だけが既に $k^{-1}$ 平衡スペクトルに入り、それより小さい $k$ はまだ $S(k,t)\approx\sigma^2 t/L$ と成長中である。$t\gg L^2/\nu$ では **すべての $k\ge 2\pi/L$** が平衡化し、$\langle|\hat h(k,\infty)|\rangle\propto k^{-1}$ が全波数域で成り立つ。

### 動的スケーリング（Family–Vicsek）

Family–Vicsek の動的スケーリング仮定

$$
W(L,t)=L^{\alpha}\,f\!\left(\frac{t}{L^{z}}\right),
\qquad
f(u\to 0)\propto u^{\beta},\quad f(u\to\infty)=\mathrm{const.}
$$

により、粗さ指数 $\alpha$、成長指数 $\beta$、動的指数 $z$ が定義される。EW は **線形** なので重み付け次元解析（拡散 $\nu$ の次元 $[\nu]=L^2/T$、ノイズ $\sigma$ の次元 $[\sigma]=L^{(1-d)/2}/T^{1/2}$）から

| 空間次元 $d$ | $\alpha$ | $\beta$ | $z=\alpha/\beta$ |
|:---:|:---:|:---:|:---:|
| $d=1$ | $1/2$ | $1/4$ | $2$ |
| $d=2$ | $0$ | $1/2$ | $0$ |

1+1 次元 EW では $\boxed{W\sim L^{1/2}\,t^{1/4}}$（成長期）が標準結果である。2D 界面では $\alpha=0$ となり $W$ は $L$ にほぼ依存せず $\sim t^{1/2}$ で増える。

### 実装との対応

[`edwards_wilkinson.py`](edwards_wilkinson.py) の離散更新

$$
\Delta h_i=\nu(\Delta h)_i\,\Delta t+\sigma\sqrt{\frac{\Delta t}{\Delta x}}\,\xi_i^n,
\qquad \xi_i^n\sim\mathcal N(0,1)
$$

は、1D 空間白色ノイズ $\sigma^2\delta(x-x')\delta(t-t')$ の Itô 離散化に対応する（$\Delta t$ と $\Delta x$ のスケールが一致すると連続極限の強度 $\sigma^2$ を保つ）。陽式拡散の安定条件 $\Delta t\le\Delta x^2/(2\nu)$ 下で、上記の連続極限スケーリングが数値的に確認できる。
