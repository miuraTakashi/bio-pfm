# 波列（3 変数離散時間マップ＋1D 拡散）

対応スクリプト: [`wavetrain.py`](wavetrain.py)。

## モデルの導出（現象論）

**Lambda–Omega 型** 反応拡散系は、局所で limit cycle 振動し、空間結合で **波列（wavetrain）** が生じる。

- **仮説**: 複素振幅または 2 変量の位相振動子＋拡散結合。
- **近似**: 遠方から見た包絡線・位相の慢変数展開（実装は具体 RD）。
- 拡散が位相勾配を平滑化し、次節の系で波数選択・欠陥が現れる。

## 支配方程式（離散時間ステップ）

周期境界上の 1 次元格子。`diffusion[l] = (roll(l,1)+roll(l,-1)-2l)/dx^2`。1 ステップ:

$$
u \leftarrow u + \Delta t\bigl(0.6u - v + d_u \nabla^2 u - w\bigr) - 0.2\,u^3,
$$

$$
v \leftarrow v + \Delta t\bigl(1.5u - 2v + \nabla^2 v - w\bigr),
$$

$$
w \leftarrow w + \Delta t\,(u+v).
$$

（$d_u$ はコード中 `du_coeff`、既定 0.01。）

## 出典（原著・標準文献）

- 連続極限で反応–拡散に結びつく **波列・進行波**の一般理論: Cross, M. C., & Hohenberg, P. C. (1993). Pattern formation outside of equilibrium. *Reviews of Modern Physics*, 65(3), 851–1112. <https://doi.org/10.1103/RevModPhys.65.851>
- Kawamura, M., Sugihara, K., Takigawa-Imamura, H., Ogawa, T., & Miura, T. (2021). Mathematical Modeling of Dynamic Cellular Association Patterns in Seminiferous Tubules. *Bulletin of Mathematical Biology*, 83, 28. <https://doi.org/10.1007/s11538-021-00863-x>
- Sugihara, K., Sekisaka, A., Ogawa, T., & Miura, T. Segmented wavetrains and sites of reversal in the mouse seminiferous tubules. <https://catalog.lib.kyushu-u.ac.jp/ja/recordID/7410558?from=gallery>
- 本スクリプトの 3 変数形は **アトラス用の特定パラメータ化**であり、単一の生物学的実験論文との対応は主張しない。

## 数値計算スキーム

- **空間**: 周期境界、`np.roll` による 3 点ラプラシアン。
- **時間**: 陽的オイラー型の離散マップ（上式）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `domain_size` | $6.28\times 4$ | 領域長 |
| `dx` | 0.1 | 空間刻み |
| `dv` | 1.0 | $\Delta t = (dx^2/dv)/4$ のスケール |
| `dt` | $(dx^2/dv)/4$ | 時間刻み |
| `du_coeff` | 0.01 | $u$ の拡散係数 |
| `amplitude` | 0.01 | 初期一様乱数の幅 |
| `n_steps` | 20000 | ステップ数 |
| `record_every` | 40 | 記録間隔 |
| `seed` | 42 | RNG シード |

## 数理解析

- 線形化（$u,v,w\approx 0$）した Fourier モード $k$ で、行列 $A(k)$ の成分は $A_{11}=0.6-d_u k^2$, $A_{12}=-1$, $A_{13}=-1$, $A_{21}=1.5$, $A_{22}=-2-k^2$, $A_{23}=-1$, $A_{31}=1$, $A_{32}=1$, $A_{33}=0$。その固有値を計算し、支配的モード（dominant mode）の **実部 / 虚部**を表示する。
- 数値場 $u(x,t)$ の支配的モードから波数 $k_{\mathrm{num}}$、波長 $\lambda_{\mathrm{num}}$ を推定。
- 同モードの位相 $\phi(t)$ の線形フィット傾きから $\omega=-d\phi/dt$、位相速度 $c=\omega/k_{\mathrm{num}}$ を推定。

## 出力

- `results/wavetrain.png`: 従来どおりの数値可視化（カイモグラフ + スナップショット）
- `results/wavetrain_analysis_dispersion.png`: 支配的モード周りの固有値の Re/Im vs $k$、最不安定 $k_*$ と数値 $k_{\mathrm{num}}$
- `results/wavetrain_analysis_wave.png`: 支配的モード位相・$\omega$・位相速度、$k_{\mathrm{num}}$ での 3 固有値
