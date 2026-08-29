# Kawasaki–Ising 細胞選別（Mochizuki–武田モデル）

対応スクリプト: [`kawasaki_ising_cell_sorting.py`](kawasaki_ising_cell_sorting.py)（Fig. 3 再現）。

## 出典

- Mochizuki, A., Iwasa, Y., & Takeda, Y. (1996). A stochastic model for cell sorting and measuring cell–cell adhesion. *Journal of Theoretical Biology*, 179(2), 129–146. [doi:10.1006/jtbi.1996.0054](https://doi.org/10.1006/jtbi.1996.0054)
- PDF: [`Mochizuki et al. 1996 - J. Theor. Biol_.pdf`](Mochizuki%20et%20al.%201996%20-%20J.%20Theor.%20Biol_.pdf)
- 物理的背景: Kawasaki, K. (1972). “Kinetics of Ising models.” In C. Domb & M. S. Green (eds.), *Phase Transitions and Critical Phenomena*, Vol. 2, 443–501. Academic Press. ISBN 978-0-12-220302-2.
- Steinberg, M. S. (1962). On the mechanism of tissue reconstruction by dissociated cells, I. Population kinetics, differential adhesiveness, and the absence of directed migration. *Proceedings of the National Academy of Sciences of the United States of America*, 48(9), 1577–1582. <https://doi.org/10.1073/pnas.48.9.1577>
- Mochizuki, A., Takeda, Y., Ide, H., & Iwasa, Y. (1997). A stochastic model for cell sorting and its application. *Forma*, 12(2), 107–122. <https://forma.katachi-jp.com/abstract/1202/12020107.html>
- Mochizuki, A., Wada, N., Ide, H., & Iwasa, Y. (1998). Cell-cell adhesion in limb-formation, estimated from photographs of cell sorting experiments based on a spatial stochastic model. *Developmental Dynamics*, 211(3), 204–214. <https://doi.org/10.1002/(SICI)1097-0177(199803)211:3%3C204::AID-AJA2%3E3.0.CO;2-L>

Steinberg (1962) の **差次接着仮説**（differential adhesion）に、細胞の **ランダム運動** を明示的に組み込んだ格子確率モデル。数学的には **スピン交換型 Ising 模型**（Kawasaki dynamics）と等価。

---

## モデルの導出（現象論）

Steinberg (1962) の **差次接着仮説**（differential adhesion）: 同型接触を増やす配置が安定。

Mochizuki–Iwasa–Takeda は、細胞の **ランダム運動** を Kawasaki 型スピン交換で明示し、接着エネルギー $E$ と詳細つり合いから遷移確率を導く（下記 `## モデル設定` 以降の Hamiltonian → $\Delta E$ → 率）。

## モデル設定

- 2 次元正方格子（Neumann 近傍, 配位数 $z=4$）。各格子点に黒細胞 $B$ または白細胞 $W$ が **1 個**（空サイトなし）。
- 細胞数 $N = N_B + N_W$ は保存。黒細胞比率 $r_B = N_B/N$ は時間不変。
- **周期境界**（境界効果除去）。
- 隣接 2 細胞が位置を交換。B–B または W–W の交換は配置を変えないため、**B–W 近接ペアのみ** が状態を更新する。

---

## 支配方程式（連続時間マルコフ過程）

### 接着エネルギー

接触 1 本あたりの接着強度を $\lambda_{BB},\lambda_{BW},\lambda_{WW}$ とする。**総接着**（Hamiltonian）:

$$
E = \lambda_{BB}\,N_{BB} + \lambda_{BW}\,N_{BW} + \lambda_{WW}\,N_{WW},
$$

ここで $N_{BB}, N_{BW}, N_{WW}$ はそれぞれ B–B, B–W, W–W 接触本数（無向エッジ数）。

**差次接着性**（differential adhesion）:

$$
A = \lambda_{BB} - 2\lambda_{BW} + \lambda_{WW}.
$$

$A>0$ … 同型接触を増やす方向（選別）。$A<0$ … 異型接触を増やす方向（市松模様）。

### 1 回の B–W 交換によるエネルギー変化

焦点 B 細胞と交換相手 W 細胞について、W 以外の $z-1$ 近傍中の黒数 $n_B(B)$、白数 $n_W(B)$（$n_B(B)+n_W(B)=z-1$）を用いると

$$
\Delta E = A\,\bigl(n_W(B) - n_B(B)\bigr).
$$

（B–B 接触が $\Delta n = n_W(B)-n_B(B)$ 本増え、B–W 接触が $2\Delta n$ 本減る。）

### 遷移確率

配置 $p$ から $p'$ への遷移（短時間 $\Delta t$ 内）:

$$
\Pr(p \to p' \text{ in } \Delta t)
= \frac{2m}{1 + \exp\!\bigl(-\Delta E / m\bigr)}\,\Delta t.
$$

- $m$ … **運動性**（random movement の強度）。$\Delta E=0$ のとき率 $=m$ となるよう分子の係数 2 を置く。
- 率の範囲: $0 \le \text{rate} \le 2m$（$\Delta E \to \pm\infty$ で飽和）。

任意 2 配置間は B–W 交換の列で到達可能 → **詳細つり合い** から平衡分布が一意:

$$
\Pr(p) = C \exp\!\left(\frac{E(p)}{m}\right).
$$

**独立パラメータは $A$ と $m$ の 2 つのみ**。平衡は $A/m$ だけに依存し、緩和速度のスケールは $m$ が決める。

---

## Ising 模型との対応

2 成分スピン $\sigma_i \in \{+1,-1\}$（$+1\leftrightarrow B$, $-1\leftrightarrow W$）の **スピン交換型 Ising** と同型。

$\lambda_{BB}=\lambda_{WW}$ のとき、耦合定数 $J$ と $A=J$、温度 $k_BT = m$ の対応:

$$
\frac{A}{m} = \frac{J}{k_BT}.
$$

近傍交換は Kawasaki dynamics（粒子数＝各スピン数保存）に相当。

---

## 巨視的相転移（粗粒化セグリゲーション）

Appendix A より、系を $n$ 個の部分領域に分け、各領域の黒比率 $x_i$ で記述したとき、平衡確率の最尤配置は一様 $x_i=r_B$ **または** 領域間で密度が分離する配置。

**一様解の安定条件**（$r_B(1-r_B) > 0$ を仮定）:

$$
r_B(1-r_B) \lessgtr \frac{m}{4A}
\quad\Longleftrightarrow\quad
\frac{A}{m} \lessgtr \frac{1}{4\,r_B(1-r_B)}.
$$

| $r_B$ | 臨界 $A/m$ |
|-------|-----------|
| 0.1 | 25 |
| 0.3 | 3.57 |
| 0.5 | **1** |

$r_B=0.5$ では $A/m \gtrsim 1$ で **粗粒化セグリゲーション**（黒優勢域と白優勢域への分離）が起こる。Fig. 3 でも $A/m \gtrsim 1$ で大きなクラスターが観察される。

---

## ミクロscopic 秩序変数

### $q_{B/B}$（最近接相関）

ランダムに選んだ黒細胞の近傍が黒である条件付き確率:

$$
q_{B/B} = \Pr(\text{neighbor is } B \mid \text{focal cell is } B).
$$

- 完全ランダム配置: $q_{B/B} = r_B$。
- 選別進行: $q_{B/B} \gg r_B$。
- 反選別（市松）: $q_{B/B} \ll r_B$。

**ダブレット密度** $r_{BB}$（ランダムに選んだ近接ペアが B–B である確率）と $q_{B/B} = r_{BB}/r_B$。

物理学では Bethe の **order of neighbors**、生態学では Lloyd の **mean crowding** に相当。

### IBC（isolated black cells）

黒細胞のうち、黒近傍を 1 つも持たない個数。選別が進むほど IBC は減少。ペア近似:

$$
\mathrm{IBC} \approx N\,r_B\,(1-q_{B/B})^{z}.
$$

---

## ペア近似による平衡解析

### 局所密度の関係（Appendix B）

$r_B$ と $q_{B/B}$ から他の条件付き密度を表せる:

$$
q_{W/B} = 1 - q_{B/B}, \qquad
q_{B/W} = \frac{r_B(1-q_{B/B})}{1-r_B}, \qquad
q_{W/W} = \frac{1 - 2r_B + r_B q_{B/B}}{1-r_B}.
$$

ダブレット密度は $r_{BB}+2r_{BW}+r_{WW}=1$。

### 総接着と $q_{B/B}$

式 (8):

$$
E = \frac{z}{2}\,N A\, r_B\, q_{B/B} + \text{（$q_{B/B}$ に依存しない項）}.
$$

平衡における $q_{B/B}$ の分布（式 9）:

$$
\Pr(q \le q_{B/B} < q+\mathrm{d}q)
\propto V(q_{B/B})\,\exp\!\left(\frac{z N A r_B}{2m}\, q_{B/B}\right)\,\mathrm{d}q,
$$

$V(q_{B/B})$ はその $q_{B/B}$ を持つ配置数（2 次元では組合せ爆発のため実用上は 1 次元で厳密計算, Appendix C）。

### 平衡条件 $\mathrm{d}q_{B/B}/\mathrm{d}t = 0$

B–W ボンド率 $2r_{BW}$。交換による $r_{BB}$ の期待変化率（式 10）:

$$
\frac{zN}{2}\,\frac{\mathrm{d}r_{BB}}{\mathrm{d}t}
= \frac{zN}{2}\,2r_{BW}\sum_{x=-z+1}^{z-1}
\frac{x\,Q(x)\,2m}{1+\exp\bigl[-(A/m)\,x\bigr]},
$$

$x = n_W(B)-n_B(B)$。ペア近似では $n_W(B), n_B(B)$ を **独立な二項分布**（各 $z-1$ 試行, 成功確率 $q_{B/W}, q_{B/B}$）とみなし $Q(x)$ を式 (11) で与える。

平衡 $\mathrm{d}r_{BB}/\mathrm{d}t=0$ より（式 12）:

$$
0 = \sum_{x=-z+1}^{z-1} \frac{x\,Q(x)}{1+\exp\bigl[-(A/m)\,x\bigr]}.
$$

この正の解 $q_{B/B}^*$ が Fig. 5 の理論曲線。**1 次元ではペア近似 ≈ 厳密解**（Fig. 6）。

---

## 非平衡性と Fig. 3 の挙動

### 数値スキーム（本リポジトリ）

[`kawasaki_ising_cell_sorting.py`](kawasaki_ising_cell_sorting.py) では:

- 格子 $100\times100$, $r_B=0.5$, $m=0.5$, **10 000 時間単位**
- 1 時間単位 = $N^2$ 回のランダム近接ペア交換試行
- 受理確率 $\min\!\bigl(1,\; 2m/(1+e^{-\Delta E/m})\bigr)$

### Fig. 3 のパラメータと観察

| パネル | $A/m$ | パターン | 論文 $q_{B/B}$ | 本実装 $q_{B/B}$ |
|--------|-------|----------|----------------|-----------------|
| (a) | −2 | 市松状 | 0.069 | 0.264 |
| (c) | 0 | ランダム | 0.498 | 0.501 |
| (f) | 2 | 大クラスター | 0.831 | 0.838 |
| (g) | 4 | 大ドメイン | — | 0.917 |
| (h) | 6 | ドメイン再細分化 | — | 0.866 |

### 非単調性（$A/m$ が大きいとき）

平衡では $q_{B/B}^*$ は $A/m$ の増加関数だが、**有限時間のシミュレーション** では $A/m \gtrsim 2$ で

1. 平衡への緩和が極めて遅い（Kawasaki 动力学の critical slowing down）
2. 初期条件（ランダム vs 塊状）依存が残る
3. **ランダム初期条件** では $q_{B/B}$ が平衡値より低く、クラスターサイズが再び小さくなる（Fig. 3 (f)→(h)）

一方 **IBC** は $q_{B/B}$ より速く平衡に近づく（Fig. 4）。$A/m$ に対して単調減少しやすい。

### 平衡 vs シミュレーションの目安

| $A/m$ の範囲 | ペア近似 vs シミュレーション |
|-------------|---------------------------|
| $-1 \lesssim A/m \lesssim 1$ | 良好（初期条件依存小） |
| $1 \lesssim A/m \lesssim 2$ | おおむね一致 |
| $A/m \gtrsim 2$ | 非平衡；$q_{B/B}$ は平衡予測から乖離 |

---

## パラメータ一覧（Fig. 3）

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `LATTICE_SIZE` | 100 | 格子一辺 $N$ |
| `FRACTION_BLACK` | 0.5 | $r_B$ |
| `MOTILITY` | 0.5 | $m$ |
| `TIME_UNITS` | 10 000 | シミュレーション時間単位数 |
| `A_OVER_M_VALUES` | −2,…,6 | $A/m$（Fig. 3 各パネル） |
| `CROP_SIZE` | 40 | 表示クロップ |
| `RNG_SEED` | 42 | 乱数シード |

環境変数: `KAWASAKI_FIG3_QUICK=1`（小格子・短時間）, `KAWASAKI_N`, `KAWASAKI_TIME`, `KAWASAKI_SEED`。

---

## 出力

- `results/kawasaki_ising_cell_sorting_fig3.png` — Fig. 3 相当（8 パネル, $A/m$ スイープ）

---

## 関連モデル（Atlas 内）

| モデル | 違い |
|--------|------|
| [Cellular Potts](../cellular_potts_cell_sorting/cellular_potts_cell_sorting.py) (Graner & Glazier 1992) | 界面エネルギー最小化・可変細胞形状 |
| [Armstrong](../1_4_nonlocal_term/armstrong_2d/) (連続非局所) | PDE 型の非局所選別 |

Kawasaki–Ising モデルは **離散・格子・確率平衡** の最小構成として、$q_{B/B}$ と IBC により **$A/m$ を実験パターンから推定** する枠組みを与える（Forma 1997; Dev. Dyn. 1998 への応用）。
