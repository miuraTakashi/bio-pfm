# Mimura 細菌コロニー反応–拡散（1D / 2D・形態図）

対応スクリプト: [`mimura_bacterial_colony.py`](mimura_bacterial_colony.py)。

Mimura–Sakaguchi–Matsushita (2000) の活性／不活性細菌＋栄養の反応–拡散系を再現する。  
**Section 3** で 1D の進行波・振動パルス・クラスタ（Fig. 3.2 / 3.4）、**Section 4** で 2D 形態 A–E（DLA / Eden / 同心円 / 円盤 / DBM）を出力する。  
Eden 以外の形態は論文キャプションの $(d,v_0)$ を基本とし、同心円（C）のみ正方形差分での再現のため 1D 振動帯側へわずかに寄せる（後述）。形態図そのもの（Fig. 3.1 / 4.7 の全面スイープ）は行わない。

格子上の離散アナログ: [`dla`](../../../2_discrete_scheme/2_1_lattice_base/dla/dla.md)（形態 A）、[`eden`](../../../2_discrete_scheme/2_1_lattice_base/eden/eden.md)（形態 B）。1D 進行波の最小モデルは [`fisher_1d`](../fisher_1d/fisher_1d.md)。Mathematica 移植元はない（文献ベースの実装）。

## モデルの導出（現象論）

- **現象**: *Bacillus subtilis* コロニーは寒天硬度 $C_a$ と栄養 $C_n$ に応じて DLA・Eden・同心円・円盤・DBM など多様な形態を示す。
- **仮説**: 細菌を活性 $u$（移動・増殖）と不活性 $w$（静止）に粗視化し、栄養 $v$ は拡散・消費される。一度不活性化すると再活性化しない。
- **保存則**: 栄養は Fick 拡散と増殖消費 $uv$ のみ。活性細菌は拡散フラックス $\nabla\cdot\bigl(d(b)\nabla u\bigr)$ に増殖 $uv$ と不活性化 $-a(u,v)\,u$ が載る。不活性 $w$ は拡散せず、不活性化の受け皿。
- **構成**: 増殖は $uv$、不活性化率 $a(u,v)$ は $u$ と $v$ の減少関数。軟寒天では定数拡散、硬寒天では集団圧による $d(b)=d\,b$。
- **パラメータ対応**: 無次元の $d=d_A/d_N$ と初期栄養 $v_0$ が実験の $(C_a^{-1}, C_n)$ に対応する。次節の無次元系（式 (2.5)–(2.6)）がこの骨格の具体形。

## 支配方程式

無次元形（式 (2.5)–(2.6)）。$u$: 活性細菌、$v$: 栄養、$w$: 不活性細菌、$b=u+w$（総細菌密度）。栄養の拡散係数は 1。

$$
\frac{\partial u}{\partial t}
=\nabla\cdot\bigl(d(b)\nabla u\bigr)+uv-a(u,v)\,u
$$

$$
\frac{\partial v}{\partial t}=\nabla^2 v-uv
$$

$$
\frac{\partial w}{\partial t}=a(u,v)\,u
$$

$$
a(u,v)=\frac{1}{(1+u/a_1)(1+v/a_2)},\qquad
a_1=\frac{1}{2400},\; a_2=\frac{1}{120}
$$

| 培地 | 拡散 | 式番号 |
|------|------|--------|
| 軟寒天 | $d(b)=d$（定数） | (3.1) 1D / (4.1) 2D |
| 硬寒天 | $d(b)=d\,b$ | (3.4) 1D / (4.2) 2D |

**境界条件**（Neumann・ゼロフラックス）

- 1D: $x\in[0,L]$（半無限の近似）。両端で $\partial_x u=\partial_x v=0$。
- 2D: 正方形プレート上の Neumann（式 (2.8)）。

**初期条件**

- 1D: 左端 droplet $u(x,0)=u_0$（$x\le r_0$）、$v\equiv v_0$、$w\equiv 0$（式 (3.2)–(3.3)）。
- 2D: 中心円盤 inoculum $u=u_0$（$x^2+y^2\le r_0^2$）、$v\equiv v_0$、$w\equiv 0$（式 (2.7)）。

ODE 閾値 $a(0,v^*)=v^*$ より $v^*\approx 0.0872$。これより十分大きい $v_0$ で進行波、近傍で消光／クラスタ。

## 出典（原著・標準文献）

- Mimura, M., Sakaguchi, H., & Matsushita, M. (2000). Reaction–diffusion modelling of bacterial colony patterns. *Physica A* **282**, 283–303. <https://doi.org/10.1016/S0378-4371(00)00085-6>
- Matsushita, M., Wakita, J., Itoh, H., Ràfols, I., Matsuyama, T., Sakaguchi, H., & Mimura, M. (1998). Interface growth and pattern formation in bacterial colonies. *Physica A* **249**, 517–524. <https://doi.org/10.1016/S0378-4371(97)00511-6>（統一モデルの先行概要）
- Hosono, Y., & Ilyas, B. (1995). Traveling waves for a simple diffusive epidemic model. *Mathematical Models and Methods in Applied Sciences* **5**, 935–966. <https://doi.org/10.1142/S0218202595000504>（定数不活性化率の進行波速度）

同フォルダの PDF: [`Mimura et al. 2000 - Reaction-diffusion modelling of bacterial colony patterns.pdf`](Mimura%20et%20al.%202000%20-%20Reaction-diffusion%20modelling%20of%20bacterial%20colony%20patterns.pdf)

## 数値計算スキーム

- **時間**: 陽的オイラー。栄養拡散 $D=1$。1D は $\Delta t=\min(\mathtt{dt},\,0.4\,\Delta x^2/4)$、2D は Mehrstellen のスペクトル半径 $16/(3\Delta x^2)$ に対し $\Delta t=\min(\mathtt{dt},\,0.45\cdot 3\Delta x^2/16)$。各ステップ後 $u,v$ を非負にクリップ。
- **空間（1D）**: 等間隔格子、3 点中心差分。非線形拡散は $\nabla\cdot(b\nabla u)$ の保存形（界面平均 $b$）。既定 $L=500$, $\Delta x=0.5$。
- **空間（2D）**: 等方寄り 9 点 Neumann（Mehrstellen: cardinal $2/3$ + 対角 $1/6$）。既定 $L=480$, $\Delta x=2$。
- **境界**: Neumann（ゼロフラックス）。1D は端点の反射ステンシル、2D は `np.pad(..., mode="edge")`。
- **対称性破れ**: 凍結 $\chi_u,\chi_v\in[1-\chi_0,1+\chi_0]$ を拡散に乗せる（構造格子の完全回転対称を破る）。
- **早期終了**: コロニー半径がプレート半径の約 46% に達したら停止。
- **負荷**: Section 4 全パネル＋GIF は数分〜十数分。短縮は `MIMURA_QUICK=1`。

## パラメータ一覧

### 共通定数・格子

`main()` / `run_section3` / `run_section4` の既定。無次元。

| 識別子 | 記号 | 既定値 | 意味 |
|--------|------|--------|------|
| `A1` | $a_1$ | $1/2400$ | 不活性化の $u$ スケール |
| `A2` | $a_2$ | $1/120$ | 不活性化の $v$ スケール |
| `L` | $L$ | 500（1D）/ 480（2D） | 領域長 |
| `dx` | $\Delta x$ | 0.5（1D）/ 2（2D；C は 1） | 格子間隔 |
| `dt` | $\Delta t$ | ケース既定を CFL で制限 | 時間刻み（1D 実効 ≈0.025、2D 既定 0.1） |
| `u0` | $u_0$ | 1.0（1D）/ 1.06（2D） | 接種振幅 |
| `r0` | $r_0$ | 3（1D）/ 4（2D） | 接種半径 |

$(a_1,a_2)$ と各図の $(d,v_0)$ は論文に合わせ固定する。

### Section 3（Fig. 3.2 / 3.4）— `Section3Case`

| パネル | 形態 | `d` | `v0` | `u0` | `t_final` | 備考 |
|--------|------|-----|------|------|-----------|------|
| (a) | travelling wave（T） | 0.1 | 0.117 | 1.0 | 2500 | |
| (b) | oscillatory（O・弱） | 0.05 | 0.129 | 1.0 | 3500 | 同心円の 1D 対応帯 |
| (c) | oscillatory（O・強） | 0.05 | 0.108 | 0.8 | 4500 | |
| (d) | clustering（E） | 0.1 | 0.0875 | 1.0 | 9000 | 長時間後にパルス消光 |
| Fig. 3.4 | nonlinear travelling | 0.05 | 0.25 | 1.0 | 5000 | $d(b)=d\,b$；`r0=4` |

### Section 4（形態 A–E）— `ColonyCase`

| パネル | 形態 | `v0` | `d` | 拡散 | `chi0` | `t_final` | `seed` | `dx` / `L` |
|--------|------|------|-----|------|--------|-----------|--------|------------|
| (A) | DLA-like（Fig. 4.4） | 0.087 | 0.05 | 定数 | 0.08 | 4500 | 11 | 既定 2 / 480 |
| (B) | Eden-like（Fig. 4.9） | 0.25 | 0.05 | $d\,b$ | 0.05 | 4000 | 12 | 既定 |
| (C) | concentric ring-like | 0.125 | 0.05 | 定数 | 0.01 | 8000 | 13 | 1 / 480 |
| (D) | disk-like（Fig. 4.1） | 0.25 | 0.25 | 定数 | 0.02 | 900 | 14 | 既定 |
| (E) | DBM-like（Fig. 4.2） | 0.071 | 0.12 | 定数 | 0.08 | 3200 | 15 | 既定 |

2D の初期振幅は `u0=1.06`, `r0=4`（文献再現で常用）。

### Concentric ring（C）の注意

論文 Fig. 4.6 キャプションは $d=0.05$, $v_0=0.1$。これは 1D 振動帯（A-ii）と分岐帯の境界付近で、粗い $\Delta x=2$ の正方形差分では花弁状分岐になりやすい。本実装では:

1. **弱い振動帯**へ寄せる: $v_0=0.125$（Fig. 3.2b の $0.129$ 近傍）
2. **リング波長を解像**: $\Delta x=1$（1D の $b$ 波列は波長 $O(50\text{–}80)$）
3. **$\chi$ を小さく**: $\chi_0=0.01$
4. 表示で動径平均を引きリングコントラストを強調（`enhance_rings=True`）

論文も「各リングは分岐構造を持ちうる」「周期成長のモデル化はさらなる検討が必要」と述べている。

### クイック実行

`MIMURA_QUICK=1`（`true` / `yes` 可）のとき。

| 識別子 | 値 |
|--------|-----|
| `L`（1D / 2D） | 300 / 320（C は $L\le 360$, $\Delta x\ge 1.25$） |
| `t_final` | 既定の $0.45$ 倍 |
| GIF フレーム | 24（既定 40） |

## 数理解析

- 拡散なし系 (2.10) の興奮性: $a(u,v)=v$ の曲線 $v=\varphi(u)$ と $v^*$ で進行／消光が分かれる。
- 定数 $a$ の流行型系では $a\lt v_0$ のとき進行波速度の下界 $c\ge 2\sqrt{d(v_0-a)}$（Hosono）。可変 $a(u,v)$ では数値で Fig. 3.1 / 4.7 型の形態図を得る。
- 本スクリプトは線形安定性の解析プロットは行わない（形態の数値再現が主目的）。

## 出力

| ファイル | 内容 |
|----------|------|
| [`results/mimura_bacterial_colony_section3_fig32.png`](results/mimura_bacterial_colony_section3_fig32.png) | Fig. 3.2 カイモグラフ（上段 $u$、下段 $b$） |
| [`results/mimura_bacterial_colony_section3_fig34.png`](results/mimura_bacterial_colony_section3_fig34.png) | Fig. 3.4 カイモグラフ＋最終プロファイル |
| [`results/mimura_bacterial_colony_section4.png`](results/mimura_bacterial_colony_section4.png) | Section 4 形態 A–E |
| `results/mimura_bacterial_colony_A.gif` … `_E.gif` | 各 2D ケースの形成過程 |

## 実行

```bash
cd python/1_continuous_scheme/1_1_diffusion_term/mimura_bacterial_colony
python3 mimura_bacterial_colony.py
```

| 環境変数 | 効果 |
|----------|------|
| `MIMURA_QUICK=1` | $L$・$t_{\mathrm{final}}$ を短縮（上表） |
| `MIMURA_SECTION3_ONLY=1` | Section 3（1D）のみ |

調整は `Section3Case` / `ColonyCase` の `t_final`, `chi0`, `dx`, `L`, `seed` など。
