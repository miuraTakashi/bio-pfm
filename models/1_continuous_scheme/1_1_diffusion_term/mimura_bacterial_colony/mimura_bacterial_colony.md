# Mimura 細菌コロニー統一モデル（Fig. 5）

**スクリプト:** [`mimura_bacterial_colony.py`](mimura_bacterial_colony.py)

## モデルの導出（現象論）

*Bacillus subtilis* コロニーの形態多様性を、活性細胞 $b$・不活性細胞 $s$・栄養 $n$ の反応拡散で記述する統一モデル（Matsushita et al., Physica A **249**, 517–524, 1998）。

- 活性 / 不活性の二状態が内部状態の粗視化。
- 栄養は拡散し、細菌に消費される（$g(n)=n$）。
- 環境の硬さ・栄養濃度は無次元パラメータ $(n_0, d)$ に対応（$d$ は細菌と栄養の拡散係数比）。

## 支配方程式（無次元形, 式 (1)–(3)）

$$\frac{\partial b}{\partial t} = \nabla\cdot(D\nabla b) + n b - a(b,n)\,b$$

$$\frac{\partial s}{\partial t} = a(b,n)\,b$$

$$\frac{\partial n}{\partial t} = \nabla^2 n - n b$$

Fig. 5 は **総細胞密度** $b+s$ のパターン。式 (1)(3) を解き、$s$ は式 (2) で積分する。

### 閉形式の補足

原著は $a(b,n)$ の具体形を簡略化しており、後続研究（例: Mimura–Sakaguchi–Matsushita, Physica A **282**, 2000）では
$a(b,n)=a_0/((1+a_1 b)(1+a_2 n))$ 型が用いられる。本実装では同型の現象論近似

$$a(b,n)=\frac{a_0}{(1+a_1 b)\,(1+a_2\max(0,\,1-n/n_0))}$$

を用い、論文キャプションの $(n_0,d)$ と各パネル用の $(a_0,a_1,t_{\mathrm{final}})$ で Fig. 5 の 5 形態を再現する。
周期 FFT だけでは栄養が全域で枯渇するため、格子外周に **栄養バス**（$n=n_0$、$b=s=0$）を置くリムを毎ステップ適用している。

**Fig. 5b（Eden-like）** のみ $D = d\,b$（論文: $d=0.05\,b$）。$\nabla\cdot(d b\nabla b)=db\nabla^2b+d|\nabla b|^2$ と分解し、$db\nabla^2b$ を **半陰式 FFT**（各ステップで $D^n=d\,b^n$ を凍結）、$d|\nabla b^n|^2$ を陽的に扱う。他パネルは $D=d$ 定数で同じ半陰式 FFT（`turing_2d` と同型）。

## 出典

- Matsushita, M. et al., *Interface growth and pattern formation in bacterial colonies*, Physica A **249**, 517–524 (1998). [`1-s2.0-S0378437197005116-main.pdf`](1-s2.0-S0378437197005116-main.pdf)
- 関連: Mimura, M., Sakaguchi, H., Matsushita, M., *Reaction–diffusion modelling of bacterial colony patterns*, Physica A **282**, 283–303 (2000).

## Fig. 5 パラメータ（論文キャプション）

| パネル | 形態 | $n_0$ | $d$ | 備考 |
|--------|------|-------|-----|------|
| (a) | DLA-like | 0.98 | 0.05 | 定数 $D$ |
| (b) | Eden-like | 1.2 | 0.05 | $D=0.05\,b$ |
| (c) | concentric ring-like | 1.2 | 0.05 | 定数 $D$ |
| (d) | disk-like | 1.5 | 0.12 | 定数 $D$ |
| (e) | DBM-like | 0.855 | 0.12 | 定数 $D$ |

## 数値計算スキーム

- 格子: 周期境界、$dx=1$、既定 $200\times200$。
- $b$, $n$: 反応項は陽的、拡散は `build_implicit_kernel_2d` による半陰式 1 ステップ（FFT）。$b$ の Eden モードのみ $D^n$ を毎ステップ更新。
- $s$: 陽的オイラー。
- 初期条件: 論文式 (5) に従い、中心 1 セルの点状 inoculum（$b$）、$n(x,0)=n_0$ 一様、$s=0$（乱数なし）。
- 外周リム: 幅 $\approx L/12$ で $n=n_0$、$b=s=0$（無限栄養浴の近似）。

## 実行

```bash
cd python/1_continuous_scheme/1_1_diffusion_term/mimura_bacterial_colony
python3 mimura_bacterial_colony.py
```

短縮: `MIMURA_FIG5_QUICK=1 python3 mimura_bacterial_colony.py`（$128^2$ 格子、$t_{\mathrm{final}}$ を 0.35 倍）。

出力:

- [`results/mimura_bacterial_colony_fig5.png`](results/mimura_bacterial_colony_fig5.png) — 最終状態 5 パネル
- [`results/mimura_bacterial_colony_fig5_{a–e}.gif`](results/) — 各パネルの形成過程（総密度 $b+s$、対数スケール表示）

## パラメータ調整

`Fig5Case` で各パネルの `t_final`, `a0`, `a1`, `seed` を変更可能。論文は $a(b,n)$ の細部を省略しているため、形態の再現は $(n_0,d)$ を固定し $(a_0,a_1)$ と積分時間でキャリブレーションしている。
