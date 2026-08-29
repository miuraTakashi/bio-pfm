# Armstrong 型非局所モデル 1D

**同梱スクリプト**

| ファイル | 内容 |
|----------|------|
| [`armstrong_1d.py`](armstrong_1d.py) | engulfment ノートブック直訳（既定パラメータ） |
| [`armstrong_fig12_1d.py`](armstrong_fig12_1d.py) | **Fig. 12 パラメータの 1D 検証**（2D と比較用） |

2D 拡張は [`../Armstrong_2d`](../Armstrong_2d/README.md)。

## モデルの導出（現象論）

**Armstrong** モデルは、成長因子 $a$ の **非局所自己抑制** で periodic pattern を生む。

- **仮説**: $a$ の拡散＋局所生成；非局所項は距離 $d$ 以内の $a$ 平均から引く。
- **保存則**: 拡散フラックス＋源項。
- 次節の 1D 非局所 RD が、毛包・器官間隔の現象論。

## 支配方程式

1D では感覚方向を $+x$ と $-x$ の 2 本とし、拡散に加えて $\chi\,\nabla\cdot(u\,K(u))$ 型の非局所項を離散畳み込みで近似（実装・記号はソース先頭 docstring および Mathematica ノートに準拠）。

## 出典（原著・標準文献）

- **Mathematica**: [`ArmstrongModel2D-2[858].nb`](../../../../Mathematica/1_continuous_scheme/1_4_nonlocal_term/ArmstrongModel2D-2%5B858%5D.nb)（ファイル名の `[` `]` は URL エンコード）
- モデルに応じた生物学的・数理文献はノートブックまたは原著論文を参照。

## 数値計算スキーム

周期境界・FFT 畳み込み・陽的オイラー。  
2 集団版は Armstrong et al. (2006), “A continuum approach to modelling cell–cell adhesion,” §4.2 の logistic 形
$g_{uu}=g_{vu}=u(1-u-v),\ g_{vv}=g_{uv}=v(1-u-v)$（$u+v \lt 1$）を用い、
Fig.9 キャプションの 4 シナリオ

- A Mixing: $(S_u,S_v,C)=(25,\,7.5,\,22.5)$
- B Engulfment: $(250,\,25,\,50)$
- C Partial engulfment: $(25,\,25,\,12.5)$
- D Complete sorting: $(25,\,7.5,\,0)$

を `t=0,5,10,500` のスナップショットで可視化する（`ARMSTRONG_FIG12_1D_MODE=fig9`）。

## Fig. 12 の 1D 検証

[`armstrong_fig12_1d.py`](armstrong_fig12_1d.py) は Fig. 12 と同じ $(S_u,S_v,C)$ を **1D 領域 $L=10$**（`dx=0.1`）で積分し、2D [`armstrong_fig12.png`](../Armstrong_2d/armstrong_fig12.png) との比較に使う。

```bash
cd python/1_continuous_scheme/1_4_nonlocal_term/Armstrong_1d
python3 armstrong_fig12_1d.py
ARMSTRONG_FIG12_1D_QUICK=1 python3 armstrong_fig12_1d.py   # t=50
ARMSTRONG_FIG12_1D_MODE=fig9 python3 armstrong_fig12_1d.py  # 論文 Fig. 9 用 1D パラメータ, L=20
```

出力: `results/armstrong_fig12_1d.png`, `data/armstrong_fig12_1d.csv`（終状態の std / mean）。

## パラメータ一覧

`simulate_armstrong_1d` の `L`, `dx`, `r0`, `dt`, `diffusion`, `chi`, `n_steps`, `seed` 等。調整の目安は [`armstrong_1d.py`](armstrong_1d.py) docstring の箇条書きを参照。

## 数理解析

非局所カーネルのフーリエ記述により線形安定性を解析しやすい場合がある。
