# Amari 二層神経場（進行波, Fig. 10）

**同梱ファイル**

| ファイル | 内容 |
|----------|------|
| [`amari_neural_field.py`](amari_neural_field.py) | Fig. 10 の進行波プロファイル再現と PDE 波速検証 |
| [`amari_neural_field_fig10.png`](amari_neural_field_fig10.png) | 進行波の理論プロファイル図（$u_1$, $u_2$） |
| [`amari_neural_field_pde_check.png`](amari_neural_field_pde_check.png) | PDE 時間発展から推定した波速の検証図 |
| [`amari_neural_field_pulse_nucleation_check.png`](amari_neural_field_pulse_nucleation_check.png) | 静止解への局所摂動からパルス生成・移動を確認する図 |
| [`BF00337259.pdf`](BF00337259.pdf) | 原著論文 PDF |

## モデルの概要

興奮層 $u_1$ と抑制層 $u_2$ の 2 層神経場を扱う。興奮層は空間畳み込み（非局所相互作用）、抑制層は局所入力 $w_3 f(u_1)$ で駆動される。活性化関数は閾値型
$f(u)=1\;(u>0),\,0\;(u\le 0)$ を使う。

## 支配方程式

$$
\tau \partial_t u_1
= -u_1
+ \int w_1(x-x') f[u_1(x')]\,dx'
- \int w_2(x-x') f[u_2(x')]\,dx'
+ h_1
$$

$$
\tau \partial_t u_2
= -u_2 + w_3 f[u_1(x)] + h_2
$$

カーネルはガウス形:

$$
w_i(x)=\frac{A_i}{\sqrt{2\pi}\sigma_i}
\exp\!\left(-\frac{x^2}{2\sigma_i^2}\right)
$$

## 数値計算の流れ

1. 共動座標 $y=x-vt$ で定常進行波 $u_i(x,t)=g_i(y)$ を仮定。  
2. 論文 §8.2 の条件 $g_1(0)=g_1(a)=0$ を `scipy.optimize.root` で解き、$(a,v)$ を推定。  
3. 得られた $(a,v)$ で $g_1(y), g_2(y)$ を評価し、Fig. 10 相当のプロファイルを描画。  
4. さらに PDE を時間積分し、波前位置の線形フィットから速度を計測して理論値 $v$ と比較。
5. 追加検証として、静止解 $(u_1,u_2)=(h_1,h_2)$ に局所ガウス摂動を与え、パルスが生じて移動するかを確認。

## 既定パラメータ（論文数値例）

- $A_1=2.0$, $A_2=4.0$
- $\sigma_1=1.0$, $\sigma_2=1.5$
- $w_3=2.0$, $h_1=-0.1$, $h_2=-1.0$, $\tau=1.0$
- 初期推定値 `a0=7.6`, `v0=7.3`（論文記載値に対応）

## 実行方法

```bash
cd python/1_continuous_scheme/1_4_nonlocal_term/amari_neural_field
python3 amari_neural_field.py
```

実行時に標準出力へ
- 推定された波の幅 `a`
- 推定された波速 `v`
- PDE からのフィット速度
- 摂動初期値テストでのパルス生成有無と右エッジ速度

が表示される。

## 出力

- `amari_neural_field_fig10.png`: Fig. 10 相当の $u_1, u_2$ プロファイル
- `amari_neural_field_pde_check.png`: PDE スナップショットと波前速度フィット
- `amari_neural_field_pulse_nucleation_check.png`: 静止解 + 局所摂動からのパルス生成・移動の確認

## 出典

- S. Amari, *Dynamics of pattern formation in lateral-inhibition type neural fields*, Biol. Cybernetics **27**, 77-87 (1977)
- 同梱論文: [`BF00337259.pdf`](BF00337259.pdf)
