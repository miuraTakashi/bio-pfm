# Euler–Bernoulli 軸方向座屈（1D、成長軸荷重）

対応スクリプト: [`euler_bernoulli_buckling.py`](euler_bernoulli_buckling.py)。

成長する円筒（M1 等）の **軸方向圧縮** により、横方向変位 $h(x,t)$ の座屈が生じる 1D デモ。周期境界・FFT 半陰式。

## モデルの導出（現象論）

- **現象**: 軸方向伸長 $\Delta L(t)=\nu t$ に伴い有効軸力 $\bar P(t)=ES\Delta L/L$ が増大し、横モードが不安定化（座屈）。
- **仮説**: Euler–Bernoulli 梁の曲率剛性 $EI h_{xxxx}$、基盤反力 $k_s h$、幾何非線形による軸力の再正規化（$P^n$ が $h_x$ の二乗積分に依存）。
- **近似**: 1D、周期境界、オーバーダンプ的な $\mu \partial_t h$。
- 次節の半陰式更新は、各フーリエモードで線形化した支配方程式から得られる。

## 支配方程式

$$
\mu \frac{\partial h}{\partial t}
= -\bar P(t)\,\frac{\partial^2 h}{\partial x^2}
- EI \frac{\partial^4 h}{\partial x^4}
- k_s h
+ \frac{ES}{2L}\Bigl(\int_0^L h_x^2\,dx\Bigr)\frac{\partial^2 h}{\partial x^2},
$$

$$
\bar P(t)=\frac{ES}{L}\Delta L(t),\quad \Delta L(t)=\nu t.
$$

半陰式（$P^n$ 明示、$h^{n+1}$ 陰的）のモード更新は [`euler_bernoulli_buckling.py`](euler_bernoulli_buckling.py) 参照。

## 出典

- Abramian, A. K., Vakulenko, S. A., van Horssen, W. T., & Lukichev, D. V. (2021). Dynamics and buckling loads for a vibrating damped Euler–Bernoulli beam connected to an inhomogeneous foundation. *Archive of Applied Mechanics*, 91(4), 1291–1308. <https://doi.org/10.1007/s00419-020-01823-y>

本実装では、この文献で扱われる減衰 Euler–Bernoulli 梁、軸方向圧縮による座屈、および弾性基盤という構成を参照し、周期境界上のオーバーダンプ系へ簡略化している。

## 数値計算スキーム

- **空間**: 周期境界、FFT（`rfft` / `irfft`）。
- **時間**: 半陰式（線形項 $EI k^4 + k_s - P^n k^2$ を陰的）。
- **短縮**: 環境変数 `EULER_BUCKLING_STEPS`。

## パラメータ一覧

| 識別子 | 意味 | 既定（コード） |
|--------|------|----------------|
| `L_MM` | 領域長 [mm] | 100 |
| `NU_ELONGATION_MM_PER_YR` | 伸長速度 $\nu$ | 0.612 |
| `N_GRID` | 格子点数 | 1000 |
| `DT` | 時間刻み [year] | 0.05 |
| `T_MAX` | 終了時刻 [year] | 60 |

幾何・材料定数は `euler_bernoulli_buckling.py` の `R_I_MM`, `T_W_MM`, `E_YOUNG`, `K_S`, `MU` 等を参照。

## 出力

- `results/euler_bernoulli_buckling.png` — $h(x,t)$ カイモグラムと $P(t)$、$\bar P(t)$

## 実行

```bash
cd python/1_continuous_scheme/1_3_biharmonic_term/euler_bernoulli_buckling
python3 euler_bernoulli_buckling.py
```

| 環境変数 | 意味 |
|----------|------|
| `EULER_BUCKLING_STEPS` | 積分ステップ数の上書き（短縮） |
