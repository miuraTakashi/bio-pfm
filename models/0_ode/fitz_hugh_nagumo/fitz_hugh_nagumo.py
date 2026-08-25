"""
FitzHugh-Nagumo equation（移植元: Mathematica/0_ode/fitz_hugh_nagumo/FitzHughNagumo.nb）。
電気生理の縮約モデル。反応項の力学のみ（拡散なし）。
dv/dt = v - v^3/3 - w + I,  dw/dt = (v + a - b*w)/tau

外部電流 I を変えると興奮 (Excitable) と振動 (Oscillatory) の
2つのレジームが現れる。Hopf 分岐点は I ≈ 0.29 付近。

Excitable では、平衡点からの v の摂動が閾値を超えると大きな発火が起こり、
閾値以下では小さな応答のみで定常点に戻る（両方を同じ図に重ねて表示）。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def reaction_term_init(a: float, b: float, i: float, tau: float):
    """反応項 fv, fw をパラメータで定義する。"""
    def fv(v: float, w: float) -> float:
        return v - v ** 3 / 3.0 - w + i

    def fw(v: float, w: float) -> float:
        return (v + a - b * w) / tau

    return fv, fw


def run_euler(fv, fw, vw0, dt: float, n_steps: int):
    """オイラー法で時系列を計算。"""
    vw = np.array(vw0, dtype=float)
    history = [vw.copy()]
    for _ in range(n_steps - 1):
        v, w = vw
        vw = vw + dt * np.array([fv(v, w), fw(v, w)])
        history.append(vw.copy())
    return np.array(history)


def find_fixed_point(a, b, I_ext):
    """v - v³/3 + I = (v+a)/b の根を Newton 法で求める。"""
    v = -1.0
    for _ in range(50):
        f = v - v ** 3 / 3.0 + I_ext - (v + a) / b
        fp = 1.0 - v ** 2 - 1.0 / b
        v -= f / fp
    w = (v + a) / b
    return v, w


def main():
    a, b, tau = 0.7, 0.8, 10.0
    dt = 0.01

    configs = [
        {
            "I": 0.0,
            "label": "Excitable",
            "sim_length": 100.0,
            "trajectories": [
                {"v_perturb": 0.8, "tag": "supra-threshold (spike)", "c": "C0"},
                {"v_perturb": 0.12, "tag": "sub-threshold (no spike)", "c": "C2"},
            ],
        },
        {
            "I": 0.5,
            "label": "Oscillatory",
            "sim_length": 200.0,
            "trajectories": [{"v_perturb": 0.1, "tag": "limit cycle", "c": "C0"}],
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, cfg in enumerate(configs):
        I_ext = cfg["I"]
        fv, fw = reaction_term_init(a, b, I_ext, tau)
        v_fp, w_fp = find_fixed_point(a, b, I_ext)
        n_steps = int(round(cfg["sim_length"] / dt))
        t = np.arange(n_steps) * dt

        vv = np.linspace(-2.5, 2.5, 300)
        w_vnull = vv - vv ** 3 / 3.0 + I_ext
        w_wnull = (vv + a) / b
        axes[row, 1].plot(vv, w_vnull, "r--", lw=1.2, label="v-nullcline")
        axes[row, 1].plot(vv, w_wnull, "g--", lw=1.2, label="w-nullcline")
        axes[row, 1].plot(v_fp, w_fp, "ko", ms=5, zorder=5, label="fixed point")

        for tr in cfg["trajectories"]:
            vw0 = [v_fp + tr["v_perturb"], w_fp]
            history = run_euler(fv, fw, vw0, dt, n_steps)
            c = tr["c"]
            tag = tr["tag"]
            axes[row, 0].plot(t, history[:, 0], color=c, ls="-", lw=1.0,
                              label=f"v ({tag})")
            axes[row, 0].plot(t, history[:, 1], color=c, ls=":", lw=1.0,
                              label=f"w ({tag})")
            axes[row, 1].plot(history[:, 0], history[:, 1], color=c, alpha=0.75,
                              lw=1.0, label=tag)

        axes[row, 0].set_xlabel("t")
        axes[row, 0].set_ylabel("v, w")
        axes[row, 0].legend(fontsize=7, loc="best")
        axes[row, 0].set_title(f"{cfg['label']}  (I = {I_ext})")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].set_xlabel("v")
        axes[row, 1].set_ylabel("w")
        axes[row, 1].legend(fontsize=7, loc="upper left")
        axes[row, 1].set_title(f"Phase plane  (I = {I_ext})")
        axes[row, 1].set_xlim(-2.5, 2.5)
        axes[row, 1].set_ylim(-1.0, 2.0)
        axes[row, 1].grid(True, alpha=0.3)

    fig.suptitle("FitzHugh-Nagumo: Excitable vs Oscillatory", fontsize=13)
    plt.tight_layout()
    out = (Path(__file__).resolve().parent / "results" / "fitz_hugh_nagumo.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
