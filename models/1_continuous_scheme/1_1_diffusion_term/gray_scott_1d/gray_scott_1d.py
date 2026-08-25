"""
Gray-Scott モデル 1D — 特徴的な (F, k) パラメータセットによる時空間パターン。
u' = -u*v^2 + F*(1-u) + Du*Laplacian(u)
v' =  u*v^2 - (F+k)*v + Dv*Laplacian(v)
反応項は RK4、拡散は陽的オイラー・周期境界で離散ラプラシアン。
横軸 x、縦軸 t のヒートマップ（カイモグラフ）。`extent` で物理座標を明示。
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_gs1d(F, k, du=0.00002, dv=0.00001, dx=0.01,
                  domain_size=1.0, sim_length=10000.0,
                  n_kymo=200, seed=42):
    """1D Gray-Scott を実行し v の (n_time × n_space) と各行の時刻 t を返す。"""
    n = int(round(domain_size / dx))
    dt = (dx ** 2 / du) / 4.0
    n_steps = int(round(sim_length / dt))
    df = dt / (dx * dx)
    rec_interval = max(1, n_steps // n_kymo)
    Fk = F + k

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.9, 1.0, size=n)
    x_idx = np.arange(1, n + 1, dtype=float)
    v = (1.0 - np.tanh((x_idx - n / 2) / 5.0) ** 2
         + rng.uniform(0.0, 0.05, size=n))

    kymo = []
    times_rec = []
    for step in range(n_steps):
        if step % rec_interval == 0:
            kymo.append(v.copy())
            times_rec.append(step * dt)

        uv2 = u * v * v
        ku1 = -uv2 + F * (1.0 - u)
        kv1 = uv2 - Fk * v

        u2 = u + 0.5 * dt * ku1
        v2 = v + 0.5 * dt * kv1
        uv2_ = u2 * v2 * v2
        ku2 = -uv2_ + F * (1.0 - u2)
        kv2 = uv2_ - Fk * v2

        u3 = u + 0.5 * dt * ku2
        v3 = v + 0.5 * dt * kv2
        uv2_ = u3 * v3 * v3
        ku3 = -uv2_ + F * (1.0 - u3)
        kv3 = uv2_ - Fk * v3

        u4 = u + dt * ku3
        v4 = v + dt * kv3
        uv2_ = u4 * v4 * v4
        ku4 = -uv2_ + F * (1.0 - u4)
        kv4 = uv2_ - Fk * v4

        u_rk = u + dt * (ku1 + 2 * ku2 + 2 * ku3 + ku4) / 6.0
        v_rk = v + dt * (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6.0

        lap_u = np.roll(u, 1) + np.roll(u, -1) - 2 * u
        lap_v = np.roll(v, 1) + np.roll(v, -1) - 2 * v
        u = u_rk + df * du * lap_u
        v = v_rk + df * dv * lap_v

    kymo.append(v.copy())
    times_rec.append(n_steps * dt)
    return np.array(kymo), np.array(times_rec, dtype=float)


PARAMS = [
    (0.020, 0.048, "Soliton"),
    (0.022, 0.052, "Chaotic"),
    (0.024, 0.054, "Expanding"),
    (0.026, 0.056, "Slow replication"),
    (0.030, 0.058, "Traveling waves"),
    (0.034, 0.059, "Stripe formation"),
    (0.038, 0.061, "Branching"),
    (0.040, 0.060, "Pulse splitting"),
    (0.042, 0.063, "Mixed mode"),
    (0.046, 0.063, "Stable stripes"),
]


def main():
    n_sets = len(PARAMS)
    ncols = 5
    nrows = (n_sets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.4 * nrows))
    axes = axes.flatten()

    for i, (F, k, label) in enumerate(PARAMS):
        print(f"  [{i+1}/{n_sets}] F={F:.3f}, k={k:.3f} ({label})")
        kymo, t_rec = simulate_gs1d(F, k)
        t_end = float(t_rec[-1])
        axes[i].imshow(
            kymo,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=[0.0, 1.0, t_end, 0.0],
        )
        axes[i].set_title(f"F={F}, k={k}\n{label}", fontsize=8)
        axes[i].set_xlabel("x", fontsize=7)
        axes[i].set_ylabel("t", fontsize=7)
        axes[i].tick_params(labelsize=5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Gray-Scott 1D — v(x, t) heatmap (x horizontal, t vertical down)", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "results" / "gray_scott_1d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
