"""
1D verification for Armstrong et al. (2006) Fig. 12 Steinberg parameters.

Uses the same antisymmetric-kernel + logistic g scheme as armstrong_1d.py
(ArmstrongCellSortingEngulfment.nb), with Fig. 12 adhesion values on domain L=10.

Compare with 2D output: ../armstrong_2d/armstrong_fig12.png

Outputs:
  armstrong_fig12_1d.png  — 4 scenarios × snapshots (t=0, 50, 125 by default)
  armstrong_fig12_1d.csv  — final-time statistics per scenario
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from armstrong_1d import list_convolve_mathematica


@dataclass(frozen=True)
class SteinbergScenario:
    name: str
    label: str
    su: float
    sv: float
    c: float


# Fig. 12 caption (2D parameters; run in 1D on L=10 for cross-check)
FIG12_SCENARIOS: tuple[SteinbergScenario, ...] = (
    SteinbergScenario("mixing", "(a) Mixing", su=10.0, sv=3.0, c=9.0),
    SteinbergScenario("engulfment", "(b) Engulfment", su=100.0, sv=10.0, c=20.0),
    SteinbergScenario("partial", "(c) Partial engulfment", su=10.0, sv=10.0, c=5.0),
    SteinbergScenario("sorting", "(d) Complete sorting", su=10.0, sv=3.0, c=0.0),
)

# Fig. 9 caption (1D paper values on L=20) — set ARMSTRONG_FIG12_1D_MODE=fig9
FIG9_SCENARIOS: tuple[SteinbergScenario, ...] = (
    SteinbergScenario("mixing", "(a) Mixing", su=25.0, sv=7.5, c=22.5),
    SteinbergScenario("engulfment", "(b) Engulfment", su=250.0, sv=25.0, c=50.0),
    SteinbergScenario("partial", "(c) Partial engulfment", su=25.0, sv=25.0, c=12.5),
    SteinbergScenario("sorting", "(d) Complete sorting", su=25.0, sv=7.5, c=0.0),
)


@dataclass(frozen=True)
class ArmstrongFig121DParams:
    L: float = 10.0
    dx: float = 0.1
    dt: float = 1.0e-4
    t_end: float = 125.0
    m: float = 1.0
    sensing_radius: float = 1.0
    n_inner: int = 1000
    seed: int = 7
    snapshot_times: tuple[float, ...] = (0.0, 50.0, 125.0)


def _build_kernel(sr: int) -> tuple[np.ndarray, int]:
    kernel = np.ones(2 * sr + 1, dtype=float)
    kernel[sr] = 0.0
    kernel[sr + 1 :] = -1.0
    return kernel, sr + 1


def simulate_scenario(
    scenario: SteinbergScenario,
    p: ArmstrongFig121DParams,
) -> tuple[np.ndarray, dict[float, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    n = int(round(p.L / p.dx))
    sr = max(1, int(round(p.sensing_radius / p.dx)))
    kernel, center = _build_kernel(sr)
    rng = np.random.default_rng(p.seed)
    u = 0.2 + 0.01 * rng.random(n)
    v = 0.2 + 0.01 * rng.random(n)
    x = (np.arange(n, dtype=float) + 0.5) * p.dx

    def guu(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        s = uu + vv
        return np.where(s < p.m, uu * (1.0 - s / p.m), 0.0)

    def guv(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        s = uu + vv
        return np.where(s < p.m, vv * (1.0 - s / p.m), 0.0)

    def ku(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return scenario.su * list_convolve_mathematica(kernel, guu(uu, vv), center) + scenario.c * list_convolve_mathematica(
            kernel, guv(uu, vv), center
        )

    def kv(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return scenario.sv * list_convolve_mathematica(kernel, guv(uu, vv), center) + scenario.c * list_convolve_mathematica(
            kernel, guu(uu, vv), center
        )

    def step(uu: np.ndarray, vv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kuu, kvv = ku(uu, vv), kv(uu, vv)
        du = (np.roll(uu, -1) + np.roll(uu, 1) - 2.0 * uu) / (p.dx * p.dx) - (
            np.roll(uu * kuu, -1) - np.roll(uu * kuu, 1)
        ) / (2.0 * p.dx)
        dv = (np.roll(vv, -1) + np.roll(vv, 1) - 2.0 * vv) / (p.dx * p.dx) - (
            np.roll(vv * kvv, -1) - np.roll(vv * kvv, 1)
        ) / (2.0 * p.dx)
        return np.maximum(uu + p.dt * du, 0.0), np.maximum(vv + p.dt * dv, 0.0)

    dt_out = p.dt * p.n_inner
    snap_targets = sorted(set(p.snapshot_times) | {p.t_end})
    snaps: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if 0.0 in snap_targets:
        snaps[0.0] = (u.copy(), v.copy())

    n_outer = max(1, int(np.ceil(p.t_end / dt_out)))
    for outer in range(n_outer):
        for _ in range(p.n_inner):
            u, v = step(u, v)
        t_now = (outer + 1) * dt_out
        for t_target in snap_targets:
            if abs(t_now - t_target) < 0.5 * dt_out and t_target not in snaps:
                snaps[t_target] = (u.copy(), v.copy())

    if p.t_end not in snaps:
        snaps[p.t_end] = (u.copy(), v.copy())

    stats = {
        "std_u": float(u.std()),
        "std_v": float(v.std()),
        "mean_u": float(u.mean()),
        "mean_v": float(v.mean()),
        "max_u": float(u.max()),
        "max_v": float(v.max()),
    }
    return x, snaps, stats


def fig12_1d_params(*, quick: bool = False, fig9: bool = False) -> ArmstrongFig121DParams:
    if fig9:
        return ArmstrongFig121DParams(
            L=20.0,
            t_end=500.0 if not quick else 50.0,
            snapshot_times=(0.0, 5.0, 10.0, 500.0) if not quick else (0.0, 5.0, 10.0, 50.0),
        )
    if quick:
        return ArmstrongFig121DParams(t_end=50.0, snapshot_times=(0.0, 25.0, 50.0))
    return ArmstrongFig121DParams(t_end=125.0, snapshot_times=(0.0, 50.0, 125.0))


def main() -> None:
    quick = os.environ.get("ARMSTRONG_FIG12_1D_QUICK", "").strip().lower() in ("1", "true", "yes")
    fig9 = os.environ.get("ARMSTRONG_FIG12_1D_MODE", "fig12").strip().lower() == "fig9"
    scenarios = FIG9_SCENARIOS if fig9 else FIG12_SCENARIOS
    params = fig12_1d_params(quick=quick, fig9=fig9)
    mode_label = "Fig. 9 (1D paper)" if fig9 else "Fig. 12 (2D params on L=10)"

    snap_keys = sorted(params.snapshot_times)
    fig, axes = plt.subplots(len(scenarios), len(snap_keys), figsize=(3.5 * len(snap_keys), 2.8 * len(scenarios)), squeeze=False)

    csv_rows: list[dict[str, str | float]] = []
    for i, scenario in enumerate(scenarios):
        print(f"{scenario.label}  Su={scenario.su}, Sv={scenario.sv}, C={scenario.c}")
        x, snaps, stats = simulate_scenario(scenario, params)
        for j, t_snap in enumerate(snap_keys):
            ax = axes[i, j]
            u_s, v_s = snaps.get(t_snap, snaps[min(snaps.keys(), key=lambda t: abs(t - t_snap))])
            ax.plot(x, u_s, color="C0", lw=1.2, label="u")
            ax.plot(x, v_s, color="C3", lw=1.2, label="v")
            ax.set_ylim(0.0, 1.05)
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(f"$t={t_snap:g}$")
            if j == 0:
                ax.set_ylabel(f"{scenario.label}\ndensity")
            if i == len(scenarios) - 1:
                ax.set_xlabel("x")
        print(
            f"  t={params.t_end:g}: std_u={stats['std_u']:.4f} std_v={stats['std_v']:.4f} "
            f"mean_u={stats['mean_u']:.3f} mean_v={stats['mean_v']:.3f}"
        )
        csv_rows.append(
            {
                "scenario": scenario.name,
                "Su": scenario.su,
                "Sv": scenario.sv,
                "C": scenario.c,
                "t_end": params.t_end,
                **stats,
            }
        )

    axes[0, -1].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        f"Armstrong et al. (2006) 1D Steinberg check — {mode_label}\n"
        rf"$L={params.L}$, $dx={params.dx}$, IC $0.2+0.01\,\mathrm{{noise}}$",
        fontsize=11,
    )
    out_png = (Path(__file__).resolve().parent / "results" / "armstrong_fig12_1d.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    out_csv = Path(__file__).with_name("data/armstrong_fig12_1d.csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
