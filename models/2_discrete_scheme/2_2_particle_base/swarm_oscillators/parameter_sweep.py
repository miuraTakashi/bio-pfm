"""
(c1, c2) パラメータスイープ用。GIF は pattern のみ（タイトル・カラーバー・目盛りなし）。

試行: 既定は少数グリッド・短ステップ。

- ``python parameter_sweep.py grid005`` … c1∈[1,6], c2∈[1,5] 刻み 0.05（101×81=8181 本、重い）→ ``parameter_sweep/`` に GIF
- ``python parameter_sweep.py grid005_npz`` … 同上グリッド → ``parameter_sweep_npz/`` に ``SO_c1_*_c2_*.npz``（座標 ``r``・位相 ``psi``）
- 環境変数 ``SWARM_SWEEP_SKIP_EXISTING=1`` … 已有 GIF / npz をスキップ（再開用）
- 環境変数 ``SWARM_SWEEP_FULL=1`` … 旧 11×9 粗グリッド（互換）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
for _d in _script_dir.parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))

import atlas_plotting as ap

from swarm_oscillators import simulate_swarm


def _fmt_param(x: float) -> str:
    """ファイル名用（例: 1.05, 2.00）。常に小数2桁で機械処理しやすくする。"""
    return f"{x:.2f}"


def snapshots_to_rt_psi(snapshots: list) -> tuple[np.ndarray, np.ndarray]:
    """simulate_swarm のスナップショット列を (T, n, 2), (T, n) float64 に積む。"""
    T = len(snapshots)
    n = int(snapshots[0][0].shape[0])
    r_out = np.empty((T, n, 2), dtype=np.float64)
    psi_out = np.empty((T, n), dtype=np.float64)
    for i, (ri, psii) in enumerate(snapshots):
        r_out[i] = np.asarray(ri, dtype=np.float64)
        psi_out[i] = np.asarray(psii, dtype=np.float64)
    return r_out, psi_out


def save_trajectory_npz(
    snapshots: list,
    out_path: Path,
    *,
    c1: float,
    c2: float,
    n: int,
    L: float,
    c3: float,
    alpha: float,
    dt: float,
    n_steps: int,
    snapshot_interval: int,
    seed: int,
) -> None:
    """軌跡を npz 保存。キー: ``r`` (T,n,2), ``psi`` (T,n), スカラーはメタデータ。"""
    r_arr, psi_arr = snapshots_to_rt_psi(snapshots)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        r=r_arr,
        psi=psi_arr,
        c1=np.float64(c1),
        c2=np.float64(c2),
        n=np.int32(n),
        L=np.float64(L),
        c3=np.float64(c3),
        alpha=np.float64(alpha),
        dt=np.float64(dt),
        n_steps=np.int32(n_steps),
        snapshot_interval=np.int32(snapshot_interval),
        seed=np.int32(seed),
    )
    print(f"Saved {out_path}  T={r_arr.shape[0]} n={n}", flush=True)


def save_pattern_gif(
    snapshots: list,
    *,
    L: float,
    dt: float,
    snapshot_interval: int,
    out_path: Path,
    fps: int = 12,
    max_frames: int = 200,
    scatter_size: float = 60,
    edge_lw: float = 0.5,
) -> None:
    """位相カラーの散布図のみ。軸・目盛り・カラーバー・タイトルなし。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.axis("off")

    r0, psi0 = snapshots[0]
    sc = ap.atlas_scatter(
        ax,
        r0[:, 0],
        r0[:, 1],
        c=psi0 % (2 * np.pi),
        phase=True,
        vmin=0,
        vmax=2 * np.pi,
        s=scatter_size,
        edgecolors="k",
        linewidths=edge_lw,
    )

    total_frames = len(snapshots)
    skip = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, skip))

    def update(_idx: int):
        i = frame_indices[_idx]
        r, psi = snapshots[i]
        sc.set_offsets(np.column_stack([r[:, 0], r[:, 1]]))
        sc.set_array(psi % (2 * np.pi))
        return (sc,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        blit=True,
        interval=max(1, 1000 // fps),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out_path}  ({len(frame_indices)} frames)", flush=True)


def default_trial_pairs() -> list[tuple[float, float]]:
    """少数試行用 (c1, c2)。例のファイル名は (1.05, 1.05)。"""
    return [
        (1.05, 1.05),
        (2.0, 2.0),
        (2.0, 3.0),
        (1.0, 1.0),
    ]


def full_grid_pairs(
    c1_min: float = 1.0,
    c1_max: float = 6.0,
    c2_min: float = 1.0,
    c2_max: float = 5.0,
    n_c1: int = 11,
    n_c2: int = 9,
) -> list[tuple[float, float]]:
    c1s = np.linspace(c1_min, c1_max, n_c1)
    c2s = np.linspace(c2_min, c2_max, n_c2)
    return [(float(a), float(b)) for a in c1s for b in c2s]


def grid_pairs_step(
    c1_min: float = 1.0,
    c1_max: float = 6.0,
    c2_min: float = 1.0,
    c2_max: float = 5.0,
    dc: float = 0.05,
) -> list[tuple[float, float]]:
    """端点を含み、刻み幅 dc（既定 0.05）の直交グリッド。"""
    n1 = int(round((c1_max - c1_min) / dc)) + 1
    n2 = int(round((c2_max - c2_min) / dc)) + 1
    c1s = np.linspace(c1_min, c1_max, n1)
    c2s = np.linspace(c2_min, c2_max, n2)
    return [(float(a), float(b)) for a in c1s for b in c2s]


def run_sweep(
    pairs: list[tuple[float, float]],
    out_dir: Path | None = None,
    *,
    n: int = 50,
    L: float = 10.0,
    c3: float = 1.0,
    alpha: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 6000,
    snapshot_interval: int = 20,
    seed: int = 42,
    fps: int = 12,
    max_frames: int = 300,
    skip_existing: bool = False,
) -> None:
    out_dir = out_dir or (_script_dir / "parameter_sweep")
    out_dir = Path(out_dir)
    for idx, (c1, c2) in enumerate(pairs):
        name = f"SO_c1_{_fmt_param(c1)}_c2_{_fmt_param(c2)}.gif"
        out_path = out_dir / name
        if skip_existing and out_path.is_file():
            if (idx + 1) % 500 == 0:
                print(f"[{idx + 1}/{len(pairs)}] skip existing", flush=True)
            continue
        snapshots = simulate_swarm(
            n=n,
            L=L,
            c1=c1,
            c2=c2,
            c3=c3,
            alpha=alpha,
            dt=dt,
            n_steps=n_steps,
            snapshot_interval=snapshot_interval,
            seed=seed,
        )
        save_pattern_gif(
            snapshots,
            L=L,
            dt=dt,
            snapshot_interval=snapshot_interval,
            out_path=out_path,
            fps=fps,
            max_frames=max_frames,
        )
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(pairs)}] done", flush=True)


def run_sweep_npz(
    pairs: list[tuple[float, float]],
    out_dir: Path | None = None,
    *,
    n: int = 50,
    L: float = 10.0,
    c3: float = 1.0,
    alpha: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 6000,
    snapshot_interval: int = 20,
    seed: int = 42,
    skip_existing: bool = False,
) -> None:
    out_dir = out_dir or (_script_dir / "parameter_sweep_npz")
    out_dir = Path(out_dir)
    for idx, (c1, c2) in enumerate(pairs):
        name = f"SO_c1_{_fmt_param(c1)}_c2_{_fmt_param(c2)}.npz"
        out_path = out_dir / name
        if skip_existing and out_path.is_file():
            if (idx + 1) % 500 == 0:
                print(f"[{idx + 1}/{len(pairs)}] skip existing npz", flush=True)
            continue
        snapshots = simulate_swarm(
            n=n,
            L=L,
            c1=c1,
            c2=c2,
            c3=c3,
            alpha=alpha,
            dt=dt,
            n_steps=n_steps,
            snapshot_interval=snapshot_interval,
            seed=seed,
        )
        save_trajectory_npz(
            snapshots,
            out_path,
            c1=c1,
            c2=c2,
            n=n,
            L=L,
            c3=c3,
            alpha=alpha,
            dt=dt,
            n_steps=n_steps,
            snapshot_interval=snapshot_interval,
            seed=seed,
        )
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(pairs)}] npz done", flush=True)


def main() -> None:
    argv1 = (sys.argv[1] if len(sys.argv) > 1 else "").lower().strip()
    skip_ex = os.environ.get("SWARM_SWEEP_SKIP_EXISTING", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    full = os.environ.get("SWARM_SWEEP_FULL", "").strip() not in ("", "0", "false", "False")

    if argv1 in ("grid005", "grid", "full005"):
        pairs = grid_pairs_step()
        print(
            f"grid005: {len(pairs)} GIFs -> {_script_dir / 'parameter_sweep'}",
            flush=True,
        )
        run_sweep(
            pairs,
            n_steps=6000,
            snapshot_interval=20,
            max_frames=300,
            fps=12,
            skip_existing=skip_ex,
        )
    elif argv1 in ("grid005_npz", "grid_npz"):
        pairs = grid_pairs_step()
        print(
            f"grid005_npz: {len(pairs)} npz -> {_script_dir / 'parameter_sweep_npz'}",
            flush=True,
        )
        run_sweep_npz(
            pairs,
            n_steps=6000,
            snapshot_interval=20,
            skip_existing=skip_ex,
        )
    elif full:
        pairs = full_grid_pairs()
        print(f"coarse grid: {len(pairs)} GIFs", flush=True)
        run_sweep(
            pairs,
            n_steps=6000,
            snapshot_interval=20,
            max_frames=450,
            fps=16,
            skip_existing=skip_ex,
        )
    else:
        run_sweep(
            default_trial_pairs(),
            n_steps=1200,
            snapshot_interval=20,
            max_frames=120,
            fps=12,
        )


if __name__ == "__main__":
    main()
