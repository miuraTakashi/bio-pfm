#!/usr/bin/env python3
"""
nematic_tensor — 2D nematic tensor デモ（フォルダ名 `nematic_tensor` に合わせたエントリポイント）。
元ノート: nematicTensor2D.ipynb

出力は次の 2 つのみ（`results/` 以下）:
  - streamlines_linewidth_timeseries.png
  - streamlines_linewidth.gif

実時間の終端は T。各保存フレーム k の時刻は t_out[k]。
(q11,q12) を streamplot に渡しているのはテンソル成分の可視化用（速度場ではない）。

注（他プロジェクトとの関係）:
  元ノート名 nematicTensor2D.ipynb は参考表記。本リポジトリ Mathematica/ に
  同名 .nb が無い場合は外部ノート由来の名称の可能性があるが、当 .py のコードは
  他プロジェクトのソースの転載・引用ではなく、このアトラス用の実装である。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from pathlib import Path

# Raster figure output (PDF より高速)
FIG_DPI = 150
_SAVE_PNG_KW = {"dpi": FIG_DPI, "bbox_inches": "tight"}

# streamplot: 矢頭なし。線の接線方向で向きのみ示す（arrowstyle='-'）
_STREAMPLOT_KW = {"arrowstyle": "-"}


# ============================================================================
# Model parameters
# ============================================================================
L = 12.8
r = -0.1
u = 1
phi = 0.1
# 実時間の終端 [0, T]（他モデルと同様に記号 T を終端時刻に用いる）
T = 5.0
K = 0.1

dx = 0.1
dt = 0.01

n = int(L/dx)

# 保存フレーム番号 k = 0 .. nOutput-1（実時間 t_out[k]；T は終端時刻）
nOutput = 6

# 実時間 t ∈ [0, T] をオイラー法（刻み dt）で N_STEPS 歩分積分し、
# その歩数を (nOutput-1) 個の区間に均等配分して各保存フレームの時刻 t_out[k] を定める。
N_STEPS = int(round(T / dt))
_n_gap = nOutput - 1
_base = N_STEPS // _n_gap
_rem = N_STEPS % _n_gap
steps_per_gap = np.array(
    [_base + (1 if i < _rem else 0) for i in range(_n_gap)],
    dtype=int,
)
t_out = np.zeros(nOutput)
for _k in range(1, nOutput):
    t_out[_k] = t_out[_k - 1] + float(steps_per_gap[_k - 1]) * dt


def _time_caption(k: int) -> str:
    """実時間 t（t_out[k]）と保存フレーム番号 k。終端時刻は T。"""
    return f"t = {t_out[k]:.5g} (frame k = {k})"


# ============================================================================
# Initial condition
# ============================================================================
q11 = np.zeros((nOutput, n, n))
q12 = np.zeros((nOutput, n, n))

_rng = np.random.default_rng()
q11[0] = _rng.standard_normal((n, n))
q12[0] = _rng.standard_normal((n, n))

# ============================================================================
# Create results directory (always next to this script, not cwd)
# ============================================================================
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {results_dir}/")

# ============================================================================
# Time integration bookkeeping (before running)
# ============================================================================
print("=== 実時間 T（終端）と保存フレーム k、各フレームの時刻 t_out[k] ===")
print(f"  実時間の終端 T = {T}, Euler 法の時間刻み dt = {dt}  →  N_STEPS = {N_STEPS} (= T/dt).")
print(f"  保存フレーム間の区間数: {nOutput - 1}; 各区間の歩数 = {steps_per_gap.tolist()}")
print(
    f"  検算: sum(steps_per_gap) * dt = {float(steps_per_gap.sum()) * dt:.12g}  (目標 T = {T})"
)
print(f"  各保存フレーム k における実時間 t_out[k]: {np.array2string(t_out, precision=6)}")
_dfcrit = dx * dx / (4.0 * K)
print(
    f"  拡散（陽解法）目安: dt ≲ dx²/(4K) = {_dfcrit:.6g}  (現在 dt = {dt})"
)

# ============================================================================
# Numerical simulation
# ============================================================================
print("Running numerical simulation...")
# Advance in scratch (a11, a12); do not overwrite stored snapshots q11[t], q12[t].
for t in range(nOutput - 1):
    a11 = np.array(q11[t], copy=True)
    a12 = np.array(q12[t], copy=True)
    for _ in range(int(steps_per_gap[t])):
        a11, a12 = (
            a11
            + dt
            * (
                r * phi * a11
                - 2 * u * (a11**2 + a12**2) * a11
                + K
                / dx
                / dx
                * (
                    np.roll(a11, 1, axis=0)
                    + np.roll(a11, -1, axis=0)
                    + np.roll(a11, 1, axis=1)
                    + np.roll(a11, -1, axis=1)
                    - 4 * a11
                )
            ),
            a12
            + dt
            * (
                r * phi * a12
                - 2 * u * (a11**2 + a12**2) * a12
                + K
                / dx
                / dx
                * (
                    np.roll(a12, 1, axis=0)
                    + np.roll(a12, -1, axis=0)
                    + np.roll(a12, 1, axis=1)
                    + np.roll(a12, -1, axis=1)
                    - 4 * a12
                )
            ),
        )
    q11[t + 1] = a11
    q12[t + 1] = a12
print("Simulation completed.")
print(
    f"  Integrated Euler substeps: {int(steps_per_gap.sum())} (expect {N_STEPS})."
)

# ============================================================================
# Visualization（streamlines_linewidth_timeseries.png と .gif のみ保存）
# ============================================================================
def _streamlines_linewidth_on_ax(ax, t_idx):
    """Draw streamlines (linewidth ∝ |Q|) on *ax* for time index *t_idx*."""
    U = q11[t_idx]
    V = q12[t_idx]
    magnitude = np.sqrt(U**2 + V**2)
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    if magnitude.max() > 0:
        mag_normalized = magnitude / magnitude.max()
    else:
        mag_normalized = magnitude
    linewidths = 0.5 + 3.0 * mag_normalized

    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.streamplot(
        X, Y, U, V, density=3.0, color="red", linewidth=linewidths, **_STREAMPLOT_KW
    )
    ax.set_title(_time_caption(t_idx), color="white", fontsize=11)
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('y', color='white')
    ax.tick_params(colors='white')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.grid(True, color='gray', alpha=0.3)
    return magnitude, linewidths


def save_streamlines_linewidth_timeseries_png(
    time_indices=None,
    filename: str = "streamlines_linewidth_timeseries.png",
):
    """One figure: selected time steps in a single horizontal row."""
    if time_indices is None:
        tl = nOutput - 1
        time_indices = (0, max(1, tl // 2), tl)
    ncols = len(time_indices)
    fig_w = 5.2 * ncols
    fig_h = 5.4
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor('black')
    axes = axes.ravel()

    print("=== Streamlines with magnitude as linewidth (timeseries row) ===")
    for ax, t_idx in zip(axes, time_indices):
        magnitude, linewidths = _streamlines_linewidth_on_ax(ax, t_idx)
        print(
            f"  k={t_idx}, t={t_out[t_idx]:.5g}: |Q| in [{magnitude.min():.3f}, {magnitude.max():.3f}], "
            f"lw in [{linewidths.min():.2f}, {linewidths.max():.2f}]"
        )

    fig.suptitle(
        f"Nematic tensor: streamlines (linewidth ∝ |Q|); t = real time, k = frame index, T = {T}",
        color="white",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    output_path = results_dir / filename
    plt.savefig(output_path, facecolor="black", **_SAVE_PNG_KW)
    plt.close()
    print(f"Saved {output_path}")


save_streamlines_linewidth_timeseries_png()

# ============================================================================
# streamlines_linewidth.gif
# ============================================================================
def save_streamlines_linewidth_gif(filename='streamlines_linewidth.gif', frames=None, fps=10):
    """
    Save streamline plots with linewidth as a GIF animation
    Parameters:
        filename: str, output GIF filename
        frames: list of frame indices to save (if None, saves all frames)
        fps: int, frames per second for the GIF
    """
    if frames is None:
        frames = range(nOutput)
    
    print(f"Creating GIF animation with {len(frames)} frames...")
    
    # Create coordinate grids
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    
    def update(frame_idx):
        """Update function for animation"""
        ax.clear()
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        
        # Get frame index
        t = frames[frame_idx]
        
        U = q11[t]
        V = q12[t]
        magnitude = np.sqrt(U**2 + V**2)
        
        # Normalize magnitude for linewidth
        if magnitude.max() > 0:
            mag_normalized = magnitude / magnitude.max()
        else:
            mag_normalized = magnitude
        linewidths = 0.5 + 3.0 * mag_normalized
        
        stream = ax.streamplot(
            X,
            Y,
            U,
            V,
            density=3.0,
            color="red",
            linewidth=linewidths,
            **_STREAMPLOT_KW,
        )
        
        ax.set_title(
            f"Streamlines (linewidth ∝ |Q|)\n{_time_caption(t)}",
            color="white",
            fontsize=12,
        )
        ax.set_xlabel('X-axis', color='white')
        ax.set_ylabel('Y-axis', color='white')
        ax.tick_params(colors='white')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.grid(True, color='gray', alpha=0.3)
        
        return stream.lines,
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames), 
                       interval=1000/fps, blit=False, repeat=True)
    
    # Save as GIF
    try:
        # Try using Pillow writer (requires pillow package)
        writer = animation.PillowWriter(fps=fps)
        print(f"Saving GIF to '{filename}'...")
        output_path = results_dir / filename
        ani.save(output_path, writer=writer)
        print(f"GIF saved successfully as '{output_path}'")
    except ImportError as e:
        print(f"Error: Pillow package is required for GIF export.")
        print(f"Install it with: pip install pillow")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)

save_streamlines_linewidth_gif(filename="streamlines_linewidth.gif", frames=range(nOutput), fps=10)

print("\n=== Output (2 files only) ===")
_k_last = nOutput - 1
_mid = max(1, _k_last // 2)
print(f"  - {results_dir / 'streamlines_linewidth_timeseries.png'}")
print(
    f"      panels: k = 0, {_mid}, {_k_last}; "
    f"t = {t_out[0]:.5g}, {t_out[_mid]:.5g}, {t_out[_k_last]:.5g}; T = {T}"
)
print(f"  - {results_dir / 'streamlines_linewidth.gif'}")
