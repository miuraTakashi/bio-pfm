"""
本多久夫先生らが発展させた上皮頂点動力学（2次元 vertex model）の簡易実装。

各細胞を多角形とみなし、エネルギー

  E = Σ_edges λ ℓ_e
    + Σ_cells [ (K_A/2)(A_α − A₀)² + (K_P/2)(P_α − P₀)² ]
    + (K_vol/2)(Σ_α A_α − L_x L_y)²

の勾配に比例する過阻尼運動で頂点位置を更新する。周期境界は最小像ベクトルを
(-L/2, L/2]（床関数版）で取り、座標は [0,L) に折り返す。

面積項・周長項・大域体積項の勾配を用いる。∂A/∂r は最小像隣接頂点公式、
∂P/∂r は隣接二辺の単位ベクトル差（最小像）で取る。
辺長が閾値未満になったら T1 transition（近傍交換）でトポロジーを更新する。

初期条件: 既定はほぼ均一な格子＋微小ジッター上の種点（全種が有界 Voronoi 領域になりやすい）。
オプションで従来のランダム配置も可（有界領域が得られるまで再試行）。
3×3 タイル複製＋周期同一視で頂点をマージする。

T1 反転後は位相の粗い検査を行い、不正な場合はロールバックする（詳細は同フォルダ honda_vertex_dynamics.md）。

注（他プロジェクトとの関係）:
  上皮頂点モデルの物理像は本田久夫先生らの研究・文献に基づくが、本ファイルは
  当リポジトリのデモ用実装であり、第三者の公開ソースコードの転載・引用ではない
  （Mathematica ノートの 1:1 移植でもない。honda_vertex_dynamics.md 参照）。
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# --- atlas heatmap helpers ---
import sys
from pathlib import Path as _Path
for _d in _Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap

from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, PillowWriter


def wrap_mod(x: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """基本領域 [0, L)² へ（% より float で安定）。"""
    y = x.copy()
    y[:, 0] -= Lx * np.floor(y[:, 0] / Lx)
    y[:, 1] -= Ly * np.floor(y[:, 1] / Ly)
    return y


def min_image(dr: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """周期トーラス上の最短差分（半開区間 (-L/2, L/2] に収める）。"""
    out = dr.copy()
    out[..., 0] -= Lx * np.floor((out[..., 0] + 0.5 * Lx) / Lx)
    out[..., 1] -= Ly * np.floor((out[..., 1] + 0.5 * Ly) / Ly)
    return out


def cell_unwrapped_xy(verts: np.ndarray, cell: list[int], Lx: float, Ly: float) -> np.ndarray:
    """細胞周りを最小像で連続平面に展開した頂点座標（先頭頂点を原点とする相対座標）。"""
    idx = list(cell)
    p0 = verts[idx[0]].copy()
    pts = [np.zeros(2)]
    acc = p0.copy()
    for k in range(1, len(idx)):
        dk = min_image(verts[idx[k]] - acc, Lx, Ly)
        acc = acc + dk
        pts.append(acc - p0)
    return np.asarray(pts, dtype=float)


def cell_absolute_unwrapped(verts: np.ndarray, cell: list[int], Lx: float, Ly: float) -> np.ndarray:
    """細胞頂点の連続平面上の絶対座標（描画用）。"""
    idx = list(cell)
    acc = verts[idx[0]].copy()
    out = [acc.copy()]
    for k in range(1, len(idx)):
        dk = min_image(verts[idx[k]] - acc, Lx, Ly)
        acc = acc + dk
        out.append(acc.copy())
    return np.asarray(out, dtype=float)


def polygon_area_unwrapped(verts: np.ndarray, cell: list[int], Lx: float, Ly: float) -> float:
    """CCW 多角形の符号付き面積（連続座標に展開してから shoelace）。"""
    pts = cell_unwrapped_xy(verts, cell, Lx, Ly)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def cell_perimeter(verts: np.ndarray, cell: list[int], Lx: float, Ly: float) -> float:
    """細胞境界の周長（CCW 頂点順、各辺は最小像長）。"""
    n = len(cell)
    s = 0.0
    for k in range(n):
        a, b = cell[k], cell[(k + 1) % n]
        dr = min_image(verts[b] - verts[a], Lx, Ly)
        s += float(np.linalg.norm(dr))
    return s


def grad_perimeter_vertex(
    verts: np.ndarray, cell: list[int], j: int, Lx: float, Ly: float
) -> np.ndarray:
    """周長 P の ∂P/∂r（cell[j] を動かす）。辺 (j−1→j) と (j→j+1) の単位ベクトル差。"""
    n = len(cell)
    jm, j0, jp = (j - 1) % n, j, (j + 1) % n
    vm, v0, vp = cell[jm], cell[j0], cell[jp]
    dr_in = min_image(verts[v0] - verts[vm], Lx, Ly)
    dr_out = min_image(verts[vp] - verts[v0], Lx, Ly)
    li = float(np.linalg.norm(dr_in)) + 1e-14
    lo = float(np.linalg.norm(dr_out)) + 1e-14
    return dr_in / li - dr_out / lo


def grad_area_vertex_min_image(
    verts: np.ndarray, cell: list[int], vid: int, Lx: float, Ly: float
) -> np.ndarray:
    """符号付き面積の ∇_{r_vid}（隣接頂点を最小像で取る多角形の標準式。周期領域で安定）。"""
    n = len(cell)
    j = cell.index(vid)
    jm, jp = cell[(j - 1) % n], cell[(j + 1) % n]
    r = verts[vid]
    dr_prev = min_image(verts[jm] - r, Lx, Ly)
    dr_next = min_image(verts[jp] - r, Lx, Ly)
    return 0.5 * np.array([dr_next[1] - dr_prev[1], dr_prev[0] - dr_next[0]])


def ensure_cells_ccw(verts: np.ndarray, cells: list[list[int]], Lx: float, Ly: float) -> None:
    """符号付き面積が正になるよう頂点順を反転（力の向きを統一）。"""
    for c in cells:
        if polygon_area_unwrapped(verts, c, Lx, Ly) < 0:
            c.reverse()


def build_edges_from_cells(cells: list[list[int]]) -> dict[frozenset, list[int]]:
    """各無向辺をキーに、共有細胞 index のリスト（長さ 1 または 2）。"""
    edge_cells: dict[frozenset, list[int]] = {}
    for ci, c in enumerate(cells):
        m = len(c)
        for k in range(m):
            a, b = c[k], c[(k + 1) % m]
            key = frozenset((a, b))
            edge_cells.setdefault(key, []).append(ci)
    return edge_cells


def _seeds_rectangular_grid(
    n_target: int, Lx: float, Ly: float, rng: np.random.Generator, jitter_frac: float = 0.12
) -> np.ndarray:
    """nx×ny ≧ n_target の格子種点（セル中心＋微小乱数）。有界 Voronoi が得られやすい。"""
    nx = max(2, int(round(np.sqrt(max(n_target, 1) * Lx / Ly))))
    ny = max(2, int(np.ceil(n_target / nx)))
    while nx * ny < n_target:
        ny += 1
    seeds: list[list[float]] = []
    hx, hy = Lx / nx, Ly / ny
    jit_scale = jitter_frac * min(hx, hy)
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * hx
            y = (j + 0.5) * hy
            jit = (rng.random(2) - 0.5) * 2.0 * jit_scale
            seeds.append([x + jit[0], y + jit[1]])
    return wrap_mod(np.asarray(seeds, dtype=float), Lx, Ly)


def _voronoi_cells_from_seeds(
    seeds: np.ndarray,
    Lx: float,
    Ly: float,
    merge_tol_frac: float,
) -> tuple[np.ndarray, list[list[int]]]:
    """種点配列から周期 Voronoi 細胞リストを構築（同一処理を grid / random で共有）。"""
    n_seeds = len(seeds)
    merge_tol = merge_tol_frac * min(Lx, Ly)
    pts_blocks: list[np.ndarray] = []
    central_start = 0
    for dx in (-Lx, 0.0, Lx):
        for dy in (-Ly, 0.0, Ly):
            if dx == 0.0 and dy == 0.0:
                central_start = len(pts_blocks) * n_seeds
            pts_blocks.append(seeds + np.array([dx, dy], dtype=float))
    pts = np.vstack(pts_blocks)

    vor = Voronoi(pts)

    verts_list: list[np.ndarray] = []

    def get_vertex_id(w: np.ndarray) -> int:
        w = wrap_mod(w.reshape(1, 2), Lx, Ly)[0]
        for vi, v in enumerate(verts_list):
            if np.linalg.norm(min_image(w - v, Lx, Ly)) < merge_tol:
                return vi
        verts_list.append(w.copy())
        return len(verts_list) - 1

    cells: list[list[int]] = []
    for k in range(n_seeds):
        pti = central_start + k
        ridx = vor.point_region[pti]
        reg = vor.regions[ridx]
        if not reg or -1 in reg:
            continue
        seed = seeds[k]
        poly = vor.vertices[np.asarray(reg, dtype=int)]
        rel = np.array([min_image(p - seed, Lx, Ly) for p in poly])
        order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
        poly_o = poly[order]
        cell_ids: list[int] = []
        last = -1
        for p in poly_o:
            vid = get_vertex_id(p)
            if vid != last:
                cell_ids.append(vid)
            last = vid
        if len(cell_ids) > 1 and cell_ids[0] == cell_ids[-1]:
            cell_ids.pop()
        if len(cell_ids) < 3:
            continue
        cells.append(cell_ids)

    if len(cells) < max(3, n_seeds // 2):
        raise RuntimeError(
            "周期 Voronoi の有界領域が不足しました。種点配置を変えるか n_seeds を調整してください。"
        )

    verts = np.stack(verts_list, axis=0)
    verts = wrap_mod(verts, Lx, Ly)
    ensure_cells_ccw(verts, cells, Lx, Ly)
    return verts, cells


def initial_periodic_voronoi(
    n_seeds: int,
    Lx: float,
    Ly: float,
    rng: np.random.Generator,
    min_sep_frac: float = 0.04,
    merge_tol_frac: float = 1e-5,
    layout: str = "grid",
    random_max_tries: int = 80,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    layout=\"grid\"（既定）: ほぼ均一な nx×ny 格子＋ジッター。全種が有界領域になりやすく空白が出にくい。
    layout=\"random\": 最小間隔 min_sep_frac * min(L) のランダム配置を random_max_tries 回試行。
    """
    if layout == "grid":
        seeds = _seeds_rectangular_grid(n_seeds, Lx, Ly, rng)
        return _voronoi_cells_from_seeds(seeds, Lx, Ly, merge_tol_frac)

    min_d = min_sep_frac * min(Lx, Ly)
    last_err: str | None = None
    for attempt in range(random_max_tries):
        s = np.zeros((n_seeds, 2))
        for i in range(n_seeds):
            for _ in range(5000):
                p = rng.random(2) * np.array([Lx, Ly], dtype=float)
                if i == 0:
                    s[i] = p
                    break
                if all(np.linalg.norm(min_image(p - s[j], Lx, Ly)) >= min_d for j in range(i)):
                    s[i] = p
                    break
            else:
                s[i] = rng.random(2) * np.array([Lx, Ly], dtype=float)
        try:
            verts, cells = _voronoi_cells_from_seeds(s, Lx, Ly, merge_tol_frac)
        except RuntimeError as e:
            last_err = str(e)
            continue
        if len(cells) == n_seeds:
            return verts, cells
        last_err = f"有界細胞数 {len(cells)} が種点数 {n_seeds} と一致しません"
    raise RuntimeError(
        last_err
        or "周期 Voronoi（random）で十分な有界領域が得られませんでした。layout=\"grid\" を使うか n_seeds を減らしてください。"
    )


def _ccw_arc_inclusive(cell: list[int], start: int, end: int, forbid: set[int]) -> list[int]:
    """
    cell 周りを CCW に進み、start から end まで（両端含む）の頂点列。
    forbid を内部（両端以外）に含む弧は不採用。2 方向のうち許容する方を選ぶ。
    """
    n = len(cell)
    i0, i1 = cell.index(start), cell.index(end)

    def ccw_chain(i_s: int, i_e: int) -> list[int]:
        out = [cell[i_s]]
        k = (i_s + 1) % n
        while True:
            out.append(cell[k])
            if k == i_e:
                break
            k = (k + 1) % n
            if len(out) > n + 2:
                raise RuntimeError("ccw_chain overflow")
        return out

    def ccw_chain_reverse(i_s: int, i_e: int) -> list[int]:
        out = [cell[i_s]]
        k = (i_s - 1) % n
        while True:
            out.append(cell[k])
            if k == i_e:
                break
            k = (k - 1) % n
            if len(out) > n + 2:
                raise RuntimeError("ccw_chain_reverse overflow")
        return out

    def interior_ok(seq: list[int]) -> bool:
        if len(seq) < 3:
            return True
        return all(v not in forbid for v in seq[1:-1])

    cand_a = ccw_chain(i0, i1)
    cand_b = ccw_chain_reverse(i0, i1)
    for arc in (cand_a, cand_b):
        if arc[0] == start and arc[-1] == end and interior_ok(arc):
            return arc
    return cand_a if len(cand_a) <= len(cand_b) else cand_b


def _mesh_consistent(verts: np.ndarray, cells: list[list[int]], Lx: float, Ly: float) -> bool:
    """面積正・辺次数・多角形の辺数の粗い妥当性チェック。"""
    if not cells:
        return False
    n_cells = len(cells)
    max_side = max(len(c) for c in cells)
    if max_side > max(14, n_cells):
        return False
    for c in cells:
        if len(c) < 3:
            return False
        if polygon_area_unwrapped(verts, c, Lx, Ly) <= 1e-14:
            return False
    ec = build_edges_from_cells(cells)
    for key, ids in ec.items():
        if len(key) != 2:
            continue
        if len(ids) != 2:
            return False
    return True


def t1_flip(
    cells: list[list[int]],
    v0: int,
    v1: int,
    edge_cells: dict[frozenset, list[int]],
    verts: np.ndarray,
    Lx: float,
    Ly: float,
) -> bool:
    """
    辺 (v0,v1) が ちょうど 2 細胞に共有されるとき T1 を実行。
    c1: ... pre1, v0, v1, post1 ... / c2: ... pre2, v1, v0, post2 ... (CCW)
    """
    key = frozenset((v0, v1))
    if key not in edge_cells or len(edge_cells[key]) != 2:
        return False
    c1i, c2i = edge_cells[key]
    c1 = cells[c1i][:]
    c2 = cells[c2i][:]

    def orient(c_low, c_high):
        i0 = c_low.index(v0)
        n = len(c_low)
        if c_low[(i0 + 1) % n] == v1:
            return c_low, c_high, c1i, c2i
        if c_low[(i0 - 1) % n] == v1:
            return c_high, c_low, c2i, c1i
        return None

    o = orient(c1, c2)
    if o is None:
        o = orient(c2, c1)
    if o is None:
        return False
    ca, cb, ia, ib = o

    i0 = ca.index(v0)
    n1 = len(ca)
    if ca[(i0 + 1) % n1] != v1:
        return False
    pre1 = ca[(i0 - 1) % n1]
    post1 = ca[(i0 + 2) % n1]

    i2 = cb.index(v1)
    n2 = len(cb)
    if cb[(i2 + 1) % n2] != v0:
        return False
    pre2 = cb[(i2 - 1) % n2]
    post2 = cb[(i2 + 2) % n2]

    forbid = {v0, v1}
    arc_post2_pre2 = _ccw_arc_inclusive(cb, post2, pre2, forbid)
    arc_post1_pre1 = _ccw_arc_inclusive(ca, post1, pre1, forbid)

    # 新細胞 A: pre1 → v0 → (cb 上 post2…pre2) → (ca 上 post1…pre1 の終端 pre1 は始点と同一なので除く)
    new_a = [pre1, v0] + arc_post2_pre2 + arc_post1_pre1[:-1]
    # 新細胞 B: pre2 → v1 → (ca 上 post1…pre1) → post2
    new_b = [pre2, v1] + arc_post1_pre1 + [post2]

    backup_a, backup_b = cells[ia][:], cells[ib][:]
    cells[ia] = new_a
    cells[ib] = new_b
    if not _mesh_consistent(verts, cells, Lx, Ly):
        cells[ia] = backup_a
        cells[ib] = backup_b
        return False
    return True


def compute_forces(
    verts: np.ndarray,
    cells: list[list[int]],
    Lx: float,
    Ly: float,
    lam: float,
    K_A: float,
    A0: float,
    K_vol: float,
    K_P: float,
    P0: float,
) -> np.ndarray:
    forces = np.zeros_like(verts)

    # 線張力: 各無向辺を一度だけ
    edge_done: set[frozenset] = set()
    for c in cells:
        m = len(c)
        for k in range(m):
            a, b = c[k], c[(k + 1) % m]
            if a == b:
                continue
            key = frozenset((a, b))
            if key in edge_done:
                continue
            edge_done.add(key)
            dr = min_image(verts[b] - verts[a], Lx, Ly)
            dist = np.linalg.norm(dr) + 1e-14
            f = lam * dr / dist
            forces[a] += f
            forces[b] -= f

    sum_A = sum(polygon_area_unwrapped(verts, c, Lx, Ly) for c in cells)
    err_vol = sum_A - Lx * Ly

    # 面積弾性 + 大域体積 + 周長弾性（細胞ごと）
    for c in cells:
        A = polygon_area_unwrapped(verts, c, Lx, Ly)
        dE_dA = K_A * (A - A0) + K_vol * err_vol
        P = cell_perimeter(verts, c, Lx, Ly)
        dE_dP = K_P * (P - P0)
        for j, vid in enumerate(c):
            ga = grad_area_vertex_min_image(verts, c, vid, Lx, Ly)
            gp = grad_perimeter_vertex(verts, c, j, Lx, Ly)
            forces[vid] -= dE_dA * ga + dE_dP * gp

    return forces


def try_all_t1(verts: np.ndarray, cells: list[list[int]], Lx: float, Ly: float, l_min: float) -> int:
    """閾値未満の辺に対して T1 を試み、成功した回数を返す（1 ステップあたり最大 max_flips 回）。"""
    flipped = 0
    max_flips = 40
    failed: set[frozenset] = set()
    for _ in range(max_flips):
        edge_cells = build_edges_from_cells(cells)
        picked = None
        for key, cids in edge_cells.items():
            if len(key) != 2 or len(cids) != 2:
                continue
            if key in failed:
                continue
            a, b = tuple(key)
            if a == b:
                continue
            dr = min_image(verts[b] - verts[a], Lx, Ly)
            if np.linalg.norm(dr) < l_min:
                picked = (a, b)
                break
        if picked is None:
            break
        a, b = picked
        key = frozenset((a, b))
        ec = build_edges_from_cells(cells)
        if t1_flip(cells, a, b, ec, verts, Lx, Ly):
            flipped += 1
        else:
            failed.add(key)
    return flipped


def simulate(
    n_steps: int = 4000,
    dt: float = 0.002,
    snapshot_every: int = 40,
    lam: float = 1.0,
    K_A: float = 6.0,
    K_vol: float = 4.0,
    K_P: float = 0.85,
    A0: float | None = None,
    P0: float | None = None,
    l_min_frac: float = 0.038,
    rng_seed: int = 42,
    n_seeds: int = 20,
    layout: str = "grid",
):
    rng = np.random.default_rng(rng_seed)
    Lx, Ly = 1.0, 1.0
    verts, cells = initial_periodic_voronoi(n_seeds, Lx, Ly, rng, layout=layout)
    n_cells = len(cells)
    # 目標細胞面積 = 基本領域 / 細胞数（平均と一致させ、大域項と整合）
    if A0 is None:
        A0 = float(Lx * Ly / max(n_cells, 1))
    if P0 is None:
        perims = [cell_perimeter(verts, c, Lx, Ly) for c in cells]
        P0 = float(np.mean(perims))

    mean_edge = 0.0
    ec0 = build_edges_from_cells(cells)
    for key in ec0:
        if len(ec0[key]) != 2:
            continue
        a, b = tuple(key)
        mean_edge += np.linalg.norm(min_image(verts[b] - verts[a], Lx, Ly))
    mean_edge /= max(1, len([k for k in ec0 if len(ec0[k]) == 2]))
    # 辺がこれより短いときのみ T1 候補（閾値を小さくし過剰反転を抑える）
    l_min = min(l_min_frac * mean_edge, 0.016 * float(np.sqrt(max(A0, 1e-12))))

    snapshots = [(verts.copy(), [c[:] for c in cells])]

    for step in range(n_steps):
        F = compute_forces(verts, cells, Lx, Ly, lam, K_A, A0, K_vol, K_P, P0)
        verts = wrap_mod(verts + dt * F, Lx, Ly)
        try_all_t1(verts, cells, Lx, Ly, l_min)
        ensure_cells_ccw(verts, cells, Lx, Ly)
        if (step + 1) % snapshot_every == 0:
            snapshots.append((verts.copy(), [c[:] for c in cells]))

    meta = dict(
        Lx=Lx,
        Ly=Ly,
        A0=A0,
        P0=P0,
        K_A=K_A,
        K_vol=K_vol,
        K_P=K_P,
        l_min=l_min,
        n_seeds=n_seeds,
        n_cells=len(cells),
        layout=layout,
    )
    return snapshots, meta


def draw_sheet(ax, verts: np.ndarray, cells: list[list[int]], Lx: float, Ly: float, title: str = ""):
    """基本領域に 3×3 複製を重ね、周期境界で途切れないように描画する。"""
    polys: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    n_c = len(cells)
    tab = ap.atlas_tab_colors(n_c)
    for ci, c in enumerate(cells):
        base = cell_absolute_unwrapped(verts, c, Lx, Ly)
        for sx in (-Lx, 0.0, Lx):
            for sy in (-Ly, 0.0, Ly):
                polys.append(base + np.array([sx, sy], dtype=float))
                colors.append(tab[ci % len(tab)])
    pc = PolyCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.6, alpha=0.85)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title)


def make_gif(
    snapshots,
    meta,
    snapshot_every: int,
    filename: Path | str | None = None,
    fps=12,
    max_frames=200,
):
    if filename is None:
        filename = (Path(__file__).resolve().parent / "results" / "honda_vertex_dynamics.gif")
    Lx, Ly = meta["Lx"], meta["Ly"]
    fig, ax = plt.subplots(figsize=(6, 6))
    total = len(snapshots)
    skip = max(1, total // max_frames)
    indices = list(range(0, total, skip))

    def update(j):
        ax.clear()
        idx = indices[j]
        v, cl = snapshots[idx]
        t = idx * snapshot_every
        draw_sheet(ax, v, cl, Lx, Ly, title=f"Honda-type vertex dynamics  step={t}")
        return []

    anim = FuncAnimation(fig, update, frames=len(indices), blit=False, interval=1000 // fps)
    anim.save(str(filename), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {filename} ({len(indices)} frames)")


def main():
    n_steps = int(os.environ.get("HONDA_VERTEX_N_STEPS", "3600"))
    snapshot_every = int(os.environ.get("HONDA_VERTEX_SNAPSHOT_EVERY", "60"))
    snapshots, meta = simulate(
        n_steps=n_steps, snapshot_every=snapshot_every, dt=0.002
    )
    v0, cells0 = snapshots[0]
    v1, cells1 = snapshots[-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    n0 = meta.get("n_cells", len(cells0))
    n1 = len(cells1)
    draw_sheet(
        axes[0],
        v0,
        cells0,
        meta["Lx"],
        meta["Ly"],
        f"Initial: periodic Voronoi ({n0} cells)",
    )
    draw_sheet(
        axes[1],
        v1,
        cells1,
        meta["Lx"],
        meta["Ly"],
        f"Final: {n1} cells (T1 transitions)",
    )
    plt.tight_layout()
    out_png = (Path(__file__).resolve().parent / "results" / "honda_vertex_dynamics.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

    make_gif(
        snapshots,
        meta,
        snapshot_every=snapshot_every,
        fps=14,
        max_frames=180,
    )


if __name__ == "__main__":
    main()
