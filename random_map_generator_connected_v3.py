# -*- coding: utf-8 -*-
"""
高连接性障碍 + 连通自由空间（显式余裕圆保障）— v3
------------------------------------------------
- 先保证 raw 自由区连通；
- 再用 "余裕圆"/clearance（把障碍膨胀）得到配置空间，在配置空间上**再次**保证连通，
  开挖通道半径 >= clearance_px + 1，确保带余裕也跑得通；
- 仍只依赖 numpy。
"""

import os, math, zipfile, argparse, textwrap
from typing import Tuple, List, Optional
import numpy as np

# ---------------- 基础 I/O ----------------

def write_pgm(path: str, img: np.ndarray) -> None:
    assert img.ndim == 2 and img.dtype == np.uint8
    H, W = img.shape
    header = f"P5\n{W} {H}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header); f.write(img.tobytes())

def write_yaml(path: str, image_filename: str, resolution: float, origin: Tuple[float,float,float],
               negate: int = 0, occupied_thresh: float = 0.65, free_thresh: float = 0.196) -> None:
    yaml_text = textwrap.dedent(f"""
    image: {image_filename}
    resolution: {resolution}
    origin: [{origin[0]}, {origin[1]}, {origin[2]}]
    negate: {negate}
    occupied_thresh: {occupied_thresh}
    free_thresh: {free_thresh}
    """)
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

# -------------- 形态学 --------------

def disk_brush(radius_px: int) -> np.ndarray:
    r = radius_px
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x*x + y*y) <= r*r
    return mask.astype(np.uint8)

def dilate(binary: np.ndarray, brush: np.ndarray) -> np.ndarray:
    H, W = binary.shape
    bh, bw = brush.shape
    ry, rx = bh//2, bw//2
    pad = np.pad(binary, ((ry,ry),(rx,rx)), mode="constant", constant_values=0)
    out = np.zeros_like(binary, dtype=np.uint8)
    ys, xs = np.where(brush==1)
    for ky, kx in zip(ys, xs):
        out |= pad[ky:ky+H, kx:kx+W]
    return out

def erode(binary: np.ndarray, brush: np.ndarray) -> np.ndarray:
    return 1 - dilate(1 - binary, brush)

def morph_close(binary: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return binary
    br = disk_brush(radius_px)
    return erode(dilate(binary, br), br)

# -------------- 多边形/栅格 --------------

def polygon_area_ccw(pts: np.ndarray) -> float:
    x = pts[:,0]; y = pts[:,1]
    return 0.5 * np.sum(x*np.roll(y,-1) - y*np.roll(x,-1))

def ensure_ccw(pts: np.ndarray) -> np.ndarray:
    return pts if polygon_area_ccw(pts) > 0 else pts[::-1].copy()

def fill_polygon_mask(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = poly.copy()
    pts[:,0] = np.clip(pts[:,0], 0, W-1); pts[:,1] = np.clip(pts[:,1], 0, H-1)
    N = len(pts)
    if N < 3: return mask
    x0 = pts[:,0]; y0 = pts[:,1]
    x1 = np.roll(pts[:,0], -1); y1 = np.roll(pts[:,1], -1)
    ymin = max(0, int(np.floor(np.min(pts[:,1])))); ymax = min(H-1, int(np.ceil(np.max(pts[:,1]))))
    for yi in range(ymin, ymax+1):
        cond = ((y0 <= yi) & (y1 > yi)) | ((y1 <= yi) & (y0 > yi))
        idx = np.where(cond)[0]
        if len(idx) == 0: continue
        xints = x0[idx] + (yi - y0[idx]) * (x1[idx] - x0[idx]) / (y1[idx] - y0[idx] + 1e-12)
        xints = np.sort(xints)
        for k in range(0, len(xints), 2):
            if k+1 >= len(xints): break
            xa = int(max(0, np.floor(xints[k]))); xb = int(min(W-1, np.ceil(xints[k+1])))
            if xb >= xa: mask[yi, xa:xb+1] = 1
    return mask

# -------------- 外形（arena） --------------

def sample_arena_polygon(H: int, W: int, rng: np.random.Generator,
                         scale_min: float = 0.45, scale_max: float = 0.95,
                         n_range: Tuple[int,int] = (8, 16),
                         jitter_range: Tuple[float,float] = (0.15, 0.45)) -> np.ndarray:
    n = int(rng.integers(n_range[0], n_range[1]+1))
    scale = float(rng.uniform(scale_min, scale_max))
    jitter = float(rng.uniform(jitter_range[0], jitter_range[1]))
    base_r = 0.5 * scale * min(H, W)
    r_max = base_r * (1.0 + jitter)
    margin = 4
    cx_min, cx_max = r_max + margin, W - r_max - margin
    cy_min, cy_max = r_max + margin, H - r_max - margin
    if cx_min >= cx_max or cy_min >= cy_max:
        r_max = 0.45 * min(H, W)
        cx_min, cx_max = r_max + margin, W - r_max - margin
        cy_min, cy_max = r_max + margin, H - r_max - margin
    cx = float(rng.uniform(max(cx_min, r_max+margin), max(cx_min+1, cx_max)))
    cy = float(rng.uniform(max(cy_min, r_max+margin), max(cy_min+1, cy_max)))
    angles = np.sort(rng.uniform(0, 2*np.pi, size=n))
    radii = base_r * (1.0 + rng.uniform(-jitter, jitter, size=n))
    xs = cx + radii * np.cos(angles); ys = cy + radii * np.sin(angles)
    poly = np.stack([xs, ys], axis=1)
    return ensure_ccw(poly)

# -------------- 网络/胶囊绘制 --------------

def sample_points_in_mask(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.argwhere(mask == 1)
    if len(idx) == 0:
        raise ValueError("mask 内无可采样点")
    choose = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
    pts = idx[choose][:, [1,0]].astype(np.float32)  # (x,y)
    return pts

def mst_edges(points: np.ndarray) -> list:
    N = len(points)
    if N <= 1: return []
    in_set = np.zeros(N, dtype=bool); in_set[0] = True
    edges = []; d = np.sum((points - points[0])**2, axis=1); parent = np.zeros(N, dtype=int)
    d[0] = np.inf; parent[:] = 0
    for _ in range(N-1):
        i = int(np.argmin(d))
        if not np.isfinite(d[i]): break
        in_set[i] = True; edges.append((parent[i], i)); d[i] = np.inf
        di = np.sum((points - points[i])**2, axis=1)
        better = di < d
        parent[better] = i; d[better] = di[better]
    return edges

def k_nn_edges(points: np.ndarray, k: int = 2, prob: float = 0.30, rng: Optional[np.random.Generator] = None) -> list:
    if rng is None: rng = np.random.default_rng()
    N = len(points); extra = []
    if N <= 2: return extra
    d2 = np.sum((points[None,:,:] - points[:,None,:])**2, axis=2)
    for i in range(N):
        order = np.argsort(d2[i])
        cnt = 0
        for j in order[1:]:
            if cnt >= k: break
            if rng.random() < prob:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a != b and (a,b) not in extra:
                    extra.append((a,b)); cnt += 1
    return extra

def draw_disk(mask: np.ndarray, cx: float, cy: float, r: int) -> None:
    H, W = mask.shape
    x0 = max(0, int(np.floor(cx - r))); x1 = min(W-1, int(np.ceil(cx + r)))
    y0 = max(0, int(np.floor(cy - r))); y1 = min(H-1, int(np.ceil(cy + r)))
    yy, xx = np.ogrid[y0:y1+1, x0:x1+1]
    sub = ((xx - cx)**2 + (yy - cy)**2) <= (r*r + 1e-6)
    mask[y0:y1+1, x0:x1+1] |= sub.astype(np.uint8)

def draw_capsule(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float, r: int) -> None:
    H, W = mask.shape
    minx = max(0, int(np.floor(min(x0, x1) - r))); maxx = min(W-1, int(np.ceil(max(x0, x1) + r)))
    miny = max(0, int(np.floor(min(y0, y1) - r))); maxy = min(H-1, int(np.ceil(max(y0, y1) + r)))
    if maxx < minx or maxy < miny: return
    xx, yy = np.meshgrid(np.arange(minx, maxx+1), np.arange(miny, maxy+1))
    vx, vy = (x1 - x0), (y1 - y0); L2 = vx*vx + vy*vy + 1e-12
    t = ((xx - x0)*vx + (yy - y0)*vy) / L2; t = np.clip(t, 0.0, 1.0)
    projx = x0 + t * vx; projy = y0 + t * vy
    dist2 = (xx - projx)**2 + (yy - projy)**2
    sub = (dist2 <= (r*r + 1e-6)).astype(np.uint8)
    mask[miny:maxy+1, minx:maxx+1] |= sub

# -------------- 连通性（自由区） --------------

from collections import deque

def label_components(binary: np.ndarray):
    H, W = binary.shape
    labels = -np.ones((H, W), dtype=np.int32)
    nid = 0; sizes = []
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
    for y in range(H):
        for x in range(W):
            if binary[y, x] == 1 and labels[y, x] < 0:
                q = deque([(y, x)]); labels[y, x] = nid; sz = 0
                while q:
                    cy, cx = q.popleft(); sz += 1
                    for dy, dx in nbrs:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] == 1 and labels[ny, nx] < 0:
                            labels[ny, nx] = nid; q.append((ny, nx))
                sizes.append(sz); nid += 1
    return labels, sizes

def boundary_points(labels: np.ndarray, label_id: int, limit: int = 400, arena_mask: Optional[np.ndarray] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng()
    H, W = labels.shape
    ys, xs = np.where(labels == label_id)
    if len(ys) == 0: return np.zeros((0,2), dtype=np.int32)
    pts = []
    for y, x in zip(ys, xs):
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                if labels[ny, nx] != label_id and (arena_mask is None or arena_mask[ny, nx] == 1):
                    pts.append((x, y)); break
    if len(pts) == 0:
        pts = list(zip(xs, ys))
    pts = np.array(pts, dtype=np.int32)
    if len(pts) > limit:
        sel = rng.choice(len(pts), size=limit, replace=False); pts = pts[sel]
    return pts

def carve_capsule_zero(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float, r: int) -> None:
    H, W = mask.shape
    minx = max(0, int(np.floor(min(x0, x1) - r))); maxx = min(W-1, int(np.ceil(max(x0, x1) + r)))
    miny = max(0, int(np.floor(min(y0, y1) - r))); maxy = min(H-1, int(np.ceil(max(y0, y1) + r)))
    if maxx < minx or maxy < miny: return
    xx, yy = np.meshgrid(np.arange(minx, maxx+1), np.arange(miny, maxy+1))
    vx, vy = (x1 - x0), (y1 - y0); L2 = vx*vx + vy*vy + 1e-12
    t = ((xx - x0)*vx + (yy - y0)*vy) / L2; t = np.clip(t, 0.0, 1.0)
    projx = x0 + t * vx; projy = y0 + t * vy
    dist2 = (xx - projx)**2 + (yy - projy)**2
    sub = (dist2 <= (r*r + 1e-6)).astype(np.uint8)
    submask = mask[miny:maxy+1, minx:maxx+1]
    submask[sub == 1] = 0
    mask[miny:maxy+1, minx:maxx+1] = submask

def ensure_free_connected_raw(occ_raw: np.ndarray, arena_mask: np.ndarray, carve_radius: int, rng: Optional[np.random.Generator] = None) -> None:
    """在 raw 自由区上打通：保证 raw ==0 的自由区在 arena 内只剩一个连通块。"""
    if rng is None: rng = np.random.default_rng()
    H, W = occ_raw.shape
    max_iters = 64
    for _ in range(max_iters):
        free = ((occ_raw == 0) & (arena_mask == 1)).astype(np.uint8)
        labels, sizes = label_components(free)
        if len(sizes) <= 1:
            return
        main_id = int(np.argmax(sizes))
        main_pts = boundary_points(labels, main_id, limit=800, arena_mask=arena_mask, rng=rng)
        if len(main_pts) == 0: return
        merged = False
        for lid in range(len(sizes)):
            if lid == main_id: continue
            sub_pts = boundary_points(labels, lid, limit=400, arena_mask=arena_mask, rng=rng)
            if len(sub_pts) == 0: continue
            A = main_pts; B = sub_pts
            if len(A) > 400: A = A[rng.choice(len(A), size=400, replace=False)]
            if len(B) > 400: B = B[rng.choice(len(B), size=400, replace=False)]
            d = A[:,None,:] - B[None,:,:]; d2 = (d[:,:,0]**2 + d[:,:,1]**2)
            i, j = np.unravel_index(np.argmin(d2), d2.shape)
            x0, y0 = int(A[i,0]), int(A[i,1]); x1, y1 = int(B[j,0]), int(B[j,1])
            carve_capsule_zero(occ_raw, x0, y0, x1, y1, r=carve_radius)
            merged = True
        if not merged: return

def ensure_free_connected_with_clearance(occ_raw: np.ndarray, arena_mask: np.ndarray, clearance_px: int, rng: Optional[np.random.Generator] = None) -> None:
    """
    基于“膨胀后的占据图”保证连通：
    - 先把 occ_raw 用半径 clearance_px 的圆盘做膨胀，得到配置空间占据图 occ_infl；
    - 在 free_infl = (occ_infl==0)&arena 上做组件标记；
    - 若有多个自由连通块，则按最近边界对，在 occ_raw 上以 **r = clearance_px + 1** 的胶囊开通道；
    - 循环直到配置空间自由区只剩 1 块或迭代上限。
    """
    if rng is None: rng = np.random.default_rng()
    H, W = occ_raw.shape
    R = int(max(1, clearance_px + 1))
    max_iters = 64
    brush = disk_brush(clearance_px)
    for _ in range(max_iters):
        occ_infl = dilate(occ_raw, brush)
        free_infl = ((occ_infl == 0) & (arena_mask == 1)).astype(np.uint8)
        labels, sizes = label_components(free_infl)
        if len(sizes) <= 1:
            return
        main_id = int(np.argmax(sizes))
        main_pts = boundary_points(labels, main_id, limit=800, arena_mask=arena_mask, rng=rng)
        if len(main_pts) == 0: return
        merged = False
        for lid in range(len(sizes)):
            if lid == main_id: continue
            sub_pts = boundary_points(labels, lid, limit=400, arena_mask=arena_mask, rng=rng)
            if len(sub_pts) == 0: continue
            A = main_pts; B = sub_pts
            if len(A) > 400: A = A[rng.choice(len(A), size=400, replace=False)]
            if len(B) > 400: B = B[rng.choice(len(B), size=400, replace=False)]
            d = A[:,None,:] - B[None,:,:]; d2 = (d[:,:,0]**2 + d[:,:,1]**2)
            i, j = np.unravel_index(np.argmin(d2), d2.shape)
            x0, y0 = int(A[i,0]), int(A[i,1]); x1, y1 = int(B[j,0]), int(B[j,1])
            carve_capsule_zero(occ_raw, x0, y0, x1, y1, r=R)
            merged = True
        if not merged: return

# -------------- 起终点 & BFS --------------

def exists_path(free: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> bool:
    H, W = free.shape
    sy, sx = start; gy, gx = goal
    if not free[sy, sx] or not free[gy, gx]: return False
    q = [(sy, sx)]; head = 0
    seen = np.zeros_like(free, dtype=np.uint8); seen[sy, sx] = 1
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
    while head < len(q):
        y, x = q[head]; head += 1
        if (y, x) == (gy, gx): return True
        for dy, dx in nbrs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and free[ny, nx]:
                seen[ny, nx] = 1; q.append((ny, nx))
    return False

# -------------- 主流程 --------------

def generate_random_map(
    width: int = 782, height: int = 906,
    resolution_m_per_px: float = 0.05,
    origin_xyz: Tuple[float,float,float] = (-7.8, -6.58982, 0.0),
    arena_scale_min: float = 0.45, arena_scale_max: float = 0.95,
    node_count_range: Tuple[int,int] = (14, 28),
    edge_thickness_px: Tuple[int,int] = (6, 12),
    node_blob_px: Tuple[int,int] = (4, 10),
    extra_knn: int = 2, extra_prob: float = 0.30,
    closing_radius_px: int = 2,
    carve_radius_px: int = 3,            # raw 连通的最小通道半径（最终会 max(clearance+1, 这个值) 用于配置空间连通）
    robot_clearance_m: float = 0.35,     # 余裕圆半径（米）
    seed: Optional[int] = None, max_retries: int = 40,
):
    rng = np.random.default_rng(seed); H, W = height, width

    # A) 外形
    arena_poly = sample_arena_polygon(H, W, rng, scale_min=arena_scale_min, scale_max=arena_scale_max)
    arena_mask = fill_polygon_mask(arena_poly, H, W)  # 1=内部, 0=外部
    occ_raw = np.ones((H, W), dtype=np.uint8)
    occ_raw[arena_mask == 1] = 0

    # B) 连通障碍网络
    n_nodes = int(rng.integers(node_count_range[0], node_count_range[1]+1))
    idx = np.argwhere(arena_mask == 1)
    choose = rng.choice(len(idx), size=min(n_nodes, len(idx)), replace=False)
    pts = idx[choose][:, [1,0]].astype(np.float32)
    for (x, y) in pts:
        r = int(rng.integers(node_blob_px[0], node_blob_px[1]+1))
        draw_disk(occ_raw, x, y, r)
    edges = mst_edges(pts); edges += k_nn_edges(pts, k=extra_knn, prob=extra_prob, rng=rng)
    for (i, j) in edges:
        (x0, y0) = pts[i]; (x1, y1) = pts[j]
        r = int(rng.integers(edge_thickness_px[0], edge_thickness_px[1]+1))
        draw_capsule(occ_raw, x0, y0, x1, y1, r=r)
    occ_raw = (occ_raw & arena_mask) | (1 - arena_mask)

    # C) 结构化闭运算（不会引入噪点）
    if closing_radius_px > 0:
        occ_raw = morph_close(occ_raw, closing_radius_px)
        occ_raw = (occ_raw & arena_mask) | (1 - arena_mask)

    # D) raw 连通（基本保障通行骨架）
    clearance_px = max(1, int(math.ceil(robot_clearance_m / resolution_m_per_px)))
    ensure_free_connected_raw(occ_raw, arena_mask, carve_radius=max(carve_radius_px, 2), rng=rng)

    # E) 配置空间（带余裕）的连通保障
    ensure_free_connected_with_clearance(occ_raw, arena_mask, clearance_px=clearance_px, rng=rng)

    # F) 导出 / 采样
    brush = disk_brush(clearance_px)
    occ_infl = dilate(occ_raw, brush)
    free_infl = (occ_infl == 0).astype(np.uint8)

    # 尝试采样相距 >= 15m 的起终点（在 inflated 自由区上）
    min_task_distance_m = 15.0; min_task_distance_px = int(min_task_distance_m / resolution_m_per_px)
    free_idx = np.argwhere(free_infl == 1)
    start = (0, 0); goal = (H-1, W-1)
    if len(free_idx) >= 2:
        for _ in range(2000):
            a = free_idx[rng.integers(0, len(free_idx))]; b = free_idx[rng.integers(0, len(free_idx))]
            if (a[0]-b[0])**2 + (a[1]-b[1])**2 >= min_task_distance_px**2:
                start, goal = (int(a[0]), int(a[1])), (int(b[0]), int(b[1])); break

    # 断言：在带余裕的 free_infl 上应该可达
    _ok = True
    try:
        if not exists_path(free_infl, start, goal):
            _ok = False
    except Exception:
        _ok = False

    pgm_raw = np.where(occ_raw == 1, 0, 254).astype(np.uint8)
    pgm_infl = np.where(occ_infl == 1, 0, 254).astype(np.uint8)
    meta = {"resolution": resolution_m_per_px, "origin": origin_xyz, "width": width, "height": height,
            "clearance_px": int(clearance_px), "ok_infl": bool(_ok),
            "start_px": (int(start[1]), int(start[0])), "goal_px": (int(goal[1]), int(goal[0]))}
    return pgm_raw, pgm_infl, meta

# ---------------- 导出/打包 ----------------

def _info_text(meta: dict) -> str:
    sx, sy = meta["start_px"]; gx, gy = meta["goal_px"]
    sx_m = (sx * meta["resolution"]) + meta["origin"][0]
    sy_m = (sy * meta["resolution"]) + meta["origin"][1]
    gx_m = (gx * meta["resolution"]) + meta["origin"][0]
    gy_m = (gy * meta["resolution"]) + meta["origin"][1]
    return textwrap.dedent(f"""
    width_px: {meta['width']}, height_px: {meta['height']}
    resolution_m_per_px: {meta['resolution']}
    origin_xyz: {meta['origin']}
    robot_clearance_px: {meta['clearance_px']} (≈ {meta['clearance_px']*meta['resolution']:.3f} m)
    ok_infl_path: {meta['ok_infl']}
    start_px: {meta['start_px']}  goal_px: {meta['goal_px']}
    start_m: ({sx_m:.3f}, {sy_m:.3f})  goal_m: ({gx_m:.3f}, {gy_m:.3f})
    """)

def save_map_bundle(name: str, pgm_raw: np.ndarray, pgm_infl: np.ndarray, meta: dict, out_dir: str, export: str = "raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    def _write_pair(stem: str, pgm_img: np.ndarray):
        pgm_name = f"{stem}.pgm"; yaml_name = f"{stem}.yaml"
        write_pgm(os.path.join(out_dir, pgm_name), pgm_img)
        write_yaml(os.path.join(out_dir, yaml_name), image_filename=pgm_name,
                   resolution=meta["resolution"], origin=meta["origin"],
                   negate=0, occupied_thresh=0.65, free_thresh=0.196)
    with open(os.path.join(out_dir, f"{name}_info.txt"), "w", encoding="utf-8") as f:
        f.write(_info_text(meta))
    export = export.lower()
    if export == "raw":
        _write_pair(name, pgm_raw)
    elif export == "inflated":
        _write_pair(name, pgm_infl)
    elif export == "both":
        _write_pair(name + "_raw", pgm_raw); _write_pair(name + "_inflated", pgm_infl)
    else:
        raise ValueError("export 必须是 raw|inflated|both 之一")

def generate_n_maps(n: int = 2, out_dir: str = "/tmp/connected_maps_v3", seed: int = 2025, export: str = "raw"):
    rng = np.random.default_rng(seed); os.makedirs(out_dir, exist_ok=True)
    names = []
    for i in range(n):
        pgm_raw, pgm_infl, meta = generate_random_map(
            width=782, height=906, resolution_m_per_px=0.05,
            origin_xyz=(-7.8, -6.58982, 0.0),
            arena_scale_min=0.45, arena_scale_max=0.95,
            node_count_range=(14, 28), edge_thickness_px=(6, 12), node_blob_px=(4, 10),
            extra_knn=2, extra_prob=0.30, closing_radius_px=2, carve_radius_px=3,
            robot_clearance_m=0.35 + 0.05*rng.random(),
            seed=int(rng.integers(0, 1<<31)), max_retries=40
        )
        name = f"ys_conn3_{i+1:02d}"; save_map_bundle(name, pgm_raw, pgm_infl, meta, out_dir=out_dir, export=export)
        names.append(name)
    zpath = os.path.join(out_dir, f"ys_connected_maps_v3_{export}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(os.path.join(out_dir, f"{name}_info.txt"), arcname=f"{name}_info.txt")
            if export == "raw":
                for ext in (".pgm", ".yaml"):
                    zf.write(os.path.join(out_dir, name + ext), arcname=name + ext)
            elif export == "inflated":
                for ext in (".pgm", ".yaml"):
                    zf.write(os.path.join(out_dir, name + ext), arcname=name + ext)
            elif export == "both":
                for suf in ("_raw","_inflated"):
                    for ext in (".pgm", ".yaml"):
                        zf.write(os.path.join(out_dir, name + suf + ext), arcname=name + suf + ext)
    return names, zpath

# ---------------- Env Engine Interface (raw-only) ----------------
def generate(**kwargs):
    """
    环境引擎约定接口（raw-only）：
      - 返回文件: {"yaml_path": <str>, "pgm_path": <str>}  —— 始终导出 raw
      - 返回内存: {"free": HxW bool, "resolution": <float>} —— 基于 raw

    关键参数（均可选）：
      width: int = 782
      height: int = 906
      resolution: float = 0.05               # m/px
      origin: (x,y,theta) = (-7.8, -6.58982, 0.0)
      seed: int | None = None

      # 生成细节（与现有实现一致，可不填沿用默认）
      arena_scale_min: float = 0.45
      arena_scale_max: float = 0.95
      node_count_range: (int,int) = (14, 28)
      edge_thickness_px: (int,int) = (6, 12)
      node_blob_px: (int,int) = (4, 10)
      extra_knn: int = 2
      extra_prob: float = 0.30
      closing_radius_px: int = 2
      carve_radius_px: int = 3
      robot_clearance_m: float = 0.35
      max_retries: int = 40

      # 文件导出（仅 raw）
      out_dir: str = "./_gen_out"
      name: str | None = None

      # 直接内存返回
      return_free: bool = False
    """
    # 基本参数
    width  = int(kwargs.get("width", 782))
    height = int(kwargs.get("height", 906))
    res    = float(kwargs.get("resolution", 0.05))
    origin = kwargs.get("origin", (-7.8, -6.58982, 0.0))
    if isinstance(origin, (list, tuple)) and len(origin) == 3:
        origin_xyz = (float(origin[0]), float(origin[1]), float(origin[2]))
    else:
        origin_xyz = (-7.8, -6.58982, 0.0)
    seed = kwargs.get("seed", None)

    # 生成地图（保持与已有实现的默认一致）
    pgm_raw, pgm_infl, meta = generate_random_map(
        width=width, height=height,
        resolution_m_per_px=res, origin_xyz=origin_xyz,
        arena_scale_min=float(kwargs.get("arena_scale_min", 0.45)),
        arena_scale_max=float(kwargs.get("arena_scale_max", 0.95)),
        node_count_range=tuple(kwargs.get("node_count_range", (14, 28))),
        edge_thickness_px=tuple(kwargs.get("edge_thickness_px", (6, 12))),
        node_blob_px=tuple(kwargs.get("node_blob_px", (4, 10))),
        extra_knn=int(kwargs.get("extra_knn", 2)),
        extra_prob=float(kwargs.get("extra_prob", 0.30)),
        closing_radius_px=int(kwargs.get("closing_radius_px", 2)),
        carve_radius_px=int(kwargs.get("carve_radius_px", 3)),
        robot_clearance_m=float(kwargs.get("robot_clearance_m", 0.35)),
        seed=seed, max_retries=int(kwargs.get("max_retries", 40))
    )

    # 分支1：直接返回 free/resolution（基于 raw）
    if bool(kwargs.get("return_free", False)):
        occ = (pgm_raw == 0)                 # 我们的 PGM: 0=占据(黑)，254/255=自由(白)
        free = (~occ).astype(bool)
        return {"free": free, "resolution": float(meta["resolution"])}

    # 分支2：写文件并返回路径（始终导出 raw）
    out_dir = kwargs.get("out_dir", "./_gen_out")
    os.makedirs(out_dir, exist_ok=True)
    if kwargs.get("name") is not None:
        stem = str(kwargs["name"])
    else:
        _sid = seed if seed is not None else int(np.random.default_rng().integers(1, 1 << 30))
        stem = f"ys_conn3_env_{_sid}"

    # save_map_bundle 会根据 export 写出对应对儿文件，我们这里强制 raw
    save_map_bundle(stem, pgm_raw, pgm_infl, meta, out_dir=out_dir, export="raw")
    yaml_path = os.path.join(out_dir, stem + ".yaml")
    pgm_path  = os.path.join(out_dir, stem + ".pgm")
    return {"yaml_path": yaml_path, "pgm_path": pgm_path}

# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高连接性障碍 + 连通自由空间（显式余裕保障）v3")
    parser.add_argument("--out", type=str, default="/tmp/connected_maps_v3", help="输出目录")
    parser.add_argument("--num", type=int, default=2, help="生成数量")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--export", type=str, default="raw", choices=["raw","inflated","both"], help="导出模式（默认 raw）")
    args = parser.parse_args()
    names, zip_path = generate_n_maps(n=args.num, out_dir=args.out, seed=args.seed, export=args.export)
    print("生成完成"); print("输出目录:", os.path.abspath(args.out)); print("地图列表:", ", ".join(names)); print("打包文件:", os.path.abspath(zip_path))

