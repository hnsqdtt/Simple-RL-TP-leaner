# -*- coding: utf-8 -*-
"""
random_map_generator_rounded_v4_fixed.py
----------------------------------------
修复点（相对你给的 v4）
1) 移除逐像素随机稀释&随机开孔 —— 噪点根源；
2) 过密时用“均匀侵蚀”降低障碍率（仅作用于 arena 内，不影响外侧硬边界）；
3) 连通性：在自由区组件的“最近边界对”之间开“胶囊通道”，可选在“带余裕的配置空间”上再保证一次；
4) 新增：--outside-noise 把 arena 外侧替换为噪点（salt/mosaic）；
5) 额外导出 *_arena.pgm 掩膜，255/254=内部、0=外部。
依赖：numpy；不需要 OpenCV。
"""

import os, math, zipfile, argparse, textwrap
from typing import Tuple, List, Optional, Iterable
import numpy as np
from collections import deque

# ---------- I/O ----------
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

# ---------- 形态学 ----------
def disk_brush(radius_px: int) -> np.ndarray:
    r = radius_px
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return ((x*x + y*y) <= r*r).astype(np.uint8)

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
    if radius_px <= 0: return binary
    br = disk_brush(radius_px)
    return erode(dilate(binary, br), br)

# ---------- 多边形/圆角/栅格 ----------
def polygon_area_ccw(pts: np.ndarray) -> float:
    x = pts[:,0]; y = pts[:,1]
    return 0.5 * np.sum(x*np.roll(y,-1) - y*np.roll(x,-1))

def ensure_ccw(pts: np.ndarray) -> np.ndarray:
    return pts if polygon_area_ccw(pts) > 0 else pts[::-1].copy()

def rounded_corner(prev, V, nxt, r: float, arc_samples: int = 16):
    a1 = prev - V; L1 = np.linalg.norm(a1); a1 = a1 / (L1 + 1e-12)
    a2 = nxt  - V; L2 = np.linalg.norm(a2); a2 = a2 / (L2 + 1e-12)
    cos_phi = float(np.clip(np.dot(a1, a2), -1.0, 1.0))
    phi = float(np.arccos(cos_phi))
    if phi < 1e-3 or phi > np.pi - 1e-3:
        return None
    tan_half = float(np.tan(phi/2.0))
    r_max = min(L1, L2) * tan_half
    if r <= 1e-6 or r > r_max - 1e-6:
        r = max(1e-6, 0.9 * r_max)
    t = r / tan_half
    T1 = V + a1 * t
    T2 = V + a2 * t
    b = a1 + a2
    nb = float(np.linalg.norm(b))
    if nb < 1e-9:
        return None
    b = b / nb
    C = V + b * (r / float(np.sin(phi/2.0)))
    ang1 = float(np.arctan2(T1[1]-C[1], T1[0]-C[0]))
    ang2 = float(np.arctan2(T2[1]-C[1], T2[0]-C[0]))
    d = (ang2 - ang1) % (2*np.pi)
    if d > np.pi:
        angles = np.linspace(ang1, ang2 - 2*np.pi, arc_samples)
    else:
        angles = np.linspace(ang1, ang2, arc_samples)
    arc = np.stack([C[0] + r*np.cos(angles), C[1] + r*np.sin(angles)], axis=1)
    arc[0] = T1; arc[-1] = T2
    return T1, T2, C, arc

def round_polygon_corners_exact(poly: np.ndarray, round_vertices: Iterable[int],
                                r_scale: Tuple[float,float] = (0.25, 0.7),
                                arc_samples: int = 18,
                                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng()
    poly = ensure_ccw(poly); N = len(poly)
    round_set = set([i % N for i in round_vertices])
    out: List[np.ndarray] = []
    for i in range(N):
        prev = poly[(i-1) % N]; V = poly[i]; nxt = poly[(i+1) % N]
        if i in round_set:
            a1 = prev - V; L1 = np.linalg.norm(a1); a1 = a1 / (L1 + 1e-12)
            a2 = nxt  - V; L2 = np.linalg.norm(a2); a2 = a2 / (L2 + 1e-12)
            cos_phi = float(np.clip(np.dot(a1, a2), -1.0, 1.0))
            phi = float(np.arccos(cos_phi))
            if phi < 1e-3 or phi > np.pi - 1e-3:
                out.append(V); continue
            tan_half = float(np.tan(phi/2.0))
            r_max = max(1e-6, min(L1, L2) * tan_half)
            frac = float(rng.uniform(r_scale[0], r_scale[1]))
            r = frac * r_max
            rc = rounded_corner(prev, V, nxt, r, arc_samples=arc_samples)
            if rc is None: out.append(V)
            else:
                T1, T2, C, arc = rc
                out.append(T1); out.extend(list(arc[1:-1])); out.append(T2)
        else:
            out.append(V)
    return np.array(out, dtype=np.float32)

def fill_polygon_mask(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = poly.copy()
    pts[:,0] = np.clip(pts[:,0], 0, W-1); pts[:,1] = np.clip(pts[:,1], 0, H-1)
    if len(pts) < 3: return mask
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

# ---------- 采样 ----------
def sample_triangle(center: Tuple[float,float], scale_px: float, rng: np.random.Generator, jitter: float = 0.35) -> np.ndarray:
    cx, cy = center
    base = np.sort(rng.uniform(0, 2*np.pi, size=3))
    r_base = scale_px * (1.0 + rng.uniform(-jitter, jitter, size=3))
    xs = cx + r_base * np.cos(base); ys = cy + r_base * np.sin(base)
    tri = np.stack([xs, ys], axis=1); tri = ensure_ccw(tri); return tri

def sample_arena_polygon(H: int, W: int, rng: np.random.Generator,
                         scale_min: float = 0.35, scale_max: float = 1.0,
                         n_range: Tuple[int,int] = (8, 16),
                         jitter_range: Tuple[float,float] = (0.15, 0.45)) -> np.ndarray:
    n = int(rng.integers(n_range[0], n_range[1]+1))
    scale = float(rng.uniform(scale_min, scale_max))
    jitter = float(rng.uniform(jitter_range[0], jitter_range[1]))
    base_r = 0.5 * scale * min(H, W)
    r_max = base_r * (1.0 + jitter); margin = 4
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
    return ensure_ccw(np.stack([xs, ys], axis=1))

# ---------- 连通性：组件 + 通道开挖 ----------
def label_components(binary: np.ndarray):
    H, W = binary.shape
    labels = -np.ones((H, W), dtype=np.int32)
    nid = 0; sizes = []
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1)]
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
    if len(pts) == 0: pts = list(zip(xs, ys))
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
    if rng is None: rng = np.random.default_rng()
    for _ in range(64):
        free = ((occ_raw == 0) & (arena_mask == 1)).astype(np.uint8)
        labels, sizes = label_components(free)
        if len(sizes) <= 1: return
        main_id = int(np.argmax(sizes))
        A = boundary_points(labels, main_id, limit=800, arena_mask=arena_mask, rng=rng)
        if len(A) == 0: return
        merged = False
        for lid in range(len(sizes)):
            if lid == main_id: continue
            B = boundary_points(labels, lid, limit=400, arena_mask=arena_mask, rng=rng)
            if len(B) == 0: continue
            if len(A) > 400: A = A[rng.choice(len(A), size=400, replace=False)]
            if len(B) > 400: B = B[rng.choice(len(B), size=400, replace=False)]
            d = A[:,None,:] - B[None,:,:]; d2 = d[:,:,0]**2 + d[:,:,1]**2
            i,j = np.unravel_index(np.argmin(d2), d2.shape)
            x0,y0 = int(A[i,0]), int(A[i,1]); x1,y1 = int(B[j,0]), int(B[j,1])
            carve_capsule_zero(occ_raw, x0,y0,x1,y1, r=carve_radius)
            merged = True
        if not merged: return

def ensure_free_connected_with_clearance(occ_raw: np.ndarray, arena_mask: np.ndarray, clearance_px: int, rng: Optional[np.random.Generator] = None) -> None:
    if rng is None: rng = np.random.default_rng()
    R = int(max(1, clearance_px + 1))
    brush = disk_brush(clearance_px)
    for _ in range(64):
        occ_infl = dilate(occ_raw, brush)
        free = ((occ_infl == 0) & (arena_mask == 1)).astype(np.uint8)
        labels, sizes = label_components(free)
        if len(sizes) <= 1: return
        main = int(np.argmax(sizes))
        A = boundary_points(labels, main, limit=800, arena_mask=arena_mask, rng=rng)
        if len(A) == 0: return
        merged = False
        for lid in range(len(sizes)):
            if lid == main: continue
            B = boundary_points(labels, lid, limit=400, arena_mask=arena_mask, rng=rng)
            if len(B) == 0: continue
            if len(A) > 400: A = A[rng.choice(len(A), size=400, replace=False)]
            if len(B) > 400: B = B[rng.choice(len(B), size=400, replace=False)]
            d = A[:,None,:] - B[None,:,:]; d2 = d[:,:,0]**2 + d[:,:,1]**2
            i,j = np.unravel_index(np.argmin(d2), d2.shape)
            x0,y0 = int(A[i,0]), int(A[i,1]); x1,y1 = int(B[j,0]), int(B[j,1])
            carve_capsule_zero(occ_raw, x0,y0,x1,y1, r=R)
            merged = True
        if not merged: return

# ---------- 主流程 ----------
def generate_random_map(
    width: int = 782, height: int = 906,
    resolution_m_per_px: float = 0.05,
    origin_xyz: Tuple[float,float,float] = (-7.8, -6.58982, 0.0),
    robot_clearance_m: float = 0.35,
    tri_count_range: Tuple[int,int] = (18, 40),
    tri_scale_px: Tuple[int,int] = (40, 160),
    round_k: int = -1,
    r_scale: Tuple[float,float] = (0.25, 0.7),
    arc_samples: int = 18,
    obstacle_density_soft_cap: float = 0.48,
    arena_scale_min: float = 0.35, arena_scale_max: float = 1.00,
    closing_radius_px: int = 0,
    connect_radius_px: int = 3,
    ensure_with_clearance: bool = True,
    outside_noise: Optional[Tuple] = None,      # None / ("salt", p) / ("mosaic", block, p)
    seed: Optional[int] = None, max_retries: int = 40,
):
    rng = np.random.default_rng(seed); H, W = height, width

    # 外形
    arena_poly = sample_arena_polygon(H, W, rng, scale_min=arena_scale_min, scale_max=arena_scale_max)
    arena_mask = fill_polygon_mask(arena_poly, H, W)  # 1=内部
    occ_raw = np.ones((H, W), dtype=np.uint8); occ_raw[arena_mask == 1] = 0

    # 三角形障碍
    n_tri = int(rng.integers(tri_count_range[0], tri_count_range[1]+1))
    margin = 6
    for _ in range(n_tri):
        scale = int(rng.integers(tri_scale_px[0], tri_scale_px[1]+1))
        cx = float(rng.integers(margin, W - margin))
        cy = float(rng.integers(margin, H - margin))
        tri = sample_triangle((cx, cy), scale, rng, jitter=0.35)
        k = int(rng.integers(0, 4)) if round_k < 0 else max(0, min(3, int(round_k)))
        if k > 0:
            chosen = list(rng.choice(3, size=k, replace=False))
            tri = round_polygon_corners_exact(tri, chosen, r_scale=r_scale, arc_samples=arc_samples, rng=rng)
        tri_mask = fill_polygon_mask(tri, H, W) & arena_mask
        occ_raw |= tri_mask.astype(np.uint8)

    # 障碍率过高 -> 均匀侵蚀（无随机孔洞）
    occ_ratio = (occ_raw & arena_mask).sum() / max(1, arena_mask.sum())
    if occ_ratio > obstacle_density_soft_cap:
        r = 1 + int((occ_ratio - obstacle_density_soft_cap) > 0.10)  # 1或2像素
        inner = erode(occ_raw & arena_mask, disk_brush(r))
        occ_raw = (inner & arena_mask) | (1 - arena_mask)

    # 可选小闭运算（不产生噪点）
    if closing_radius_px > 0:
        inner = morph_close(occ_raw & arena_mask, closing_radius_px)
        occ_raw = (inner & arena_mask) | (1 - arena_mask)

    # raw 连通保障
    ensure_free_connected_raw(occ_raw, arena_mask, carve_radius=max(2, connect_radius_px), rng=rng)

    # 配置空间连通保障（带余裕）
    clearance_px = max(1, int(np.ceil(robot_clearance_m / resolution_m_per_px)))
    if ensure_with_clearance:
        ensure_free_connected_with_clearance(occ_raw, arena_mask, clearance_px=clearance_px, rng=rng)

    # 外侧噪点（仅作用 arena 外）
    if outside_noise is not None:
        mode = outside_noise[0].lower()
        if mode == "salt":
            p = float(outside_noise[1]) if len(outside_noise)>=2 else 0.5
            nz = (np.random.default_rng(seed).random((H,W)) < p).astype(np.uint8)
        else:
            block = int(outside_noise[1]) if len(outside_noise)>=2 else 6
            p = float(outside_noise[2]) if len(outside_noise)>=3 else 0.5
            gh, gw = (H + block - 1)//block, (W + block - 1)//block
            grid = (np.random.default_rng(seed).random((gh, gw)) < p).astype(np.uint8)
            nz = np.repeat(np.repeat(grid, block, axis=0), block, axis=1)[:H,:W]
        occ_out = np.where(nz==1, 0, 1).astype(np.uint8)
        occ_raw = np.where(arena_mask==0, occ_out, occ_raw)

    # 导出
    pgm_raw = np.where(occ_raw == 1, 0, 254).astype(np.uint8)
    occ_infl = dilate(occ_raw, disk_brush(clearance_px))
    pgm_infl = np.where(occ_infl == 1, 0, 254).astype(np.uint8)
    arena_u8 = (arena_mask*254).astype(np.uint8)

    meta = {"resolution": resolution_m_per_px, "origin": origin_xyz, "width": width, "height": height, "clearance_px": int(clearance_px)}
    return pgm_raw, pgm_infl, arena_u8, meta

# ---------- 导出/打包 ----------
def save_map_bundle(name: str, pgm_raw: np.ndarray, pgm_infl: np.ndarray, arena_u8: np.ndarray, meta: dict, out_dir: str, export: str = "raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    def _write_pair(stem: str, pgm_img: np.ndarray):
        pgm_name = f"{stem}.pgm"; yaml_name = f"{stem}.yaml"
        write_pgm(os.path.join(out_dir, pgm_name), pgm_img)
        write_yaml(os.path.join(out_dir, yaml_name), image_filename=pgm_name,
                   resolution=meta["resolution"], origin=meta["origin"],
                   negate=0, occupied_thresh=0.65, free_thresh=0.196)
    write_pgm(os.path.join(out_dir, f"{name}_arena.pgm"), arena_u8)
    export = export.lower()
    if export == "raw":
        _write_pair(name, pgm_raw)
    elif export == "inflated":
        _write_pair(name, pgm_infl)
    elif export == "both":
        _write_pair(name + "_raw", pgm_raw); _write_pair(name + "_inflated", pgm_infl)
    else:
        raise ValueError("export 必须是 raw|inflated|both 之一")

def generate_n_maps(n: int = 2, out_dir: str = "/tmp/rounded_maps_v4_fixed", seed: int = 2025,
                    round_k: int = -1, export: str = "raw",
                    arena_scale_min: float = 0.35, arena_scale_max: float = 1.0,
                    outside_noise: Optional[Tuple] = None):
    rng = np.random.default_rng(seed); os.makedirs(out_dir, exist_ok=True)
    names = []
    for i in range(n):
        pgm_raw, pgm_infl, arena_u8, meta = generate_random_map(
            width=782, height=906, resolution_m_per_px=0.05,
            origin_xyz=(-7.8, -6.58982, 0.0),
            robot_clearance_m=0.35 + 0.1*rng.random(),
            tri_count_range=(18, 40), tri_scale_px=(40, 160),
            round_k=round_k, r_scale=(0.25, 0.7), arc_samples=18,
            obstacle_density_soft_cap=0.48,
            arena_scale_min=arena_scale_min, arena_scale_max=arena_scale_max,
            closing_radius_px=0, connect_radius_px=3,
            ensure_with_clearance=True, outside_noise=outside_noise,
            seed=int(rng.integers(0, 1<<31)), max_retries=40
        )
        name = f"ys_round4_fix_{i+1:02d}"
        save_map_bundle(name, pgm_raw, pgm_infl, arena_u8, meta, out_dir=out_dir, export=export); names.append(name)
    zpath = os.path.join(out_dir, f"ys_rounded_maps_v4_fixed_{export}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(os.path.join(out_dir, name + "_arena.pgm"), arcname=name + "_arena.pgm")
            if export == "raw":
                for ext in (".pgm", ".yaml"): zf.write(os.path.join(out_dir, name + ext), arcname=name + ext)
            elif export == "inflated":
                for ext in (".pgm", ".yaml"): zf.write(os.path.join(out_dir, name + ext), arcname=name + ext)
            elif export == "both":
                for suf in ("_raw","_inflated"):
                    for ext in (".pgm", ".yaml"): zf.write(os.path.join(out_dir, name + suf + ext), arcname=name + suf + ext)
    return names, zpath

# ---------- Env Engine Interface (raw-only) ----------
def generate(**kwargs):
    """
    环境引擎约定接口（raw-only）：
      - 返回文件: {"yaml_path": <str>, "pgm_path": <str>} —— 始终导出 raw
      - 返回内存: {"free": HxW bool, "resolution": <float>} —— 基于 raw（True=free）

    常用参数（可选，均有默认值）：
      width:int=782, height:int=906, resolution:float=0.05, origin:(x,y,theta)=(-7.8,-6.58982,0.0)
      round_k:int=-1, arena_scale_min:float=0.35, arena_scale_max:float=1.0
      outside_noise: None | ("salt", p) | ("mosaic", block, p)
      connect_radius_px:int=3, closing_radius_px:int=0, ensure_with_clearance:bool=True
      robot_clearance_m:float=0.35, max_retries:int=40, seed:int|None=None

      # 保存到文件：
      out_dir:str="./_gen_out", name:str|None=None
      # 直接返回free：
      return_free:bool=False
    """
    import os, numpy as np

    # 基本参数
    width  = int(kwargs.get("width", 782))
    height = int(kwargs.get("height", 906))
    res    = float(kwargs.get("resolution", 0.05))
    origin = kwargs.get("origin", (-7.8, -6.58982, 0.0))
    origin_xyz = (float(origin[0]), float(origin[1]), float(origin[2])) if isinstance(origin, (list, tuple)) and len(origin)==3 else (-7.8, -6.58982, 0.0)
    seed   = kwargs.get("seed", None)

    # 生成一张地图（沿用本文件默认风格，可被 kwargs 覆盖）
    pgm_raw, pgm_infl, arena_u8, meta = generate_random_map(
        width=width, height=height, resolution_m_per_px=res, origin_xyz=origin_xyz,
        robot_clearance_m=float(kwargs.get("robot_clearance_m", 0.35)),
        tri_count_range=tuple(kwargs.get("tri_count_range", (18, 40))),
        tri_scale_px=tuple(kwargs.get("tri_scale_px", (40, 160))),
        round_k=int(kwargs.get("round_k", -1)),
        r_scale=tuple(kwargs.get("r_scale", (0.25, 0.7))),
        arc_samples=int(kwargs.get("arc_samples", 18)),
        obstacle_density_soft_cap=float(kwargs.get("obstacle_density_soft_cap", 0.48)),
        arena_scale_min=float(kwargs.get("arena_scale_min", 0.35)),
        arena_scale_max=float(kwargs.get("arena_scale_max", 1.0)),
        closing_radius_px=int(kwargs.get("closing_radius_px", 0)),
        connect_radius_px=int(kwargs.get("connect_radius_px", 3)),
        ensure_with_clearance=bool(kwargs.get("ensure_with_clearance", True)),
        outside_noise=kwargs.get("outside_noise", None),
        seed=seed if seed is not None else None,
        max_retries=int(kwargs.get("max_retries", 40)),
    )

    # 分支1：直接返回 free/resolution（raw）
    if bool(kwargs.get("return_free", False)):
        occ = (pgm_raw == 0)          # 本文件PGM语义：0=占据(黑)，254/255=自由(白)
        free = (~occ).astype(bool)
        return {"free": free, "resolution": float(meta["resolution"])}

    # 分支2：写文件并返回路径（raw-only）
    out_dir = kwargs.get("out_dir", "./_gen_out")
    os.makedirs(out_dir, exist_ok=True)
    if kwargs.get("name") is not None:
        stem = str(kwargs["name"])
    else:
        _sid = seed if seed is not None else int(np.random.default_rng().integers(1, 1 << 30))
        stem = f"ys_round4_fix_{_sid}"

    # 仅导出 raw（save_map_bundle 会顺带写出 *_arena.pgm，保留无妨）
    save_map_bundle(stem, pgm_raw, pgm_infl, arena_u8, meta, out_dir=out_dir, export="raw")
    yaml_path = os.path.join(out_dir, stem + ".yaml")
    pgm_path  = os.path.join(out_dir, stem + ".pgm")
    return {"yaml_path": yaml_path, "pgm_path": pgm_path}

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="不规则外围封闭地图 + 圆角三角形障碍 v4（无噪点修复版）")
    parser.add_argument("--out", type=str, default="/tmp/rounded_maps_v4_fixed", help="输出目录")
    parser.add_argument("--num", type=int, default=2, help="生成数量")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--round-k", type=int, default=-1, help="-1 表示每个三角形随机 0~3；否则固定 0~3")
    parser.add_argument("--export", type=str, default="raw", choices=["raw","inflated","both"], help="导出模式（默认 raw）")
    parser.add_argument("--arena-scale-min", type=float, default=0.35, help="外形最小尺度（0~1）")
    parser.add_argument("--arena-scale-max", type=float, default=1.0, help="外形最大尺度（0~1）")
    parser.add_argument("--outside-noise", type=str, default="", help="外侧噪点: '' / 'salt:p' / 'mosaic:block,p'")
    args = parser.parse_args()

    onoise = None
    if args.outside_noise:
        s = args.outside_noise.split(":")
        if s[0]=="salt":
            p = float(s[1]) if len(s)>=2 else 0.5
            onoise = ("salt", p)
        elif s[0]=="mosaic":
            rest = s[1] if len(s)>=2 else ""
            block, p = 6, 0.5
            if rest:
                ss = rest.split(",")
                if len(ss)>=1 and ss[0]: block = int(ss[0])
                if len(ss)>=2 and ss[1]: p = float(ss[1])
            onoise = ("mosaic", block, p)

    names, zip_path = generate_n_maps(n=args.num, out_dir=args.out, seed=args.seed,
                                      round_k=args.round_k, export=args.export,
                                      arena_scale_min=args.arena_scale_min,
                                      arena_scale_max=args.arena_scale_max,
                                      outside_noise=onoise)
    print("生成完成"); print("输出目录:", os.path.abspath(args.out)); print("地图列表:", ", ".join(names)); print("打包文件:", os.path.abspath(zip_path))
