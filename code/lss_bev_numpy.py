# -*- coding: utf-8 -*-
"""
lss_bev_numpy.py
================
纯 numpy 实现的简化版 Lift-Splat-Shoot（LSS）视锥投影 -> BEV 栅格 全流程。

对应章节：《七、端到端智驾》第 7 节「工程实践」。

流程：
    0) 构造 6 路环视相机的内参 K 与 相机->自车 外参 (R, t)
    1) frustum  : 在特征图分辨率上构造 (D, H, W) 的视锥采样网格（像素 + 离散深度 bin）
    2) lift     : 特征图 c (C,H,W) 与 每像素离散深度分布 alpha (D,H,W) 做外积
                  -> 视锥特征 (D,H,W,C)，即 c ⊗ alpha
    3) project  : 视锥点经 K^-1 反投影到相机系，再经外参刚体变换到自车系
    4) splat    : voxel pooling —— 把落在 BEV 范围内的视锥点按栅格索引累加
                  提供两种实现：naive scatter(np.add.at) 与 sort+cumsum trick
    5) stats    : 非零栅格数、显存/内存占用、各阶段耗时
    6) sweep    : 深度 bin 数 D = 30 / 59 / 118 的显存与耗时对比

只依赖 numpy，不需要 torch。
运行：  python lss_bev_numpy.py
"""

import time
import numpy as np

np.random.seed(20260804)
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------------
# 全局配置：尽量贴近 nuScenes + BEVDet 的真实量级
# ----------------------------------------------------------------------------
IMG_H, IMG_W = 900, 1600      # nuScenes 原图分辨率
STRIDE = 32                   # 骨干网络下采样倍率 -> 特征图 28 x 50
FEAT_H, FEAT_W = IMG_H // STRIDE, IMG_W // STRIDE
C_FEAT = 64                   # BEV 特征通道数（BEVDet 默认 64）
N_CAM = 6                     # 环视 6 路

# BEV 栅格：x/y 各 [-50, 50) m，分辨率 0.5 m -> 200 x 200
XBOUND = (-50.0, 50.0, 0.5)
YBOUND = (-50.0, 50.0, 0.5)
ZBOUND = (-5.0, 3.0)          # z 方向直接压成一层（pillar），只做范围过滤

BEV_W = int((XBOUND[1] - XBOUND[0]) / XBOUND[2])   # 200
BEV_H = int((YBOUND[1] - YBOUND[0]) / YBOUND[2])   # 200

DTYPE = np.float32
MB = 1024.0 * 1024.0


# ----------------------------------------------------------------------------
# 0. 相机内外参
# ----------------------------------------------------------------------------
def make_intrinsics(stride=STRIDE):
    """nuScenes CAM_FRONT 量级的内参，按特征图 stride 缩放。"""
    fx = fy = 1266.417
    cx, cy = 816.267, 491.507
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=DTYPE)
    S = np.diag([1.0 / stride, 1.0 / stride, 1.0]).astype(DTYPE)
    return (S @ K).astype(DTYPE)          # 特征图坐标系下的内参


def yaw_matrix(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=DTYPE)


def pitch_matrix(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=DTYPE)


# 相机光学系 (x右, y下, z前) -> 自车系 (x前, y左, z上) 的固定轴变换
CAM2EGO_AXIS = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]], dtype=DTYPE)


def make_extrinsics():
    """6 路环视：航向角、安装位置、下倾角，量级参考量产环视布置。"""
    cfg = [
        # name,        yaw(deg), x(m),  y(m),  z(m), pitch_down(deg)
        ("CAM_FRONT",       0.0,  1.55,  0.00, 1.50, 2.0),
        ("CAM_FRONT_LEFT", 55.0,  1.45,  0.45, 1.50, 2.0),
        ("CAM_FRONT_RIGHT", -55.0, 1.45, -0.45, 1.50, 2.0),
        ("CAM_BACK_LEFT", 110.0,  1.00,  0.90, 1.55, 4.0),
        ("CAM_BACK_RIGHT", -110.0, 1.00, -0.90, 1.55, 4.0),
        ("CAM_BACK",      180.0, -0.55,  0.00, 1.55, 6.0),
    ]
    names, Rs, ts = [], [], []
    for name, yaw, x, y, z, pitch in cfg:
        R = yaw_matrix(yaw) @ pitch_matrix(pitch) @ CAM2EGO_AXIS
        names.append(name)
        Rs.append(R.astype(DTYPE))
        ts.append(np.array([x, y, z], dtype=DTYPE))
    return names, np.stack(Rs), np.stack(ts)


# ----------------------------------------------------------------------------
# 1. frustum：视锥采样网格
# ----------------------------------------------------------------------------
def create_frustum(d_min, d_max, d_step):
    """返回 (D, H, W, 3)，每个元素是 [u*d, v*d, d]（齐次待反投影形式）。"""
    depths = np.arange(d_min, d_max, d_step, dtype=DTYPE)          # (D,)
    D = depths.shape[0]
    us = np.arange(FEAT_W, dtype=DTYPE) + 0.5                      # 特征像素中心
    vs = np.arange(FEAT_H, dtype=DTYPE) + 0.5
    vv, uu = np.meshgrid(vs, us, indexing="ij")                    # (H, W)
    uu = np.broadcast_to(uu, (D, FEAT_H, FEAT_W))
    vv = np.broadcast_to(vv, (D, FEAT_H, FEAT_W))
    dd = depths[:, None, None] * np.ones((1, FEAT_H, FEAT_W), dtype=DTYPE)
    frustum = np.stack([uu * dd, vv * dd, dd], axis=-1)            # (D,H,W,3)
    return frustum.astype(DTYPE), depths


def frustum_to_ego(frustum, K, R, t):
    """视锥点 -> 相机系 -> 自车系。返回 (D,H,W,3)。"""
    Kinv = np.linalg.inv(K).astype(DTYPE)
    pts_cam = frustum @ Kinv.T                 # (D,H,W,3)  等价 d * K^-1 [u,v,1]^T
    pts_ego = pts_cam @ R.T + t                # 刚体变换
    return pts_ego


# ----------------------------------------------------------------------------
# 2. lift：特征 ⊗ 深度分布
# ----------------------------------------------------------------------------
def softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def make_fake_inputs(D):
    """伪造：每路相机的特征图 (C,H,W) 与 离散深度分布 logits (D,H,W)。"""
    feats = np.random.randn(N_CAM, C_FEAT, FEAT_H, FEAT_W).astype(DTYPE) * 0.5
    # 深度 logits 加一个随行号变化的先验：图像下方(近处地面)偏近，上方偏远
    row_prior = np.linspace(1.0, -1.0, FEAT_H, dtype=DTYPE)[None, :, None]
    base = np.random.randn(N_CAM, D, FEAT_H, FEAT_W).astype(DTYPE) * 0.8
    ramp = np.linspace(-1.5, 1.5, D, dtype=DTYPE)[None, :, None, None]
    logits = base + ramp * row_prior[None]
    alpha = softmax(logits, axis=1)            # 沿 D 维归一化 -> 每像素一个深度分布
    return feats, alpha.astype(DTYPE)


def lift(feat, alpha):
    """外积：c (C,H,W) ⊗ alpha (D,H,W) -> (D,H,W,C)。"""
    # (D,H,W,1) * (1,H,W,C)
    return (alpha[..., None] * feat.transpose(1, 2, 0)[None]).astype(DTYPE)


# ----------------------------------------------------------------------------
# 3. splat：voxel pooling
# ----------------------------------------------------------------------------
def bev_index(pts_ego):
    """自车系点 -> BEV 栅格索引，返回 (flat_idx, valid_mask)。"""
    x, y, z = pts_ego[..., 0], pts_ego[..., 1], pts_ego[..., 2]
    ix = np.floor((x - XBOUND[0]) / XBOUND[2]).astype(np.int64)
    iy = np.floor((y - YBOUND[0]) / YBOUND[2]).astype(np.int64)
    valid = ((ix >= 0) & (ix < BEV_W) & (iy >= 0) & (iy < BEV_H) &
             (z >= ZBOUND[0]) & (z < ZBOUND[1]))
    flat = ix * BEV_H + iy
    return flat, valid


def splat_scatter(flat_idx, feats, n_cell):
    """朴素 scatter：np.add.at，逐元素原子加，正确但慢。"""
    bev = np.zeros((n_cell, feats.shape[1]), dtype=DTYPE)
    np.add.at(bev, flat_idx, feats)
    return bev


def splat_bincount(flat_idx, feats, n_cell):
    """逐通道 bincount 的向量化 scatter，实际工程里常见的 numpy 写法。"""
    bev = np.empty((n_cell, feats.shape[1]), dtype=DTYPE)
    for c in range(feats.shape[1]):
        bev[:, c] = np.bincount(flat_idx, weights=feats[:, c],
                                minlength=n_cell).astype(DTYPE)
    return bev


def splat_cumsum_trick(flat_idx, feats, n_cell):
    """LSS 原论文的 cumsum trick：排序 -> 前缀和 -> 段末做差。"""
    order = np.argsort(flat_idx, kind="stable")
    idx_s = flat_idx[order]
    f_s = feats[order]
    csum = np.cumsum(f_s, axis=0)
    # 每段（同一栅格）的最后一个位置
    tail = np.ones(idx_s.shape[0], dtype=bool)
    tail[:-1] = idx_s[1:] != idx_s[:-1]
    csum_tail = csum[tail]
    seg_sum = np.empty_like(csum_tail)
    seg_sum[0] = csum_tail[0]
    seg_sum[1:] = csum_tail[1:] - csum_tail[:-1]
    bev = np.zeros((n_cell, feats.shape[1]), dtype=DTYPE)
    bev[idx_s[tail]] = seg_sum
    return bev


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def run_pipeline(d_min=1.0, d_max=60.0, d_step=1.0, verbose=True,
                 splat_fn=splat_cumsum_trick, repeat=3):
    K = make_intrinsics()
    names, Rs, ts = make_extrinsics()
    frustum, depths = create_frustum(d_min, d_max, d_step)
    D = depths.shape[0]
    feats, alpha = make_fake_inputs(D)

    n_pts_total = N_CAM * D * FEAT_H * FEAT_W
    lift_bytes = n_pts_total * C_FEAT * np.dtype(DTYPE).itemsize
    geom_bytes = n_pts_total * 3 * np.dtype(DTYPE).itemsize

    if verbose:
        print("=" * 74)
        print("简化 LSS：视锥 -> BEV  (pure numpy)")
        print("=" * 74)
        print(f"输入图像          : {IMG_W} x {IMG_H} x {N_CAM} 路")
        print(f"骨干 stride       : {STRIDE}  ->  特征图 {FEAT_W} x {FEAT_H}, C={C_FEAT}")
        print(f"深度 bin          : [{d_min}, {d_max}) step {d_step}  ->  D={D}")
        print(f"BEV 栅格          : {BEV_W} x {BEV_H} x {XBOUND[2]}m "
              f"(x/y ∈ [{XBOUND[0]:.0f}, {XBOUND[1]:.0f}) m)")
        print(f"视锥点总数        : {n_pts_total:,}  "
              f"( = {N_CAM} x {D} x {FEAT_H} x {FEAT_W} )")
        print(f"视锥特征张量      : {lift_bytes/MB:8.2f} MB  (float32)")
        print(f"视锥几何张量      : {geom_bytes/MB:8.2f} MB  (float32)")
        print("-" * 74)

    # ---- 阶段 1: geometry ----
    t0 = time.perf_counter()
    all_pts = np.empty((N_CAM,) + frustum.shape, dtype=DTYPE)
    for i in range(N_CAM):
        all_pts[i] = frustum_to_ego(frustum, K, Rs[i], ts[i])
    t_geom = time.perf_counter() - t0

    # ---- 阶段 2: lift ----
    t0 = time.perf_counter()
    all_lift = np.empty((N_CAM, D, FEAT_H, FEAT_W, C_FEAT), dtype=DTYPE)
    for i in range(N_CAM):
        all_lift[i] = lift(feats[i], alpha[i])
    t_lift = time.perf_counter() - t0

    # ---- 阶段 3: index + filter ----
    t0 = time.perf_counter()
    flat, valid = bev_index(all_pts)
    flat_v = flat[valid]
    feat_v = all_lift[valid]
    t_index = time.perf_counter() - t0

    # ---- 阶段 4: splat（重复计时取最小值） ----
    n_cell = BEV_W * BEV_H
    ts_splat = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        bev = splat_fn(flat_v, feat_v, n_cell)
        ts_splat.append(time.perf_counter() - t0)
    t_splat = min(ts_splat)

    bev_map = bev.reshape(BEV_W, BEV_H, C_FEAT)
    occupied = np.count_nonzero(np.abs(bev_map).sum(axis=-1) > 1e-8)
    kept = int(valid.sum())

    if verbose:
        print(f"[1] geometry  反投影+外参变换 : {t_geom*1e3:8.2f} ms")
        print(f"[2] lift      c ⊗ alpha 外积  : {t_lift*1e3:8.2f} ms")
        print(f"[3] index     栅格索引+范围过滤: {t_index*1e3:8.2f} ms")
        print(f"[4] splat     voxel pooling   : {t_splat*1e3:8.2f} ms"
              f"   [{splat_fn.__name__}]")
        print(f"    ---------------------------------------------")
        print(f"    合计                       : "
              f"{(t_geom+t_lift+t_index+t_splat)*1e3:8.2f} ms")
        print("-" * 74)
        print(f"落在 BEV 范围内的视锥点   : {kept:,} / {n_pts_total:,} "
              f"({100.0*kept/n_pts_total:.1f}%)")
        print(f"被丢弃（出界/超高）的点   : {n_pts_total-kept:,} "
              f"({100.0*(n_pts_total-kept)/n_pts_total:.1f}%)")
        print(f"BEV 非零栅格数            : {occupied:,} / {n_cell:,} "
              f"({100.0*occupied/n_cell:.1f}%)")
        print(f"BEV 特征图                : {BEV_W}x{BEV_H}x{C_FEAT} = "
              f"{bev.nbytes/MB:.2f} MB")
        print(f"每个非零栅格平均落点数    : {kept/max(occupied,1):.1f}")
        # 环形密度剖面：按到自车的距离分环统计非零率
        print("-" * 74)
        print("BEV 非零栅格的距离分布（自车为中心的同心环）:")
        gx = (np.arange(BEV_W) + 0.5) * XBOUND[2] + XBOUND[0]
        gy = (np.arange(BEV_H) + 0.5) * YBOUND[2] + YBOUND[0]
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        rad = np.sqrt(GX ** 2 + GY ** 2)
        nz = np.abs(bev_map).sum(axis=-1) > 1e-8
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]:
            m = (rad >= lo) & (rad < hi)
            print(f"   {lo:2d}-{hi:2d} m : 非零 {nz[m].sum():5d} / {m.sum():5d} "
                  f"栅格 = {100.0*nz[m].sum()/m.sum():5.1f}%")
        print("=" * 74)

    return dict(D=D, n_pts=n_pts_total, lift_mb=lift_bytes / MB,
                t_geom=t_geom, t_lift=t_lift, t_index=t_index, t_splat=t_splat,
                kept=kept, occupied=occupied, bev=bev_map)


def compare_splat_impls():
    """三种 voxel pooling 实现的耗时对比（D=59）。"""
    print()
    print("=" * 74)
    print("voxel pooling 三种实现对比  (D=59, 6 路相机)")
    print("=" * 74)
    K = make_intrinsics()
    _, Rs, ts = make_extrinsics()
    frustum, depths = create_frustum(1.0, 60.0, 1.0)
    D = depths.shape[0]
    feats, alpha = make_fake_inputs(D)
    all_pts = np.stack([frustum_to_ego(frustum, K, Rs[i], ts[i])
                        for i in range(N_CAM)])
    all_lift = np.stack([lift(feats[i], alpha[i]) for i in range(N_CAM)])
    flat, valid = bev_index(all_pts)
    flat_v, feat_v = flat[valid], all_lift[valid]
    n_cell = BEV_W * BEV_H
    print(f"参与 pooling 的点数: {flat_v.shape[0]:,}  特征维度: {C_FEAT}")
    print(f"{'实现':<26}{'耗时(ms)':>12}{'相对最快':>12}   校验")
    print("-" * 74)
    ref = None
    rows = []
    for fn in (splat_scatter, splat_bincount, splat_cumsum_trick):
        t0 = time.perf_counter()
        out = fn(flat_v, feat_v, n_cell)
        dt = (time.perf_counter() - t0) * 1e3
        if ref is None:
            ref = out
            chk = "baseline"
        else:
            chk = f"max|Δ|={np.abs(out-ref).max():.2e}"
        rows.append((fn.__name__, dt, chk))
    fastest = min(r[1] for r in rows)
    for name, dt, chk in rows:
        print(f"{name:<26}{dt:12.2f}{dt/fastest:11.2f}x   {chk}")
    print("=" * 74)


def sweep_depth_bins():
    """深度 bin 数对显存与耗时的影响。"""
    print()
    print("=" * 74)
    print("深度 bin 数 D 的影响：显存 / 耗时")
    print("=" * 74)
    cfgs = [("D=30  [1,60) step2.0", 1.0, 60.0, 2.0),
            ("D=59  [1,60) step1.0", 1.0, 60.0, 1.0),
            ("D=118 [1,60) step0.5", 1.0, 60.0, 0.5)]
    print(f"{'配置':<24}{'D':>5}{'视锥点数':>13}{'lift显存(MB)':>15}"
          f"{'lift(ms)':>11}{'splat(ms)':>11}{'非零栅格':>10}")
    print("-" * 92)
    res = []
    for name, lo, hi, st in cfgs:
        r = run_pipeline(lo, hi, st, verbose=False, repeat=3)
        res.append((name, r))
        print(f"{name:<24}{r['D']:>5}{r['n_pts']:>13,}{r['lift_mb']:>15.1f}"
              f"{r['t_lift']*1e3:>11.1f}{r['t_splat']*1e3:>11.1f}"
              f"{r['occupied']:>10,}")
    print("-" * 92)
    base = res[1][1]
    for name, r in res:
        print(f"{name:<24} 相对 D=59: 显存 x{r['lift_mb']/base['lift_mb']:.2f}, "
              f"lift 耗时 x{r['t_lift']/base['t_lift']:.2f}, "
              f"splat 耗时 x{r['t_splat']/base['t_splat']:.2f}, "
              f"非零栅格 {r['occupied']:,}")
    print("=" * 74)


def bev_ascii(bev_map, step=8):
    """把 BEV 非零掩码降采样成 ASCII 图，直观看覆盖形状。"""
    nz = (np.abs(bev_map).sum(axis=-1) > 1e-8)
    h, w = nz.shape
    print()
    print("=" * 74)
    print(f"BEV 覆盖 ASCII 图（{step}x{step} 栅格降采样，'#'=该块有特征, "
          f"'.'=空, 'E'=自车）")
    print("上=车头(+x), 左=车左(+y)；每字符代表 "
          f"{step*XBOUND[2]:.0f}m x {step*YBOUND[2]:.0f}m")
    print("=" * 74)
    rows = []
    for i in range(h - step, -1, -step):          # +x 在上
        line = []
        for j in range(w - step, -1, -step):      # +y 在左
            blk = nz[i:i + step, j:j + step]
            line.append("#" if blk.any() else ".")
        rows.append("".join(line))
    cx, cy = (h // 2) // step, (w // 2) // step
    rows[len(rows) - 1 - cx] = (rows[len(rows) - 1 - cx][:cy] + "E" +
                                rows[len(rows) - 1 - cx][cy + 1:])
    for r in rows:
        print("   " + r)
    print("=" * 74)


if __name__ == "__main__":
    main_res = run_pipeline(1.0, 60.0, 1.0, verbose=True)
    bev_ascii(main_res["bev"], step=8)
    compare_splat_impls()
    sweep_depth_bins()
