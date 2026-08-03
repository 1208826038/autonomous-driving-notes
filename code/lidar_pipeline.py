"""
激光雷达点云目标提取最小流水线（教学版）
体素降采样 -> RANSAC 地面拟合与剔除 -> 欧式聚类(KD 树) -> PCA 定向 3D 包围框

坐标系：自车系 ego frame，x 前、y 左、z 上，单位 m。
"""
import numpy as np
from scipy.spatial import cKDTree

RNG = np.random.default_rng(20260804)


# ---------------- 0. 合成一帧点云：地面 + 两辆车 + 一个行人 ----------------
def _box_surface(center, size, yaw, n):
    """在长方体的 5 个可见面（不含底面）上均匀撒点，模拟激光只能打到表面。"""
    l, w, h = size
    # 各面面积权重：顶 + 前后 + 左右
    faces, areas = [], []
    for ax, sgn in [(2, +1), (0, +1), (0, -1), (1, +1), (1, -1)]:
        faces.append((ax, sgn))
        areas.append({0: w * h, 1: l * h, 2: l * w}[ax])
    areas = np.array(areas) / np.sum(areas)
    counts = RNG.multinomial(n, areas)
    pts = []
    half = np.array([l, w, h]) / 2.0
    for (ax, sgn), c in zip(faces, counts):
        p = RNG.uniform(-half, half, size=(c, 3))
        p[:, ax] = sgn * half[ax]
        pts.append(p)
    p = np.vstack(pts)
    p += RNG.normal(0, 0.02, p.shape)          # 2 cm 测距噪声
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return p @ R.T + np.asarray(center)


def make_scene():
    """返回 (points Nx3, 真值列表)。点数随距离平方衰减，复现真实稀疏性。"""
    def gz_at(x, y):                            # 地面高程模型（微坡 + 路拱）
        return -1.73 + 0.015 * x - 0.010 * y

    objs = [  # 名称, 底面中心(x,y), 尺寸(l,w,h), 航向 yaw(rad)
        ("车A-同向", (12.0, 3.20), (4.50, 1.90, 1.50), np.deg2rad(8.0)),
        ("车B-对向", (25.0, -3.50), (4.70, 1.95, 1.60), np.deg2rad(-172.0)),
        ("行人",     (8.0, -1.60), (0.60, 0.50, 1.70), np.deg2rad(35.0)),
    ]
    objs = [(n, (x, y, gz_at(x, y) + s[2] / 2), s, yaw) for n, (x, y), s, yaw in objs]
    K = 6.0e5                                   # 点密度常数（点·m²/表面积）
    clouds, gt = [], []
    for name, c, s, yaw in objs:
        r = np.hypot(c[0], c[1])
        area = 2 * (s[0] * s[2] + s[1] * s[2]) + s[0] * s[1]
        n = max(int(K * area / (r * r) / 100), 40)
        clouds.append(_box_surface(c, s, yaw, n))
        gt.append((name, c, s, np.rad2deg(yaw), n))

    # 地面：z = -1.73 + 0.015x - 0.01y（微坡+路拱），30 m 内 x-y 平面撒点
    n_g = 26000
    gx = RNG.uniform(0.0, 45.0, n_g)
    gy = RNG.uniform(-12.0, 12.0, n_g)
    gz = gz_at(gx, gy) + RNG.normal(0, 0.025, n_g)
    clouds.append(np.stack([gx, gy, gz], axis=1))
    return np.vstack(clouds).astype(np.float64), gt


# ---------------- 1. 体素降采样 ----------------
def voxel_downsample(pts, voxel=0.12):
    """每个立方体格子只保留质心，点数降一个量级但保形。"""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    m = inv.max() + 1
    cnt = np.bincount(inv, minlength=m).reshape(-1, 1)
    acc = np.zeros((m, 3))
    np.add.at(acc, inv, pts)
    return acc / cnt


# ---------------- 2. RANSAC 地面平面拟合 ----------------
def ransac_ground(pts, dist_th=0.15, iters=250, max_slope_deg=20.0):
    """随机取 3 点定平面，统计内点，返回 (法向 n, 截距 d, 地面掩码)。
    平面方程 n·p + d = 0，约束法向与 z 轴夹角小于 max_slope_deg 以排除墙面。"""
    best_mask, best_cnt, best_plane = None, -1, None
    N = len(pts)
    cos_lim = np.cos(np.deg2rad(max_slope_deg))
    for _ in range(iters):
        idx = RNG.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nvec = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nvec)
        if nn < 1e-6:
            continue
        nvec /= nn
        if nvec[2] < 0:
            nvec = -nvec
        if nvec[2] < cos_lim:                   # 太陡，不是地面
            continue
        d = -nvec @ p0
        mask = np.abs(pts @ nvec + d) < dist_th
        c = int(mask.sum())
        if c > best_cnt:
            best_cnt, best_mask, best_plane = c, mask, (nvec, d)
    # 用全部内点做一次最小二乘精化（SVD 求最小奇异向量）
    inl = pts[best_mask]
    ctr = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - ctr, full_matrices=False)
    nvec = Vt[-1] * (1 if Vt[-1][2] > 0 else -1)
    d = -nvec @ ctr
    mask = np.abs(pts @ nvec + d) < dist_th
    return nvec, d, mask


# ---------------- 3. 欧式聚类（KD 树 + BFS 区域生长） ----------------
def euclidean_cluster(pts, eps=0.55, min_pts=12, max_pts=20000):
    tree = cKDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    clusters = []
    for s in range(len(pts)):
        if visited[s]:
            continue
        queue, comp, visited[s] = [s], [], True
        while queue:
            i = queue.pop()
            comp.append(i)
            for j in tree.query_ball_point(pts[i], eps):
                if not visited[j]:
                    visited[j] = True
                    queue.append(j)
        if min_pts <= len(comp) <= max_pts:
            clusters.append(np.asarray(comp))
    return clusters


# ---------------- 4. PCA 定向包围框 ----------------
def pca_bbox(cluster_pts, plane):
    """BEV 平面内做 PCA 求主轴 -> yaw；旋转到主轴系取 min/max -> 长宽；
    高度用 zmax 减去地面平面在框心处的高度，避免地面剔除截断底部。"""
    xy = cluster_pts[:, :2]
    ctr_xy = xy.mean(axis=0)
    cov = np.cov((xy - ctr_xy).T)
    w_, V = np.linalg.eigh(cov)
    axis = V[:, np.argmax(w_)]                  # 最大特征值对应主轴
    yaw = np.arctan2(axis[1], axis[0])
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # 归一到 (-90°, 90°]，框有 180° 模糊
    c, s = np.cos(-yaw), np.sin(-yaw)
    loc = (xy - ctr_xy) @ np.array([[c, -s], [s, c]]).T
    lo, hi = loc.min(axis=0), loc.max(axis=0)
    l, w = hi - lo
    mid = (hi + lo) / 2
    c2, s2 = np.cos(yaw), np.sin(yaw)
    cx, cy = ctr_xy + np.array([[c2, -s2], [s2, c2]]) @ mid
    nvec, d = plane
    z_ground = -(nvec[0] * cx + nvec[1] * cy + d) / nvec[2]
    h = cluster_pts[:, 2].max() - z_ground
    return dict(center=(cx, cy, z_ground + h / 2), size=(l, w, h),
                yaw_deg=np.rad2deg(yaw), n=len(cluster_pts))


# ---------------- 演示 main ----------------
if __name__ == "__main__":
    raw, gt = make_scene()
    print(f"[0] 原始点云          : {len(raw):6d} 点")
    for name, c, s, y, n in gt:
        print(f"    真值 {name:8s} 距离 {np.hypot(c[0], c[1]):5.1f} m  "
              f"尺寸 {s[0]:.2f}x{s[1]:.2f}x{s[2]:.2f}  yaw {y:7.1f}°  打到 {n:4d} 点")

    ds = voxel_downsample(raw, voxel=0.12)
    print(f"[1] 体素降采样 0.12 m : {len(ds):6d} 点  (保留 {100*len(ds)/len(raw):.1f}%)")

    nvec, d, gmask = ransac_ground(ds, dist_th=0.15, iters=250)
    print(f"[2] RANSAC 地面       : 法向 ({nvec[0]:+.4f},{nvec[1]:+.4f},{nvec[2]:+.4f})  "
          f"d={d:+.4f}  坡度 {np.rad2deg(np.arccos(nvec[2])):.2f}°")
    obj = ds[~gmask]
    print(f"    地面内点 {int(gmask.sum()):5d}  剩余非地面 {len(obj):4d} 点")

    cls = euclidean_cluster(obj, eps=0.55, min_pts=12)
    print(f"[3] 欧式聚类 eps=0.55 : 检出 {len(cls)} 个目标")

    boxes = [pca_bbox(obj[c], (nvec, d)) for c in cls]
    boxes.sort(key=lambda b: np.hypot(b['center'][0], b['center'][1]))
    print(f"[4] PCA 定向包围框:")
    print(f"    {'#':>2} {'center(x,y,z)':>24} {'size(l,w,h)':>20} {'yaw':>8} {'pts':>5}")
    for i, b in enumerate(boxes):
        cx, cy, cz = b['center']
        l, w, h = b['size']
        print(f"    {i:2d} ({cx:7.2f},{cy:7.2f},{cz:7.2f}) "
              f"({l:5.2f},{w:5.2f},{h:5.2f}) {b['yaw_deg']:7.1f}° {b['n']:5d}")
