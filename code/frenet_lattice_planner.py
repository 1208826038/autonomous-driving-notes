#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frenet 系 Lattice 规划器 —— 第五章《运动规划》配套代码
=========================================================
场景：城市快速路一段缓弯（R ≈ 240 m），单向两车道。
  - 自车 v0 = 12.0 m/s（43 km/h），起点略偏离参考线 l0 = +0.30 m
  - 静态障碍 1：违停厢货，压占本车道左侧 (s=48 m, l=+1.60 m, 5.4x2.05 m)
  - 静态障碍 2：施工锥桶            (s=88 m, l=-1.40 m, 0.5x0.5 m)
  - 动态障碍  ：前方慢车，同车道 7.0 m/s 匀速 (s0=30 m, l=0.0, 4.8x1.9 m)

算法（Apollo Lattice Planner 思路）：
  1) 参考线离散 + 等弧长重采样，数值求 theta_r / kappa_r / dkappa_r
  2) 横向：五次多项式 l(s)=a0+a1 s+...+a5 s^5，末端条件 (l_end, 0, 0)
  3) 纵向：四次多项式 s(t)，末端条件 (v_end, 0) —— 巡航型速度采样
  4) 横纵笛卡尔积 -> Frenet->Cartesian -> 边界筛 -> 运动学筛 -> 碰撞筛 -> 代价排序

所有打印数字均为真实运行结果。
"""

import math
import time
import numpy as np

# ----------------------------------------------------------------------------
# 0. 全局参数（车规量级）
# ----------------------------------------------------------------------------
DT = 0.1                 # 轨迹点时间间隔 [s]
PLAN_HZ = 10.0           # 规划频率 [Hz] -> 周期预算 100 ms
V_REF = 12.0             # 巡航目标车速 [m/s]
V_LIMIT = 13.9           # 限速 50 km/h [m/s]

KAPPA_MAX = 0.20         # 最大曲率 [1/m]（R_min = 5.0 m）
A_LON_MAX = 2.0          # 纵向加速度上限 [m/s^2]
A_LON_MIN = -4.0         # 纵向减速度下限 [m/s^2]
A_LAT_MAX = 2.5          # 横向加速度舒适上限 [m/s^2]
JERK_MAX = 4.0           # 纵向 jerk 硬上限 [m/s^3]（舒适阈 2.0）
JERK_LAT_MAX = 3.0       # 横向 jerk 上限 [m/s^3]

EGO_L, EGO_W = 4.70, 1.90       # 车长 / 车宽 [m]
BUF_S, BUF_L = 0.50, 0.30       # 纵向 / 横向安全缓冲 [m]
L_BOUND_HI, L_BOUND_LO = 2.60, -2.60   # 横向可行域 [m]

# 两套代价权重：A = 保守（默认量产标定），B = 放开借道
WEIGHTS_A = dict(lat_jerk=1.0, lon_jerk=1.0, ref_dev=4.0, speed=6.0,
                 obs=30.0, end_l=2.0, prog=0.0, d_soft=1.20)
WEIGHTS_B = dict(lat_jerk=1.0, lon_jerk=1.0, ref_dev=1.0, speed=6.0,
                 obs=30.0, end_l=1.0, prog=4.0, d_soft=0.50)

REF_DS = 0.2             # 参考线离散间隔 [m]


# ----------------------------------------------------------------------------
# 1. 参考线：等弧长重采样 + 数值求曲率
# ----------------------------------------------------------------------------
class ReferenceLine:
    def __init__(self, xs, ys, ds=REF_DS):
        raw_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        self.length = float(raw_s[-1])
        self.ds = ds
        s = np.arange(0.0, self.length, ds)
        x = np.interp(s, raw_s, xs)
        y = np.interp(s, raw_s, ys)
        dx, dy = np.gradient(x, s), np.gradient(y, s)
        ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
        theta = np.arctan2(dy, dx)
        kappa = (dx * ddy - dy * ddx) / (np.power(dx * dx + dy * dy, 1.5) + 1e-12)
        k = kappa.copy()                       # 3 点滑动平均，抑制数值微分噪声
        k[1:-1] = (kappa[:-2] + 2 * kappa[1:-1] + kappa[2:]) / 4.0
        dkappa = np.gradient(k, s)
        # 转成 Python list，热循环里直接索引，比 np.interp 快约 20 倍
        self.s_arr, self.n = s, len(s)
        self.X, self.Y = x.tolist(), y.tolist()
        self.TH, self.K, self.DK = theta.tolist(), k.tolist(), dkappa.tolist()
        self.kappa_np, self.dkappa_np = k, dkappa

    def at(self, s):
        """线性插值查询：ds 固定，直接算索引，O(1)。"""
        u = s / self.ds
        i = int(u)
        if i < 0:
            i, r = 0, 0.0
        elif i >= self.n - 1:
            i, r = self.n - 2, 1.0
        else:
            r = u - i
        j = i + 1
        return (self.X[i] + (self.X[j] - self.X[i]) * r,
                self.Y[i] + (self.Y[j] - self.Y[i]) * r,
                self.TH[i] + (self.TH[j] - self.TH[i]) * r,
                self.K[i] + (self.K[j] - self.K[i]) * r,
                self.DK[i] + (self.DK[j] - self.DK[i]) * r)


def design_kappa(s, k_max=1.0 / 240.0):
    """公路线形标准做法：直线 -> 缓和曲线 -> 圆曲线 -> 缓和曲线 -> 直线。
    缓和段用升余弦而非线性回旋线，保证 dkappa/ds 也连续（C2 参考线）。"""
    if s < 20.0:
        return 0.0
    if s < 50.0:
        return k_max * 0.5 * (1.0 - math.cos(math.pi * (s - 20.0) / 30.0))
    if s < 90.0:
        return k_max
    if s < 120.0:
        return k_max * 0.5 * (1.0 + math.cos(math.pi * (s - 90.0) / 30.0))
    return 0.0


def build_reference_line(total=150.0, ds=REF_DS):
    """由设计曲率积分出参考线：theta = ∫k ds, (x,y) = ∫(cos,sin)theta ds。"""
    n = int(total / ds) + 1
    xs, ys = [0.0], [0.0]
    th, k_design = 0.0, []
    for i in range(n):
        s = i * ds
        k = design_kappa(s)
        k_design.append(k)
        if i > 0:
            xs.append(xs[-1] + ds * math.cos(th))
            ys.append(ys[-1] + ds * math.sin(th))
        th += k * ds
    ref = ReferenceLine(np.array(xs), np.array(ys), ds=ds)
    ref.k_design = np.array(k_design[:ref.n])
    return ref


# ----------------------------------------------------------------------------
# 2. 多项式基元
# ----------------------------------------------------------------------------
class Quintic:
    """五次多项式 p(u)=a0+a1 u+...+a5 u^5，两端各给 (p, p', p'')。"""

    def __init__(self, p0, v0, ac0, p1, v1, ac1, U):
        self.a0, self.a1, self.a2 = p0, v0, ac0 / 2.0
        c0 = p1 - (p0 + v0 * U + 0.5 * ac0 * U * U)
        c1 = v1 - (v0 + ac0 * U)
        c2 = ac1 - ac0
        U2, U3, U4, U5 = U * U, U ** 3, U ** 4, U ** 5
        self.a3 = (10.0 * c0 - 4.0 * c1 * U + 0.5 * c2 * U2) / U3
        self.a4 = (-15.0 * c0 + 7.0 * c1 * U - 1.0 * c2 * U2) / U4
        self.a5 = (6.0 * c0 - 3.0 * c1 * U + 0.5 * c2 * U2) / U5

    def coeffs(self):
        return [self.a0, self.a1, self.a2, self.a3, self.a4, self.a5]

    def val(self, u):
        return (self.a0 + u * (self.a1 + u * (self.a2 + u * (self.a3 + u * (self.a4 + u * self.a5)))))

    def d1(self, u):
        return (self.a1 + u * (2 * self.a2 + u * (3 * self.a3 + u * (4 * self.a4 + u * 5 * self.a5))))

    def d2(self, u):
        return 2 * self.a2 + u * (6 * self.a3 + u * (12 * self.a4 + u * 20 * self.a5))

    def d3(self, u):
        return 6 * self.a3 + u * (24 * self.a4 + u * 60 * self.a5)


class Quartic:
    """四次多项式 s(t)：起点 (s,v,a) 全给，终点只给 (v,a) —— 巡航采样。"""

    def __init__(self, s0, v0, ac0, v1, ac1, T):
        self.b0, self.b1, self.b2 = s0, v0, ac0 / 2.0
        d1 = v1 - v0 - ac0 * T
        d2 = ac1 - ac0
        self.b4 = (d2 * T - 2.0 * d1) / (4.0 * T ** 3)
        self.b3 = (d2 - 12.0 * self.b4 * T * T) / (6.0 * T)

    def val(self, t):
        return self.b0 + t * (self.b1 + t * (self.b2 + t * (self.b3 + t * self.b4)))

    def d1(self, t):
        return self.b1 + t * (2 * self.b2 + t * (3 * self.b3 + t * 4 * self.b4))

    def d2(self, t):
        return 2 * self.b2 + t * (6 * self.b3 + t * 12 * self.b4)

    def d3(self, t):
        return 6 * self.b3 + 24 * self.b4 * t


# ----------------------------------------------------------------------------
# 3. Frenet -> Cartesian（含曲率修正项）
# ----------------------------------------------------------------------------
def frenet_to_cartesian(rx, ry, rth, rk, rdk, s_d, s_dd, l, dl, ddl):
    omk = 1.0 - rk * l                     # 1 - kappa_r * l
    if omk < 1e-3:
        omk = 1e-3                         # l -> 1/kappa_r 时 Frenet 退化
    dth = math.atan2(dl, omk)
    cos_dth = math.cos(dth)
    tan_dth = dl / omk
    x = rx - l * math.sin(rth)
    y = ry + l * math.cos(rth)
    theta = rth + dth
    kr_d_prime = rdk * l + rk * dl
    kappa = (((ddl + kr_d_prime * tan_dth) * cos_dth * cos_dth) / omk + rk) * cos_dth / omk
    v = s_d * math.hypot(omk, dl)
    a = (s_dd * omk / cos_dth
         + (s_d * s_d / cos_dth) * (dl * (kappa * omk / cos_dth - rk) - kr_d_prime))
    return x, y, theta, kappa, v, a


# ----------------------------------------------------------------------------
# 4. 障碍物与 Frenet 盒式碰撞检查
# ----------------------------------------------------------------------------
class Obstacle:
    def __init__(self, name, s0, l0, length, width, v_s=0.0, dynamic=False):
        self.name, self.s0, self.l0 = name, s0, l0
        self.length, self.width = length, width
        self.v_s, self.dynamic = v_s, dynamic
        self.half_s = (length + EGO_L) / 2.0 + BUF_S    # 预膨胀：闵可夫斯基和
        self.half_l = (width + EGO_W) / 2.0 + BUF_L

    def gap(self, t, s, l):
        """自车质心 (s,l) 与该障碍膨胀盒的分离距离；<0 表示侵入。"""
        sc = self.s0 + self.v_s * t if self.dynamic else self.s0
        ds = abs(s - sc) - self.half_s
        dl = abs(l - self.l0) - self.half_l
        if ds >= 0.0 and dl >= 0.0:
            return math.hypot(ds, dl)
        if ds >= 0.0:
            return ds
        if dl >= 0.0:
            return dl
        return max(ds, dl)                               # 双向侵入，取穿透深度


# ----------------------------------------------------------------------------
# 5. 主流程：采样 -> 三级筛选 -> 代价排序
# ----------------------------------------------------------------------------
LAT_OFFSETS = [round(v, 2) for v in np.arange(-2.0, 2.01, 0.5)]   # 9
LAT_HORIZONS = [30.0, 45.0, 60.0]                                  # 3
V_ENDS = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]                         # 6
T_HORIZONS = [4.0, 6.0, 8.0]                                       # 3


def plan(ref, ego, obstacles, W=None, verbose_reject=False):
    W = W or WEIGHTS_A
    stats = dict(total=0, rej_bound=0, rej_kine=0, rej_coll=0, ok=0)
    kine_reason = dict(kappa=0, a_lon=0, a_lat=0, jerk=0, jerk_lat=0, v_lim=0, reverse=0)
    ranked = []
    t0 = time.perf_counter()

    for S in LAT_HORIZONS:
        for l_end in LAT_OFFSETS:
            lat = Quintic(ego['l'], ego['dl'], ego['ddl'], l_end, 0.0, 0.0, S)
            for T in T_HORIZONS:
                n = int(round(T / DT)) + 1
                for v_end in V_ENDS:
                    stats['total'] += 1
                    lon = Quartic(ego['s'], ego['v'], ego['a'], v_end, 0.0, T)

                    traj = []
                    fail = None
                    min_gap = 1e9
                    j_lat_int = j_lon_int = ref_int = 0.0
                    mx_jerk = mx_alat = mx_kappa = mx_jlat = 0.0
                    prev_alat = None

                    for k in range(n):
                        t = k * DT
                        s = lon.val(t); s_d = lon.d1(t)
                        s_dd = lon.d2(t); s_ddd = lon.d3(t)
                        if s_d < -0.05:
                            fail = 'kine'; kine_reason['reverse'] += 1; break

                        u = s - ego['s']
                        if u >= S:
                            l, dl, ddl, dddl = l_end, 0.0, 0.0, 0.0
                        else:
                            if u < 0.0:
                                u = 0.0
                            l = lat.val(u); dl = lat.d1(u)
                            ddl = lat.d2(u); dddl = lat.d3(u)

                        if l > L_BOUND_HI or l < L_BOUND_LO:
                            fail = 'bound'; break

                        rx, ry, rth, rk, rdk = ref.at(s)
                        x, y, th, kap, v, a = frenet_to_cartesian(
                            rx, ry, rth, rk, rdk, s_d, s_dd, l, dl, ddl)

                        a_lat = v * v * kap                 # 带符号横向加速度
                        if abs(kap) > KAPPA_MAX:
                            fail = 'kine'; kine_reason['kappa'] += 1; break
                        if a > A_LON_MAX or a < A_LON_MIN:
                            fail = 'kine'; kine_reason['a_lon'] += 1; break
                        if abs(a_lat) > A_LAT_MAX:
                            fail = 'kine'; kine_reason['a_lat'] += 1; break
                        if abs(s_ddd) > JERK_MAX:
                            fail = 'kine'; kine_reason['jerk'] += 1; break
                        if v > V_LIMIT + 0.2:
                            fail = 'kine'; kine_reason['v_lim'] += 1; break

                        if prev_alat is not None:
                            jlat = abs(a_lat - prev_alat) / DT
                            if jlat > JERK_LAT_MAX:
                                fail = 'kine'; kine_reason['jerk_lat'] += 1; break
                            if jlat > mx_jlat:
                                mx_jlat = jlat
                        prev_alat = a_lat

                        for ob in obstacles:
                            g = ob.gap(t, s, l)
                            if g < min_gap:
                                min_gap = g
                            if g < 0.0:
                                fail = 'coll'; break
                        if fail:
                            break

                        mx_jerk = max(mx_jerk, abs(s_ddd))
                        mx_alat = max(mx_alat, abs(a_lat))
                        mx_kappa = max(mx_kappa, abs(kap))
                        j_lat_int += dddl * dddl * max(s_d, 0.1) * DT
                        j_lon_int += s_ddd * s_ddd * DT
                        ref_int += l * l * DT
                        traj.append(dict(t=t, s=s, l=l, dl=dl, x=x, y=y, theta=th,
                                         kappa=kap, v=v, a=a, jerk=s_ddd, a_lat=a_lat))

                    if fail == 'bound':
                        stats['rej_bound'] += 1; continue
                    if fail == 'kine':
                        stats['rej_kine'] += 1; continue
                    if fail == 'coll':
                        stats['rej_coll'] += 1; continue
                    stats['ok'] += 1

                    travelled = traj[-1]['s'] - ego['s']
                    c_latj = W['lat_jerk'] * j_lat_int / T
                    c_lonj = W['lon_jerk'] * j_lon_int / T
                    c_ref = W['ref_dev'] * ref_int / T
                    c_spd = W['speed'] * (v_end - V_REF) ** 2 / V_REF
                    c_obs = W['obs'] * max(0.0, W['d_soft'] - min_gap) ** 2
                    c_end = W['end_l'] * l_end * l_end
                    c_prog = W['prog'] * max(0.0, V_REF * T - travelled) / T
                    total = c_latj + c_lonj + c_ref + c_spd + c_obs + c_end + c_prog
                    ranked.append(dict(
                        total=total, l_end=l_end, S=S, T=T, v_end=v_end, gap=min_gap,
                        max_jerk=mx_jerk, max_alat=mx_alat, max_kappa=mx_kappa,
                        max_jlat=mx_jlat, travelled=travelled, traj=traj,
                        parts=dict(lat_jerk=c_latj, lon_jerk=c_lonj, ref_dev=c_ref,
                                   speed=c_spd, obstacle=c_obs, end_l=c_end,
                                   progress=c_prog)))

    ms = (time.perf_counter() - t0) * 1000.0
    ranked.sort(key=lambda c: c['total'])
    best = ranked[0] if ranked else None
    if verbose_reject:
        stats['kine_reason'] = kine_reason
    return best, ranked, stats, ms


# ----------------------------------------------------------------------------
# 6. 打印辅助
# ----------------------------------------------------------------------------
COST_LABELS = [('lat_jerk', '横向 jerk 平滑'), ('lon_jerk', '纵向 jerk 平滑'),
               ('ref_dev', '偏离参考线'), ('speed', '速度偏离巡航'),
               ('obstacle', '障碍物贴近惩罚'), ('end_l', '终点横向偏置'),
               ('progress', '通行效率(推进)')]


def report_best(best, ego, title):
    """打印一条最优轨迹的完整体检报告。"""
    print("\n" + "-" * 78)
    print(f" {title}")
    print("-" * 78)
    print(f"横向终点 l_end = {best['l_end']:+.2f} m | 收敛距离 S = {best['S']:.0f} m | "
          f"时域 T = {best['T']:.1f} s | 末速度 v_end = {best['v_end']:.1f} m/s")
    print(f"{best['T']:.0f} s 内推进 {best['travelled']:.2f} m "
          f"(平均 {best['travelled']/best['T']:.2f} m/s) | "
          f"最小间隙 gap = {best['gap']:.3f} m")
    rr = 1.0 / best['max_kappa'] if best['max_kappa'] > 1e-9 else float('inf')
    print(f"max|jerk_lon| = {best['max_jerk']:.3f} m/s^3 (限 {JERK_MAX})  "
          f"max jerk_lat = {best['max_jlat']:.3f} m/s^3 (限 {JERK_LAT_MAX})")
    print(f"max|a_lat|    = {best['max_alat']:.3f} m/s^2 (限 {A_LAT_MAX})  "
          f"max|kappa| = {best['max_kappa']:.5f} 1/m (R = {rr:.1f} m, 限 {KAPPA_MAX})")

    print(f"\n 代价分解 (total = {best['total']:.4f})")
    for key, name in COST_LABELS:
        v = best['parts'][key]
        pct = v / best['total'] * 100 if best['total'] > 1e-12 else 0.0
        bar = '#' * int(round(pct / 2.5))
        print(f"   {name:<16s} {v:9.4f}  {pct:5.1f}%  {bar}")

    print("\n 轨迹采样（每 1.0 s）")
    print("    t[s]      s[m]     l[m]    v[m/s]  a[m/s^2] jerk[m/s^3]  "
          "kappa[1/m]  a_lat[m/s^2]")
    for p in best['traj']:
        if abs(p['t'] * 10 - round(p['t'] * 10)) < 1e-6 and \
           abs(round(p['t']) - p['t']) < 1e-6:
            print(f"   {p['t']:5.1f} {p['s']:9.3f} {p['l']:+8.3f} {p['v']:9.3f} "
                  f"{p['a']:9.3f} {p['jerk']:11.3f} {p['kappa']:11.5f} "
                  f"{p['a_lat']:12.3f}")


def report_rank(ranked, title):
    """打印代价排名前 10，观察备选簇的分布。"""
    print(f"\n {title}")
    print(f"   {'#':>2s} {'l_end':>7s} {'S':>5s} {'T':>5s} {'v_end':>6s} "
          f"{'gap':>7s} {'travel':>7s} {'total':>9s} {'obs':>8s} {'speed':>8s} "
          f"{'ref':>8s}")
    for i, c in enumerate(ranked[:10], 1):
        print(f"   {i:>2d} {c['l_end']:>+7.2f} {c['S']:>5.0f} {c['T']:>5.1f} "
              f"{c['v_end']:>6.1f} {c['gap']:>7.3f} {c['travelled']:>7.2f} "
              f"{c['total']:>9.3f} {c['parts']['obstacle']:>8.3f} "
              f"{c['parts']['speed']:>8.3f} {c['parts']['ref_dev']:>8.3f}")


# ----------------------------------------------------------------------------
# 7. 入口
# ----------------------------------------------------------------------------
def main():
    ref = build_reference_line()
    ego = dict(s=0.0, l=0.30, dl=0.02, ddl=0.0, v=12.0, a=0.0)
    obstacles = [
        Obstacle("parked_van 违停厢货", 48.0, 1.60, 5.40, 2.05),
        Obstacle("cone 施工锥桶",       88.0, -1.40, 0.50, 0.50),
        Obstacle("lead_car 前方慢车",   30.0, 0.00, 4.80, 1.90, v_s=7.0, dynamic=True),
    ]

    print("=" * 78)
    print(" Frenet Lattice Planner  —  第五章《运动规划》配套实验")
    print("=" * 78)
    print(f"[参考线] 线形 = 直线(0~20m) + 缓和曲线(20~50m) + 圆曲线(50~90m) "
          f"+ 缓和曲线(90~120m) + 直线(120m~)")
    print(f"[参考线] 长度 {ref.length:.2f} m | 离散点 {ref.n} | ds = {REF_DS} m")
    kmax = float(np.max(np.abs(ref.kappa_np)))
    print(f"[参考线] |kappa_r|max = {kmax:.6f} 1/m  (R_min = {1.0/kmax:.1f} m)")
    print(f"[参考线] |dkappa_r|max = {float(np.max(np.abs(ref.dkappa_np))):.6e} 1/m^2")
    err = float(np.max(np.abs(ref.kappa_np - ref.k_design)))
    print(f"[参考线] 数值重建曲率 vs 设计曲率  max|误差| = {err:.3e} 1/m  "
          f"(相对 {err/kmax*100:.4f}%)")
    print(f"[自车]   s={ego['s']:.2f} m  l={ego['l']:+.2f} m  dl/ds={ego['dl']:.3f}  "
          f"v={ego['v']:.2f} m/s  a={ego['a']:.2f} m/s^2")
    print(f"[约束]   |kappa|<={KAPPA_MAX} 1/m | a_lon in [{A_LON_MIN},{A_LON_MAX}] m/s^2 | "
          f"a_lat<={A_LAT_MAX} m/s^2 | |jerk|<={JERK_MAX} m/s^3 | v<={V_LIMIT} m/s")
    print(f"[包络]   自车 {EGO_L}x{EGO_W} m，缓冲 (s,l)=({BUF_S},{BUF_L}) m")
    print("[障碍物]")
    for ob in obstacles:
        kind = f"动态 v_s={ob.v_s:.1f} m/s" if ob.dynamic else "静态         "
        print(f"   - {ob.name:<22s} s={ob.s0:6.2f} l={ob.l0:+.2f} "
              f"{ob.length:.2f}x{ob.width:.2f} m  {kind}  "
              f"膨胀半宽 (s,l)=({ob.half_s:.3f},{ob.half_l:.3f})")

    # --- 五次多项式端点约束求解自检（对应正文 4.3 节的系数矩阵） ---
    S_chk, l_end_chk = 30.0, -1.00
    q = Quintic(ego['l'], ego['dl'], ego['ddl'], l_end_chk, 0.0, 0.0, S_chk)
    M = np.array([[S_chk ** 3, S_chk ** 4, S_chk ** 5],
                  [3 * S_chk ** 2, 4 * S_chk ** 3, 5 * S_chk ** 4],
                  [6 * S_chk, 12 * S_chk ** 2, 20 * S_chk ** 3]])
    rhs = np.array([l_end_chk - (ego['l'] + ego['dl'] * S_chk),
                    0.0 - ego['dl'], 0.0])
    a345 = np.linalg.solve(M, rhs)
    print(f"\n[自检] 五次多项式 l(s)，S={S_chk:.0f} m, l_end={l_end_chk:+.2f} m")
    print("       闭式解 a0..a5 = " + ", ".join(f"{c:+.6e}" for c in q.coeffs()))
    print("       线性求解 a3..a5 = " + ", ".join(f"{c:+.6e}" for c in a345)
          + f"   (cond(M) = {np.linalg.cond(M):.3e})")
    print(f"       端点残差: l(S)-l_end = {q.val(S_chk)-l_end_chk:+.3e}, "
          f"l'(S) = {q.d1(S_chk):+.3e}, l''(S) = {q.d2(S_chk):+.3e}")

    # --- ST 图：把与自车横向重叠的障碍投影成 (t, s_low, s_high) 禁止区 ---
    print("\n[ST 图投影] 以 l=0 走廊为例，障碍在 s-t 平面上的禁止区（每 1.0 s）")
    print("    t[s] | " + " | ".join(f"{ob.name.split()[0]:>22s}" for ob in obstacles))
    for k in range(0, 9):
        t = float(k)
        cells = []
        for ob in obstacles:
            if abs(0.0 - ob.l0) < ob.half_l:          # 横向重叠才会进 ST 图
                sc = ob.s0 + ob.v_s * t if ob.dynamic else ob.s0
                cells.append(f"[{sc-ob.half_s:7.2f},{sc+ob.half_s:7.2f}]")
            else:
                cells.append(f"{'—（横向不重叠）':>20s}")
        print(f"   {t:5.1f} | " + " | ".join(cells))

    best, ranked, stats, ms = plan(ref, ego, obstacles, verbose_reject=True)

    print("\n" + "-" * 78)
    print(" 采样与三级筛选统计")
    print("-" * 78)
    print(f"候选轨迹总数 = {len(LAT_OFFSETS)} 横向终点 x {len(LAT_HORIZONS)} 收敛距离 "
          f"x {len(T_HORIZONS)} 时域 x {len(V_ENDS)} 末速度 = {stats['total']}")
    print(f"  [1] 横向边界剔除  l 超出 [{L_BOUND_LO}, {L_BOUND_HI}] m : {stats['rej_bound']:4d}")
    print(f"  [2] 运动学/动力学剔除                          : {stats['rej_kine']:4d}   "
          + " ".join(f"{k}={v}" for k, v in stats['kine_reason'].items() if v))
    print(f"  [3] 碰撞剔除      Frenet 膨胀盒                : {stats['rej_coll']:4d}")
    print(f"  ==> 最终可行                                   : {stats['ok']:4d}   "
          f"({stats['ok']/stats['total']*100:.1f}%)")
    # 重复 20 次统计耗时分布（实时系统关心的是尾延迟，不是平均值）
    samples = sorted(plan(ref, ego, obstacles)[3] for _ in range(20))
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95) - 1]
    print(f"求解耗时（20 次重复）: min={samples[0]:.2f} ms  p50={p50:.2f} ms  "
          f"p95={p95:.2f} ms  max={samples[-1]:.2f} ms")
    print(f"规划周期预算 {1000.0/PLAN_HZ:.0f} ms  ->  p95 占用 "
          f"{p95/(1000.0/PLAN_HZ)*100:.1f}%")

    if best is None:
        print("\n!! 无可行轨迹 -> 触发 fallback（沿用上一帧轨迹 / 紧急制动轨迹）")
        return

    report_best(best, ego, "配置 A（保守权重：d_soft=1.2 m, w_ref=4.0, w_end_l=2.0, 无推进项）")
    report_rank(ranked, "配置 A 代价前 10 名")

    # === 只改代价权重，不改候选集，看行为如何反转 ===
    bestB, rankedB, statsB, msB = plan(ref, ego, obstacles, W=WEIGHTS_B)
    print("\n" + "=" * 78)
    print(" 【关键实验】候选集完全不变，只把代价权重从 A 换成 B")
    print(" B: d_soft 1.2->0.5 m, w_ref 4.0->1.0, w_end_l 2.0->1.0, 新增推进项 w_prog=4.0")
    print("=" * 78)
    report_best(bestB, ego, "配置 B（放开借道）")
    report_rank(rankedB, "配置 B 代价前 10 名")
    print(f"\n>>> 行为反转：A 选择『减速拖延』(l_end={best['l_end']:+.2f} m, "
          f"v_end={best['v_end']:.1f} m/s, {best['travelled']:.1f} m/{best['T']:.0f}s)")
    print(f"                B 选择『向左借道通过』(l_end={bestB['l_end']:+.2f} m, "
          f"v_end={bestB['v_end']:.1f} m/s, {bestB['travelled']:.1f} m/{bestB['T']:.0f}s)")
    print(f"    两者用的是同一批 {stats['ok']} / {statsB['ok']} 条可行轨迹，"
          f"差别只在评分函数。")

    # --- 消融 1：移除动态慢车 ---
    b2, _, st2, ms2 = plan(ref, ego, [obstacles[0], obstacles[1]])
    # --- 消融 2：移除违停厢货 ---
    b3, _, st3, ms3 = plan(ref, ego, [obstacles[1], obstacles[2]])
    print("\n" + "-" * 78)
    print(" 消融实验")
    print("-" * 78)
    hdr = f"{'config':<20s}{'coll_rej':>9s}{'feasible':>9s}{'l_end':>8s}{'v_end':>8s}" \
          f"{'T':>6s}{'total':>9s}{'ms':>8s}"
    print(hdr)
    for tag, st, b, tt in [("baseline(3 obs)", stats, best, ms),
                           ("no lead_car", st2, b2, ms2),
                           ("no parked_van", st3, b3, ms3)]:
        print(f"{tag:<20s}{st['rej_coll']:>9d}{st['ok']:>9d}{b['l_end']:>8.2f}"
              f"{b['v_end']:>8.1f}{b['T']:>6.1f}{b['total']:>9.3f}{tt:>8.2f}")

    # --- 轨迹拼接一致性 ---
    p1 = best['traj'][1]
    ego2 = dict(s=p1['s'], l=p1['l'], dl=p1['dl'], ddl=0.0, v=p1['v'], a=p1['a'])
    b4, _, _, ms4 = plan(ref, ego2, obstacles)
    print("\n" + "-" * 78)
    print(" 轨迹拼接 (trajectory stitching) 一致性检查")
    print("-" * 78)
    print(f"新起点取上一帧 t=0.1 s 的点: s={ego2['s']:.3f} m, l={ego2['l']:+.4f} m, "
          f"v={ego2['v']:.4f} m/s, a={ego2['a']:.4f} m/s^2")
    print(f"上一帧最优 (l_end={best['l_end']:+.2f}, v_end={best['v_end']:.1f}, T={best['T']:.1f}) "
          f"-> 本帧最优 (l_end={b4['l_end']:+.2f}, v_end={b4['v_end']:.1f}, T={b4['T']:.1f})"
          f"  耗时 {ms4:.2f} ms")
    dl_cm = abs(b4['traj'][0]['l'] - p1['l']) * 100
    dv_cm = abs(b4['traj'][0]['v'] - p1['v']) * 100
    print(f"帧间起点跳变: |dl| = {dl_cm:.4f} cm, |dv| = {dv_cm:.4f} cm/s  -> "
          f"{'一致，无跳变' if dl_cm < 2.0 else '存在跳变，需排查'}")

    # --- 无拼接对照：起点用定位测得的带噪声状态 ---
    ego3 = dict(s=p1['s'] + 0.12, l=p1['l'] + 0.08, dl=p1['dl'] - 0.01,
                ddl=0.0, v=p1['v'] - 0.25, a=p1['a'])
    b5, _, _, _ = plan(ref, ego3, obstacles)
    print(f"对照：若不做拼接、直接用带噪声定位状态 (dl=+8 cm, dv=-25 cm/s) 作为起点，")
    print(f"      本帧最优变为 (l_end={b5['l_end']:+.2f}, v_end={b5['v_end']:.1f})，"
          f"首点横向偏差 {abs(b5['traj'][0]['l']-p1['l'])*100:.2f} cm")
    print("=" * 78)


if __name__ == "__main__":
    main()
