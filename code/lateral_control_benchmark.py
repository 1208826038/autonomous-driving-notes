"""
ch06 横向控制器对比仿真  (autonomous-driving-notes / 第六章配套代码)
================================================================
被控对象 : 二自由度动力学自行车模型（线性轮胎 + 附着极限硬饱和 mu=0.9）
执行链路 : 150 ms 纯延时 + EPS 一阶滞后(50 ms) + 前轮转角速率限制(560 deg/s 方向盘)
纵向     : PID + 加速度前馈，动力/制动一阶滞后 250 ms + 100 ms 延时
参考轨迹 : 直线 -> 3.5 m 变道(72 km/h) -> 减速直线 -> 曲率 0.05 1/m 定曲率弯(36 km/h) -> 加速直线
对比控制器:
  ① Pure Pursuit  变预瞄 ld = 0.3v + 3
  ② Pure Pursuit  短预瞄 ld = 4 m       (复现"高速变道方向盘抖动")
  ③ Stanley       k = 2.0
  ④ LQR + 曲率前馈  (增益调度)
  ⑤ LQR + 曲率前馈 + 延时补偿
运行: python lateral_control_benchmark.py
"""
import numpy as np
from scipy.linalg import solve_discrete_are

# ---------------- 车辆 / 执行器参数 ----------------
M, IZ = 1600.0, 2500.0          # 质量 kg，横摆转动惯量 kg*m^2
LF, LR = 1.20, 1.60             # 质心到前/后轴 m
L = LF + LR                     # 轴距 2.8 m
CF, CR = 110000.0, 130000.0     # 前/后轴总侧偏刚度 N/rad
G, MU = 9.81, 0.9
FZF, FZR = M * G * LR / L, M * G * LF / L
STEER_RATIO = 16.0
DELTA_MAX = np.deg2rad(35.0)            # 前轮转角上限
RATE_MAX = np.deg2rad(35.0)             # 前轮角速度上限 -> 方向盘 560 deg/s
TAU_EPS, T_DELAY = 0.05, 0.15           # EPS 时间常数 / 转向纯延时
TAU_PWT, T_DELAY_LON = 0.25, 0.10       # 动力总成时间常数 / 纵向纯延时
AX_MIN, AX_MAX = -4.0, 2.0              # 纵向加速度限
DT, DS = 0.01, 0.05                     # 控制周期 100 Hz / 路径离散步长
SAT_HIT = [False]

# 不足转向梯度 / 稳定性因数 / 特征车速
K_US = FZF / CF - FZR / CR              # rad per g
K_S = K_US / G                          # s^2/m
V_CHAR = np.sqrt(L * G / K_US)


# ---------------- 参考轨迹：按曲率剖面积分生成 ----------------
def build_path():
    segs = [(0, 20, 'zero'), (20, 80, 'sine'), (80, 140, 'zero'),
            (140, 145, 'ramp_up'), (145, 175, 'const'), (175, 180, 'ramp_dn'),
            (180, 220, 'zero')]
    s = np.arange(0.0, 220.0, DS)
    kap = np.zeros_like(s)
    amp = 0.0061087                                  # 标定为横向位移 3.50 m
    for a, b, kind in segs:
        m = (s >= a) & (s < b)
        u = s[m] - a
        if kind == 'sine':
            kap[m] = amp * np.sin(2 * np.pi * u / (b - a))
        elif kind == 'const':
            kap[m] = 0.05
        elif kind == 'ramp_up':
            kap[m] = 0.05 * u / (b - a)
        elif kind == 'ramp_dn':
            kap[m] = 0.05 * (1 - u / (b - a))
    psi = np.cumsum(kap) * DS
    return s, np.cumsum(np.cos(psi)) * DS, np.cumsum(np.sin(psi)) * DS, psi, kap


def build_speed(s):
    """速度剖面：20 m/s 巡航 -> 2.5 m/s^2 减速到 10 m/s 过弯 -> 加速回 18 m/s"""
    v = np.full_like(s, 20.0)
    m = (s >= 80) & (s < 140)
    v[m] = np.sqrt(400 - 2 * 2.5 * (s[m] - 80))
    v[(s >= 140) & (s < 180)] = 10.0
    m = s >= 180
    v[m] = np.minimum(18.0, np.sqrt(100 + 2 * 1.2 * (s[m] - 180)))
    return v


S, PX, PY, PPSI, PK = build_path()
PV = build_speed(S)
N_PATH = len(S)


def nearest(x, y, i0=0):
    lo, hi = max(0, i0 - 40), min(N_PATH, i0 + 800)
    return lo + int(np.argmin((PX[lo:hi] - x) ** 2 + (PY[lo:hi] - y) ** 2))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lat_err(x, y, i):
    return -np.sin(PPSI[i]) * (x - PX[i]) + np.cos(PPSI[i]) * (y - PY[i])


# ---------------- 轮胎与被控对象 ----------------
def tire_fy(alpha, c, fz):
    """线性 C*alpha，但受附着极限 mu*Fz 硬饱和"""
    f, fmax = c * alpha, MU * fz
    if abs(f) > fmax:
        SAT_HIT[0] = True
        return np.sign(f) * fmax
    return f


def plant_deriv(st, delta, ax):
    """st = [X, Y, psi, vx, vy, r]"""
    _, _, psi, vx, vy, r = st
    vxs = max(vx, 1.0)
    af = delta - (vy + LF * r) / vxs
    ar = -(vy - LR * r) / vxs
    fyf, fyr = tire_fy(af, CF, FZF), tire_fy(ar, CR, FZR)
    return np.array([vx * np.cos(psi) - vy * np.sin(psi),
                     vx * np.sin(psi) + vy * np.cos(psi),
                     r,
                     ax + vy * r,
                     (fyf + fyr) / M - vx * r,
                     (LF * fyf - LR * fyr) / IZ])


def rk4(st, delta, ax, dt):
    k1 = plant_deriv(st, delta, ax)
    k2 = plant_deriv(st + dt / 2 * k1, delta, ax)
    k3 = plant_deriv(st + dt / 2 * k2, delta, ax)
    k4 = plant_deriv(st + dt * k3, delta, ax)
    return st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------- LQR：误差动力学 + 离散黎卡提 + 增益调度 ----------------
Q_LQR = np.diag([3.0, 0.5, 12.0, 0.3])
R_LQR = np.array([[250.0]])


def error_model(vx):
    A = np.array([
        [0, 1, 0, 0],
        [0, -(CF + CR) / (M * vx), (CF + CR) / M,
         (-CF * LF + CR * LR) / (M * vx)],
        [0, 0, 0, 1],
        [0, -(CF * LF - CR * LR) / (IZ * vx), (CF * LF - CR * LR) / IZ,
         -(CF * LF ** 2 + CR * LR ** 2) / (IZ * vx)]])
    B = np.array([[0.0], [CF / M], [0.0], [CF * LF / IZ]])
    return A, B


def lqr_gain(vx, Q=None, R=None):
    Q = Q_LQR if Q is None else Q
    R = R_LQR if R is None else R
    A, B = error_model(vx)
    Ad = np.eye(4) + A * DT + 0.5 * (A @ A) * DT ** 2        # 二阶离散化
    Bd = (np.eye(4) * DT + 0.5 * A * DT ** 2) @ B
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad).flatten()


V_TBL = np.arange(3.0, 36.0, 1.0)                            # 增益调度查表
K_TBL = np.array([lqr_gain(v) for v in V_TBL])


def sched_gain(vx):
    vx = float(np.clip(vx, V_TBL[0], V_TBL[-1]))
    return np.array([np.interp(vx, V_TBL, K_TBL[:, j]) for j in range(4)])


KV = LR * M / (CF * L) - LF * M / (CR * L)                   # == 稳定性因数 K_s


def lqr_ff(kappa, vx, k3):
    """Ackermann + 不足转向补偿 - 稳态航向误差反馈抵消项"""
    return (L * kappa + KV * vx ** 2 * kappa
            - k3 * (LR * kappa - LF * M * vx ** 2 * kappa / (CR * L)))


# ---------------- 横向控制器 ----------------
def ctl_pure_pursuit(st, i, k, l0):
    """几何法，参考点是后轴中心（状态 X,Y 在质心，需后移 lr）"""
    x, y, psi, vx = st[0], st[1], st[2], st[3]
    rx, ry = x - LR * np.cos(psi), y - LR * np.sin(psi)      # 后轴中心
    ld = max(k * vx + l0, 2.0)
    j = min(nearest(rx, ry, max(0, i - 80)) + int(ld / DS), N_PATH - 1)
    alpha = wrap(np.arctan2(PY[j] - ry, PX[j] - rx) - psi)
    return np.arctan2(2 * L * np.sin(alpha), ld)


def ctl_stanley(st, i, k=2.0, ks=1.0):
    """参考点是前轴中心（质心前移 lf）"""
    x, y, psi, vx = st[0], st[1], st[2], st[3]
    fx, fy = x + LF * np.cos(psi), y + LF * np.sin(psi)      # 前轴中心
    j = nearest(fx, fy, i)
    return wrap(PPSI[j] - psi) + np.arctan2(-k * lat_err(fx, fy, j), vx + ks)


def ctl_lqr(st, i, buf=None):
    """buf 非 None 则启用延时补偿：用模型 + 在途指令把状态前推 T_DELAY"""
    if buf is not None:
        for u in buf:
            st = rk4(st, u, 0.0, DT)
        i = nearest(st[0], st[1], i)
    x, y, psi, vx, vy, r = st
    vxs = max(vx, 2.0)
    K = sched_gain(vxs)
    e1 = lat_err(x, y, i)
    e3 = wrap(psi - PPSI[i])
    e2 = vy + vxs * np.sin(e3)
    e4 = r - vxs * PK[i]
    return -float(K @ np.array([e1, e2, e3, e4])) + lqr_ff(PK[i], vxs, K[2])


# ---------------- 仿真主循环 ----------------
def simulate(name, t_delay=T_DELAY):
    ndly = max(0, int(round(t_delay / DT)))
    ndly_lon = int(round(T_DELAY_LON / DT))
    st = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
    buf, buf_a = [0.0] * ndly, [0.0] * ndly_lon
    d_act, d_prev, ax_act, ei = 0.0, None, 0.0, 0.0
    i, t = 0, 0.0
    log = {k: [] for k in ('t', 'e', 's', 'rate', 'ay', 'd', 've', 'v')}
    while i < N_PATH - 900 and t < 40.0:
        i = nearest(st[0], st[1], i)
        # ---- 横向 ----
        if name == 'PP':
            cmd = ctl_pure_pursuit(st, i, 0.3, 3.0)
        elif name == 'PP_SHORT':
            cmd = ctl_pure_pursuit(st, i, 0.0, 4.0)
        elif name == 'STANLEY':
            cmd = ctl_stanley(st, i)
        elif name == 'LQR':
            cmd = ctl_lqr(st, i)
        else:
            cmd = ctl_lqr(st, i, buf=list(buf))
        cmd = float(np.clip(cmd, -DELTA_MAX, DELTA_MAX))
        rate_cmd = 0.0 if d_prev is None else (cmd - d_prev) / DT
        d_prev = cmd
        buf.append(cmd)
        d_tgt = buf.pop(0) if ndly else cmd                   # 纯延时
        d_new = d_act + (d_tgt - d_act) * DT / TAU_EPS        # EPS 一阶滞后
        d_new = np.clip(d_new, d_act - RATE_MAX * DT, d_act + RATE_MAX * DT)
        d_act = float(np.clip(d_new, -DELTA_MAX, DELTA_MAX))
        # ---- 纵向 PID + 前馈 ----
        v_ref = PV[i]
        a_ff = (PV[min(i + 200, N_PATH - 1)] ** 2 - v_ref ** 2) / (2 * 10.0)
        ev = v_ref - st[3]
        ei = float(np.clip(ei + ev * DT, -3.0, 3.0))          # 抗积分饱和
        a_cmd = float(np.clip(a_ff + 1.2 * ev + 0.35 * ei, AX_MIN, AX_MAX))
        buf_a.append(a_cmd)
        ax_act += (buf_a.pop(0) - ax_act) * DT / TAU_PWT
        # ---- 推进 ----
        af = d_act - (st[4] + LF * st[5]) / max(st[3], 1.0)
        ar = -(st[4] - LR * st[5]) / max(st[3], 1.0)
        ay = (tire_fy(af, CF, FZF) + tire_fy(ar, CR, FZR)) / M
        st = rk4(st, d_act, ax_act, DT)
        for k_, v_ in zip(('t', 'e', 's', 'rate', 'ay', 'd', 've', 'v'),
                          (t, lat_err(st[0], st[1], i), S[i], rate_cmd, ay,
                           d_act, ev, st[3])):
            log[k_].append(v_)
        t += DT
    return {k: np.array(v) for k, v in log.items()}


def metrics(lg):
    e, s = lg['e'], lg['s']
    return dict(max_e=np.abs(e).max() * 100,
                lc_e=np.abs(e[(s > 20) & (s < 82)]).max() * 100,
                ss_e=(e[(s > 155) & (s < 173)].mean() * 100
                      if ((s > 155) & (s < 173)).any() else float('nan')),
                rms=np.sqrt((e ** 2).mean()) * 100,
                sw_rate=np.abs(lg['rate']).max() * STEER_RATIO * 180 / np.pi,
                max_sw=np.abs(lg['d']).max() * STEER_RATIO * 180 / np.pi,
                max_ay=np.abs(lg['ay']).max(),
                v_err=np.abs(lg['ve']).max())


CASES = [('PP', 'PurePursuit ld=0.3v+3'), ('PP_SHORT', 'PurePursuit ld=4m固定'),
         ('STANLEY', 'Stanley k=2.0'), ('LQR', 'LQR + 曲率前馈'),
         ('LQRD', 'LQR + 前馈 + 延时补偿')]

if __name__ == '__main__':
    ay_ss, kap = 10.0 ** 2 * 0.05, 0.05
    print('=' * 86)
    print(f'车辆 m={M:.0f}kg Iz={IZ:.0f} L={L:.2f}m lf={LF} lr={LR} '
          f'Cf={CF/1000:.0f}kN/rad Cr={CR/1000:.0f}kN/rad mu={MU}')
    print(f'不足转向梯度 K_us = {K_US:.5f} rad/g = {np.rad2deg(K_US):.2f} deg/g '
          f'-> {"不足转向(understeer)" if K_US > 0 else "过度转向"}')
    print(f'稳定性因数 K_s = {K_S:.5f} s^2/m   特征车速 V_ch = {V_CHAR:.1f} m/s '
          f'= {V_CHAR*3.6:.0f} km/h')
    print(f'执行链 转向纯延时 {T_DELAY*1000:.0f} ms + EPS tau {TAU_EPS*1000:.0f} ms，'
          f'速率上限 {np.rad2deg(RATE_MAX)*STEER_RATIO:.0f} deg/s(方向盘)，'
          f'控制 {1/DT:.0f} Hz')
    print(f'弯道 kappa=0.05 1/m (R=20 m) @ 10 m/s -> ay = {ay_ss:.2f} m/s^2 '
          f'= {ay_ss/G:.2f} g；前轴稳态侧偏角 '
          f'{np.rad2deg(M*ay_ss*LR/L/CF):.2f} deg')
    print(f'  Ackermann {np.rad2deg(L*kap):.2f} deg + 不足转向 '
          f'{np.rad2deg(K_US*ay_ss/G):.2f} deg = 稳态前轮 '
          f'{np.rad2deg(L*kap+K_US*ay_ss/G):.2f} deg '
          f'(方向盘 {np.rad2deg(L*kap+K_US*ay_ss/G)*STEER_RATIO:.1f} deg)')
    print(f'LQR Q=diag(3,0.5,12,0.3)  R=250  Kv={KV:.5f} s^2/m')
    print('=' * 86)
    print(f"{'控制器':<24}{'最大|e|':>9}{'变道段max':>11}{'弯道稳态e':>11}"
          f"{'RMS':>8}{'方向盘峰值转速':>15}{'峰值ay':>10}")
    print('-' * 86)
    res = {}
    for key, label in CASES:
        SAT_HIT[0] = False
        r = metrics(simulate(key))
        r['sat'] = SAT_HIT[0]
        res[label] = r
        print(f'{label:<24}{r["max_e"]:>7.1f}cm{r["lc_e"]:>9.1f}cm'
              f'{r["ss_e"]:>9.1f}cm{r["rms"]:>6.1f}cm'
              f'{r["sw_rate"]:>12.1f}d/s{r["max_ay"]:>8.2f}m/s2')
    print('-' * 86)
    for k, v in res.items():
        print(f'{k:<24} 最大方向盘转角 {v["max_sw"]:6.1f} deg  峰值侧向加速度 '
              f'{v["max_ay"]/G:.2f} g  轮胎附着饱和: '
              f'{"是" if v["sat"] else "否"}  纵向速度最大误差 '
              f'{v["v_err"]:.2f} m/s')
    print('=' * 86)
    print('增益调度表 (同一组 Q/R 在不同车速下解离散黎卡提方程)')
    print(f"{'vx[m/s]':>8}{'km/h':>7}{'k1':>9}{'k2':>9}{'k3':>9}{'k4':>9}"
          f"{'R=200m前馈[deg方向盘]':>24}")
    for v in (5, 10, 15, 20, 25, 30, 35):
        K = lqr_gain(float(v))
        ff = L / 200.0 + KV * v ** 2 / 200.0
        print(f'{v:>8}{v*3.6:>7.0f}{K[0]:>9.4f}{K[1]:>9.4f}{K[2]:>9.4f}'
              f'{K[3]:>9.4f}{np.rad2deg(ff)*STEER_RATIO:>24.2f}')
    print('=' * 86)
    print('延时敏感性：LQR(无延时补偿) 在不同纯延时下的表现')
    print(f"{'纯延时[ms]':>12}{'最大|e|[cm]':>14}{'RMS[cm]':>11}"
          f"{'方向盘峰值转速[deg/s]':>24}{'是否发散':>10}")
    for d_ms in (0, 50, 100, 150, 200, 250, 300):
        r = metrics(simulate('LQR', t_delay=d_ms / 1000.0))
        print(f'{d_ms:>12}{r["max_e"]:>14.2f}{r["rms"]:>11.2f}'
              f'{r["sw_rate"]:>24.1f}{"是" if r["max_e"] > 100 else "否":>10}')
    print('=' * 86)
