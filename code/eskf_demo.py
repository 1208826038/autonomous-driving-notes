"""2D 简化 ESKF 组合导航：IMU 100 Hz 预测 + GNSS 5 Hz 位置更新（含一段失锁）。

误差状态 8 维: [dpx, dpy, dvx, dvy, dtheta, dbax, dbay, dbg]
名义状态: p(2), v(2), theta(1), b_a(2, 体轴), b_g(1)
"""
import numpy as np

np.random.seed(20260804)

DT = 0.01                      # IMU 周期 100 Hz
T_END = 120.0                  # 总时长 s
GNSS_DT = 0.2                  # GNSS 5 Hz
OUTAGE = (40.0, 60.0)          # 人为 GNSS 失锁窗口（隧道）
GNSS_SIGMA = 0.05              # RTK 固定解水平精度 5 cm (1σ)

# 真值零偏（未标定的消费级 MEMS 典型量级）
BA_TRUE = np.array([0.060, -0.040])        # m/s^2  (~6 mg)
BG_TRUE = np.deg2rad(0.020)                # rad/s  (=72 deg/h)
NA_STD, NG_STD = 0.015, np.deg2rad(0.10)   # IMU 白噪声 (100 Hz 采样下)
BA_RW, BG_RW = 1e-4, np.deg2rad(1e-3)      # 零偏随机游走驱动


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


J2 = np.array([[0.0, -1.0], [1.0, 0.0]])   # dR/dtheta = R @ J2


def true_motion(t):
    """真值轨迹：0-30s 直线加速，30-90s 定常圆弧转弯，90s 后直线。"""
    v = 12.0 + 3.0 * np.sin(0.05 * t)              # 纵向车速 m/s
    a_lon = 3.0 * 0.05 * np.cos(0.05 * t)          # 纵向加速度
    omega = np.deg2rad(4.0) if 30.0 <= t < 90.0 else 0.0
    return v, a_lon, omega


# ---------- 1. 生成真值轨迹与带误差的 IMU 数据 ----------
n = int(T_END / DT)
p_t, v_t, th_t = np.zeros(2), np.array([12.0, 0.0]), 0.0
traj, imu = [], []
for k in range(n):
    t = k * DT
    v_lon, a_lon, omega = true_motion(t)
    a_body = np.array([a_lon, omega * v_lon])      # 纵向 + 向心加速度
    traj.append((t, p_t.copy(), v_t.copy(), th_t))
    am = a_body + BA_TRUE + np.random.randn(2) * NA_STD
    gm = omega + BG_TRUE + np.random.randn() * NG_STD
    imu.append((am, gm))
    p_t = p_t + v_t * DT
    v_t = rot(th_t) @ np.array([v_lon, 0.0])
    th_t = th_t + omega * DT

# ---------- 2. ESKF ----------
p = traj[0][1] + np.array([1.0, -1.0])   # 初始位置给 1 m 误差
v = traj[0][2].copy()
th = traj[0][3] + np.deg2rad(1.0)        # 初始航向给 1° 误差
ba, bg = np.zeros(2), 0.0                # 零偏初值为 0（完全未标定）

P = np.diag([1.0, 1.0, 0.5, 0.5, np.deg2rad(3) ** 2, 0.1, 0.1, np.deg2rad(0.1) ** 2])
Qc = np.diag([NA_STD ** 2, NA_STD ** 2, NG_STD ** 2, BA_RW ** 2, BA_RW ** 2, BG_RW ** 2])
G = np.zeros((8, 6))
H = np.zeros((2, 8)); H[0, 0] = H[1, 1] = 1.0
Rg = np.eye(2) * GNSS_SIGMA ** 2

log = []
next_gnss = 0.0
for k in range(n):
    t, pt, vt, tht = traj[k]
    am, gm = imu[k]
    a_hat = am - ba
    w_hat = gm - bg
    Rm = rot(th)

    # --- 名义状态积分（一阶欧拉，车规实现应用中点法）---
    p = p + v * DT
    v = v + (Rm @ a_hat) * DT
    th = th + w_hat * DT

    # --- 误差状态传播 ---
    F = np.eye(8)
    F[0:2, 2:4] = np.eye(2) * DT
    F[2:4, 4] = (Rm @ J2 @ a_hat) * DT
    F[2:4, 5:7] = -Rm * DT
    F[4, 7] = -DT
    G[2:4, 0:2] = -Rm; G[4, 2] = -1.0; G[5:7, 3:5] = np.eye(2); G[7, 5] = 1.0
    P = F @ P @ F.T + G @ Qc @ G.T * DT

    # --- GNSS 更新（失锁期间跳过）---
    if t >= next_gnss:
        next_gnss += GNSS_DT
        if not (OUTAGE[0] <= t < OUTAGE[1]):
            z = pt + np.random.randn(2) * GNSS_SIGMA
            y = z - p
            S = H @ P @ H.T + Rg
            if float(y @ np.linalg.solve(S, y)) < 13.8:      # 卡方门控 2 自由度 99.9%
                K = P @ H.T @ np.linalg.inv(S)
                dx = K @ y
                p += dx[0:2]; v += dx[2:4]; th += dx[4]
                ba += dx[5:7]; bg += dx[7]
                I_KH = np.eye(8) - K @ H
                P = I_KH @ P @ I_KH.T + K @ Rg @ K.T        # Joseph 形式

    log.append((t, np.linalg.norm(p - pt), np.rad2deg(th - tht),
                ba.copy(), bg, np.sqrt(P[0, 0] + P[1, 1])))

# ---------- 3. 输出 ----------
print("t(s)  |pos_err|(m)  yaw_err(deg)  ba_x  ba_y  bg(deg/h)  sqrt(trP_pos)(m)  GNSS")
for tq in [1, 5, 10, 20, 30, 39.9, 45, 50, 55, 59.9, 61, 65, 80, 100, 119.9]:
    t, e, ey, b_a, b_g, sp = log[min(int(tq / DT), n - 1)]
    tag = "OUT" if OUTAGE[0] <= t < OUTAGE[1] else "ON"
    print(f"{t:6.1f} {e:11.3f} {ey:12.3f} {b_a[0]:+7.4f} {b_a[1]:+7.4f} "
          f"{np.rad2deg(b_g)*3600:9.1f} {sp:15.3f}  {tag}")

on = [e for t, e, *_ in log if t < OUTAGE[0] or t > OUTAGE[1] + 1.0]
out = [e for t, e, *_ in log if OUTAGE[0] <= t < OUTAGE[1]]
print(f"\nGNSS 在线段: RMS={np.sqrt(np.mean(np.square(on))):.3f} m  max={max(on):.3f} m")
print(f"失锁 20 s 段: 末端误差={out[-1]:.3f} m  max={max(out):.3f} m")
print(f"失锁后重捕: 0.2 s 内收敛回 {log[int((OUTAGE[1]+0.4)/DT)][1]:.3f} m")
print(f"零偏真值 ba=({BA_TRUE[0]:+.4f},{BA_TRUE[1]:+.4f}) m/s^2, bg={np.rad2deg(BG_TRUE)*3600:.1f} deg/h")
print(f"零偏估计 ba=({ba[0]:+.4f},{ba[1]:+.4f}) m/s^2, bg={np.rad2deg(bg)*3600:.1f} deg/h")

# 对照：完全不做 GNSS 更新的纯惯导
p2, v2, th2 = traj[0][1] + np.array([1.0, -1.0]), traj[0][2].copy(), traj[0][3] + np.deg2rad(1.0)
pure = []
for k in range(n):
    t, pt, _, _ = traj[k]
    am, gm = imu[k]
    p2 = p2 + v2 * DT
    v2 = v2 + (rot(th2) @ am) * DT
    th2 = th2 + gm * DT
    pure.append((t, np.linalg.norm(p2 - pt)))
print("\n纯惯导（零 GNSS、零偏未估计）漂移:")
for tq in [1, 5, 10, 20, 60, 120]:
    t, e = pure[min(int(tq / DT), n - 1)]
    print(f"  t={t:6.1f}s  误差={e:10.2f} m")
