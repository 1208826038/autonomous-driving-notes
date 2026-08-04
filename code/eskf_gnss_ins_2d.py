"""2D 误差状态卡尔曼滤波（ESKF, Error-State Kalman Filter）GNSS/INS 松耦合组合导航

名义状态 (nominal state)：p(2) v(2) theta(1) ba(2) bg(1)
误差状态 (error state) ：dp(2) dv(2) dtheta(1) dba(2) dbg(1)  -> 8 维
IMU 100 Hz 预测，GNSS 5 Hz 位置更新，60~90 s 人为失锁。

运行： python eskf_gnss_ins_2d.py
"""
import numpy as np

np.random.seed(20260804)

DT, T_END = 0.01, 120.0            # IMU 100 Hz
GNSS_DT = 0.2                      # GNSS 5 Hz
OUTAGE = (60.0, 90.0)              # 人为 GNSS 失锁窗口（隧道 / 城市峡谷）

BA_TRUE = np.array([0.050, -0.030])        # 加计上电零偏 m/s^2
BG_TRUE = np.deg2rad(0.30)                 # 陀螺上电零偏 rad/s (0.30 deg/s = 1080 deg/h)
SIG_A, SIG_G = 1.86e-3, 1.22e-4            # 噪声密度 m/s^2/rtHz, rad/s/rtHz（车规 MEMS）
SIG_BA, SIG_BG = 6.0e-5, 3.0e-6            # 零偏随机游走
SIG_GNSS = 0.05                            # RTK 固定解水平精度 5 cm
J = np.array([[0.0, -1.0], [1.0, 0.0]])    # 2D 旋转生成元 dR/dtheta = R @ J


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def profile(t, speed):
    """返回该时刻的真实体轴比力 a_b=[纵向, 侧向] 与偏航角速率 w"""
    if t < 15.0:      al, w = 0.80, 0.0          # 起步加速到 12 m/s
    elif t < 35.0:    al, w = 0.00, 0.0          # 直线巡航
    elif t < 50.0:    al, w = 0.00, 0.12         # 左转弯（103 度）
    elif t < 70.0:    al, w = 0.00, 0.0          # 直线（失锁开始）
    elif t < 85.0:    al, w = 0.00, -0.10        # 失锁中右转弯（86 度）
    elif t < 105.0:   al, w = 0.30, 0.0          # 加速直行
    else:             al, w = -0.60, 0.0         # 减速
    return np.array([al, speed * w]), w


# ---------- 1. 生成真值轨迹与 IMU 量测 ----------
N = int(T_END / DT)
p_t, v_t, th_t = np.zeros(2), np.zeros(2), 0.0
truth, imu = np.zeros((N, 5)), np.zeros((N, 3))
for k in range(N):
    ab, w = profile(k * DT, np.linalg.norm(v_t))
    truth[k] = [p_t[0], p_t[1], v_t[0], v_t[1], th_t]
    imu[k, :2] = ab + BA_TRUE + np.random.randn(2) * SIG_A / np.sqrt(DT)
    imu[k, 2] = w + BG_TRUE + np.random.randn() * SIG_G / np.sqrt(DT)
    v_t = v_t + rot(th_t) @ ab * DT
    p_t = p_t + v_t * DT
    th_t = th_t + w * DT

# ---------- 2. ESKF ----------
p, v, th = truth[0, :2].copy(), truth[0, 2:4].copy(), truth[0, 4]
ba, bg = np.zeros(2), 0.0
P = np.diag([5.0**2, 5.0**2, 1.0**2, 1.0**2, np.deg2rad(5)**2,
             0.20**2, 0.20**2, np.deg2rad(1.0)**2])
Q = np.zeros((8, 8))
Q[2:4, 2:4] = np.eye(2) * SIG_A**2 * DT
Q[4, 4] = SIG_G**2 * DT
Q[5:7, 5:7] = np.eye(2) * SIG_BA**2 * DT
Q[7, 7] = SIG_BG**2 * DT
H = np.zeros((2, 8)); H[:2, :2] = np.eye(2)
Rn = np.eye(2) * SIG_GNSS**2

# 开环纯惯导基线（不做任何修正，零偏未知）
p_o, v_o, th_o = truth[0, :2].copy(), truth[0, 2:4].copy(), truth[0, 4]

err, err_ol, bhist, rej = np.zeros(N), np.zeros(N), np.zeros((N, 3)), 0
next_gnss = 0.0
for k in range(N):
    t = k * DT
    am, wm = imu[k, :2], imu[k, 2]

    # --- 预测：名义状态机械编排 ---
    Rm = rot(th)
    acc = Rm @ (am - ba)
    v = v + acc * DT
    p = p + v * DT
    th = th + (wm - bg) * DT

    # --- 预测：误差状态协方差传播 ---
    F = np.eye(8)
    F[0:2, 2:4] = np.eye(2) * DT
    F[2:4, 4] = (Rm @ J @ (am - ba)) * DT
    F[2:4, 5:7] = -Rm * DT
    F[4, 7] = -DT
    P = F @ P @ F.T + Q

    # --- 更新：GNSS 位置观测 ---
    if t >= next_gnss:
        next_gnss += GNSS_DT
        if not (OUTAGE[0] <= t < OUTAGE[1]):
            z = truth[k, :2] + np.random.randn(2) * SIG_GNSS
            y = z - p
            S = H @ P @ H.T + Rn
            if y @ np.linalg.solve(S, y) < 9.21:          # 卡方门控 2 自由度 99%
                K = P @ H.T @ np.linalg.inv(S)
                dx = K @ y
                p, v, th = p + dx[0:2], v + dx[2:4], th + dx[4]   # 误差注入
                ba, bg = ba + dx[5:7], bg + dx[7]
                IKH = np.eye(8) - K @ H
                P = IKH @ P @ IKH.T + K @ Rn @ K.T                # Joseph 形式
            else:
                rej += 1

    # --- 开环纯惯导 ---
    Ro = rot(th_o)
    v_o = v_o + Ro @ am * DT
    p_o = p_o + v_o * DT
    th_o = th_o + wm * DT

    err[k] = np.linalg.norm(p - truth[k, :2])
    err_ol[k] = np.linalg.norm(p_o - truth[k, :2])
    bhist[k] = [ba[0], ba[1], bg]

# ---------- 3. 输出 ----------
idx = lambda s: int(s / DT)
print("=" * 62)
print(" 2D ESKF GNSS/INS 组合导航  |  IMU 100Hz  GNSS 5Hz  失锁 60~90s")
print("=" * 62)
print(f"轨迹总长 {np.sum(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1)):.1f} m,"
      f" 最高车速 {np.max(np.linalg.norm(truth[:, 2:4], axis=1)):.2f} m/s,"
      f" GNSS 拒收 {rej} 次")
print("\n[A] 关键时刻定位误差（m）")
print(f"{'t(s)':>6}{'ESKF':>10}{'开环纯惯导':>14}   状态")
for s, tag in [(10, "GNSS 可用"), (30, "GNSS 可用"), (59.9, "失锁前一刻"),
               (65, "失锁 5s"), (75, "失锁 15s(转弯中)"), (89.9, "失锁 30s"),
               (90.5, "GNSS 恢复 0.5s"), (92, "GNSS 恢复 2s"), (119.9, "收尾")]:
    print(f"{s:>6.1f}{err[idx(s)]:>10.3f}{err_ol[idx(s)]:>14.1f}   {tag}")

m_on = np.r_[np.arange(idx(20), idx(60)), np.arange(idx(95), N)]
m_off = np.arange(idx(60), idx(90))
print(f"\n[B] 分段统计")
print(f"  GNSS 可用段 RMSE = {np.sqrt(np.mean(err[m_on]**2)):.3f} m,"
      f" 最大 {np.max(err[m_on]):.3f} m")
print(f"  失锁 30s 段 RMSE = {np.sqrt(np.mean(err[m_off]**2)):.3f} m,"
      f" 最大 {np.max(err[m_off]):.3f} m（发生在 t={idx(60)*DT + np.argmax(err[m_off])*DT:.1f}s）")
print(f"  开环纯惯导 120s 末端误差 = {err_ol[-1]:.1f} m（放大 {err_ol[-1]/max(err[-1],1e-9):.0f} 倍）")

print(f"\n[C] 零偏在线估计收敛（真值 ba=[{BA_TRUE[0]:.4f},{BA_TRUE[1]:.4f}] m/s^2, "
      f"bg={np.rad2deg(BG_TRUE):.4f} deg/s）")
print(f"{'t(s)':>6}{'ba_x':>10}{'ba_y':>10}{'bg(deg/s)':>12}")
for s in [5, 15, 30, 50, 59.9, 89.9, 119.9]:
    b = bhist[idx(s)]
    print(f"{s:>6.1f}{b[0]:>10.4f}{b[1]:>10.4f}{np.rad2deg(b[2]):>12.4f}")
e = bhist[idx(59.9)] - np.r_[BA_TRUE, BG_TRUE]
print(f"  60s 时残余零偏误差: ba=[{e[0]:+.4f},{e[1]:+.4f}] m/s^2, "
      f"bg={np.rad2deg(e[2]):+.4f} deg/s")
print(f"  该残余零偏对 30s 失锁的理论位置贡献 ~ {0.5*np.linalg.norm(e[:2])*30**2:.2f} m（加计项）")

rec = next(k for k in range(idx(90), N) if err[k] < 0.05)
print(f"\n[D] GNSS 恢复：失锁末 {err[idx(89.99)]:.3f} m -> 首帧修正后 {err[idx(90.01)]:.3f} m,"
      f" {(rec - idx(90)) * DT:.2f} s 内收敛到 0.05 m 以内")
print("=" * 62)
