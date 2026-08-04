"""
ch06 车辆动力学与稳定性校核
==========================
1) 二自由度模型的横摆/侧向模态：固有频率、阻尼比随车速的变化
2) ISO 4138 稳态回转：数值反演不足转向梯度，验证解析式 K_us = Wf/Cf - Wr/Cr
3) ISO 7401 阶跃转向响应：上升时间 / 超调 / 稳态横摆角速度增益
4) 开环频率特性：LQR 闭环的相位裕度与延时裕度（解释仿真中 200/250 ms 的分界）
"""
import numpy as np
from lateral_control_benchmark import (M, IZ, LF, LR, L, CF, CR, G, K_US, K_S,
                                       V_CHAR, STEER_RATIO, DT, lqr_gain,
                                       error_model)


def ss_2dof(vx):
    """[vy, r] 状态矩阵"""
    A = np.array([[-(CF + CR) / (M * vx), (-CF * LF + CR * LR) / (M * vx) - vx],
                  [-(CF * LF - CR * LR) / (IZ * vx),
                   -(CF * LF ** 2 + CR * LR ** 2) / (IZ * vx)]])
    B = np.array([[CF / M], [CF * LF / IZ]])
    return A, B


print('=' * 78)
print('1) 横摆/侧向模态（二自由度特征根）')
print(f"{'vx[m/s]':>8}{'km/h':>7}{'wn[rad/s]':>11}{'fn[Hz]':>9}{'zeta':>8}"
      f"{'实部':>9}{'横摆增益 r/delta[1/s]':>22}")
for vx in (5, 10, 15, 20, 25, 30, 35, 40):
    A, _ = ss_2dof(float(vx))
    det, tr = np.linalg.det(A), np.trace(A)
    wn = np.sqrt(abs(det))
    zeta = -tr / (2 * wn)
    gain = vx / (L + K_S * vx ** 2)              # 稳态横摆角速度增益
    print(f'{vx:>8}{vx*3.6:>7.0f}{wn:>11.2f}{wn/2/np.pi:>9.2f}{zeta:>8.3f}'
          f'{np.linalg.eigvals(A).real.max():>9.2f}{gain:>22.3f}')
print(f'解析特征车速 V_ch = sqrt(gL/K_us) = {V_CHAR:.2f} m/s = {V_CHAR*3.6:.0f} km/h'
      f'（横摆增益在此处取极大值 {V_CHAR/(L+K_S*V_CHAR**2):.3f} 1/s）')

print('=' * 78)
print('2) ISO 4138 稳态回转：定半径 R=100 m，逐步提速，反演不足转向梯度')
print(f"{'v[m/s]':>8}{'ay[m/s^2]':>11}{'ay[g]':>8}{'前轮delta[deg]':>16}"
      f"{'方向盘[deg]':>13}{'alpha_f-alpha_r[deg]':>22}")
R = 100.0
rows = []
for vx in (10, 15, 20, 25, 30, 35):
    ay = vx ** 2 / R
    af = (M * ay * LR / L) / CF
    ar = (M * ay * LF / L) / CR
    delta = L / R + af - ar
    rows.append((ay / G, delta))
    print(f'{vx:>8}{ay:>11.2f}{ay/G:>8.3f}{np.rad2deg(delta):>16.3f}'
          f'{np.rad2deg(delta)*STEER_RATIO:>13.2f}{np.rad2deg(af-ar):>22.3f}')
xs = np.array([r[0] for r in rows])
ys = np.array([r[1] for r in rows])
slope = np.polyfit(xs, ys, 1)[0]
print(f'数值拟合斜率 d(delta)/d(ay/g) = {slope:.5f} rad/g = '
      f'{np.rad2deg(slope):.3f} deg/g')
print(f'解析式    K_us = Wf/Cf - Wr/Cr  = {K_US:.5f} rad/g = '
      f'{np.rad2deg(K_US):.3f} deg/g   -> 相对误差 '
      f'{abs(slope-K_US)/K_US*100:.4f}%')

print('=' * 78)
print('3) ISO 7401 阶跃转向响应（方向盘阶跃 40 deg，即前轮 2.5 deg）')
print(f"{'vx[m/s]':>8}{'r_ss[deg/s]':>13}{'峰值r[deg/s]':>14}{'超调[%]':>10}"
      f"{'上升时间T90[s]':>16}{'峰值时间[s]':>13}{'ay_ss[g]':>10}")
delta_step = np.deg2rad(40.0 / STEER_RATIO)
for vx in (10, 20, 30):
    A, B = ss_2dof(float(vx))
    x = np.zeros((2, 1))
    hist, ts = [], []
    for k in range(int(4.0 / 0.001)):
        x = x + 0.001 * (A @ x + B * delta_step)
        hist.append(float(x[1, 0]))
        ts.append(k * 0.001)
    hist, ts = np.array(hist), np.array(ts)
    r_ss = hist[-1]
    pk = hist.max() if r_ss > 0 else hist.min()
    tp = ts[int(np.argmax(hist))]
    i90 = int(np.argmax(hist >= 0.9 * r_ss))
    print(f'{vx:>8}{np.rad2deg(r_ss):>13.3f}{np.rad2deg(pk):>14.3f}'
          f'{(pk/r_ss-1)*100:>10.1f}{ts[i90]:>16.3f}{tp:>13.3f}'
          f'{r_ss*vx/G:>10.3f}')

print('=' * 78)
print('4) LQR 闭环开环频率特性与延时裕度（vx = 20 m/s）')
vx = 20.0
A, B = error_model(vx)
K = lqr_gain(vx)
print(f'LQR 增益 K = [{K[0]:.4f}, {K[1]:.4f}, {K[2]:.4f}, {K[3]:.4f}]')
ws = np.logspace(-1, 2, 4000)
mag, pha = [], []
for w in ws:
    Lo = (K @ np.linalg.solve(1j * w * np.eye(4) - A, B))[0]   # 开环传递 K(jwI-A)^-1 B
    mag.append(abs(Lo))
    pha.append(np.angle(Lo))
mag, pha = np.array(mag), np.unwrap(np.array(pha))
i = int(np.argmin(np.abs(mag - 1.0)))
wc = ws[i]
pm = np.pi + pha[i]                       # 相位裕度（负反馈）
print(f'穿越频率 wc = {wc:.2f} rad/s = {wc/2/np.pi:.2f} Hz')
print(f'相位裕度 PM = {np.rad2deg(pm):.1f} deg')
print(f'延时裕度 tau_max = PM/wc = {pm/wc*1000:.0f} ms  '
      f'(EPS 一阶滞后 50 ms 另占 {np.rad2deg(np.arctan(wc*0.05)):.1f} deg 相位)')
tau_eff = (pm - np.arctan(wc * 0.05)) / wc
print(f'扣掉 EPS 滞后后可容忍的纯延时 ≈ {tau_eff*1000:.0f} ms '
      f'-> 与仿真中 200 ms 稳定 / 250 ms 发散的分界一致')
print('=' * 78)
