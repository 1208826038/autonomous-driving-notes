# -*- coding: utf-8 -*-
"""
ch06 车辆控制 · 扩展验证套件 (autonomous-driving-notes 第六章配套代码 2/2)
========================================================================
复用 lateral_control_benchmark.py 的同一被控对象（二自由度动力学自行车 +
线性/饱和轮胎 + EPS 一阶滞后 + 纯延时 + 转角速率限幅 + 纵向 PID/前馈），
在其上追加下列实验：

  ①  MPC（condensed QP，Δu 盒约束，FISTA 投影梯度求解）并入横向控制器对比
  ②  车速鲁棒性扫描：40 / 80 / 120 km/h（等侧向加速度工况自动缩放轨迹）
  ③  执行器纯延时敏感性：0 / 100 / 200 ms × 5 种控制器
  ④  LQR 的 Q/R 权重扫描（增益 + 闭环指标）
  ⑤  轮胎模型对比：线性 / Fiala / Pacejka 魔术公式
  ⑥  附着椭圆：纵向减速度吃掉多少侧向余量
  ⑦  Smith 预估器 vs 状态外推（含 ±20% 侧偏刚度模型失配）
  ⑧  MPC 的 Np/Nc 选型与单步 QP 求解耗时
  ⑨  ACC 上下位分层：模式切换抖振（无迟滞 vs 迟滞 + 最短驻留）
  ⑩  低速病态：误差模型 1/vx 奇异性数值体检
  ⑪  控制频率选型：10 / 20 / 50 / 100 / 200 Hz 的离散化误差与抖动

运行:  python ch06_control_suite.py
"""
import sys
import time
from collections import deque

import numpy as np

try:                                       # Windows 控制台重定向时保证中文不炸
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:                          # pragma: no cover
    pass

from lateral_control_benchmark import (    # noqa: E402
    M, IZ, LF, LR, L, CF, CR, G, MU, FZF, FZR, STEER_RATIO, DELTA_MAX,
    RATE_MAX, TAU_EPS, T_DELAY, TAU_PWT, T_DELAY_LON, AX_MIN, AX_MAX,
    DT, DS, K_US, K_S, V_CHAR, KV, SAT_HIT,
    rk4, wrap, tire_fy, error_model, lqr_gain, lqr_ff,
    build_path, build_speed, Q_LQR, R_LQR,
)

np.set_printoptions(suppress=True, precision=4)
LINE = '=' * 92
THIN = '-' * 92


# =====================================================================
# 0. 通用：参考路径容器 / 被控对象仿真主循环
# =====================================================================
class Path:
    """按曲率剖面积分生成的参考路径（弧长等间隔 DS）。"""

    def __init__(self, s, x, y, psi, kap, v, tag=''):
        self.s, self.x, self.y = s, x, y
        self.psi, self.kap, self.v = psi, kap, v
        self.n = len(s)
        self.tag = tag

    def nearest(self, x, y, i0=0):
        lo, hi = max(0, i0 - 40), min(self.n, i0 + 800)
        return lo + int(np.argmin((self.x[lo:hi] - x) ** 2 +
                                  (self.y[lo:hi] - y) ** 2))

    def lat_err(self, x, y, i):
        return (-np.sin(self.psi[i]) * (x - self.x[i]) +
                np.cos(self.psi[i]) * (y - self.y[i]))

    def kap_at(self, s_query):
        j = int(np.clip(s_query / DS, 0, self.n - 1))
        return self.kap[j]


def _integrate(kap, v):
    psi = np.cumsum(kap) * DS
    return (np.cumsum(np.cos(psi)) * DS, np.cumsum(np.sin(psi)) * DS, psi, v)


# 默认工况：与 lateral_control_benchmark 完全一致（用于交叉校验）
_S0, _PX0, _PY0, _PPSI0, _PK0 = build_path()
PATH_DEFAULT = Path(_S0, _PX0, _PY0, _PPSI0, _PK0, build_speed(_S0), 'default')


def build_path_iso(v, dy=3.5, ay_curve=2.5):
    """等侧向加速度工况：变道长度 3v（约 3 s），弯道曲率 kappa = ay/v^2。

    这样 40/80/120 km/h 三档跑的是"同样激烈程度"的机动，
    对比出来的差异才纯粹来自控制器与车速，而不是工况难度。
    """
    lc = 3.0 * v                                    # 变道段长度
    ramp = 1.0 * v                                  # 弯道曲率过渡段
    arc = 3.0 * v                                   # 定曲率段
    s0, s1 = 30.0, 30.0 + lc                        # 直线 -> 变道
    s2 = s1 + 2.0 * v                               # 直线恢复
    s3, s4, s5 = s2 + ramp, s2 + ramp + arc, s2 + 2 * ramp + arc
    total = s5 + 40.0
    s = np.arange(0.0, total, DS)
    kap = np.zeros_like(s)
    amp = 2.0 * np.pi * dy / lc ** 2                # 解析标定：横移 = amp*lc^2/(2pi)
    k_c = ay_curve / v ** 2
    m = (s >= s0) & (s < s1)
    kap[m] = amp * np.sin(2 * np.pi * (s[m] - s0) / lc)
    m = (s >= s2) & (s < s3)
    kap[m] = k_c * (s[m] - s2) / ramp
    m = (s >= s3) & (s < s4)
    kap[m] = k_c
    m = (s >= s4) & (s < s5)
    kap[m] = k_c * (1 - (s[m] - s4) / ramp)
    px, py, ppsi, _ = _integrate(kap, None)
    return Path(s, px, py, ppsi, kap, np.full_like(s, v), f'{v * 3.6:.0f}km/h'), \
        dict(lc=(s0, s1), arc=(s3 + 0.3 * arc, s4), k_c=k_c, amp=amp,
             ay_lc=amp * v ** 2, ay_arc=ay_curve)


def simulate(ctl, path, t_delay=T_DELAY, tau_eps=TAU_EPS, t_end=40.0,
             stop_pts=900, dt_ctrl=DT):
    """统一被控对象。ctl(st, i, path, buf_in_flight) -> 前轮转角 [rad]"""
    ndly = max(0, int(round(t_delay / DT)))
    ndly_lon = int(round(T_DELAY_LON / DT))
    st = np.array([0.0, 0.0, 0.0, float(path.v[0]), 0.0, 0.0])
    buf, buf_a = [0.0] * ndly, [0.0] * ndly_lon
    d_act, d_prev, ax_act, ei = 0.0, None, 0.0, 0.0
    i, t, cmd = 0, 0.0, 0.0
    n_hold = max(1, int(round(dt_ctrl / DT)))       # 控制降频：零阶保持
    log = {k: [] for k in ('t', 'e', 's', 'rate', 'ay', 'd', 've', 'v', 'beta')}
    step = 0
    while i < path.n - stop_pts and t < t_end:
        i = path.nearest(st[0], st[1], i)
        if step % n_hold == 0:                      # 控制器按 dt_ctrl 触发
            cmd = ctl(st, i, path, list(buf))
            cmd = float(np.clip(cmd, -DELTA_MAX, DELTA_MAX))
        rate_cmd = 0.0 if d_prev is None else (cmd - d_prev) / DT
        d_prev = cmd
        buf.append(cmd)
        d_tgt = buf.pop(0) if ndly else cmd         # 纯延时（环形缓冲）
        d_new = d_act + (d_tgt - d_act) * DT / tau_eps          # EPS 一阶滞后
        d_new = np.clip(d_new, d_act - RATE_MAX * DT,
                        d_act + RATE_MAX * DT)                  # 转角速率限幅
        d_act = float(np.clip(d_new, -DELTA_MAX, DELTA_MAX))
        v_ref = path.v[i]
        a_ff = (path.v[min(i + 200, path.n - 1)] ** 2 - v_ref ** 2) / (2 * 10.0)
        ev = v_ref - st[3]
        ei = float(np.clip(ei + ev * DT, -3.0, 3.0))            # 抗积分饱和
        a_cmd = float(np.clip(a_ff + 1.2 * ev + 0.35 * ei, AX_MIN, AX_MAX))
        buf_a.append(a_cmd)
        ax_act += (buf_a.pop(0) - ax_act) * DT / TAU_PWT
        af = d_act - (st[4] + LF * st[5]) / max(st[3], 1.0)
        ar = -(st[4] - LR * st[5]) / max(st[3], 1.0)
        ay = (tire_fy(af, CF, FZF) + tire_fy(ar, CR, FZR)) / M
        st = rk4(st, d_act, ax_act, DT)
        for k_, v_ in zip(('t', 'e', 's', 'rate', 'ay', 'd', 've', 'v', 'beta'),
                          (t, path.lat_err(st[0], st[1], i), path.s[i], rate_cmd,
                           ay, d_act, ev, st[3],
                           np.arctan2(st[4], max(st[3], 1.0)))):
            log[k_].append(v_)
        t += DT
        step += 1
        if abs(path.lat_err(st[0], st[1], i)) > 60.0:           # 发散提前退出
            break
    return {k: np.array(v) for k, v in log.items()}


def metrics(lg, lc_win=(20, 82), arc_win=(155, 173)):
    e, s = lg['e'], lg['s']
    m_lc = (s > lc_win[0]) & (s < lc_win[1])
    m_arc = (s > arc_win[0]) & (s < arc_win[1])
    return dict(max_e=np.abs(e).max() * 100,
                lc_e=(np.abs(e[m_lc]).max() * 100) if m_lc.any() else np.nan,
                ss_e=(e[m_arc].mean() * 100) if m_arc.any() else np.nan,
                rms=np.sqrt((e ** 2).mean()) * 100,
                sw_rate=np.abs(lg['rate']).max() * STEER_RATIO * 180 / np.pi,
                max_sw=np.abs(lg['d']).max() * STEER_RATIO * 180 / np.pi,
                max_ay=np.abs(lg['ay']).max(),
                max_beta=np.rad2deg(np.abs(lg['beta']).max()),
                v_err=np.abs(lg['ve']).max())


# =====================================================================
# 1. 控制器工厂
# =====================================================================
def make_sched(Q=None, R=None):
    """增益调度：每档车速离线解一次离散黎卡提，运行时线性插值。"""
    Q = Q_LQR if Q is None else Q
    R = R_LQR if R is None else R
    vt = np.arange(3.0, 36.0, 1.0)
    kt = np.array([lqr_gain(float(v), Q, R) for v in vt])

    def sched(vx):
        vx = float(np.clip(vx, vt[0], vt[-1]))
        return np.array([np.interp(vx, vt, kt[:, j]) for j in range(4)])
    sched.table = (vt, kt)
    return sched


SCHED_DEFAULT = make_sched()


def ctl_pp(k, l0):
    """Pure Pursuit：参考点后轴中心，预瞄 ld = k*v + l0。"""
    def f(st, i, path, buf):
        x, y, psi, vx = st[0], st[1], st[2], st[3]
        rx, ry = x - LR * np.cos(psi), y - LR * np.sin(psi)
        ld = max(k * vx + l0, 2.0)
        j = min(path.nearest(rx, ry, max(0, i - 80)) + int(ld / DS), path.n - 1)
        alpha = wrap(np.arctan2(path.y[j] - ry, path.x[j] - rx) - psi)
        return np.arctan2(2 * L * np.sin(alpha), ld)
    return f


def ctl_stanley(k=2.0, ks=1.0):
    """Stanley：参考点前轴中心，delta = theta_e + atan(k*e/(v+ks))。"""
    def f(st, i, path, buf):
        x, y, psi, vx = st[0], st[1], st[2], st[3]
        fx, fy = x + LF * np.cos(psi), y + LF * np.sin(psi)
        j = path.nearest(fx, fy, i)
        return (wrap(path.psi[j] - psi) +
                np.arctan2(-k * path.lat_err(fx, fy, j), vx + ks))
    return f


def err_state(st, i, path, vxs):
    x, y, psi, _, vy, r = st
    e1 = path.lat_err(x, y, i)
    e3 = wrap(psi - path.psi[i])
    e2 = vy + vxs * np.sin(e3)
    e4 = r - vxs * path.kap[i]
    return np.array([e1, e2, e3, e4])


def ctl_lqr(sched=None, delay_comp=False):
    sched = SCHED_DEFAULT if sched is None else sched

    def f(st, i, path, buf):
        if delay_comp:                     # 用在途指令把状态前推 tau_d
            for u in buf:
                st = rk4(st, u, 0.0, DT)
            i = path.nearest(st[0], st[1], i)
        vxs = max(st[3], 2.0)
        K = sched(vxs)
        return (-float(K @ err_state(st, i, path, vxs)) +
                lqr_ff(path.kap[i], vxs, K[2]))
    return f


# ---------------------------------------------------------------------
# MPC：condensed QP + Δu 盒约束 + FISTA 投影梯度
# ---------------------------------------------------------------------
class MPC:
    """线性时变 MPC。

    误差动力学          e_dot = A e + B delta + E * (vx*kappa)
    离散预测堆叠        X = Phi x0 + Gam U + Psi D
    增量化              U = S dU + 1 u_prev,  S 为下三角全 1
    代价                J = X' Qbar X + dU' Rbar dU
                        = 1/2 dU' H dU + f' dU + const
                        H = 2 (Gt' Qbar Gt + Rbar),  f = 2 Gt' Qbar x_free
    约束                |dU_k| <= rate_max * dt_mpc （盒约束，投影即可）
    """

    def __init__(self, Np=25, Nc=8, dt_mpc=0.02,
                 q=(50.0, 1.0, 60.0, 1.0), r_du=5.0e4, q_term=5.0,
                 iters=45, delay_comp=False, name='MPC'):
        self.Np, self.Nc, self.dtm = Np, Nc, dt_mpc
        self.q, self.r_du, self.q_term = np.array(q), r_du, q_term
        self.iters, self.delay_comp, self.name = iters, delay_comp, name
        self.cache, self.u_prev = {}, 0.0
        self.solve_ms, self.gap = [], []

    # ---- 离线：按车速缓存预测矩阵 ----
    def _mats(self, vx):
        key = int(round(vx))
        if key in self.cache:
            return self.cache[key]
        vxs = max(float(key), 3.0)
        A, B = error_model(vxs)
        E = np.array([[0.0],
                      [-((CF * LF - CR * LR) / (M * vxs) + vxs)],
                      [0.0],
                      [-(CF * LF ** 2 + CR * LR ** 2) / (IZ * vxs)]])
        dtm = self.dtm
        Ad = np.eye(4) + A * dtm + 0.5 * (A @ A) * dtm ** 2
        Bd = (np.eye(4) * dtm + 0.5 * A * dtm ** 2) @ B
        Ed = (np.eye(4) * dtm + 0.5 * A * dtm ** 2) @ E
        Np_, Nc_ = self.Np, self.Nc
        pows = [np.eye(4)]
        for _ in range(Np_):
            pows.append(pows[-1] @ Ad)
        Phi = np.zeros((4 * Np_, 4))
        Gam = np.zeros((4 * Np_, Nc_))
        Psi = np.zeros((4 * Np_, Np_))
        for k in range(1, Np_ + 1):
            rr = slice(4 * (k - 1), 4 * k)
            Phi[rr, :] = pows[k]
            for j in range(k):
                mcol = min(j, Nc_ - 1)
                Gam[rr, mcol] += (pows[k - 1 - j] @ Bd).flatten()
                Psi[rr, j] = (pows[k - 1 - j] @ Ed).flatten()
        qd = np.tile(self.q, Np_)
        qd[-4:] *= self.q_term                      # 终端权重
        Qbar = np.diag(qd)
        Rbar = self.r_du * np.eye(Nc_)
        Stri = np.tril(np.ones((Nc_, Nc_)))
        Gt = Gam @ Stri
        H = 2.0 * (Gt.T @ Qbar @ Gt + Rbar)
        GtQ = 2.0 * Gt.T @ Qbar
        alpha = 1.0 / np.linalg.eigvalsh(H).max()
        out = dict(Phi=Phi, Gam=Gam, Psi=Psi, gsum=Gam.sum(axis=1),
                   H=H, GtQ=GtQ, alpha=alpha,
                   Hun=np.linalg.inv(H))
        self.cache[key] = out
        return out

    # ---- 在线：FISTA 投影梯度解盒约束 QP ----
    def _solve(self, H, f, alpha, du_max):
        n = len(f)
        z = np.zeros(n)
        y = z.copy()
        tk = 1.0
        for _ in range(self.iters):
            g = H @ y + f
            zn = np.clip(y - alpha * g, -du_max, du_max)
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y = zn + (tk - 1.0) / tn * (zn - z)
            z, tk = zn, tn
        return z

    def __call__(self, st, i, path, buf):
        if self.delay_comp:
            for u in buf:
                st = rk4(st, u, 0.0, DT)
            i = path.nearest(st[0], st[1], i)
        vxs = max(st[3], 3.0)
        mt = self._mats(vxs)
        x0 = err_state(st, i, path, vxs)
        dvec = np.array([vxs * path.kap[min(int(i + vxs * k * self.dtm / DS),
                                            path.n - 1)]
                         for k in range(self.Np)])
        t0 = time.perf_counter()
        x_free = mt['Phi'] @ x0 + mt['gsum'] * self.u_prev + mt['Psi'] @ dvec
        f = mt['GtQ'] @ x_free
        du = self._solve(mt['H'], f, mt['alpha'], RATE_MAX * self.dtm)
        self.solve_ms.append((time.perf_counter() - t0) * 1000.0)
        du_un = -mt['Hun'] @ f                      # 无约束解析解（用于对比）
        self.gap.append(float(np.abs(du_un[0] - du[0])))
        u = float(np.clip(self.u_prev + du[0], -DELTA_MAX, DELTA_MAX))
        self.u_prev = u
        return u

    def reset(self):
        self.u_prev = 0.0
        self.solve_ms, self.gap = [], []


# ---------------------------------------------------------------------
# Smith 预估器（含模型失配）
# ---------------------------------------------------------------------
def model_deriv(st, delta, cf, cr):
    _, _, psi, vx, vy, r = st
    vxs = max(vx, 1.0)
    af = delta - (vy + LF * r) / vxs
    ar = -(vy - LR * r) / vxs
    fyf, fyr = cf * af, cr * ar
    return np.array([vx * np.cos(psi) - vy * np.sin(psi),
                     vx * np.sin(psi) + vy * np.cos(psi),
                     r, 0.0,
                     (fyf + fyr) / M - vx * r,
                     (LF * fyf - LR * fyr) / IZ])


def rk4m(st, delta, cf, cr, dt):
    k1 = model_deriv(st, delta, cf, cr)
    k2 = model_deriv(st + dt / 2 * k1, delta, cf, cr)
    k3 = model_deriv(st + dt / 2 * k2, delta, cf, cr)
    k4 = model_deriv(st + dt * k3, delta, cf, cr)
    return st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class SmithLQR:
    """Smith 预估器：并联"无延时模型 - 有延时模型"，把纯延时移出反馈环。

        x_corr = x_meas + (x_model_nodelay - x_model_delayed)
    """

    def __init__(self, t_delay, cf_scale=1.0, cr_scale=1.0, sched=None):
        self.n = max(1, int(round(t_delay / DT)))
        self.cf, self.cr = CF * cf_scale, CR * cr_scale
        self.sched = SCHED_DEFAULT if sched is None else sched
        self.hist = deque([0.0] * self.n, maxlen=self.n + 4)
        self.xm = None
        self.xmd = None

    def __call__(self, st, i, path, buf):
        if self.xm is None:
            self.xm, self.xmd = st.copy(), st.copy()
        corr = st + (self.xm - self.xmd)
        corr[3] = st[3]
        j = path.nearest(corr[0], corr[1], max(0, i - 60))
        vxs = max(st[3], 2.0)
        K = self.sched(vxs)
        u = (-float(K @ err_state(corr, j, path, vxs)) +
             lqr_ff(path.kap[j], vxs, K[2]))
        u = float(np.clip(u, -DELTA_MAX, DELTA_MAX))
        ud = self.hist.popleft() if len(self.hist) >= self.n else 0.0
        self.hist.append(u)
        self.xm = rk4m(self.xm, u, self.cf, self.cr, DT)
        self.xmd = rk4m(self.xmd, ud, self.cf, self.cr, DT)
        self.xm[3] = self.xmd[3] = st[3]
        return u


# =====================================================================
# 实验 ①：五+二方横向控制器同台对比（默认工况，150 ms 延时）
# =====================================================================
def exp1_controllers():
    print(LINE)
    print('实验① 横向控制器同台对比  |  被控对象: 2-DOF 动力学自行车 + '
          f'{T_DELAY * 1000:.0f} ms 纯延时 + EPS {TAU_EPS * 1000:.0f} ms + '
          f'速率限 {np.rad2deg(RATE_MAX) * STEER_RATIO:.0f} deg/s')
    print(f'工况: 直线 -> 3.5 m 变道(72 km/h) -> 减速 -> kappa=0.05 弯(36 km/h, '
          f'{0.05 * 100 / G:.2f} g) -> 加速   控制 {1 / DT:.0f} Hz')
    print(THIN)
    mpc = MPC()
    mpc_d = MPC(delay_comp=True, name='MPC+延时补偿')
    cases = [
        ('PurePursuit ld=0.3v+3', ctl_pp(0.3, 3.0)),
        ('PurePursuit ld=4m 固定', ctl_pp(0.0, 4.0)),
        ('Stanley k=2.0', ctl_stanley()),
        ('LQR + 曲率前馈', ctl_lqr()),
        ('LQR + 前馈 + 延时补偿', ctl_lqr(delay_comp=True)),
        ('MPC Np=25 Nc=8', mpc),
        ('MPC + 延时补偿', mpc_d),
    ]
    print(f"{'控制器':<24}{'最大|e|':>10}{'变道max':>10}{'弯道稳态e':>11}"
          f"{'RMS':>9}{'盘峰值转速':>13}{'峰值ay':>10}{'质心侧偏':>10}{'饱和':>6}")
    res = {}
    for label, ctl in cases:
        SAT_HIT[0] = False
        if isinstance(ctl, MPC):
            ctl.reset()
        r = metrics(simulate(ctl, PATH_DEFAULT))
        r['sat'] = SAT_HIT[0]
        res[label] = r
        print(f'{label:<24}{r["max_e"]:>8.1f}cm{r["lc_e"]:>8.1f}cm'
              f'{r["ss_e"]:>9.1f}cm{r["rms"]:>7.1f}cm'
              f'{r["sw_rate"]:>10.1f}d/s{r["max_ay"] / G:>9.2f}g'
              f'{r["max_beta"]:>9.2f}°{"是" if r["sat"] else "否":>6}')
    print(THIN)
    print(f'MPC 单步 QP 求解耗时: 均值 {np.mean(mpc.solve_ms):.3f} ms / '
          f'p95 {np.percentile(mpc.solve_ms, 95):.3f} ms / '
          f'max {np.max(mpc.solve_ms):.3f} ms  (Np=25, Nc=8, FISTA 45 迭代)')
    print(f'约束激活强度: |无约束解 - 投影解| 首元素均值 '
          f'{np.mean(mpc.gap) * 1e3:.4f} mrad, 最大 {np.max(mpc.gap) * 1e3:.4f} mrad')
    return res


# =====================================================================
# 实验 ②：车速鲁棒性扫描 40 / 80 / 120 km/h
# =====================================================================
def exp2_speed_sweep():
    print(LINE)
    print('实验② 车速鲁棒性扫描（等侧向加速度工况：变道 3 s 完成，弯道 ay=2.5 m/s²）')
    print(THIN)
    for kmh in (40, 80, 120):
        v = kmh / 3.6
        path, info = build_path_iso(v)
        lcw, arcw = info['lc'], info['arc']
        print(f'--- {kmh} km/h (v={v:.2f} m/s)  变道段 {lcw[0]:.0f}~{lcw[1]:.0f} m, '
              f'kappa_lc={info["amp"]:.5f} 1/m, ay_lc={info["ay_lc"]:.2f} m/s² '
              f'({info["ay_lc"] / G:.2f} g); 弯道 kappa={info["k_c"]:.5f} 1/m '
              f'(R={1 / info["k_c"]:.0f} m), ay={info["ay_arc"]:.2f} m/s² '
              f'({info["ay_arc"] / G:.2f} g)')
        print(f"  {'控制器':<24}{'最大|e|':>10}{'变道max':>10}{'弯道稳态e':>11}"
              f"{'RMS':>9}{'盘峰值转速':>13}{'峰值ay':>10}")
        mpc = MPC(delay_comp=True)
        for label, ctl in [('PurePursuit ld=0.3v+3', ctl_pp(0.3, 3.0)),
                           ('Stanley k=2.0', ctl_stanley()),
                           ('LQR + 曲率前馈', ctl_lqr()),
                           ('LQR + 前馈 + 延时补偿', ctl_lqr(delay_comp=True)),
                           ('MPC + 延时补偿', mpc)]:
            SAT_HIT[0] = False
            if isinstance(ctl, MPC):
                ctl.reset()
            lg = simulate(ctl, path, stop_pts=700, t_end=60.0)
            r = metrics(lg, lc_win=lcw, arc_win=arcw)
            print(f'  {label:<24}{r["max_e"]:>8.1f}cm{r["lc_e"]:>8.1f}cm'
                  f'{r["ss_e"]:>9.1f}cm{r["rms"]:>7.1f}cm'
                  f'{r["sw_rate"]:>10.1f}d/s{r["max_ay"] / G:>9.2f}g')


# =====================================================================
# 实验 ③：执行器纯延时敏感性 0 / 100 / 200 ms
# =====================================================================
def exp3_delay():
    print(LINE)
    print('实验③ 执行器纯延时敏感性（默认工况，逐控制器 × 0/100/200 ms）')
    print(THIN)
    print(f"{'控制器':<24}" + ''.join(f'{f'{d} ms':>26}' for d in (0, 100, 200)))
    print(f"{'':<24}" + ''.join(f"{'最大|e| / RMS / 盘转速':>26}" for _ in range(3)))
    mpc = MPC(delay_comp=True)
    rows = [('PurePursuit ld=0.3v+3', lambda: ctl_pp(0.3, 3.0)),
            ('Stanley k=2.0', lambda: ctl_stanley()),
            ('LQR + 曲率前馈', lambda: ctl_lqr()),
            ('LQR + 前馈 + 延时补偿', lambda: ctl_lqr(delay_comp=True)),
            ('MPC + 延时补偿', lambda: mpc)]
    for label, mk in rows:
        cells = []
        for d_ms in (0, 100, 200):
            ctl = mk()
            if isinstance(ctl, MPC):
                ctl.reset()
            r = metrics(simulate(ctl, PATH_DEFAULT, t_delay=d_ms / 1000.0))
            cells.append(f'{r["max_e"]:>8.1f}/{r["rms"]:>6.1f}/{r["sw_rate"]:>8.0f}')
        print(f'{label:<24}' + ''.join(f'{c:>26}' for c in cells))
    print(THIN)
    print('单位: 最大|e| [cm] / RMS [cm] / 方向盘峰值转速 [deg/s]')


# =====================================================================
# 实验 ④：LQR 的 Q/R 权重扫描
# =====================================================================
def exp4_qr_sweep():
    print(LINE)
    print('实验④ LQR 权重扫描：Q=diag(q1, 0.5, q3, 0.3)，R 变化 (vx=20 m/s 处的增益)')
    print(THIN)
    print(f"{'q1(ey)':>8}{'q3(epsi)':>10}{'R':>8}{'k1':>9}{'k2':>9}{'k3':>9}"
          f"{'k4':>9}{'最大|e|':>10}{'RMS':>9}{'盘峰值转速':>13}{'峰值ay':>9}")
    combos = [(3.0, 12.0, 50.0), (3.0, 12.0, 250.0), (3.0, 12.0, 1000.0),
              (1.0, 12.0, 250.0), (10.0, 12.0, 250.0), (30.0, 12.0, 250.0),
              (3.0, 4.0, 250.0), (3.0, 40.0, 250.0), (10.0, 40.0, 1000.0)]
    for q1, q3, R in combos:
        Q = np.diag([q1, 0.5, q3, 0.3])
        Rm = np.array([[R]])
        sched = make_sched(Q, Rm)
        K = lqr_gain(20.0, Q, Rm)
        SAT_HIT[0] = False
        r = metrics(simulate(ctl_lqr(sched), PATH_DEFAULT))
        print(f'{q1:>8.1f}{q3:>10.1f}{R:>8.0f}{K[0]:>9.4f}{K[1]:>9.4f}'
              f'{K[2]:>9.4f}{K[3]:>9.4f}{r["max_e"]:>8.1f}cm{r["rms"]:>7.1f}cm'
              f'{r["sw_rate"]:>10.1f}d/s{r["max_ay"] / G:>8.2f}g')


# =====================================================================
# 实验 ⑤：轮胎模型对比 线性 / Fiala / Pacejka
# =====================================================================
def pacejka(alpha, Fz, mu=MU, B=None, C=1.30, E=0.97, Ca=None):
    """Magic Formula: Fy = D sin(C atan(B a - E(B a - atan(B a))))"""
    D = mu * Fz
    Ca = CF / 2.0 if Ca is None else Ca
    B = Ca / (C * D) if B is None else B            # 由 dFy/da|0 = B*C*D = Ca 定标
    x = B * alpha
    return D * np.sin(C * np.arctan(x - E * (x - np.arctan(x))))


def fiala(alpha, Fz, Ca, mu=MU):
    a_sl = 3.0 * mu * Fz / Ca
    a = np.clip(alpha, -a_sl, a_sl)
    z = 1.0 - Ca * np.abs(a) / (3.0 * mu * Fz)
    fy = -mu * Fz * (1 - z ** 3) * np.sign(a)
    return np.where(np.abs(alpha) >= a_sl, -mu * Fz * np.sign(alpha), fy) * -1.0


def exp5_tire():
    print(LINE)
    print('实验⑤ 轮胎模型对比（单前轮，Fz = m g lr / L / 2 = '
          f'{FZF / 2:.0f} N, mu = {MU}）')
    Ca = CF / 2.0
    B = Ca / (1.30 * MU * FZF / 2)
    print(f'Pacejka 参数标定: C=1.30, D=mu*Fz={MU * FZF / 2:.0f} N, '
          f'B=Ca/(C*D)={B:.3f}, E=0.97  (保证 dFy/dalpha|0 = Ca = {Ca / 1000:.0f} kN/rad)')
    print(f'Fiala 滑移角  a_sl = 3 mu Fz / Ca = '
          f'{np.rad2deg(3 * MU * FZF / 2 / Ca):.2f} deg')
    print(THIN)
    print(f"{'alpha[deg]':>11}{'线性 C*a[N]':>14}{'Fiala[N]':>12}"
          f"{'Pacejka[N]':>13}{'线性/Pacejka':>14}{'线性误差':>11}")
    for ad in (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15):
        a = np.deg2rad(ad)
        f_lin = Ca * a
        f_fia = float(fiala(np.array([a]), FZF / 2, Ca)[0])
        f_pac = pacejka(a, FZF / 2)
        print(f'{ad:>11.1f}{f_lin:>14.0f}{f_fia:>12.0f}{f_pac:>13.0f}'
              f'{f_lin / f_pac:>14.3f}{(f_lin / f_pac - 1) * 100:>10.1f}%')
    a_pk = np.deg2rad(np.linspace(0.1, 20, 4000))
    fy = pacejka(a_pk, FZF / 2)
    ip = int(np.argmax(fy))
    print(THIN)
    print(f'Pacejka 峰值 Fy = {fy[ip]:.0f} N @ alpha = {np.rad2deg(a_pk[ip]):.2f} deg '
          f'(= {fy[ip] / (MU * FZF / 2):.3f} mu Fz)；'
          f'20 deg 处跌落到 {pacejka(np.deg2rad(20), FZF / 2) / fy[ip] * 100:.1f}% 峰值')
    for thr in (0.05, 0.10, 0.20):
        idx = np.where(np.abs(Ca * a_pk / fy - 1) > thr)[0]
        print(f'  线性模型误差首次超过 {thr * 100:.0f}%: alpha = '
              f'{np.rad2deg(a_pk[idx[0]]):.2f} deg')


# =====================================================================
# 实验 ⑥：附着椭圆
# =====================================================================
def exp6_ellipse():
    print(LINE)
    print('实验⑥ 附着椭圆：纵向加减速吃掉多少侧向余量  (ax² + ay² <= (mu g)²)')
    print(THIN)
    for mu in (0.9, 0.5, 0.2):
        lim = mu * G
        print(f'mu={mu} (a_max={lim:.2f} m/s²={mu:.2f} g)  '
              f'干沥青/湿滑/冰雪；R=20 m 弯的极限车速 '
              f'{np.sqrt(lim * 20):.1f} m/s = {np.sqrt(lim * 20) * 3.6:.0f} km/h')
        print(f"  {'ax[m/s²]':>10}{'ax[g]':>8}{'可用ay[m/s²]':>14}{'可用ay[g]':>11}"
              f"{'余量占比':>10}{'R=50m 最高车速[km/h]':>22}")
        for ax in (0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -lim * 0.95):
            if abs(ax) >= lim:
                continue
            ay = np.sqrt(max(lim ** 2 - ax ** 2, 0.0))
            print(f'  {ax:>10.2f}{ax / G:>8.2f}{ay:>14.2f}{ay / G:>11.2f}'
                  f'{ay / lim * 100:>9.1f}%{np.sqrt(ay * 50) * 3.6:>22.1f}')


# =====================================================================
# 实验 ⑦：Smith 预估器 vs 状态外推（含模型失配）
# =====================================================================
def exp7_smith():
    print(LINE)
    print('实验⑦ 延时补偿方法对比（默认工况，纯延时 150 ms）')
    print(THIN)
    print(f"{'方法':<34}{'最大|e|':>10}{'变道max':>10}{'弯道稳态e':>11}"
          f"{'RMS':>9}{'盘峰值转速':>13}{'峰值ay':>9}")
    rows = [('无补偿 LQR', ctl_lqr()),
            ('状态外推(模型准确)', ctl_lqr(delay_comp=True)),
            ('Smith 预估器(模型准确)', SmithLQR(T_DELAY)),
            ('Smith 预估器(Cf 低估 20%)', SmithLQR(T_DELAY, cf_scale=0.8)),
            ('Smith 预估器(Cf 高估 20%)', SmithLQR(T_DELAY, cf_scale=1.2)),
            ('Smith 预估器(Cf/Cr 同降 30%)',
             SmithLQR(T_DELAY, cf_scale=0.7, cr_scale=0.7))]
    for label, ctl in rows:
        SAT_HIT[0] = False
        r = metrics(simulate(ctl, PATH_DEFAULT))
        print(f'{label:<34}{r["max_e"]:>8.1f}cm{r["lc_e"]:>8.1f}cm'
              f'{r["ss_e"]:>9.1f}cm{r["rms"]:>7.1f}cm'
              f'{r["sw_rate"]:>10.1f}d/s{r["max_ay"] / G:>8.2f}g')


# =====================================================================
# 实验 ⑧：MPC 的 Np / Nc 选型与求解耗时
# =====================================================================
def exp8_np_nc():
    print(LINE)
    print('实验⑧ MPC 预测/控制时域选型（dt_mpc=0.02 s，含延时补偿，默认工况）')
    print(THIN)
    print(f"{'Np':>5}{'Nc':>5}{'预瞄[s]':>10}{'H维度':>9}{'最大|e|':>10}{'RMS':>9}"
          f"{'弯道稳态e':>11}{'盘峰值转速':>13}{'QP均值[ms]':>13}{'QP p95[ms]':>13}")
    for Np_, Nc_ in ((10, 4), (15, 6), (25, 8), (40, 12), (60, 15)):
        mpc = MPC(Np=Np_, Nc=Nc_, delay_comp=True)
        SAT_HIT[0] = False
        r = metrics(simulate(mpc, PATH_DEFAULT))
        print(f'{Np_:>5}{Nc_:>5}{Np_ * 0.02:>10.2f}{f"{Nc_}x{Nc_}":>9}'
              f'{r["max_e"]:>8.1f}cm{r["rms"]:>7.1f}cm{r["ss_e"]:>9.1f}cm'
              f'{r["sw_rate"]:>10.1f}d/s{np.mean(mpc.solve_ms):>13.3f}'
              f'{np.percentile(mpc.solve_ms, 95):>13.3f}')
    print(THIN)
    print('注：耗时为 numpy/Python 实现的相对量级；量产 C++ + OSQP/qpOASES 热启动')
    print('    通常再快 5~20 倍，但结论（随 Nc 平方、随 Np 线性增长）一致。')


# =====================================================================
# 实验 ⑨：ACC 上下位分层与模式切换抖振
# =====================================================================
def exp9_acc():
    print(LINE)
    print('实验⑨ ACC 上下位分层：定速/跟车模式切换抖振（前车 t=4 s 切入，'
          '之后正弦扰动车速）')
    print(THIN)
    dt = 0.02
    tau_h, d0 = 1.6, 5.0
    v_set = 25.0

    def run(hyst, dwell):
        v, x = 22.0, 0.0
        xl, vl = 45.0, 22.0
        a_act, mode, t_mode = 0.0, 'CC', 0.0
        buf = deque([0.0] * int(round(0.10 / dt)))
        sw, jerks, gaps, a_prev = 0, [], [], 0.0
        for k in range(int(30.0 / dt)):
            t = k * dt
            if t >= 4.0:
                vl = 20.0 + 1.5 * np.sin(2 * np.pi * 0.25 * (t - 4.0))
                if abs(t - 4.0) < dt:
                    xl = x + 22.0
            xl += vl * dt
            gap = xl - x
            d_des = d0 + tau_h * v
            a_cc = 0.6 * (v_set - v)                        # 定速上位
            a_acc = 0.45 * (gap - d_des) + 1.1 * (vl - v)   # 跟车上位
            margin = hyst if mode == 'CC' else -hyst
            want = 'ACC' if a_acc < a_cc - margin else 'CC'
            if want != mode and (t - t_mode) >= dwell:
                mode, t_mode, sw = want, t, sw + 1
            a_des = a_acc if mode == 'ACC' else a_cc
            a_des = float(np.clip(a_des, AX_MIN, AX_MAX))
            buf.append(a_des)
            a_tgt = buf.popleft()
            a_act += (a_tgt - a_act) * dt / TAU_PWT         # 下位: MAP + 动力滞后
            v = max(0.0, v + a_act * dt)
            x += v * dt
            jerks.append(abs(a_act - a_prev) / dt)
            a_prev = a_act
            gaps.append(gap)
        return sw, max(jerks), min(gaps), np.mean(gaps)

    print(f"{'策略':<34}{'模式切换次数':>14}{'最大|jerk|[m/s³]':>18}"
          f"{'最小间距[m]':>13}{'平均间距[m]':>13}")
    for label, h, d in [('无迟滞、无最短驻留', 0.0, 0.0),
                        ('迟滞 0.15 m/s²', 0.15, 0.0),
                        ('迟滞 0.15 + 驻留 0.5 s', 0.15, 0.5),
                        ('迟滞 0.30 + 驻留 1.0 s', 0.30, 1.0)]:
        sw, jk, gmin, gavg = run(h, d)
        print(f'{label:<34}{sw:>14d}{jk:>18.2f}{gmin:>13.2f}{gavg:>13.2f}')


# =====================================================================
# 实验 ⑩：低速病态 1/vx
# =====================================================================
def exp10_lowspeed():
    print(LINE)
    print('实验⑩ 低速病态体检：误差动力学含 1/vx 项，vx->0 时条件数与增益爆炸')
    print(THIN)
    print(f"{'vx[m/s]':>9}{'km/h':>7}{'|A|_max':>12}{'cond(A_d)':>13}"
          f"{'k1':>9}{'k2':>9}{'k3':>9}{'k4':>9}{'运动学 kappa 增益':>18}")
    for vx in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        A, B = error_model(vx)
        Ad = np.eye(4) + A * DT + 0.5 * (A @ A) * DT ** 2
        try:
            K = lqr_gain(vx)
            ks = f'{K[0]:>9.4f}{K[1]:>9.4f}{K[2]:>9.4f}{K[3]:>9.4f}'
        except Exception as e:                              # noqa: BLE001
            ks = f'{"DARE 失败":>36}'
        print(f'{vx:>9.1f}{vx * 3.6:>7.1f}{np.abs(A).max():>12.1f}'
              f'{np.linalg.cond(Ad):>13.4f}{ks}{vx / L:>18.4f}')
    print(THIN)
    print('运动学模型 dpsi/dt = v tan(delta)/L 在 vx->0 时完全良态（增益 -> 0），')
    print('这就是"低于 v_min 切纯运动学 / 冻结横向控制"的数学依据。')


# =====================================================================
# 实验 ⑪：控制频率选型
# =====================================================================
def exp11_rate():
    print(LINE)
    print('实验⑪ 控制频率选型（LQR + 前馈 + 延时补偿，默认工况，'
          '控制器零阶保持降频）')
    print(THIN)
    print(f"{'控制频率[Hz]':>14}{'周期[ms]':>11}{'最大|e|':>10}{'RMS':>9}"
          f"{'弯道稳态e':>11}{'盘峰值转速':>13}{'峰值ay':>9}")
    for hz in (10, 20, 50, 100, 200):
        dt_c = 1.0 / hz
        SAT_HIT[0] = False
        r = metrics(simulate(ctl_lqr(delay_comp=True), PATH_DEFAULT,
                             dt_ctrl=max(dt_c, DT)))
        print(f'{hz:>14}{1000 / hz:>11.1f}{r["max_e"]:>8.1f}cm{r["rms"]:>7.1f}cm'
              f'{r["ss_e"]:>9.1f}cm{r["sw_rate"]:>10.1f}d/s{r["max_ay"] / G:>8.2f}g')
    print(THIN)
    print('注：本仿真物理积分步长恒为 10 ms，只降控制器触发频率，'
          '故 200 Hz 与 100 Hz 同结果。')


# =====================================================================
# 实验 ⑫：ISO 3888-2 双移线合格判据
# =====================================================================
def exp12_iso3888():
    print(LINE)
    print('实验⑫ ISO 3888 双移线简化判据（默认工况变道段，车道半宽 1.75 m，'
          '车宽 1.85 m）')
    print(THIN)
    half_lane, w_car = 1.75, 1.85
    margin_allow = half_lane - w_car / 2                    # 0.825 m
    print(f'允许最大横向偏差 = 半车道 {half_lane} - 半车宽 {w_car / 2:.3f} = '
          f'{margin_allow:.3f} m = {margin_allow * 100:.1f} cm')
    print(f"{'控制器':<24}{'变道段max|e|[cm]':>18}{'裕度[cm]':>11}"
          f"{'峰值ay[g]':>11}{'盘转速[deg/s]':>15}{'判定':>8}")
    mpc = MPC(delay_comp=True)
    for label, ctl in [('PurePursuit ld=0.3v+3', ctl_pp(0.3, 3.0)),
                       ('PurePursuit ld=4m 固定', ctl_pp(0.0, 4.0)),
                       ('Stanley k=2.0', ctl_stanley()),
                       ('LQR + 曲率前馈', ctl_lqr()),
                       ('LQR + 前馈 + 延时补偿', ctl_lqr(delay_comp=True)),
                       ('MPC + 延时补偿', mpc)]:
        SAT_HIT[0] = False
        if isinstance(ctl, MPC):
            ctl.reset()
        r = metrics(simulate(ctl, PATH_DEFAULT))
        ok = (r['lc_e'] < margin_allow * 100 and r['max_ay'] / G < 0.8
              and r['sw_rate'] < 560.0 and not SAT_HIT[0])
        why = 'PASS' if ok else ('转速超限' if r['sw_rate'] >= 560 else
                                 ('撞桩' if r['lc_e'] >= margin_allow * 100
                                  else '附着超限'))
        print(f'{label:<24}{r["lc_e"]:>18.1f}'
              f'{margin_allow * 100 - r["lc_e"]:>11.1f}{r["max_ay"] / G:>11.2f}'
              f'{r["sw_rate"]:>15.1f}{why:>8}')


# =====================================================================
# 实验 ⑬：稳态前轮转角 / 增益调度 / 特征车速解析核对
# =====================================================================
def exp13_analytic():
    print(LINE)
    print('实验⑬ 解析量核对（供正文推导对拍）')
    print(THIN)
    Wf, Wr = M * G * LR / L, M * G * LF / L
    print(f'轴荷    Wf = m g lr / L = {Wf:.1f} N ({Wf / (M * G) * 100:.1f}%), '
          f'Wr = {Wr:.1f} N ({Wr / (M * G) * 100:.1f}%)')
    print(f'不足转向梯度 K_us = Wf/Cf - Wr/Cr = {Wf / CF:.5f} - {Wr / CR:.5f} '
          f'= {K_US:.5f} rad/g = {np.rad2deg(K_US):.3f} deg/g')
    print(f'稳定性因数 K_s = K_us/g = {K_S:.6f} s²/m ; '
          f'另一等价写法 Kv = m/L (lr/Cf - lf/Cr) = {KV:.6f} s²/m '
          f'(相对差 {abs(KV - K_S) / K_S * 100:.4f}%)')
    print(f'特征车速 V_ch = sqrt(gL/K_us) = {V_CHAR:.3f} m/s = {V_CHAR * 3.6:.1f} km/h')
    print(f'临界车速 V_cr（若过度转向 K_us<0）不存在：K_us = {K_US:.5f} > 0 -> '
          f'全速域稳定')
    print(THIN)
    print(f"{'kappa[1/m]':>12}{'R[m]':>8}{'v[m/s]':>9}{'ay[m/s²]':>11}{'ay[g]':>8}"
          f"{'Ackermann[°]':>14}{'不足转向[°]':>13}{'稳态前轮[°]':>13}{'方向盘[°]':>12}")
    for kap, v in ((0.05, 10.0), (0.05, 12.0), (0.01, 25.0), (0.005, 33.3),
                   (0.002, 33.3), (0.20, 3.0)):
        ay = kap * v ** 2
        ack = L * kap
        us = K_US * ay / G
        d = ack + us
        print(f'{kap:>12.3f}{1 / kap:>8.0f}{v:>9.1f}{ay:>11.2f}{ay / G:>8.3f}'
              f'{np.rad2deg(ack):>14.3f}{np.rad2deg(us):>13.3f}'
              f'{np.rad2deg(d):>13.3f}{np.rad2deg(d) * STEER_RATIO:>12.1f}')
    print(THIN)
    print('横摆增益 r/delta = v / (L + K_s v²)，在 v = V_ch 处取极大：')
    print(f"{'v[m/s]':>9}{'km/h':>7}{'r/delta[1/s]':>15}{'侧向加速度增益 ay/delta[g/rad]':>32}")
    for v in (5, 10, 20, 30, V_CHAR, 35, 40):
        g_r = v / (L + K_S * v ** 2)
        print(f'{v:>9.2f}{v * 3.6:>7.1f}{g_r:>15.4f}{g_r * v / G:>32.4f}')


# =====================================================================
# 实验 ⑭：EPS 一阶滞后时间常数敏感性
# =====================================================================
def exp15_mpc_weights():
    print(LINE)
    print('实验⑮ MPC 权重扫描：Q=diag(q1, 1, q3, 1)，R_du 惩罚方向盘增量'
          '（Np=25, Nc=8, 默认工况 150 ms 延时）')
    print(THIN)
    print(f"{'q1(ey)':>8}{'q3(epsi)':>10}{'R_du':>10}"
          f"{'无补偿 max/RMS/转速/饱和':>34}{'延时补偿 max/RMS/弯道e/转速':>34}")
    for q1, q3, rdu in ((30, 60, 2e4), (30, 60, 1e5), (50, 60, 2e4),
                        (50, 60, 5e4), (50, 200, 5e4), (80, 60, 5e4),
                        (80, 200, 2e4), (120, 260, 3e5)):
        out = []
        for dc in (False, True):
            mpc = MPC(q=(q1, 1.0, q3, 1.0), r_du=rdu, delay_comp=dc)
            SAT_HIT[0] = False
            r = metrics(simulate(mpc, PATH_DEFAULT))
            r['sat'] = SAT_HIT[0]
            out.append(r)
        a, b = out
        c1 = (f'{a["max_e"]:8.1f}/{a["rms"]:6.1f}/{a["sw_rate"]:7.0f}/'
              f'{"是" if a["sat"] else "否"}')
        c2 = (f'{b["max_e"]:7.1f}/{b["rms"]:5.1f}/{b["ss_e"]:6.1f}/'
              f'{b["sw_rate"]:6.0f}')
        print(f'{q1:>8.0f}{q3:>10.0f}{rdu:>10.0f}{c1:>34}{c2:>34}')
    print(THIN)
    print('单位: 最大|e| [cm] / RMS [cm] / 弯道稳态 e [cm] / 方向盘峰值转速 [deg/s]')
    print('规律: R_du 太小 -> 无补偿时高频抖并触发附着饱和；q1 越大越激进；')
    print('      q3(航向权重) 越大弯道稳态误差越大（航向被"按"住，横向偏置被容忍）。')


def exp14_eps():
    print(LINE)
    print('实验⑭ EPS 一阶滞后时间常数敏感性（LQR + 前馈 + 延时补偿，纯延时 150 ms）')
    print(THIN)
    print(f"{'tau_eps[ms]':>13}{'带宽[Hz]':>11}{'最大|e|':>10}{'RMS':>9}"
          f"{'盘峰值转速':>13}{'wc=6.2rad/s 处相位滞后[°]':>28}")
    for tau_ms in (20, 30, 50, 80, 120, 200):
        tau = tau_ms / 1000.0
        SAT_HIT[0] = False
        r = metrics(simulate(ctl_lqr(delay_comp=True), PATH_DEFAULT, tau_eps=tau))
        print(f'{tau_ms:>13}{1 / (2 * np.pi * tau):>11.2f}{r["max_e"]:>8.1f}cm'
              f'{r["rms"]:>7.1f}cm{r["sw_rate"]:>10.1f}d/s'
              f'{np.rad2deg(np.arctan(6.20 * tau)):>28.1f}')


if __name__ == '__main__':
    t_all = time.perf_counter()
    print(LINE)
    print('ch06 车辆控制 · 扩展验证套件')
    print(f'车辆: m={M:.0f} kg, Iz={IZ:.0f} kg·m², L={L:.2f} m (lf={LF}, lr={LR}), '
          f'Cf={CF / 1000:.0f} kN/rad, Cr={CR / 1000:.0f} kN/rad, mu={MU}')
    print(f'执行链: 纯延时 {T_DELAY * 1000:.0f} ms, EPS tau {TAU_EPS * 1000:.0f} ms, '
          f'速率上限 {np.rad2deg(RATE_MAX) * STEER_RATIO:.0f} deg/s(方向盘), '
          f'转向比 {STEER_RATIO}, 控制 {1 / DT:.0f} Hz')
    exp1_controllers()
    exp2_speed_sweep()
    exp3_delay()
    exp4_qr_sweep()
    exp5_tire()
    exp6_ellipse()
    exp7_smith()
    exp8_np_nc()
    exp9_acc()
    exp10_lowspeed()
    exp11_rate()
    exp12_iso3888()
    exp13_analytic()
    exp14_eps()
    exp15_mpc_weights()
    print(LINE)
    print(f'全部实验完成，总耗时 {time.perf_counter() - t_all:.1f} s')
    print(LINE)
