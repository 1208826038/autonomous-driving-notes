"""无保护左转决策器：TTC + 间隙接受 logit + Stackelberg 让行博弈 -> GO / CREEP / WAIT。"""
import math
from dataclasses import dataclass

SQ, INF = math.sqrt, float("inf")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def fs(x, w=6):
    """把可能为 inf 的时间量格式化成定宽字符串。"""
    return f"{'  >99':>{w}}" if x == INF else f"{x:{w}.2f}"


@dataclass
class Veh:
    name: str
    d: float          # 到冲突点距离 m
    v: float          # 车速 m/s
    a: float = 0.0    # 观测纵向加速度 m/s^2（正=加速抢行）
    svo: float = 0.5  # 社会价值取向 0=激进自利 1=利他礼让


@dataclass
class Params:
    t_crit: float = 5.0       # 临界间隙 s（HCM 无保护左转经验区间 4.1~5.5）
    beta: float = 0.7         # 间隙接受 logit 斜率
    margin: float = 1.5       # GO 所需时间裕度 s
    a_ego: float = 1.8        # 自车起步加速度 m/s^2
    l_cross: float = 14.0     # 需清空的冲突区长度 m（含车长）
    latency: float = 0.25     # 感知-决策-执行链路延迟 s
    s_buf: float = 5.0        # 冲突点纵向安全缓冲 m
    a_worst: float = 1.0      # 对向车"最坏加速"假设 m/s^2
    a_comf: float = 1.5       # 对向车舒适减速上限 m/s^2
    a_max_brake: float = 6.0  # 对向车物理最大减速 m/s^2
    p_go: float = 0.80        # 信任对方让行才 GO 的概率门限
    p_creep: float = 0.55     # 蠕行试探的概率门限
    wait_timeout: float = 8.0 # frozen-robot 解锁时间 s


def ego_clear_time(p: Params, v0=0.0):
    """自车从 v0 加速清空冲突区所需时间（含链路延迟）。"""
    return (-v0 + SQ(v0 ** 2 + 2 * p.a_ego * p.l_cross)) / p.a_ego + p.latency


def arrival_time(d, v, a):
    """解 d = v t + 0.5 a t^2 求到达冲突点时间；减速到停仍够不着则为 inf。"""
    if d <= 0:
        return 0.0
    if abs(a) < 1e-3:
        return d / v if v > 0.1 else INF
    disc = v * v + 2 * a * d
    if disc < 0:
        return INF
    t = (-v + SQ(disc)) / a
    return t if t > 0 else INF


def required_decel(o: Veh, t_clear, p: Params, extra_a=0.0):
    """对向车为不侵入冲突缓冲区，在 t_clear 内需附加的减速度（<=0 表示无需减速）。"""
    base = 2.0 * (o.v * t_clear - (o.d - p.s_buf)) / (t_clear ** 2)
    return base + o.a + extra_a


def yield_prob(o: Veh, a_req, p: Params):
    """Stackelberg：自车为领导者先动，估计跟随者选择"让行"的概率。"""
    if a_req <= 0:
        return 1.0
    if a_req > p.a_max_brake:
        return 0.0
    a_tol = 1.0 + 2.5 * o.svo            # SVO 越高越肯为别人踩刹车
    if o.a > 0.3:                        # 观测到正在加速 = 抢行意图
        a_tol -= 1.0
    return sigmoid((a_tol - a_req) / 0.6)


def decide(ego_v, oncoming, p: Params, wait_time=0.0):
    tc = ego_clear_time(p, ego_v)
    rep, crit = {"t_clear": round(tc, 2), "rows": []}, None
    for o in oncoming:
        t_nom = arrival_time(o.d, o.v, max(-3.0, min(2.0, o.a)))
        a_nom = required_decel(o, tc, p)
        a_wst = required_decel(o, tc, p, p.a_worst)
        py = yield_prob(o, a_nom, p)
        rep["rows"].append(
            f"{o.name}: d={o.d:5.1f}m v={o.v:4.1f}m/s a={o.a:+4.1f} svo={o.svo:.2f} | "
            f"TTC={o.d / max(o.v, .1):5.2f}s t_arr={fs(t_nom)}s "
            f"a_req={a_nom:+5.2f} a_req_worst={a_wst:+5.2f} P(让)={py:.2f}")
        key = (a_wst, a_nom)
        if crit is None or key > crit[0]:      # 取"最难让"的那辆为临界车
            crit = (key, o, t_nom, a_nom, a_wst, py)
    if crit is None:
        return "GO", "无冲突车流", rep
    _, o, t_nom, a_nom, a_wst, py = crit
    pet = t_nom - tc
    pa = sigmoid((t_nom - p.t_crit) / p.beta)
    rep.update(crit=o.name, pet=fs(pet, 1).strip(), p_accept=round(pa, 3),
               a_req=round(a_nom, 2), a_req_worst=round(a_wst, 2), p_yield=round(py, 3))

    if a_wst <= 0:
        return "GO", (f"即使 {o.name} 以 {p.a_worst}m/s² 最坏加速，也进不了 "
                      f"{p.s_buf}m 缓冲区（a_req_worst={a_wst:+.2f}），间隙绝对充分"), rep
    if a_wst > p.a_max_brake:
        return "WAIT", (f"{o.name} 让行需 {a_wst:.2f}m/s² > 物理极限 "
                        f"{p.a_max_brake}m/s²，硬安全门不通过"), rep
    if a_nom <= 0:
        if pet >= p.margin or pa >= 0.5:
            return "GO", (f"名义轨迹下 {o.name} 无需减速(a_req={a_nom:+.2f})，"
                          f"PET={fs(pet, 1).strip()}s、P(接受)={pa:.2f} 达标"), rep
        return "CREEP", (f"名义安全但裕度不足(PET={pet:.2f}s<{p.margin}s，"
                         f"P(接受)={pa:.2f})，蠕行前压压缩 t_clear 再判"), rep
    if a_nom <= p.a_comf and py >= p.p_go:
        return "GO", (f"仅需 {o.name} 舒适减速 {a_nom:.2f}m/s²，"
                      f"博弈估计其让行概率 {py:.2f} ≥ {p.p_go}"), rep
    if py >= p.p_creep:
        return "CREEP", (f"需 {o.name} 减速 {a_nom:.2f}m/s² 但 P(让)={py:.2f} 仅中等，"
                         f"蠕行释放意图、观察其响应"), rep
    if wait_time > p.wait_timeout and a_nom <= 0.6 * p.a_max_brake:
        return "CREEP", (f"已等待 {wait_time:.1f}s > {p.wait_timeout}s，"
                         f"主动蠕行打破 frozen-robot 僵局"), rep
    return "WAIT", (f"需 {o.name} 减速 {a_nom:.2f}m/s² 而 P(让)={py:.2f} 偏低"
                    f"（已等 {wait_time:.1f}s），保持等待"), rep


if __name__ == "__main__":
    P = Params()
    cases = [
        ("S1 大间隙：单车 130m @50km/h",            0.0, [Veh("A", 130, 13.9)], 0.0),
        ("S2 临界间隙：单车 68m @50km/h",           0.0, [Veh("A", 68, 13.9)], 0.0),
        ("S3 小间隙：单车 30m @60km/h",             0.0, [Veh("A", 30, 16.7)], 0.0),
        ("S4 对方加速抢行：50m @12m/s a=+1.5 激进", 0.0, [Veh("A", 50, 12.0, +1.5, 0.15)], 0.0),
        ("S5 对方减速礼让：45m @11m/s a=-1.5 礼貌", 0.0, [Veh("A", 45, 11.0, -1.5, 0.85)], 0.0),
        ("S6 车队三车连续 40/72/108m @13m/s",       0.0, [Veh("A", 40, 13.0), Veh("B", 72, 13.0),
                                                          Veh("C", 108, 13.0)], 0.0),
        ("S7 僵持解锁：45m @12.5m/s 激进 已等9.5s", 0.0, [Veh("A", 45, 12.5, 0.0, 0.10)], 9.5),
        ("S8 已蠕行至 2m/s，同 S2 间隙",            2.0, [Veh("A", 68, 13.9)], 3.0),
    ]
    for title, v0, onc, wt in cases:
        act, why, rep = decide(v0, onc, P, wt)
        print(f"\n=== {title} ===")
        for r in rep["rows"]:
            print("   " + r)
        print(f"   t_clear={rep['t_clear']}s  临界车={rep.get('crit')}  "
              f"PET={rep.get('pet')}s  P(接受)={rep.get('p_accept')}  "
              f"P(让)={rep.get('p_yield')}")
        print(f"   >>> {act}  |  {why}")
