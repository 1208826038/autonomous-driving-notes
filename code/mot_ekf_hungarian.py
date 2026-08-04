"""多目标跟踪最小可运行实现：CV-EKF + 马氏距离门控 + 匈牙利关联 + 航迹生命周期管理。

场景：120 帧 @ 20 Hz（雷达周期 50 ms），ego 系 x 前 y 左，3 个真值目标 + 泊松杂波
  GT1 正前方前车  x: 42→33 m, y=0.0      匀速接近，CV 模型完全匹配
  GT2 右邻道车    x: 18→24 m, y:-3.6→0.0 t∈[1.5,4.0]s 抢道切入，横向峰值加速度 3.6 m/s²
  GT3 对向来车    x: 68→35 m, y=+3.6     t∈[2.0,3.6]s 被大车遮挡，检测消失 32 帧后重现
消融两个工程开关：航迹去重合并 merge、机动自适应过程噪声 adapt（穷人版 IMM）。
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

DT, N_FRAMES = 0.05, 120                       # 雷达周期 50 ms
SIGMA_A = 1.2                                  # 过程噪声：加速度标准差 m/s^2
R_MEAS = np.diag([0.25 ** 2, 0.40 ** 2])       # 观测噪声：纵向 25 cm / 横向 40 cm
GATE_CHI2 = 9.21                               # chi-square, 2 自由度, 99% 分位
NIS_ALERT = 4.61                               # chi-square, 2 自由度, 90% 分位：机动报警线
PD, CLUTTER_RATE = 0.95, 1.2                   # 检测概率 / 每帧杂波数泊松均值
M_OF_N, MAX_COAST_CONF, MAX_COAST_TENT = (3, 5), 5, 2
MERGE_DIST, Q_BOOST = 1.8, 9.0                 # 航迹去重半径 m / 机动时 Q 放大倍数

F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
G = np.array([[DT ** 2 / 2, 0], [0, DT ** 2 / 2], [DT, 0], [0, DT]])
Q0 = G @ (np.eye(2) * SIGMA_A ** 2) @ G.T      # 离散白噪声加速度模型 Q = G diag(σa²) Gᵀ
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


def ground_truth(t):
    """真值 {gt_id: (x, y, visible)}，ego 系 x 前 y 左，单位 m。"""
    s = min(max((t - 1.5) / 2.5, 0.0), 1.0)                       # 切入进度 0→1
    y2 = -3.6 + 3.6 * (s - np.sin(2 * np.pi * s) / (2 * np.pi))   # 余弦加加速度换道
    return {1: (42.0 - 1.5 * t, 0.0, True),
            2: (18.0 + 1.0 * t, y2, True),
            3: (68.0 - 5.5 * t, 3.6, not (2.0 <= t < 3.6))}


def simulate_frame(t, rng):
    dets = [[x + rng.normal(0, 0.25), y + rng.normal(0, 0.40)]
            for x, y, vis in ground_truth(t).values() if vis and rng.random() < PD]
    dets += [[rng.uniform(5.0, 80.0), rng.uniform(-9.0, 9.0)]
             for _ in range(rng.poisson(CLUTTER_RATE))]
    rng.shuffle(dets)
    return np.array(dets).reshape(-1, 2)


class Track:
    _next_id = 1

    def __init__(self, z):
        self.id, Track._next_id = Track._next_id, Track._next_id + 1
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([0.5, 0.5, 100.0, 100.0])   # 单次观测定不了速度，速度先验极宽
        self.state, self.hits, self.coast, self.hist = "TENTATIVE", 1, 0, [1]
        self.q_scale, self.nis = 1.0, 0.0

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q_scale * Q0

    def innovation(self, z):
        return z - H @ self.x, H @ self.P @ H.T + R_MEAS

    def update(self, z, adapt):
        nu, S = self.innovation(z)
        Sinv = np.linalg.inv(S)
        self.nis = float(nu @ Sinv @ nu)                   # 归一化新息平方 NIS
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ nu
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_MEAS @ K.T   # Joseph 形式，保对称正定
        if adapt:   # NIS 连续超 90% 分位 ⇒ 判定机动，临时放大 Q（穷人版 IMM）
            self.q_scale = Q_BOOST if self.nis > NIS_ALERT else max(1.0, self.q_scale * 0.5)
        self.hits, self.coast = self.hits + 1, 0
        self.hist.append(1)

    def miss(self):
        self.coast += 1
        self.hist.append(0)

    def manage(self):
        recent = sum(self.hist[-M_OF_N[1]:])
        if self.state == "TENTATIVE":
            if recent >= M_OF_N[0]:
                self.state = "CONFIRMED"
            elif self.coast >= MAX_COAST_TENT:
                self.state = "DELETED"
        elif self.state == "CONFIRMED" and self.coast > 0:
            self.state = "COASTED"
        elif self.state == "COASTED":
            self.state = "CONFIRMED" if self.coast == 0 else (
                "DELETED" if self.coast >= MAX_COAST_CONF else "COASTED")


def associate(tracks, dets):
    """代价 = 马氏距离平方，门外置 1e6；匈牙利求全局最优二分匹配 O(n³)。"""
    if not tracks or len(dets) == 0:
        return [], list(range(len(dets)))
    C = np.full((len(tracks), len(dets)), 1e6)
    for i, trk in enumerate(tracks):
        for j, z in enumerate(dets):
            nu, S = trk.innovation(z)
            d2 = float(nu @ np.linalg.inv(S) @ nu)
            if d2 <= GATE_CHI2:                            # 卡方门控
                C[i, j] = d2
    ri, ci = linear_sum_assignment(C)
    pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e6]
    hit = {j for _, j in pairs}
    return pairs, [j for j in range(len(dets)) if j not in hit]


def dedup(tracks):
    """航迹去重：同一物理目标被拆成两条航迹时，保留命中数多的那条。"""
    keep = []
    for tr in sorted(tracks, key=lambda t: (-t.hits, t.id)):
        if all(np.hypot(tr.x[0] - o.x[0], tr.x[1] - o.x[1]) > MERGE_DIST for o in keep):
            keep.append(tr)
    return keep


def run(seed, merge=True, adapt=True, verbose=False):
    rng = np.random.default_rng(seed)
    Track._next_id = 1
    tracks, ms, gt2trk = [], [], {}
    sq = n_ok = n_fn = n_fp = n_gt = ids = 0
    nis_cv, nis_mnv = [], []

    for k in range(N_FRAMES):
        t = k * DT
        dets = simulate_frame(t, rng)
        for trk in tracks:
            trk.predict()
        t0 = time.perf_counter()
        pairs, unmatched = associate(tracks, dets)
        ms.append((time.perf_counter() - t0) * 1e3)

        hit_idx = {i for i, _ in pairs}
        for i, j in pairs:
            tracks[i].update(dets[j], adapt)
        for i, trk in enumerate(tracks):
            if i not in hit_idx:
                trk.miss()
        for trk in tracks:
            trk.manage()
        tracks += [Track(dets[j]) for j in unmatched]
        tracks = [tr for tr in tracks if tr.state != "DELETED"]
        if merge:
            tracks = dedup(tracks)

        # ---- 评测：可见真值与已确认航迹做贪心最近邻配对（门限 2.5 m）----
        conf = [tr for tr in tracks if tr.state in ("CONFIRMED", "COASTED")]
        gt = {g: v for g, v in ground_truth(t).items() if v[2]}
        n_gt += len(gt)
        used = set()
        for g, (gx, gy, _) in gt.items():
            best, bd = None, 2.5
            for tr in conf:
                d = np.hypot(tr.x[0] - gx, tr.x[1] - gy)
                if tr.id not in used and d < bd:
                    best, bd = tr, d
            if best is None:
                n_fn += 1
                continue
            used.add(best.id)
            sq, n_ok = sq + bd ** 2, n_ok + 1
            if g == 2 and best.nis > 0:
                (nis_mnv if 1.5 <= t < 4.0 else nis_cv).append(best.nis)
            if g in gt2trk and gt2trk[g] != best.id:
                ids += 1
                if verbose:
                    print(f"  [帧{k:3d} t={t:.2f}s] !! ID SWITCH  GT{g}: T{gt2trk[g]} -> T{best.id}")
            gt2trk[g] = best.id
        n_fp += len(conf) - len(used)

        if verbose and (k % 12 == 0 or k in (30, 40, 41, 72, 73)):
            st = " ".join(f"T{tr.id}:{tr.state[:4]}" for tr in tracks[:6])
            print(f"帧{k:3d} t={t:4.2f}s | 检测{len(dets):2d} 匹配{len(pairs):2d} "
                  f"新建{len(unmatched):2d} 航迹{len(tracks):2d} | 关联{ms[-1]:5.3f}ms | {st}")

    return dict(ids=ids, fn=n_fn, fp=n_fp, gt=n_gt, ms=float(np.mean(ms)),
                ms_max=float(np.max(ms)), rmse=float(np.sqrt(sq / max(n_ok, 1))),
                mota=1 - (n_fn + n_fp + ids) / n_gt, nis_cv=float(np.mean(nis_cv)),
                nis_mnv=float(np.mean(nis_mnv)), nis_max=float(np.max(nis_mnv)),
                alive=[f"T{tr.id}({tr.state})" for tr in tracks])


def main():
    print("=" * 84)
    print("多目标跟踪仿真  dt=50ms  帧数=120  门限 chi2(2,0.99)=9.21  确认逻辑 3-of-5")
    print("=" * 84)
    r = run(20260804, merge=True, adapt=True, verbose=True)
    print("-" * 84)
    print(f"关联耗时 均值/最大 : {r['ms']:.3f} ms / {r['ms_max']:.3f} ms")
    print(f"ID 切换次数 (IDS)  : {r['ids']}")
    print(f"漏跟 FN / 虚跟 FP  : {r['fn']} / {r['fp']}   (GT 总量 {r['gt']})")
    print(f"位置 RMSE          : {r['rmse']:.3f} m")
    print(f"MOTA               : {r['mota']:.4f}")
    print(f"GT2 NIS 匀速段均值 : {r['nis_cv']:.3f}   (理论 E[NIS]=2.0)")
    print(f"GT2 NIS 机动段均值 : {r['nis_mnv']:.3f}   峰值 {r['nis_max']:.3f}")
    print(f"存活航迹           : {r['alive']}")

    print("=" * 84)
    print("消融实验：5 个随机种子平均")
    print(f"{'配置':<26}{'IDS':>6}{'FN':>6}{'FP':>6}{'RMSE(m)':>10}{'MOTA':>9}{'关联ms':>9}")
    print("-" * 84)
    for name, mg, ad in [("① 基线（都不开）", False, False), ("② +航迹去重", True, False),
                         ("③ +自适应Q", False, True), ("④ 两者全开", True, True)]:
        a = [run(s, mg, ad) for s in (20260804, 7, 42, 1234, 99)]
        print(f"{name:<24}{np.mean([x['ids'] for x in a]):6.1f}"
              f"{np.mean([x['fn'] for x in a]):6.1f}{np.mean([x['fp'] for x in a]):6.1f}"
              f"{np.mean([x['rmse'] for x in a]):10.3f}{np.mean([x['mota'] for x in a]):9.4f}"
              f"{np.mean([x['ms'] for x in a]):9.3f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
