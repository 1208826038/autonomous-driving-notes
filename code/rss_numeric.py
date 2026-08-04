"""RSS 纵向/横向安全距离数值算例 + 间隙接受 logit 曲线，为 ch04 提供真实数字。"""
import math


def rss_long(vr, vf, rho=0.6, a_max=3.0, b_min=4.0, b_max=8.0):
    """同向行驶纵向最小安全距离（后车 vr 追前车 vf）。"""
    t1 = vr * rho                                  # 响应期匀速位移
    t2 = 0.5 * a_max * rho ** 2                    # 响应期最坏加速的额外位移
    t3 = (vr + rho * a_max) ** 2 / (2 * b_min)     # 后车以 b_min 刹停的距离
    t4 = vf ** 2 / (2 * b_max)                     # 前车以 b_max 刹停的距离
    return max(0.0, t1 + t2 + t3 - t4), (t1, t2, t3, t4)


def rss_long_opposite(v1, v2, rho=0.6, a_max=3.0, b_min=4.0):
    """对向行驶纵向最小安全距离（双方都要刹停）。"""
    d1 = (v1 + rho * a_max) ** 2 / (2 * b_min)
    d2 = (v2 + rho * a_max) ** 2 / (2 * b_min)
    return v1 * rho + 0.5 * a_max * rho ** 2 + d1 + v2 * rho + 0.5 * a_max * rho ** 2 + d2


def rss_lat(v1, v2, rho=0.6, a_lat=1.0, b_lat=1.5, mu=0.4):
    """横向最小安全距离：两车横向相向靠拢，mu 为不可约横向波动裕度。"""
    def half(v):
        v_end = v + rho * a_lat
        return rho * (v + v_end) / 2 + v_end ** 2 / (2 * b_lat)
    return mu + max(0.0, half(v1) + half(v2))


def logit_gap(t, t_crit=5.0, beta=0.7):
    return 1.0 / (1.0 + math.exp(-(t - t_crit) / beta))


if __name__ == "__main__":
    print("=== RSS 纵向（同向跟车）ρ=0.6s a_max=3.0 b_min=4.0 b_max=8.0 ===")
    print(f"{'vr(m/s)':>8}{'vf(m/s)':>8}{'vρ':>8}{'½aρ²':>8}{'后车刹停':>10}"
          f"{'-前车刹停':>11}{'d_min(m)':>10}{'时距(s)':>9}")
    for vr, vf in [(8.3, 8.3), (13.9, 13.9), (16.7, 16.7), (22.2, 22.2),
                   (27.8, 27.8), (33.3, 33.3), (22.2, 0.0), (33.3, 25.0)]:
        d, (t1, t2, t3, t4) = rss_long(vr, vf)
        thw = d / vr if vr > 0 else 0
        print(f"{vr:8.1f}{vf:8.1f}{t1:8.2f}{t2:8.2f}{t3:10.2f}{-t4:11.2f}{d:10.2f}{thw:9.2f}")

    print("\n=== RSS 纵向（对向）ρ=0.6s a_max=3.0 b_min=4.0 ===")
    for v1, v2 in [(13.9, 13.9), (16.7, 16.7), (8.3, 13.9), (22.2, 22.2)]:
        print(f"  v1={v1:5.1f} v2={v2:5.1f} -> d_min={rss_long_opposite(v1, v2):7.2f} m")

    print("\n=== RSS 横向 ρ=0.6s a_lat=1.0 b_lat=1.5 μ=0.4m ===")
    for v1, v2 in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.5), (1.5, 1.0), (2.0, 0.0)]:
        print(f"  v_lat1={v1:4.1f} v_lat2={v2:4.1f} -> d_lat_min={rss_lat(v1, v2):5.2f} m")

    print("\n=== 间隙接受 logit  P=σ((t-t_crit)/β)，t_crit=5.0s β=0.7 ===")
    print("   t(s): " + " ".join(f"{t:5.1f}" for t in [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]))
    print("   P   : " + " ".join(f"{logit_gap(t):5.2f}" for t in
                                 [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]))

    print("\n=== 自车清空冲突区时间 t_clear（a=1.8m/s² 延迟0.25s）===")
    for L in [10, 12, 14, 16, 18]:
        for v0 in [0.0, 1.0, 2.0, 3.0]:
            t = (-v0 + math.sqrt(v0 ** 2 + 2 * 1.8 * L)) / 1.8 + 0.25
            print(f"  L={L:3d}m v0={v0:3.1f}m/s -> t_clear={t:5.2f}s", end="")
        print()
