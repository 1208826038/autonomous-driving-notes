# -*- coding: utf-8 -*-
"""
第八章《系统架构与中间件》配套定量分析脚本
=========================================================
三个互相独立又互相印证的分析：

  ① 端到端时延蒙特卡洛
     给定「曝光 → 传输 → ISP → 推理 → 融合 → 预测 → 规划 → 控制 → CAN → 执行器」
     各阶段的耗时分布（基线 + Gamma 抖动 + 低概率长尾离群），
     用蒙特卡洛仿真出整链路时延的 P50/P95/P99/P99.9 与各截止期下的超时率。
     对比三种系统配置：通用 Linux / PREEMPT_RT 化 / RT + 零拷贝 + INT8。

  ② 时间不同步 → 融合位置错位
     Δt ∈ {0,5,10,20,40} ms 与相对速度 v_rel 的笛卡尔积，
     算出纵向错位；再算转弯工况下的横向错位；
     与典型数据关联门限（1.0 m）比对，判断是否会导致关联失败。

  ③ 多传感器原始数据带宽估算
     逐路算原始/压缩带宽，汇总整车，
     再与 CAN / CAN FD / 100BASE-T1 / 1000BASE-T1 / Multi-Gig 的
     「有效可用吞吐」（含协议开销与设计负载率）比对，给出够用/不够用判定。

运行：python arch_latency_sync_bandwidth.py
依赖：numpy
"""

import unicodedata

import numpy as np

SEED = 20260804
rng = np.random.default_rng(SEED)
N_MC = 400_000  # 蒙特卡洛样本数


# --------------------------------------------------------------------------
# 通用：CJK 宽度对齐的表格打印
# --------------------------------------------------------------------------
def dw(s):
    """显示宽度：CJK 全角字符算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, n, align="l"):
    s = str(s)
    k = max(0, n - dw(s))
    if align == "l":
        return s + " " * k
    if align == "r":
        return " " * k + s
    return " " * (k // 2) + s + " " * (k - k // 2)


def table(headers, rows, aligns=None, indent=""):
    cols = len(headers)
    aligns = aligns or ["l"] * cols
    widths = [dw(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], dw(c))
    line = indent + "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line]
    out.append(
        indent + "| " + " | ".join(pad(h, widths[i], "c") for i, h in enumerate(headers)) + " |"
    )
    out.append(indent + "+" + "+".join("=" * (w + 2) for w in widths) + "+")
    for r in rows:
        out.append(
            indent
            + "| "
            + " | ".join(pad(c, widths[i], aligns[i]) for i, c in enumerate(r))
            + " |"
        )
    out.append(line)
    print("\n".join(out))


def banner(title):
    print()
    print("=" * 96)
    print("  " + title)
    print("=" * 96)


def sub(title):
    print()
    print("-" * 96)
    print("  " + title)
    print("-" * 96)


# ==========================================================================
# ① 端到端时延蒙特卡洛
# ==========================================================================
# 每个阶段建模为：base(确定性下界) + Gamma(k, θ)(常规抖动，右偏)
#                 + 以概率 p 叠加 Exp(scale) 的长尾离群（调度抢占 / 缺页 / GC / GPU 争用）
#
# 字段：(阶段名, base_ms, gamma_k, gamma_theta, p_outlier, outlier_scale_ms)
STAGES_BASE = [
    ("曝光积分 + 逐行读出",       11.00, 3.0, 0.17, 0.0010,  2.0),
    ("串行器传输 GMSL2/MIPI",      1.80, 2.0, 0.10, 0.0020,  1.0),
    ("ISP 流水线",                 3.00, 2.0, 0.20, 0.0050,  3.0),
    ("送显存 + 预处理(resize)",    2.50, 2.0, 0.30, 0.0120,  6.0),
    ("感知 DNN 推理 (GPU/DLA)",   28.00, 3.0, 1.00, 0.0200, 15.0),
    ("检测后处理 + NMS",           3.50, 2.0, 0.50, 0.0100,  5.0),
    ("多传感器融合",               6.00, 2.5, 0.48, 0.0100,  6.0),
    ("轨迹预测",                   8.00, 2.5, 0.60, 0.0080,  6.0),
    ("行为决策 + 运动规划",       18.00, 2.5, 1.60, 0.0200, 20.0),
    ("横纵向控制解算",             2.00, 2.0, 0.15, 0.0050,  3.0),
    ("CAN FD 下发执行器",          1.20, 2.0, 0.15, 0.0100,  2.0),
    ("执行器响应(EPS 力矩建立)",  25.00, 3.0, 1.00, 0.0030, 10.0),
]

# 三种系统配置：
#   p_mul       : 离群概率倍率
#   out_mul     : 离群幅度倍率
#   base_over   : 对特定阶段的 base 覆盖（优化后）
SCENARIOS = [
    (
        "A 基线：通用 Linux + socket/DDS 有拷贝传输",
        1.00,
        1.00,
        {},
    ),
    (
        "B RT 化：PREEMPT_RT + CPU 隔离 + 绑核 + mlockall",
        0.12,
        0.35,
        {},
    ),
    (
        "C RT + 共享内存零拷贝 + INT8 量化 + DLA 卸载",
        0.10,
        0.30,
        {
            "送显存 + 预处理(resize)": 0.80,
            "感知 DNN 推理 (GPU/DLA)": 17.00,
            "多传感器融合": 4.00,
        },
    ),
]

DEADLINES = [150.0, 180.0, 200.0]


def simulate(p_mul, out_mul, base_over, n=N_MC, seed=SEED):
    """返回 (总时延数组, 每阶段时延数组字典)"""
    r = np.random.default_rng(seed)
    per_stage = {}
    total = np.zeros(n)
    for name, base, k, theta, p, oscale in STAGES_BASE:
        b = base_over.get(name, base)
        x = b + r.gamma(k, theta, n)
        hit = r.random(n) < (p * p_mul)
        x = x + hit * r.exponential(oscale * out_mul, n)
        per_stage[name] = x
        total += x
    return total, per_stage


def pct(a, q):
    return float(np.percentile(a, q))


banner("① 端到端时延蒙特卡洛仿真（摄像头曝光 → 执行器力矩建立）")
print(f"  样本数 N = {N_MC:,}    随机种子 = {SEED}")
print("  模型：每阶段 = 确定性基线 + Gamma 抖动 + 低概率指数长尾（抢占/缺页/GPU 争用）")

results = {}
for sname, pmul, omul, bover in SCENARIOS:
    tot, ps = simulate(pmul, omul, bover)
    results[sname] = (tot, ps)

sub("1-1  三种系统配置的端到端时延分布对比（单位 ms）")
rows = []
for sname, _, _, _ in SCENARIOS:
    tot, _ = results[sname]
    rows.append(
        [
            sname,
            f"{tot.mean():.1f}",
            f"{pct(tot,50):.1f}",
            f"{pct(tot,95):.1f}",
            f"{pct(tot,99):.1f}",
            f"{pct(tot,99.9):.1f}",
            f"{tot.max():.1f}",
            f"{pct(tot,99.9)-pct(tot,50):.1f}",
        ]
    )
table(
    ["系统配置", "均值", "P50", "P95", "P99", "P99.9", "最大", "P99.9-P50"],
    rows,
    ["l", "r", "r", "r", "r", "r", "r", "r"],
)

sub("1-2  各截止期（deadline）下的超时率")
rows = []
for sname, _, _, _ in SCENARIOS:
    tot, _ = results[sname]
    row = [sname]
    for d in DEADLINES:
        rate = float((tot > d).mean())
        if rate == 0.0:
            row.append("0 (<2.5e-6)")
        elif rate < 1e-4:
            row.append(f"{rate*1e6:.1f} ppm")
        else:
            row.append(f"{rate*100:.3f} %")
    rows.append(row)
table(
    ["系统配置"] + [f"deadline {int(d)} ms" for d in DEADLINES],
    rows,
    ["l", "r", "r", "r"],
)

sub("1-3  时延预算分解（配置 A 基线，各阶段 P50 / P99 与累计 P50）")
tot_a, ps_a = results[SCENARIOS[0][0]]
cum = 0.0
rows = []
for name, base, k, theta, p, oscale in STAGES_BASE:
    x = ps_a[name]
    p50 = pct(x, 50)
    p99 = pct(x, 99)
    cum += p50
    rows.append(
        [
            name,
            f"{base:.1f}",
            f"{p50:.2f}",
            f"{p99:.2f}",
            f"{p99-p50:.2f}",
            f"{cum:.1f}",
            f"{p50/pct(tot_a,50)*100:.1f}%",
        ]
    )
table(
    ["阶段", "基线", "P50", "P99", "抖动(P99-P50)", "累计P50", "占比"],
    rows,
    ["l", "r", "r", "r", "r", "r", "r"],
)

sub("1-4  抖动来源归因（配置 A：谁贡献了尾部时延）")
rows = []
for name, base, k, theta, p, oscale in STAGES_BASE:
    x = ps_a[name]
    jitter = pct(x, 99.9) - pct(x, 50)
    rows.append([name, f"{pct(x,99.9):.2f}", f"{jitter:.2f}", f"{p*100:.2f}%", f"{oscale:.1f}"])
rows.sort(key=lambda r: -float(r[2]))
table(
    ["阶段", "P99.9(ms)", "尾部抖动(ms)", "离群概率", "离群尺度(ms)"],
    rows,
    ["l", "r", "r", "r", "r"],
)

sub("1-5  一个反直觉结论：单阶段 99% 达标 ≠ 链路 99% 达标")
n_stage = len(STAGES_BASE)
for q in (0.99, 0.995, 0.999):
    print(
        f"  若每个阶段独立地以 {q*100:.1f}% 概率达标，{n_stage} 级串联链路整体达标率 "
        f"= {q**n_stage*100:.2f}%  →  超时率 {(1-q**n_stage)*100:.2f}%"
    )


# ==========================================================================
# ② 时间不同步 → 融合位置错位
# ==========================================================================
banner("② 传感器时间不同步导致的融合位置错位")

DTS_MS = [0, 1, 5, 10, 20, 40]
VRELS = [5, 10, 15, 20, 30, 50, 70]
GATE_M = 1.0  # 典型数据关联门限（马氏/欧氏），超过则关联失败

sub("2-1  纵向位置错位  Δx = v_rel · Δt   （单位 cm；★ 表示超过 1.0 m 关联门限）")
rows = []
for v in VRELS:
    row = [f"{v} m/s ({v*3.6:.0f} km/h)"]
    for dt in DTS_MS:
        e = v * dt / 1000.0
        flag = " ★" if e > GATE_M else ""
        row.append(f"{e*100:.0f}{flag}")
    rows.append(row)
table(
    ["相对速度"] + [f"Δt={d} ms" for d in DTS_MS],
    rows,
    ["l"] + ["r"] * len(DTS_MS),
)

sub("2-2  转弯工况横向错位  Δy = d · ω · Δt   （自车横摆角速度 ω，单位 cm）")
YAWS = [0.05, 0.10, 0.20, 0.30, 0.50]  # rad/s
DIST = 50.0  # 目标距离 m
rows = []
for w in YAWS:
    row = [f"{w:.2f} rad/s ({np.degrees(w):.1f}°/s)"]
    for dt in DTS_MS:
        e = DIST * w * dt / 1000.0
        flag = " ★" if e > GATE_M else ""
        row.append(f"{e*100:.1f}{flag}")
    rows.append(row)
table(
    [f"横摆角速度 (目标 {DIST:.0f} m 处)"] + [f"Δt={d} ms" for d in DTS_MS],
    rows,
    ["l"] + ["r"] * len(DTS_MS),
)

sub("2-3  不同步在下游放大：速度估计与 TTC 误差")
print("  设融合器用「摄像头位置 - 激光位置」跨源差分估计相对速度，帧间隔 T = 100 ms。")
print("  时间戳误差 Δt 会被 1/T 放大成速度误差：δv = v_rel · Δt / T")
rows = []
for dt in [1, 5, 10, 20, 40]:
    for v in [10, 30]:
        dv = v * (dt / 1000.0) / 0.100
        # TTC = R / v_rel ；取 R = 60 m
        R = 60.0
        ttc0 = R / v
        ttc1 = R / max(v + dv, 0.1)
        rows.append(
            [
                f"{dt}",
                f"{v}",
                f"{dv:.2f}",
                f"{dv/v*100:.1f}%",
                f"{ttc0:.2f}",
                f"{ttc1:.2f}",
                f"{(ttc1-ttc0)*1000:.0f}",
            ]
        )
table(
    ["Δt(ms)", "v_rel(m/s)", "速度误差δv(m/s)", "相对误差", "真TTC(s)", "误算TTC(s)", "TTC偏差(ms)"],
    rows,
    ["r"] * 7,
)

sub("2-4  同步精度等级与达标手段（错位按 v_rel = 30 m/s 折算）")
LEVELS = [
    ("软件收包时间戳（无同步）", 20e-3, "应用层 recv() 打戳，含协议栈+调度抖动"),
    ("NTP over Ethernet", 2e-3, "毫秒级，仅够做日志对齐"),
    ("SOME/IP 应用层对时", 1e-3, "依赖单向时延对称假设"),
    ("gPTP 802.1AS 软件时间戳", 100e-6, "网卡驱动层打戳，受中断延迟影响"),
    ("gPTP 802.1AS 硬件时间戳", 1e-6, "PHY/MAC 硬件打戳，车载主流"),
    ("GNSS PPS + 硬件触发", 100e-9, "秒脉冲直连传感器 trigger 引脚"),
]
def fmt_us(us):
    if us >= 1000:
        return f"{us/1000:,.0f} ms"
    if us >= 1:
        return f"{us:,.0f} μs"
    return f"{us*1000:,.0f} ns"


def fmt_mm(mm):
    if mm >= 1000:
        return f"{mm/1000:.2f} m"
    if mm >= 1:
        return f"{mm:.2f} mm"
    return f"{mm*1000:.1f} μm"


rows = []
for name, err, how in LEVELS:
    rows.append([name, fmt_us(err * 1e6), fmt_mm(30 * err * 1000), how])
table(["同步方案", "典型误差", "@30m/s 错位", "实现方式"], rows, ["l", "r", "r", "l"])


# ==========================================================================
# ③ 多传感器原始数据带宽估算
# ==========================================================================
banner("③ 多传感器原始数据带宽估算与总线选型判定")

# ---- 3-1 单路传感器带宽 ----
def cam_raw(w, h, bpp, fps):
    """返回 Mbps"""
    return w * h * bpp * fps / 1e6


SENSORS = [
    # (名称, 数量, 单路原始 Mbps, 单路压缩/结构化 Mbps, 备注)
    ("前视 8MP 主摄 (3840×2160 YUV422 30fps)", 1, cam_raw(3840, 2160, 16, 30), 25.0,
     "16 bpp；压缩为 H.265 25 Mbps"),
    ("侧前视 8MP ×2 (3840×2160 YUV422 30fps)", 2, cam_raw(3840, 2160, 16, 30), 25.0,
     "同上"),
    ("侧后视 3MP ×2 (2048×1536 YUV422 30fps)", 2, cam_raw(2048, 1536, 16, 30), 10.0,
     "H.265 10 Mbps"),
    ("后视 8MP ×1 (3840×2160 YUV422 30fps)", 1, cam_raw(3840, 2160, 16, 30), 25.0,
     "H.265 25 Mbps"),
    ("环视鱼眼 3MP ×4 (2048×1536 YUV422 30fps)", 4, cam_raw(2048, 1536, 16, 30), 10.0,
     "泊车用，可降帧"),
    ("舱内 DMS 2MP ×1 (1920×1080 YUV422 30fps)", 1, cam_raw(1920, 1080, 16, 30), 6.0,
     "座舱域"),
    ("128 线激光雷达 (2.6 M pts/s, 16 B/pt)", 1, 2.6e6 * 16 * 8 / 1e6, 2.6e6 * 16 * 8 / 1e6,
     "UDP 点云，不压缩"),
    ("转镜激光雷达 AT128 (1.53 M pts/s, 8 B/pt)", 1, 1.53e6 * 8 * 8 / 1e6, 1.53e6 * 8 * 8 / 1e6,
     "紧凑封包，≈100 Mbps"),
    ("4D 成像雷达 (5000 pts/帧, 20 Hz, 16 B/pt)", 1, 5000 * 20 * 16 * 8 / 1e6,
     5000 * 20 * 16 * 8 / 1e6, "点云模式"),
    ("角雷达 ×4 (目标列表 64 obj, 32 B, 20 Hz)", 4, 64 * 32 * 20 * 8 / 1e6,
     64 * 32 * 20 * 8 / 1e6, "结构化目标"),
    ("GNSS/IMU 组合导航 (100 Hz, 100 B)", 1, 100 * 100 * 8 / 1e6, 100 * 100 * 8 / 1e6,
     "CAN/串口即可"),
    ("超声波雷达 ×12 (20 Hz, 8 B)", 12, 20 * 8 * 8 / 1e6, 20 * 8 * 8 / 1e6, "LIN/CAN"),
]

sub("3-1  逐路传感器带宽（Mbps）")
rows = []
tot_raw = tot_cmp = 0.0
for name, n, raw1, cmp1, note in SENSORS:
    tot_raw += n * raw1
    tot_cmp += n * cmp1
    rows.append(
        [name, str(n), f"{raw1:,.2f}", f"{n*raw1:,.2f}", f"{n*cmp1:,.2f}", note]
    )
rows.append(["【合计】", "", "", f"{tot_raw:,.2f}", f"{tot_cmp:,.2f}", "≈ 压缩后降至 1/%.0f" % (tot_raw / tot_cmp)])
table(
    ["传感器", "数量", "单路原始", "小计原始", "小计压缩/结构化", "备注"],
    rows,
    ["l", "r", "r", "r", "r", "l"],
)
print(f"\n  整车原始（未压缩）总带宽 : {tot_raw/1000:.2f} Gbps")
print(f"  整车压缩/结构化总带宽   : {tot_cmp:.1f} Mbps  ({tot_cmp/1000:.3f} Gbps)")
print(f"  压缩比                  : {tot_raw/tot_cmp:.1f} : 1")
print(f"  按 1 TB = 8e12 bit 计，原始数据落盘速率 = {tot_raw*1e6/8/1e9:.2f} GB/s "
      f"→ 一小时 {tot_raw*1e6/8/1e12*3600:.1f} TB")

# ---- 3-2 总线有效吞吐计算 ----
sub("3-2  各总线「有效可用吞吐」推导（含协议开销与设计负载率）")


def can_classic(bitrate=500e3, payload=8, load=0.5):
    """经典 CAN：标准帧 11 位 ID，含填充位近似 130 bit/帧"""
    bits = 130.0
    fps = bitrate / bits
    return payload * 8 * fps * load / 1e6, fps


def canfd(nom=500e3, data=5e6, payload=64, load=0.5):
    """CAN FD：仲裁段走 nominal 速率，数据段走 data 速率"""
    t_arb = 30 / nom          # SOF+ID+控制头 ≈ 30 bit @nominal
    t_data = (payload * 8 + 21 + 20) / data  # 数据 + CRC + 填充余量
    t_tail = 15 / nom         # CRC 界定符 + ACK + EOF + IFS @nominal
    t = t_arb + t_data + t_tail
    fps = 1.0 / t
    return payload * 8 * fps * load / 1e6, fps, t * 1e6


def eth(line, load=0.7, mtu=1500):
    """以太网：IP/UDP 头 28 B，帧间隙+前导+FCS+头 38 B"""
    payload = mtu - 28
    frame = mtu + 38
    eff = payload / frame
    return line * eff * load, eff


cc, cc_fps = can_classic()
cf, cf_fps, cf_us = canfd()
buses = []
buses.append(["CAN 2.0B 500 kbps", "0.5", f"{cc_fps:.0f} 帧/s", "8 B", f"{cc:.3f}", "50%",
              "事件型控制报文"])
buses.append(["CAN FD 500k/2M", "2.0",
              f"{canfd(data=2e6)[1]:.0f} 帧/s", "64 B", f"{canfd(data=2e6)[0]:.3f}", "50%",
              "量产最常见 CAN FD 配置"])
buses.append(["CAN FD 500k/5M", "5.0", f"{cf_fps:.0f} 帧/s", "64 B", f"{cf:.3f}", "50%",
              f"单帧 {cf_us:.0f} μs"])
buses.append(["LIN 20 kbps", "0.02", "≈100 帧/s", "8 B", f"{0.02*0.5*0.5:.4f}", "50%",
              "车窗/座椅等低速"])
buses.append(["FlexRay 10 Mbps", "10.0", "静态段时隙", "254 B", f"{10*0.75*0.8:.2f}", "80%",
              "时间触发，确定性强"])
e100, ef100 = eth(100)
e1000, ef1000 = eth(1000)
e2g5, _ = eth(2500)
e10g, _ = eth(10000)
buses.append(["100BASE-T1", "100", "—", "1500 B", f"{e100:.1f}", "70%", f"链路效率 {ef100*100:.1f}%"])
buses.append(["1000BASE-T1", "1000", "—", "1500 B", f"{e1000:.1f}", "70%", "智驾域控主干"])
buses.append(["MultiGBASE-T1 2.5G", "2500", "—", "1500 B", f"{e2g5:.1f}", "70%", "中央计算互联"])
buses.append(["MultiGBASE-T1 10G", "10000", "—", "1500 B", f"{e10g:.1f}", "70%", "跨域主干/记录仪"])
buses.append(["GMSL2 (点对点)", "6000", "—", "—", "6000", "100%", "摄像头专用非对称串行"])
buses.append(["MIPI CSI-2 D-PHY 4lane", "10000", "—", "—", "10000", "100%", "SoC 板内"])
table(
    ["总线", "线速率(Mbps)", "最大帧率", "载荷", "有效吞吐(Mbps)", "设计负载率", "备注"],
    buses,
    ["l", "r", "r", "r", "r", "r", "l"],
)

# ---- 3-3 选型判定 ----
sub("3-3  链路选型判定：这条数据流该走什么总线")
LINKS = [
    ("单颗 8MP 原始 YUV422 30fps", cam_raw(3840, 2160, 16, 30)),
    ("单颗 8MP RAW12 30fps", cam_raw(3840, 2160, 12, 30)),
    ("单颗 8MP H.265 压缩流", 25.0),
    ("128 线激光雷达点云", 2.6e6 * 16 * 8 / 1e6),
    ("AT128 激光雷达点云", 1.53e6 * 8 * 8 / 1e6),
    ("4D 成像雷达点云", 5000 * 20 * 16 * 8 / 1e6),
    ("角雷达目标列表", 64 * 32 * 20 * 8 / 1e6),
    ("整车压缩流汇聚到域控", tot_cmp),
    ("整车原始流汇聚（假想）", tot_raw),
]
CANDS = [
    ("CAN FD 5M", cf),
    ("100BASE-T1", e100),
    ("1000BASE-T1", e1000),
    ("2.5G-T1", e2g5),
    ("10G-T1", e10g),
]
rows = []
for lname, need in LINKS:
    row = [lname, f"{need:,.1f}"]
    pick = "—"
    for bname, cap in CANDS:
        if need <= cap:
            row.append("✔")
            if pick == "—":
                pick = bname
        else:
            row.append("✘")
    if pick == "—":
        pick = "需 GMSL2/CSI-2 点对点"
    row.append(pick)
    rows.append(row)
table(
    ["数据流", "需求(Mbps)"] + [c[0] for c in CANDS] + ["最低可行选型"],
    rows,
    ["l", "r"] + ["c"] * len(CANDS) + ["l"],
)

sub("3-4  CAN FD 到底能扛多少：换算成「每秒能发多少个目标」")
print(f"  CAN FD 500k/5M，64 B 载荷，单帧耗时 {cf_us:.0f} μs，理论 {cf_fps:.0f} 帧/s")
print(f"  50% 负载率下可用 {cf_fps*0.5:.0f} 帧/s = {cf:.2f} Mbps")
obj_bytes = 48
print(f"  一个融合目标（位置/速度/尺寸/朝向/协方差/类别）约 {obj_bytes} B，"
      f"每帧 64 B 只能装 {64//obj_bytes} 个")
for hz in (10, 20, 50):
    n_obj = int(cf_fps * 0.5 / hz * (64 // obj_bytes))
    print(f"    → {hz:2d} Hz 输出时，CAN FD 最多承载 {n_obj} 个目标/帧")
print("  结论：CAN FD 传「目标列表」绰绰有余，传「点云/图像」差 2~3 个数量级。")

print()
print("=" * 96)
print("  分析结束")
print("=" * 96)
