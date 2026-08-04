# 智能驾驶知识点笔记（autonomous-driving-notes）

> 本仓库聚焦**智能驾驶领域中"超出 BMS（电池管理系统）范畴"的那部分工程知识点**。
> 与同账号下的 `bms-simulink-learning`、`iso26262-aspice-bms-guide` 形成互补：那里讲电池与功能安全体系，这里讲"车怎么看懂世界、怎么决策、怎么开"。

## 这个仓库讲什么

一套面向**嵌入式 / 控制 / 算法工程师**的智能驾驶知识体系，从传感器原始数据一路讲到车辆执行：

- **感知层**：摄像头、激光雷达、毫米波雷达如何"看见"物体（检测、分割、跟踪）。
- **融合与估计层**：多传感器如何对齐、滤波、估计自车与周边状态。
- **定位与地图层**：GNSS/IMU/激光/视觉如何确定"我在哪"，高精地图是什么。
- **决策规划层**：车"该干什么"（行为决策）与"具体怎么走"（运动规划）。
- **控制层**：轨迹如何变成方向盘转角与制动指令。
- **前沿与系统层**：端到端大模型、车载计算平台与 SOA 中间件架构。

> 说明：功能安全（ISO 26262 / ASPICE）与 BMS 强相关，已在 `iso26262-aspice-bms-guide` 单独成册，本仓库仅在系统架构章节做必要衔接，不重复展开。

## 章节目录（8 章，合计约 16000 行 / 71 万字符）

每章都不是提纲式速记，而是**能当教材读、当面试稿背、当工程手册查**的长文：公式真推导、mermaid 架构图、多张对比表格、**实际跑过并把真实输出贴回文档**的 Python 代码、12–14 条"现象→原因→对策"踩坑清单、14–16 道带实质答案的面试题。

| 章节 | 文件 | 主题 | 篇幅 |
|------|------|------|------|
| 一、环境感知 | [chapters/ch01-perception.md](chapters/ch01-perception.md) | 摄像头成像与 ISP、LiDAR 点云、毫米波 FMCW 与 CFAR、三类传感器全维度对比 | 826 行 |
| 二、多传感器融合 | [chapters/ch02-sensor-fusion.md](chapters/ch02-sensor-fusion.md) | 时空对齐与标定、KF/EKF/UKF 推导、IMM、多目标跟踪与数据关联 | 1942 行 |
| 三、定位与高精地图 | [chapters/ch03-localization-mapping.md](chapters/ch03-localization-mapping.md) | GNSS/RTK、IMU 误差模型与 Allan 方差、ESKF 组合导航、OpenDRIVE | 1596 行 |
| 四、行为决策 | [chapters/ch04-behavior-decision.md](chapters/ch04-behavior-decision.md) | 轨迹预测、FSM/行为树、POMDP、博弈论、RSS 安全形式化 | 1773 行 |
| 五、运动规划 | [chapters/ch05-motion-planning.md](chapters/ch05-motion-planning.md) | Frenet 坐标系、Hybrid A\*、Lattice 采样、ST 图与 QP 轨迹优化 | 2091 行 |
| 六、车辆控制 | [chapters/ch06-vehicle-control.md](chapters/ch06-vehicle-control.md) | 车辆动力学、Pure Pursuit/Stanley/LQR/MPC 四法对比、延时补偿与标定 | 2650 行 |
| 七、端到端智驾 | [chapters/ch07-end2end.md](chapters/ch07-end2end.md) | BEV（LSS/BEVFormer）、Occupancy、UniAD/VAD、世界模型与 VLA | 1767 行 |
| 八、系统架构与中间件 | [chapters/ch08-system-architecture.md](chapters/ch08-system-architecture.md) | E/E 架构演进、车载网络、AUTOSAR 双栈、时间同步与功能安全冗余 | 3359 行 |

## 配套可运行代码

`code/` 目录是各章"工程实践"小节里**真实运行过**的验证脚本（纯 Python + numpy，不需要车载环境）。文档中引用的每一个实验数字都来自这些脚本的输出，不是编造的。

| 脚本 | 章节 | 做什么 |
|------|------|--------|
| `lidar_pipeline.py` | ch01 | 体素降采样 → RANSAC 地面剔除 → 欧式聚类 → PCA 定向包围框 |
| `mot_ekf_hungarian.py` | ch02 | CV 模型 EKF + 马氏距离门 + 匈牙利关联 + 航迹生命周期管理 |
| `eskf_gnss_ins_2d.py` | ch03 | 2D ESKF 组合导航，含 GNSS 失锁窗口与零偏估计收敛 |
| `left_turn_decider.py`、`rss_numeric.py` | ch04 | 无保护左转间隙接受决策；RSS 安全距离数值算例 |
| `frenet_lattice_planner.py` | ch05 | Frenet 系 Lattice 规划器：横纵采样 + 碰撞剔除 + 代价评分 |
| `ch06_control_suite.py` | ch06 | 四种横向控制器对比，含车速扫描、延时敏感性、Q/R 权重扫描 |
| `lss_bev_numpy.py` | ch07 | 纯 numpy 复刻 LSS 视锥 → BEV，含深度 bin 数敏感性扫描 |
| `arch_latency_sync_bandwidth.py` | ch08 | 端到端时延蒙特卡洛 P50/P95/P99、同步误差、总线带宽估算 |

运行输出保存为同目录下的 `*_out.txt`。

## 写作规范（保持各章风格统一）

每章建议遵循以下结构，便于速查与面试复用：

1. **引言**：用一个真实/生动的场景（事故、Demo、工程痛点）引出问题。
2. **核心概念**：定义、分类、关键参数，配 `mermaid` 框图/流程图。
3. **机制深拆**：数学表达（公式用 `$...$` 或 `$$...$$`）、算法步骤、数据流。
4. **工程实践**：给出可运行的伪代码 / Python 片段 / C 思路，标注嵌入式落地的坑。
5. **常见坑**：分条列出工程易错点（坐标系、时延、标定、数值稳定性等）。
6. **面试要点**：10 道左右高频面试题 + 一句话答案。
7. **结语**：一页纸回顾 + 延伸阅读。

语言：简体中文，术语中英对照；示例尽量贴近真实车载工程，避免纯教科书式叙述。

## 学习路线建议

```
感知 → 融合 → 定位建图 → 决策 → 规划 → 控制
        ↘ 端到端（并行了解）
系统架构贯穿始终（理解各模块如何跑在车端计算平台上）
```

## 仓库结构

```
autonomous-driving-notes/
├── README.md
├── chapters/            # 8 章正文
│   ├── ch01-perception.md
│   ├── ch02-sensor-fusion.md
│   ├── ch03-localization-mapping.md
│   ├── ch04-behavior-decision.md
│   ├── ch05-motion-planning.md
│   ├── ch06-vehicle-control.md
│   ├── ch07-end2end.md
│   └── ch08-system-architecture.md
└── code/                # 各章配套验证脚本 + 真实运行输出
    ├── *.py
    └── *_out.txt
```

---
📌 本仓库持续补充中。欢迎结合实车/仿真项目把章节里的伪代码落成可用实现。
