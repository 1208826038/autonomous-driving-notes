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

## 章节目录（8 章，每章 ≥1000 字）

| 章节 | 文件 | 主题 |
|------|------|------|
| 一、环境感知 | [chapters/ch01-perception.md](chapters/ch01-perception.md) | 摄像头/LiDAR/Radar 目标检测与语义分割 |
| 二、多传感器融合 | [chapters/ch02-sensor-fusion.md](chapters/ch02-sensor-fusion.md) | 卡尔曼族滤波（KF/EKF/UKF）、粒子滤波、多目标跟踪 |
| 三、定位与高精地图 | [chapters/ch03-localization-mapping.md](chapters/ch03-localization-mapping.md) | GNSS/IMU/LiDAR SLAM 与 OpenDRIVE 高精地图 |
| 四、行为决策 | [chapters/ch04-behavior-decision.md](chapters/ch04-behavior-decision.md) | 有限状态机 / 博弈 / 强化学习决策 |
| 五、运动规划 | [chapters/ch05-motion-planning.md](chapters/ch05-motion-planning.md) | A* / RRT* / Lattice 与轨迹优化 |
| 六、车辆控制 | [chapters/ch06-vehicle-control.md](chapters/ch06-vehicle-control.md) | PID / LQR / MPC 与车辆动力学 |
| 七、端到端智驾 | [chapters/ch07-end2end.md](chapters/ch07-end2end.md) | BEV / Transformer / Occupancy 网络 |
| 八、系统架构与中间件 | [chapters/ch08-system-architecture.md](chapters/ch08-system-architecture.md) | ROS2 / DDS / 计算平台(Orin) / 冗余架构 |

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
└── chapters/
    ├── ch01-perception.md
    ├── ch02-sensor-fusion.md
    ├── ch03-localization-mapping.md
    ├── ch04-behavior-decision.md
    ├── ch05-motion-planning.md
    ├── ch06-vehicle-control.md
    ├── ch07-end2end.md
    └── ch08-system-architecture.md
```

---
📌 本仓库持续补充中。欢迎结合实车/仿真项目把章节里的伪代码落成可用实现。
