#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五章《运动规划》交付校验脚本。

检查项：
  1. 行数 >= 800
  2. mermaid 代码块 >= 2
  3. Markdown 表格 >= 4
  4. 数学公式（$...$ / $$...$$）存在且成对
  5. 章节结构完整（0 引言 / 1 核心概念 / 2~6 机制 / 7 工程实践 / 8 常见坑 /
                    9 面试要点 / 10 结语）
  6. 常见坑条目数 12~14，面试问答条目数 14~16
  7. 真实运行输出已嵌入（与 lattice_out.txt 中的关键行比对）
  8. 字数统计（中文字符 / 总字符 / 正文估算字数）
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, '..', 'chapters', 'ch05-motion-planning.md')
OUT = os.path.join(HERE, 'lattice_out.txt')

ok_all = True


def check(label, cond, detail=''):
    global ok_all
    mark = 'PASS' if cond else 'FAIL'
    if not cond:
        ok_all = False
    print(f"  [{mark}] {label}{('  ' + detail) if detail else ''}")
    return cond


text = open(DOC, encoding='utf-8').read()
lines = text.split('\n')

print("=" * 70)
print(" ch05-motion-planning.md 交付校验")
print("=" * 70)

# ---- 1. 行数 ----
n_lines = len(lines)
print("\n[1] 篇幅")
check("行数 >= 800", n_lines >= 800, f"实际 {n_lines} 行")

# ---- 2. mermaid ----
mermaid = re.findall(r'```mermaid', text)
print("\n[2] 图表")
check("mermaid 图 >= 2", len(mermaid) >= 2, f"实际 {len(mermaid)} 个")

# ---- 3. 表格 ----
# 一个表格 = 连续出现的 | --- | 分隔行
seps = [i for i, ln in enumerate(lines)
        if re.match(r'^\s*\|[\s:\-\|]+\|\s*$', ln) and '-' in ln]
check("对比表格 >= 4", len(seps) >= 4, f"实际 {len(seps)} 个")

# ---- 4. 数学公式 ----
block = re.findall(r'\$\$', text)
inline = re.findall(r'(?<!\$)\$(?!\$)', text)
print("\n[3] 数学公式")
check("块级公式 $$ 成对", len(block) % 2 == 0, f"{len(block)//2} 个块级公式")
check("行内公式 $ 成对", len(inline) % 2 == 0, f"{len(inline)//2} 处行内公式")
check("块级公式 >= 20", len(block) // 2 >= 20)

# ---- 5. 章节结构 ----
print("\n[4] 章节结构")
required = [
    (r'^# 五、运动规划', '章节标题'),
    (r'^## 0\. 引言', '0 引言'),
    (r'^## 1\. 核心概念', '1 核心概念'),
    (r'^## 2\. ', '2 机制深拆'),
    (r'^## 3\. ', '3 机制深拆'),
    (r'^## 4\. ', '4 机制深拆'),
    (r'^## 5\. ', '5 机制深拆'),
    (r'^## 6\. ', '6 机制深拆'),
    (r'^## 7\. 工程实践', '7 工程实践'),
    (r'^### 车规\s*/?\s*实时落地的坑', '7.x 车规落地的坑'),
    (r'^## 8\. 常见坑', '8 常见坑'),
    (r'^## 9\. 面试要点', '9 面试要点'),
    (r'^## 10\. 结语', '10 结语'),
    (r'^本章约 \d+ 字', '字数页脚'),
]
for pat, name in required:
    check(name, any(re.match(pat, ln) for ln in lines))

# ---- 6. 条目数 ----
print("\n[5] 条目数量")
pits = [ln for ln in lines if re.match(r'^### 坑 \d+', ln)]
qs = [ln for ln in lines if re.match(r'^\*\*Q\d+', ln)]
check("常见坑 12~14 条", 12 <= len(pits) <= 14, f"实际 {len(pits)} 条")
check("面试问答 14~16 条", 14 <= len(qs) <= 16, f"实际 {len(qs)} 条")

# ---- 7. 引言长度 ----
intro = []
grab = False
for ln in lines:
    if re.match(r'^## 0\. 引言', ln):
        grab = True
        continue
    if grab and re.match(r'^## 1\.', ln):
        break
    if grab:
        intro.append(ln)
intro_cn = len(re.findall(r'[\u4e00-\u9fff]', '\n'.join(intro)))
check("引言 >= 300 中文字", intro_cn >= 300, f"实际 {intro_cn} 字")

# ---- 8. 真实输出嵌入校验 ----
print("\n[6] 真实运行输出嵌入校验")
if os.path.exists(OUT):
    real = open(OUT, encoding='utf-8').read()
    keys = [
        '数值重建曲率 vs 设计曲率  max|误差| = 4.566e-07',
        'cond(M) = 4.263e+05',
        '==> 最终可行                                   :  123   (25.3%)',
        '横向终点 l_end = +0.00 m | 收敛距离 S = 30 m',
        '横向终点 l_end = -1.00 m | 收敛距离 S = 45 m',
        'no parked_van             105      281',
        '帧间起点跳变: |dl| = 0.0000 cm',
    ]
    for k in keys:
        in_real = k in real
        in_doc = k in text
        check(f"输出行一致: {k[:42]}...", in_real and in_doc,
              f"(源{'有' if in_real else '无'}/文档{'有' if in_doc else '无'})")
else:
    check("lattice_out.txt 存在", False)

# ---- 9. 关键覆盖点 ----
print("\n[7] 需覆盖内容点名检查")
topics = {
    '路径-速度解耦': r'路径-速度解耦',
    '时空联合规划': r'时空联合规划',
    'Frenet 坐标系': r'Frenet',
    '参考线平滑': r'参考线.{0,6}平滑|平滑.{0,6}参考线',
    '曲率修正项': r'曲率修正项',
    '自行车模型': r'自行车模型',
    '最小转弯半径': r'最小转弯半径',
    '非完整约束': r'非完整约束',
    'Dubins/Reeds-Shepp': r'Reeds-?Shepp',
    'A* 可采纳/一致': r'可采纳.*一致|一致.*可采纳',
    'Hybrid A*': r'Hybrid A',
    '代价地图膨胀': r'膨胀层|膨胀半径',
    'Lattice': r'Lattice',
    '五次多项式': r'五次多项式',
    'state lattice': r'状态格|state lattice',
    'RRT/RRT*': r'RRT\\?\*',
    'ST 图': r'ST 图',
    '时空走廊': r'时空走廊',
    'DP+QP 两阶段': r'DP 粗解|DP 粗解 \+ QP',
    'Apollo EM': r'EM Planner|Apollo EM',
    '速度-曲率耦合': r'速度-曲率耦合|v\^?2.?\\kappa|v\^2 \\kappa',
    'QP/OSQP': r'OSQP',
    '凸走廊': r'凸走廊|凸化',
    'MPC': r'MPC',
    'jerk 舒适': r'jerk',
    '一致性/迟滞': r'迟滞',
    '概率占用/风险场': r'风险场',
    '应急规划': r'应急规划|contingency',
    '规划vs控制频率': r'控制频率|100 Hz',
    '轨迹拼接': r'轨迹拼接|stitching',
    'fallback': r'fallback',
}
miss = []
for name, pat in topics.items():
    if not re.search(pat, text, re.IGNORECASE):
        miss.append(name)
check(f"需覆盖点 {len(topics)} 项全部命中", not miss,
      ('缺失: ' + ', '.join(miss)) if miss else '')

# ---- 10. 字数 ----
print("\n[8] 字数统计")
cn = len(re.findall(r'[\u4e00-\u9fff]', text))
en_words = len(re.findall(r'[A-Za-z]+', text))
total_chars = len(text)
# 正文（剔除代码块）字数
no_code = re.sub(r'```.*?```', '', text, flags=re.S)
cn_body = len(re.findall(r'[\u4e00-\u9fff]', no_code))
print(f"       总字符数           : {total_chars}")
print(f"       中文字符数         : {cn}")
print(f"       中文字符数(不含代码): {cn_body}")
print(f"       英文单词数         : {en_words}")
print(f"       估算字数(中文+英文): {cn + en_words}")
print(f"       行数               : {n_lines}")

print("\n" + "=" * 70)
print(" 结论: " + ("全部通过" if ok_all else "存在未通过项"))
print("=" * 70)
sys.exit(0 if ok_all else 1)
