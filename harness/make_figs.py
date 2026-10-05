# -*- coding: utf-8 -*-
"""论文配图：图1 三链证据对照 + 图2 断层结构（纯数据驱动，输出至 report/fig/）"""
import sys, os
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np

FONTDIR = os.environ.get("CC_EVAL_FONTS", r"fonts")  # 指向 Noto Sans SC 字体目录
for f in os.listdir(FONTDIR):
    font_manager.fontManager.addfont(os.path.join(FONTDIR, f))
plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report', 'fig')
os.makedirs(OUT, exist_ok=True)

# ---------------- 图1：三链证据 ----------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) 开源：KNO 随机带 vs 通用能力缩放
ax = axes[0]
models = ['0.5B基', '0.5B', '1.5B', '3B', '7B', 'L1B', 'L3B', 'L8B']
kno = [6.5, 6.5, 15.8, 11.8, 14.0, 6.2, 10.5, 18.8]
poly = [66.7, 60.0, 80.0, 90.0, 83.3, 73.3, 70.0, 72.4]
x = np.arange(len(models))
ax.bar(x - 0.2, kno, 0.4, label='等级知识（定级）', color='#9e2b25')
ax.bar(x + 0.2, poly, 0.4, label='通用能力（多音字定音）', color='#1f4e79')
ax.axhline(14.3, ls='--', c='gray', lw=1, label='随机基线 14.3%')
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel('准确率（%）'); ax.set_ylim(0, 105)
ax.set_title('(a) 开源模型：等级知识贴地，通用能力缩放', fontsize=10)
ax.legend(fontsize=8, loc='upper left', ncols=3)

# (b) 商用：条文定位 vs 情境教学
ax = axes[1]
cats = ['等级知识定级', '标准条文定位', '情境化教学设计']
lo = [12.0, 0, 98]; hi = [35.2, 8.9, 100]
ax.bar(cats, [h - l for l, h in zip(lo, hi)], bottom=lo, color=['#c0504d', '#c0504d', '#4f81bd'], width=0.5)
for i, (l, h) in enumerate(zip(lo, hi)):
    ax.text(i, h + 2, f'{l}–{h}%', ha='center', fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 105); ax.set_ylabel('得分区间（%）')
ax.set_title('(b) 商用模型：会教学，不懂标准', fontsize=10)

# (c) 代际：通用升、标准降
ax = axes[2]
gen = [36.6, 50.8]; std = [35.2, 22.5]
x = np.arange(2)
ax.bar(x - 0.2, gen, 0.35, label='通用生成', color='#1f4e79')
ax.bar(x + 0.2, std, 0.35, label='等级知识', color='#9e2b25')
ax.set_xticks(x); ax.set_xticklabels(['前代', '新一代'], fontsize=9)
ax.set_ylabel('得分（%）'); ax.set_ylim(0, 105)
for i, v in enumerate(gen): ax.text(i - 0.2, v + 1, str(v), ha='center', fontsize=9)
for i, v in enumerate(std): ax.text(i + 0.2, v + 1, str(v), ha='center', fontsize=9)
ax.annotate('+14.2', xy=(0.5, 54), fontsize=9, color='#1f4e79', ha='center')
ax.annotate('−12.7', xy=(0.5, 12), fontsize=9, color='#9e2b25', ha='center')
ax.set_title('(c) 代际更新：通用升、标准降', fontsize=10)
ax.legend(fontsize=8)

fig.suptitle('图 1　通用语言能力与标准专门知识分离的三条证据链', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_evidence.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------- 图2：断层结构 ----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
levels = ['1', '2', '3', '4', '5', '6', '7-9']
qwen = [45/57*100, 0/57*100, 29/57*100, 0/57*100, 3/57*100, 1/57*100, 1/57*100]
llama = [1/57*100, 6/57*100, 28/57*100, 0, 0, 0, 40/57*100]
x = np.arange(len(levels))
ax.bar(x - 0.21, qwen, 0.42, label='Qwen2.5-7B', color='#5b3f8c')
ax.bar(x + 0.21, llama, 0.42, label='Llama-3.1-8B', color='#9e2b25')
ax.axhline(14.3, ls='--', c='gray', lw=1, label='随机基线 14.3%')
ax.set_xticks(x); ax.set_xticklabels([f'{l} 级' for l in levels], fontsize=9)
ax.set_ylabel('分类命中率（%）'); ax.set_ylim(0, 95)
ax.set_title('图 2　等级知识的断层结构：端点可及、中段缺失\n（类别均衡题库，每级 57 题）', fontsize=11)
ax.legend(fontsize=9)
for i, v in enumerate(qwen):
    if v > 3: ax.text(i - 0.21, v + 1.5, f'{v:.0f}%', ha='center', fontsize=8)
for i, v in enumerate(llama):
    if v > 3: ax.text(i + 0.21, v + 1.5, f'{v:.0f}%', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_gap_structure.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('saved:', os.listdir(OUT))
