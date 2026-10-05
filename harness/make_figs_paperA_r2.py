# -*- coding: utf-8 -*-
"""论文 A 图 1/图 2 重绘（审稿四轮修订：SCO 正名为作文评分、PED 复合、双锚定）"""
import sys, os, json
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FONTDIR = os.environ.get("CC_EVAL_FONTS", r"fonts")  # 指向 Noto Sans SC 字体目录
for f in os.listdir(FONTDIR):
    font_manager.fontManager.addfont(os.path.join(FONTDIR, f))
plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'report', 'fig')

# ---------- 图 1：双锚定四级导出 ----------
fig, ax = plt.subplots(figsize=(13.0, 7.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

anchor1 = [('《中文水平等级标准》\n（GF 0025—2021）', 88), ('《文化和国情教学\n参考框架》', 73),
           ('《国际中文教材\n评价标准》', 58), ('《国际中文教师专业\n能力标准》', 43)]
anchor2 = [('真实学习者中介语\n语料（HSK 作文语料）', 22)]
ax.text(13, 97, '锚点一：国家标准体系', ha='center', fontsize=11, fontweight='bold', color='#1f4e79')
ax.text(13, 30.5, '锚点二：学习者语料', ha='center', fontsize=11, fontweight='bold', color='#8a4b8f')
for txt, y in anchor1:
    ax.add_patch(FancyBboxPatch((4, y - 6.5), 18, 10, boxstyle='round,pad=0.35', fc='#1f4e79', ec='none', alpha=0.92))
    ax.text(13, y - 1.5, txt, ha='center', va='center', fontsize=8.5, color='white')
for txt, y in anchor2:
    ax.add_patch(FancyBboxPatch((4, y - 6.5), 18, 10, boxstyle='round,pad=0.35', fc='#8a4b8f', ec='none', alpha=0.92))
    ax.text(13, y - 1.5, txt, ha='center', va='center', fontsize=8.5, color='white')

judges = [('等级归属判断', 88), ('语音合规判断', 77), ('文化适切判断', 66), ('偏误识别纠正', 55),
          ('作文评分对齐', 44), ('限级内容生成', 33), ('标准条文定位与\n教学设计决策', 16)]
ax.text(45, 97, '判断类型', ha='center', fontsize=11, fontweight='bold', color='#2e6b3f')
for txt, y in judges:
    ax.add_patch(FancyBboxPatch((36, y - 5), 18, 9 if '\n' not in txt else 11, boxstyle='round,pad=0.35', fc='#2e6b3f', ec='none', alpha=0.9))
    ax.text(45, y - (0.5 if '\n' not in txt else 0), txt, ha='center', va='center', fontsize=9, color='white')

tasks = [('KNO 定级', 88), ('PHO 语音', 77), ('CUL 文化', 66), ('ERR 纠错', 55),
         ('SCO 作文评分', 44), ('GEN 限级生成', 33), ('PED 标准与教学\n（条文定位+情境设计）', 16)]
ax.text(77, 97, '任务族（代码）', ha='center', fontsize=11, fontweight='bold', color='#7a4a1f')
for txt, y in tasks:
    ax.add_patch(FancyBboxPatch((68, y - 5), 18, 9 if '\n' not in txt else 11, boxstyle='round,pad=0.35', fc='#7a4a1f', ec='none', alpha=0.9))
    ax.text(77, y - (0.5 if '\n' not in txt else 0), txt, ha='center', va='center', fontsize=8.5, color='white')

# 锚点→判断 连线
links1 = [(88, 88), (88, 77), (88, 33), (73, 66), (58, 16), (43, 16)]
for y1, y2 in links1:
    ax.add_patch(FancyArrowPatch((22.8, y1 - 1.5), (35.6, y2 - 0.5), arrowstyle='-|>',
                                 mutation_scale=13, color='#666666', lw=1.1, alpha=0.75))
links1b = [(22, 55), (22, 44)]
for y1, y2 in links1b:
    ax.add_patch(FancyArrowPatch((22.8, y1 - 1.5), (35.6, y2 - 0.5), arrowstyle='-|>',
                                 mutation_scale=13, color='#8a4b8f', lw=1.1, alpha=0.75))
# 判断→任务 一一对应
for _, y in judges:
    ax.add_patch(FancyArrowPatch((54.8, y - 0.5), (67.6, y - 0.5), arrowstyle='-|>',
                                 mutation_scale=13, color='#666666', lw=1.1, alpha=0.75))
# 任务→维度合计
ax.add_patch(FancyBboxPatch((90, 34), 9, 40, boxstyle='round,pad=0.4', fc='#5b3f8c', ec='none', alpha=0.92))
ax.text(94.5, 68, '评测标准体系', ha='center', fontsize=10, fontweight='bold', color='white')
ax.text(94.5, 57, '【基准版】\n6主维 1555题\n（本文验证）', ha='center', va='center', fontsize=8, color='white')
ax.text(94.5, 43, '【演进版】\n36维 2176题\n（后续扩展）', ha='center', va='center', fontsize=8, color='#d9d2e9')
for _, y in tasks:
    ax.add_patch(FancyArrowPatch((86.8, y - 0.5), (89.6, 54 + (y - 54) * 0.25), arrowstyle='-|>',
                                 mutation_scale=12, color='#666666', lw=1.0, alpha=0.7))
ax.set_title('图 1　评测维度的双锚定导出：国家标准与学习者语料', fontsize=12.5, pad=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figA1_mapping.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 图 2：v1.0 六大主维热力图（SCO 正名、删 PHO 列） ----------
d = json.load(open(os.path.join(ROOT, 'results/multimodel_summary.json'), encoding='utf-8'))
models = list(d.keys())
dims = ['KNO', 'CUL', 'ERR', 'SCO', 'GEN', 'PED']
labels = ['KNO\n等级知识', 'CUL\n文化判断', 'ERR\n偏误纠正', 'SCO\n作文评分', 'GEN\n限级生成', 'PED\n标准与教学']
mat = np.array([[d[m][k] for k in dims] for m in models])
fig, ax = plt.subplots(figsize=(9.0, 6.2))
cmap = plt.get_cmap('RdYlGn')
im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([f'商用 M{i+1}' for i in range(len(models))], fontsize=9)
for i in range(len(models)):
    for j in range(len(dims)):
        v = mat[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                color='black' if 25 < v < 80 else 'white')
ax.set_xticks(np.arange(-0.5, len(dims)), minor=True)
ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
ax.grid(which='minor', color='white', lw=1.2)
ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('得分', fontsize=9)
ax.set_title('图 2　十一个商用模型在 CC-Eval v1.0 六大主维上的得分热力图\n（型号按双盲评审要求匿名编号）', fontsize=11.5, pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figA2_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('figA1/figA2 redrawn')
