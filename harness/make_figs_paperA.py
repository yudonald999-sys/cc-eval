# -*- coding: utf-8 -*-
"""论文 A 三图：图1 维度导出映射 / 图2 11商用×6维热力图 / 图3 干预敏感度"""
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
os.makedirs(OUT, exist_ok=True)

# ---------- 图 1：标准文件→判断类型→任务族→维度 四级映射 ----------
fig, ax = plt.subplots(figsize=(12.5, 6.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
cols = [
    ('标准文件（锚）', ['《中文水平等级标准》\n（GF 0025—2021）', '《文化和国情教学\n参考框架》', '《国际中文教材\n评价标准》', '《国际中文教师专业\n能力标准》'], '#1f4e79'),
    ('判断类型', ['等级归属判断', '语音合规判断', '文化适切判断', '偏误识别纠正', '标准条文定位', '限级内容生成', '教学设计决策'], '#2e6b3f'),
    ('任务族（代码）', ['KNO 定级', 'PHO 语音', 'CUL 文化', 'ERR 纠错', 'SCO 条文', 'GEN 生成', 'PED 教学'], '#7a4a1f'),
]
xs = [6, 33, 60]
for ci, (title, items, color) in enumerate(cols):
    x = xs[ci]
    ax.text(x + 8, 96, title, ha='center', fontsize=11, fontweight='bold', color=color)
    n = len(items)
    h = 10 if n <= 4 else 9
    y0 = 88
    for i, it in enumerate(items):
        y = y0 - i * (h + 3.2)
        box = FancyBboxPatch((x, y - h), 16, h, boxstyle='round,pad=0.35',
                             fc=color, ec='none', alpha=0.92 if ci < 2 else 0.88)
        ax.add_patch(box)
        ax.text(x + 8, y - h / 2, it, ha='center', va='center', fontsize=9, color='white')
# 映射箭头（稀疏示意，避免蛛网）
pairs = [(0, 0), (0, 1), (1, 2), (2, 4), (3, 6), (3, 3)]
for si, ti in pairs:
    sy = 88 - si * (10 + 3.2) - 5
    ty = 88 - ti * (9 + 3.2) - 4.5
    ax.add_patch(FancyArrowPatch((22.8, sy), (32.6, ty), arrowstyle='-|>',
                                 mutation_scale=13, color='#666666', lw=1.1, alpha=0.75))
pairs2 = [(0, 0), (1, 1), (2, 2), (3, 6), (4, 4), (6, 5), (6, 3)]
for si, ti in pairs2:
    sy = 88 - si * (9 + 3.2) - 4.5
    ty = 88 - ti * (9 + 3.2) - 4.5
    ax.add_patch(FancyArrowPatch((49.8, sy), (59.6, ty), arrowstyle='-|>',
                                 mutation_scale=13, color='#666666', lw=1.1, alpha=0.75))
# 维度合计框
box = FancyBboxPatch((82, 30), 14, 44, boxstyle='round,pad=0.4', fc='#5b3f8c', ec='none', alpha=0.92)
ax.add_patch(box)
ax.text(89, 66, '评测维度', ha='center', fontsize=11, fontweight='bold', color='white')
ax.text(89, 52, '36 维\n2176 题', ha='center', va='center', fontsize=13, color='white', fontweight='bold')
for ti in range(7):
    ty = 88 - ti * (9 + 3.2) - 4.5
    ax.add_patch(FancyArrowPatch((76.8, ty), (81.6, 52 + (ty - 52) * 0.35), arrowstyle='-|>',
                                 mutation_scale=12, color='#666666', lw=1.0, alpha=0.7))
ax.set_title('图 1　评测维度的四级导出：从标准文件到任务族', fontsize=12.5, pad=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figA1_mapping.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 图 2：11 商用模型 × 6 维热力图 ----------
d = json.load(open(os.path.join(ROOT, 'results/multimodel_summary.json'), encoding='utf-8'))
models = list(d.keys())
dims = ['KNO', 'PHO*', 'CUL', 'ERR', 'SCO', 'GEN', 'PED']
dim_keys = ['KNO', None, 'CUL', 'ERR', 'SCO', 'GEN', 'PED']
mat = np.full((len(models), len(dims)), np.nan)
for i, m in enumerate(models):
    for j, k in enumerate(dim_keys):
        if k and k in d[m]:
            mat[i, j] = d[m][k]
# PHO 无商用汇总值，列标注后隐藏 NaN
mat_masked = np.ma.masked_invalid(mat)
fig, ax = plt.subplots(figsize=(9.5, 6.2))
cmap = plt.get_cmap('RdYlGn').copy()
cmap.set_bad('#f2f2f2')
im = ax.imshow(mat_masked, cmap=cmap, vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(['KNO\n等级知识', 'PHO\n语音知识', 'CUL\n文化判断', 'ERR\n偏误纠正', 'SCO\n条文定位', 'GEN\n限级生成', 'PED\n教学设计'], fontsize=9)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([f'商用 M{i+1}' for i in range(len(models))], fontsize=9)
for i in range(len(models)):
    for j in range(len(dims)):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                    color='black' if 25 < v < 80 else 'white')
        else:
            ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='#999999')
ax.set_xticks(np.arange(-0.5, len(dims)), minor=True)
ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
ax.grid(which='minor', color='white', lw=1.2)
ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('得分', fontsize=9)
ax.set_title('图 2　十一个商用模型在六个主维度上的得分热力图\n（型号按双盲评审要求匿名；PHO 维度施测中，暂缺）', fontsize=11.5, pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figA2_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 图 3：干预敏感度 ----------
fig, ax = plt.subplots(figsize=(8.8, 4.6))
groups = ['留出词\n（零重叠）', '训练词\n（均衡采样）', '原题库\n（跨题面）', '原题库\n（原始口径）']
base = [4.3, 15.4, 6.2, 9.0]
inj = [42.1, 28.6, 34.2, 30.2]
x = np.arange(len(groups))
b1 = ax.bar(x - 0.2, [b if b is not None else 0 for b in base], 0.38, label='注入前（基座）', color='#9e2b25')
b2 = ax.bar(x + 0.2, inj, 0.38, label='注入后（低秩适配微调）', color='#1f4e79')
for i, (b, v) in enumerate(zip(base, inj)):
    ax.text(i - 0.2, b + 1, f'{b}', ha='center', fontsize=9)
    ax.text(i + 0.2, v + 1, f'{v}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(14.3, ls='--', c='gray', lw=1, label='随机基线 14.3%')
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
ax.set_ylabel('准确率（%）'); ax.set_ylim(0, 55)
ax.set_title('图 3　评测工具对知识状态变化的敏感度（注入实验，0.5B 基座）', fontsize=11.5)
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figA3_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('saved:', [f for f in os.listdir(OUT) if f.startswith('figA')])
