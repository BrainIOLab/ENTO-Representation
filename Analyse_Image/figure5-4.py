
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math

neuron_path = os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'dict_performs.pkl')
with open(neuron_path, "rb") as f:
    dict_performs= pickle.load(f)  # 获取全部神经元响应

# 加载神经元响应
neuron_path = r'E:\Image_paper\Project\dict_view.pkl'
with open(neuron_path, "rb") as f:
    dict_view= pickle.load(f)  # 获取全部神经元响应

obj = 'view_pigeon'
md_list = []
for idx, rec in dict_performs[obj].items():
    md_list.append((idx, float(rec['Neighbor_correlation_mean'])))
best_idx = max(md_list, key=lambda t: t[1])[0]   # 最大 MD 的 idx

resp = dict_performs[obj][best_idx]['resp']



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ==== 你给的函数 ====
def tuning_center_bin(resp):
    resp = np.asarray(resp, dtype=float)
    n = resp.size
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    w = resp - np.nanmin(resp)
    w = np.clip(w, 0.0, None)
    if not np.isfinite(w).any() or w.sum() == 0:
        peak_idx = int(np.nanargmax(resp))
        return peak_idx
    C = np.nansum(w * np.cos(theta))
    S = np.nansum(w * np.sin(theta))
    ang = (np.arctan2(S, C) + 2*np.pi) % (2*np.pi)
    center_idx = int(np.round(ang / (2*np.pi) * n)) % n
    peak_idx = int(np.nanargmax(resp))
    return peak_idx  # 按你当前版本：返回峰值bin

def modulation_depth(resp):
    r = np.asarray(resp, dtype=float)
    if r.size != 18:
        raise ValueError("resp 必须是长度为18的数组")
    return float(np.max(r) - np.min(r))

# ==== 样式 ====
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

# x 轴：0..340，每 20°
X = np.arange(18) * 20

# 6×6 网格
rows, cols = 6, 6
assert resp.shape == (18, rows*cols), "resp 形状应为 (18, 36)"

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(rows, cols)
for j in range(rows*cols):
    r, c = divmod(j, cols)
    ax = fig.add_subplot(gs[r, c])
    y = resp[:, j]  # (18,)
    # 计算每个神经元的调谐中心 & 调制深度（在画图中计算）
    cbin = int(tuning_center_bin(y))     # 0..17
    cdeg = cbin * 20
    md_val = modulation_depth(y)
    # 曲线
    ax.plot(X, y, lw=linewidth_plot, color='black', alpha=0.95)
    # 纵轴与刻度
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(axis='y', labelsize=fontsize_)
    # 调谐中心（竖线 + 标注）
    ax.axvline(cdeg, ls='--', lw=0.7, color='red', alpha=0.85)
    ax.text(cdeg, 0.98, f'C={cbin}\n{cdeg}°', ha='center', va='top',
            fontsize=fontsize_-1, color='red')
    # 在右上角补充 MD（可选：保留 3 位小数）
    ax.set_title(f'NC={md_val:.3f}',
                 fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # 底部行保留少量刻度
    if r == rows - 1:
        ax.set_xticks([0, 180, 360])
        ax.set_xticklabels([0, 180, 360], fontsize=fontsize_, fontname=fontname_)
    else:
        ax.set_xticks([])
    # 边框
    for side in ['bottom', 'left', 'top', 'right']:
        ax.spines[side].set_linewidth(linewidth_ax)
plt.subplots_adjust(wspace=0.4, hspace=0.6, right=0.98, left=0.08, top=0.92, bottom=0.08)
plt.savefig(r'E:\Image_paper\Project\plot\figure5\view_pigeon_Neighbor_correlation_mean.eps', dpi=600, format='eps')
