import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math


def analyse_reducedim(resp_class, reducedim_method, n_components):
    if reducedim_method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)  # 降维到2维
        X_reduced = pca.fit_transform(resp_class)
    if reducedim_method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components)  # 降维到2维
        X_reduced = tsne.fit_transform(resp_class)
    if reducedim_method == 'MDS':
        from sklearn.manifold import MDS
        mds = MDS(n_components=n_components)  # 降维到2维
        X_reduced = mds.fit_transform(resp_class)
    if reducedim_method == 'Isomap':
        from sklearn.manifold import Isomap
        isomap = Isomap(n_components=n_components)  # 降维到2维
        X_reduced = isomap.fit_transform(resp_class)
    from sklearn.metrics import pairwise_distances
    rdm = pairwise_distances(X_reduced, metric='euclidean')
    return X_reduced, rdm

def _procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    经典 Procrustes：去中心、单位化、最优正交对齐后残差范数。
    返回归一化残差（越小越相似）。
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc) + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc) + 1e-12)
    # 最优旋转
    U, _, Vt = np.linalg.svd(Xn.T @ Yn, full_matrices=False)
    R = U @ Vt
    X_aligned = Xn @ R
    resid = np.linalg.norm(X_aligned - Yn)
    return float(resid)

def metric_circularity_procrustes(X_reduced: np.ndarray):
    X = np.asarray(X_reduced, float)
    # 与理想等间隔单位圆的 Procrustes 距离
    n = X.shape[0]
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n,2)
    proc = _procrustes_distance(X, circle)
    return   proc

def compute_global_dist(RDM):
    n = RDM.shape[0]
    discriminability = np.sum(RDM) / (n * n - n)  # 1. 全局区分度（去对角线）
    return discriminability


# 加载神经元响应
neuron_path = r'E:\Image_paper\Project\dict_view.pkl'
with open(neuron_path, "rb") as f:
    dict_view= pickle.load(f)  # 获取全部神经元响应

dict_performs={}
obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']
ft_list=['Modulation depth','Neighbor correlation']
reducedim_method='MDS'
n_components=2
n_neuron=36

for obj in obj_list:
    dict_performs[obj]={}
    for ft in ft_list:
        dict_performs[obj][ft] = {}

        dict_mertics = {}
        for neu in dict_view.keys():
            dict_mertics[neu] = dict_view[neu][obj]['metrics'][ft]
        sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
        resp_view = [dict_view[neu][obj]['resp'] for neu in sorted_dict.keys()]
        resp_view = np.array(resp_view)


        dict_performs[obj][ft]['PCD']=[]
        dict_performs[obj][ft]['GD']=[]
        range_list = list(range(0, resp_view.shape[0] - n_neuron + 1))
        for nstart in range_list:
            resp_split = resp_view[nstart:nstart + n_neuron, :]
            resp_split = np.transpose(resp_split)
            # 性能指标
            X_reduced, rdm = analyse_reducedim(resp_split, reducedim_method, n_components)
            #计算与标准圆的procrustes_distance
            pd=metric_circularity_procrustes(X_reduced)
            #计算全局平均距离
            global_distance = compute_global_dist(rdm)
            dict_performs[obj][ft]['PCD'].append(pd)
            dict_performs[obj][ft]['GD'].append(global_distance)
            print(obj,ft,nstart,pd,global_distance)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ===== 样式 =====
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'
fts_row = ['Modulation depth', 'Neighbor correlation']   # 两行
metrics_col = ['PCD', 'GD']                              # 两列
colors = plt.cm.tab10(np.linspace(0, 1, len(obj_list)))  # 6 条曲线配色
def set_axes_style(ax, ylabel):
    ax.set_xlabel('Sliding window index', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.tick_params(labelsize=fontsize_)
def set_ylim_from_all(ax, arrays):
    ys = np.concatenate([np.asarray(a, float) for a in arrays if len(a)>0])
    ys = ys[np.isfinite(ys)]
    if ys.size:
        pad = 0.05 * (ys.max() - ys.min() + 1e-12)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
fig = plt.figure(figsize=(8.5, 5.0))
gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45)
for r, ft in enumerate(fts_row):
    for c, metric_key in enumerate(metrics_col):
        ax = fig.add_subplot(gs[r, c])
        # 画 6 条曲线（不同 obj）
        all_y_for_ylim = []
        for i, obj in enumerate(obj_list):
            ys = np.asarray(dict_performs[obj][ft][metric_key], dtype=float)
            if ys.size == 0:
                continue
            xs = np.arange(1, ys.size + 1)
            ax.plot(xs, ys, linewidth=linewidth_plot, color=colors[i], label=obj)
            all_y_for_ylim.append(ys)
        arrow = '↓ better' if metric_key == 'PCD' else '↑ better'
        set_axes_style(ax, f'{metric_key} ({arrow})')
        set_ylim_from_all(ax, all_y_for_ylim)
        ax.set_title(f'{ft} — {metric_key}', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
        ax.legend(loc='best', fontsize=fontsize_-1, frameon=False)
plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ===== 样式 =====
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

fts_row = ['Modulation depth', 'Neighbor correlation']   # 行：2 个特征
metrics_col = ['PCD', 'GD']                              # 列：PCD/GD
colors = plt.cm.tab10(np.linspace(0, 1, len(obj_list)))  # 6 条曲线配色
SHOW_RAW = True                                          # 是否叠加原始曲线
RAW_ALPHA = 0.25

def _best_odd_window(n, pref):
    w = min(pref, n if n%2==1 else n-1)
    if w < 3: return 3 if n >= 3 else (n|1)
    return w if w%2==1 else w-1

def smooth_series(y, method='savgol', win_pref=9, poly=2):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not ok.any():
        return np.zeros_like(y)
    idx = np.arange(len(y))
    y_filled = np.interp(idx, idx[ok], y[ok])
    n = len(y_filled)
    if n < 5:
        return y_filled
    if method == 'savgol':
        try:
            from scipy.signal import savgol_filter
            w = _best_odd_window(n, win_pref)
            w = max(w, poly + 3 if (poly+1)%2==0 else poly + 2)
            w = min(w, n if n%2==1 else n-1)
            if w < 3: w = 3
            if w <= poly: poly = max(1, min(poly, w-2))
            return savgol_filter(y_filled, window_length=w, polyorder=poly, mode='interp')
        except Exception:
            method = 'ma'
    if method == 'ma':
        w = _best_odd_window(n, win_pref if win_pref%2==1 else win_pref-1)
        w = max(3, w)
        pad = w//2
        y_pad = np.pad(y_filled, (pad, pad), mode='edge')
        kernel = np.ones(w) / w
        return np.convolve(y_pad, kernel, mode='valid')
    return y_filled

def set_axes_style(ax, ylabel):
    ax.set_xlabel('Sliding window index', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.tick_params(labelsize=fontsize_)

def set_ylim_from_all(ax, arrays):
    ys = np.concatenate([np.asarray(a, float) for a in arrays if len(a)>0])
    ys = ys[np.isfinite(ys)]
    if ys.size:
        pad = 0.05 * (ys.max() - ys.min() + 1e-12)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

fig = plt.figure(figsize=(8.5, 5.0))
gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45)

for r, ft in enumerate(fts_row):
    for c, metric_key in enumerate(metrics_col):
        ax = fig.add_subplot(gs[r, c])
        all_y_for_ylim = []
        for i, obj in enumerate(obj_list):
            y = np.asarray(dict_performs[obj][ft][metric_key], float)
            if y.size == 0:
                continue
            x = np.arange(1, y.size + 1)
            if SHOW_RAW:
                ax.plot(x, y, color='0.6', linewidth=linewidth_plot, alpha=RAW_ALPHA)
            y_sm = smooth_series(y, method='savgol', win_pref=15, poly=2)  # ← 已修正
            ax.plot(x, y_sm, color=colors[i], linewidth=linewidth_plot, label=obj)
            all_y_for_ylim.append(y_sm)
        arrow = '↓ better' if metric_key == 'PCD' else '↑ better'
        set_axes_style(ax, f'{metric_key} ({arrow})')
        set_ylim_from_all(ax, all_y_for_ylim)
        ax.set_title(f'{ft} — {metric_key} (smoothed)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
        ax.legend(loc='best', fontsize=fontsize_-1, frameon=False)

plt.tight_layout()
plt.show()
