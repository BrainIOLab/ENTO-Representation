import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score


import numpy as np
from collections import defaultdict

def build_balanced_batches(dict_view, obj, per_bin=2, repeats=100, seed=None):
    """
    返回长度=repeats的列表，每个元素是长度=18*per_bin的神经元ID列表。
    策略：
      1) 优先不放回；
      2) bin 数量不足时，仅对该 bin 允许放回；
      3) bin 为 0 时，从最近的相邻 bin 借（环状距离），必要时跨多层邻居；
    仍严格输出每个中心各 per_bin 个，总数 18*per_bin。
    """
    rng = np.random.default_rng(seed)
    target_bins = list(range(18))

    # 分桶
    buckets = defaultdict(list)
    for neu, d in dict_view.items():
        if obj not in d:
            continue
        entry = d[obj]
        resp = entry.get('resp', None)
        tc   = entry.get('metrics', {}).get('Tuning center', None)
        if isinstance(tc, (int, np.integer)) and resp is not None and np.asarray(resp).shape == (18,):
            if 0 <= int(tc) <= 17:
                buckets[int(tc)].append(neu)

    # 预计算每个bin的“邻居借样池”（环状最近→更远）
    # 对于空bin，借样顺序：±1, ±2, ...（mod 18）
    borrow_order = {b: [] for b in target_bins}
    for b in target_bins:
        order = []
        for k in range(1, 9):  # 最多扩到 9 步足够覆盖全环
            order.append((b - k) % 18)
            order.append((b + k) % 18)
        borrow_order[b] = order

    neu_batches = []
    for _ in range(repeats):
        selected = []
        for b in target_bins:
            pool = buckets[b]

            if len(pool) >= per_bin:
                # 足够：不放回
                choices = rng.choice(pool, size=per_bin, replace=False)
                selected.extend(choices.tolist())
            elif len(pool) > 0:
                # 仅 1 个：第一个不放回，第二个允许放回
                first = rng.choice(pool, size=1, replace=False).tolist()
                second = rng.choice(pool, size=per_bin-1, replace=True).tolist()
                selected.extend(first + second)
            else:
                # 为 0：从邻居借
                borrowed = []
                for nb in borrow_order[b]:
                    if len(buckets[nb]) > 0:
                        need = per_bin - len(borrowed)
                        take = min(need, len(buckets[nb]))
                        borrowed.extend(rng.choice(buckets[nb], size=take, replace=False).tolist())
                        if len(borrowed) == per_bin:
                            break
                # 如果还不够（极端情况），最后允许从所有非空bin里放回补齐
                if len(borrowed) < per_bin:
                    all_pool = np.concatenate([np.array(buckets[nb]) for nb in target_bins if len(buckets[nb])>0])
                    extra = rng.choice(all_pool, size=per_bin-len(borrowed), replace=True).tolist()
                    borrowed.extend(extra)
                selected.extend(borrowed)

        rng.shuffle(selected)
        assert len(selected) == 18 * per_bin
        neu_batches.append(selected)

    return neu_batches



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

def pd_per_object(X_reduced: np.ndarray, n_obj=6, n_view=18):
    """
    X_reduced: (n_obj*n_view, 2)，按物体顺序堆叠
    返回：pds(长度 n_obj 的数组), pd_mean(标量)
    """
    X = np.asarray(X_reduced, float)
    assert X.shape == (n_obj * n_view, 2), f"X_reduced shape {X.shape} 不符合 {n_obj*n_view}×2"

    pds = []
    for o in range(n_obj):
        start = o * n_view
        end   = start + n_view
        pts_o = X[start:end, :]  # (18, 2)
        pds.append(metric_circularity_procrustes(pts_o))

    pds = np.array(pds, dtype=float)
    return float(np.nanmean(pds)),float(np.min(pds))

def compute_class_distance(rdm, labels):
    #计算平均类内距离，和平均类间距离
    intra, inter = [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                intra.append(rdm[i, j])
            else:
                inter.append(rdm[i, j])
    return np.mean(intra), np.mean(inter)
# 加载神经元响应
neuron_path = r'E:\Image_paper\Project\dict_view.pkl'
with open(neuron_path, "rb") as f:
    dict_view= pickle.load(f)  # 获取全部神经元响应+

neuron_path = r'E:\Image_paper\Project\plot\figure6\dict_view_resp.pkl'
with open(neuron_path, "rb") as f:
    dict_view_resp= pickle.load(f)  # 获取全部神经元响应

dict_performs={}
obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']
for obj in obj_list:
    per_bin = 2
    repeats = 1000
    neu_batches = build_balanced_batches(dict_view, obj, per_bin=per_bin, repeats=repeats, seed=42)

    reducedim_method='MDS'
    n_components=2
    dict_performs[obj] = {}
    for idx,neulist in enumerate(neu_batches):
        print(obj, idx)
        Modulation_depth = [dict_view[neu][obj]['metrics']['Modulation depth'] for neu in neulist]
        Neighbor_correlation = [dict_view[neu][obj]['metrics']['Neighbor correlation'] for neu in neulist]

        dict_performs[obj][idx]={}
        dict_performs[obj][idx]['Modulation depth']=np.mean(Modulation_depth)
        dict_performs[obj][idx]['Modulation depth'] = np.median(Modulation_depth)

        resp_class = [dict_view_resp[neu]['resp'] for neu in neulist]
        resp_class = np.transpose(resp_class)
        labels=dict_view_resp[neulist[0]]['labels']

        X_reduced, rdm = analyse_reducedim(resp_class, 'MDS', 2)
        dist_intra, dist_inter = compute_class_distance(rdm, labels)

        mean_pd,min_pd=pd_per_object(X_reduced, n_obj = 6, n_view = 18)

        model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        y_pred = cross_val_predict(model, resp_class, labels, cv=cv)
        accuracy = accuracy_score(labels, y_pred)  # 准确率

        dict_performs[obj][idx]['ACC'] = accuracy
        dict_performs[obj][idx]['dist_intra'] = dist_intra
        dict_performs[obj][idx]['dist_inter'] = dist_inter
        dict_performs[obj][idx]['mean PCD'] = mean_pd
        dict_performs[obj][idx]['min PCD'] = min_pd
        dict_performs[obj][idx]['X_reduced'] = X_reduced

with open(r'E:\Image_paper\Project\plot\figure6\dict_performs_viewclass.pkl', "wb") as f:
    pickle.dump(dict_performs, f)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 允许的键
_ALLOWED_KEYS = {'ACC', 'dist_intra', 'dist_inter', 'PCD'}

def _gather_xy(dict_performs, obj_list, x_key, y_key):
    assert x_key in _ALLOWED_KEYS and y_key in _ALLOWED_KEYS, f"keys must be in {_ALLOWED_KEYS}"
    Xs, Ys = [], []
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            if x_key in rec and y_key in rec:
                try:
                    x = float(rec[x_key]); y = float(rec[y_key])
                except Exception:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    Xs.append(x); Ys.append(y)
    return np.array(Xs, float), np.array(Ys, float)

def plot_xy_scatter(dict_performs, obj_list, x_key, y_key, add_trend=True, figsize=(3.2, 3.0),
                    fontsize_=8, linewidth_ax=0.5, fontname_='Arial', fontweight_='normal',
                    color='tab:blue', save_path=None):
    xs, ys = _gather_xy(dict_performs, obj_list, x_key, y_key)
    if xs.size == 0:
        print("No valid points to plot."); return

    # 计算并打印皮尔逊相关
    r = np.corrcoef(xs, ys)[0,1] if xs.size > 1 else np.nan
    print(f"{x_key} vs {y_key} | N={xs.size} | Pearson r = {r:.3f}")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    ax.scatter(xs, ys, s=16, alpha=0.9, color=color, edgecolors='none')

    # 趋势线（整体）
    if add_trend and xs.size >= 2:
        coef = np.polyfit(xs, ys, deg=1)
        xx = np.linspace(xs.min(), xs.max(), 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, lw=0.8, color='k', linestyle='--', alpha=0.7)
        # 在图内角落标注 r
        ax.text(0.98, 0.02, f"r={r:.2f}", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=fontsize_-1)

    # 轴标签与样式
    ax.set_xlabel(x_key.replace('_', ' '), fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel(y_key.replace('_', ' '), fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.tick_params(labelsize=fontsize_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)

    plt.subplots_adjust(bottom=0.22, left=0.18, right=0.96, top=0.95)
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

# ===== 示例用法 =====
plot_xy_scatter(dict_performs, obj_list, 'dist_inter', 'dist_intra')

plot_xy_scatter(dict_performs, obj_list, 'PCD', 'dist_inter')

plot_xy_scatter(dict_performs, obj_list, 'PCD', 'dist_intra')







import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize

# —— 兼容提取 PCD（标量 / (数组, 均值) / 任意可迭代取均值）——
def _extract_val(v):
    try:
        if np.isscalar(v):                 # 标量
            return float(v)
        if isinstance(v, (tuple, list)) and len(v) == 2 and np.isscalar(v[1]):
            return float(v[1])             # (pds_array, pd_mean) 取第二个
        arr = np.asarray(v, dtype=float).ravel()
        return float(np.nanmean(arr))      # 其他情况取均值
    except Exception:
        return np.nan

# —— 收集 x=dist_intra, y=dist_inter, c=<PCD键>, acc=ACC —— #
def gather_xyc_with_acc(dict_performs, obj_list, c_key):
    xs, ys, cs, accs = [], [], [], []
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            try:
                x = float(rec['dist_intra'])
                y = float(rec['dist_inter'])
                c = _extract_val(rec.get(c_key, np.nan))
                a = float(rec.get('ACC', np.nan))
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(c) and np.isfinite(a):
                xs.append(x); ys.append(y); cs.append(c); accs.append(a)
    return np.array(xs, float), np.array(ys, float), np.array(cs, float), np.array(accs, float)

def scatter_panel(ax, xs, ys, cs, accs, cmap='viridis',
                  fontsize_=8, linewidth_ax=0.5, fontname_='Arial', fontweight_='normal',
                  xlabel='Within-class distance', ylabel='Between-class distance', colorbar=False,cbarnam='PCD'):
    if xs.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return None

    norm = Normalize(vmin=np.nanmin(cs), vmax=np.nanmax(cs))

    # mask：ACC>=0.95 圆点；ACC<0.95 画“×”
    m_good = accs >= 0.95
    m_bad  = ~m_good

    # 圆点
    sc = ax.scatter(xs[m_good], ys[m_good], c=cs[m_good], cmap=cmap, norm=norm,
                    s=18, edgecolors='none')

    # × 点（同一色带）
    ax.scatter(xs[m_bad], ys[m_bad], c=cs[m_bad], cmap=cmap, norm=norm,
               s=28, marker='x', linewidths=0.9)

    # 轴与样式
    ax.set_xlabel(xlabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.tick_params(labelsize=fontsize_)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_linewidth(linewidth_ax)

    # colorbar 更小
    if colorbar:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)  # 更小的色条
        cbar.set_label(cbarnam, fontsize=fontsize_)
        cbar.ax.tick_params(labelsize=fontsize_)
        vmin, vmax = sc.norm.vmin, sc.norm.vmax
        mid = 0.5 * (vmin + vmax)
        ticks = [vmin, mid, vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    return sc

# ================= 绘图：1×2 子图（mean PCD 与 min PCD） =================
fontsize_ = 8
linewidth_ax = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(5.5, 2.5))
gs = gridspec.GridSpec(1, 2)

# 左：mean PCD
ax1 = fig.add_subplot(gs[0, 0])
xs1, ys1, cs1, accs1 = gather_xyc_with_acc(dict_performs, obj_list, c_key='mean PCD')
scatter_panel(ax1, xs1, ys1, cs1, accs1, cmap='coolwarm',  # 你也可用 'viridis'/'cividis'/'coolwarm'
              fontsize_=fontsize_, linewidth_ax=linewidth_ax, fontname_=fontname_, fontweight_=fontweight_,cbarnam='mean PCD')
# 右：min PCD
ax2 = fig.add_subplot(gs[0, 1])
xs2, ys2, cs2, accs2 = gather_xyc_with_acc(dict_performs, obj_list, c_key='min PCD')
scatter_panel(ax2, xs2, ys2, cs2, accs2, cmap='coolwarm',
              fontsize_=fontsize_, linewidth_ax=linewidth_ax, fontname_=fontname_, fontweight_=fontweight_,cbarnam='min PCD')

plt.subplots_adjust(wspace=0.3, hspace=0.15, right=0.95, left=0.1, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-4.eps'), dpi=600,
            format='eps')
















import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# === 工具函数 ===
def _collect(dict_performs, obj_list, keys):
    """收集给定键的数值；keys 为列表，比如 ['dist_inter','dist_intra','mean PCD','min PCD','ACC']"""
    out = {k: [] for k in keys}
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            ok = True
            vals = {}
            for k in keys:
                if k not in rec:
                    ok = False; break
                try:
                    vals[k] = float(rec[k])
                except Exception:
                    ok = False; break
            if ok and all(np.isfinite(vals[k]) for k in keys):
                for k in keys:
                    out[k].append(vals[k])
    return {k: np.array(v, float) for k, v in out.items()}

def _scatter_one(ax, x, y, color, label, show_trend=True, fontsize_=8):
    ax.scatter(x, y, s=16, alpha=0.9, color=color, edgecolors='none', label=label)
    r = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan
    if show_trend and x.size >= 2:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, lw=0.8, color=color, linestyle='--', alpha=0.9)
        ax.text(0.98, 0.02, f"{label}: r={r:.2f}", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=fontsize_-1, color=color)
    return r

def _scatter_with_bad(ax, x, y, acc, color, label, thr=0.95, show_trend=True, fontsize_=8):
    good = acc >= thr
    bad  = ~good
    # good: 圆点
    ax.scatter(x[good], y[good], s=16, alpha=0.9, color=color, edgecolors='none', label=label)
    # bad: 叉号（同色）
    ax.scatter(x[bad],  y[bad],  s=28, alpha=0.9, color=color, marker='x', linewidths=0.9)
    # 拟合线与 r
    r = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan
    if show_trend and x.size >= 2:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, lw=0.8, color=color, linestyle='--', alpha=0.9)
        ax.text(0.98, 0.02, f"{label}: r={r:.2f}", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=fontsize_-1, color=color)
    return r

# === 统一样式参数 ===
fontsize_ = 8
linewidth_ax = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'

# === 收集所需数据 ===
# 图1需要：dist_inter, dist_intra
d12 = _collect(dict_performs, obj_list, ['dist_inter','dist_intra'])
# 图2需要：dist_inter, mean PCD, min PCD
d2  = _collect(dict_performs, obj_list, ['dist_inter','mean PCD','min PCD'])
# 图3需要：dist_intra, mean PCD, min PCD, ACC
d3  = _collect(dict_performs, obj_list, ['dist_intra','mean PCD','min PCD','ACC'])

# === 1×3 画布 ===
fig = plt.figure(figsize=(8.2, 2.5))
gs  = gridspec.GridSpec(1, 3, wspace=0.35, left=0.08, right=0.98, top=0.92, bottom=0.22)

# ---------- 子图1：dist_inter vs dist_intra ----------
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(d12['dist_inter'], d12['dist_intra'], s=16, alpha=0.9, color='tab:blue', edgecolors='none')
# 拟合线 & r（可留可去）
if d12['dist_inter'].size >= 2:
    coef = np.polyfit(d12['dist_inter'], d12['dist_intra'], 1)
    xx = np.linspace(d12['dist_inter'].min(), d12['dist_inter'].max(), 100)
    yy = np.polyval(coef, xx)
    ax1.plot(xx, yy, lw=0.8, color='k', linestyle='--', alpha=0.7)
    r = np.corrcoef(d12['dist_inter'], d12['dist_intra'])[0, 1]
    ax1.text(0.98, 0.02, f"r={r:.2f}", transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=fontsize_-1)
ax1.set_xlabel('Between-class distance', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax1.set_ylabel('Within-class distance',  fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax1.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax1.spines[sp].set_linewidth(linewidth_ax)
ax1.tick_params(labelsize=fontsize_)

# ---------- 子图2：dist_inter vs mean/min PCD（两色、两拟合线、两个r） ----------
ax2 = fig.add_subplot(gs[0, 1])
r_mean = _scatter_one(ax2, d2['dist_inter'], d2['mean PCD'], color='tab:blue',   label='mean PCD', show_trend=True, fontsize_=fontsize_)
r_min  = _scatter_one(ax2, d2['dist_inter'], d2['min PCD'],  color='tab:orange', label='min PCD',  show_trend=True, fontsize_=fontsize_)
ax2.set_xlabel('Between-class distance', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax2.set_ylabel('PCD',                     fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax2.spines[sp].set_linewidth(linewidth_ax)
ax2.tick_params(labelsize=fontsize_)
ax2.legend(frameon=False, fontsize=fontsize_-1, ncol=2)

# ---------- 子图3：dist_intra vs mean/min PCD（ACC<0.95 标 ×） ----------
ax3 = fig.add_subplot(gs[0, 2])
r_mean3 = _scatter_with_bad(ax3, d3['dist_intra'], d3['mean PCD'], d3['ACC'],
                            color='tab:blue',   label='mean PCD', thr=0.95, show_trend=True, fontsize_=fontsize_)
r_min3  = _scatter_with_bad(ax3, d3['dist_intra'], d3['min PCD'],  d3['ACC'],
                            color='tab:orange', label='min PCD',  thr=0.95, show_trend=True, fontsize_=fontsize_)
ax3.set_xlabel('Within-class distance',  fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax3.set_ylabel('PCD',                     fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax3.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax3.spines[sp].set_linewidth(linewidth_ax)
ax3.tick_params(labelsize=fontsize_)
ax3.legend(frameon=False, fontsize=fontsize_-1, ncol=2)

plt.subplots_adjust(wspace=0.2, hspace=0.15, right=0.95, left=0.1, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-5.eps'), dpi=600,
            format='eps')



from matplotlib.ticker import FormatStrFormatter

# === 1×3 画布 ===
fig = plt.figure(figsize=(8.4, 2.5))
gs  = gridspec.GridSpec(1, 3, wspace=0.35, left=0.08, right=0.98, top=0.92, bottom=0.22)

# ---------- 子图1：dist_inter vs dist_intra ----------
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(d12['dist_inter'], d12['dist_intra'], s=16, alpha=0.9, color='tab:blue', edgecolors='none')
if d12['dist_inter'].size >= 2:
    coef = np.polyfit(d12['dist_inter'], d12['dist_intra'], 1)
    xx = np.linspace(d12['dist_inter'].min(), d12['dist_inter'].max(), 100)
    yy = np.polyval(coef, xx)
    ax1.plot(xx, yy, lw=0.8, color='k', linestyle='--', alpha=0.7)
    r = np.corrcoef(d12['dist_inter'], d12['dist_intra'])[0, 1]
    ax1.text(0.98, 0.02, f"r={r:.2f}", transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=fontsize_-1)
ax1.set_xlabel('Between-class distance', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax1.set_ylabel('Within-class distance',  fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax1.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax1.spines[sp].set_linewidth(linewidth_ax)
ax1.tick_params(labelsize=fontsize_)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# ---------- 子图2：dist_inter (x) vs PCD (y)（两条系列，各自r，legend两行） ----------
ax2 = fig.add_subplot(gs[0, 1])
x2 = d2['dist_inter']
y2_mean = d2['mean PCD']
y2_min  = d2['min PCD']

# 相关系数
r_mean2 = np.corrcoef(x2, y2_mean)[0,1] if x2.size > 1 else float('nan')
r_min2  = np.corrcoef(x2, y2_min )[0,1] if x2.size > 1 else float('nan')

# mean PCD
ax2.scatter(x2, y2_mean, s=16, alpha=0.9, color='tab:blue', edgecolors='none')
if x2.size >= 2:
    coef = np.polyfit(x2, y2_mean, 1)
    xx = np.linspace(x2.min(), x2.max(), 100)
    yy = np.polyval(coef, xx)
    ax2.plot(xx, yy, lw=0.8, color='tab:blue', linestyle='--', alpha=0.9,
             label=f'mean PCD (r={r_mean2:.2f})')

# min PCD
ax2.scatter(x2, y2_min, s=16, alpha=0.9, color='tab:orange', edgecolors='none')
if x2.size >= 2:
    coef = np.polyfit(x2, y2_min, 1)
    yy = np.polyval(coef, xx)  # 复用上面 xx
    ax2.plot(xx, yy, lw=0.8, color='tab:orange', linestyle='--', alpha=0.9,
             label=f'min PCD (r={r_min2:.2f})')

ax2.set_xlabel('Between-class distance', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax2.set_ylabel('PCD',                     fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax2.spines[sp].set_linewidth(linewidth_ax)
ax2.tick_params(labelsize=fontsize_)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.legend(frameon=False, fontsize=fontsize_-1, ncol=1)  # 两行

# ---------- 子图3：dist_intra (x) vs PCD (y)（ACC<0.95 记为×；两条系列，各自r，legend两行） ----------
ax3 = fig.add_subplot(gs[0, 2])
x3 = d3['dist_intra']
y3_mean = d3['mean PCD']
y3_min  = d3['min PCD']
acc3    = d3['ACC']

r_mean3 = np.corrcoef(x3, y3_mean)[0,1] if x3.size > 1 else float('nan')
r_min3  = np.corrcoef(x3, y3_min )[0,1] if x3.size > 1 else float('nan')

# mean：按 ACC 分两种标记
good = acc3 >= 0.95
ax3.scatter(x3[good], y3_mean[good], s=16, alpha=0.9, color='tab:blue', edgecolors='none')
ax3.scatter(x3[~good], y3_mean[~good], s=28, alpha=0.9, color='tab:blue', marker='x', linewidths=0.9)
if x3.size >= 2:
    coef = np.polyfit(x3, y3_mean, 1)
    xx3 = np.linspace(x3.min(), x3.max(), 100)
    yy3 = np.polyval(coef, xx3)
    mean_handle, = ax3.plot(xx3, yy3, lw=0.8, color='tab:blue', linestyle='--', alpha=0.9,
                            label=f'mean PCD (r={r_mean3:.2f})')

# min：同样处理
ax3.scatter(x3[good], y3_min[good], s=16, alpha=0.9, color='tab:orange', edgecolors='none')
ax3.scatter(x3[~good], y3_min[~good], s=28, alpha=0.9, color='tab:orange', marker='x', linewidths=0.9)
if x3.size >= 2:
    coef = np.polyfit(x3, y3_min, 1)
    yy3m = np.polyval(coef, xx3)
    min_handle, = ax3.plot(xx3, yy3m, lw=0.8, color='tab:orange', linestyle='--', alpha=0.9,
                           label=f'min PCD (r={r_min3:.2f})')

ax3.set_xlabel('Within-class distance',  fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax3.set_ylabel('PCD',                     fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax3.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax3.spines[sp].set_linewidth(linewidth_ax)
ax3.tick_params(labelsize=fontsize_)
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.legend(frameon=False, fontsize=fontsize_-1, ncol=1)  # 两行（两条目、1 列）

plt.subplots_adjust(wspace=0.2, hspace=0.15, right=0.95, left=0.1, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-5.eps'), dpi=600, format='eps')
plt.show()



# ---------- 从 dict_performs 中挑选 3 组（按你的新条件） ----------
records = []  # (obj, idx, rec)
for obj, d in dict_performs.items():
    for idx, rec in d.items():
        # 必须具备以下键
        required = ['dist_intra', 'dist_inter', 'mean PCD', 'X_reduced']
        if not all(k in rec for k in required):
            continue
        try:
            di  = float(rec['dist_intra'])
            de  = float(rec['dist_inter'])
            mp  = float(rec['mean PCD'])   # 若想用 min PCD，改成 float(rec['min PCD'])
            Xr  = np.asarray(rec['X_reduced'])
        except Exception:
            continue
        if Xr.shape == (6*18, 2) and np.isfinite([di, de, mp]).all():
            records.append((obj, idx, rec))

if not records:
    raise ValueError("dict_performs 中没有可用记录。")

# ① 最大类间距离
pick_inter_max = max(records, key=lambda t: float(t[2]['dist_inter']))
# ② 最小类内距离
pick_intra_min = min(records, key=lambda t: float(t[2]['dist_intra']))
# ③ 最大 PCD（此处用 mean PCD）
pick_pcd_min   = min(records, key=lambda t: float(t[2]['mean PCD']))  # 或改成 ['min PCD']

# 保持与后续绘图代码兼容的结构（tag 不会被用作标题，后续代码不显示 title）
selections = [
    ('max inter', pick_inter_max),
    ('min intra', pick_intra_min),
    ('min mean PCD', pick_pcd_min),
]


from matplotlib.ticker import FormatStrFormatter

# —— 1) 统计三张图所有点，求统一正方形范围 —— #
xs_all, ys_all = [], []
for _, item in selections:   # selections = [('max inter', ...), ('max intra', ...), ('min mean PCD', ...)]
    _, _, rec = item
    Xr = np.asarray(rec['X_reduced'])
    xs_all.append(Xr[:, 0]); ys_all.append(Xr[:, 1])
xs_all = np.concatenate(xs_all); ys_all = np.concatenate(ys_all)

xy_min = min(xs_all.min(), ys_all.min())
xy_max = max(xs_all.max(), ys_all.max())
pad = 0.05 * (xy_max - xy_min) if xy_max > xy_min else 1.0
lo, hi = xy_min - pad, xy_max + pad

# 统一刻度：比如 5 个刻度（含两端）；你也可改成 6/7 等
num_ticks = 5
ticks_shared = np.linspace(lo, hi, num_ticks)

# —— 2) 画 1×3，并统一范围与刻度 —— #
fig = plt.figure(figsize=(8.5, 2.5))
gs  = gridspec.GridSpec(1, 3)

n_obj, n_view = 6, 18
cmap = plt.cm.tab10

for ii, (_, item) in enumerate(selections):
    ax = fig.add_subplot(gs[0, ii])
    obj, idx, rec = item
    X_reduced = np.asarray(rec['X_reduced'])

    # 绘制 6 个物体闭合轨迹
    for o in range(n_obj):
        s, e = o * n_view, (o + 1) * n_view
        pts = X_reduced[s:e, :]
        pts_closed = np.vstack([pts, pts[0]])
        col = cmap(o % 10)
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], linewidth=linewidth_plot, alpha=0.95, color=col)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, edgecolors='k', facecolors=col, alpha=0.9, linewidths=0.3)

    # —— 统一比例 / 范围 / 刻度 —— #
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks(ticks_shared); ax.set_yticks(ticks_shared)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # 轴标签与样式
    ax.set_xlabel('dim 1', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel('dim 2', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    for sp in ['top','right']: ax.spines[sp].set_linewidth(linewidth_ax)
    for sp in ['bottom','left']: ax.spines[sp].set_linewidth(linewidth_ax)
    ax.tick_params(labelsize=fontsize_, width=linewidth_ax)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname(fontname_); lab.set_fontweight(fontweight_)

plt.subplots_adjust(wspace=0.3, hspace=0.15, right=0.95, left=0.05, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-7.eps'), dpi=600, format='eps')