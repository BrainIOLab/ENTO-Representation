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

neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class= pickle.load(f)  # 获取全部神经元响应+

neuron_path = r'E:\Image_paper\Project\plot\figure6\dict_view_resp.pkl'
with open(neuron_path, "rb") as f:
    dict_view_resp= pickle.load(f)  # 获取全部神经元响应

dict_performs={}
obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']
for obj in obj_list:
    per_bin = 2
    repeats = 100
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

        resp_class_ = [dict_class[neu]['class']['resp'] for neu in neulist]
        resp_class_ = np.transpose(resp_class_)
        labels_=dict_class[neulist[0]]['class']['name']
        labels_=[i.split('_')[2] for i in labels_]

        model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        y_pred = cross_val_predict(model, resp_class_, labels_, cv=cv)
        accuracy = accuracy_score(labels_, y_pred)  # 准确率

        dict_performs[obj][idx]['ACC'] = accuracy
        dict_performs[obj][idx]['dist_intra'] = dist_intra
        dict_performs[obj][idx]['dist_inter'] = dist_inter
        dict_performs[obj][idx]['mean PCD'] = mean_pd
        dict_performs[obj][idx]['min PCD'] = min_pd
        dict_performs[obj][idx]['X_reduced'] = X_reduced



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


d2  = _collect(dict_performs, obj_list, ['ACC','mean PCD','min PCD'])


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 已有：d2 = _collect(dict_performs, obj_list, ['ACC','mean PCD','min PCD'])

fig = plt.figure(figsize=(2.6, 2.5))
gs  = gridspec.GridSpec(1, 1)
ax  = fig.add_subplot(gs[0, 0])

x = d2['ACC']
y_mean = d2['mean PCD']
y_min  = d2['min PCD']

# 计算相关系数
r_mean = np.corrcoef(x, y_mean)[0, 1] if x.size > 1 else np.nan
r_min  = np.corrcoef(x, y_min )[0, 1] if x.size > 1 else np.nan

# 散点
ax.scatter(x, y_mean, s=16, alpha=0.9, color='tab:blue',   edgecolors='none')
ax.scatter(x, y_min,  s=16, alpha=0.9, color='tab:orange', edgecolors='none')

# 拟合线
if x.size >= 2:
    # 用共同的 xx，便于对比
    xx = np.linspace(x.min(), x.max(), 100)
    coef_mean = np.polyfit(x, y_mean, 1)
    coef_min  = np.polyfit(x, y_min,  1)
    ax.plot(xx, np.polyval(coef_mean, xx), lw=0.9, ls='--', color='tab:blue',   alpha=0.9,
            label=f'mean PCD (r={r_mean:.2f})')
    ax.plot(xx, np.polyval(coef_min,  xx), lw=0.9, ls='--', color='tab:orange', alpha=0.9,
            label=f'min PCD (r={r_min:.2f})')
else:
    ax.plot([], [], lw=0.9, ls='--', color='tab:blue',   label='mean PCD')
    ax.plot([], [], lw=0.9, ls='--', color='tab:orange', label='min PCD')

# 轴与样式
ax.set_xlabel('Accuracy', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('PCD',      fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_linewidth(linewidth_ax)
ax.tick_params(labelsize=fontsize_)
ax.legend(frameon=False, fontsize=fontsize_-1, ncol=1)  # legend 中显示 r

plt.subplots_adjust(wspace=0.2, hspace=0.15, right=0.96, left=0.18, top=0.95, bottom=0.2)
# 保存（可选）
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-8.eps'), dpi=600, format='eps')
plt.show()