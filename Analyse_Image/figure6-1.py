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


# 加载神经元响应
neuron_path = r'E:\Image_paper\Project\dict_view.pkl'
with open(neuron_path, "rb") as f:
    dict_view= pickle.load(f)  # 获取全部神经元响应

obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']
neurons = list(dict_view.keys())
N = len(neurons)
V = 18
X = np.full((len(neurons), len(obj_list)*V), np.nan, dtype=float)
labels = []

# 先做列标签（优先使用能取到的 name 作为角度名，否则用 0..17）
for obj in obj_list:
    angle_names = None
    for neu in neurons:
        d = dict_view[neu].get(obj)
        if d and 'name' in d and len(d['name']) == V:
            angle_names = [str(i.split('_')[2]) for i in d['name']]
            break
    labels.extend([f'{ang}' for ang in angle_names])

dict_view_resp={}
for i, neu in enumerate(neurons):
    col = 0

    for obj in obj_list:
        try:
            resp = np.asarray(dict_view[neu][obj]['resp']).reshape(-1)
            vec = np.full(V, np.nan)
            vec[:min(V, len(resp))] = resp[:min(V, len(resp))]
            X[i, col:col+V] = vec
        except KeyError:
            # 该神经元缺这个对象：保持 nan
            pass
        col += V
    dict_view_resp[neu]={}
    dict_view_resp[neu]['resp'] = X[i, :]
    dict_view_resp[neu]['labels']=labels

with open(r'E:\Image_paper\Project\plot\figure6\dict_view_resp.pkl', "wb") as f:
    pickle.dump(dict_view_resp, f)

X_samples = X.T  # (S, N_neuron)
y = np.asarray(labels)  # (S,)

reducedim_method='MDS'
X_reduced, rdm = analyse_reducedim(X_samples, reducedim_method, 2)

# model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
# cv = StratifiedKFold(n_splits=5, shuffle=True)
# y_pred = cross_val_predict(model, X_samples, y, cv=cv)
# accuracy = accuracy_score(y, y_pred)  # 准确率


def per_class_accuracy(y_true, y_pred, classes):
    """返回按 classes 顺序的 per-class accuracy（一类的召回率）"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    denom = cm.sum(axis=1).astype(float)         # 每类真实样本数
    acc = np.divide(np.diag(cm), denom, out=np.full(C, np.nan), where=denom != 0)
    return acc  # shape: (C,)
classes = list(dict.fromkeys(y))
C = len(classes)
from sklearn.model_selection import StratifiedKFold, cross_val_score
acc_by_n = {n: [] for n in range(1, 70,1)}
acc_by_n_2 = {n: [] for n in range(1, 70,1)}

repeats=20
percls_by_n     = {n: np.full((repeats, C), np.nan) for n in range(1, 70,1)}  # ★ 新增：各类准确率
rng = np.random.default_rng(42)
for n in range(1, 70,1):
    print(n)
    for r in range(repeats):
        sel = rng.choice(N, size=n, replace=False)     # 选择 n 个神经元的索引
        X = X_samples[:, sel]                          # (200, n) 样本×特征
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
        scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)  # 5折准确率
        acc_by_n[n].append(float(np.mean(scores)))

        # —— 各类别准确率（放到 array）——
        y_pred = cross_val_predict(model, X, y, cv=cv)
        per_cls_acc = per_class_accuracy(y, y_pred, classes)  # shape (C,)
        percls_by_n[n][r, :] = per_cls_acc





import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ====== 风格参数（两幅子图统一） ======
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

# ====== 1×2 画布 ======
fig = plt.figure(figsize=(2.7, 2.5))
gs = gridspec.GridSpec(1, 1)   # 1 行 2 列

# ========= 子图1：per-class accuracy =========
ax1 = fig.add_subplot(gs[0, 0])

ns_list = sorted(percls_by_n.keys())               # 不同神经元数 n 的有序列表
percls_array = np.stack([percls_by_n[n] for n in ns_list], axis=0)  # (num_n, repeats, C)
percls_mean  = np.nanmean(percls_array, axis=1)    # (num_n, C)
percls_std   = np.nanstd(percls_array, axis=1, ddof=1)

for j, cls in enumerate(classes):
    m = percls_mean[:, j]
    ax1.plot(ns_list, m, '-', lw=linewidth_plot, label=str(cls))

ax1.axhline(1/6, ls='--', lw=0.7, color='k')
ax1.set_yticks([0.2,0.4,0.6,0.8,1.0])
ax1.set_ylim([0.2,1.05])
ax1.set_xlabel('Number of neurons', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax1.set_ylabel('Per-class accuracy', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax1.legend(
    ncol=2, fontsize=fontsize_-1, frameon=False,
    handlelength=1.0, handletextpad=0.3, columnspacing=0.6,
    labelspacing=0.2, borderaxespad=0.2, markerscale=0.8, loc='best'
)
# 统一边框与刻度样式
for spine in ['top','right']:
    ax1.spines[spine].set_visible(False)
for spine in ['bottom','left']:
    ax1.spines[spine].set_linewidth(linewidth_ax)
ax1.tick_params(labelsize=fontsize_, width=linewidth_ax)
for label in ax1.get_xticklabels()+ax1.get_yticklabels():
    label.set_fontname(fontname_); label.set_fontweight(fontweight_)
plt.subplots_adjust(wspace=1.2, hspace=0.15, right=0.95, left=0.2, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-1.eps'), dpi=600,
            format='eps')








# ====== 1×2 画布 ======
fig = plt.figure(figsize=(1.5, 1.5))
gs = gridspec.GridSpec(1, 1)   # 1 行 2 列

# ========= 子图2：MDS降维后连线（仅用 X_reduced） =========
ax2 = fig.add_subplot(gs[0, 0])
n_obj = 6
n_view = 18
assert X_reduced.shape[0] == n_obj * n_view and X_reduced.shape[1] == 2
cmap = plt.cm.tab10
for o in range(n_obj):
    start = o * n_view
    end   = start + n_view
    pts = X_reduced[start:end, :]                # (18, 2)
    pts_closed = np.vstack([pts, pts[0]])        # 首尾相接
    # 画线（闭合）+ 散点
    ax2.plot(pts_closed[:, 0], pts_closed[:, 1],
             linewidth=linewidth_plot, alpha=0.95, color=cmap(o % 10),
             label=obj_list[o])
    ax2.scatter(pts[:, 0], pts[:, 1], s=14, edgecolors='k',
                facecolors=cmap(o % 10), alpha=0.9, linewidths=0.3)
ax2.axis('equal')
ax2.set_xlabel('dim 1', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax2.set_ylabel('dim 2', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
# ax2.legend(frameon=False, fontsize=fontsize_-1, ncol=2)
# 统一边框与刻度样式（与子图1一致）
for spine in ['top','right']:
    ax2.spines[spine].set_linewidth(linewidth_ax)
for spine in ['bottom','left']:
    ax2.spines[spine].set_linewidth(linewidth_ax)
ax2.tick_params(labelsize=fontsize_, width=linewidth_ax)
for label in ax2.get_xticklabels()+ax2.get_yticklabels():
    label.set_fontname(fontname_); label.set_fontweight(fontweight_)
ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)  # 隐藏刻度线与刻度文字
ax2.set_xlabel(''); ax2.set_ylabel('')                                         # 去掉轴标签

# （可选）确保没有自定义刻度残留
ax2.set_xticks([]); ax2.set_yticks([])
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure6', 'figure6-2.eps'), dpi=600,
            format='eps')
