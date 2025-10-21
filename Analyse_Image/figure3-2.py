import os
import pickle
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score
import matplotlib.pyplot as plt
from matplotlib import gridspec



neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应


neurons = list(dict_class.keys())              # len = 253
N = len(neurons)
# 构造 (N, 200) 的响应矩阵：神经元 × 刺激
resp_all = np.array([dict_class[neu]['class']['resp'] for neu in neurons])  # (N, 200)
name_any = dict_class[neurons[0]]['class']['name']  # 用任一神经元的 name 顺序
labels_fine = np.array([n.split('_')[2] for n in name_any])  # 形如 str 标签
y = labels_fine
labels = np.array([n.split('_')[1] for n in name_any])  # 形如 str 标签
y2 = labels
def per_class_accuracy(y_true, y_pred, classes):
    """返回按 classes 顺序的 per-class accuracy（一类的召回率）"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    denom = cm.sum(axis=1).astype(float)         # 每类真实样本数
    acc = np.divide(np.diag(cm), denom, out=np.full(C, np.nan), where=denom != 0)
    return acc  # shape: (C,)
classes = list(dict.fromkeys(labels_fine))
C = len(classes)
from sklearn.model_selection import StratifiedKFold, cross_val_score
acc_by_n = {n: [] for n in range(1, 250,20)}
acc_by_n_2 = {n: [] for n in range(1, 250,20)}

repeats=50
percls_by_n     = {n: np.full((repeats, C), np.nan) for n in range(1, 250, 20)}  # ★ 新增：各类准确率
rng = np.random.default_rng(42)
for n in range(1, 250,20):
    print(n)
    for r in range(repeats):
        sel = rng.choice(N, size=n, replace=False)     # 选择 n 个神经元的索引
        X = resp_all[sel, :].T                          # (200, n) 样本×特征
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
        scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)  # 5折准确率
        acc_by_n[n].append(float(np.mean(scores)))

        # —— 各类别准确率（放到 array）——
        y_pred = cross_val_predict(model, X, y, cv=cv)
        per_cls_acc = per_class_accuracy(y, y_pred, classes)  # shape (C,)
        percls_by_n[n][r, :] = per_cls_acc

        cv = StratifiedKFold(n_splits=5, shuffle=True)
        model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
        scores = cross_val_score(model, X, y2, cv=cv, n_jobs=-1)  # 5折准确率
        acc_by_n_2[n].append(float(np.mean(scores)))


fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(4.6, 2.3))
gs = gridspec.GridSpec(1, 2)   # 1 行 4 列
ax = fig.add_subplot(gs[0, 0])

ns    = np.array(sorted(acc_by_n.keys()))
means = np.array([np.mean(acc_by_n[n]) for n in ns])
std   = np.array([np.std(acc_by_n[n], ddof=1) for n in ns])
sem   = std / np.sqrt([len(acc_by_n[n]) for n in ns])  # repeats 次数
ax.errorbar(ns, means, yerr=std, fmt='o-', lw=linewidth_plot, ms=3.0, capsize=2,
            color='r', ecolor='r', elinewidth=linewidth_plot)
ax.axhline(1/8, ls='--', lw=0.7, color='r', label='chance (0.125)')
ns    = np.array(sorted(acc_by_n_2.keys()))
means = np.array([np.mean(acc_by_n_2[n]) for n in ns])
std   = np.array([np.std(acc_by_n_2[n], ddof=1) for n in ns])
sem   = std / np.sqrt([len(acc_by_n_2[n]) for n in ns])  # repeats 次数
ax.errorbar(ns, means, yerr=std, fmt='o-', lw=1.0, ms=3.0, capsize=2,
            color='k', ecolor='k', elinewidth=0.8)
ax.axhline(1/5, ls='--', lw=0.7, color='k', label='chance (0.5)')
ax.set_xlabel('Number of neurons', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Decoding accuracy', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylim(0, 0.8)
ax.set_xlim(0, ns.max()+1)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.legend(frameon=False, fontsize=fontsize_)


ax = fig.add_subplot(gs[0, 1])
ns_list = sorted(percls_by_n.keys())               # 不同神经元数 n 的有序列表
percls_array = np.stack([percls_by_n[n] for n in ns_list], axis=0)  # (num_n, repeats, C)
percls_mean  = np.nanmean(percls_array, axis=1)    # (num_n, C)
percls_std   = np.nanstd(percls_array, axis=1, ddof=1)  # (num_n, C)
for j, cls in enumerate(classes):
    m = percls_mean[:, j]          # (num_n,)
    s = percls_std[:, j]
    ax.plot(ns_list, m, '-', lw=0.9, label=str(cls))
ax.axhline(1/8, ls='--', lw=0.7, color='k')
ax.set_xlabel('Number of neurons', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Per-class accuracy', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylim(0, 0.8)
ax.legend(
    ncol=2,
    fontsize=fontsize_-1,   # 字号略小
    frameon=False,
    handlelength=1.0,       # 线段更短
    handletextpad=0.3,      # 线与文字的间距
    columnspacing=0.6,      # 列间距
    labelspacing=0.2,       # 行间距
    borderaxespad=0.2,      # 图例与轴的间距
    markerscale=0.8,        # 点标记缩小
    loc='best'       # 位置可按需改
)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)


plt.subplots_adjust(wspace=0.4, hspace=0.35, right=0.95, left=0.1, top=0.90, bottom=0.2)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3', 'figure3_3.eps'), dpi=600, format='eps')




