import os
import pickle
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
from collections import defaultdict

# 1) 取出每个神经元的 200 维类别响应，并做 z-score（防止幅值影响相关性）
def build_resp_matrix(dict_class, zscore=True, drop_constant=True):
    neurons = list(dict_class.keys())
    # 仅保留确实有 'class'->'resp' 的键
    neurons = [n for n in neurons if ('class' in dict_class[n] and 'resp' in dict_class[n]['class'])]
    R = []
    keep_neurons = []
    for n in neurons:
        r = np.asarray(dict_class[n]['class']['resp'], dtype=float).flatten()
        if drop_constant and (np.nanstd(r) == 0 or np.allclose(np.nanstd(r), 0)):
            continue
        if zscore:
            m, s = np.nanmean(r), np.nanstd(r)
            if s == 0 or np.isnan(s):
                continue
            r = (r - m) / s
        # 处理 NaN
        if np.isnan(r).any():
            # 用该神经元非 NaN 均值填补
            r = np.where(np.isnan(r), np.nanmean(r), r)
            if np.isnan(r).any():
                continue
        R.append(r)
        keep_neurons.append(n)
    R = np.vstack(R) if len(R) else np.empty((0,0))
    return keep_neurons, R

# 2) 相关性去重：把 |corr|>=thr 的神经元连边，找连通分量，每个分量只留一个代表
def prune_by_corr(dict_class, thr=0.90, prefer_metric='Separation ratio'):
    neurons, R = build_resp_matrix(dict_class, zscore=True, drop_constant=True)
    if R.shape[0] <= 1:
        return neurons, [], dict_class  # 无需处理

    # 相关矩阵（行=神经元）
    C = np.corrcoef(R)
    np.fill_diagonal(C, 0.0)

    # 构图：|corr| >= thr 视为相连
    N = len(neurons)
    visited = np.zeros(N, dtype=bool)
    components = []

    for i in range(N):
        if visited[i]:
            continue
        # BFS/DFS
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.where(np.abs(C[u]) >= thr)[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

    # 选择代表：优先分离比最高；若没有该指标，选响应方差最大的
    kept_idx = []
    removed_idx = []
    for comp in components:
        if len(comp) == 1:
            kept_idx.append(comp[0])
            continue

        # 计算每个候选的“代表性分数”
        scores = []
        for idx in comp:
            name = neurons[idx]
            score = None
            # 1) Separation ratio（如果有）
            try:
                score = float(dict_class[name]['class']['metrics']['Separation ratio'])
            except Exception:
                score = None
            # 2) 退化：用响应方差
            if score is None:
                score = float(np.var(R[idx]))
            scores.append(score)

        rep_local = comp[int(np.argmax(scores))]
        kept_idx.append(rep_local)
        removed_idx.extend([k for k in comp if k != rep_local])

    kept_idx = sorted(set(kept_idx))
    removed_idx = sorted(set(removed_idx))

    kept = [neurons[i] for i in kept_idx]
    removed = [neurons[i] for i in removed_idx]

    # 构建去重后的 dict
    dict_class_pruned = {k: dict_class[k] for k in kept}

    print(f"[去重完成] 原始: {len(neurons)}  -> 保留: {len(kept)}  删除: {len(removed)}  (阈值 |r|>={thr})")
    return kept, removed, dict_class_pruned


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

def cal_class_performs(resp_split, label_fine):
    #根据排序获取神经元的响应矩阵，也从低到高进行排序
    resp_class=[dict_class[neu]['class']['resp'] for neu in sorted_dict.keys()]
    resp_class=np.array(resp_class)
    name_class=dict_class['ento_exp1_ch14']['class']['name']
    label_fine=[i.split('_')[2] for i in name_class]
    class_fine = list(dict.fromkeys(label_fine))

    n_neuron=80        #小批量神经元的个数
    dict_performs={}

    for nstart in range(0,resp_class.shape[0]-n_neuron+1):
        dict_performs[nstart]={}
        resp_split = resp_class[nstart:nstart + n_neuron, :]
        resp_split=np.transpose(resp_split)
        #解码准确率
        model = SVC(kernel='linear', decision_function_shape='ovr')        #linear
        cv = StratifiedKFold(n_splits=5)  # 创建交叉验证
        y_pred = cross_val_predict(model, resp_split, label_fine, cv=cv)
        accuracy = accuracy_score(label_fine, y_pred)  # 准确率
        print(nstart,accuracy)
        #降维及rdm矩阵
        X_reduced, rdm = analyse_reducedim(resp_split, 'MDS', 2)
        #类内距离、类间距离、分离比
        dist_intra, dist_inter=compute_class_distance(rdm, label_fine)
        Separability = dist_inter / (dist_intra + 1e-9)  # 计算类分离比
        dict_performs[nstart]['Accuracy']=accuracy
        dict_performs[nstart]['Intra-class dist.'] = dist_intra
        dict_performs[nstart]['Inter-class dist.'] = dist_inter
        dict_performs[nstart]['Separability'] = Separability
    return dict_performs


for nn in range(168):
    name='dict_class{}.pkl'.format(nn)
    neuron_path = r'E:\Image_paper\Project/'+name
    with open(neuron_path, "rb") as f:
        dict_class = pickle.load(f)  # 获取全部神经元响应

    # ====== 运行：删除高度相关的神经元 ======
    kept_neurons, removed_neurons, dict_class = prune_by_corr(dict_class, thr=0.90)

    dict_performs={}
    metric_list=['Color','Shape','Texture','V1-like','V2-like',
                 'Alexnet Conv1','Alexnet Conv2','Alexnet Conv3','Alexnet Conv4','Alexnet Conv5']
    for metric_name in metric_list:
        dict_mertics = {}
        for neu in dict_class.keys():
            dict_mertics[neu]=dict_class[neu]['fit_corr'][metric_name]
        sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
        dict_performs[metric_name]=cal_class_performs(sorted_dict, dict_class)



    # save_path = r'E:\Image_paper\Project\dict_performs_class_rbf.pkl'
    # with open(save_path, 'wb') as f:
    #     pickle.dump(dict_performs, f)



    fontsize_ = 8
    linewidth_ax = 0.5
    linewidth_plot = 0.7
    fontname_ = 'Arial'
    fontweight_ = 'normal'
    # 为每条折线指定不同颜色（可按需改）
    line_color = {
        'Selectivity index': '#1f77b4',  # 蓝
        'Separation ratio':  '#ff7f0e',  # 橙
        'Shape':             '#2ca02c',  # 绿
        'Color':             '#e41a1c',  # 红
        'Texture':           '#984ea3',  # 紫
        'Alexnet Conv1':     '#4daf4a',  # 鲜绿
        'Alexnet Conv3':     '#377eb8',  # 靛蓝
        'Alexnet Conv5':     '#a65628',  # 棕
    }


    fig = plt.figure(figsize=(12, 2.8))
    gs = gridspec.GridSpec(1, 4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    n_points = 6  # 取 n 个中心点
    half_window = 10  # 每个点两侧各取 m 个样本
    from matplotlib.lines import Line2D
    plot_performs=['Color', 'Texture', 'Shape', 'Alexnet Conv1', 'Alexnet Conv3', 'Alexnet Conv5']
    for ppf in plot_performs:
        idx = np.array(sorted(dict_performs['Color'].keys()))
        series = {
            'Accuracy':           np.array([dict_performs[ppf][i]['Accuracy']            for i in idx]),
            'Intra-class dist.':  np.array([dict_performs[ppf][i]['Intra-class dist.']   for i in idx]),
            'Inter-class dist.':  np.array([dict_performs[ppf][i]['Inter-class dist.']   for i in idx]),
            'Separability':       np.array([dict_performs[ppf][i]['Separability']        for i in idx]),
        }

        clr = line_color.get(ppf, 'tab:red')  # ← 取到具体颜色字符串
        for ax, (title, y) in zip(axes, series.items()):
            # —— 选中心位置（避开边界，确保有完整窗口）
            idx_min = idx[0] + half_window
            idx_max = idx[-1] - half_window
            centers = np.linspace(idx_min, idx_max, n_points, dtype=int)
            # —— 计算每个中心的窗口均值与 SEM
            mean_vals, sem_vals, center_x = [], [], []
            for c in centers:
                mask = (idx >= c - half_window) & (idx <= c + half_window)
                y_win = y[mask]
                if y_win.size == 0:
                    continue
                mean_vals.append(np.nanmean(y_win))
                sd = np.nanstd(y_win, ddof=1) if y_win.size > 1 else 0.0
                sem_vals.append(sd / np.sqrt(max(y_win.size, 1)))
                center_x.append(c)
            center_x  = np.asarray(center_x)
            mean_vals = np.asarray(mean_vals)
            sem_vals  = np.asarray(sem_vals)
            # —— 仅绘制误差线（均值连线 + 误差棒），不画原始曲线
            ax.errorbar(center_x, mean_vals, yerr=sem_vals,
                        fmt='o-', lw=1.0, ms=3.0, capsize=2,
                        color=clr, ecolor=clr, zorder=3,label=ppf,)
            ax.set_xlabel('Start index', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)

    legend_handles = [Line2D([0], [0], color=line_color[k], marker='o', lw=1.2, label=k)
                      for k in plot_performs]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.995, 1.0),
               ncol=4, frameon=False, fontsize=fontsize_)

    plt.subplots_adjust(wspace=0.35, hspace=0.3, right=0.98, left=0.12, top=0.8, bottom=0.18)
    plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3', name.split('.')[0]+'.png'), dpi=600, format='png')