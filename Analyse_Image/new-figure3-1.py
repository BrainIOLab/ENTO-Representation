
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn.metrics import pairwise_distances
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

def rdm_spearman(rdm1: np.ndarray, rdm2: np.ndarray):

    iu = np.triu_indices_from(rdm1, k=1)
    v1 = rdm1[iu].astype(float)
    v2 = rdm2[iu].astype(float)
    rho, p = spearmanr(v1, v2)
    r, p_p = pearsonr(v1, v2)
    return float(rho), r

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
#选择一个神经元集群
numbers=[65,73,81,85,95,99,107,123,132,135,141,142,149,157,164,165]
nnb=1
for nnb in numbers:
    name = 'dict_class{}.pkl'.format(nnb)
    neuron_path = r'E:\Image_paper\Project/' + name
    with open(neuron_path, "rb") as f:
        dict_class = pickle.load(f)  # 获取全部神经元响应

    kept_neurons, removed_neurons, dict_class_kept = prune_by_corr(dict_class, thr=0.90)

    #整体神经元集群降维
    reducedim_method='MDS'      # Isomap,  tSNE,  MDS
    resp_all = np.array([dict_class[k]['class']['resp'] for k in dict_class])
    resp_all = np.transpose(resp_all)
    name_class = dict_class[list(dict_class.keys())[0]]['class']['name']
    label_fine = np.array([i.split('_')[2] for i in name_class])
    class_fine = list(dict.fromkeys(label_fine))
    label_coarse = np.array([i.split('_')[1] for i in name_class])
    class_coarse = list(dict.fromkeys(label_coarse))

    REPEATS = 20                #重复多少次
    cv_splits = 5               #交叉验证
    N_samples, N_neurons = resp_all.shape
    N_classes = len(class_fine)


    #遍历每种特征
    feature_path = r'E:\Image_paper\Project\dict_img_feature.pkl'
    with open(feature_path, "rb") as f:
        img_dict = pickle.load(f)  # 图像特征字典，每张图像是一个键
    feat_list=['Color','Shape','Texture', 'Alexnet Conv1', 'Alexnet Conv3', 'Alexnet Conv5']
    n_dim = 3
    reducedim_method = 'MDS'
    dict_rdm = {}
    for ft in feat_list:
        print(ft)
        feat_resp = []
        for img in name_class:
            feat_resp.append(img_dict[img][ft])
        feat_resp = np.array(feat_resp)
        _, rdm_feature = analyse_reducedim(feat_resp, reducedim_method, n_dim)
        dict_rdm[ft]=rdm_feature


    n_neuron = 80  # 小批量神经元的个数
    dict_performs={}
    for ft in feat_list:
        dict_performs[ft]={}
        dict_mertics = {}
        for neu in dict_class_kept.keys():
            dict_mertics[neu] = dict_class_kept[neu]['fit_corr'][ft]
        sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
        resp_class = [dict_class_kept[neu]['class']['resp'] for neu in sorted_dict.keys()]
        resp_class = np.array(resp_class)
        acc_list=[]
        rdmsp_list=[]
        range_list=list(range(0, resp_class.shape[0] - n_neuron + 1))
        for nstart in range_list:
            print(nstart)
            resp_split = resp_class[nstart:nstart + n_neuron, :]
            resp_split = np.transpose(resp_split)
            # 解码准确率
            model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
            cv = StratifiedKFold(n_splits=5)
            y_pred = cross_val_predict(model, resp_split, label_fine, cv=cv)
            accuracy = accuracy_score(label_fine, y_pred)  # 准确率
            print(ft, nstart, accuracy)
            acc_list.append(accuracy)
            _, rdm_neurons = analyse_reducedim(resp_split, reducedim_method, n_dim)
            r, _ = rdm_spearman(dict_rdm[ft], rdm_neurons)
            rdmsp_list.append(r)
        dict_performs[ft]['acc']=acc_list
        dict_performs[ft]['rdm'] = rdmsp_list

    for ft in ['Selectivity index','Separation ratio']:
        dict_performs[ft]={}
        dict_mertics = {}
        for neu in dict_class_kept.keys():
            dict_mertics[neu] = dict_class_kept[neu]['class']['metrics'][ft]
        sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
        resp_class = [dict_class_kept[neu]['class']['resp'] for neu in sorted_dict.keys()]
        resp_class = np.array(resp_class)
        acc_list=[]
        range_list=list(range(0, resp_class.shape[0] - n_neuron + 1))
        for nstart in range_list:
            print(nstart)
            resp_split = resp_class[nstart:nstart + n_neuron, :]
            resp_split = np.transpose(resp_split)
            # 解码准确率
            model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
            cv = StratifiedKFold(n_splits=5)
            y_pred = cross_val_predict(model, resp_split, label_fine, cv=cv)
            accuracy = accuracy_score(label_fine, y_pred)  # 准确率
            print(ft, nstart, accuracy)
            acc_list.append(accuracy)
        dict_performs[ft]['acc']=acc_list


    pkl_path=os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'r_{}.pickle'.format(name.split('.')[0]))
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict_performs, f)


    pkl_path = os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'acc_overall.pickle')
    with open(pkl_path, 'rb') as f:
        acc_overall = pickle.load(f)

    # 加载 acc_per_class
    pkl_path = os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'acc_per_class.pickle')
    with open(pkl_path, 'rb') as f:
        acc_per_class = pickle.load(f)

    #绘制子群的性能变化##################################################################################################
    fontsize_ = 8
    linewidth_ax = 0.5
    linewidth_plot = 0.7
    fontname_ = 'Arial'
    fontweight_ = 'normal'
    n_list = sorted(set(list(range(1, N_neurons + 1, 10)) + [N_neurons]))

    fig = plt.figure(figsize=(8.5, 2.5))
    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0])

    xs = np.array(n_list)  # x 轴：实际评估过的神经元数量
    mean_overall = acc_overall[xs - 1].mean(axis=1)
    mean_per_class = {c: acc_per_class[c][xs - 1].mean(axis=1) for c in class_fine}
    # === 画图（风格不变） ===
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[::2]  # 0,2,4,... 共10色
    for ci, c in enumerate(class_fine):
        ax.plot(xs, mean_per_class[c],linewidth=linewidth_plot,label=c,color=colors[ci % len(colors)])
    ax.plot(xs, mean_overall,linewidth=1.2,linestyle='--',color='k',label='Overall')
    ax.set_xlabel('Number of neurons', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel('Decoding accuracy', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(0.0, 0.8)
    ax.set_yticks([0,0.2,0.4,0.6,0.8])
    ax.set_xticks([0,50,100,150,200,250])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.tick_params(labelsize=fontsize_)


    ax = fig.add_subplot(gs[0, 1])

    n_points    = 6
    half_window = 10
    idx = np.asarray(range_list, dtype=int)
    L = idx.size
    win_size = 2*half_window + 1
    if L < win_size:
        raise ValueError(f"L={L} < window={win_size}, 无法形成完整窗口")
    idx_min = idx[0] + half_window
    idx_max = idx[-1] - half_window
    centers = np.linspace(idx_min, idx_max, n_points)
    centers = np.rint(centers).astype(int)
    centers = np.unique(np.clip(centers, idx_min, idx_max))
    for i, ft in enumerate(dict_performs.keys()):
        y = np.asarray(dict_performs[ft]['acc'], dtype=float)
        if y.size != L:
            raise ValueError(f"{ft}: y.size={y.size} 与 idx.size={L} 不一致")
        mean_vals, sem_vals, center_x = [], [], []
        for c in centers:
            mask = (idx >= c - half_window) & (idx <= c + half_window)
            y_win = y[mask]
            if y_win.size == 0:
                continue
            m  = np.nanmean(y_win)
            sd = np.nanstd(y_win, ddof=1) if y_win.size > 1 else 0.0   # ← SD（不除以 sqrt(n)）
            sem = sd / np.sqrt(max(y_win.size, 1))      # ← SEM
            mean_vals.append(m); sem_vals.append(sem); center_x.append(c)
        if not center_x:
            continue
        center_x  = np.asarray(center_x)
        mean_vals = np.asarray(mean_vals)
        sem_vals = np.asarray(sem_vals)
        ax.errorbar(center_x, mean_vals, yerr=sem_vals,   # ← 用 SD 作为误差
                    fmt='o-', lw=linewidth_plot, ms=3.5, capsize=2,
                    label=ft, zorder=3)

    ax.set_xlabel('Fitting degree', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel('Decoding accuracy',          fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.set_ylim(0.59, 0.663)
    ax.set_yticks([0.60,0.62,0.64,0.66])
    # ax.set_xticks([0,50,100,150])
    ax.tick_params(labelsize=fontsize_)


    ax = fig.add_subplot(gs[0, 2])
    n_points    = 6
    half_window = 10
    idx = np.asarray(range_list, dtype=int)
    L = idx.size
    win_size = 2*half_window + 1
    if L < win_size:
        raise ValueError(f"L={L} < window={win_size}, 无法形成完整窗口")
    idx_min = idx[0] + half_window
    idx_max = idx[-1] - half_window
    centers = np.linspace(idx_min, idx_max, n_points)
    centers = np.rint(centers).astype(int)
    centers = np.unique(np.clip(centers, idx_min, idx_max))
    for i, ft in enumerate(feat_list):
        y = np.asarray(dict_performs[ft]['rdm'], dtype=float)
        if y.size != L:
            raise ValueError(f"{ft}: y.size={y.size} 与 idx.size={L} 不一致")
        mean_vals, sem_vals, center_x = [], [], []
        for c in centers:
            mask = (idx >= c - half_window) & (idx <= c + half_window)
            y_win = y[mask]
            if y_win.size == 0:
                continue
            m  = np.nanmean(y_win)
            sd = np.nanstd(y_win, ddof=1) if y_win.size > 1 else 0.0   # ← SD（不除以 sqrt(n)）
            sem = sd / np.sqrt(max(y_win.size, 1))      # ← SEM
            mean_vals.append(m); sem_vals.append(sem); center_x.append(c)
        if not center_x:
            continue
        center_x  = np.asarray(center_x)
        mean_vals = np.asarray(mean_vals)
        sem_vals = np.asarray(sem_vals)
        ax.errorbar(center_x,mean_vals, yerr=sem_vals,   # ← 用 SD 作为误差
                    fmt='o-', lw=linewidth_plot, ms=3.5, capsize=2,
                    label=ft, zorder=3)

    ax.set_xlabel('Fitting degree', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel('Decoding accuracy',          fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    # ax.set_yticks([0,0.1,0.2,0.3,0.4])
    # ax.set_xticks([0,50,100,150])
    ax.tick_params(labelsize=fontsize_)
    plt.subplots_adjust(wspace=0.4, hspace=0.1, right=0.95, left=0.1, top=0.95, bottom=0.15)
    plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'figure3_performs_temp1.eps'), dpi=600, format='eps')

