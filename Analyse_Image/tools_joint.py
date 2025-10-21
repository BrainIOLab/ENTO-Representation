import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

def plot_rdm_class(rdm):
    plt.figure(figsize=(3, 3))
    sns.heatmap(rdm, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=True)
    plt.tight_layout()
    plt.show()

def build_view_rdm(n_angles=36):
    angles = np.linspace(0, 360, n_angles, endpoint=False)  # shape: (36,)
    angle_diff = np.abs(angles[:, None] - angles[None, :])  # |θ_i - θ_j|, shape: (36, 36)
    circular_diff = np.minimum(angle_diff, 360 - angle_diff)
    rdm_target = circular_diff / 180.0
    return rdm_target

def build_class_rdm(n_objects,n_classes,labels):
    # 计算目标rdm矩阵，归一化到0~1
    objects_per_class = n_objects // n_classes
    # 构建 RDM：类内 0，类间 1
    rdm_target = np.zeros((n_objects, n_objects))
    for i in range(n_objects):
        for j in range(n_objects):
            rdm_target[i, j] = 0 if labels[i] == labels[j] else 1
    return rdm_target

def circular_gaussian(theta, mu, sigma, amp=None):
    #根据高斯调谐参数生成响应矩阵
    delta = np.angle(np.exp(1j * np.deg2rad(theta[:, None] - mu[None, :])), deg=True)
    base_response = np.exp(-0.5 * (delta / sigma[None, :]) ** 2).T
    if amp is not None:
        return amp[:, None] * base_response
    return base_response

def cal_class_response(params_class, config):
    n = config.num_neurons
    mu = params_class[:n] % config.num_categories
    amp = params_class[n:2*n]
    sigma = params_class[2*n:]
    x = np.arange(config.num_categories)  # 类别索引为 0 到 num_categories - 1
    resp_class = np.zeros((n, len(x)))
    for i in range(n):
        dist = np.minimum(np.abs(x - mu[i]), config.num_categories - np.abs(x - mu[i]))
        resp_class[i] = amp[i] * np.exp(-0.5 * (dist / sigma[i]) ** 2)
    return resp_class

def combine_view_class_response(resp_class, resp_view, mode='multiplicative'):
    n_neurons, n_categories = resp_class.shape
    _, n_angles = resp_view.shape
    # 初始化结果张量
    resp_all = np.zeros((n_neurons, n_categories, n_angles))
    for i in range(n_neurons):
        class_response = resp_class[i][:, np.newaxis]  # shape: (n_categories, 1)
        view_response = resp_view[i][np.newaxis, :]  # shape: (1, n_angles)
        if mode == 'multiplicative':
            resp_all[i] = class_response * view_response
        elif mode == 'additive':
            resp_all[i] = class_response + view_response
        elif mode == 'max':
            resp_all[i] = np.maximum(class_response, view_response)
        elif mode == 'min':
            resp_all[i] = np.minimum(class_response, view_response)
        elif mode == 'mean':
            resp_all[i] = (class_response + view_response) / 2
        else:
            raise ValueError(f"不支持的联合方式: {mode}")
    return resp_all

def plot_combined_tuning(resp_view, resp_class, resp_joint, config,savepath):
    num_neurons = resp_joint.shape[0]
    n_cols = 5  # 三维图每行子图数量
    n_rows_3d = int(np.ceil(num_neurons / n_cols))  # 3D子图行数
    fig = plt.figure(figsize=(n_cols * 2.5, (1 + n_rows_3d) * 2.5))
    gs = gridspec.GridSpec(1 + n_rows_3d, n_cols, height_ratios=[1] + [3]*n_rows_3d)

    angles = config.angles
    categories = np.arange(config.num_categories)
    n_cat, n_ang = resp_joint.shape[1], resp_joint.shape[2]
    X, Y = np.meshgrid(np.arange(n_ang), np.arange(n_cat))

    # ==== 第一行：视角调谐曲线 ====
    ax0 = fig.add_subplot(gs[0, 0:2])
    for i in range(resp_view.shape[0]):
        ax0.plot(angles, resp_view[i], lw=0.8, alpha=0.8)
    ax0.set_title('View Tuning')
    ax0.set_xlabel('Angle')
    ax0.set_ylabel('Response')
    ax0.tick_params(labelsize=7)

    # ==== 第一行：类别调谐曲线 ====
    ax1 = fig.add_subplot(gs[0, 3:5])
    for i in range(resp_class.shape[0]):
        ax1.plot(categories, resp_class[i], lw=0.8, alpha=0.8)
    ax1.set_title('Class Tuning')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Response')
    ax1.tick_params(labelsize=7)

    # ==== 第二/三行：联合调谐 3D 图 ====
    for idx in range(num_neurons):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col], projection='3d')
        Z = resp_joint[idx]
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.95)

        ax.set_title(f'Neuron {idx}', fontsize=9)
        ax.set_xlabel('Angle Idx', fontsize=8)
        ax.set_ylabel('Class Idx', fontsize=8)
        ax.set_zlabel('Resp', fontsize=8)
        ax.tick_params(labelsize=6, pad=0.5)
        ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_combined_tuning_continuously(resp_view1, resp_view2, resp_joint, config, savepath):
    num_neurons = resp_joint.shape[0]
    n_cols = 5  # 三维图每行子图数量
    n_rows_3d = int(np.ceil(num_neurons / n_cols))  # 3D子图行数
    fig = plt.figure(figsize=(n_cols * 2.5, (1 + n_rows_3d) * 2.5))
    gs = gridspec.GridSpec(1 + n_rows_3d, n_cols, height_ratios=[1] + [3]*n_rows_3d)

    n_cat, n_ang = resp_joint.shape[1], resp_joint.shape[2]
    X, Y = np.meshgrid(np.arange(n_ang), np.arange(n_cat))
    angles1=config.angles1
    # ==== 第一行：视角调谐曲线 ====
    ax0 = fig.add_subplot(gs[0, 0:2])
    for i in range(resp_view1.shape[0]):
        ax0.plot(angles1, resp_view1[i], lw=0.8, alpha=0.8)
    ax0.set_title('View Tuning')
    ax0.set_xlabel('Angle')
    ax0.set_ylabel('Response')
    ax0.tick_params(labelsize=7)

    # ==== 第一行：类别调谐曲线 ====
    ax1 = fig.add_subplot(gs[0, 3:5])
    angles2 = config.angles2
    for i in range(resp_view2.shape[0]):
        ax1.plot(angles2, resp_view2[i], lw=0.8, alpha=0.8)
    ax1.set_title('Class Tuning')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Response')
    ax1.tick_params(labelsize=7)

    # ==== 第二/三行：联合调谐 3D 图 ====
    for idx in range(num_neurons):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col], projection='3d')
        Z = resp_joint[idx]
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.95)

        ax.set_title(f'Neuron {idx}', fontsize=9)
        ax.set_xlabel('Angle Idx', fontsize=8)
        ax.set_ylabel('Class Idx', fontsize=8)
        ax.set_zlabel('Resp', fontsize=8)
        ax.tick_params(labelsize=6, pad=0.5)
        ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()

from sklearn.manifold import Isomap, TSNE, MDS, LocallyLinearEmbedding
# import umap
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
def compute_geodesic_rdm_class(R, method='isomap', n_neighbors=10, n_components=3, metric='euclidean'):
    # 已知响应矩阵，计算测地距离
    X = R.T  # 转为 samples × features
    if method == 'isomap':
        model = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric=metric)
        model.fit(X)
        D = model.dist_matrix_
        embedded = model.transform(X)
    elif method == 'pca':
        model = PCA(n_components=n_components)
        embedded = model.fit_transform(X)
        D = squareform(pdist(X, metric=metric))
    # elif method == 'umap':
    #     model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric)
    #     embedded = model.fit_transform(X)
    #     D = squareform(pdist(embedded, metric=metric))
    elif method == 'tsne':
        model = TSNE(n_components=n_components, perplexity=n_neighbors, metric=metric)
        embedded = model.fit_transform(X)
        D = squareform(pdist(embedded, metric=metric))
    elif method == 'mds':
        rdm = squareform(pdist(X, metric='euclidean'))
        # MDS 嵌入
        model = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', random_state=0)
        embedded = model.fit_transform(rdm)
        D = squareform(pdist(embedded, metric=metric))
    elif method == 'lle':
        model = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
        embedded = model.fit_transform(X)
        D = squareform(pdist(embedded, metric=metric))

    else:
        raise ValueError(f"Unsupported method: {method}")
    # 归一化距离
    D_norm = (D - D.min()) / (D.max() - D.min() + 1e-8)
    return D, D_norm, embedded

def view_rdm_from_joint(rdm_joint, config):
    num_classes = config.num_categories
    views_per_class = config.objects_per_category  # e.g. 36
    sub_rdm_all = np.zeros((num_classes, views_per_class, views_per_class))
    sub_rdm_norm_all = np.zeros_like(sub_rdm_all)
    for i in range(num_classes):
        idx_start = i * views_per_class
        idx_end = (i + 1) * views_per_class
        sub_rdm = rdm_joint[idx_start:idx_end, idx_start:idx_end]
        sub_rdm_norm = (sub_rdm - sub_rdm.min()) / (sub_rdm.max() - sub_rdm.min() + 1e-8)
        sub_rdm_all[i] = sub_rdm
        sub_rdm_norm_all[i] = sub_rdm_norm
    return sub_rdm_all, sub_rdm_norm_all

def cal_rdm_loss(D, D_target):
    losses =  np.mean((D - D_target) ** 2, axis=(1, 2))
    mean_loss = np.mean(losses)  # 所有类别的平均损失
    return mean_loss,losses

def safe_log(x, eps=1e-6):
    if x+ eps<0:
        x=0
    return np.log(x + eps)

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

from sklearn.model_selection import cross_val_score
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
def compute_accu_ri(resp, labels, per=0.1, repeat=50):
    #冗余定义为：重复50次随机丢失10%神经元后的准确率缺失平均
    n_neurons = resp.shape[0]
    n_drop = max(1, int(n_neurons * per))
    base_acc = np.mean(cross_val_score(SVC(kernel='rbf'), resp.T, labels, cv=5))
    drops = []
    for _ in range(repeat):
        drop_indices = random.sample(range(n_neurons), n_drop)
        resp_dropped = np.delete(resp, drop_indices, axis=0)
        drop_acc = np.mean(cross_val_score(SVC(kernel='rbf'), resp_dropped.T, labels, cv=5))
        drops.append(base_acc - drop_acc)
    return base_acc,np.mean(drops)

def compute_structure_auc(rdm, rdm_target):
    try:
        auc = roc_auc_score(rdm_target.flatten(), rdm.flatten())
    except:
        auc = np.nan  # e.g., if rdm_target is all 1 or all 0
    return auc

def cal_joint_performs(rdm_joint,config,resp_joint_reshape,embedded_joint):
    rdm_class=rdm_joint.copy()
    rdm_view,rdm_view_norm=view_rdm_from_joint(rdm_joint, config)

    dict_={}
    rdm_view_target = config.view_target_rdm
    rdm_loss,_ = cal_rdm_loss(rdm_view_norm, rdm_view_target)  # 计算rdm间的均方误差和其log
    dict_['RDM mse']=rdm_loss
    dict_['Log-RDM mse'] = safe_log(rdm_loss)
    dict_['embedded_joint'] = embedded_joint

    dict_['Intra-class distance'],dict_['Inter-class distance']=compute_class_distance(rdm_class, config.labels)     #计算类内距离和类间距离
    dict_['Separation ratio'] = dict_['Inter-class distance'] / (dict_['Intra-class distance'] + 1e-9)  # 计算类分离比
    dict_['Accuracy'], dict_['Redundancy'] = compute_accu_ri(resp_joint_reshape, config.labels, per=0.0, repeat=50)
    dict_['AUC'] = compute_structure_auc(rdm_class, config.class_target_rdm)
    return dict_

def cal_joint_performs_continuously(rdm_joint,config):
    num_angles = config.angles1.shape[0]  # 36
    view1_losses = []
    for a1 in range(num_angles):
        idxs = [a1 * num_angles + a2 for a2 in range(num_angles)]
        sub_rdm = rdm_joint[np.ix_(idxs, idxs)]
        loss = np.mean((sub_rdm - config.view_target_rdm1) ** 2)
        view1_losses.append(loss)
    mse_view1 = np.mean(view1_losses)
    log_mse_view1 = np.log(mse_view1 + 1e-9)

    view2_losses = []
    for a2 in range(num_angles):
        idxs = [a1 * num_angles + a2 for a1 in range(num_angles)]
        sub_rdm = rdm_joint[np.ix_(idxs, idxs)]
        loss = np.mean((sub_rdm - config.view_target_rdm2) ** 2)
        view2_losses.append(loss)

    mse_view2 = np.mean(view2_losses)
    log_mse_view2 = np.log(mse_view2 + 1e-9)

    dict_={}
    dict_['log_mse_view1']=log_mse_view1
    dict_['log_mse_view2'] = log_mse_view2

    return dict_


from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes

def compute_isomap_rdm(R, n_neighbors=10):
    #已知响应矩阵，计算测地距离
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2, metric='euclidean')
    isomap.fit(R.T)
    D = isomap.dist_matrix_
    D_norm = (D - D.min()) / (D.max() - D.min() + 1e-8)
    embedded = isomap.transform(R.T)  # 获取降维后的样本位置（2D）
    return D, D_norm, embedded

def procrustes_disc(resp,resp_noise):
    # Isomap 嵌入结果, 比较两个低维点集的形状相似度（允许旋转、缩放、平移）
    X_clean = compute_isomap_rdm(resp)[2]  # embedded
    X_noisy = compute_isomap_rdm(resp_noise)[2]
    # 计算 Procrustes 距离
    _, _, disparity = procrustes(X_clean, X_noisy)
    return disparity

def neighborhood_preservation(resp, resp_noise, k=10):
    #对比每个样本在原始空间与扰动后嵌入中，k - 近邻是否一致, 值域 [0, 1]，越高表示局部结构越稳定
    X_clean = compute_isomap_rdm(resp)[2]  # embedded
    X_noisy = compute_isomap_rdm(resp_noise)[2]
    nbrs1 = NearestNeighbors(n_neighbors=k + 1).fit(X_clean)
    nbrs2 = NearestNeighbors(n_neighbors=k + 1).fit(X_noisy)
    indices1 = nbrs1.kneighbors(return_distance=False)[:, 1:]
    indices2 = nbrs2.kneighbors(return_distance=False)[:, 1:]
    # 逐样本比较邻居是否重叠
    overlap = [len(set(indices1[i]) & set(indices2[i])) / k for i in range(X_clean.shape[0])]
    return np.mean(overlap)

def cal_number_performs(rdm_joint,config,resp_joint_reshape,resp_joint,resp_view):
    rdm_class=rdm_joint.copy()
    rdm_view,rdm_view_norm=view_rdm_from_joint(rdm_joint, config)

    dict_={}
    rdm_view_target = config.view_target_rdm
    proc_disc=[]
    neigh_preser=[]
    for n in range(resp_joint.shape[1]):
        proc_disc.append(procrustes_disc(resp_view, resp_joint[:,n,:]))
        neigh_preser.append(neighborhood_preservation(resp_view, resp_joint[:,n,:], k=10))
    dict_['PD']=np.mean(proc_disc)
    dict_['NP'] = np.mean(neigh_preser)
    dict_['Accuracy'], _ = compute_accu_ri(resp_joint_reshape, config.labels, per=0.0, repeat=50)
    dict_['AUC'] = compute_structure_auc(rdm_class, config.class_target_rdm)
    return dict_

def plot_class_rdm_comparison_with_gridspec(class_target_rdm, rdm_joint,savepath):
    fig = plt.figure(figsize=(5.5, 2.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(class_target_rdm, cmap='coolwarm', xticklabels=False, yticklabels=False, ax=ax1, cbar=True)
    ax1.set_title('Target Class RDM', fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(rdm_joint, cmap='coolwarm', xticklabels=False, yticklabels=False, ax=ax2, cbar=True)
    ax2.set_title('Joint Manifold RDM', fontsize=10)

    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_per_object_view_rdm_from_joint(rdm_joint, config, n_cols,savepath):
    import math
    num_classes = config.num_categories
    views_per_class = config.objects_per_category

    n_rows = math.ceil(num_classes / n_cols)
    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2))
    gs = gridspec.GridSpec(n_rows, n_cols)

    for i in range(num_classes):
        idx_start = i * views_per_class
        idx_end = (i + 1) * views_per_class

        # 取该类的子矩阵并归一化
        sub_rdm = rdm_joint[idx_start:idx_end, idx_start:idx_end]
        sub_rdm_norm = (sub_rdm - sub_rdm.min()) / (sub_rdm.max() - sub_rdm.min() + 1e-8)

        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        sns.heatmap(sub_rdm_norm, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=False, ax=ax)
        ax.set_title(f'Class {i}', fontsize=9)

    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_3d_trajectory_by_class(embedded_joint, config, title,savepath):
    """
    将 3D 嵌入按类别绘制为连续轨迹，每类一个颜色
    """
    num_categories = config.num_categories
    views_per_class = config.objects_per_category
    colors = plt.cm.get_cmap('tab10', num_categories)  # tab10 色表有 10 种颜色

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_categories):
        idx_start = i * views_per_class
        idx_end = (i + 1) * views_per_class
        traj = embedded_joint[idx_start:idx_end, :]  # shape: (36, 3)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(i), label=f'Class {i}', linewidth=1.8)
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(i), s=10, alpha=0.8)  # 可选散点叠加

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def round_sig(x, sig=3):
    #保留3位有效数字
    from math import log10, floor
    if x == 0:
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def plot_joint_performes(res_dict,title_name,savepath):
    from matplotlib.ticker import ScalarFormatter
    #绘制性能指标的变化
    # fig = plt.figure(figsize=(12, 4.5))
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 2)
    # plt.suptitle(title_name)

    para_range = list(res_dict.keys())
    performes=list(res_dict[para_range[0]].keys())
    for index, p in enumerate(performes):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        para=[round_sig(res_dict[i][p],3) for i in para_range]
        ax.plot(para_range, para, color='gray', alpha=0.8)
        formatter = ScalarFormatter(useOffset=False,useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))  # 强制所有值用科学计数法
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(p)
    # plt.subplots_adjust(wspace=0.8, hspace=0.5, right=0.95, left=0.15, top=0.95, bottom=0.1)
    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_number_performes(res_dict,title_name,savepath):
    from matplotlib.ticker import ScalarFormatter
    #绘制性能指标的变化
    # fig = plt.figure(figsize=(12, 4.5))
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 2)
    # plt.suptitle(title_name)

    para_range = list(res_dict.keys())
    performes=list(res_dict[para_range[0]].keys())
    for index, p in enumerate(performes):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        para=[round_sig(res_dict[i][p],3) for i in para_range]
        ax.plot(para_range, para, color='gray', alpha=0.8)
        ax.set_ylabel(p)
        ax.set_ylim([0,1])
    # plt.subplots_adjust(wspace=0.8, hspace=0.5, right=0.95, left=0.15, top=0.95, bottom=0.1)
    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_3d_trajectory_by_view(embedded_joint, label_type='view2', title='', savepath='none'):
    """
    将 3D 嵌入结果按 view1 或 view2 绘制为连续轨迹
    label_type: 'view1' or 'view2'
    """
    assert label_type in ['view1', 'view2'], "label_type must be 'view1' or 'view2'"
    num_angles = 36
    num_total = num_angles * num_angles
    assert embedded_joint.shape[0] == num_total, f"Expected shape ({num_total}, 3), got {embedded_joint.shape}"

    colors = plt.cm.get_cmap('hsv', num_angles)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    if label_type == 'view1':
        # 固定 view1，遍历 view2
        for v1 in range(num_angles):
            idxs = [v1 * num_angles + v2 for v2 in range(num_angles)]
            traj = embedded_joint[idxs, :]  # shape: (36, 3)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(v1), label=f'View1-{v1}', linewidth=1.2)
            ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(v1), s=8, alpha=0.8)
    else:
        # 固定 view2，遍历 view1
        for v2 in range(num_angles):
            idxs = [v1 * num_angles + v2 for v1 in range(num_angles)]
            traj = embedded_joint[idxs, :]  # shape: (36, 3)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(v2), label=f'View2-{v2}', linewidth=1.2)
            ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(v2), s=8, alpha=0.8)

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(title)
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()

    if savepath != 'none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_view1_view2_rdms_from_joint(rdm_joint, config, n_cols=6, savepath1='none', savepath2='none'):
    """
    从 rdm_joint 中提取：
    - 每个 angle1 固定时，angle2 间的 RDM（横向切块）
    - 每个 angle2 固定时，angle1 间的 RDM（纵向切块）
    并可视化。
    """
    num_angles = config.angles1.shape[0]
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import numpy as np
    import math

    # ========== 第一视角维度：angle1 固定，angle2 RDM ==========
    n_rows = math.ceil(num_angles / n_cols)
    fig1 = plt.figure(figsize=(n_cols * 2.0, n_rows * 2.0))
    gs1 = gridspec.GridSpec(n_rows, n_cols)

    for a1 in range(num_angles):
        idxs = [a1 * num_angles + a2 for a2 in range(num_angles)]
        sub_rdm = rdm_joint[np.ix_(idxs, idxs)]
        sub_rdm_norm = (sub_rdm - sub_rdm.min()) / (sub_rdm.max() - sub_rdm.min() + 1e-8)

        ax = fig1.add_subplot(gs1[a1 // n_cols, a1 % n_cols])
        sns.heatmap(sub_rdm_norm, cmap='viridis', xticklabels=False, yticklabels=False, cbar=False, ax=ax)
        ax.set_title(f'View1={a1}', fontsize=8)

    fig1.suptitle("Angle2 Structure under each View1", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath1 != 'none':
        plt.savefig(savepath1, dpi=300)
        plt.close()

    # ========== 第二视角维度：angle2 固定，angle1 RDM ==========
    fig2 = plt.figure(figsize=(n_cols * 2.0, n_rows * 2.0))
    gs2 = gridspec.GridSpec(n_rows, n_cols)

    for a2 in range(num_angles):
        idxs = [a1 * num_angles + a2 for a1 in range(num_angles)]
        sub_rdm = rdm_joint[np.ix_(idxs, idxs)]
        sub_rdm_norm = (sub_rdm - sub_rdm.min()) / (sub_rdm.max() - sub_rdm.min() + 1e-8)

        ax = fig2.add_subplot(gs2[a2 // n_cols, a2 % n_cols])
        sns.heatmap(sub_rdm_norm, cmap='viridis', xticklabels=False, yticklabels=False, cbar=False, ax=ax)
        ax.set_title(f'View2={a2}', fontsize=8)

    fig2.suptitle("Angle1 Structure under each View2", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath2 != 'none':
        plt.savefig(savepath2, dpi=300)
        plt.close()