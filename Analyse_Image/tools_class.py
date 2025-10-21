import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import random
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score

def build_class_rdm(n_objects,n_classes,labels):
    # 计算目标rdm矩阵，归一化到0~1
    objects_per_class = n_objects // n_classes
    # 构建 RDM：类内 0，类间 1
    rdm_target = np.zeros((n_objects, n_objects))
    for i in range(n_objects):
        for j in range(n_objects):
            rdm_target[i, j] = 0 if labels[i] == labels[j] else 1
    return rdm_target

from sklearn.manifold import Isomap, TSNE, MDS, LocallyLinearEmbedding
import umap
from sklearn.metrics import pairwise_distances
import numpy as np

def compute_geodesic_rdm_class(R, method='isomap', n_neighbors=10, n_components=2, metric='euclidean'):
    # 已知响应矩阵，计算测地距离
    X = R.T  # 转为 samples × features
    if method == 'isomap':
        model = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric=metric)
        model.fit(X)
        D = model.dist_matrix_
        embedded = model.transform(X)
    elif method == 'umap':
        model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric)
        embedded = model.fit_transform(X)
        D = squareform(pdist(embedded, metric=metric))
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

def cal_represent_class(init_params,config):
    #由参数计算响应矩阵和表征
    amp_all = init_params[0:config.num_neurons]
    sigma_tuning_all = init_params[config.num_neurons:2 * config.num_neurons]
    sigma_intra_all = init_params[2 * config.num_neurons:3 * config.num_neurons]
    neuron_centers = np.linspace(0, config.num_categories - 1, config.num_neurons)  # 神经元调谐中心（等距分布）
    responses = np.zeros((config.num_samples, config.num_neurons))
    for i in range(config.num_neurons):
        mu_i = neuron_centers[i]
        amp_i = amp_all[i]
        sigma_tuning_i = sigma_tuning_all[i]
        sigma_intra_i = sigma_intra_all[i]
        for c in range(config.num_categories):
            angle_diff = 2 * np.pi * (c - mu_i) / config.num_categories
            mu_ic = amp_i * np.exp(-(1 - np.cos(angle_diff)) / (2 * sigma_tuning_i ** 2))
            idx = np.where(config.labels == c)[0]
            responses[idx, i] = np.random.normal(loc=mu_ic, scale=sigma_intra_i, size=config.objects_per_category)
    resp = responses.T  # 转置后为 神经元 × 样本
    # 计算表征
    rdm, rdm_norm, embedded = compute_geodesic_rdm_class(resp, method=config.method, n_neighbors=10)
    return resp, rdm, rdm_norm, embedded

def cal_represent_class_optimize(init_params,config):
    mu = init_params[:config.num_neurons]
    sigma_inter = init_params[config.num_neurons : 2 * config.num_neurons]
    sigma_intra = init_params[config.num_neurons*2 :3 * config.num_neurons]
    amp = init_params[3 * config.num_neurons:]

    responses = np.zeros((config.num_samples, config.num_neurons))
    for i in range(config.num_neurons):
        mu_i = mu[i]
        amp_i = amp[i]
        sigma_tuning_i = sigma_inter[i]
        sigma_intra_i = sigma_intra[i]
        for c in range(config.num_categories):
            angle_diff = 2 * np.pi * (c - mu_i) / config.num_categories
            mu_ic = amp_i * np.exp(-(1 - np.cos(angle_diff)) / (2 * sigma_tuning_i ** 2))
            idx = np.where(config.labels == c)[0]
            responses[idx, i] = np.random.normal(loc=mu_ic, scale=sigma_intra_i, size=config.objects_per_category)
    resp = responses.T  # 转置后为 神经元 × 样本
    # 计算表征
    rdm, rdm_norm, embedded = compute_geodesic_rdm_class(resp, method=config.method, n_neighbors=10)
    return resp, rdm, rdm_norm, embedded

def compute_class_selectivity(resp):
    #选择性指标，神经元响应分布是否偏向特定刺激
    sel = 1 - (np.mean(resp, axis=1)**2) / (np.mean(resp**2, axis=1) + 1e-9)
    return np.mean(sel)

def compute_class_sparsity(resp):
    #稀疏性指标，每个物体在所有神经元上的激活是否有限
    a = np.mean(resp, axis=0)
    b = np.mean(resp**2, axis=0)
    sparsity = 1 - np.mean((a**2)/(b + 1e-9))
    return sparsity

def compute_class_energy(resp):
    #计算每个神经元的平均能量，然后计算神经元集群的总能量
    energy_per_neuron = np.mean(resp**2, axis=1)  # 每个神经元的能量
    energy = np.sum(energy_per_neuron)
    return energy

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

def cal_class_perform(resp_class, rdm_class, config):
    dict_ = {}
    dict_['Selectivity']=compute_class_selectivity(resp_class)        #计算选择性指标
    dict_['Sparsity'] = compute_class_sparsity(resp_class)      #计算稀疏性指标
    dict_['Energy']=compute_class_energy(resp_class)      #计算能量坐标
    dict_['Intra-class distance'],dict_['Inter-class distance']=compute_class_distance(rdm_class, config.labels)     #计算类内距离和类间距离
    dict_['Separation ratio'] = dict_['Inter-class distance'] / (dict_['Intra-class distance'] + 1e-9) #计算类分离比
    dict_['Accuracy'],dict_['Redundancy']=compute_accu_ri(resp_class, config.labels, per=0.1, repeat=50)
    dict_['AUC']=compute_structure_auc(rdm_class, config.rdm_class_target)
    return dict_

# 绘制图像
def visualize_class_represent(resp_class, rdm_class, embedded_class_2d, dict_perform, savepath, config):
    fig = plt.figure(figsize=(8, 5.5))
    gs = gridspec.GridSpec(4, 6)  # 4行3列的网格
    title_name="Selectivity:{:.4f}; Sparsity:{:.4f}; Energy:{:.4f} \n Intra-class:{:.4f}; Inter-class:{:.4f}; Separation ratio:{:.4f} \n Accuracy:{:.4f}; Redundancy:{:.4f}; AUC:{:.4f};".format(
        dict_perform['Selectivity'],dict_perform['Sparsity'],dict_perform['Energy'],
        dict_perform['Intra-class distance'], dict_perform['Inter-class distance'],dict_perform['Separation ratio'],
        dict_perform['Accuracy'], dict_perform['Redundancy'], dict_perform['AUC'], fontsize=12)
    plt.suptitle(title_name)
    # 1. 平均响应曲线
    responses=resp_class.T
    mean_responses = responses.reshape(config.num_categories, config.objects_per_category, config.num_neurons)
    mean_responses=np.mean(mean_responses,1)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    for i in range(config.num_neurons):
        if i==4:
            ax1.plot(mean_responses[:, i], color='red')
        else:
            ax1.plot(mean_responses[:, i], color='gray')
    ax1.set_title("Mean category response")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Mean response")
    # 2. 单个神经元响应示例
    ax2 = fig.add_subplot(gs[2:4, 0:2])  # 第3行第2列
    ax2.plot(responses[:, 4], color='black')
    ax2.set_title("Neuron 5")
    ax2.set_xlabel("Object index")
    ax2.set_ylabel("Response")
    # 3. RDM
    ax3 = fig.add_subplot(gs[0:2, 2:4])
    sns.heatmap(rdm_class, cmap='viridis', xticklabels=False, yticklabels=False, ax=ax3)
    ax3.set_title('RDM')
    # 4. RDM traget
    ax3 = fig.add_subplot(gs[0:2, 4:6])
    sns.heatmap(config.rdm_class_target, cmap='viridis', xticklabels=False, yticklabels=False, ax=ax3)
    ax3.set_title('RDM target')
    # 5. 2D嵌入
    ax4 = fig.add_subplot(gs[2:4,4:6])
    sc_2d = ax4.scatter(embedded_class_2d[:, 0], embedded_class_2d[:, 1], c=config.labels, cmap='Set1', s=5)
    ax4.set_title('Isomap embedding (2D)')
    cbar_2d = fig.colorbar(sc_2d, ax=ax4, ticks=np.arange(10))
    cbar_2d.set_label('Class Label')
    plt.tight_layout()
    if savepath != 'none':
        plt.savefig(savepath, dpi=150)
        plt.close()
    else:
        pass

def plot_class_performes(res_dict,savepath,xl):
    #绘制性能指标的变化
    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(3, 3)
    para_range = list(res_dict.keys())

    performes=list(res_dict[para_range[0]].keys())
    for index, p in enumerate(performes):
        ax = fig.add_subplot(gs[index // 3, index % 3])
        para=[res_dict[i][p] for i in para_range]
        ax.plot(para_range, para, color='gray', alpha=0.8)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))  # 强制所有值用科学计数法
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(p)
        ax.set_xlabel(xl)
    plt.subplots_adjust(wspace=0.7, hspace=0.7, right=0.95, left=0.15, top=0.95, bottom=0.1)
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        pass

