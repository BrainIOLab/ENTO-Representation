import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
import pickle
import os

def load_view_rdm():
    loadpath = os.path.join('results', 'view', 'symmetric', 'symmetric.pkl')
    with open(loadpath, 'rb') as f:
        rdm_view_norm = pickle.load(f)
    return rdm_view_norm

def build_view_rdm(angles):
    #归一化到0~1，建立均匀的视角表征
    delta = np.abs(angles[:, None] - angles[None, :])
    circular_delta = np.minimum(delta, 360 - delta)
    return circular_delta / 180.0

def build_gaussian_rdm(angles, center_deg=180, sigma=30):
    delta = np.abs(angles[:, None] - center_deg)
    circular_delta = np.minimum(delta, 360 - delta)
    similarity = np.exp(-0.5 * (circular_delta / sigma) ** 2)
    rdm = 1 - np.outer(similarity, similarity)
    rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min())
    return rdm

def circular_gaussian(theta, mu, sigma, amp=None):
    #根据高斯调谐参数生成响应矩阵
    delta = np.angle(np.exp(1j * np.deg2rad(theta[:, None] - mu[None, :])), deg=True)
    base_response = np.exp(-0.5 * (delta / sigma[None, :]) ** 2).T
    if amp is not None:
        return amp[:, None] * base_response
    return base_response

def compute_isomap_rdm(R, n_neighbors=5):
    #已知响应矩阵，计算测地距离
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2, metric='euclidean')
    isomap.fit(R.T)
    D = isomap.dist_matrix_
    D_norm = (D - D.min()) / (D.max() - D.min() + 1e-8)
    embedded = isomap.transform(R.T)  # 获取降维后的样本位置（2D）
    return D, D_norm, embedded

def compute_pca_rdm(R, n_components=2):
    #计算欧式距离的 RDM 矩阵，并用 PCA 降维到 2D
    X = R.T
    D = np.sqrt(((X[:, np.newaxis] - X[np.newaxis, :]) ** 2).sum(axis=2))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # 降维后的坐标
    D_norm = (D - D.min()) / (D.max() - D.min() + 1e-8)
    return D, D_norm, X_pca  # 返回距离矩阵和降维后的数据

def apply_noise(R, noise_type="none", noise_std=0.00, rng=np.random):
    #对神经响应矩阵 R 添加噪声
    if noise_type == 'none':
        R_noisy=R
    else:
        if noise_type == "additive":
            R_noisy = R + rng.normal(0, noise_std, size=R.shape)
        elif noise_type == "multiplicative":
            R_noisy = R * (1 + rng.normal(0, noise_std, size=R.shape))
        elif noise_type == "poisson":
            R_scaled = R * noise_std  # 将连续响应放大为 spike count 级别
            R_poisson = rng.poisson(R_scaled)
            R_noisy = R_poisson / noise_std  # 再缩放回原来的幅度范围
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        R_noisy = np.clip(R_noisy, 0, 1.0)  # 限制响应范围
    return R_noisy

def cal_view_represent(params, config, method):
    #根据高斯调谐参数，计算响应矩阵，RDM矩阵，和归一化后的RDM矩阵
    mu = params[:config.num_neurons] % 360
    amp= params[config.num_neurons:2 * config.num_neurons]
    sigma = params[2 * config.num_neurons:]
    R = circular_gaussian(config.angles, mu, sigma, amp)
    R = apply_noise(R, noise_type=config.noise_type, noise_std=config.noise_std, rng=np.random)
    if method=='isomap':
        rdm_view, rdm_view_norm, embedded_view_2d = compute_isomap_rdm(R)
    if method=='pca':
        rdm_view, rdm_view_norm, embedded_view_2d = compute_pca_rdm(R)
    return R, rdm_view, rdm_view_norm, embedded_view_2d


def generate_180symmetric_response(angles, mu, sigma, amp):
    n_half = len(angles) // 2
    angles_half = angles[:n_half]  # 0~180°

    # 在0~180°上计算响应
    theta = angles_half[:, None]    # [n_half, 1]
    mu = mu[None, :]                # [1, n_neurons]
    dist_sq = (theta - mu) ** 2
    R_half = np.exp(-0.5 * dist_sq / (sigma[None, :] ** 2)).T  # shape: [n_neurons, n_half]
    R_half *= amp[:, None]

    # 构造对称另一半（水平翻转）
    R_full = np.concatenate([R_half, np.flip(R_half, axis=1)], axis=1)
    return R_full

def cal_view_represent_symmetric(params, config, method):
    #根据高斯调谐参数，计算响应矩阵，RDM矩阵，和归一化后的RDM矩阵
    mu = params[:config.num_neurons] % 360
    amp= params[config.num_neurons:2 * config.num_neurons]
    sigma = params[2 * config.num_neurons:]
    # R = circular_gaussian(config.angles, mu, sigma, amp)
    R=generate_180symmetric_response(config.angles, mu, sigma, amp)
    R = apply_noise(R, noise_type=config.noise_type, noise_std=config.noise_std, rng=np.random)
    if method=='isomap':
        rdm_view, rdm_view_norm, embedded_view_2d = compute_isomap_rdm(R)
    if method=='pca':
        rdm_view, rdm_view_norm, embedded_view_2d = compute_pca_rdm(R)
    return R, rdm_view, rdm_view_norm, embedded_view_2d

def cal_rdm_loss(D, D_target):
    loss = np.mean((D - D_target) ** 2)
    return loss

def safe_log(x, eps=1e-6):
    if x+ eps<0:
        x=0
    return np.log(x + eps)

def compute_rdm_metrics(RDM):
    n = RDM.shape[0]
    discriminability = np.sum(RDM) / (n * n - n)            # 1. 全局区分度（去对角线）
    return discriminability

def compute_sum_energy(R):
    # 计算神经元集群的平均能量
    energy_per_neuron = np.mean(R**2, axis=1)  # 每个神经元的能量
    sum_energy = np.sum(energy_per_neuron)
    return sum_energy

def compute_mutual_information(R, n_clusters=20):
    #计算神经元集群对刺激的互信息
    n_neurons, n_stimuli = R.shape
    stimuli = np.arange(n_stimuli)  # S ∈ [0, 359]
    response_vectors = R.T  # shape: (360, n_neurons)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10,random_state=0).fit(response_vectors)
    discrete_response = kmeans.labels_  # length: 360
    mi = mutual_info_score(stimuli, discrete_response)  # in nats
    mi_bits = mi / np.log(2)  # convert from nats to bits
    return mi_bits

def compute_redundancy(R, compute_mutual_info_func):
    #计算神经元的冗余信息量
    n_neurons = R.shape[0]
    I_full = compute_mutual_info_func(R)
    redundancy = []
    for i in range(n_neurons):
        R_minus_i = np.delete(R, i, axis=0)  # 删除第 i 个神经元
        I_minus_i = compute_mutual_info_func(R_minus_i)
        red_i = 1 - I_minus_i / (I_full + 1e-9)  # 防止除 0
        redundancy.append(red_i)
    return np.mean(np.array(redundancy))

def circular_gradient(R, d_theta):
    #视角神经元响应，循环求导
    dR = np.zeros_like(R)
    dR[:, 1:-1] = (R[:, 2:] - R[:, :-2]) / (2 * d_theta)         # 中心差分
    dR[:, 0] = (R[:, 1] - R[:, -1]) / (2 * d_theta)              # 环形左端
    dR[:, -1] = (R[:, 0] - R[:, -2]) / (2 * d_theta)             # 环形右端
    return dR

def compute_fisher_information(R, theta_deg):
    #计算神经元集群的FI信息
    theta_rad = np.deg2rad(theta_deg)    # 角度转弧度
    d_theta = theta_rad[1] - theta_rad[0]  # scalar 步长
    dR = circular_gradient(R, d_theta)    # 对角度求导：R'(\theta)
    # Fisher Information 公式：FI_i = (dR_i)^2 / R_i
    eps = 1e-9
    FI_per_neuron = (dR ** 2) / (R + eps)
    # 总 FI：所有神经元求和
    FI_total = np.sum(FI_per_neuron, axis=0)
    FI_total_mean=np.mean(FI_total)
    return FI_total_mean

def compute_mutual_info(R, n_bins=20):
    n_neurons, n_angles = R.shape
    X = np.arange(n_angles)  # 视角变量 0~359
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')     # 离散化响应
    R_disc = discretizer.fit_transform(R.T).T  # 转置 -> (360, 5) -> 离散化 -> 再转置回来
    mi_individual = [mutual_info_score(R_disc[i], X) for i in range(n_neurons)]
    mi_sum = np.sum(mi_individual)
    return mi_sum

def compute_selectivities(R):
    # 计算所有神经元的平均选择性
    def selectivity(responses):
        # 计算单个神经元的选择性
        mean = np.mean(responses)
        mean_sq = np.mean(responses ** 2)
        return 1 - (mean ** 2) / (mean_sq + 1e-9)  # 防止除0
    return np.mean([selectivity(R[i]) for i in range(R.shape[0])])

def compute_sparsity(R):
    #计算神经元的稀疏性
    mean_response = np.mean(R, axis=0)
    mean_sq_response = np.mean(R ** 2, axis=0)
    sparsity = (mean_response ** 2) / (mean_sq_response + 1e-9)
    return 1-np.mean(sparsity)

def cal_view_performes(resp_view, rdm_view, rdm_view_norm, config):
    # 计算性能指标
    dict_={}
    rdm_view_target=config.rdm_view_target
    rdm_loss = cal_rdm_loss(rdm_view_norm, rdm_view_target)     # 计算rdm间的均方误差和其log
    dict_['RDM mse']=rdm_loss
    dict_['Log-RDM mse'] = safe_log(rdm_loss)
    dict_['Global distance'] = compute_rdm_metrics(rdm_view)    # 计算全局距离
    dict_['Energy'] = compute_sum_energy(resp_view)      # 计算集群总能量
    dict_['Mutual info'] = compute_mutual_information(resp_view)    # 计算互信息（bin化然后计算互信息）
    dict_['Redundant info']= compute_redundancy(resp_view, compute_mutual_information)  # 计算冗余信息
    dict_['Log-redundant info']=safe_log(dict_['Redundant info'])      # 冗余信息量的log化
    dict_['Fisher info'] = compute_fisher_information(resp_view, config.angles)  # 计算Fisher信息
    dict_['Selectivity'] = compute_selectivities(resp_view)  # 计算选择性
    dict_['Sparsity'] = compute_sparsity(resp_view)  # 计算稀疏性
    return dict_

def generate_gaussian_mu(std_devs):
    #生成高斯分布的mu，中心点为180°
    center = 180
    mu_dict = {}
    mu_values = [
        center - 2 * std_devs,  # -2σ
        center - std_devs,  # -1σ
        center,  # 中心点
        center + std_devs,  # +1σ
        center + 2 * std_devs  # +2σ
    ]
    # 限制到 0~360 范围
    mu_values = np.clip(mu_values, 0, 360)
    return mu_values

from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial import procrustes

def make_symmetric(D):
    D_sym = 0.5 * (D + D.T)
    np.fill_diagonal(D_sym, 0)
    return D_sym

def normalize_rdm(D):
    return (D - D.min()) / (D.max() - D.min() + 1e-9)

def structure_preservation(rdm,rdm_noise):
    v_clean = make_symmetric(rdm)
    v_noisy = make_symmetric(rdm_noise)
    v_clean = squareform(v_clean)
    v_noisy = squareform(v_noisy)
    rho, _ = spearmanr(v_clean, v_noisy)

    rdm_norm = normalize_rdm(rdm)
    rdm_noise_norm = normalize_rdm(rdm_noise)
    mse = np.mean((rdm_norm - rdm_noise_norm) ** 2)
    return rho,mse

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

def cal_noise_perform(rdm,rdm_noise,resp, resp_noise):
    dict_={}
    pear,mse=structure_preservation(rdm, rdm_noise)
    proc_disc=procrustes_disc(resp, resp_noise)
    neigh_preser=neighborhood_preservation(resp, resp_noise, k=10)
    # dict_['cos_sim'] =cos_sim
    dict_['Pearson'] =pear
    dict_['Mse'] =mse
    dict_['PD'] =proc_disc
    dict_['NPS'] = neigh_preser
    return pear,mse,proc_disc,neigh_preser