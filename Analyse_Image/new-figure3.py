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

def scatter_2d_image(X_reduced, name_class, savepath, img_size):
    figure_size = 720
    import cv2
    index = np.argmax(np.max(X_reduced, 0) - np.min(X_reduced, 0))  # 建立映射
    min_old, max_old = np.min(X_reduced, 0)[index], np.max(X_reduced, 0)[index]
    min_new, max_new = img_size / 2, figure_size - img_size / 2
    mapped_data = np.interp(X_reduced, (min_old, max_old), (min_new, max_new))  # 转换为图像尺寸空间
    image_bg = np.ones((figure_size, figure_size, 3), dtype=np.uint8) * 255  # 创建白色背景
    for i_index, i in enumerate(name_class):
        imgpath = os.path.join(r'E:\Image_paper\Project\Acute_Image\image\img_rgb', i.split('.')[0][0:-3], i)
        print(imgpath)
        image = cv2.imread(imgpath)
        imgpath_mask = imgpath.replace('img_rgb', 'img_mask')
        img_mask = cv2.imread(imgpath_mask, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (img_size, img_size))
        resized_image_mask = cv2.resize(img_mask, (img_size, img_size))
        resized_image_mask[resized_image_mask > 0] = 1
        resized_image_mask = 1 - resized_image_mask
        mask_expanded = np.stack([resized_image_mask] * 3, axis=-1)
        image_fg = resized_image * mask_expanded  # 将图像的前景抠出来，背景为0
        mapped_y, mapped_x = mapped_data[i_index, :]  # 获取降维后的坐标
        mapped_x = figure_size - mapped_x
        arr_reversed = np.where(mask_expanded == 0, 1, 0)  # mask反转
        temp = image_bg[int(mapped_x - img_size / 2):int(mapped_x - img_size / 2) + img_size,
               int(mapped_y - img_size / 2):int(mapped_y - img_size / 2) + img_size]
        temp = temp * arr_reversed + image_fg
        image_bg[int(mapped_x - img_size / 2):int(mapped_x - img_size / 2) + img_size,
        int(mapped_y - img_size / 2):int(mapped_y - img_size / 2) + img_size] = temp
    image_bg = np.uint8(image_bg)
    cv2.imwrite(savepath,image_bg)

#选择一个神经元集群
name = 'dict_class1.pkl'
neuron_path = r'E:\Image_paper\Project/' + name
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应

#整体神经元集群降维
reducedim_method='MDS'      # Isomap,  tSNE,  MDS
resp_all = np.array([dict_class[k]['class']['resp'] for k in dict_class])
resp_all = np.transpose(resp_all)
name_class = dict_class[list(dict_class.keys())[0]]['class']['name']
label_fine = np.array([i.split('_')[2] for i in name_class])
class_fine = list(dict.fromkeys(label_fine))
label_coarse = np.array([i.split('_')[1] for i in name_class])
class_coarse = list(dict.fromkeys(label_coarse))

#绘制整体神经元集群的降维
savepath = r'E:\Image_paper\Project\plot\figure3n\{}.jpg'.format(reducedim_method)
X_reduced, rdm = analyse_reducedim(resp_all, reducedim_method, 2)      #降维
scatter_2d_image(X_reduced, name_class, savepath, 40)

#求解码准确率与神经元数量的关系##################################################################################################
REPEATS = 20                #重复多少次
cv_splits = 5               #交叉验证
N_samples, N_neurons = resp_all.shape
N_classes = len(class_fine)

# 取样本的神经元数量列表：1, 11, 21, ..., N_neurons（保证包含最后一个）
dict_accuracy={}
dict_accuracy['Neurons fine']=[]
dict_accuracy['Neurons coarse']=[]
n_list = sorted(set(list(range(1, N_neurons + 1, 10)) + [N_neurons]))
acc_overall = np.zeros((N_neurons, REPEATS), dtype=float)
acc_per_class = {c: np.zeros((N_neurons, REPEATS), dtype=float) for c in class_fine}
for r in range(REPEATS):
    rng = np.random.default_rng()       # 可复现
    perm = rng.permutation(N_neurons)                # 本次重复的神经元顺序（随后前缀累加）
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True)
    for n in n_list:
        print(n,r)
        cols = perm[:n]
        Xn = resp_all[:, cols]
        model = SVC(kernel='linear', decision_function_shape='ovr')
        y_pred = cross_val_predict(model, Xn, label_fine, cv=cv, n_jobs=None)
        acc = accuracy_score(label_fine, y_pred)
        acc_overall[n - 1, r]=acc
        for c in class_fine:
            mask = (label_fine == c)
            acc_per_class[c][n-1, r] = accuracy_score(label_fine[mask], y_pred[mask])
        if n==n_list[-1]:       #统计最终性能指标
            dict_accuracy['Neurons fine'].append(acc)
            model = SVC(kernel='linear', decision_function_shape='ovr')
            y_pred = cross_val_predict(model, Xn, label_coarse, cv=cv, n_jobs=None)
            acc1 = accuracy_score(label_coarse, y_pred)
            dict_accuracy['Neurons coarse'].append(acc1)

#求图像特征的解码准确率##################################################################################################
feature_path = r'E:\Image_paper\Project\dict_img_feature.pkl'
with open(feature_path, "rb") as f:
    img_dict = pickle.load(f)  # 图像特征字典，每张图像是一个键
feat_list = ['Color', 'Shape', 'Texture', 'V1-like', 'V2-like',
             'Alexnet Conv1', 'Alexnet Conv2', 'Alexnet Conv3', 'Alexnet Conv4', 'Alexnet Conv5']
for ft in feat_list:
    dict_accuracy[ft] = []
    feat_resp = []
    labels = []
    for img in img_dict.keys():
        if img_dict[img]['task'] == 'class':
            feat_resp.append(img_dict[img][ft])
            labels.append(img.split('_')[2])
    feat_resp = np.array(feat_resp)
    for ii in range(REPEATS):
        print(ft, ii)
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        model = SVC(kernel='linear', decision_function_shape='ovr')
        y_pred = cross_val_predict(model, feat_resp, labels, cv=cv)
        fine_accuracy = accuracy_score(labels, y_pred)  # 准确率
        dict_accuracy[ft].append(fine_accuracy)

pkl_path=os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'dict_accuracy.pickle')
with open(pkl_path, 'wb') as f:
    pickle.dump(dict_accuracy, f)

pkl_path=os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'acc_overall.pickle')
with open(pkl_path, 'wb') as f:
    pickle.dump(acc_overall, f)

pkl_path=os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'acc_per_class.pickle')
with open(pkl_path, 'wb') as f:
    pickle.dump(acc_per_class, f)

#绘制柱状图，解码准确率##################################################################################################
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(3.8, 1.7))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
color_map = { 'Neurons fine':'#374E55',  'Neurons coarse':'#374E55',  'Color':'#9E9E9E',  'Shape':'#31A354',
    'Texture':'#762A83',  'V1-like':'#A1D99B',  'V2-like':'#2171B5',  'Alexnet Conv1':'#EF3B2C',
    'Alexnet Conv2':'#EF3B2C',  'Alexnet Conv3':'#EF3B2C',  'Alexnet Conv4':'#EF3B2C',  'Alexnet Conv5':'#EF3B2C',}
means=[np.mean(dict_accuracy[k]) for k in dict_accuracy.keys()]
stds=[np.std(dict_accuracy[k]) for k in dict_accuracy.keys()]
names = list(dict_accuracy.keys())
colors = [color_map.get(k, '0.65') for k in names]
bars = ax.bar(range(len(names)), means, yerr=stds, capsize=2,
              color=colors, ecolor='k', edgecolor='k', linewidth=0.6)
ax.axhline(1/8, ls='--', lw=0.8, color='k')
ax.text(len(names)-0.2, 1/8 + 0.01, 'chance\n0.125', ha='left', va='bottom',
        fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Decoding accuracy',  # ← 文案也改为 SD
              fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.2,0.4,0.6,0.8,1.0])
ax.set_xticks([])
# 柱内标签与柱顶数值（显示均值；如需显示 ±SD：f'{m:.2f}±{s:.2f}'）
for bar, label, m, s in zip(bars, names, means, stds):
    x = bar.get_x() + bar.get_width() / 2.0
    ax.text(x, bar.get_height() / 2.0, label,
            rotation=90, ha='center', va='center', color='white',
            fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, clip_on=True)
    ax.text(x, bar.get_height() + 0.02, f'{m:.2f}',
            ha='center', va='bottom',
            fontsize=fontsize_-1, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(wspace=0.5, hspace=0.4, right=0.90, left=0.2, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'figure3_bar_decode.eps'), dpi=600, format='eps')


#绘制柱状图，rdm相似性##################################################################################################
n_dim = 10
reducedim_method = 'MDS'
dict_rdm = {}
dict_rdm_spearman = {}
for ft in feat_list:
    print(ft)
    feat_resp = []
    for img in name_class:
        feat_resp.append(img_dict[img][ft])
    feat_resp = np.array(feat_resp)
    _, rdm_feature = analyse_reducedim(feat_resp, reducedim_method, n_dim)
    _, rdm_neurons = analyse_reducedim(resp_all, reducedim_method, n_dim)
    r, _ = rdm_spearman(rdm_feature, rdm_neurons)
    dict_rdm_spearman[ft]=r
    dict_rdm[ft]=rdm_feature

fig = plt.figure(figsize=(3.5, 1.6))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
means=[np.mean(dict_rdm_spearman[k]) for k in dict_rdm_spearman.keys()]
# stds=[np.std(dict_rdm_spearman[k]) for k in dict_rdm_spearman.keys()]
names = list(dict_rdm_spearman.keys())
colors = [color_map.get(k, '0.65') for k in names]
bars = ax.bar(range(len(names)), means, capsize=2,
              color=colors, ecolor='k', edgecolor='k', linewidth=0.6)
ax.set_ylabel('RDM similarity',  # ← 文案也改为 SD
              fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylim(0.1, 0.31)
ax.set_yticks([0, 0.1,0.2,0.3])
ax.set_xticks([])
# 柱内标签与柱顶数值（显示均值；如需显示 ±SD：f'{m:.2f}±{s:.2f}'）
for bar, label, m in zip(bars, names, means):
    x = bar.get_x() + bar.get_width() / 2.0
    if ' ' in label:
        label=label.split(' ')[1]
    ax.text(x, bar.get_height() / 2.0, label,
            rotation=90, ha='center', va='center', color='white',
            fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, clip_on=True)
    ax.text(x, bar.get_height(), f'{m:.2f}',
            ha='center', va='bottom',
            fontsize=fontsize_-1, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

plt.subplots_adjust(wspace=0.5, hspace=0.4, right=0.90, left=0.15, top=0.95, bottom=0.1)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'figure3_bar_rdmsl.eps'), dpi=600, format='eps')

#绘制子群的性能变化##################################################################################################
#遍历每种特征
feat_list=['Color','Shape','Texture', 'Alexnet Conv1', 'Alexnet Conv3', 'Alexnet Conv5']
n_neuron = 80  # 小批量神经元的个数
dict_performs={}
for ft in feat_list:
    dict_performs[ft]={}
    dict_mertics = {}
    for neu in dict_class.keys():
        dict_mertics[neu] = dict_class[neu]['fit_corr'][ft]
    sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
    resp_class = [dict_class[neu]['class']['resp'] for neu in sorted_dict.keys()]
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
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        y_pred = cross_val_predict(model, resp_split, label_fine, cv=cv)
        accuracy = accuracy_score(label_fine, y_pred)  # 准确率
        print(ft, nstart, accuracy)
        acc_list.append(accuracy)
        _, rdm_neurons = analyse_reducedim(resp_split, reducedim_method, n_dim)
        r, _ = rdm_spearman(dict_rdm[ft], rdm_neurons)
        rdmsp_list.append(r)
    dict_performs[ft]['acc']=acc_list
    dict_performs[ft]['rdm'] = rdmsp_list


pkl_path=os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'dict_performs.pickle')
with open(pkl_path, 'wb') as f:
    pickle.dump(dict_performs, f)

#绘制子群的性能变化##################################################################################################
fig = plt.figure(figsize=(8, 2.5))
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
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax)
ax.spines['left'].set_linewidth(linewidth_ax)
ax.tick_params(labelsize=fontsize_)
ax = fig.add_subplot(gs[0, 1])
n_points = 6     # 取多少个中心点
half_window = 10  # 每个中心点两侧各取多少个样本（窗口=2*half_window+1）
L = len(range_list)
valid_start = half_window
valid_end   = L - 1 - half_window
centers_idx = np.linspace(valid_start, valid_end, n_points).astype(int)
xs = [range_list[i] for i in centers_idx]  # 对应的神经元窗口起点
for ft in feat_list:
    acc_list = dict_performs[ft]['acc']  # 长度 L
    means = []
    stds  = []
    for ci in centers_idx:
        w = acc_list[ci - half_window : ci + half_window + 1]  # 窗口
        w = np.asarray(w, dtype=float)
        means.append(np.nanmean(w))
        stds.append(np.nanstd(w, ddof=1) if len(w) > 1 else 0.0)
    ax.errorbar(xs, means,  fmt='-o', capsize=2, linewidth=linewidth_plot, markersize=3.5, label=ft)
ax.set_xlabel('Goodness of fit', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Decoding accuracy',          fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax)
ax.spines['left'].set_linewidth(linewidth_ax)
ax.tick_params(labelsize=fontsize_)


ax = fig.add_subplot(gs[0, 2])
n_points = 6     # 取多少个中心点
half_window = 10  # 每个中心点两侧各取多少个样本（窗口=2*half_window+1）
L = len(range_list)
valid_start = half_window
valid_end   = L - 1 - half_window
centers_idx = np.linspace(valid_start, valid_end, n_points).astype(int)
xs = [range_list[i] for i in centers_idx]  # 对应的神经元窗口起点
for ft in feat_list:
    acc_list = dict_performs[ft]['rdm']  # 长度 L
    means = []
    stds  = []
    for ci in centers_idx:
        w = acc_list[ci - half_window : ci + half_window + 1]  # 窗口
        w = np.asarray(w, dtype=float)
        means.append(np.nanmean(w))
        stds.append(np.nanstd(w, ddof=1) if len(w) > 1 else 0.0)
    ax.errorbar(xs, means,  fmt='-o', capsize=2, linewidth=linewidth_plot, markersize=3.5, label=ft)
ax.set_xlabel('Goodness of fit', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('RDM similarity',          fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax)
ax.spines['left'].set_linewidth(linewidth_ax)
ax.tick_params(labelsize=fontsize_)
plt.subplots_adjust(wspace=0.5, hspace=0.2, right=0.95, left=0.1, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'figure3_performs.eps'), dpi=600, format='eps')