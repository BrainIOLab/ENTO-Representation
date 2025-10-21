import os
import pickle
import numpy as np
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

name = 'dict_class1pkl'
neuron_path = r'E:\Image_paper\Project/' + name
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应


dict_mertics = {}
for neu in dict_class.keys():
    dict_mertics[neu]=dict_class[neu]['class']['metrics']['Separation ratio']
sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
resp_class = [dict_class[neu]['class']['resp'] for neu in sorted_dict.keys()]
resp_class = np.array(resp_class)
name_class = dict_class['ento_exp1_ch14']['class']['name']
label_fine = [i.split('_')[2] for i in name_class]
class_fine = list(dict.fromkeys(label_fine))

dict_acc={}
dict_seqa={}
dict_seqas={}
n_neuron = 80  # 小批量神经元的个数
for nstart in range(0, resp_class.shape[0] - n_neuron + 1):
    resp_split = resp_class[nstart:nstart + n_neuron, :]
    resp_split = np.transpose(resp_split)
    # 解码准确率
    model = SVC(kernel='linear', decision_function_shape='ovr')  # linear
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    y_pred = cross_val_predict(model, resp_split, label_fine, cv=cv)
    accuracy = accuracy_score(label_fine, y_pred)  # 准确率
    print(nstart, accuracy)

    X_reduced, rdm = analyse_reducedim(resp_split, 'MDS', 3)
    # 类内距离、类间距离、分离比
    dist_intra, dist_inter = compute_class_distance(rdm, label_fine)
    Separability = dist_inter / (dist_intra + 1e-9)  # 计算类分离比

    dict_acc[accuracy]=resp_split
    dict_seqa[nstart]=resp_split
    dict_seqas[round(Separability,4)] = resp_split

sorted_keys = list(dict_seqas.keys())
sorted_keys.sort()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # 兼容旧版本
from sklearn.preprocessing import LabelEncoder
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(2, 3)
acc_list = [0.485, 0.54, 0.58, 0.61, 0.65, 0.695]
seqa_list = [0,3,29,123,143,158]
seqas_list = [1.0871,1.2136,1.2452,1.271,1.3047,1.3562]
cmap = plt.cm.get_cmap('tab20', 8)
le = LabelEncoder()
labels_num = le.fit_transform(label_fine)  # 0..7
for iindex, axx in enumerate(seqas_list):
    # resp_split = dict_acc[axx]            # (200, n_neuron)
    resp_split = dict_seqas[axx]  # (200, n_neuron)
    r, c = divmod(iindex, 3)
    # === 3D 子图 ===
    ax = fig.add_subplot(gs[r, c], projection='3d')
    # 3D MDS
    X_reduced, rdm = analyse_reducedim(resp_split, 'MDS', 3)  # <-- 改为3维
    sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                    c=labels_num, cmap=cmap, s=20)
    ax.set_axis_off()        # 对 3D 也适用，直接隐藏轴框、刻度、标签
    ax.grid(False)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    # # 每个子图都放色条会拥挤，如需仅放一个，可移到循环外
    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_label('Label')
plt.subplots_adjust(wspace=0.1, hspace=0.1, right=1, left=0., top=1, bottom=0.0)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3', 'figure3-5.eps'), dpi=600, format='eps')


fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3)
acc_list = [0.485, 0.54, 0.58, 0.61, 0.65, 0.695]
seqa_list = [0,3,29,123,143,158]
seqas_list = [1.0871,1.2136,1.2452,1.271,1.3047,1.3562]
cmap = plt.cm.get_cmap('tab20', 8)
le = LabelEncoder()
labels_num = le.fit_transform(label_fine)  # 0..7
for iindex, axx in enumerate(seqas_list):
    # resp_split = dict_acc[axx]            # (200, n_neuron)
    resp_split = dict_seqas[axx]  # (200, n_neuron)
    r, c = divmod(iindex, 3)
    # === 3D 子图 ===
    ax = fig.add_subplot(gs[r, c])
    # 3D MDS
    X_reduced, rdm = analyse_reducedim(resp_split, 'MDS', 2)
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_num, cmap=cmap, s=30, edgecolors='k')
    # ax.set_title(f'Acc={axx:.3f}', fontsize=10)
    # ax.set_xlabel('dim 1')
    # ax.set_ylabel('dim 2')
    # ax.set_zlabel('dim 3')
    ax.set_axis_off()        # 对 3D 也适用，直接隐藏轴框、刻度、标签
    ax.grid(False)
    # ax.set_xlim(-3,3)
    # ax.set_ylim(-3, 3)
    # ax.set_zlim(-3, 3)

    # # 每个子图都放色条会拥挤，如需仅放一个，可移到循环外
    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_label('Label')

plt.tight_layout()
plt.show()
