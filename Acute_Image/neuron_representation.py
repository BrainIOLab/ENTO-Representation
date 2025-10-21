import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm
import cv2
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_curve, auc

def reducedim_analysis(neuron_resp, label_num,class_names,savename):

    neuron_resp=np.transpose(neuron_resp)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2,2)
    cmap = plt.cm.get_cmap('tab20', 8)  # 16类
    result = np.zeros((720 * 2, 720 * 2, 3), dtype=np.uint8)

    reducedim_method_list=['PCA','tSNE','MDS','Isomap']
    for p_index,reducedim_method in enumerate(reducedim_method_list):
        if reducedim_method == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)  # 降维到2维
            X_reduced = pca.fit_transform(neuron_resp)
        if reducedim_method == 'tSNE':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)  # 降维到2维
            X_reduced = tsne.fit_transform(neuron_resp)
        if reducedim_method == 'MDS':
            from sklearn.manifold import MDS
            mds = MDS(n_components=2)  # 降维到2维
            X_reduced = mds.fit_transform(neuron_resp)
        if reducedim_method == 'Isomap':
            from sklearn.manifold import Isomap
            isomap = Isomap(n_components=2)  # 降维到2维
            X_reduced = isomap.fit_transform(neuron_resp)

        fisher_ratio = cal_fisher_ratio(X_reduced, label_num)
        silhouette_avg = silhouette_score(X_reduced, label_num, metric='euclidean')
        fisher_ratio=round(fisher_ratio,2)
        silhouette_avg=round(silhouette_avg, 2)
        print("Fisher判别比:", round(fisher_ratio,2),"轮廓系数:", round(silhouette_avg,2))

        # 绘制散点图
        ax1 = fig.add_subplot(gs[p_index // 2, p_index % 2])
        scatter = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=label_num, cmap=cmap, s=50, edgecolors='k')
        ax1.axis('equal')
        cbar = fig.colorbar(scatter, ax=ax1, label='Label')  # 注意这里需要传入fig和ax1
        ax1.set_title('{}_{:.2f}_{:.2f}'.format(reducedim_method,fisher_ratio,silhouette_avg))
        ax1.set_xlabel('dim 1')
        ax1.set_ylabel('dim 2')
        # plt.show()

        # 绘制图像
        img_size=44                                                                     # 图像中单个物体图像的大小
        figure_size=720
        index = np.argmax(np.max(X_reduced, 0) - np.min(X_reduced, 0))  # 建立映射
        min_old, max_old = np.min(X_reduced, 0)[index], np.max(X_reduced, 0)[index]
        min_new, max_new = img_size / 2, figure_size - img_size / 2
        mapped_data = np.interp(X_reduced, (min_old, max_old), (min_new, max_new))  # 转换为图像尺寸空间
        image_bg = np.ones((figure_size, figure_size, 3), dtype=np.uint8) * 255  # 创建白色背景
        for i_index,i in enumerate(class_names):
            imgpath = os.path.join(r'image','img_rgb', i.split('.')[0][0:-3],i)
            image = cv2.imread(imgpath)
            imgpath_mask = imgpath.replace('img_rgb', 'img_mask')
            img_mask = cv2.imread(imgpath_mask, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (img_size, img_size))
            resized_image_mask = cv2.resize(img_mask, (img_size, img_size))
            resized_image_mask[resized_image_mask>0] = 1
            resized_image_mask=1-resized_image_mask
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
        i = p_index // 2  # 行索引 (0或1)
        j = p_index % 2  # 列索引 (0或1)
        # 计算放置位置
        y_start, y_end = i * 720, (i + 1) * 720
        x_start, x_end = j * 720, (j + 1) * 720
        # 将图像放入对应位置
        result[y_start:y_end, x_start:x_end] = image_bg

    plt.subplots_adjust(wspace=0.3, hspace=0.4, right=0.95, left=0.1, top=0.95, bottom=0.10)
    plt.savefig(os.path.join('results','neuron_representation', '{}_scatter_{}.png'.format('reducedim',savename)), dpi=600,format='png')
    cv2.imwrite(os.path.join('results','neuron_representation', '{}_image_{}.png'.format('reducedim',savename)),result)

def rsm_analysis(neuron_resp,class_labels,savename):
    correlation_matrix = np.corrcoef(neuron_resp, rowvar=False)  # 计算皮尔逊相关性矩阵
    # 可视化相关性矩阵
    plt.figure(figsize=(5, 4))
    sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
    plt.xticks(ticks=list(range(12,200,25)), labels=class_labels, rotation=45, ha='right')
    plt.yticks(ticks=list(range(12,200,25)), labels=class_labels, rotation=0)
    plt.title('neuron representation rsm')
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.95, left=0.25, top=0.90, bottom=0.25)
    plt.savefig(os.path.join('results', 'neuron_representation', 'rsm_{}.png'.format(savename)), dpi=600,format='png')

def return_best_resp(neuron_dict, decoder):
    # 获取neuron_data中最佳响应，返回最大相关性
    feature_lists = list(neuron_dict['decode'].keys())
    resp_lists = list(neuron_dict['decode'][feature_lists[0]]['class'].keys())
    model_list = list(neuron_dict['decode'][feature_lists[0]]['class'][resp_lists[0]].keys())
    for model_name in model_list:
        df = pd.DataFrame(index=resp_lists, columns=feature_lists, dtype=float)
        for f in feature_lists:
            for r in resp_lists:
                df.at[r, f] = round(neuron_dict['decode'][f]['class'][r][model_name]['correlation'], 2)

    max_value = df.values.max()  # 或 df.max().max()
    max_row, max_col = np.where(df == max_value)
    max_resp_label = df.index[max_row[0]]
    max_feature_label = df.columns[max_col[0]]
    resp_values = neuron_dict['decode'][max_feature_label]['class'][max_resp_label][decoder]['Y_test']
    return resp_values

def cal_fisher_ratio(X_reduced,label_num):
    from collections import Counter
    num_classes = len(Counter(label_num))
    num_samples = len(label_num)
    label_num=np.array(label_num)

    num_neurons = X_reduced.shape[1]
    # 计算整体均值（所有样本的均值）
    mu_all = np.mean(X_reduced, axis=0)
    S_W = np.zeros((num_neurons, num_neurons))  # 类内散度矩阵
    S_B = np.zeros((num_neurons, num_neurons))  # 类间散度矩阵
    for i in range(num_classes):
        class_data = X_reduced[label_num == i]
        mu_i = np.mean(class_data, axis=0)
        class_scatter = np.zeros((num_neurons, num_neurons))
        for x in class_data:
            class_scatter += np.outer(x - mu_i, x - mu_i)
        S_W += class_scatter
        N_i = class_data.shape[0]  # 当前类别的样本数
        S_B += N_i * np.outer(mu_i - mu_all, mu_i - mu_all)
    fisher_ratio = np.trace(S_B) / np.trace(S_W)
    return fisher_ratio

def neuron_decode(neuron_resp,label_num,class_labels,savename):
    neuron_resp=np.transpose(neuron_resp)
    label_num=np.array(label_num)

    model_names = ['SVM', 'LDA', 'BP','XGB']
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(2,2)

    for mindex,model_name in enumerate(model_names):
        # print(model_name)
        if model_name == 'SVM':
            model = SVC(kernel='rbf', decision_function_shape='ovr')
        if model_name == 'LDA':
            model = LinearDiscriminantAnalysis()
        if model_name == 'BP':
            model = MLPClassifier(hidden_layer_sizes=(16,), solver='sgd', max_iter=300, early_stopping=True)
        if model_name == 'XGB':
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        Y_test = []
        Y_index = []
        Y_pred = []
        k_fold=5
        for k in range(k_fold):
            test_indices = np.arange(k, neuron_resp.shape[0], k_fold)  # 获取测试集索引
            train_indices = np.setdiff1d(np.arange(neuron_resp.shape[0]), test_indices)  # 获取训练集的索引（除去测试集的索引）
            x_train = neuron_resp[train_indices, :]
            y_train = label_num[train_indices]
            x_test = neuron_resp[test_indices, :]
            y_test = label_num[test_indices]
            Y_test.extend(y_test)
            Y_index.extend(list(test_indices))
            model.fit(x_train, y_train)
            Y_pred.extend(model.predict(x_test))
        # 计算性能指标
        sorted_indices = np.argsort(Y_index)  # 从小到大排序
        Y_index_sorted = np.array(Y_index)[sorted_indices]
        Y_test_sorted = np.array(Y_test)[sorted_indices]
        Y_pred_sorted = np.array(Y_pred)[sorted_indices]

        cm = confusion_matrix(Y_test_sorted, Y_pred_sorted)  # 混淆矩阵
        accuracy = accuracy_score(Y_test_sorted, Y_pred_sorted)  # 准确率
        precision = precision_score(Y_test_sorted, Y_pred_sorted, average='macro')  # 宏平均精确率
        recall = recall_score(Y_test_sorted, Y_pred_sorted, average='macro')  # 宏平均召回率
        f1 = f1_score(Y_test_sorted, Y_pred_sorted, average='macro')  # 宏平均F1分数
        print(model_name, 'accuracy:', accuracy, 'f1:', f1)

        ax1 = fig.add_subplot(gs[mindex // 2, mindex % 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=class_labels,
                    yticklabels=class_labels,ax=ax1)
        ax1.set_title(model_name+' '+str(round(accuracy,2)))
        ax1.set_xlabel('Predicted Labels')  # 替代plt.xlabel()
        ax1.set_ylabel('True Labels')  # 替代plt.ylabel()
    plt.subplots_adjust(wspace=0.8, hspace=0.8, right=0.95, left=0.2, top=0.95, bottom=0.18)
    plt.savefig(os.path.join('results', 'neuron_representation', 'decode_{}.png'.format(savename)), dpi=600,format='png')

if __name__ == "__main__":

    # expname_list=['ento_exp1','ento_exp2','ento_exp3','ento_exp4','ento_exp5','ento_exp6','ento_exp7','ento_exp8','ento_exp9',
    #               'ento_exp10','ento_exp11','ento_exp12','ento_exp13','ento_exp15','ento_exp16','ento_exp17','ento_exp18','ento_exp19','ento_exp20','ento_exp21']
    # expname_list = ['ni_exp1','ni_exp2','ni_exp4','ni_exp5','ni_exp6','ni_exp7','ni_exp8','ni_exp9',
    #               'ni_exp10','ni_exp11','ni_exp13','ni_exp14','ni_exp15','ni_exp16','ni_exp18']
    # expname_list = ['mvl_exp1','mvl_exp2','mvl_exp3','mvl_exp4','mvl_exp5','mvl_exp6','mvl_exp7','mvl_exp8','mvl_exp9',
    #               'mvl_exp10','mvl_exp11','mvl_exp12','mvl_exp13']
    expname_list=['ento_exp1','ento_exp2','ento_exp3','ento_exp4','ento_exp5','ento_exp6','ento_exp7','ento_exp8','ento_exp9',
                  'ento_exp10','ento_exp11','ento_exp12','ento_exp13','ento_exp15','ento_exp16','ento_exp17','ento_exp18','ento_exp19','ento_exp20','ento_exp21',
                  'ni_exp1', 'ni_exp2', 'ni_exp4', 'ni_exp5', 'ni_exp6', 'ni_exp7', 'ni_exp8', 'ni_exp9',
                                 'ni_exp10','ni_exp11','ni_exp13','ni_exp14','ni_exp15','ni_exp16','ni_exp18']
    savename=expname_list[0]
    decoder='svr'   #'regressor','xgb'

    # 1.加载神经元响应，确定最佳试次组合的响应
    neuron_path = r'results/neuron_mapping'
    neuron_resp=[]
    corr_thre=0.1
    for exp in expname_list:
        for filename in os.listdir(os.path.join(neuron_path,exp)):
            if filename.endswith('.pickle') or filename.endswith('.pkl'):
                file_path = os.path.join(neuron_path, exp, filename)
                print(file_path)
                with open(file_path, 'rb') as f:
                    neuron_dict = pickle.load(f)
                if neuron_dict['trial_corr_class']>corr_thre:
                    neuron_resp.append(return_best_resp(neuron_dict, decoder))
    neuron_resp=np.array(neuron_resp)
    # 类别标签
    class_labels = ['butterfly', 'elephant',  'faces',  'pigeon',
                     'backpack', 'beermug',  'cowboyhat','electricguitar']
    # 图像类别
    class_names=neuron_dict['name_class']
    label_num=[class_labels.index(i.split('_')[2]) for i in class_names]

    # 2. 表征相似性分析
    rsm_analysis(neuron_resp, class_labels,savename)

    # 3.降维分析
    reducedim_analysis(neuron_resp, label_num, class_names,savename)

    # 4.解码分析
    neuron_decode(neuron_resp,label_num,class_labels,savename)

############################################################################################################################################
    neuron_path = r'H:\Image_Paper\Code_Acute_Image\results\neuron_mapping'
    neuron_resp=[]
    corr_thre=0.1
    corr_color_features_pca=[]
    corr_shape_features_pca=[]
    corr_texture_features_pca=[]
    corr_hmax_c2_pca=[]
    corr_alexnet_features5_pca=[]
    corr_alexnet_features12_pca = []
    for exp in expname_list:
        for filename in os.listdir(os.path.join(neuron_path,exp)):
            if filename.endswith('.pickle') or filename.endswith('.pkl'):
                file_path = os.path.join(neuron_path, exp, filename)
                print(file_path)
                with open(file_path, 'rb') as f:
                    neuron_dict = pickle.load(f)
                corr_color_features_pca.append(neuron_dict['decode']['color_features_pca']['class']['mean']['svr']['correlation'])
                corr_shape_features_pca.append(
                    neuron_dict['decode']['shape_features_pca']['class']['mean']['svr']['correlation'])
                corr_texture_features_pca.append(
                    neuron_dict['decode']['texture_features_pca']['class']['mean']['svr']['correlation'])
                corr_hmax_c2_pca.append(
                    neuron_dict['decode']['hmax_c2_pca']['class']['mean']['svr']['correlation'])
                corr_alexnet_features5_pca.append(
                    neuron_dict['decode']['alexnet_features.5_pca']['class']['mean']['svr']['correlation'])
                corr_alexnet_features12_pca.append(
                    neuron_dict['decode']['alexnet_features.12_pca']['class']['mean']['svr']['correlation'])


import matplotlib.pyplot as plt

# 数据列表
corr_lists = [
    corr_color_features_pca,
    corr_shape_features_pca,
    corr_texture_features_pca,
    corr_hmax_c2_pca,
    corr_alexnet_features5_pca,
    corr_alexnet_features12_pca
]

titles = [
    'Color Features',
    'Shape Features',
    'Texture Features',
    'HMAX C2',
    'AlexNet Layer 5',
    'AlexNet Layer 12'
]

# 绘图
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
axs = axs.flatten()

for i, (data, title) in enumerate(zip(corr_lists, titles)):
    axs[i].hist(data, bins=20, color='gray', edgecolor='black')
    axs[i].set_title(title, fontsize=10)
    axs[i].set_xlabel('Correlation')
    axs[i].set_ylabel('Count')
    axs[i].tick_params(labelsize=8)

plt.tight_layout()
plt.show()


#######################################################################
def return_best_resp_view(neuron_dict, decoder):
    # 获取neuron_data中最佳响应，返回最大相关性
    feature_lists = list(neuron_dict['decode'].keys())
    resp_lists = list(neuron_dict['decode'][feature_lists[0]]['view'].keys())
    model_list = list(neuron_dict['decode'][feature_lists[0]]['view'][resp_lists[0]].keys())
    for model_name in model_list:
        df = pd.DataFrame(index=resp_lists, columns=feature_lists, dtype=float)
        for f in feature_lists:
            for r in resp_lists:
                df.at[r, f] = round(neuron_dict['decode'][f]['view'][r][model_name]['correlation'], 2)
    max_value =  np.nanmax(df.values)
    max_row, max_col = np.where(df == max_value)
    max_resp_label = df.index[max_row[0]]
    max_feature_label = df.columns[max_col[0]]
    resp_values = neuron_dict['decode'][max_feature_label]['view'][max_resp_label][decoder]['Y_test']
    return resp_values

neuron_path = r'H:\Image_Paper\Code_Acute_Image\results\neuron_mapping'
neuron_resp = []
corr_thre = 0.1
for exp in expname_list:
    for filename in os.listdir(os.path.join(neuron_path, exp)):
        if filename.endswith('.pickle') or filename.endswith('.pkl'):
            file_path = os.path.join(neuron_path, exp, filename)
            print(file_path)
            with open(file_path, 'rb') as f:
                neuron_dict = pickle.load(f)
            if neuron_dict['trial_corr_view'] > corr_thre:
                neuron_resp.append(return_best_resp_view(neuron_dict, decoder))
neuron_resp = np.array(neuron_resp)
name_view=neuron_dict['name_view']

import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
def circular_smooth(arr, window=3):
    """
    对一维数组进行环形滑动平均（考虑首尾相接）
    """
    padded = np.concatenate([arr[-1:], arr, arr[:1]])  # 头尾各加1个，确保3个窗口
    smooth = np.convolve(padded, np.ones(window)/window, mode='valid')
    return smooth

def plot_neuron_view_tuning(neuron_resp, name_view, neuron_idx=0):
    """
    绘制一个神经元对6个物体的视角调谐曲线（18个视角）
    """
    n_images = len(name_view)
    assert neuron_resp.shape[1] == n_images

    # 建立物体名到其对应视角响应的映射
    object_dict = defaultdict(lambda: [None] * 18)

    for i, name in enumerate(name_view):
        # 提取物体名和视角编号
        match = re.match(r'view_.*?_(.*?)_(\d+)\.png', name)
        if match:
            object_name = match.group(1)
            view_id = int(match.group(2))
            object_dict[object_name][view_id] = neuron_resp[neuron_idx, i]

    # 绘图
    plt.figure(figsize=(7, 4))
    for obj, responses in object_dict.items():
        if None not in responses:  # 确保该物体视角完整
            resp_array = np.array(responses)
            smooth_resp = circular_smooth(resp_array, window=3)
            plt.plot(range(18), smooth_resp, label=obj)

    plt.xlabel('View Index (0–17)')
    plt.ylabel('Response')
    plt.title(f'Neuron {neuron_idx} - View Tuning Across Objects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_neuron_view_tuning(neuron_resp, name_view, neuron_idx=100)  # 绘制第0号神经元

import matplotlib.pyplot as plt
import numpy as np

# 假设变量定义
# neuron_resp: shape (499, 108)
# name_view: list of 108 names like 'view_animate_elephant_00.png'
# object_names: 提取的类别名，例如 ['elephant', 'beermug', ...]

# 提取6个物体名的顺序
object_names = sorted(list(set([n.split('_')[2] for n in name_view])))
object_indices = {obj: [] for obj in object_names}

# 获取每类物体对应的索引
for idx, name in enumerate(name_view):
    obj = name.split('_')[2]
    object_indices[obj].append(idx)

# 指定要绘制的12个神经元索引（示例）
selected_neurons = np.linspace(0, 498, 12, dtype=int)

# 创建 3×4 子图
fig, axs = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
axs = axs.flatten()

for i, neuron_idx in enumerate(selected_neurons):
    ax = axs[i]
    for obj in object_names:
        idxs = object_indices[obj]
        responses = neuron_resp[neuron_idx, idxs]

        # 平滑：环形卷积窗口大小为3
        responses = np.convolve(np.r_[responses[-1], responses, responses[0]], np.ones(3) / 3, mode='valid')

        ax.plot(responses, label=obj)

    ax.set_title(f'Neuron {neuron_idx}', fontsize=10)
    ax.set_ylim([0, np.nanmax(neuron_resp)])
    ax.grid(True)

    if i == len(selected_neurons) - 1:
        ax.legend(fontsize=6)

plt.tight_layout()
plt.show()

neuron_resp = np.array(neuron_resp)
name_view=neuron_dict['name_view']

view_labels = ['elephant','faces','pigeon','beermug','cowboyhat','electricguitar']
label_num=[view_labels.index(i.split('_')[2]) for i in name_view]


savename='view'
rsm_analysis(neuron_resp, view_labels, savename)

# 3.降维分析
reducedim_analysis(neuron_resp, label_num, name_view, savename)

# 4.解码分析
neuron_decode(neuron_resp, label_num, view_labels, savename)

correlation_matrix = np.corrcoef(neuron_resp, rowvar=False)  # 计算皮尔逊相关性矩阵
# 可视化相关性矩阵
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
plt.xticks(ticks=list(range(9, 108, 18)), labels=view_labels, rotation=45, ha='right')
plt.yticks(ticks=list(range(9, 108, 18)), labels=view_labels, rotation=0)
plt.title('neuron representation rsm')
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.95, left=0.25, top=0.90, bottom=0.25)
plt.savefig(os.path.join('results', 'neuron_representation', 'rsm_{}.png'.format(savename)), dpi=600, format='png')