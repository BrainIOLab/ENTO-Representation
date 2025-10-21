import os
fontsize_    = 8
fontname_    = 'Arial'
fontweight_  = 'normal'
linewidth_ax = 0.7

# ---- 工具函数：统一子图外观 ----
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def moving_avg_same(raster_trial, bin_size=50, step=1):
    """
    raster_trial: (time x trials)，如 (4000, 200)
    bin_size: 窗宽（例如 50）
    step: 步长（=1 时与卷积结果等长；>1 时做下采样）
    返回:
        rate: (time x trials) 若 step=1；若 step>1 则 (ceil(time/step) x trials)
    """
    kernel = np.ones(bin_size, dtype=float) / bin_size  # 平均核
    # 沿时间轴对每个 trial 做 1D 卷积，same 模式保证长度不变
    rate_full = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                                    axis=0, arr=raster_trial)
    # 步长=1：保持与输入同维度；步长>1：做等间隔采样
    if step == 1:
        return rate_full
    else:
        return rate_full[::step, :]

def assembly_neuron_resp(ndata_class):
    # 确定神经元响应特征组合形式
    resp = {}
    resp['trial_1']=ndata_class[:,0]                        # 取第一个试次
    resp['trial_2']=ndata_class[:,1]                        # 取第二个试次
    resp['trial_3']=ndata_class[:,2]                        # 取第三个试次
    resp['mean']=np.mean(ndata_class, 1)                    # 取平均
    resp['max']=np.max(ndata_class, 1)                      # 取最大
    resp['trial_1_2']=np.mean(ndata_class[:,[0, 1]], 1)     # 取1、2试次
    resp['trial_1_3']=np.mean(ndata_class[:,[0, 2]], 1)     # 取1、3试次
    resp['trial_2_3']=np.mean(ndata_class[:,[1, 2]], 1)     # 取2、3试次

    # 取3个数据中相近的两个的平均值
    b = np.sort(ndata_class, 1)
    a = np.diff(b, axis=1)
    c = np.argmin(a, 1)
    d = (b[:,0] + b[:,1]) / 2
    e = (b[:,1] + b[:,2]) / 2
    f = d * (1 - c) + e * c
    resp['trial_corr']=f
    return resp

def get_final_response(resp_class):
    # 计算相关系数矩阵 (3x3)
    corr_matrix = np.corrcoef(resp_class, rowvar=False)

    # 找出非对角线中最大相关性
    tril_idx = np.tril_indices_from(corr_matrix, k=-1)
    max_idx = np.argmax(corr_matrix[tril_idx])
    i, j = tril_idx[0][max_idx], tril_idx[1][max_idx]

    # 取相关性最高的两列试次
    final_response = resp_class[:, [i, j]].mean(axis=1)
    max_corr = corr_matrix[i, j]
    return final_response, max_corr

def _beautify_ax(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    if title:
        ax.set_title(title, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.tick_params(axis='both', width=linewidth_ax)
    plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

def plot_hist_grid(data_dict, main_title, bins=20, xlim=None, xlabel='', ylabel='Count',
                   figsize_base=(3.8, 2.5), savepath=None):
    """
    data_dict: {feature_name: [values...]}
    每个键绘一个子图；每行4列。
    """
    features = list(data_dict.keys())
    n = len(features)
    ncols = 4
    nrows = math.ceil(n / ncols)

    # 动态放缩画布大小：每格子近似以 (3.8, 2.5) 为基准
    fig_w = figsize_base[0] * ncols
    fig_h = figsize_base[1] * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.3, hspace=0.5)

    for i, feat in enumerate(features):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])

        vals = np.asarray(data_dict[feat], dtype=float)
        vals = vals[~np.isnan(vals)]  # 去掉NaN
        if vals.size == 0:
            # 空数据占位
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            _beautify_ax(ax, xlabel=xlabel, ylabel=ylabel, title=feat)
            continue

        ax.hist(vals, bins=bins, alpha=0.8, edgecolor='black', linewidth=0.5)
        if xlim is not None:
            ax.set_xlim(xlim)
        _beautify_ax(ax, xlabel=xlabel, ylabel=ylabel, title=feat)

    fig.suptitle(main_title, y=0.995, fontsize=fontsize_, fontname=fontname_, fontweight='bold')

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def analyse_decode_class(resp_class, name_class, model_name, kflod, savename):
    #类别使用五折交叉验证，用全部类别图像训练，然后在视角图像上预测
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

    cv = StratifiedKFold(n_splits=kflod)  # 创建交叉验证
    if model_name == 'SVM':
        model = SVC(kernel='rbf', decision_function_shape='ovr')
    if model_name == 'LDA':
        model = LinearDiscriminantAnalysis()
    if model_name == 'BP':
        model = MLPClassifier(hidden_layer_sizes=(16,), solver='sgd', max_iter=300, early_stopping=True)

    dict_decode = {}
    Xdata = np.transpose(resp_class)
    #精细类别解码
    labels_fine = [i.split('_')[2] for i in name_class]
    class_fine = list(dict.fromkeys(labels_fine))
    y_pred = cross_val_predict(model, Xdata, labels_fine, cv=cv)
    dict_decode['fine_cm'] = confusion_matrix(labels_fine, y_pred, labels=class_fine)  # 混淆矩阵
    dict_decode['fine_accuracy'] = accuracy_score(labels_fine, y_pred)  # 准确率
    dict_decode['fine_precision'] = precision_score(labels_fine, y_pred, average='macro')  # 宏平均精确率
    dict_decode['fine_recall'] = recall_score(labels_fine, y_pred, average='macro')  # 宏平均召回率
    dict_decode['fine_f1'] = f1_score(labels_fine, y_pred, average='macro')  # 宏平均F1分数
    dict_decode['accuracy_face'] = dict_decode['fine_cm'][2, 2]/25  # 宏平均F1分数
    dict_decode['accuracy_pigeon'] = dict_decode['fine_cm'][3, 3] / 25  # 宏平均F1分数

    #粗类别解码
    labels_coarse = [i.split('_')[1] for i in name_class]
    class_coarse = list(dict.fromkeys(labels_coarse))
    y_pred = cross_val_predict(model, Xdata, labels_coarse, cv=cv)
    dict_decode['coarse_cm'] = confusion_matrix(labels_coarse, y_pred, labels=class_coarse)  # 混淆矩阵
    dict_decode['coarse_accuracy'] = accuracy_score(labels_coarse, y_pred)  # 准确率
    dict_decode['coarse_precision'] = precision_score(labels_coarse, y_pred, average='macro')  # 宏平均精确率
    dict_decode['coarse_recall'] = recall_score(labels_coarse, y_pred, average='macro')  # 宏平均召回率
    dict_decode['coarse_f1'] = f1_score(labels_coarse, y_pred, average='macro')  # 宏平均F1分数

    #绘制图像
    if savename != "None":
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import gridspec
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 2)

        ax = fig.add_subplot(gs[0, 0])
        sns.heatmap(dict_decode['fine_cm'], annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=class_fine,
                    yticklabels=class_fine, ax=ax)
        ax.set_title(model_name + ' ' + 'fine ' + 'class ' + str(round(dict_decode['fine_accuracy'], 2)))
        ax.set_xlabel('Predicted Labels')  # 替代plt.xlabel()
        ax.set_ylabel('True Labels')  # 替代plt.ylabel()

        ax = fig.add_subplot(gs[0, 1])
        sns.heatmap(dict_decode['coarse_cm'], annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=class_coarse,
                    yticklabels=class_coarse, ax=ax)
        ax.set_title(
            model_name + ' ' + 'coarse ' + 'class ' + str(round(dict_decode['coarse_accuracy'], 2)))
        ax.set_xlabel('Predicted Labels')  # 替代plt.xlabel()
        ax.set_ylabel('True Labels')  # 替代plt.ylabel()

        plt.subplots_adjust(wspace=0.5, hspace=1.1, right=0.95, left=0.2, top=0.9, bottom=0.4)
        plt.savefig(os.path.join(r'results', 'class', 'decode_class_{}.png'.format(savename)), dpi=300, format='png')

    return dict_decode


def analyse_decode_view(resp_view, name_view, resp_class, name_class, model_name, kflod, savename):
    #类别使用五折交叉验证，用全部类别图像训练，然后在视角图像上预测
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

    if model_name == 'SVM':
        model = SVC(kernel='rbf', decision_function_shape='ovr')
    if model_name == 'LDA':
        model = LinearDiscriminantAnalysis()
    if model_name == 'BP':
        model = MLPClassifier(hidden_layer_sizes=(16,), solver='sgd', max_iter=300, early_stopping=True)

    dict_decode = {}
    #使用类别训练模型,在视角粗类别解码
    X_test = np.transpose(resp_view)
    labels_coarse = [i.split('_')[1] for i in name_view]
    class_coarse = list(dict.fromkeys(labels_coarse))
    X_train = np.transpose(resp_class)
    Y_train = [i.split('_')[1] for i in name_class]
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)  # 6-8个
    dict_decode['coarse_cm'] = confusion_matrix(labels_coarse, y_pred, labels=class_coarse)  # 混淆矩阵
    dict_decode['coarse_accuracy'] = accuracy_score(labels_coarse, y_pred)  # 准确率
    dict_decode['coarse_precision'] = precision_score(labels_coarse, y_pred, average='macro')  # 宏平均精确率
    dict_decode['coarse_recall'] = recall_score(labels_coarse, y_pred, average='macro')  # 宏平均召回率
    dict_decode['coarse_f1'] = f1_score(labels_coarse, y_pred, average='macro')  # 宏平均F1分数

    # 使用类别训练模型,在视角精细类别解码
    labels_fine = [i.split('_')[2] for i in name_view]
    class_fine = list(dict.fromkeys(labels_fine))

    X_train = np.transpose(resp_class)
    Y_train = [i.split('_')[2] for i in name_class]
    mask = [y in class_fine for y in Y_train]
    X_selected = X_train[mask, :]  # (N_selected, 835)
    Y_selected = np.array(Y_train)[mask]  # (N_selected,)
    model.fit(X_selected, Y_selected)
    y_pred = model.predict(X_test)  # 6-8个
    dict_decode['fine_cm'] = confusion_matrix(labels_fine, y_pred, labels=class_fine)  # 混淆矩阵
    dict_decode['fine_accuracy'] = accuracy_score(labels_fine, y_pred)  # 准确率
    dict_decode['fine_precision'] = precision_score(labels_fine, y_pred, average='macro')  # 宏平均精确率
    dict_decode['fine_recall'] = recall_score(labels_fine, y_pred, average='macro')  # 宏平均召回率
    dict_decode['fine_f1'] = f1_score(labels_fine, y_pred, average='macro')  # 宏平均F1分数

    #绘制图像
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    sns.heatmap(dict_decode['fine_cm'], annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_fine,
                yticklabels=class_fine, ax=ax)
    ax.set_title(model_name + ' ' + 'fine ' + 'view ' + str(round(dict_decode['fine_accuracy'], 2)))
    ax.set_xlabel('Predicted Labels')  # 替代plt.xlabel()
    ax.set_ylabel('True Labels')  # 替代plt.ylabel()

    ax = fig.add_subplot(gs[0, 1])
    sns.heatmap(dict_decode['coarse_cm'], annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_coarse,
                yticklabels=class_coarse, ax=ax)
    ax.set_title(
        model_name + ' ' + 'coarse ' + 'view ' + str(round(dict_decode['coarse_accuracy'], 2)))
    ax.set_xlabel('Predicted Labels')  # 替代plt.xlabel()
    ax.set_ylabel('True Labels')  # 替代plt.ylabel()

    plt.subplots_adjust(wspace=0.5, hspace=1.1, right=0.95, left=0.2, top=0.9, bottom=0.4)
    plt.savefig(os.path.join(r'results', 'neuron_represent', 'decode_view_{}.png'.format(savename)), dpi=300, format='png')

    return dict_decode

#可视化
def scatter_2d(X_reduced, name_class, reducedim_method, savename):
    from sklearn.preprocessing import LabelEncoder
    labels_fine=[i.split('_')[2] for i in name_class]
    labels_coarse=[i.split('_')[1] for i in name_class]

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 2)

    cmap = plt.cm.get_cmap('tab20', 8)                  # 16类
    ax = fig.add_subplot(gs[0, 0])                      #绘制调谐曲线
    le = LabelEncoder()
    labels_num = le.fit_transform(labels_fine)          # 自动转为 0,1,2,... 数字
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_num, cmap=cmap, s=30, edgecolors='k')
    ax.axis('equal')
    cbar = fig.colorbar(scatter, ax=ax, label='Label')  # 注意这里需要传入fig和ax1
    ax.set_title('{}'.format(reducedim_method))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['orange', 'red'])
    ax = fig.add_subplot(gs[0, 1])
    le = LabelEncoder()
    labels_num = le.fit_transform(labels_coarse)
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_num, cmap=cmap, s=30, edgecolors='k')
    ax.axis('equal')
    cbar = fig.colorbar(scatter, ax=ax, label='Label')  # 注意这里需要传入fig和ax1
    ax.set_title('{}'.format(reducedim_method))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')

    plt.subplots_adjust(wspace=0.5, hspace=0.5, right=0.95, left=0.1, top=0.90, bottom=0.18)
    plt.savefig(os.path.join(r'results', 'class', 'scatter2d_{}_{}.png'.format(savename, reducedim_method)), dpi=300, format='png')

def scatter_2d_image(X_reduced, name_class, reducedim_method, savename,rtype,img_size):
    img_size = img_size  # 图像中单个物体图像的大小
    figure_size = 720
    import cv2
    index = np.argmax(np.max(X_reduced, 0) - np.min(X_reduced, 0))  # 建立映射
    min_old, max_old = np.min(X_reduced, 0)[index], np.max(X_reduced, 0)[index]
    min_new, max_new = img_size / 2, figure_size - img_size / 2
    mapped_data = np.interp(X_reduced, (min_old, max_old), (min_new, max_new))  # 转换为图像尺寸空间
    image_bg = np.ones((figure_size, figure_size, 3), dtype=np.uint8) * 255  # 创建白色背景
    for i_index, i in enumerate(name_class):
        imgpath = os.path.join(r'Acute_Image\image\img_rgb', i.split('.')[0][0:-3], i)
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
    cv2.imwrite(os.path.join(r'results', rtype, 'image_{}_{}.png'.format(savename, reducedim_method)),image_bg)


def plot_rdm(rdm, name_class, reducedim_method, savename):
    labels_coarse=[i.split('_')[1] for i in name_class]
    class_coarse=list(dict.fromkeys(labels_coarse))

    labels_fine=[i.split('_')[2] for i in name_class]
    class_fine=list(dict.fromkeys(labels_fine))

    import seaborn as sns
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])  # 绘制调谐曲线
    cmap = 'coolwarm'
    vmax = float(np.max(rdm))
    heatmap = sns.heatmap(
        rdm, ax=ax, cmap=cmap, cbar=True, vmin=0, vmax=vmax,
        square=True, xticklabels=False, yticklabels=False
    )
    num_classes=len(class_fine)
    num_objects_per_class=len(labels_fine)/num_classes
    N = num_classes * num_objects_per_class
    centers = np.arange(num_objects_per_class / 2, N, num_objects_per_class)
    # X 轴标签在顶部或底部都行；下面演示放在底部：
    ax.set_xticks(centers)
    ax.set_xticklabels(class_fine, rotation=45, ha='right')
    ax.set_yticks(centers)
    ax.set_yticklabels(class_fine, rotation=0)
    boundaries = [i * num_objects_per_class for i in range(1, num_classes)]
    for b in boundaries:
        ax.axhline(b, color='w', lw=0.6, ls='-')   # 水平分割线
        ax.axvline(b, color='w', lw=0.6, ls='-')   # 垂直分割线

    plt.subplots_adjust(wspace=0.5, hspace=0.5, right=0.95, left=0.25, top=0.90, bottom=0.25)
    plt.savefig(os.path.join(r'results', 'class', 'rdm_{}_{}.png'.format(savename, reducedim_method)), dpi=300, format='png')


def analyse_reducedim_2d(resp_class, name_class, reducedim_method, savename):

    neuron_resp=np.transpose(resp_class)
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

    from sklearn.metrics import pairwise_distances
    rdm = pairwise_distances(X_reduced, metric='euclidean')

    if savename != "None":
        #可视化散点图
        scatter_2d(X_reduced, name_class, reducedim_method, savename)
        #可视化散点图-图像
        scatter_2d_image(X_reduced, name_class, reducedim_method, savename, 'class', 44)
        #可视化rdm
        plot_rdm(rdm, name_class, reducedim_method, savename)

    return rdm, X_reduced



def scatter_2d_single(X_reduced, reducedim_method, savename):

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=(3.5, 3))
    gs = gridspec.GridSpec(1, 1)

    cmap = plt.cm.get_cmap('tab20', 8)                  # 16类
    ax = fig.add_subplot(gs[0, 0])                      #绘制调谐曲线
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c='red', cmap=cmap, s=30, edgecolors='k')
    ax.plot(
        np.append(X_reduced[:, 0], X_reduced[0, 0]),
        np.append(X_reduced[:, 1], X_reduced[0, 1]),
        '-o', color='blue', markersize=2, linewidth=1
    )
    ax.axis('equal')
    ax.set_title('{}'.format(reducedim_method))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')

    plt.subplots_adjust(wspace=0.5, hspace=0.5, right=0.90, left=0.2, top=0.90, bottom=0.18)
    plt.savefig(os.path.join(r'results', 'view', 'scatter2d_single_{}_{}.png'.format(savename, reducedim_method)), dpi=300, format='png')

def plot_rdm_single(rdm, reducedim_method, savename):

    import seaborn as sns
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])  # 绘制调谐曲线
    cmap = 'coolwarm'
    vmax = float(np.max(rdm))
    heatmap = sns.heatmap(
        rdm, ax=ax, cmap=cmap, cbar=True, vmin=0, vmax=vmax,
        square=True, xticklabels=False, yticklabels=False
    )
    n = rdm.shape[0]  # 18
    tick_pos = np.linspace(0, n - 1, 4, dtype=int)   # 4 个位置
    tick_labels = [str(i + 1) for i in tick_pos]     # 标签从 1 开始

    ax.set_xticks(tick_pos + 0.5)  # heatmap 的 tick 对应 cell 中心，所以 +0.5
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_pos + 0.5)
    ax.set_yticklabels(tick_labels)

    ax.set_title(savename)
    plt.subplots_adjust(wspace=0.5, hspace=0.5, right=0.95, left=0.25, top=0.90, bottom=0.25)
    plt.savefig(os.path.join(r'results', 'view', 'rdm_single_{}_{}.png'.format(savename, reducedim_method)), dpi=300, format='png')

def analyse_reducedim_2d_singleview(resp_view, name_view, reducedim_method, savename):

    neuron_resp=np.transpose(resp_view)
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

    from sklearn.metrics import pairwise_distances
    rdm = pairwise_distances(X_reduced, metric='euclidean')

    if savename != "None":
        #可视化散点图
        scatter_2d_single(X_reduced, reducedim_method, savename)

        #可视化散点图-图像
        scatter_2d_image(X_reduced, name_view, reducedim_method, savename, 'view', 120)

        #可视化rdm
        plot_rdm_single(rdm, reducedim_method, savename)

    return rdm, X_reduced

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

def cal_class_represent(rdm,name_class):
    dict_class_metrics={}
    labels_fine = [i.split('_')[2] for i in name_class]
    # labels_coarse = [i.split('_')[1] for i in name_class]
    dist_intra_fine, dist_inter_fine = compute_class_distance(rdm, labels_fine)
    # dist_intra_coarse, dist_inter_coarse = compute_class_distance(rdm, labels_coarse)
    Separability_fine = dist_inter_fine / (dist_intra_fine + 1e-9)  # 计算类分离比
    # Separability_coarse = dist_inter_coarse / (dist_intra_coarse + 1e-9)  # 计算类分离比

    dict_class_metrics['Intra-class distance'] = dist_intra_fine
    dict_class_metrics['Inter-class distance'] = dist_inter_fine
    dict_class_metrics['Separability rate'] = Separability_fine

    # dict_class_metrics['dist_intra_coarse'] = dist_intra_fine
    # dict_class_metrics['dist_inter_coarse'] = dist_inter_fine
    # dict_class_metrics['Separability_coarse'] = Separability_fine

    return dict_class_metrics


def scatter_3d(X_reduced, name_class, reducedim_method, savename):
    from sklearn.preprocessing import LabelEncoder
    labels_fine = [i.split('_')[2] for i in name_class]
    labels_coarse = [i.split('_')[1] for i in name_class]
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from mpl_toolkits.mplot3d import Axes3D
    # 创建3D图形
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2)
    # 精细标签的3D图
    cmap = plt.cm.get_cmap('tab20', 20)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    le = LabelEncoder()
    labels_num = le.fit_transform(labels_fine)
    scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                           c=labels_num, cmap=cmap, s=50, edgecolors='k', alpha=0.8)
    ax1.set_title('{} - Fine Labels'.format(reducedim_method))
    ax1.set_xlabel('dim 1')
    ax1.set_ylabel('dim 2')
    ax1.set_zlabel('dim 3')
    # 添加颜色条
    cbar1 = fig.colorbar(scatter1, ax=ax1, label='Fine Label', shrink=0.6, pad=0.12)
    # 粗粒度标签的3D图
    from matplotlib.colors import ListedColormap
    cmap_coarse = ListedColormap(['orange', 'red', 'blue', 'green', 'purple', 'brown'])
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    le = LabelEncoder()
    labels_num_coarse = le.fit_transform(labels_coarse)
    scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                           c=labels_num_coarse, cmap=cmap_coarse, s=50, edgecolors='k', alpha=0.8)
    ax2.set_title('{} - Coarse Labels'.format(reducedim_method))
    ax2.set_xlabel('dim 1')
    ax2.set_ylabel('dim 2')
    ax2.set_zlabel('dim 3')
    cbar2 = fig.colorbar(scatter2, ax=ax2, label='Coarse Label', shrink=0.6, pad=0.12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5, right=0.95, left=0.1, top=0.90, bottom=0.25)
    plt.savefig(os.path.join(r'results', 'neuron_represent', 'scatter3d_{}_{}.png'.format(savename, reducedim_method)),
                dpi=300, format='png', bbox_inches='tight')


def analyse_reducedim_3d(resp_class, name_class, reducedim_method, savename):
    """
    3D降维分析和表征相似性矩阵
    """
    neuron_resp = np.transpose(resp_class)

    # 3D降维
    if reducedim_method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)  # 降维到3维
        X_reduced = pca.fit_transform(neuron_resp)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    elif reducedim_method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)  # 降维到3维
        X_reduced = tsne.fit_transform(neuron_resp)

    elif reducedim_method == 'MDS':
        from sklearn.manifold import MDS
        mds = MDS(n_components=3, random_state=42)  # 降维到3维
        X_reduced = mds.fit_transform(neuron_resp)

    elif reducedim_method == 'Isomap':
        from sklearn.manifold import Isomap
        isomap = Isomap(n_components=3, n_neighbors=5)  # 降维到3维
        X_reduced = isomap.fit_transform(neuron_resp)

    # 计算表征相似性矩阵
    from sklearn.metrics import pairwise_distances
    rdm = pairwise_distances(X_reduced, metric='euclidean')

    #可视化散点图
    scatter_3d(X_reduced, name_class, reducedim_method, savename)
    plot_rdm(rdm,name_class,reducedim_method,savename+'3d')

    return rdm, X_reduced

#定义类别调谐特性#########################################################################################################
def icc1(X, eps=1e-12):
    """
    一致性指标：ICC(1)
    X: (C, T) 数组 (C=类别数, T=重复次数)
    返回：ICC(1) 值
    """
    C, T = X.shape
    if C < 2 or T < 2:
        return np.nan
    # 每类均值
    m_i = X.mean(axis=1, keepdims=True)  # (C,1)
    # 类间方差均方
    MS_between = T * np.var(m_i.squeeze(-1), ddof=1)
    # 类内方差均方
    SSE = np.sum((X - m_i) ** 2)
    MS_within = SSE / (C * (T - 1))
    denom = MS_between + (T - 1) * MS_within
    return (MS_between - MS_within) / (denom + eps)

# ---------- 2. Split-half correlation ----------
def split_half_corr(X, n_iter=100, random_state=None):
    """
    一致性指标：Split-half correlation
    X: (C, T) 数组
    n_iter: 重复次数（取平均）
    返回：平均 split-half 相关
    """
    rng = np.random.RandomState(random_state)  # 兼容老版本 NumPy
    C, T = X.shape
    if T < 2:
        return np.nan
    corrs = []
    for _ in range(n_iter):
        idx = rng.permutation(T)
        half1_idx = idx[:T//2]
        half2_idx = idx[T//2:]
        R1 = X[:, half1_idx].mean(axis=1)
        R2 = X[:, half2_idx].mean(axis=1)
        # Pearson 相关
        if np.std(R1) < 1e-8 or np.std(R2) < 1e-8:
            corrs.append(0.0)
        else:
            corr = np.corrcoef(R1, R2)[0,1]
            corrs.append(corr)
    return float(np.mean(corrs))

# ---------- 3. SNR ----------
def snr(X, eps=1e-12):
    """
    一致性指标：信噪比 SNR
    X: (C, T) 数组
    返回：SNR = Var_between / Var_within
    """
    # 每类均值 (C,)
    R_mean = X.mean(axis=1)
    var_between = np.var(R_mean, ddof=1)
    var_within = X.var(axis=1, ddof=1).mean()
    return float(var_between / (var_within + eps))


def _mutual_info_single(labels, values):
    """
    计算离散互信息 MI(C;R)，对连续响应 values 分箱。
    labels: (M,), 取值 0..C-1
    values: (M,), 连续值
    """
    nbins = 10
    eps = 1e-12
    values = np.asarray(values)
    labels = np.asarray(labels)
    vmin, vmax = np.min(values), np.max(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0

    bins = np.linspace(vmin, vmax, nbins + 1)
    # digitize 结果范围 1..nbins
    digitized = np.digitize(values, bins[:-1], right=False)
    digitized[digitized > nbins] = nbins

    C = int(labels.max()) + 1
    M = labels.size

    # P(c)
    Pc = np.bincount(labels, minlength=C).astype(float)
    Pc /= Pc.sum() + eps

    # P(r)（忽略 0 桶）
    Pr = np.bincount(digitized, minlength=nbins + 1).astype(float)[1:]
    Pr /= Pr.sum() + eps

    # P(c,r)
    Pcr = np.zeros((C, nbins), dtype=float)
    for c in range(C):
        idx = (labels == c)
        if np.any(idx):
            dr = digitized[idx]
            hist = np.bincount(dr, minlength=nbins + 1).astype(float)[1:]
            Pcr[c] = hist / (M + eps)

    # MI = sum P(c,r) log( P(c,r)/(P(c)P(r)) )
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = (Pc[:, None] * Pr[None, :]) + eps
        log_term = np.where(Pcr > 0, np.log(Pcr / denom), 0.0)
        MI = float(np.sum(Pcr * log_term))
    return MI


def compute_class_tuning(resp_class):
    class_dict = {}

    num_class = 8
    num_object = 25
    resp_matrix = resp_class.reshape(num_class, num_object)  # 转换为类别×个体的形式

    eps = 1e-12
    R_mean = resp_matrix.mean(axis=1)

    # 类内方差：对每类沿试次求方差，再在类别上取平均
    var_within_class = resp_matrix.var(axis=1, ddof=1)  # (C,)
    var_within = float(var_within_class.mean())
    # 类间方差：对类均值在类别上求方差
    var_between = float(np.var(R_mean, ddof=1))
    # 分离比
    separation_ratio = float(var_between / (var_within + eps))
    # 类别选择性
    R_bar = float(R_mean.mean())
    R_max = float(R_mean.max())
    CSI = float((R_max - R_bar) / (R_max + R_bar + eps))
    # 类别稀疏性指标
    mean_R = float(R_mean.mean())
    mean_R2 = float((R_mean ** 2).mean())
    sparseness = float((1.0 - (mean_R ** 2) / (mean_R2 + eps)) / (1.0 - 1.0 / num_class + eps))
    # 一致性指标：单因子随机效应 ICC(1)
    # ICC1 = icc1(resp_matrix)
    # 试次相关系数（Split-half correlation）
    halfcorr = split_half_corr(resp_matrix, n_iter=200)
    # 互信息
    labels_all = np.repeat(np.arange(num_class), num_object)  # (C*T,)
    values_all = resp_matrix.reshape(num_class * num_object)
    mutual_info = float(_mutual_info_single(labels_all, values_all))

    class_dict['Within variance'] = var_within
    class_dict['Between variance'] = var_between
    class_dict['Separation ratio'] = separation_ratio
    class_dict['Selectivity index'] = CSI
    class_dict['Sparseness'] = sparseness
    class_dict['Split-half reliability'] = halfcorr
    class_dict['Mutual information'] = mutual_info
    return class_dict


#定义视角调谐特性#########################################################################################################
def smooth_moving_average(r, window=3):
    """环形移动平均平滑"""
    r = np.asarray(r, float)
    V = len(r)
    half = window // 2
    r_pad = np.concatenate([r[-half:], r, r[:half]])  # 环形补齐
    smoothed = np.convolve(r_pad, np.ones(window) / window, mode='valid')
    return smoothed

def circular_tuning_stats(r, eps=1e-12):
    """计算首选视角 mu，调谐强度 R，圆方差 CV"""
    V = len(r)
    theta = np.linspace(0, 2*np.pi, V, endpoint=False)
    r_shifted = r - np.min(r) + 1e-9   # 保证非负
    A = np.sum(r_shifted) + eps
    C = np.sum(r_shifted * np.cos(theta))
    S = np.sum(r_shifted * np.sin(theta))
    R = np.sqrt(C*C + S*S) / A
    CV = 1 - R
    mu = np.arctan2(S, C)   # 弧度
    return mu, R, CV

def modulation_depth(r, eps=1e-12):
    rmax, rmin = np.max(r), np.min(r)
    return (rmax - rmin) / (rmax + rmin + eps)

def neighbor_corr_circular(r):
    """邻近平滑度：同时考虑左 shift 和右 shift"""
    r = np.asarray(r, float)
    r_right = np.roll(r, -1)   # 右邻
    r_left  = np.roll(r,  1)   # 左邻

    def safe_corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0,1])

    corr_right = safe_corr(r, r_right)
    corr_left  = safe_corr(r, r_left)

    return (corr_left + corr_right) / 2.0

def smoothness_index_circular(r):
    def _si(x, y):
        eps = 1e-12
        num = np.sum(np.abs(x - y))
        den = np.sum(np.abs(x + y)) + eps
        return 1.0 - float(num / den)
    r_left  = np.roll(r, +1)  # 左邻
    r_right = np.roll(r, -1)  # 右邻

    si_left  = _si(r, r_left)
    si_right = _si(r, r_right)
    si_mean  = 0.5 * (si_left + si_right)

    return si_mean

def mutual_info_view(r, nbins=6, eps=1e-12):
    """互信息 MI(View; Response)，视角=18，响应分箱"""
    V = len(r)
    labels = np.arange(V, dtype=int)
    vmin, vmax = float(np.min(r)), float(np.max(r))
    if vmax == vmin:
        return 0.0
    bins = np.linspace(vmin, vmax, nbins+1)
    digit = np.digitize(r, bins[:-1], right=False)
    digit[digit > nbins] = nbins

    Pv = np.ones(V) / V
    Pr = np.bincount(digit, minlength=nbins+1).astype(float)[1:]
    Pr /= Pr.sum() + eps

    Pvr = np.zeros((V, nbins), float)
    for v in range(V):
        Pvr[v, digit[v]-1] += 1
    Pvr /= V

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = Pv[:,None] * Pr[None,:] + eps
        logterm = np.where(Pvr>0, np.log(Pvr/denom), 0.0)
        MI = float(np.sum(Pvr * logterm))
    return MI

def fisher_info_poisson(r, eps=1e-12):
    """Poisson 假设下 Fisher 信息（均值）"""
    V = len(r)
    dtheta = 2*np.pi / V
    r_shifted = r - np.min(r) + 1e-9   # 保证非负
    r_p = np.roll(r_shifted, -1)
    r_m = np.roll(r_shifted,  1)
    dr = (r_p - r_m) / (2*dtheta)
    J = (dr*dr) / (r_shifted + eps)
    return J, float(np.mean(J))

def mirror_correlation_full(resp_obj):
    r = np.asarray(resp_obj, dtype=float)
    V = len(r)
    all_corrs = []

    for c in range(V):
        # 从中心点开始，向右（顺时针）取 V 个点
        right = [r[(c + d) % V] for d in range(V)]
        # 从中心点开始，向左（逆时针）取 V 个点
        left = [r[(c - d) % V] for d in range(V)]

        left = np.array(left)
        right = np.array(right)

        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(left, right)[0,1])
        all_corrs.append(corr)

    mc = float(np.max(all_corrs))
    best_center = int(np.argmax(all_corrs))
    return mc

def tbi_variance_ratio_collapsed(R, zscore=True, eps=1e-8):
    """
    基于方差比的 TBI（视角不可类比的改造版）
    输入:
        R: shape (6, 18)  行=物体/类别，列=视角（不同物体的视角不可类比）
    思路:
        - 类别强度 S_class: 先对每个物体在视角上取均值 -> 得到 6×1 的向量，
          再对这个 6 维向量计算方差，表示“不同物体整体响应差异”的强度。
        - 视角强度 S_view : 对每个物体（行）在其 18 个视角上计算方差，
          再对 6 个物体取平均，表示“物体内部随视角波动”的强度。
        - TBI = (S_class - S_view) / (S_class + S_view + eps)
          >0 偏类别调谐；<0 偏视角调谐；≈0 二者相当。
    返回:
        tbi, purity, S_class, S_view
    """
    # 类别强度：先跨视角求均值 -> 6个物体总体响应
    obj_mean = np.nanmean(R, axis=1)          # shape (6,)
    ss_class = np.nanvar(obj_mean, ddof=1)    # 6维上的方差

    # 视角强度：每个物体内部跨视角方差，然后在物体间取平均
    ss_view_each = np.nanvar(R, axis=1, ddof=1)   # shape (6,)
    ss_view = np.nanmean(ss_view_each)

    ss_tot = ss_class + ss_view + eps
    S_class = ss_class / ss_tot
    S_view  = ss_view  / ss_tot

    tbi = (S_class - S_view) / (S_class + S_view + eps)
    purity = float(max(S_class, S_view))
    return float(tbi), float(purity), float(S_class), float(S_view)

def tbi_tuning_index_collapsed(R, zscore=True, eps=1e-8):
    """
    基于选择性指数的 TBI（视角不可类比的改造版）
    输入:
        R: (6, 18)
    思路:
        - 视角调谐强度 TI_view:
            对每个物体的 18 视角响应计算 TI_row = (max-min)/(max+min+eps)，
            再在 6 个物体上取平均。
        - 类别调谐强度 TI_class:
            先对每个物体跨视角取均值 -> 6 维向量，
            对该 6 维向量计算 TI_class = (max-min)/(max+min+eps)。
        - TBI = (TI_class - TI_view)/(TI_class + TI_view + eps)
    返回:
        tbi, purity, TI_class, TI_view
    """

    # 视角选择性：逐物体（行）计算
    row_max = np.nanmax(R, axis=1)
    row_min = np.nanmin(R, axis=1)
    ti_view_each = (row_max - row_min) / (row_max + row_min + eps)
    TI_view = float(np.nanmean(ti_view_each))

    # 类别选择性：先跨视角求均值，再在6个物体上做 TI
    obj_mean = np.nanmean(R, axis=1)   # shape (6,)
    omx, omn = np.nanmax(obj_mean), np.nanmin(obj_mean)
    TI_class = float((omx - omn) / (omx + omn + eps))

    tbi = (TI_class - TI_view) / (TI_class + TI_view + eps)
    purity = float(max(TI_class, TI_view))
    return float(tbi), float(purity), float(TI_class), float(TI_view)

def _nanmean(x, axis=None):
    return np.nanmean(x, axis=axis)


def kendalls_w_from_ranks(R_cat_by_view):
    from scipy.stats import spearmanr, rankdata
    """
    R_cat_by_view: array shape (k_categories, n_views)，元素为“分数/响应”
    先按列对6个类别做秩转换，再计算 Kendall's W
    """
    X = np.asarray(R_cat_by_view, float)
    k, n = X.shape  # k=类别数(6), n=视角数(18)
    # 列内秩（并列取平均秩）
    ranks = np.vstack([rankdata(X[:, j], method='average') for j in range(n)]).T  # (k,n)
    R_i = np.sum(ranks, axis=1)                         # 每个类别的秩和
    R_bar = np.mean(R_i)
    S = np.sum((R_i - R_bar)**2)                        # 秩和的离差平方和
    W = 12 * S / (n**2 * (k**3 - k) + 1e-12)            # Kendall's W（无并列修正近似）
    return float(np.clip(W, 0, 1))


def view_sparseness_single(resp_obj, eps=1e-12, nonneg=True):
    """
    视角稀疏性（Vinje & Gallant, 2000）
    S = [1 - (mean(r)^2 / mean(r^2))] / [1 - 1/V],  V = 18
    resp_obj: shape (18,)
    """
    r = np.asarray(resp_obj, float)
    if nonneg:
        # 避免正负抵消：移到非负域
        r = r - np.min(r) + 1e-9

    V = r.size
    r_mean   = np.nanmean(r)
    r_sqmean = np.nanmean(r**2)

    if r_sqmean < eps:
        return 0.0

    S = (1.0 - (r_mean**2) / (r_sqmean + eps)) / (1.0 - 1.0 / V)
    return float(np.clip(S, 0.0, 1.0))

def compute_view_tuning(resp_obj):

    view_dict={}
    # # 调谐强度R，R越大调谐越集中；圆方差CV，CV越小越集中
    # _, R, CV = circular_tuning_stats(resp_obj)
    # # 调谐深度MD
    # MD = modulation_depth(resp_obj)
    sparseness=view_sparseness_single(resp_obj)
    # 邻近平滑度NC，越接近1越连续
    NC = neighbor_corr_circular(resp_obj)
    # 平滑度指数,越接近1越平滑
    smoothness_index = smoothness_index_circular(resp_obj)
    # 计算互信息和Fisher信息
    MI = mutual_info_view(resp_obj, nbins=6)
    J_all, J_mean = fisher_info_poisson(resp_obj)
    # 计算对称性
    mc = mirror_correlation_full(resp_obj)

    view_dict['View sparseness'] = sparseness
    # view_dict['CV'] = CV
    # view_dict['MD'] = MD
    view_dict['Neighbor correlation'] = NC
    view_dict['Smoothness index'] = smoothness_index
    view_dict['Mutual information'] = MI
    view_dict['Fisher information'] = J_mean
    view_dict['Mirror correlation'] = mc

    return view_dict

def compute_viewclass_tuning(resp_view):

    viewclass_dict={}
    # 计算视角/类别选择性
    tbi_var, _, _, _ = tbi_variance_ratio_collapsed(resp_view, zscore=True,
                                                    eps=1e-8)  # >0 偏类别调谐；<0 偏视角调谐；≈0 二者相当。
    tbi_tuning, _, _, _ = tbi_tuning_index_collapsed(resp_view, zscore=True,
                                                     eps=1e-8)  # >0 偏类别调谐；<0 偏视角调谐；≈0 二者相当。
    # 试次相关系数（Split-half correlation）
    halfcorr = split_half_corr(resp_view, n_iter=200)

    kendall_w = kendalls_w_from_ranks(resp_view)

    viewclass_dict['TBI-var'] = tbi_var
    viewclass_dict['TBI-tuning'] = tbi_tuning
    viewclass_dict['Split-half reliability'] = halfcorr
    viewclass_dict['Kendall’s W'] = kendall_w
    return viewclass_dict

#定义视角表征性能#########################################################################################################
def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))

def _rank_matrix(D: np.ndarray) -> np.ndarray:
    """对每一行按距离升序做秩次，自己点记为0。"""
    n = D.shape[0]
    R = np.zeros_like(D, float)
    for i in range(n):
        order = np.argsort(D[i])
        r = np.empty(n, float)
        r[order] = np.arange(n, dtype=float)
        r[i] = 0.0
        R[i] = r
    return R

def _pca_project_to_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:2].T)

def metric_trustworthiness_continuity(X_reduced: np.ndarray, X_high: np.ndarray, k: int = 3):
    """
    计算邻近保持度 (Trustworthiness, Continuity)
    输入:
      - X_reduced: (18,2/3)，降维后的流形坐标
      - X_high:    (18, n_features)，原始高维神经元响应
      - k: 邻居数 (默认3)
    返回:
      dict: {"trustworthiness": ..., "continuity": ...}
    """
    X_high=np.transpose(X_high)
    # 高维 RDM (欧氏距离，也可改成相关距离)
    D_high = _pairwise_dists(X_high)
    # 低维 RDM
    D_low  = _pairwise_dists(X_reduced)
    R_low  = _rank_matrix(D_low)
    R_high = _rank_matrix(D_high)

    n = X_reduced.shape[0]
    T_sum = 0.0
    C_sum = 0.0
    for i in range(n):
        Nl = set(np.where((R_low[i] > 0) & (R_low[i] <= k))[0])
        Nh = set(np.where((R_high[i] > 0) & (R_high[i] <= k))[0])
        impostors = Nl - Nh
        misses    = Nh - Nl
        if impostors:
            T_sum += np.sum(R_high[i, list(impostors)] - k)
        if misses:
            C_sum += np.sum(R_low[i, list(misses)] - k)

    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    trust = 1.0 - norm * T_sum
    conti = 1.0 - norm * C_sum
    return  float(trust), float(conti)

# ============== 2) 邻近平滑度（按环序）=============

def metric_neighbor_smoothness(X_reduced: np.ndarray):
    """
    假定行序 0..17 为环相邻（首尾相接）。
    返回：
      - neighbor_dist_CV：相邻距离的变异系数（越小越均匀/平滑）
      - dir_cosine_mean：相邻步进方向的平均余弦相似度（越接近1越平滑）
    """
    X = np.asarray(X_reduced, float)
    # 相邻距离
    step = np.roll(X, -1, axis=0) - X
    step_norm = np.linalg.norm(step, axis=1)
    mu = step_norm.mean()
    sd = step_norm.std(ddof=1) if len(step_norm) > 2 else 0.0
    neighbor_dist_CV = float(sd / (mu + 1e-12))

    # 相邻方向的连续性：cos(Δx_k, Δx_{k+1})
    dir1 = step[:-1]
    dir2 = step[1:]
    dir1 = np.vstack([dir1, step[-1:]])  # 环：最后一步与第一步也配对
    dir2 = np.vstack([dir2, step[:1]])

    def _safe_cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12: return 0.0
        return float(np.dot(a, b) / (na * nb))
    cosines = [ _safe_cos(a, b) for a, b in zip(dir1, dir2) ]
    dir_cosine_mean = float(np.mean(cosines))

    return neighbor_dist_CV, dir_cosine_mean

# ============== 3) 路径平滑性（轨迹一/二阶）=============

def metric_trajectory_smoothness(X_reduced: np.ndarray):
    """
    返回：
      - d1_var：一阶相邻步长的方差（越小越平滑）
      - curvature_energy：离散二阶差分能量（越小越平滑）
    """
    X = np.asarray(X_reduced, float)
    d1 = np.roll(X, -1, axis=0) - X
    d2 = np.roll(X, -1, axis=0) - 2 * X + np.roll(X, 1, axis=0)
    traj_d1_var = float(np.var(np.linalg.norm(d1, axis=1)))
    traj_curvature_energy = float(np.sum(np.linalg.norm(d2, axis=1) ** 2))
    return traj_d1_var, traj_curvature_energy

# ============== 4) 环形结构一致性（圆拟合 & Procrustes 到理想圆）=============

def _fit_circle_R2_2d(X2: np.ndarray) -> float:
    """
    最小二乘圆拟合（Kasa），返回 R^2（越大越接近圆）
    """
    x, y = X2[:, 0], X2[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    a, b1, c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc, yc = -a / 2.0, -b1 / 2.0
    r = np.sqrt((xc**2 + yc**2) - c)
    dist = np.sqrt((x - xc)**2 + (y - yc)**2)
    ss_res = np.sum((dist - r)**2)
    ss_tot = np.sum((dist - dist.mean())**2) + 1e-12
    R2 = 1.0 - ss_res / ss_tot
    return float(max(min(R2, 1.0), -np.inf))

def _procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    经典 Procrustes：去中心、单位化、最优正交对齐后残差范数。
    返回归一化残差（越小越相似）。
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc) + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc) + 1e-12)
    # 最优旋转
    U, _, Vt = np.linalg.svd(Xn.T @ Yn, full_matrices=False)
    R = U @ Vt
    X_aligned = Xn @ R
    resid = np.linalg.norm(X_aligned - Yn)
    return float(resid)

def metric_circularity_procrustes(X_reduced: np.ndarray):
    """
    返回：
      - circularity_R2：圆拟合优度（越大越像圆）
      - procrustes_to_circle：与理想等间隔圆的 Procrustes 残差（越小越像圆）
    """
    X = np.asarray(X_reduced, float)
    # 如为 3D，先 PCA->2D
    X2 = _pca_project_to_2d(X) if X.shape[1] == 3 else X.copy()

    # 圆拟合 R^2
    circ_R2 = _fit_circle_R2_2d(X2)

    # 与理想等间隔单位圆的 Procrustes 距离
    n = X2.shape[0]
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n,2)
    proc = _procrustes_distance(X2, circle)

    return  circ_R2,  proc

def compute_global_dist(RDM):
    n = RDM.shape[0]
    discriminability = np.sum(RDM) / (n * n - n)  # 1. 全局区分度（去对角线）
    return discriminability


def cal_view_represent(rdm, X_reduced, X_high, k = 3):

    dict_view_metrics = {}
    # 1) 保持度
    trustworthiness, continuity= metric_trustworthiness_continuity(X_reduced, X_high, k=k)
    # 2) 邻近平滑度
    neighbor_dist_CV, dir_cosine_mean = metric_neighbor_smoothness(X_reduced)
    # 3) 轨迹平滑性
    traj_d1_var, traj_curvature_energy= metric_trajectory_smoothness(X_reduced)
    # 4) 环形一致性
    circ_R2,  proc= metric_circularity_procrustes(X_reduced)
    # 4) 全局表征距离
    global_distance=compute_global_dist(rdm)

    dict_view_metrics['trustworthiness']=trustworthiness
    dict_view_metrics['continuity'] = continuity
    dict_view_metrics['neighbor_dist_CV'] = neighbor_dist_CV
    dict_view_metrics['dir_cosine_mean'] = dir_cosine_mean
    dict_view_metrics['traj_d1_var'] = traj_d1_var
    dict_view_metrics['traj_curvature_energy'] = traj_curvature_energy
    dict_view_metrics['circ_R2'] = circ_R2
    dict_view_metrics['proc'] = proc
    dict_view_metrics['global_distance'] = global_distance

    return dict_view_metrics

def plot_category_rates(spk_rate, label_fine, class_fine,ax, para, paraname):
    """
    spk_rate: (T x N)，T=时间点(4000)，N=刺激数(200)
    label_fine: 长度 N，类别标签（字符串）
    class_fine: 长度 8 的类别名称（字符串）
    """
    para=round(para,2)
    spk_rate = np.asarray(spk_rate)
    labels = np.asarray(label_fine)
    T, N = spk_rate.shape
    assert N == len(labels), "label_fine 长度应与刺激数一致"

    time = np.arange(T)

    for cname in class_fine:
        idx = np.where(labels == cname)[0]  # 属于该类别的刺激索引
        if idx.size == 0:
            continue

        X = spk_rate[:, idx]  # (T, n_c)
        mean_c = X.mean(axis=1)
        sem_c = X.std(axis=1, ddof=1) / np.sqrt(X.shape[1])  # SEM
        ax.plot(time, mean_c, label=cname, linewidth=1.5)
        ax.fill_between(time, mean_c - sem_c, mean_c + sem_c, alpha=0.25)
        ax.set_xlim([0,500])
    ax.set_ylabel('Firing rate (a.u.)')
    ax.set_title(paraname+': '+str(para))

def compute_sorted_metrics(neuron_dict, rtype, metric_name):
    dict_mertics={}
    for neu in neuron_dict.keys():
        dict_mertics[neu]=neuron_dict[neu][rtype]['metrics'][metric_name]
    #从低到高对字典进行排序
    sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
    #根据排序获取神经元的响应矩阵，也从低到高进行排序
    resp_class=[neuron_dict[neu][rtype]['resp'] for neu in sorted_dict.keys()]
    resp_class=np.array(resp_class)
    name_class=neuron_dict[neu][rtype]['name']
    label_fine=[i.split('_')[2] for i in name_class]
    class_fine = list(dict.fromkeys(label_fine))
    return sorted_dict, resp_class, label_fine, class_fine, name_class

def compute_sorted_metrics_view(neuron_dict, rtype, metric_name):
    dict_mertics={}
    for neu in neuron_dict.keys():
        dict_mertics[neu]=neuron_dict[neu][rtype]['metrics'][metric_name]
    #从低到高对字典进行排序
    sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
    #根据排序获取神经元的响应矩阵，也从低到高进行排序
    resp_view=[neuron_dict[neu][rtype]['resp'] for neu in sorted_dict.keys()]
    resp_view=np.array(resp_view)
    return sorted_dict, resp_view

def compute_sorted_metrics_img(neuron_dict, rtype, metric_name):
    dict_mertics={}
    for neu in neuron_dict.keys():
        dict_mertics[neu]=neuron_dict[neu]['fit_corr'][metric_name]
    #从低到高对字典进行排序
    sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
    #根据排序获取神经元的响应矩阵，也从低到高进行排序
    resp_class=[neuron_dict[neu][rtype]['resp'] for neu in sorted_dict.keys()]
    resp_class=np.array(resp_class)
    name_class=neuron_dict[neu][rtype]['name']
    label_fine=[i.split('_')[2] for i in name_class]
    class_fine = list(dict.fromkeys(label_fine))
    return sorted_dict, resp_class, label_fine, class_fine, name_class

def plot_class_represent(resp_class,label_fine, class_fine, metric_name,sorted_dict,n_neuron,dict_class,X_reduced_show):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import h5py
    n_class = 8
    samples_per_class = 25
    #可视化热力图响应矩阵################################################################################################################
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(4, 5,width_ratios=[1.7, 1, 1, 1, 1])

    #对每个神经元响应类别进行排序
    resp_sorted = np.zeros_like(resp_class)
    for i in range(n_class):
        start = i * samples_per_class
        end = (i + 1) * samples_per_class
        block = resp_class[:, start:end]   # (259, 25)
        # 对每一行（神经元）的25个值排序（从高到低）
        sorted_block = -np.sort(-block, axis=1)  # 先取负号，再排序，再取负号 => 从大到小
        resp_sorted[:, start:end] = sorted_block

    ax = fig.add_subplot(gs[0:2, 0])
    im = ax.imshow(resp_sorted, aspect='auto', interpolation='nearest')  # 去掉 vmin/vmax 则用全范围
    cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Response')
    # 类别分界线
    for i in range(1, n_class):
        ax.axvline(x=i * samples_per_class - 0.5, color='r', linestyle='--', linewidth=1)
    # 类别中心位置
    centers = [samples_per_class * (i + 0.5) - 0.5 for i in range(n_class)]
    ax.set_xticks(centers)
    ax.set_xticklabels(class_fine, rotation=45, ha='right')
    ax.set_ylabel('Neuron (sorted low→high, n=259)')
    ax.set_title(metric_name)
    ax.set_yticks([])   # 去掉刻度标签以更清爽（需要时可注释掉）

    #绘制参数分布################################################################################################################
    values = list(sorted_dict.values())
    ax = fig.add_subplot(gs[3, 0])   # 这里只有一个子图，用 [0, 0]
    # 直方图
    ax.hist(values, bins=20, color='skyblue', edgecolor='k', alpha=0.7)
    ax.set_xlabel(metric_name+' value')
    ax.set_ylabel('Count')

    #绘制神经元响应示例################################################################################################################
    nsample=4
    items = list(sorted_dict.items())
    indices = np.linspace(0, len(items)-1, nsample, dtype=int)
    neuron_sample=[items[i][0] for i in indices]
    rootpath=r'E:\Image_paper\Project\Acute_Image\results\neuron_response'
    for ii,neu in enumerate(neuron_sample):
        ax = fig.add_subplot(gs[ii, 1])
        raster_path=os.path.join(rootpath,neu.split('ch')[0][:-1],neu.split('ch')[0][:-1]+'_raster.mat')
        with h5py.File(raster_path, 'r') as f:
            raster_trial = np.array(f['raster_trial']).T
        raster_trial=np.mean(raster_trial[3000:3500,:,0:200,int(neu.split('ch')[1])-1],1)
        spk_rate = moving_avg_same(raster_trial, bin_size=50, step=1)
        spk_rate=spk_rate*1000
        plot_category_rates(spk_rate, label_fine, class_fine, ax, items[indices[ii]][1], metric_name)
        # 只在最后一个子图显示 x 轴
        if ii < nsample - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (s)')

    #随参数变化的性能变化################################################################################################################
    params_show=['fine_accuracy','Intra-class distance', 'Inter-class distance', 'Separability rate']

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    x = range(0, resp_class.shape[0] - n_neuron + 1)

    for i, param in enumerate(params_show):
        y = [dict_class[idx][param] for idx in x]
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(x, y, linewidth=1.2)
        # if i==0:
        #     y = [dict_class[idx]['coarse_accuracy'] for idx in x]
        #     ax.plot(x, y, linewidth=1.2)
        #     y = [dict_class[idx]['accuracy_face'] for idx in x]
        #     ax.plot(x, y, linewidth=1.2)
        #     y = [dict_class[idx]['accuracy_pigeon'] for idx in x]
        #     ax.plot(x, y, linewidth=1.2)
        if i < len(params_show) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Neuron index')
        ax.set_ylabel(param)
        ax.set_title(param, fontsize=10)

    # 绘制流形变化
    from sklearn.preprocessing import LabelEncoder

    le_fine = LabelEncoder().fit(label_fine)
    cmap_fine = plt.cm.get_cmap('tab20', len(le_fine.classes_))
    for i, X_reduced in enumerate(X_reduced_show):
        row = i % 4
        col_offset = 3 + (i // 4)  # 前 4 个放第 3 列，后 4 个放第 4 列
        ax = fig.add_subplot(gs[row, col_offset])

        # 使用 fine 标签着色
        y = le_fine.transform(label_fine)
        sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                        c=y, cmap=cmap_fine, s=10, linewidths=0)
        ax.axis('equal')
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.set_title(f'X{i + 1}', fontsize=8)
    plt.subplots_adjust(wspace=0.6, hspace=0.3, right=0.95, left=0.1, top=0.90, bottom=0.1)
    plt.savefig(os.path.join('results',"class", metric_name+".png"), format='png', dpi=300, bbox_inches='tight')


def plot_viewclass_represent(resp_class,label_fine, class_fine, metric_name,sorted_dict,n_neuron,dict_class,X_reduced_show):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import h5py
    n_class = 6
    samples_per_class = 18
    #可视化热力图响应矩阵################################################################################################################
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(4, 5,width_ratios=[1.7, 1, 1, 1, 1])

    #对每个神经元响应类别进行排序
    resp_sorted = np.zeros_like(resp_class)
    for i in range(n_class):
        start = i * samples_per_class
        end = (i + 1) * samples_per_class
        block = resp_class[:, start:end]   # (259, 25)
        # 对每一行（神经元）的25个值排序（从高到低）
        sorted_block = -np.sort(-block, axis=1)  # 先取负号，再排序，再取负号 => 从大到小
        resp_sorted[:, start:end] = sorted_block

    ax = fig.add_subplot(gs[0:2, 0])
    im = ax.imshow(resp_sorted, aspect='auto', interpolation='nearest')  # 去掉 vmin/vmax 则用全范围
    cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Response')
    # 类别分界线
    for i in range(1, n_class):
        ax.axvline(x=i * samples_per_class - 0.5, color='r', linestyle='--', linewidth=1)
    # 类别中心位置
    centers = [samples_per_class * (i + 0.5) - 0.5 for i in range(n_class)]
    ax.set_xticks(centers)
    ax.set_xticklabels(class_fine, rotation=45, ha='right')
    ax.set_ylabel('Neuron (sorted low→high, n=259)')
    ax.set_title(metric_name)
    ax.set_yticks([])   # 去掉刻度标签以更清爽（需要时可注释掉）

    #绘制参数分布################################################################################################################
    values = list(sorted_dict.values())
    ax = fig.add_subplot(gs[3, 0])   # 这里只有一个子图，用 [0, 0]
    # 直方图
    ax.hist(values, bins=20, color='skyblue', edgecolor='k', alpha=0.7)
    ax.set_xlabel(metric_name+' value')
    ax.set_ylabel('Count')

    #绘制神经元响应示例################################################################################################################
    nsample=4
    items = list(sorted_dict.items())
    indices = np.linspace(0, len(items)-1, nsample, dtype=int)
    neuron_sample=[items[i][0] for i in indices]
    rootpath=r'E:\Image_paper\Project\Acute_Image\results\neuron_response'
    for ii,neu in enumerate(neuron_sample):
        ax = fig.add_subplot(gs[ii, 1])
        raster_path=os.path.join(rootpath,neu.split('ch')[0][:-1],neu.split('ch')[0][:-1]+'_raster.mat')
        with h5py.File(raster_path, 'r') as f:
            raster_trial = np.array(f['raster_trial']).T
        raster_trial=np.mean(raster_trial[3000:3500,:,200:,int(neu.split('ch')[1])-1],1)
        spk_rate = moving_avg_same(raster_trial, bin_size=50, step=1)
        spk_rate=spk_rate*1000
        plot_category_rates(spk_rate, label_fine, class_fine, ax, items[indices[ii]][1], metric_name)
        # 只在最后一个子图显示 x 轴
        if ii < nsample - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (s)')

    #随参数变化的性能变化################################################################################################################
    params_show=['fine_accuracy','Intra-class distance', 'Inter-class distance', 'Separability rate']

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    x = range(0, resp_class.shape[0] - n_neuron + 1)

    for i, param in enumerate(params_show):
        y = [dict_class[idx][param] for idx in x]
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(x, y, linewidth=1.2)
        # if i==0:
        #     y = [dict_class[idx]['coarse_accuracy'] for idx in x]
        #     ax.plot(x, y, linewidth=1.2)
        #     y = [dict_class[idx]['accuracy_face'] for idx in x]
        #     ax.plot(x, y, linewidth=1.2)
        #     y = [dict_class[idx]['accuracy_pigeon'] for idx in x]
        #     ax.plot(x, y, linewidth=1.2)
        if i < len(params_show) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Neuron index')
        ax.set_ylabel(param)
        ax.set_title(param, fontsize=10)

    # 绘制流形变化
    from sklearn.preprocessing import LabelEncoder

    le_fine = LabelEncoder().fit(label_fine)
    cmap_fine = plt.cm.get_cmap('tab20', len(le_fine.classes_))
    for i, X_reduced in enumerate(X_reduced_show):
        row = i % 4
        col_offset = 3 + (i // 4)  # 前 4 个放第 3 列，后 4 个放第 4 列
        ax = fig.add_subplot(gs[row, col_offset])

        # 使用 fine 标签着色
        y = le_fine.transform(label_fine)
        sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                        c=y, cmap=cmap_fine, s=10, linewidths=0)
        ax.axis('equal')
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.set_title(f'X{i + 1}', fontsize=8)
    plt.subplots_adjust(wspace=0.6, hspace=0.3, right=0.95, left=0.1, top=0.90, bottom=0.1)
    plt.savefig(os.path.join('results','viewclass', metric_name+".png"), format='png', dpi=300, bbox_inches='tight')

def plot_view_represent(resp_view,metric_name,sorted_dict, n_neuron, dict_view, X_reduced_show):
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(6, 5,width_ratios=[1.5, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[0:2, 0])
    im = ax.imshow(resp_view, aspect='auto', interpolation='nearest')  # 去掉 vmin/vmax 则用全范围
    cbar = fig.colorbar(im, ax=ax)
    ax.set_ylabel('Neuron (sorted low→high, n=259)')
    ax.set_title(metric_name)
    ax.set_yticks([])   # 去掉刻度标签以更清爽（需要时可注释掉）

    #绘制参数分布################################################################################################################
    values = list(sorted_dict.values())
    ax = fig.add_subplot(gs[3, 0])   # 这里只有一个子图，用 [0, 0]
    # 直方图
    ax.hist(values, bins=20, color='skyblue', edgecolor='k', alpha=0.7)
    ax.set_xlabel(metric_name+' value')
    ax.set_ylabel('Count')

    #绘制神经元响应示例################################################################################################################
    nsample=4
    indices = np.linspace(0, resp_view.shape[0]-1, nsample, dtype=int)

    for ii, neu in enumerate(indices):
        ax = fig.add_subplot(gs[ii, 1])   # 每个神经元一个子图
        tuning = resp_view[neu, :]        # (1 × n_view)，该神经元的视角响应
        ax.plot(tuning, marker='o', linewidth=1.2)
        ax.set_title(metric_name+': '+str(round(list(sorted_dict.values())[neu],2)), fontsize=8)
        ax.set_ylabel('Resp')
        # 只在最后一个子图显示 x 轴
        if ii < nsample - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('View index')

    #随参数变化的性能变化################################################################################################################

    x = range(0, resp_view.shape[0] - n_neuron + 1)
    metrics_plan = [
        ('trustworthiness',        'Trustworthiness'),
        ('continuity',             'Continuity'),
        ('neighbor_dist_CV',       'Neighbor Dist CV'),
        ('dir_cosine_mean',        'Directional Cosine'),
        ('traj_d1_var',            'Trajectory d1 Var'),
        ('traj_curvature_energy',  'Curvature Energy'),
        ('circ_R2',                'Circular $R^2$'),
        ('proc',                   'Procrustes Dist'),
        ('global_distance',        'Global Distance'),
    ]
    for i, (k, ylabel) in enumerate(metrics_plan):
        row = i // 3   # 每行 3 个
        col = 2 + (i % 3)  # 从列 2 开始
        ax = fig.add_subplot(gs[row, col])
        y = [dict_view[idx][k] for idx in x]
        ax.plot(x, y, linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=8)

        # 只有最后一行显示 x 轴标签
        if row < (len(metrics_plan) - 1) // 3:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Start index (nstart)')


    # 断言有 8 份
    X_list = X_reduced_show
    for i, X_reduced in enumerate(X_list):
        row = i % 4
        col_offset = 4 + (i // 4)  # 前 4 个放第 3 列，后 4 个放第 4 列
        ax = fig.add_subplot(gs[col_offset, row])
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1],  s=10, linewidths=0)
        ax.plot(
            np.append(X_reduced[:, 0], X_reduced[0, 0]),
            np.append(X_reduced[:, 1], X_reduced[0, 1]),
            '-o', color='blue', markersize=2, linewidth=1)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'X{i + 1}', fontsize=8)

    plt.subplots_adjust(wspace=0.6, hspace=0.5, right=0.95, left=0.1, top=0.95, bottom=0.05)
    plt.savefig(os.path.join('results','view',metric_name+".png"), format='png', dpi=300, bbox_inches='tight')