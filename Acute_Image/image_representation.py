#解码，选取一个特征或者多个特征组合进行解码
import pickle
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import tqdm
import seaborn as sns
import os
from sklearn.metrics import silhouette_score

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

def reducedim_analyse(img_dict,feature_name,img_size,figure_size):

    feature_matrix = return_feature_value(img_dict, feature_name)  # 返回该特征的所有值
    labels, _ = return_category_value(img_dict)  # 返回特征标签
    cmap = plt.cm.get_cmap('tab20', 8)  # 16类

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2,2)
    result = np.zeros((720 * 2, 720 * 2, 3), dtype=np.uint8)

    reducedim_method_list=['PCA','tSNE','MDS','Isomap']
    for p_index,reducedim_method in enumerate(reducedim_method_list):

        if reducedim_method == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)  # 降维到2维
            X_reduced = pca.fit_transform(feature_matrix)
        if reducedim_method == 'tSNE':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)  # 降维到2维
            X_reduced = tsne.fit_transform(feature_matrix)
        if reducedim_method == 'MDS':
            from sklearn.manifold import MDS
            mds = MDS(n_components=2)  # 降维到2维
            X_reduced = mds.fit_transform(feature_matrix)
        if reducedim_method == 'Isomap':
            from sklearn.manifold import Isomap
            isomap = Isomap(n_components=2)  # 降维到2维
            X_reduced = isomap.fit_transform(feature_matrix)

        fisher_ratio = cal_fisher_ratio(X_reduced, labels)
        silhouette_avg = silhouette_score(X_reduced, labels, metric='euclidean')
        fisher_ratio=round(fisher_ratio,2)
        silhouette_avg=round(silhouette_avg, 2)
        print("Fisher判别比:", round(fisher_ratio,2),"轮廓系数:", round(silhouette_avg,2))

        # 绘制散点图
        ax1 = fig.add_subplot(gs[p_index // 2, p_index % 2])
        scatter = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap=cmap, s=50, edgecolors='k')
        ax1.axis('equal')
        cbar = fig.colorbar(scatter, ax=ax1, label='Label')  # 注意这里需要传入fig和ax1
        ax1.set_title('{}_{}_{:.2f}_{:.2f}'.format(feature_name,reducedim_method,fisher_ratio,silhouette_avg))
        ax1.set_xlabel('dim 1')
        ax1.set_ylabel('dim 2')
        # plt.show()

        # 绘制图像
        img_list = list(img_dict.keys())
        index = np.argmax(np.max(X_reduced, 0) - np.min(X_reduced, 0))  # 建立映射
        min_old, max_old = np.min(X_reduced, 0)[index], np.max(X_reduced, 0)[index]
        min_new, max_new = img_size / 2, figure_size - img_size / 2
        mapped_data = np.interp(X_reduced, (min_old, max_old), (min_new, max_new))  # 转换为图像尺寸空间
        image_bg = np.ones((figure_size, figure_size, 3), dtype=np.uint8) * 255  # 创建白色背景
        for i in tqdm.tqdm(range(len(img_dict))):
            imgpath = img_dict[img_list[i]]['imgpath']
            # imgpath = os.path.join(*imgpath[4:])
            # imgpath = os.path.join(r'image\img_rgb', imgpath)
            image = cv2.imread(imgpath)
            imgpath_mask = imgpath.replace('img_rgb', 'img_mask')
            img_mask = cv2.imread(imgpath_mask, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (img_size, img_size))
            resized_image_mask = cv2.resize(img_mask, (img_size, img_size))
            # resized_image_mask[resized_image_mask <= 200] = 0
            # resized_image_mask[resized_image_mask > 200] = 1
            resized_image_mask[resized_image_mask>0] = 1
            resized_image_mask=1-resized_image_mask
            mask_expanded = np.stack([resized_image_mask] * 3, axis=-1)
            image_fg = resized_image * mask_expanded  # 将图像的前景抠出来，背景为0
            mapped_y, mapped_x = mapped_data[i, :]  # 获取降维后的坐标
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
    plt.savefig(os.path.join('results','image_features','image_class','{}_{}_scatter.png'.format(feature_name,'reducedim')), dpi=600,format='png')
    cv2.imwrite(os.path.join('results','image_features', 'image_class','{}_{}_image.png'.format(feature_name,'reducedim')),result)


def show_RSM(repre_dict,feature):
    class_labels = ['butterfly', 'elephant',  'faces',  'pigeon',
                     'backpack', 'beermug',  'cowboyhat','electricguitar']
    # 可视化相关性矩阵
    correlation_matrix=repre_dict[feature]['RSM']
    plt.figure(figsize=(5, 4))
    sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
    plt.xticks(ticks=list(range(12,200,25)), labels=class_labels, rotation=45, ha='right')
    plt.yticks(ticks=list(range(12,200,25)), labels=class_labels, rotation=0)
    plt.title(feature)
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.95, left=0.25, top=0.90, bottom=0.25)
    plt.savefig(os.path.join('results', 'image_features', 'rsm_{}.png'.format(feature)), dpi=600,
                format='png')
    # plt.show()

def RSM_analyse(imgclass_dict,repre_dict):
    feature_names=get_feature_names(imgclass_dict)
    for feature_name in tqdm.tqdm(feature_names):
        if max(imgclass_dict[list(imgclass_dict.keys())[0]][feature_name].shape)>1:                   #只有多维度特征计算RSM
            feature_matrix=return_feature_value(imgclass_dict, feature_name)                     #返回该特征的所有值
            correlation_matrix = np.corrcoef(np.transpose(feature_matrix), rowvar=False)    #计算皮尔逊相关性矩阵
            if correlation_matrix.shape==(len(imgclass_dict),len(imgclass_dict)):
                repre_dict[feature_name]['RSM']=correlation_matrix
            else:
                raise ValueError("特征维度不匹配！")
    return repre_dict

def show_accuracy(repre_dict,model_name):
    value = []
    for key in list(repre_dict.keys()):
        value.append(repre_dict[key][model_name]['accuracy'])
    value = np.array(value)
    plt.figure(figsize=(18, 6))
    plt.bar(list(repre_dict.keys()),value, color='skyblue')
    plt.xticks(rotation=90)  # 设置 x 轴标签的旋转角度为 45 度
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig(os.path.join('results','image_features', 'decode_accuracy_{}.png'.format(model_name)), dpi=300, format='png')
    # plt.show()


def decode_analyse(imgclass_dict,repre_dict):
    feature_names=get_feature_names(imgclass_dict)                                       #读取所有特征名称
    labels_numeric,category_dict=return_category_value(imgclass_dict)                    #读取所有图像的标签，
    cv = StratifiedKFold(n_splits=5)  # 创建交叉验证
    model_names=['linearSVM']
    for feature_name in feature_names:
        repre_dict[feature_name]={}
        Xdata=return_feature_value(imgclass_dict, feature_name)                              #读取一个特征的所有图像
        # from sklearn.preprocessing import StandardScaler
        # # 初始化 StandardScaler
        # scaler = StandardScaler()
        # # 对数据进行标准化
        # Xdata = scaler.fit_transform(Xdata)
        #模型训练和预测
        for model_name in model_names:
            repre_dict[feature_name]['decode_' + model_name]={}
            if model_name=='linearSVM':
                model = SVC(kernel='linear')
            if model_name=='LDA':
                model = LinearDiscriminantAnalysis()
            if model_name=='BP':
                model = MLPClassifier(hidden_layer_sizes=(64,), solver='sgd', max_iter=100)
            y_pred = cross_val_predict(model, Xdata, labels_numeric, cv=cv)                 #使用交叉验证生成预测结果
            #计算性能指标
            cm = confusion_matrix(labels_numeric, y_pred)                                   #混淆矩阵
            accuracy = accuracy_score(labels_numeric, y_pred)                               #准确率
            precision = precision_score(labels_numeric, y_pred, average='macro')            #宏平均精确率
            recall = recall_score(labels_numeric, y_pred, average='macro')                  #宏平均召回率
            f1 = f1_score(labels_numeric, y_pred, average='macro')                          #宏平均F1分数

            print(feature_name,model_name,'accuracy:',accuracy,'f1:',f1)
            repre_dict[feature_name]['decode_'+model_name]['cm'] = cm
            repre_dict[feature_name]['decode_'+model_name]['accuracy'] = accuracy
            repre_dict[feature_name]['decode_'+model_name]['precision'] = precision
            repre_dict[feature_name]['decode_' + model_name]['recall'] = recall
            repre_dict[feature_name]['decode_' + model_name]['f1'] = f1
    return repre_dict

def show_confusionmatrix(repre_dict,feature,model_name):
    # 可视化混淆矩阵
    category_dict = {'butterfly': 0, 'elephant': 1, 'faces': 2,  'pigeon': 3,
                     'backpack': 4, 'beermug': 5,  'cowboyhat': 6, 'electricguitar': 7}
    cm=repre_dict[feature][model_name]['cm']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(category_dict.keys()))
    disp.plot(cmap=plt.cm.Blues)  # 可以使用不同的 colormap，如 plt.cm.Blues, plt.cm.Reds 等
    plt.xticks(rotation=45)  # 设置 x 轴标签的旋转角度为 45 度
    plt.title(feature+'_'+model_name+'_'+str(round(repre_dict[feature][model_name]['accuracy'],2)))
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.95, left=0.05, top=0.90, bottom=0.25)
    plt.savefig(os.path.join('results', 'image_features', 'decode_confusionmatrix_{}_{}.png'.format(model_name,feature)), dpi=600,
                format='png')
    # plt.show()

def return_category_value(img_dict):
    # 返回所有图像的类别，n_image×1
    category_dict = {'butterfly': 0, 'elephant': 1, 'faces': 2,  'pigeon': 3,
                     'backpack': 4, 'beermug': 5,  'cowboyhat': 6, 'electricguitar': 7}
    numeric_labels = []
    for key in list(img_dict.keys()):
        numeric_labels.append(category_dict[key.split('_')[2]])
    numeric_labels = np.array(numeric_labels)
    return numeric_labels, category_dict

def return_feature_value(img_dict, feature_name):
    # 返回所有图像的某个特征，n_image×n_dinm
    value = []
    for key in list(img_dict.keys()):
        value.append(img_dict[key][feature_name])
    value = np.array(value)
    return value

def get_feature_names(img_dict):
    # 返回字典里所有特征的名称
    feature_list=['color','shape','texture','v1like','v2like','hmax','alexnet','vgg16',
                   'resnet50','inceptionv3','mobilenetv2','densenet121','inceptionv3','cst']
    key = list(img_dict.keys())[0]
    feature_names = []
    for feature_key, feature_value in img_dict[key].items():
        if feature_key.split('_')[0] in feature_list and feature_key.split('_')[-1] != 'names':
            feature_names.append(feature_key)
    return feature_names

if __name__ == "__main__":
    # 读取字典数据
    with open(r"image/img_features/img_dict_pca.pkl", "rb") as f:
        img_dict = pickle.load(f)

    #获取图像类别字典
    imgclass_dict= {}
    for i in list(img_dict.keys()):
        if 'class' in i:
            imgclass_dict[i]=img_dict[i]

    #获取图像视角字典
    imgview_dict= {}
    for i in list(img_dict.keys()):
        if 'view' in i:
            imgview_dict[i]=img_dict[i]

###########图像类别########################################################################################################
    #解码，图像类别
    repre_dict={}                                                                   # 存放结果
    repre_dict=decode_analyse(imgclass_dict, repre_dict)                            # 特征解码
    #
    with open(r'image/img_features/repre_dict.pkl', 'wb') as file:
        pickle.dump(repre_dict, file)

    show_confusionmatrix(repre_dict,'v1like_channel_r','decode_linearSVM')          # 可视化混淆矩阵
    show_accuracy(repre_dict, 'decode_linearSVM')

    #表征相似性矩阵
    repre_dict=RSM_analyse(imgclass_dict, repre_dict)                                    # 计算表征相似性矩阵（RSM）
    show_RSM(repre_dict,'color_features_pca')                                      # 可视化RSM
    show_RSM(repre_dict, 'shape_features_pca')
    show_RSM(repre_dict, 'texture_features_pca')
    show_RSM(repre_dict, 'cst_features')
    show_RSM(repre_dict, 'alexnet_features.2_pca')
    show_RSM(repre_dict, 'alexnet_features.12_pca')

    # #降维，并可视化
    reducedim_method='tSNE'                                                         # 选择降维方法,PCA,tSNE,MDS,Isomap
    img_size=44                                                                     # 图像中单个物体图像的大小
    figure_size=720                                                                 # 图像的大小
    reducedim_analyse(imgclass_dict,  'color_features', img_size, figure_size)
    reducedim_analyse(imgclass_dict,  'shape_features', img_size, figure_size)
    reducedim_analyse(imgclass_dict,  'texture_features', img_size, figure_size)
    reducedim_analyse(imgclass_dict,  'cst_features', img_size, figure_size)
    reducedim_analyse(imgclass_dict,  'alexnet_features.2', img_size, figure_size)
    reducedim_analyse(imgclass_dict,  'alexnet_features.12', img_size, figure_size)

    ###########图像视角########################################################################################################
    reducedim_method='tSNE'                                                         # 选择降维方法,PCA,tSNE,MDS,Isomap
    img_size=44                                                                     # 图像中单个物体图像的大小
    figure_size=720
    reducedim_analyse(imgview_dict,  'color_features', img_size, figure_size)
    reducedim_analyse(imgview_dict,  'shape_features', img_size, figure_size)
    reducedim_analyse(imgview_dict,  'texture_features', img_size, figure_size)
    reducedim_analyse(imgview_dict,  'cst_features', img_size, figure_size)
    reducedim_analyse(imgview_dict,  'alexnet_features.2', img_size, figure_size)
    reducedim_analyse(imgview_dict,  'alexnet_features.12', img_size, figure_size)






