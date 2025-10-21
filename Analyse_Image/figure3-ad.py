import os
import pickle
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score
import matplotlib.pyplot as plt
from matplotlib import gridspec

#计算整体神经元集群的解码准确率和混淆矩阵################################################################################
dict_accuracy={}

neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应

resp_neuron=[dict_class[neu]['class']['resp'] for neu in dict_class.keys()]
resp_neuron=np.transpose(np.array(resp_neuron))
temp=dict_class['ento_exp15_ch20']['class']['name']
labels_neuron=[i.split('_')[1] for i in temp]
cv = StratifiedKFold(n_splits=5)  # 创建交叉验证
model = SVC(kernel='rbf', decision_function_shape='ovr')
y_pred = cross_val_predict(model, resp_neuron, labels_neuron, cv=cv)
classes =  list(dict.fromkeys(labels_neuron))
cm=confusion_matrix(labels_neuron, y_pred,labels=classes)  # 混淆矩阵

neuron_accuracy = accuracy_score(labels_neuron, y_pred)  # 准确率
dict_accuracy['Neurons']=neuron_accuracy

#绘制cm混淆矩阵###############################################################################################################
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

import seaborn as sns
fig = plt.figure(figsize=(3.2, 3.2))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
hm=sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=classes,
            yticklabels=classes, ax=ax,
            annot_kws=dict(size=fontsize_, family=fontname_, weight=fontweight_))
cbar = hm.collections[0].colorbar          # <— 关键
cbar.ax.tick_params(labelsize=fontsize_)
for t in cbar.ax.get_yticklabels():
    t.set_fontname(fontname_)
    t.set_fontweight(fontweight_)
ax.set_xlabel('Predicted Labels',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)  # 替代plt.xlabel()
ax.set_ylabel('True Labels',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)  # 替代plt.ylabel()
# 轴刻度字体
ax.set_xticklabels(classes, rotation=90, ha='right',
                   fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_yticklabels(classes, rotation=0,
                   fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(wspace=0.5, hspace=1.1, right=0.95, left=0.3, top=0.9, bottom=0.4)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3', 'figure3_ad2.eps'), dpi=600, format='eps')

#计算图像特征的解码准确率################################################################################
feature_path = r'E:\Image_paper\Project\dict_img_feature.pkl'
with open(feature_path, "rb") as f:
    img_dict = pickle.load(f)  # 图像特征字典，每张图像是一个键
feat_list=['Color','Shape','Texture','V1-like','V2-like',
           'Alexnet Conv1','Alexnet Conv2','Alexnet Conv3','Alexnet Conv4','Alexnet Conv5']

for ft in feat_list:
    feat_resp=[]
    labels=[]
    for img in img_dict.keys():
        if img_dict[img]['task']=='class':
            feat_resp.append(img_dict[img][ft])
            labels.append(img.split('_')[1])
    feat_resp=np.array(feat_resp)
    cv = StratifiedKFold(n_splits=5)  # 创建交叉验证
    model = SVC(kernel='rbf', decision_function_shape='ovr')
    y_pred = cross_val_predict(model, feat_resp, labels, cv=cv)
    fine_accuracy = accuracy_score(labels, y_pred)  # 准确率
    dict_accuracy[ft]=fine_accuracy

#绘制图像#######################################################################
color_map = {
    'Neurons':          '#374E55',
    'Color':            '#9E9E9E',  # 红橙
    'Shape':            '#31A354',  # 绿青
    'Texture':          '#762A83',  # 肉粉
    'V1-like':          '#A1D99B',  # 淡青绿
    'V2-like':          '#2171B5',  # 洋红
    'Alexnet Conv1':    '#EF3B2C',  # 青蓝
    'Alexnet Conv2':    '#EF3B2C',  # 靛蓝
    'Alexnet Conv3':    '#EF3B2C',  # 褐色
    'Alexnet Conv4':    '#EF3B2C',  # 灰蓝
    'Alexnet Conv5':    '#EF3B2C',  # 墨青
}

fig = plt.figure(figsize=(3.5, 2.8))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])

names = [k for k in dict_accuracy]
vals  = [dict_accuracy[k] for k in names]
colors = [color_map.get(k, '0.65') for k in names]
bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='k', linewidth=0.6)
# chance level（8 类 = 0.125）
ax.axhline(1/8, ls='--', lw=0.8, color='k')
ax.text(len(names)-0.2, 1/8 + 0.01, 'chance\n0.125', ha='left', va='bottom',
        fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Decoding accuracy',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.2,0.4,0.6,0.8,1.0])
# ax.set_xticks(range(len(names)))
# ax.set_xticklabels(names, rotation=45, ha='right',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

ax.set_xticks([])

# 在柱子中部写标签，在柱顶写数值
for bar, label, v in zip(bars, names, vals):
    x = bar.get_x() + bar.get_width() / 2.0
    # 柱内标签（竖排）
    ax.text(x, bar.get_height() / 2.0, label,
            rotation=90, ha='center', va='center',
            color='white',
            fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,
            clip_on=True)
    # 柱顶数值
    ax.text(x, bar.get_height() + 0.02, f'{v:.2f}',
            ha='center', va='bottom',
            fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
# 样式
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

plt.subplots_adjust(wspace=0.4, hspace=1, right=0.90, left=0.15, top=0.85, bottom=0.3)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3', 'figure3_ad1.eps'), dpi=600, format='eps')






