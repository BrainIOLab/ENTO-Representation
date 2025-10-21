import os
import pickle
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import h5py
from matplotlib import gridspec


neuron_path = r'E:\Image_paper\Project\dict_view.pkl'
with open(neuron_path, "rb") as f:
    dict_view = pickle.load(f)  # 获取全部神经元响应

obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']


fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(7.3, 2.8))
gs = gridspec.GridSpec(1, 6)
for iindex,obj in enumerate(obj_list):
    neuron_ids   = list(dict_view.keys())
    all_resp     = []   # [n_neuron, 18]
    preferred_id = []   # 每个神经元的调谐中心(0~17)
    for neu in neuron_ids:
        resp = np.array(dict_view[neu][obj]['resp'])
        all_resp.append(resp)
        preferred_id.append(np.argmax(resp))   # 响应最大值对应的视角索引
    all_resp = np.array(all_resp)
    preferred_id = np.array(preferred_id)
    sort_idx = np.argsort(preferred_id)
    resp_sorted = all_resp[sort_idx]

    ax = fig.add_subplot(gs[0, iindex])
    im = ax.imshow(resp_sorted, aspect='auto', interpolation='none',cmap='jet')  # 去掉 vmin/vmax 则用全范围
    ax.set_yticks([])  # 去掉刻度标签以更清爽（需要时可注释掉）
    ax.set_xticks([])  # 去掉刻度标签以更清爽（需要时可注释掉）
    if iindex==0:
        ax.set_ylabel('Neurons (preferred view, n=253)',fontsize=10, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(wspace=0.25, hspace=0.05, right=0.88, left=0.12, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure1', 'f1_view.eps'), dpi=600, format='eps')


#绘制视角调谐################################################################################################################

obj='view_electricguitar'
dict_mertics={}
for neu in dict_view.keys():
    dict_mertics[neu]=dict_view[neu][obj]['metrics']['Modulation depth']
# 从低到高对字典进行排序
dict_sort = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))


low_MD_neuron='ni_exp15_ch5'
High_MD_neuron='ento_exp18_ch23'
fig = plt.figure(figsize=(2.1, 3.2))
gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[0, 0])
resp=dict_view[low_MD_neuron][obj]["resp"]
angles = np.linspace(0, 360, len(resp), endpoint=False)
ax.plot(angles, resp, '-o', markersize=4, linewidth=linewidth_plot, color='steelblue',label='Modulation depth=0.14')
resp=dict_view[High_MD_neuron][obj]["resp"]
ax.plot(angles, resp, '-^', markersize=4, linewidth=linewidth_plot, color='darkorange',label='Modulation depth=0.57')
# ax.set_xlabel('View angle (deg)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_xticklabels([])
ax.set_ylabel('Norm. response ', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
ax.set_ylim([0.2,1])
ax.set_xticks([0, 90, 180, 270, 360])
# ax.set_xticklabels(['0', '90', '180', '270', '360'],
#                    fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.legend(loc='upper center', frameon=False, fontsize=fontsize_, handlelength=1.8,bbox_to_anchor=(0.5, 1.6))


obj='view_electricguitar'
dict_mertics={}
for neu in dict_view.keys():
    dict_mertics[neu]=dict_view[neu][obj]['metrics']['Neighbor correlation']
# 从低到高对字典进行排序
dict_sort = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))

low_MD_neuron='ni_exp1_ch14'
High_MD_neuron='ento_exp9_ch5'
# fig = plt.figure(figsize=(1.5, 3))
# gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[1, 0])
resp=dict_view[low_MD_neuron][obj]["resp"]
ax.plot(angles, resp, '-o', markersize=4, linewidth=linewidth_plot, color='steelblue',label='Neighbor correlation=0.29')
resp=dict_view[High_MD_neuron][obj]["resp"]
ax.plot(angles, resp, '-^', markersize=4, linewidth=linewidth_plot, color='darkorange',label='Neighbor correlation=0.89')
# ax.set_xlabel('View angle (deg)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_xlabel('View angle (deg)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Norm. response ', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
ax.set_ylim([0.2,1])
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xticklabels(['0', '90', '180', '270', '360'],
                   fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.legend(loc='upper center', frameon=False, fontsize=fontsize_, handlelength=1.8,bbox_to_anchor=(0.5, 1.6))

plt.subplots_adjust(wspace=0.4, hspace=0.7, right=0.85, left=0.25, top=0.85, bottom=0.13)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure1', 'f1_viewtuning.eps'), dpi=600, format='eps')