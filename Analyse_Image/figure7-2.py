import numpy as np
import os
from tools_view import *
from tools_plot import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle

#设置基础参数
class view_config:
    def __init__(self):
        self.num_neurons = 5  # 设置神经元数量
        self.num_angles = 360  # 设置角度的数量
        self.angles = np.linspace(0, 360, self.num_angles, endpoint=False)  # 计算角度
        self.rdm_view_target = build_view_rdm(self.angles)      # 设定视角表征的目标RDM
        self.noise_type = 'none'        # 设定噪声类型， "none", "additive", "multiplicative", "poisson"
        self.noise_std = 0.2            # 设定噪声的程度

fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

#绘制调谐,mu############################################################################################################################
fig = plt.figure(figsize=(1.4, 1.5))
gs = gridspec.GridSpec(2, 1)                                     # 1行4列，第4列用于颜色条
sigma_range=[10,60]
for index,sigma in enumerate(sigma_range):
    config=view_config()                                         # 设置基础参数
    mu_view = np.arange(0, 360, 360 / config.num_neurons)        # mu均匀分布
    # mu_view = generate_gaussian_mu(20)                         # mu均匀分布
    amp_view = np.ones(config.num_neurons) * 1                   # 幅值设置为1
    sigma_view = np.ones(config.num_neurons) * sigma             # 调谐宽度为50
    params_view = np.concatenate([mu_view, amp_view, sigma_view])
    resp_view, rdm_view, rdm_view_norm, embedded_view_2d=cal_view_represent(params_view, config, 'isomap')      #计算响应和表征距离矩阵
    ax = fig.add_subplot(gs[index, 0])
    for i in range(config.num_neurons):
        if i==2:
            ax.plot(config.angles, resp_view[i, :], color='red', linewidth=linewidth_plot, )
        else:
            ax.plot(config.angles, resp_view[i, :], color='gray', linewidth=linewidth_plot, )
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)  # 去除上边框
    ax.spines['right'].set_visible(False)  # 去除右边框
    ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
    ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
    ax.tick_params(axis='both', width=linewidth_ax)  # 设置 X 和 Y 轴的刻度线宽度为 2
    plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

    ax.set_xticks([0,360])  # 设置刻度位置
    if index ==0:
        ax.set_xticklabels([])  # 不显示x轴刻度标签
    else:
        ax.set_xticklabels([0,360])  # 设置刻度标签，确保 0 被显示
        ax.set_xlabel('View', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-2)
    ax.set_ylabel('Response', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(wspace=0.05, hspace=0.8, right=0.9, left=0.31, top=0.95, bottom=0.2)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-5.eps'), dpi=600, format='eps')



#绘制流形嵌入,sigma#########################################################################################################
config=view_config()   #设置基础参数
fig = plt.figure(figsize=(3.0, 1.4))
gs = gridspec.GridSpec(2, 4,height_ratios=[1, 1])
sigma_plot = [10, 20, 30, 60]
for sindex, sigma in enumerate(sigma_plot):
    # 设定调谐参数
    config = view_config()  # 设置基础参数
    mu_view = np.arange(0, 360, 360 / config.num_neurons)  # mu均匀分布
    amp_view = np.ones(config.num_neurons) * 1  # 幅值设置为1
    sigma_view = np.ones(config.num_neurons) * sigma  # 调谐宽度为50
    params_view = np.concatenate([mu_view, amp_view, sigma_view])

    resp_view, rdm_view, rdm_view_norm, embedded_view_2d = cal_view_represent(params_view, config, 'isomap')  # 计算响应和表征距离矩阵
    # print('isomap', np.min(rdm_view),np.max(rdm_view))
    ax = fig.add_subplot(gs[sindex // len(sigma_plot) + sindex // len(sigma_plot), sindex % len(sigma_plot)])
    plot_embedded_2d(embedded_view_2d, ax)
    ax.set_ylim([-1.6, 1.6])
    ax.set_xlim([-1.6, 1.6])
    # if sindex==3:
    #     ax.set_xticks([-1., 1.])
    #     ax.spines['bottom'].set_visible(True)
    #     ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
    #     ax.tick_params(axis='both', width=linewidth_ax)  # 设置 X 和 Y 轴的刻度线宽度为 2
    #     plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    #     plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax = fig.add_subplot(gs[sindex // len(sigma_plot) + sindex // len(sigma_plot) + 1, sindex % len(sigma_plot)])
    plot_rdm_no_box(ax, rdm_view)

plt.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85, left=0.05, top=0.95, bottom=0.10)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-6.png'), dpi=600, format='png')
