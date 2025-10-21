import numpy as np
import os
from tools_joint import *
import pickle

class joint_config:
    def __init__(self):
        self.num_neurons = 20  # 设置神经元数量
        self.angles = np.linspace(0, 360, 36, endpoint=False)  # 设置角度间隔，360个
        self.view_target_rdm = build_view_rdm(n_angles=36)  # 设定视角表征的目标RDM
        self.noise_type = 'none'  # 设定噪声类型， "none", "additive", "multiplicative", "poisson"
        self.noise_std = 0.2  # 设定噪声的程度

        self.objects_per_category = len(self.angles)
        self.num_categories = 10
        self.num_samples = self.num_categories * self.objects_per_category
        self.labels = np.repeat(np.arange(self.num_categories), self.objects_per_category)
        self.class_target_rdm = build_class_rdm(self.num_samples,self.num_categories,self.labels)  # 设定视角表征的目标RDM
        self.method='isomap'


#类别和视角################################################################################################################
config=joint_config()
sigma_class_range = np.linspace(0.1, 4, 40)
sigma_view_range = np.linspace(10, 140, 40)
dict_sigma={}
for sigma_class_ in sigma_class_range:
    for sigma_view_ in sigma_view_range:
        print(sigma_class_,sigma_view_)

        #计算视角响应
        mu_view = np.arange(0, 360, 360 / config.num_neurons)  # mu均匀分布
        amp_view = np.ones(config.num_neurons) * 1  # 幅值设置为1
        sigma_view = np.ones(config.num_neurons) * sigma_view_  # 调谐宽度为50
        params_view = np.concatenate([mu_view, amp_view, sigma_view])
        resp_view = circular_gaussian(config.angles, mu_view, sigma_view, amp_view)  # shape: (num_neurons, 36)
        #计算类别响应
        mu_class = np.linspace(0, config.num_categories - 1, config.num_neurons)
        amp_class = np.ones(config.num_neurons) * 1  # 幅值设置为1
        sigma_class = np.ones(config.num_neurons) * sigma_class_  # 调谐宽度为1
        params_class = np.concatenate([mu_class, amp_class, sigma_class])
        resp_class=cal_class_response(params_class, config)
        #计算联合调谐的响应矩阵，计算表征距离矩阵
        resp_joint=combine_view_class_response(resp_class, resp_view, mode='additive')    #(n_neurons, n_categories, n_angles)
        resp_joint_reshape = resp_joint.reshape(resp_joint.shape[0], -1)  # shape: (neurons, samples)
        rdm_joint, rdm_joint_norm, embedded_joint= compute_geodesic_rdm_class(resp_joint_reshape, method='isomap')

        dict_sigma[(sigma_class_, sigma_view_)] = cal_joint_performs(rdm_joint,config,resp_joint_reshape,embedded_joint)
# 保存性能指标
with open(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-20.pkl'), 'wb') as file:
    pickle.dump(dict_sigma, file)



pkl_path = os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-20.pkl')
with open(pkl_path, 'rb') as f:
    dict_sigma = pickle.load(f)  # 读取
#可视化################################################################################################################
fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

sigma_class_range = np.linspace(0.1, 4, 40)
sigma_view_range = np.linspace(10, 140, 40)
view_matrix = np.zeros((len(sigma_class_range), len(sigma_view_range)))
Intra_matrix = np.zeros((len(sigma_class_range), len(sigma_view_range)))
Inter_matrix = np.zeros((len(sigma_class_range), len(sigma_view_range)))
Acc_matrix = np.zeros((len(sigma_class_range), len(sigma_view_range)))
# 提取 accu_svr 值填入矩阵
for i, sigma_class in enumerate(sigma_class_range):
    for j, sigma_view in enumerate(sigma_view_range):
        key = (sigma_class, sigma_view)
        if key in dict_sigma:
            view_matrix[i, j] = dict_sigma[key]['Log-RDM mse']
            Intra_matrix[i, j] = dict_sigma[key]['Intra-class distance']
            Inter_matrix[i, j] = dict_sigma[key]['Inter-class distance']
            Acc_matrix[i, j] = dict_sigma[key]['Accuracy']
        else:
            view_matrix[i, j] = np.nan  # 或设置为 0，根据需要处理缺失值
            Intra_matrix[i, j] = np.nan  # 或设置为 0，根据需要处理缺失值
            Inter_matrix[i, j] = np.nan  # 或设置为 0，根据需要处理缺失值
            Acc_matrix[i, j] = np.nan

# 绘图
import matplotlib.pyplot as plt
from matplotlib import gridspec
fig = plt.figure(figsize=(3.6, 1.4))
gs = gridspec.GridSpec(1, 2,width_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(view_matrix, origin='lower', aspect='auto',
                extent=[sigma_class_range[0], sigma_class_range[-1],
                        sigma_view_range[0], sigma_view_range[-1]],
                cmap='viridis')
X, Y = np.meshgrid(sigma_class_range, sigma_view_range)
CS = ax1.contour(X, Y, view_matrix, levels=[-11],
                 colors='red', linewidths=0.5)
ax1.clabel(CS, inline=True, fontsize=7, fmt="%.1f")
ax1.set_xlabel('Sigma class',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-5)
ax1.set_ylabel('Sigma view',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-5)
ax1.set_title('Log(RDM mse)',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax1.spines['top'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax1.spines['right'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax1.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax1.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
ax1.set_xticks([0.1,4])
ax1.set_yticks([10,140])
# ax1.set_yticklabels(['1', '2', '3', '4'])
ax1.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)  # 设置 X 和 Y 轴的刻度线宽度为 2
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

cbar = fig.colorbar(im, ax=ax1)
# cbar.set_label('SVR accuracy',fontsize=fontsize_-1, fontname=fontname_, fontweight=fontweight_)
cbar.ax.tick_params(labelsize=fontsize_-1)
cbar.set_ticks([-3, -6,-9,-12])
for label in cbar.ax.get_yticklabels():
    label.set_fontname(fontname_)
    label.set_fontweight(fontweight_)

ax1 = fig.add_subplot(gs[0, 1])
im = ax1.imshow(Acc_matrix, origin='lower', aspect='auto',
                extent=[sigma_class_range[0], sigma_class_range[-1],
                        sigma_view_range[0], sigma_view_range[-1]],
                cmap='viridis')
X, Y = np.meshgrid(sigma_class_range, sigma_view_range)
CS = ax1.contour(X, Y, Acc_matrix, levels=[0.9],
                 colors='red', linewidths=0.5)
ax1.clabel(CS, inline=True, fontsize=7, fmt="%.1f")
ax1.set_xlabel('Sigma class',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-5)
ax1.set_ylabel('Sigma view',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-5)
ax1.set_title('Accuracy',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax1.spines['top'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax1.spines['right'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax1.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax1.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
ax1.set_xticks([0.1,4])
ax1.set_yticks([10,140])
# ax1.set_xticks([0.1,0.2,0.3,0.4,0.5])
# ax1.set_xticklabels(['0.1', '0.2', '0.3', '0.4','0.5'])
# ax1.set_yticks([1,2,3,4])
# ax1.set_yticklabels(['1', '2', '3', '4'])
ax1.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)  # 设置 X 和 Y 轴的刻度线宽度为 2
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
cbar = fig.colorbar(im, ax=ax1)
# cbar.set_label('SVR accuracy',fontsize=fontsize_-1, fontname=fontname_, fontweight=fontweight_)
cbar.ax.tick_params(labelsize=fontsize_-1)
# cbar.set_ticks([0.2, 0.4,0.6,0.8,1.0])
cbar.set_ticks([0, 0.2,0.4,0.6,0.8,1])
for label in cbar.ax.get_yticklabels():
    label.set_fontname(fontname_)
    label.set_fontweight(fontweight_)

plt.subplots_adjust(wspace=0.4, hspace=0.5, right=0.95, left=0.1, top=0.88, bottom=0.20)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-3.eps'), dpi=600, format='eps')


embedded_joint=dict_sigma[(1.8, 16.666666666666668)]['embedded_joint']
print(list(dict_sigma.keys())[365])
num_categories = config.num_categories
views_per_class = config.objects_per_category
colors = plt.cm.get_cmap('tab10', num_categories)  # tab10 色表有 10 种颜色
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_categories):
    idx_start = i * views_per_class
    idx_end = (i + 1) * views_per_class
    traj = embedded_joint[idx_start:idx_end, :]  # shape: (36, 3)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(i), label=f'Class {i}', linewidth=0.5)
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(i), s=2, alpha=0.8)  # 可选散点叠加
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_,pad=-4)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
# ax.view_init(elev=30, azim=135)
ax.grid(False)
ax.set_axis_off()
ax.set_xlabel('Dim 1',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-8)
ax.set_ylabel('Dim 2',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-8)
ax.set_zlabel('Dim 3',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-8)
# ax.set_title('Multiplicative',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,pad=-10)
ax.view_init(elev=-82, azim=18)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-4.eps'), dpi=600, format='eps')






embedded_joint=dict_sigma[(3.1999999999999997, 36.66666666666667)]['embedded_joint']
print(list(dict_sigma.keys())[365])
num_categories = config.num_categories
views_per_class = config.objects_per_category
colors = plt.cm.get_cmap('tab10', num_categories)  # tab10 色表有 10 种颜色
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_categories):
    idx_start = i * views_per_class
    idx_end = (i + 1) * views_per_class
    traj = embedded_joint[idx_start:idx_end, :]  # shape: (36, 3)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(i), label=f'Class {i}', linewidth=0.5)
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(i), s=2, alpha=0.8)  # 可选散点叠加
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_,pad=-4)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
# ax.view_init(elev=30, azim=135)
ax.grid(False)
ax.set_axis_off()
ax.set_xlabel('Dim 1',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-8)
ax.set_ylabel('Dim 2',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-8)
ax.set_zlabel('Dim 3',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-8)
# ax.set_title('Multiplicative',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,pad=-10)
ax.view_init(elev=173, azim=171)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-7.eps'), dpi=600, format='eps')