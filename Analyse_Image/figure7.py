import numpy as np
import os
from tools_joint import *

class joint_config:
    def __init__(self):
        self.num_neurons = 10  # 设置神经元数量
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

config=joint_config()

#计算视角响应
mu_view = np.arange(0, 360, 360 / config.num_neurons)  # mu均匀分布
amp_view = np.ones(config.num_neurons) * 1  # 幅值设置为1
sigma_view = np.ones(config.num_neurons) * 50  # 调谐宽度为50
params_view = np.concatenate([mu_view, amp_view, sigma_view])
resp_view = circular_gaussian(config.angles, mu_view, sigma_view, amp_view)  # shape: (num_neurons, 36)
#计算类别响应
mu_class = np.linspace(0, config.num_categories - 1, config.num_neurons)
amp_class = np.ones(config.num_neurons) * 1  # 幅值设置为1
sigma_class = np.ones(config.num_neurons) * 1  # 调谐宽度为1
params_class = np.concatenate([mu_class, amp_class, sigma_class])
resp_class=cal_class_response(params_class, config)
#计算联合调谐的响应矩阵，计算表征距离矩阵
resp_joint=combine_view_class_response(resp_class, resp_view, mode='additive')    #(n_neurons, n_categories, n_angles)
resp_joint_reshape = resp_joint.reshape(resp_joint.shape[0], -1)  # shape: (neurons, samples)
rdm_joint, rdm_joint_norm, embedded_joint= compute_geodesic_rdm_class(resp_joint_reshape, method='isomap')

# ===================== 绘制调谐曲线（按 sigma 扩展） =====================
fontsize_ = 7
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(1.5, 1.8))
gs = gridspec.GridSpec(2, 1)

angles = config.angles
categories = np.arange(config.num_categories)

# ========= 第一行：视角调谐（sigma_view: 10 -> 140，等间隔的10条） =========
ax = fig.add_subplot(gs[0, 0])
sigmas_view = np.linspace(10, 140, 10)
mu_view_fixed = 180.0                                   # 固定调谐中心（角度）
amp_view_plot = np.ones_like(sigmas_view)

# 向量化计算 (10, 36)
resp_view_plot = circular_gaussian(
    angles,
    mu=np.full_like(sigmas_view, mu_view_fixed, dtype=float),
    sigma=sigmas_view.astype(float),
    amp=amp_view_plot.astype(float)
)

for i in range(resp_view_plot.shape[0]):
    color = 'red' if i == 3 else 'gray'                 # 可选：第6条高亮
    ax.plot(angles + 1, resp_view_plot[i], color=color, alpha=0.9, linewidth=linewidth_plot)

ax.set_xlim([1, 360]); ax.set_xticks([1, 360])
ax.set_title(
    fr'View tuning',
    fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_
)
ax.set_xlabel('Angle', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-4)
ax.set_ylabel('Response', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

# ========= 第二行：类别调谐（sigma_class: 1 -> 10，等间隔的10条） =========
ax = fig.add_subplot(gs[1, 0])
sigmas_class = np.linspace(0.1, 4, 10)
mu_class_fixed = 5.0                                    # 固定调谐中心（第5类，可按需调整）
amp_class_plot = np.ones_like(sigmas_class)

# 利用已有 cal_class_response 生成 (10, num_categories) 的类别调谐
params_class_plot = np.concatenate([                     # 顺序与你的函数保持一致：[mu, amp, sigma]
    np.full_like(sigmas_class, mu_class_fixed, dtype=float),
    amp_class_plot.astype(float),
    sigmas_class.astype(float)
])
resp_class_plot = cal_class_response(params_class_plot, config)  # 形状应为 (10, 10)

for i in range(resp_class_plot.shape[0]):
    color = 'red' if i == 3 else 'gray'
    ax.plot(categories + 1, resp_class_plot[i], color=color, linewidth=linewidth_plot, alpha=0.9)

ax.set_xlim([1, 10]); ax.set_xticks([1, 10])
ax.set_title(
    fr'Class tuning',
    fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_
)
ax.set_xlabel('Class', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-4)
ax.set_ylabel('Response', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

plt.subplots_adjust(wspace=0.6, hspace=1.3, right=0.85, left=0.25, top=0.85, bottom=0.2)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-1.eps'), dpi=600, format='eps')



#绘制三维联合调谐######################################################################################################
n_cat, n_ang = resp_joint.shape[1], resp_joint.shape[2]
X, Y = np.meshgrid(np.arange(n_ang), np.arange(n_cat))
num_neurons = resp_joint.shape[0]
angles = config.angles
categories = np.arange(config.num_categories)
idx=5
labelpad_=-5

fig = plt.figure(figsize=(1.4,1.4))
gs = gridspec.GridSpec(1,1)

ax = fig.add_subplot(gs[0, 0], projection='3d')
resp_joint=combine_view_class_response(resp_class, resp_view, mode='additive')    #(n_neurons, n_categories, n_angles)
Z = resp_joint[idx]
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.95)
# ax.set_title(f'Additive',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, pad=-4)
ax.set_xlabel('Angle Idx',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=labelpad_)
ax.set_ylabel('Class Idx',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=labelpad_)
ax.set_zlabel('Response',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=-2)
ax.zaxis.label.set_rotation(270)  # 明确设定z轴标签角度为90度
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.view_init(elev=30, azim=135)
ax.grid(False)
ax.set_xticks([0,36])
ax.set_xticklabels(['0', '360'])
ax.set_yticks([1,10])
# ax.set_xticks([])  # Remove x-axis ticks
# ax.set_yticks([])  # Remove y-axis ticks
ax.set_zticks([0,2])  # Remove z-axis ticks
ax.tick_params(axis='both', pad=-3)      # x/y 的标签内缩
ax.tick_params(axis='z',   pad=-3)
plt.subplots_adjust(wspace=0.6, hspace=1.2, right=0.80, left=0.05, top=0.92, bottom=0.2)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-2.eps'), dpi=600, format='eps')




import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os

# 使用你已有的 RDM（任选其一）
rdm = config.view_target_rdm if 'view_target_rdm' in globals() else config.view_target_rdm  # shape: (36, 36)
# 统一风格
fontsize_ = 8
linewidth_ax = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'
fig = plt.figure(figsize=(2.2, 2.2))
gs  = gridspec.GridSpec(1, 1)
ax  = fig.add_subplot(gs[0, 0])
# 可选：归一化到 [0,1] 便于比较
# rdm_vis = rdm / (rdm.max() + 1e-12)
rdm_vis = rdm
im = ax.imshow(rdm_vis, origin='lower', cmap='coolwarm', interpolation='nearest', aspect='equal')
ax.invert_yaxis()  # 翻转 y 轴
# 角度刻度：在索引 0/9/18/27/35 处标 0/90/180/270/360
ticks_idx   = [0, 35]
ticks_label = ['0', '360']
ax.set_xticks(ticks_idx); ax.set_xticklabels(ticks_label)
ax.set_yticks(ticks_idx); ax.set_yticklabels(ticks_label)
ax.set_xlabel('Angle (deg)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-3)
ax.set_ylabel('Angle (deg)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-3)
# 轴与网格样式
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax.spines[sp].set_linewidth(linewidth_ax)
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)
# 色条更小
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.ax.tick_params(labelsize=fontsize_)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['0', '1'])
plt.tight_layout()
# 保存（按需）
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-viewRDM.eps'), dpi=600, format='eps')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 取 RDM（根据你的类）
rdm = config.class_target_rdm if 'class_target_rdm' in globals() else config.class_target_rdm  # (360, 360)

# 基本参数
n_cls   = config.num_categories        # 10
per_cls = config.objects_per_category  # 36
N       = n_cls * per_cls              # 360

# 画图
fig, ax = plt.subplots(figsize=(2.2, 2.2))
im = ax.imshow(rdm, origin='lower', cmap='coolwarm', aspect='equal', interpolation='nearest')
ax.invert_yaxis()  # 翻转 y 轴
# 轴刻度：按“类别中心”标注 1..10
centers = np.arange(per_cls//2, N, per_cls)          # 每个类别块中心：18, 54, ..., 342
ax.set_xticks(centers); ax.set_xticklabels([str(i) for i in range(1, n_cls+1)])
ax.set_yticks(centers); ax.set_yticklabels([str(i) for i in range(1, n_cls+1)])
ax.set_xlabel('Class', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=0)
ax.set_ylabel('Class', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=-1)
ax.tick_params(axis='both', width=linewidth_ax, labelsize=fontsize_)
# 类别边界（每 36 个样本一条）
for k in range(0, N+1, per_cls):
    ax.axhline(k-0.5, color='k', lw=0.4, alpha=0.6)
    ax.axvline(k-0.5, color='k', lw=0.4, alpha=0.6)

# 色条仅显示 0 和 1（若你的 RDM 本就 0/1）
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.ax.tick_params(labelsize=fontsize_)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['0', '1'])
# 细节样式
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure7', 'figure7-classRDM.eps'), dpi=600, format='eps')

