import os
import pickle
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import h5py
from matplotlib import gridspec
from matplotlib.ticker import FixedLocator

def sort_resp(resp_class, n_class=8, samples_per_class=25):
    resp_sorted_by_row = resp_class  # 排序后的矩阵 (268, 200)
    resp_reshaped = resp_sorted_by_row.reshape(resp_sorted_by_row.shape[0], n_class, samples_per_class)
    resp_sorted_final = np.zeros_like(resp_sorted_by_row)
    for n in range(resp_reshaped.shape[0]):
        data=resp_reshaped[n,:,:]
        data_sorted = np.sort(data, axis=1)[:, ::-1]

        row_means = data_sorted.mean(axis=1)  # 每个神经元的平均响应
        row_order = np.argsort(row_means)[::-1]  # 平均值从高到低的索引
        data_sorted_by_row = data_sorted[row_order]  # 排序后的矩阵 (268, 200)
        resp_sorted_final[n]=data_sorted_by_row.reshape(-1)
    return resp_sorted_final


def moving_avg_same(raster_trial, bin_size=50, step=1):
    kernel = np.ones(bin_size, dtype=float) / bin_size  # 平均核
    # 沿时间轴对每个 trial 做 1D 卷积，same 模式保证长度不变
    rate_full = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                                    axis=0, arr=raster_trial)
    # 步长=1：保持与输入同维度；步长>1：做等间隔采样
    if step == 1:
        return rate_full
    else:
        return rate_full[::step, :]

def plot_category_rates(spk_rate, label_fine, class_fine, ax):
    spk_rate = np.asarray(spk_rate)
    labels = np.asarray(label_fine)
    T, N = spk_rate.shape
    assert N == len(labels), "label_fine 长度应与刺激数一致"
    time = np.arange(T)

    # colors = plt.cm.Set1(np.linspace(0, 1, len(class_fine)))
    #
    # colors = plt.cm.tab10(np.linspace(0, 1, n_class))

    for cname in class_fine:
        idx = np.where(labels == cname)[0]  # 属于该类别的刺激索引
        if idx.size == 0:
            continue
        X = spk_rate[:, idx]  # (T, n_c)
        mean_c = X.mean(axis=1)
        sem_c = X.std(axis=1, ddof=1) / np.sqrt(X.shape[1])  # SEM
        ax.plot(time, mean_c, label=cname, linewidth=0.7)
        ax.fill_between(time, mean_c - sem_c, mean_c + sem_c, alpha=0.25)


neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应


dict_mertics={}
for neu in dict_class.keys():
    dict_mertics[neu]=dict_class[neu]['class']['metrics']['Selectivity index']

# 从低到高对字典进行排序
dict_sort = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
resp_class = [dict_class[neu]['class']['resp'] for neu in dict_sort.keys()]
resp_class = np.array(resp_class)

name_class = dict_class[neu]['class']['name']
label_fine = [i.split('_')[2] for i in name_class]
class_fine = list(dict.fromkeys(label_fine))

n_class=8
samples_per_class=25

fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

#绘制响应热力图#################################################################################################################
fig = plt.figure(figsize=(2.8, 3.7))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
# resp_class=sort_resp(resp_class, n_class=8, samples_per_class=25)
# resp_class = minmax_scale(resp_class, feature_range=(0, 1), axis=1)
im = ax.imshow(resp_class, aspect='auto', interpolation='none',cmap='jet')  # 去掉 vmin/vmax 则用全范围
cbar = fig.colorbar(im, ax=ax, fraction=0.04, shrink=0.6)
# 设置刻度字体大小和字体类型
cbar.ax.tick_params(labelsize=8)  # 字号
cbar.set_ticks([0, 1])          # 只显示 0 和 1
cbar.set_ticklabels(['0', '1']) # 可选：自定义刻度标签
cbar.set_label('Norm. response',   # 标签文字
               fontsize=10,         # 字号
               fontname=fontname_,         # 字体
               fontweight=fontweight_)     # 粗细
for label in cbar.ax.get_yticklabels():
    label.set_fontname(fontname_)  # 字体类型，可换为 'Arial' 等
# 类别分界线
for i in range(1, n_class):
    ax.axvline(x=i * samples_per_class - 0.5, color='r', linestyle='-', linewidth=1.2)
# 类别中心位置
centers = [samples_per_class * (i + 0.5) - 0.5 for i in range(n_class)]
ax.set_xticks(centers)
# ax.set_xticklabels(class_fine, rotation=45, ha='right')
# ax.set_xticklabels(class_fine, rotation=45, ha='right',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Neurons (increasing selectivity, n=253)',fontsize=10, fontname=fontname_, fontweight=fontweight_)
# ax.set_xticks([])  # 去掉刻度标签以更清爽（需要时可注释掉）
ax.set_yticks([])  # 去掉刻度标签以更清爽（需要时可注释掉）
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_visible(False) # 设置下边框线条宽度
ax.spines['left'].set_visible(False)  # 设置左边框线条宽度

plt.subplots_adjust(wspace=0.4, hspace=0.05, right=0.88, left=0.12, top=0.95, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure1', 'f1_class.eps'), dpi=600, format='eps')


#绘制选择性#################################################################################################################
high_selc_neuron='ento_exp8_ch14'
low_selc_neuron='ni_exp15_ch4'

fig = plt.figure(figsize=(2.0, 3.5))
gs = gridspec.GridSpec(2, 1)
rootpath=r'E:\Image_paper\Project\Acute_Image\results\neuron_response'

ax = fig.add_subplot(gs[0, 0])
neu=low_selc_neuron
raster_path=os.path.join(rootpath,neu.split('ch')[0][:-1],neu.split('ch')[0][:-1]+'_raster.mat')
with h5py.File(raster_path, 'r') as f:
    raster_trial = np.array(f['raster_trial']).T
raster_trial=np.mean(raster_trial[2900:3600,:,0:200,int(neu.split('ch')[1])-1],1)
spk_rate = moving_avg_same(raster_trial, bin_size=50, step=1)
spk_rate=spk_rate*1000
plot_category_rates(spk_rate, label_fine, class_fine, ax)
ax.set_xlim([100, 600])
ax.xaxis.set_major_locator(FixedLocator([200, 400, 600]))
ax.set_xticklabels(['100','300','500'],
                   fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_xlabel('Time (ms)',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Firing rate (Hz)',
              fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_title(
    'Class selectivity={:.2f}\nSeparation ratio={:.2f}'.format(
        round(dict_class[neu]['class']['metrics']['Selectivity index'], 2),
        round(dict_class[neu]['class']['metrics']['Separation ratio'], 2)
    ),loc='left',
    fontsize=fontsize_+1,
    fontname=fontname_,
    fontweight=fontweight_
)
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

ax = fig.add_subplot(gs[1, 0])
neu=high_selc_neuron
raster_path=os.path.join(rootpath,neu.split('ch')[0][:-1],neu.split('ch')[0][:-1]+'_raster.mat')
with h5py.File(raster_path, 'r') as f:
    raster_trial = np.array(f['raster_trial']).T
raster_trial=np.mean(raster_trial[2900:3600,:,0:200,int(neu.split('ch')[1])-1],1)
spk_rate = moving_avg_same(raster_trial, bin_size=50, step=1)
spk_rate=spk_rate*1000
plot_category_rates(spk_rate, label_fine, class_fine, ax)
ax.set_xlim([100, 600])
ax.xaxis.set_major_locator(FixedLocator([200, 400, 600]))
ax.set_xticklabels(['100','300','500'],
                   fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_xlabel('Time (ms)',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Firing rate (Hz)',
              fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_title(
    'Class selectivity={:.2f}\nSeparation ratio={:.2f}'.format(
        round(dict_class[neu]['class']['metrics']['Selectivity index'], 2),
        round(dict_class[neu]['class']['metrics']['Separation ratio'], 2)
    ),loc='left',
    fontsize=fontsize_+1,
    fontname=fontname_,
    fontweight=fontweight_
)
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

plt.subplots_adjust(wspace=0.4, hspace=1, right=0.95, left=0.25, top=0.85, bottom=0.15)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure1', 'f1_firerate.eps'), dpi=600, format='eps')

#绘制分离比#################################################################################################################
fig = plt.figure(figsize=(4.2, 2.8))
gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[0, 0])
neu=high_selc_neuron
resp=dict_class[neu]['class']['resp']

n_class, k = 8, 25
resp_by_class = resp.reshape(n_class, k)  # (8,25)
# 散点（按类着色，带水平抖动）
for c in range(n_class):
    x_center = c
    jitter = (np.random.rand(k) - 0.5) * 0.3
    xs = x_center + jitter
    ax.scatter(xs, resp_by_class[c], s=14, alpha=0.8, zorder=2)
# 方差范围（均值±标准差）+ 均值点
means = resp_by_class.mean(axis=1)
stds  = resp_by_class.std(axis=1)
for c in range(n_class):
    ax.vlines(c, means[c]-stds[c], means[c]+stds[c], colors='k', linewidth=1, zorder=3)
    ax.scatter([c], [means[c]], color='k', s=15, zorder=4)
# 轴与标签
labels = class_fine if 'class_fine' in globals() else [f'C{i+1}' for i in range(n_class)]
ax.set_xticks(range(n_class))
ax.set_xticklabels([])
# ax.set_xlabel('Category',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Response (norm)',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_xlim(-0.5, n_class-0.5)
# 可选美化
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)


ax = fig.add_subplot(gs[1, 0])
neu=low_selc_neuron
resp=dict_class[neu]['class']['resp']

n_class, k = 8, 25
resp_by_class = resp.reshape(n_class, k)  # (8,25)
# 散点（按类着色，带水平抖动）
for c in range(n_class):
    x_center = c
    jitter = (np.random.rand(k) - 0.5) * 0.3
    xs = x_center + jitter
    ax.scatter(xs, resp_by_class[c], s=14, alpha=0.8, zorder=2)
# 方差范围（均值±标准差）+ 均值点
means = resp_by_class.mean(axis=1)
stds  = resp_by_class.std(axis=1)
for c in range(n_class):
    ax.vlines(c, means[c]-stds[c], means[c]+stds[c], colors='k', linewidth=1, zorder=3)
    ax.scatter([c], [means[c]], color='k', s=15, zorder=4)
# 轴与标签
labels = class_fine if 'class_fine' in globals() else [f'C{i+1}' for i in range(n_class)]
ax.set_xticks(range(n_class))
# ax.set_xticklabels(labels)
ax.set_xticklabels(labels, rotation=60, ha='right',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
# ax.set_xlabel('Category',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_ylabel('Response (norm)',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
ax.set_xlim(-0.5, n_class-0.5)
# 可选美化
ax.spines['top'].set_visible(False)  # 去除上边框
ax.spines['right'].set_visible(False)  # 去除右边框
ax.spines['bottom'].set_linewidth(linewidth_ax)  # 设置下边框线条宽度
ax.spines['left'].set_linewidth(linewidth_ax)  # 设置左边框线条宽度
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(wspace=0.4, hspace=0.3, right=0.95, left=0.15, top=0.95, bottom=0.25)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure1', 'f1_class_para.eps'), dpi=600, format='eps')