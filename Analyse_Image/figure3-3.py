import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tools_new import *

neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应

name_class = dict_class['ento_exp15_ch20']['class']['name']
label_fine = [i.split('_')[2] for i in name_class]
neurons = list(dict_class.keys())

N_NEURON_SUB = 100
REPEATS = 1000
rng = np.random.default_rng()
dict_perform={}
for r in range(REPEATS):
    print(r)
    dict_perform[r]={}
    sel_neurons = rng.choice(neurons, size=100, replace=False)
    resp = np.array([dict_class[neu]['class']['resp'] for neu in sel_neurons])

    metric_list = ['Color', 'Shape', 'Texture', 'V1-like', 'V2-like',
                   'Alexnet Conv1', 'Alexnet Conv2', 'Alexnet Conv3', 'Alexnet Conv4', 'Alexnet Conv5']
    for metric_name in metric_list:
        fitcorr = np.array([dict_class[neu]['fit_corr'][metric_name] for neu in sel_neurons], dtype=float)
        dict_perform[r][metric_name+'_median']=np.median(fitcorr)
        dict_perform[r][metric_name + '_mean'] = np.mean(fitcorr)

    metric_list = ['Selectivity index', 'Separation ratio']
    for metric_name in metric_list:
        fitcorr = np.array([dict_class[neu]['class']['metrics'][metric_name] for neu in sel_neurons], dtype=float)
        dict_perform[r][metric_name+'_median']=np.median(fitcorr)
        dict_perform[r][metric_name + '_mean'] = np.mean(fitcorr)

    # 高斯收缩版（更稳定）
    X = resp.T  # 变成 (N × D)
    K = 8  # 类别数
    n_per_class = 25
    y = np.repeat(np.arange(K), n_per_class)
    I = mi_gaussian_lda_approx(X, y, ridge=1e-3, base=2)

    print("Cluster MI (Gaussian):", I)
    dict_perform[r]['MI'] = round(I, 4)

    dict_temp=cal_class_performs(resp, label_fine)
    dict_perform[r].update(dict_temp)

with open('dict_performs_class_true.pkl', 'wb') as f:
    pickle.dump(dict_perform, f)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

features=['Color_mean','Shape_mean','Texture_mean', 'V1-like_mean', 'V2-like_mean',
       'Alexnet Conv1_mean', 'Alexnet Conv2_mean', 'Alexnet Conv3_mean', 'Alexnet Conv4_mean', 'Alexnet Conv5_mean']
Y_names = ['Accuracy']
X_names = features


for y in Y_names:
    ok = df[X_names + [y]].dropna()
    X = ok[X_names].values
    Y = ok[y].values
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3,3,21), cv=5))
    model.fit(X, Y)
    # 回溯系数
    coefs = model.named_steps['ridgecv'].coef_
    res = pd.Series(coefs, index=X_names).sort_values(key=np.abs, ascending=False)
    print(f'\n[{y}] Ridge 重要性（绝对值排序）\n', res.head(12))

# 简易“偏相关”：把 y 对除x以外的特征做回归残差；x 对其他特征做回归残差；两残差相关
from sklearn.linear_model import LinearRegression
def partial_corr(df, x, y, controls):
    ok = df[[x,y]+controls].dropna()
    Xc = ok[controls].values
    rgr = LinearRegression().fit(Xc, ok[y].values); ry = ok[y].values - rgr.predict(Xc)
    rgr = LinearRegression().fit(Xc, ok[x].values); rx = ok[x].values - rgr.predict(Xc)
    return spearmanr(rx, ry)

pc = partial_corr(df, 'Alexnet Conv5_mean', 'Accuracy',
                  [c for c in X_names if c!='Alexnet Conv5_mean'])
print('\n偏相关 Alexnet Conv5_mean ~ Accuracy | 其他：', pc)









MI=[dict_perform[t]['MI'] for t in dict_perform.keys()]
Alexnet_Conv5=[dict_perform[t]['Alexnet Conv5_mean'] for t in dict_perform.keys()]
ACC=[dict_perform[t]['Accuracy'] for t in dict_perform.keys()]

r = np.corrcoef(MI, ACC)[0, 1]
plt.figure(figsize=(5, 4))
plt.scatter(MI, ACC, s=10)
plt.xlabel('Cluster MI (bits)')
plt.ylabel('Accuracy')
plt.title('MI vs Accuracy (r = {:.3f})'.format(r))
plt.tight_layout()

plt.figure(figsize=(5, 3.2))
plt.hist(MI, bins=20, edgecolor='black', alpha=0.8)
plt.xlabel('Mutual Information (bits)')
plt.ylabel('Count')
plt.title('MI Histogram')



fontsize_ = 8
linewidth_ax = 0.5
linewidth_plot = 0.7
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(4.6, 2.3))
gs = gridspec.GridSpec(1, 5)   # 1 行 4 列


import numpy as np
import matplotlib.pyplot as plt

lists=['Color_mean','Shape_mean','Texture_mean', 'V1-like_mean', 'V2-like_mean',
       'Alexnet Conv1_mean', 'Alexnet Conv2_mean', 'Alexnet Conv3_mean', 'Alexnet Conv4_mean', 'Alexnet Conv5_mean']
# lists=['Alexnet Conv1_mean', 'Alexnet Conv2_mean', 'Alexnet Conv3_mean', 'Alexnet Conv4_mean', 'Alexnet Conv5_mean']

for ff in lists:
    MI  = np.array([dict_perform[t]['MI'] for t in dict_perform.keys()], dtype=float)
    AX5 = np.array([dict_perform[t][ff] for t in dict_perform.keys()], dtype=float)
    ACC = np.array([dict_perform[t]['Accuracy'] for t in dict_perform.keys()], dtype=float)

    def equal_freq_edges(mi, n_bins=5):
        mi = np.asarray(mi, float)
        mi = mi[np.isfinite(mi)]
        q = np.linspace(0, 100, n_bins + 1)              # 0%, 20%, 40%, 60%, 80%, 100%
        # numpy<=1.22 用 interpolation；新版本用 method='linear'
        edges = np.percentile(mi, q, interpolation='linear')
        edges[0]  = mi.min()                              # 保守处理
        edges[-1] = mi.max()
        return edges

    n_bins=5
    edges = equal_freq_edges(MI, n_bins=5)

    bin_sel_list = []
    all_y = []
    for i in range(n_bins):
        left, right = edges[i], edges[i+1]
        if i < n_bins - 1:
            sel = (MI >= left) & (MI < right)
        else:
            sel = (MI >= left) & (MI <= right)
        bin_sel_list.append(sel)
        if np.any(sel):
            all_y.append(ACC[sel])

    for i in range(n_bins):
        ax = fig.add_subplot(gs[0, i])
        sel = bin_sel_list[i]
        left, right = edges[i], edges[i+1]
        ax.set_title(f'Bin {i+1}\n[{left:.3g}, {right:.3g}]', fontsize=9)
        if np.any(sel):
            x = AX5[sel]
            y = ACC[sel]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            bins_idx = np.array_split(np.arange(len(x_sorted)), 4)

            # 3) 逐bin统计均值与方差（ddof=1 为样本方差）
            mean_x = np.array([x_sorted[idx].mean() for idx in bins_idx])
            var_x = np.array([x_sorted[idx].var(ddof=1) for idx in bins_idx])
            mean_y = np.array([y_sorted[idx].mean() for idx in bins_idx])
            var_y = np.array([y_sorted[idx].var(ddof=1) for idx in bins_idx])
            std_x = np.sqrt(var_x)
            std_y = np.sqrt(var_y)
            # ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y,
            #              fmt='o-', linewidth=1.5, capsize=3)
            ax.plot(mean_y,label=ff)

            ax.text(0.02, 0.95, f'n={x_sorted.size}', transform=ax.transAxes,
                    va='top', ha='left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9)

        ax.set_xlabel('Alexnet Conv5 (mean)', fontsize=9)
        ax.set_ylim([0.35,0.45])
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 0.8))
plt.tight_layout()
plt.show()



MI = np.array([dict_perform[t]['MI'] for t in dict_perform.keys()], dtype=float)
AX5 = np.array([dict_perform[t]['Alexnet Conv1_mean'] for t in dict_perform.keys()], dtype=float)
ACC = np.array([dict_perform[t]['Accuracy'] for t in dict_perform.keys()], dtype=float)

# 清理无效值（保持三者配对）
mask = np.isfinite(MI) & np.isfinite(AX5) & np.isfinite(ACC)
mi  = MI[mask]
ax5 = AX5[mask]
acc = ACC[mask]

plt.figure(figsize=(4.6, 3.2))
sc = plt.scatter(ax5, mi, c=acc, s=20, alpha=0.85, cmap='viridis')  # 颜色由 ACC 决定
cbar = plt.colorbar(sc)
cbar.set_label('Accuracy')

plt.xlabel('Alexnet Conv5 (mean)')
plt.ylabel('Mutual Information (bits)')  # 如果是 nat 改成 nat
plt.title('MI vs Alexnet_Conv5 colored by ACC')
plt.grid(alpha=0.3, linestyle=':', linewidth=0.8)
plt.tight_layout()
plt.show()





#
# dict_performs={}
# metric_list=['Selectivity index','Separation ratio']    #'Selectivity index',  'Separation ratio'
# for metric_name in metric_list:
#     dict_mertics = {}
#     for neu in dict_class.keys():
#         dict_mertics[neu]=dict_class[neu]['class']['metrics'][metric_name]
#     sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
#     dict_performs[metric_name]=cal_class_performs(sorted_dict, dict_class)
#
#
# metric_list=['Color','Shape','Texture','V1-like','V2-like',
#              'Alexnet Conv1','Alexnet Conv2','Alexnet Conv3','Alexnet Conv4','Alexnet Conv5']
# for metric_name in metric_list:
#     dict_mertics = {}
#     for neu in dict_class.keys():
#         dict_mertics[neu]=dict_class[neu]['fit_corr'][metric_name]
#     sorted_dict = dict(sorted(dict_mertics.items(), key=lambda x: x[1]))
#     dict_performs[metric_name]=cal_class_performs(sorted_dict, dict_class)
#
# save_path = r'E:\Image_paper\Project\dict_performs_class_rbf.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(dict_performs, f)
#
# fontsize_ = 8
# linewidth_ax = 0.5
# linewidth_plot = 0.7
# fontname_ = 'Arial'
# fontweight_ = 'normal'
# # 为每条折线指定不同颜色（可按需改）
# line_color = {
#     'Selectivity index': '#1f77b4',  # 蓝
#     'Separation ratio':  '#ff7f0e',  # 橙
#     'Shape':             '#2ca02c',  # 绿
#     'Color':             '#e41a1c',  # 红
#     'Texture':           '#984ea3',  # 紫
#     'Alexnet Conv1':     '#4daf4a',  # 鲜绿
#     'Alexnet Conv3':     '#377eb8',  # 靛蓝
#     'Alexnet Conv5':     '#a65628',  # 棕
# }
#
# fig = plt.figure(figsize=(12, 2.8))
# gs = gridspec.GridSpec(1, 4)
# axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
# n_points = 7  # 取 10 个中心点
# half_window = 20  # 每个点两侧各取 15 个样本
# from matplotlib.lines import Line2D
# plot_performs=['Selectivity index', 'Separation ratio', 'Color', 'Texture', 'Shape', 'Alexnet Conv1', 'Alexnet Conv3', 'Alexnet Conv5']
# for ppf in plot_performs:
#     idx = np.array(sorted(dict_performs['Selectivity index'].keys()))
#     series = {
#         'Accuracy':           np.array([dict_performs[ppf][i]['Accuracy']            for i in idx]),
#         'Intra-class dist.':  np.array([dict_performs[ppf][i]['Intra-class dist.']   for i in idx]),
#         'Inter-class dist.':  np.array([dict_performs[ppf][i]['Inter-class dist.']   for i in idx]),
#         'Separability':       np.array([dict_performs[ppf][i]['Separability']        for i in idx]),
#     }
#
#     clr = line_color.get(ppf, 'tab:red')  # ← 取到具体颜色字符串
#     for ax, (title, y) in zip(axes, series.items()):
#         # —— 选中心位置（避开边界，确保有完整窗口）
#         idx_min = idx[0] + half_window
#         idx_max = idx[-1] - half_window
#         centers = np.linspace(idx_min, idx_max, n_points, dtype=int)
#         # —— 计算每个中心的窗口均值与 SEM
#         mean_vals, sem_vals, center_x = [], [], []
#         for c in centers:
#             mask = (idx >= c - half_window) & (idx <= c + half_window)
#             y_win = y[mask]
#             if y_win.size == 0:
#                 continue
#             mean_vals.append(np.nanmean(y_win))
#             sd = np.nanstd(y_win, ddof=1) if y_win.size > 1 else 0.0
#             sem_vals.append(sd / np.sqrt(max(y_win.size, 1)))
#             center_x.append(c)
#         center_x  = np.asarray(center_x)
#         mean_vals = np.asarray(mean_vals)
#         sem_vals  = np.asarray(sem_vals)
#         # —— 仅绘制误差线（均值连线 + 误差棒），不画原始曲线
#         ax.errorbar(center_x, mean_vals, yerr=sem_vals,
#                     fmt='o-', lw=1.0, ms=3.0, capsize=2,
#                     color=clr, ecolor=clr, zorder=3,label=ppf,)
#         ax.set_xlabel('Start index', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
#         ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_linewidth(linewidth_ax); ax.spines['left'].set_linewidth(linewidth_ax)
#
# legend_handles = [Line2D([0], [0], color=line_color[k], marker='o', lw=1.2, label=k)
#                   for k in plot_performs]
# fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.995, 1.0),
#            ncol=4, frameon=False, fontsize=fontsize_)
#
# plt.subplots_adjust(wspace=0.35, hspace=0.3, right=0.98, left=0.12, top=0.8, bottom=0.18)
# plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3', 'rbf.png'), dpi=600, format='png')