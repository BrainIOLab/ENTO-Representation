import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib import font_manager

def round_sig(x, sig=3):
    #保留3位有效数字
    from math import log10, floor
    if x == 0:
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def plot_rdm(rdm, savepath, title='RDM'):
    fontsize_ = 8
    linewidth_ax = 0.5
    linewidth_plot = 0.7
    fontname_ = 'Arial'
    fontweight_ = 'normal'
    plt.figure(figsize=(1, 1))
    angles = np.linspace(0, 360, rdm.shape[0], endpoint=False)  # 或 angles = np.arange(rdm.shape[0])
    angle_ticks = [0, 90, 180, 270, 360]
    tick_indices = [np.argmin(np.abs(angles - a)) for a in angle_ticks]
    ax=sns.heatmap(rdm, cmap='coolwarm', xticklabels=False, yticklabels=False, square=True,cbar=False)
    # plt.xticks(ticks=tick_indices, labels=angle_ticks,fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # plt.yticks(ticks=tick_indices, labels=angle_ticks,fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # plt.title(title, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # 设置 colorbar 字体
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=fontsize_)  # 设置刻度字号
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontname(fontname_)
    #     label.set_fontweight(fontweight_)
    plt.savefig(savepath, dpi=600, format='png')
    # plt.tight_layout()
    # plt.show()

def save_rdm_class(rdm, savepath, title='RDM'):
    fontsize_ = 8
    linewidth_ax = 0.5
    linewidth_plot = 0.7
    fontname_ = 'Arial'
    fontweight_ = 'normal'
    plt.figure(figsize=(1, 1))
    angles = np.linspace(0, 360, rdm.shape[0], endpoint=False)  # 或 angles = np.arange(rdm.shape[0])
    angle_ticks = [0, 90, 180, 270, 360]
    tick_indices = [np.argmin(np.abs(angles - a)) for a in angle_ticks]
    ax=sns.heatmap(rdm, cmap='coolwarm', xticklabels=False, yticklabels=False, square=True,cbar=False)
    # plt.xticks(ticks=tick_indices, labels=angle_ticks,fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # plt.yticks(ticks=tick_indices, labels=angle_ticks,fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # plt.title(title, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # 设置 colorbar 字体
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=fontsize_)  # 设置刻度字号
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontname(fontname_)
    #     label.set_fontweight(fontweight_)
    plt.savefig(savepath, dpi=600, format='png')
    # plt.tight_layout()
    # plt.show()

def plot_rdm_class(rdm):

    plt.figure(figsize=(3, 3))
    sns.heatmap(rdm, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=True)
    plt.tight_layout()
    plt.show()

def visualize_view_represent(resp_view, rdm_view, embedded_view_2d, config, view_perform, savepath):
    #已知调谐曲线，绘制表征及性能指标
    angles=config.angles
    n_neurons=config.num_neurons

    fig = plt.figure(figsize=(9, 5.5))
    gs = gridspec.GridSpec(2, 3)
    title_name="d loss:{:.6f}; log_d loss:{:.6f}; disc:{:.4f}; energy:{:.4f}; ri:{:.6f}; log_ri:{:.6f}; \n fi:{:.4f}; mi:{:.4f}; ssi:{:.4f}; spi:{:.4f};".format(
        view_perform['RDM mse'],view_perform['Log-RDM mse'], np.mean(view_perform['Global distance']), view_perform['Energy'],
        view_perform['Redundant info'], view_perform['Log-redundant info'],view_perform['Fisher info'], view_perform['Mutual info'],
        view_perform['Selectivity'], view_perform['Sparsity'], fontsize=12)
    plt.suptitle(title_name)

    ax = fig.add_subplot(gs[0, 0:2])            #绘制调谐曲线
    for i in range(n_neurons):
        ax.plot(angles, resp_view[i,:], color='gray', alpha=0.6)
    ax.set_ylabel("Response")
    ax.set_title(f"Tuning Curves")
    ax.set_ylim([0,1])

    ax = fig.add_subplot(gs[0, 2])              #绘制给定rdm矩阵
    sns.heatmap(config.rdm_view_target, ax=ax, cmap='coolwarm', cbar=True)
    ax.set_title('Target RDM')
    ax.set_xticks([])
    ax.set_yticks([])

    # ax = fig.add_subplot(gs[1, 0])              #绘制2D嵌入
    # sc = ax.scatter(embedded_view_2d[0:-1:4, 0], embedded_view_2d[0:-1:4, 1], c=np.arange(90), cmap='hsv', s=10)
    # ax.set_title('2D Embedding')
    # x_mid = (embedded_view_2d[:, 0].max() + embedded_view_2d[:, 0].min()) / 2
    # y_mid = (embedded_view_2d[:, 1].max() + embedded_view_2d[:, 1].min()) / 2
    # radius = max(np.ptp(embedded_view_2d[:, 0]), np.ptp(embedded_view_2d[:, 1])) / 2 * 1.1
    # ax.set_xlim(x_mid - radius, x_mid + radius)
    # ax.set_ylim(y_mid - radius, y_mid + radius)
    # plt.colorbar(sc, ax=ax)
    #
    # ax = fig.add_subplot(gs[1, 1], projection='3d')              #绘制3D嵌入
    # isomap = Isomap(n_neighbors=10, n_components=3, metric='euclidean')
    # embedded_view_3d = isomap.fit_transform(resp_view.T)
    # colors = plt.cm.hsv(np.linspace(0, 1, 90))
    # ax.scatter(embedded_view_3d[0:-1:4, 0], embedded_view_3d[0:-1:4, 1], embedded_view_3d[0:-1:4, 2], c=colors, s=10)
    # ax.set_title(f'3D Embedding')
    # x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    # x_range, y_range, z_range = np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)
    # mid = lambda lim: np.mean(lim)
    # radius = 0.5 * max(x_range, y_range, z_range)
    # ax.set_xlim(mid(x_limits) - radius, mid(x_limits) + radius)
    # ax.set_ylim(mid(y_limits) - radius, mid(y_limits) + radius)
    # ax.set_zlim(mid(z_limits) - radius, mid(z_limits) + radius)

    ax = fig.add_subplot(gs[1, 2])              #绘制给定rdm矩阵
    sns.heatmap(rdm_view, ax=ax, cmap='coolwarm', cbar=True)
    ax.set_title('RDM')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(wspace=0.2, hspace=0.4, right=0.95, left=0.1, top=0.80, bottom=0.1)
    # plt.show()
    if savepath!='none':
        plt.savefig(savepath, dpi=600)
        plt.close()
    else:
        pass

def plot_view_performes(res_dict,title_name,savepath):
    #绘制性能指标的变化
    # fig = plt.figure(figsize=(12, 4.5))
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 2)
    # plt.suptitle(title_name)

    para_range = list(res_dict.keys())
    performes=list(res_dict[para_range[0]].keys())
    for index, p in enumerate(performes):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        para=[round_sig(res_dict[i][p],3) for i in para_range]
        ax.plot(para_range, para, color='gray', alpha=0.8)
        formatter = ScalarFormatter(useOffset=False,useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))  # 强制所有值用科学计数法
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(p)
    # plt.subplots_adjust(wspace=0.8, hspace=0.5, right=0.95, left=0.15, top=0.95, bottom=0.1)
    plt.tight_layout()
    if savepath!='none':
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_embedded_2d(embedded, ax):
    sc = ax.scatter(embedded[0:-1:6, 0], embedded[0:-1:6, 1], c=np.arange(60), cmap='hsv', s=0.5)
    x_mid = (embedded[:, 0].max() + embedded[:, 0].min()) / 2
    y_mid = (embedded[:, 1].max() + embedded[:, 1].min()) / 2
    radius = max(np.ptp(embedded[:, 0]), np.ptp(embedded[:, 1])) / 2 * 1.1
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_rdm_no_box(ax, geodesic_distances):

    heatmap=sns.heatmap(geodesic_distances, ax=ax, cmap='coolwarm', cbar=False, vmin=0, vmax=np.max(geodesic_distances))
    # cbar = heatmap.collections[0].colorbar  # 获取 heatmap 对应的 colorbar
    # cbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
    # cbar.ax.tick_params(labelsize=7)  # 设置刻度字体大小为10
    # cbar.ax.yaxis.set_tick_params(labelsize=7)
    #
    # vmin = np.min(geodesic_distances)
    # vmax = np.max(geodesic_distances)
    # cbar.set_ticks([vmin, vmax])
    # cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])  # 可根据需要保留小数点
    # font_prop = font_manager.FontProperties(family='Arial', size=7)
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontproperties(font_prop)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.yaxis.tick_right()
    # ax.set_yticks([20,340])
    # ax.tick_params(axis='y', length=0)  # length=0 隐藏刻度线
    # ax.set_yticklabels([ f'{np.max(geodesic_distances):.1f}',0],rotation=0)
    # plt.yticks(fontsize=7, fontname='Arial', fontweight='normal')

def click_plot_class(Energy,Global_distance,Energy_f,Global_distance_f):
    # 可视化散点图
    fig = plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(4, 7)
    ax = fig.add_subplot(gs[0:4, 0:4])
    sc_all = ax.scatter(Energy, Global_distance, color='black', alpha=0.2, s=2, label='All')
    sc_filtered = ax.scatter(Energy_f, Global_distance_f, color='red', alpha=0.5, s=5, label='Filtered')
    ax.set_xlabel("Energy")
    ax.set_ylabel("Global distance")
    ax.set_title("Solutions")
    # 用于记录点击显示
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    def on_click(event):
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        coords = np.array(list(zip(Energy, Global_distance)))  # 仅标红色点
        distances = np.sqrt((coords[:, 0] - x_click) ** 2 + (coords[:, 1] - y_click) ** 2)
        min_idx = np.argmin(distances)
        if distances[min_idx] < 0.01:  # 距离阈值，防止误点
            x_sel, y_sel = coords[min_idx]
            annot.xy = (x_sel, y_sel)
            annot.set_text(f"Index: {min_idx}")
            annot.set_visible(True)
            fig.canvas.draw_idle()
            print(f"Clicked point index: {min_idx}")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

def click_plot_view(Energy,Global_distance,Energy_f,Global_distance_f):
    # 可视化散点图
    fig = plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(4, 7)
    ax = fig.add_subplot(gs[0:4, 0:4])
    sc_all = ax.scatter(Energy, Global_distance, color='black', alpha=0.2, s=2, label='All')
    sc_filtered = ax.scatter(Energy_f, Global_distance_f, color='red', alpha=0.5, s=5, label='Filtered')
    ax.set_xlabel("Energy")
    ax.set_ylabel("Global distance")
    ax.set_title("Solutions")
    # 用于记录点击显示
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    def on_click(event):
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        coords = np.array(list(zip(Energy_f, Global_distance_f)))  # 仅标红色点
        distances = np.sqrt((coords[:, 0] - x_click) ** 2 + (coords[:, 1] - y_click) ** 2)
        min_idx = np.argmin(distances)
        if distances[min_idx] < 0.01:  # 距离阈值，防止误点
            x_sel, y_sel = coords[min_idx]
            annot.xy = (x_sel, y_sel)
            annot.set_text(f"Index: {min_idx}")
            annot.set_visible(True)
            fig.canvas.draw_idle()
            print(f"Clicked point index: {min_idx}")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
