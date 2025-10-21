import os
import pickle
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import h5py
from matplotlib import gridspec

#加载数据###############################################################################
neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class = pickle.load(f)  # 获取全部神经元响应

fontsize_ = 8
linewidth_ax = 0.3
linewidth_plot = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'

fig = plt.figure(figsize=(4.2, 0.6))
gs = gridspec.GridSpec(1, 4)   # 1 行 4 列
neu_list = ['ento_exp15_ch20', 'ni_exp6_ch4', 'ni_exp10_ch3', 'ento_exp10_ch7']
for i, neu in enumerate(neu_list):
    resp = dict_class[neu]['class']['resp']
    x = np.arange(1, len(resp)+1)
    ax = fig.add_subplot(gs[0, i])
    ax.plot(x, resp, '-', markersize=2.5, linewidth=linewidth_plot, color='k')
    # 不显示任何坐标轴或刻度
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # 坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(wspace=0.4, hspace=0.7, right=0.95, left=0.15, top=0.90, bottom=0.10)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure2', 'figure2_1.eps'), dpi=600, format='eps')

#绘制神经元发放的拟合程度############################################################################################################################
rename_dict = {
    'Color': 'color_features_pca',
    'Shape': 'shape_features_pca',
    'Texture': 'texture_features_pca',
    'V1-like': 'v1like_features_pca',
    'V2-like': 'v2like_features_pca',
    'Alexnet Conv1': 'alexnet_features.2_pca',
    'Alexnet Conv2': 'alexnet_features.5_pca',
    'Alexnet Conv3': 'alexnet_features.7_pca',
    'Alexnet Conv4': 'alexnet_features.9_pca',
    'Alexnet Conv5': 'alexnet_features.12_pca',
}

color_map = {
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
fig = plt.figure(figsize=(3.8, 5.8))
gs = gridspec.GridSpec(len(rename_dict), 2)   # 1 行 4 列
for iindex,feat in enumerate(rename_dict.keys()):
    ax = fig.add_subplot(gs[iindex, 1])
    dict_mertics={}
    for neu in dict_class.keys():
        dict_mertics[neu]=dict_class[neu]['fit_corr'][feat]
    # 从低到高对字典进行排序
    max_key = max(dict_mertics, key=dict_mertics.get)
    keys_in_order = list(dict_mertics.keys())
    pos_insert = keys_in_order.index(max_key)  # 0-based
    print(feat+':'+str(round(dict_mertics[max_key],2)))
    rootpath = r'E:\Image_paper\Project\Acute_Image\results\neuron_mapping'
    expname = max_key.split('ch')[0][:-1]
    with open(os.path.join(rootpath, expname, max_key + '.pkl'), "rb") as f:
        neuron = pickle.load(f)  # 图像特征字典，每张图像是一个键
    Y_test=neuron['decode'][rename_dict[feat]]['class']['mean']['svr']['Y_test']
    Y_test=(Y_test-np.min(Y_test))/(np.max(Y_test)-np.min(Y_test))
    Y_pred=neuron['decode'][rename_dict[feat]]['class']['mean']['svr']['Y_pred']
    Y_pred = (Y_pred - np.min(Y_pred)) / (np.max(Y_pred) - np.min(Y_pred))
    ax.plot(Y_test, '-', markersize=2.5, linewidth=linewidth_plot, color='k')
    ax.plot(Y_pred, '-', markersize=2.5, linewidth=linewidth_plot, color='r')
    val = float(dict_mertics[max_key])  # 相关性
    title_txt = f"{feat}: r={val:.2f}"
    # ax.set_title(title_txt, fontsize=fontsize_ , fontname=fontname_, fontweight=fontweight_, pad=5)
    ax.set_yticks([0, 1])
    ax.set_yticks([])
    if iindex ==9 :
        ax.set_xticks([0, 200])
        ax.set_xlabel('Stimulus index', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    else:
        ax.set_xticks([])
    if iindex==4:
        ax.set_ylabel('Response Magnifude', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_, labelpad=3)
    else:
        ax.set_ylabel(' ')
    ax.text(
        0.7, 1.2, f"r = {val:.2f}",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,
        color='k'
        # 可选边框：
        # , bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', lw=0.5, alpha=0.9)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

# fig = plt.figure(figsize=(1.7, 5.5))
# gs = gridspec.GridSpec(len(rename_dict), 1)   # 1 行 4 列
bins = np.linspace(-0.0, 0.7, 30)  # [-1,1] 间等分成8个bin
for iindex,feat in enumerate(rename_dict.keys()):
    ax = fig.add_subplot(gs[iindex, 0])
    fit_corr=[dict_class[neu]['fit_corr'][feat] for neu in dict_class.keys()]
    ax.hist(fit_corr, bins=bins, color=color_map[feat])  # 可改 bins
    # ax.set_xlabel(feat, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # ax.set_ylabel('Count', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    med = float(np.median(fit_corr))

    ax.axvline(med, linestyle='--', linewidth=0.7, color='k')
    ymax = ax.get_ylim()[1]
    ax.set_xlim([-0.0,0.7])
    ax.set_yticks([0, 20])
    ax.tick_params(axis='both', which='both',
                   direction='in', length=3, width=0.6)  # 刻度与标签间距
    ax.spines['left'].set_position(('outward', 6))  # 像素为单位，正值向外移
    label = feat.replace(' ', '\n', 1)
    ax.set_ylabel(label, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.text(
        0.45, 1.2, f"median = {med:.2f}",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,
        color='k'
        # 可选边框：
        # , bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', lw=0.5, alpha=0.9)
    )
    if iindex < 9:
        ax.set_xlabel('')
        ax.set_xticks([])
    else:
        ax.set_xticks([0,0.2,0.4,0.6])
        ax.set_xlabel('Correlation (r)',fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

plt.subplots_adjust(wspace=0.1, hspace=0.35, right=0.95, left=0.20, top=0.90, bottom=0.10)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure2', 'figure2_6.eps'), dpi=600, format='eps')


#绘制拟合度的分布######################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# ========= 依赖的外部对象 =========
use_color_map = 'color_map' in globals()
feat_names = list(rename_dict.keys())
fit_corr_ary = []
for feat in feat_names:
    vals = [dict_class[neu]['fit_corr'][feat] for neu in dict_class.keys()]
    fit_corr_ary.append(vals)
fit_corr_ary = np.array(fit_corr_ary, dtype=float)
# ========= 统计量：mean & SEM（忽略 NaN）=========
means  = np.nanmean(fit_corr_ary, axis=1)
counts = np.sum(~np.isnan(fit_corr_ary), axis=1)
sd     = np.nanstd(fit_corr_ary, axis=1, ddof=1)
sem    = np.divide(sd, np.sqrt(np.maximum(counts, 1)),
                   out=np.zeros_like(sd), where=counts > 1)
# ========= 画图（横向柱状图；x 轴在上）=========
fig = plt.figure(figsize=(2.7, 2.8))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(feat_names))
colors = [color_map.get(fn, '0.65') for fn in feat_names]
bars = ax.barh(y, means,height=0.8, color=colors, edgecolor='k', linewidth=0.5, zorder=2)
ax.errorbar(means, y, xerr=sem, fmt='none', ecolor='k', elinewidth=0.7, capsize=2, zorder=3)

# 在柱子中间写特征名
for bar, label in zip(bars, feat_names):
    xx = bar.get_x() + bar.get_width() / 2.0
    yy = bar.get_y() + bar.get_height() / 2.0
    ax.text(xx, yy, label, ha='center', va='center',
            color='white', fontsize=fontsize_-1, fontname=fontname_, fontweight=fontweight_)
# 轴标签与样式
ax.set_xlabel('Fit corr. (mean ± SEM)', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_,labelpad=8)  # 先设在下方，马上移动到上方
ax.set_yticks([])  # 隐藏 y 轴刻度（名字已写在柱内）
xmax = np.nanmax(means + sem) if np.isfinite(np.nanmax(means + sem)) else 1.0
ax.set_xlim(0, xmax * 1.08)
ax.invert_yaxis()
ax.xaxis.tick_bottom()
ax.xaxis.set_label_position('bottom')
ax.tick_params(axis='x', bottom=True, top=False)   # 开底部刻度，关顶部刻度

ax.spines['bottom'].set_linewidth(linewidth_ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(linewidth_ax)
plt.xticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.yticks(fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
plt.subplots_adjust(right=0.95, left=0.20, top=0.9, bottom=0.2)
save_path = os.path.join(r'E:\Image_paper\Project\plot', 'figure2', 'figure2_7.eps')
plt.savefig(save_path, dpi=600, format='eps')
# plt.show()


#绘制神经元rank排序############################################################################################################################
feat='Color'
dict_mertics={}
for neu in dict_class.keys():
    dict_mertics[neu]=dict_class[neu]['fit_corr'][feat]
# 从低到高对字典进行排序
max_key = max(dict_mertics, key=dict_mertics.get)
resp=dict_class[max_key]['class']['resp']
name=dict_class[max_key]['class']['name']
import numpy as np
from pathlib import Path
from PIL import Image

# ====== 1) 选出前10/后10的索引、名称 ======
resp = np.asarray(resp, dtype=float)
name = np.asarray(name)              # 与 resp 一一对应
assert len(resp) == len(name)

valid = np.isfinite(resp)
idx_all = np.where(valid)[0]
resp_v = resp[valid]
name_v = name[valid]

k = min(10, resp_v.size)

# Top-10（按值降序）
idx_top_unsorted = np.argpartition(resp_v, -k)[-k:]
idx_top_sorted_local = idx_top_unsorted[np.argsort(resp_v[idx_top_unsorted])[::-1]]
idx_top = idx_all[idx_top_sorted_local]
top_names = list(name[idx_top])

# Bottom-10（按值升序）
idx_bot_unsorted = np.argpartition(resp_v, k)[:k]
idx_bot_sorted_local = idx_bot_unsorted[np.argsort(resp_v[idx_bot_unsorted])]
idx_bot = idx_all[idx_bot_sorted_local]
bot_names = list(name[idx_bot])

print("Top-10 names:", top_names)
print("Bottom-10 names:", bot_names)

# ====== 2) 根据 name 保存原图到 plot 目录（JPG、按序号命名） ======
img_root = Path(r"E:\Image_paper\Project\Acute_Image\image\img_rgb")  # 你的图像库根目录
out_root = Path(r"E:\Image_paper\Project\plot")
out_top = out_root / "top10"
out_bot = out_root / "bottom10"
out_top.mkdir(parents=True, exist_ok=True)
out_bot.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 建索引：文件名(不含扩展名) -> 完整路径
print("Indexing images under:", img_root)
stem2path = {}
for p in img_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
        stem2path[p.stem] = p  # 若同名多处，可自定义优先级策略

def save_by_names(names, out_dir, start_idx=1, prefix=""):
    missing = []
    idx = start_idx
    for nm in names:
        p = stem2path.get(Path(nm).stem)  # 防止 nm 已带扩展名
        if p is None:
            missing.append(nm)
            continue
        try:
            img = Image.open(p)
            # 统一转为 RGB，避免 PNG 透明通道写 JPG 报错
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")
            out_path = out_dir / f"{prefix}{idx:04d}.jpg"
            img.save(out_path, format="JPEG", quality=95, optimize=True)
            idx += 1
        except Exception as e:
            print(f"[ERROR] save {p} -> {e}")
            missing.append(nm)
    return missing, idx

# 保存 Top-10 与 Bottom-10
miss_top, next_idx = save_by_names(top_names, out_top, start_idx=1,  prefix="top_")
miss_bot, _        = save_by_names(bot_names, out_bot, start_idx=1,  prefix="bot_")

if miss_top:
    print("Missing in top10:", miss_top)
if miss_bot:
    print("Missing in bottom10:", miss_bot)
print("Done. Saved to:", out_top, "and", out_bot)
