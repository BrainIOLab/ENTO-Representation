
def cal_class_performs(resp_split, label_fine):
    #根据排序获取神经元的响应矩阵，也从低到高进行排序
    resp_class=[dict_class[neu]['class']['resp'] for neu in sorted_dict.keys()]
    resp_class=np.array(resp_class)
    name_class=dict_class['ento_exp15_ch20']['class']['name']
    label_fine=[i.split('_')[2] for i in name_class]
    class_fine = list(dict.fromkeys(label_fine))

    n_neuron=100        #小批量神经元的个数
    dict_performs={}

    for nstart in range(0,resp_class.shape[0]-n_neuron+1):
        dict_performs[nstart]={}
        resp_split = resp_class[nstart:nstart + n_neuron, :]
        resp_split=np.transpose(resp_split)
        #解码准确率
        model = SVC(kernel='rbf', decision_function_shape='ovr')        #linear
        cv = StratifiedKFold(n_splits=5)  # 创建交叉验证
        y_pred = cross_val_predict(model, resp_split, label_fine, cv=cv)
        accuracy = accuracy_score(label_fine, y_pred)  # 准确率
        print(nstart,accuracy)
        #降维及rdm矩阵
        X_reduced, rdm = analyse_reducedim_2d(resp_split, 'MDS')
        #类内距离、类间距离、分离比
        dist_intra, dist_inter=compute_class_distance(rdm, label_fine)
        Separability = dist_inter / (dist_intra + 1e-9)  # 计算类分离比
        dict_performs[nstart]['Accuracy']=accuracy
        dict_performs[nstart]['Intra-class dist.'] = dist_intra
        dict_performs[nstart]['Inter-class dist.'] = dist_inter
        dict_performs[nstart]['Separability'] = Separability
    return dict_performs




fig = plt.figure(figsize=(3, 3))
gs = gridspec.GridSpec(2, 3)
keys = list(dict_rdm.keys())
for idx, key in enumerate(keys):

    rdm=dict_rdm[key]
    r, c = divmod(idx, 3)
    ax = fig.add_subplot(gs[r, c])
    im = ax.imshow(rdm, cmap='viridis', aspect='equal', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize_)
    # ax.set_xlabel('Images', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    # ax.set_ylabel('Images', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    idx_by_class = [[i for i, lab in enumerate(label_fine) if lab == c] for c in class_fine]
    counts = [len(block) for block in idx_by_class]
    starts = np.cumsum([0] + counts[:-1])
    centers = [s + (n / 2) - 0.5 for s, n in zip(starts, counts)]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    # 类别分割线（细线）
    for b in np.cumsum(counts)[:-1]:
        ax.axhline(b - 0.5, color='w', linewidth=0.6, alpha=0.8)
        ax.axvline(b - 0.5, color='w', linewidth=0.6, alpha=0.8)
    ax.set_xticklabels(class_fine, rotation=90, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_yticklabels(class_fine, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.tick_params(labelsize=fontsize_)
    plt.subplots_adjust(wspace=0.35, hspace=0.3, right=0.90, left=0.25, top=0.95, bottom=0.25)
    plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure3n', 'rdm.eps'), dpi=600, format='eps')



# 假设 dict_rdm 恰好有 6 个键
keys = list(dict_rdm.keys())[:6]  # 若多于6个只取前6个；也可改为你想要的顺序

fig = plt.figure(figsize=(8, 5))  # 2x3 画布
gs = gridspec.GridSpec(2, 3)

for idx, key in enumerate(keys):
    r, c = divmod(idx, 3)
    ax = fig.add_subplot(gs[r, c])

    rdm = rdm_neurons
    im = ax.imshow(rdm, cmap='viridis', aspect='equal', interpolation='nearest')

    # 独立colorbar（紧凑）
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize_)

    # 类别分块线与刻度
    idx_by_class = [[i for i, lab in enumerate(label_fine) if lab == c_] for c_ in class_fine]
    counts = [len(block) for block in idx_by_class]
    starts = np.cumsum([0] + counts[:-1])
    centers = [s + (n / 2) - 0.5 for s, n in zip(starts, counts)]

    ax.set_xticks(centers)
    ax.set_yticks(centers)

    for b in np.cumsum(counts)[:-1]:
        ax.axhline(b - 0.5, color='w', linewidth=0.6, alpha=0.8)
        ax.axvline(b - 0.5, color='w', linewidth=0.6, alpha=0.8)

    # 仅底行显示x刻度标签，左列显示y刻度标签，避免过挤
    if r == 1:
        ax.set_xticklabels(class_fine, rotation=90,
                           fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    else:
        ax.set_xticklabels([])

    if c == 0:
        ax.set_yticklabels(class_fine,
                           fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    else:
        ax.set_yticklabels([])

    # 轴样式与标题
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.tick_params(labelsize=fontsize_)
    ax.set_title('Neurons', fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

# 统一排版边距
plt.subplots_adjust(wspace=0.35, hspace=0.35, right=0.90, left=0.15, top=0.93, bottom=0.20)
# 保存（可选）
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot','figure3n','rdm_grid_2x3_neurons.eps'), dpi=600, format='eps')
# plt.show()


