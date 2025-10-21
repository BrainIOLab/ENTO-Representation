import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math

import numpy as np
from collections import defaultdict

def build_balanced_batches(dict_view, obj, per_bin=2, repeats=100, seed=None):
    """
    返回长度=repeats的列表，每个元素是长度=18*per_bin的神经元ID列表。
    策略：
      1) 优先不放回；
      2) bin 数量不足时，仅对该 bin 允许放回；
      3) bin 为 0 时，从最近的相邻 bin 借（环状距离），必要时跨多层邻居；
    仍严格输出每个中心各 per_bin 个，总数 18*per_bin。
    """
    rng = np.random.default_rng(seed)
    target_bins = list(range(18))

    # 分桶
    buckets = defaultdict(list)
    for neu, d in dict_view.items():
        if obj not in d:
            continue
        entry = d[obj]
        resp = entry.get('resp', None)
        tc   = entry.get('metrics', {}).get('Tuning center', None)
        if isinstance(tc, (int, np.integer)) and resp is not None and np.asarray(resp).shape == (18,):
            if 0 <= int(tc) <= 17:
                buckets[int(tc)].append(neu)

    # 预计算每个bin的“邻居借样池”（环状最近→更远）
    # 对于空bin，借样顺序：±1, ±2, ...（mod 18）
    borrow_order = {b: [] for b in target_bins}
    for b in target_bins:
        order = []
        for k in range(1, 9):  # 最多扩到 9 步足够覆盖全环
            order.append((b - k) % 18)
            order.append((b + k) % 18)
        borrow_order[b] = order

    neu_batches = []
    for _ in range(repeats):
        selected = []
        for b in target_bins:
            pool = buckets[b]

            if len(pool) >= per_bin:
                # 足够：不放回
                choices = rng.choice(pool, size=per_bin, replace=False)
                selected.extend(choices.tolist())
            elif len(pool) > 0:
                # 仅 1 个：第一个不放回，第二个允许放回
                first = rng.choice(pool, size=1, replace=False).tolist()
                second = rng.choice(pool, size=per_bin-1, replace=True).tolist()
                selected.extend(first + second)
            else:
                # 为 0：从邻居借
                borrowed = []
                for nb in borrow_order[b]:
                    if len(buckets[nb]) > 0:
                        need = per_bin - len(borrowed)
                        take = min(need, len(buckets[nb]))
                        borrowed.extend(rng.choice(buckets[nb], size=take, replace=False).tolist())
                        if len(borrowed) == per_bin:
                            break
                # 如果还不够（极端情况），最后允许从所有非空bin里放回补齐
                if len(borrowed) < per_bin:
                    all_pool = np.concatenate([np.array(buckets[nb]) for nb in target_bins if len(buckets[nb])>0])
                    extra = rng.choice(all_pool, size=per_bin-len(borrowed), replace=True).tolist()
                    borrowed.extend(extra)
                selected.extend(borrowed)

        rng.shuffle(selected)
        assert len(selected) == 18 * per_bin
        neu_batches.append(selected)

    return neu_batches

def analyse_reducedim(resp_class, reducedim_method, n_components):
    if reducedim_method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)  # 降维到2维
        X_reduced = pca.fit_transform(resp_class)
    if reducedim_method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components)  # 降维到2维
        X_reduced = tsne.fit_transform(resp_class)
    if reducedim_method == 'MDS':
        from sklearn.manifold import MDS
        mds = MDS(n_components=n_components)  # 降维到2维
        X_reduced = mds.fit_transform(resp_class)
    if reducedim_method == 'Isomap':
        from sklearn.manifold import Isomap
        isomap = Isomap(n_components=n_components)  # 降维到2维
        X_reduced = isomap.fit_transform(resp_class)
    from sklearn.metrics import pairwise_distances
    rdm = pairwise_distances(X_reduced, metric='euclidean')
    return X_reduced, rdm

def _procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    经典 Procrustes：去中心、单位化、最优正交对齐后残差范数。
    返回归一化残差（越小越相似）。
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc) + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc) + 1e-12)
    # 最优旋转
    U, _, Vt = np.linalg.svd(Xn.T @ Yn, full_matrices=False)
    R = U @ Vt
    X_aligned = Xn @ R
    resid = np.linalg.norm(X_aligned - Yn)
    return float(resid)

def metric_circularity_procrustes(X_reduced: np.ndarray):
    X = np.asarray(X_reduced, float)
    # 与理想等间隔单位圆的 Procrustes 距离
    n = X.shape[0]
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n,2)
    proc = _procrustes_distance(X, circle)
    return   proc

def compute_global_dist(RDM):
    n = RDM.shape[0]
    discriminability = np.sum(RDM) / (n * n - n)  # 1. 全局区分度（去对角线）
    return discriminability


# 加载神经元响应
neuron_path = r'E:\Image_paper\Project\dict_view.pkl'
with open(neuron_path, "rb") as f:
    dict_view= pickle.load(f)  # 获取全部神经元响应

neuron_path = r'E:\Image_paper\Project\dict_class1.pkl'
with open(neuron_path, "rb") as f:
    dict_class= pickle.load(f)  # 获取全部神经元响应

dict_performs={}
obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']
feat_list=['Color','Shape','Texture', 'Alexnet Conv1', 'Alexnet Conv3', 'Alexnet Conv5']
for obj in obj_list:

    per_bin = 2
    repeats = 100
    neu_batches = build_balanced_batches(dict_view, obj, per_bin=per_bin, repeats=repeats, seed=42)

    ##
    reducedim_method='MDS'
    n_components=2

    dict_performs[obj] = {}
    for idx,neulist in enumerate(neu_batches):
        Modulation_depth = [dict_view[neu][obj]['metrics']['Modulation depth'] for neu in neulist]
        Neighbor_correlation = [dict_view[neu][obj]['metrics']['Neighbor correlation'] for neu in neulist]

        resp_split = [dict_view[neu][obj]['resp'] for neu in neulist]
        resp_split = np.transpose(resp_split)
        # 性能指标
        X_reduced, rdm = analyse_reducedim(resp_split, reducedim_method, n_components)
        # 计算与标准圆的procrustes_distance
        pd = metric_circularity_procrustes(X_reduced)
        # 计算全局平均距离
        global_distance = compute_global_dist(rdm)
        dict_performs[obj][idx]={}
        dict_performs[obj][idx]['Modulation_depth_mean']=np.mean(Modulation_depth)
        dict_performs[obj][idx]['Modulation_depth_median'] = np.median(Modulation_depth)
        dict_performs[obj][idx]['Neighbor_correlation_mean']=np.mean(Neighbor_correlation)
        dict_performs[obj][idx]['Neighbor_correlation_median'] = np.median(Neighbor_correlation)
        dict_performs[obj][idx]['PCD']=pd
        dict_performs[obj][idx]['GD']=global_distance

        for ft in feat_list:
            tempa=[dict_class[neu]['fit_corr'][ft] for neu in neulist]
            dict_performs[obj][idx][ft] = np.mean(tempa)

        print(obj, idx, pd, global_distance)




import numpy as np

# ===== 组装数据：X (n_samples, n_features) & y =====
def collect_matrix(dict_performs, obj_list, feat_list, y_key):
    rows_X, rows_y = [], []
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            try:
                # 取 6 个特征
                x = [float(rec[ft]) for ft in feat_list]
                if not np.all(np.isfinite(x)):
                    continue
                y = float(rec[y_key])
                if not np.isfinite(y):
                    continue
                rows_X.append(x)
                rows_y.append(y)
            except Exception:
                pass
    if len(rows_X) < 10:
        raise RuntimeError(f"{y_key}: 有效样本过少。")
    X = np.array(rows_X, float)
    y = np.array(rows_y, float)
    return X, y  # (n, 6), (n,)

def zscore_colwise(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd

def partial_r2_all(ZX, Zy):
    """
    计算每个特征在控制其它特征后的 partial R^2。
    ZX: 标准化后的自变量 (n, p)
    Zy: 标准化后的因变量 (n,)
    返回：p 维向量
    """
    n, p = ZX.shape
    pr2 = np.zeros(p, float)
    # 预先加常数项
    one = np.ones((n, 1), float)
    for j in range(p):
        # X_-j
        mask = np.ones(p, bool); mask[j] = False
        Z_minus = np.hstack([one, ZX[:, mask]])
        # y 对 X_-j 的残差
        by = np.linalg.pinv(Z_minus) @ Zy
        y_res = Zy - Z_minus @ by
        # x_j 对 X_-j 的残差
        bx = np.linalg.pinv(Z_minus) @ ZX[:, j]
        x_res = ZX[:, j] - Z_minus @ bx
        # 偏相关平方
        num = np.nansum((y_res - y_res.mean())*(x_res - x_res.mean()))
        den = np.sqrt(np.nansum((y_res - y_res.mean())**2) * np.nansum((x_res - x_res.mean())**2)) + 1e-12
        r = num / den
        pr2[j] = r**2
    return pr2

def fit_beta_and_pr2(X, y, B=1000, seed=0):
    """
    标准化 OLS：返回带符号 β、partial R^2 以及二者的 95% bootstrap CI
    """
    n, p = X.shape
    ZX = zscore_colwise(X)
    Zy = zscore_colwise(y.reshape(-1,1)).ravel()

    # 带常数 OLS
    Xmat = np.hstack([np.ones((n,1), float), ZX])  # [1, Z1..Zp]
    beta = np.linalg.pinv(Xmat) @ Zy               # b0, b1..bp
    betas = beta[1:].astype(float)                 # 取各特征的 β（带符号）

    pr2 = partial_r2_all(ZX, Zy)

    # bootstrap
    rng = np.random.default_rng(seed)
    beta_boot = np.zeros((B, p), float)
    pr2_boot  = np.zeros((B, p), float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ZX_b, Zy_b = ZX[idx], Zy[idx]
        Xm = np.hstack([np.ones((n,1), float), ZX_b])
        btmp = np.linalg.pinv(Xm) @ Zy_b
        beta_boot[b] = btmp[1:]
        pr2_boot[b]  = partial_r2_all(ZX_b, Zy_b)

    # 95% CI
    beta_ci = np.percentile(beta_boot, [2.5, 97.5], axis=0)     # shape (2, p)
    pr2_ci  = np.percentile(pr2_boot,  [2.5, 97.5], axis=0)

    # 打包
    return {
        'beta': betas,              # (p,)
        'beta_ci': beta_ci,         # (2,p) -> [lo, hi] per feature
        'beta_boot': beta_boot,     # (B,p)
        'partial_r2': pr2,          # (p,)
        'partial_r2_ci': pr2_ci,    # (2,p)
        'partial_r2_boot': pr2_boot # (B,p)
    }

# ====== 从 dict_performs 取出矩阵，分别拟合 PCD 与 GD ======
X_pcd, y_pcd = collect_matrix(dict_performs, obj_list, feat_list, y_key='PCD')
X_gd,  y_gd  = collect_matrix(dict_performs, obj_list, feat_list, y_key='GD')

# 可调：bootstrap 轮数
B = 1000

res_pcd_feat = fit_beta_and_pr2(X_pcd, y_pcd, B=B, seed=0)
res_gd_feat  = fit_beta_and_pr2(X_gd,  y_gd,  B=B, seed=1)

# ====== 打印一个简表核对 ======
def pretty_arr(a): return ", ".join([f"{v:.3f}" for v in a])

print("Features:", feat_list)
print("\n=== PCD: signed β ===")
print(pretty_arr(res_pcd_feat['beta']))
print("95% CI lo:", pretty_arr(res_pcd_feat['beta_ci'][0]))
print("95% CI hi:", pretty_arr(res_pcd_feat['beta_ci'][1]))
print("Partial R^2:", pretty_arr(res_pcd_feat['partial_r2']))
print("Partial R^2 95% CI lo:", pretty_arr(res_pcd_feat['partial_r2_ci'][0]))
print("Partial R^2 95% CI hi:", pretty_arr(res_pcd_feat['partial_r2_ci'][1]))

print("\n=== GD: signed β ===")
print(pretty_arr(res_gd_feat['beta']))
print("95% CI lo:", pretty_arr(res_gd_feat['beta_ci'][0]))
print("95% CI hi:", pretty_arr(res_gd_feat['beta_ci'][1]))
print("Partial R^2:", pretty_arr(res_gd_feat['partial_r2']))
print("Partial R^2 95% CI lo:", pretty_arr(res_gd_feat['partial_r2_ci'][0]))
print("Partial R^2 95% CI hi:", pretty_arr(res_gd_feat['partial_r2_ci'][1]))



# ========= 统计检验（β 的 t 检验；partial R^2 的 F 检验） =========
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def zscore_colwise(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd

def ols_beta_se_p(X, y):
    X = zscore_colwise(X)
    y = zscore_colwise(y.reshape(-1,1)).ravel()
    n, p = X.shape
    Xd = np.hstack([np.ones((n,1)), X])
    pinv = np.linalg.pinv(Xd)
    beta_all = pinv @ y
    yhat = Xd @ beta_all
    resid = y - yhat
    df = max(n - (p+1), 1)
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * (pinv @ pinv.T)
    se_all = np.sqrt(np.clip(np.diag(cov), 0, None))
    b = beta_all[1:]
    se = se_all[1:]
    t = b / (se + 1e-12)
    try:
        from scipy.stats import t as student_t
        p = 2 * (1 - student_t.cdf(np.abs(t), df=df))
    except Exception:
        from math import erf, sqrt
        def norm_cdf(z): return 0.5*(1+erf(z/np.sqrt(2)))
        p = 2 * (1 - norm_cdf(np.abs(t)))
    return b, se, t, p, df

def partial_r2_and_pF(X, y):
    X = zscore_colwise(X)
    y = zscore_colwise(y.reshape(-1,1)).ravel()
    n, p = X.shape
    one = np.ones((n,1))
    Xfull = np.hstack([one, X])
    beta_full = np.linalg.pinv(Xfull) @ y
    resid_full = y - Xfull @ beta_full
    RSS_full = float(resid_full @ resid_full)
    df2 = max(n - (p+1), 1)
    pr2 = np.zeros(p)
    pF  = np.zeros(p)
    for j in range(p):
        mask = np.ones(p, bool); mask[j] = False
        Xred = np.hstack([one, X[:, mask]])
        beta_red = np.linalg.pinv(Xred) @ y
        resid_red = y - Xred @ beta_red
        RSS_red = float(resid_red @ resid_red)
        F = ((RSS_red - RSS_full) / 1.0) / (RSS_full / df2 + 1e-12)
        pr2[j] = F / (F + df2) if F > 0 else 0.0
        try:
            from scipy.stats import f as fdist
            pF[j] = 1 - fdist.cdf(F, 1, df2)
        except Exception:
            # 粗略近似
            import math
            if pr2[j] > 0 and pr2[j] < 1:
                t2 = pr2[j]/(1-pr2[j]) * df2
                def norm_cdf(z): return 0.5*(1+math.erf(z/np.sqrt(2)))
                pF[j] = 2*(1-norm_cdf(np.sqrt(t2)))
            else:
                pF[j] = 1.0
    return pr2, pF

# 计算检验结果
b_pcd, se_pcd, t_pcd, p_t_pcd, df_pcd = ols_beta_se_p(X_pcd, y_pcd)
pr2_pcd_chk, pF_pcd = partial_r2_and_pF(X_pcd, y_pcd)

b_gd,  se_gd,  t_gd,  p_t_gd,  df_gd  = ols_beta_se_p(X_gd,  y_gd)
pr2_gd_chk,  pF_gd  = partial_r2_and_pF(X_gd,  y_gd)

# 用你已经算好的 CI & 点估计（res_*_feat），并配合 p 值做标注
labels = feat_list
colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

fontsize_ = 8
linewidth_ax = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'

def style_axis(ax):
    ax.tick_params(labelsize=fontsize_)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth_ax); ax.spines['bottom'].set_linewidth(linewidth_ax)

def stars(p):
    return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 5e-2 else ''))

def bar_with_ci_and_sig(ax, vals, ci, pvals, ylabel,flas_=True,
                        put_xaxis_at_zero=False, y_from_zero=False, colors=None):
    p = len(vals)
    xpos = np.arange(p)
    bars = ax.bar(xpos, vals, width=0.65, align='center',
                  color=colors, edgecolor='k', linewidth=0.6)
    centers = np.array([b.get_x() + b.get_width()/2 for b in bars])
    yerr = np.vstack([vals - ci[0], ci[1] - vals])
    ax.errorbar(centers, vals, yerr=yerr, fmt='none', ecolor='k', elinewidth=0.8, capsize=2)

    # 轴范围
    if put_xaxis_at_zero:
        lo = min(np.min(ci[0]), 0.0)
        hi = max(np.max(ci[1]), 0.0)
        pad = 0.08 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.spines['bottom'].set_position(('data', 0.0))
        ax.xaxis.set_ticks_position('bottom')
    elif y_from_zero:
        hi = max(0.0, float(np.max(ci[1])))
        ax.set_ylim(0.0, max(0.01, hi * 1.15))

    # 标注显著性星号
    for i, (x, v) in enumerate(zip(centers, vals)):
        s = stars(pvals[i])
        if s:
            if put_xaxis_at_zero:
                top = max(v, ci[1, i])
                bot = min(v, ci[0, i])
                if v >= 0:
                    ytxt = top + 0.03*(ax.get_ylim()[1] - ax.get_ylim()[0])
                    va = 'bottom'
                else:
                    ytxt = bot - 0.03*(ax.get_ylim()[1] - ax.get_ylim()[0])
                    va = 'top'
            else:
                top = max(v, ci[1, i])
                ytxt = top + 0.03*(ax.get_ylim()[1] - ax.get_ylim()[0])
                va = 'bottom'
            ax.text(x, ytxt, s, ha='center', va=va, fontsize=fontsize_+1, fontweight='bold')
    if flas_:
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize_, fontname=fontname_)
    else:
        ax.set_xticks(centers)
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    style_axis(ax); ax.margins(x=0.03)

# ========= 2×2 图：带显著性标注 =========
fig = plt.figure(figsize=(4, 4))
gs = gridspec.GridSpec(2, 2)

# (1,1) PCD: β（x 轴放到 y=0），显著性来自 t 检验 p 值
ax = fig.add_subplot(gs[0, 0])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_pcd_feat['beta']),
    ci=np.asarray(res_pcd_feat['beta_ci']),
    pvals=p_t_pcd,
    flas_=False,
    ylabel='β (PCD, signed)',
    put_xaxis_at_zero=True,
    colors=colors)

# (1,2) PCD: partial R²，显著性来自 F 检验 p 值
ax = fig.add_subplot(gs[1, 0])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_pcd_feat['partial_r2']),
    ci=np.asarray(res_pcd_feat['partial_r2_ci']),
    pvals=pF_pcd,
    ylabel='Partial $R^2$ (PCD)',
    y_from_zero=True,
    colors=colors)

# (2,1) GD: β（x 轴放到 y=0）
ax = fig.add_subplot(gs[0, 1])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_gd_feat['beta']),
    ci=np.asarray(res_gd_feat['beta_ci']),
    pvals=p_t_gd,
    flas_=False,
    ylabel='β (GD, signed)',
    put_xaxis_at_zero=True,
    colors=colors)

# (2,2) GD: partial R²
ax = fig.add_subplot(gs[1, 1])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_gd_feat['partial_r2']),
    ci=np.asarray(res_gd_feat['partial_r2_ci']),
    pvals=pF_gd,
    ylabel='Partial $R^2$ (GD)',
    y_from_zero=True,
    colors=colors)

plt.subplots_adjust(wspace=0.6, hspace=0.2, right=0.95, left=0.15, top=0.95, bottom=0.17)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'figure5-3.eps'), dpi=600,
            format='eps')
