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


dict_performs={}
obj_list=['view_elephant','view_faces','view_pigeon','view_beermug','view_cowboyhat','view_electricguitar']
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
        print(obj, idx, pd, global_distance)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ===== 样式 =====
fontsize_ = 8
linewidth_ax = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'

# 调色：每个 obj 一种颜色
uniq_objs = list(dict.fromkeys(obj_list))
palette = plt.cm.tab10(np.linspace(0, 1, max(6, len(uniq_objs))))
color_map = {o: palette[i % len(palette)] for i, o in enumerate(uniq_objs)}

def gather_xy(dict_performs, obj_list, x_key, y_key):
    """从 dict_performs 收集 (x, y, obj)，按 x 升序返回"""
    data = []
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            try:
                x = float(rec[x_key]); y = float(rec[y_key])
                if np.isfinite(x) and np.isfinite(y):
                    data.append((obj, x, y))
            except Exception:
                pass
    if not data:
        return np.array([]), np.array([]), []
    data_sorted = sorted(data, key=lambda t: t[1])
    objs = [t[0] for t in data_sorted]
    xs   = np.array([t[1] for t in data_sorted], float)
    ys   = np.array([t[2] for t in data_sorted], float)
    return xs, ys, objs

def plot_panel(ax, xs, ys, objs, xlabel, ylabel, add_trend=True):
    # 分对象散点
    for o in uniq_objs:
        mask = np.array([ob == o for ob in objs])
        if mask.any():
            ax.scatter(xs[mask], ys[mask], s=12, alpha=0.85, color=color_map[o], label=o)
    # 趋势线（整体一条，可选）
    if add_trend and xs.size >= 2:
        coef = np.polyfit(xs, ys, deg=1)
        xx = np.linspace(xs.min(), xs.max(), 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, lw=0.8, color='k', linestyle='--', alpha=0.7)

    # 样式
    xlabel=xlabel[:-5]
    ax.set_xlabel(xlabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    ax.tick_params(labelsize=fontsize_)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth_ax)
    ax.spines['left'].set_linewidth(linewidth_ax)

# ===== 2×2：第一行 Modulation_depth_mean；第二行 Neighbor_correlation_mean =====
panels = [
    ('Modulation_depth_mean', 'PCD', 'PCD (↓ better)'),
    ('Modulation_depth_mean', 'GD',  'GD (↑ better)'),
    ('Neighbor_correlation_mean', 'PCD', 'PCD (↓ better)'),
    ('Neighbor_correlation_mean', 'GD',  'GD (↑ better)'),
]
fig = plt.figure(figsize=(4.4, 4.2))
gs = gridspec.GridSpec(2, 2)
for i, (x_key, y_key, y_label) in enumerate(panels):
    r, c = divmod(i, 2)
    ax = fig.add_subplot(gs[r, c])
    xs, ys, objs = gather_xy(dict_performs, obj_list, x_key, y_key)
    plot_panel(ax, xs, ys, objs, xlabel=x_key.replace('_', ' '), ylabel=y_label, add_trend=True)

from matplotlib.lines import Line2D  # 用于全局图例的代理句柄
# ===== 共用一个 legend（底部）=====
legend_handles = [Line2D([0],[0], marker='o', linestyle='None', markersize=4.5,
                         markerfacecolor=color_map[o], markeredgecolor=color_map[o], label=o)
                  for o in uniq_objs]

# 放在底部正中；bbox_to_anchor 的 y 设为 -0.08 让图例在图框外下方
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.00),
           ncol=3, frameon=False, fontsize=fontsize_)
plt.subplots_adjust(wspace=0.4, hspace=0.3, right=0.95, left=0.15, top=0.95, bottom=0.2)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'figure5-1.eps'), dpi=600,
            format='eps')






import numpy as np

# ===== 收集散点数据（与上面绘图一致）=====
def collect_xy(dict_performs, obj_list, x_key, y_key):
    xs, ys = [], []
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            try:
                x = float(rec[x_key]); y = float(rec[y_key])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x); ys.append(y)
            except Exception:
                pass
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if xs.size < 3:
        raise RuntimeError(f"样本过少：{x_key} vs {y_key}")
    return xs, ys

# ===== 统计量：Pearson/Spearman + CI =====
def pearson_ci(x, y, alpha=0.05):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    r = (x*y).sum() / np.sqrt((x*x).sum() * (y*y).sum() + 1e-12)
    n = len(x)
    # p 值（t 检验）
    df = max(n-2, 1)
    t = r * np.sqrt(df / max(1e-12, 1-r*r))
    try:
        from scipy.stats import t as tdist
        p = 2*(1-tdist.cdf(abs(t), df=df))
    except Exception:
        # 正态近似
        from math import erf, sqrt
        def norm_cdf(z): return 0.5*(1+erf(z/np.sqrt(2)))
        p = 2*(1-norm_cdf(abs(t)))
    # Fisher z 置信区间
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1/np.sqrt(max(n-3, 1))
    from math import erf, sqrt
    def norm_ppf(q):
        # 简单近似的反正态分位（避免依赖）
        import math
        # Wichura-ish 近似略复杂；这里用 scipy 时会覆盖掉。fallback用二分也可。
        # 为稳妥，给一个小表近似；或直接用 1.96 对 95%CI：
        return 1.96 if abs(q-0.975)<1e-6 else 1.645  # 仅常用 95%/90% 情况
    # 为保险：alpha=0.05 -> zcrit=1.96
    zcrit = 1.96 if abs(alpha-0.05)<1e-6 else norm_ppf(1-alpha/2)
    lo = np.tanh(z - zcrit*se)
    hi = np.tanh(z + zcrit*se)
    return float(r), float(p), float(lo), float(hi)

def spearman_corr(x, y):
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        # 简单降级：秩变换 + pearson
        xr = np.argsort(np.argsort(x))
        yr = np.argsort(np.argsort(y))
        r, p, _, _ = pearson_ci(xr, yr)
        return float(r), float(p)

# ===== 偏相关（可选）：控制 z 后 x ~ y 的相关 =====
def partial_corr(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    # 线性回归残差
    Z = np.column_stack([np.ones_like(z), z])
    bx = np.linalg.pinv(Z) @ x; x_res = x - Z @ bx
    by = np.linalg.pinv(Z) @ y; y_res = y - Z @ by
    r, p, lo, hi = pearson_ci(x_res, y_res)
    return r, p, lo, hi

# ===== 组合并输出 =====
pairs = [
    ('Modulation_depth_mean',        'PCD', 'MD vs PCD'),
    ('Modulation_depth_mean',        'GD',  'MD vs GD'),
    ('Neighbor_correlation_mean',    'PCD', 'NC vs PCD'),
    ('Neighbor_correlation_mean',    'GD',  'NC vs GD'),
]

results = {}
for xk, yk, name in pairs:
    x, y = collect_xy(dict_performs, obj_list, xk, yk)
    r, p, lo, hi = pearson_ci(x, y)
    rho, p_s = spearman_corr(x, y)
    results[name] = {
        'pearson_r': r, 'p_value': p, 'ci95': (lo, hi),
        'spearman_rho': rho, 'p_value_spearman': p_s,
        'n': len(x)
    }



# （可选）偏相关：在另一个指标为协变量时
# 对 PCD：控制 NC 看 MD~PCD；控制 MD 看 NC~PCD
# 对 GD：同理
x_MD, y_PCD = collect_xy(dict_performs, obj_list, 'Modulation_depth_mean', 'PCD')
x_NC, _     = collect_xy(dict_performs, obj_list, 'Neighbor_correlation_mean', 'PCD')
pc_MD_PCD, p_MD_PCD, lo_MD_PCD, hi_MD_PCD = partial_corr(x_MD, y_PCD, x_NC)
pc_NC_PCD, p_NC_PCD, lo_NC_PCD, hi_NC_PCD = partial_corr(x_NC, y_PCD, x_MD)

x_MD, y_GD  = collect_xy(dict_performs, obj_list, 'Modulation_depth_mean', 'GD')
x_NC, _     = collect_xy(dict_performs, obj_list, 'Neighbor_correlation_mean', 'GD')
pc_MD_GD,  p_MD_GD,  lo_MD_GD,  hi_MD_GD  = partial_corr(x_MD, y_GD, x_NC)
pc_NC_GD,  p_NC_GD,  lo_NC_GD,  hi_NC_GD  = partial_corr(x_NC, y_GD, x_MD)

results_partial = {
    'PCD | control NC (MD~PCD)': {'r': pc_MD_PCD, 'p': p_MD_PCD, 'ci95': (lo_MD_PCD, hi_MD_PCD)},
    'PCD | control MD (NC~PCD)': {'r': pc_NC_PCD, 'p': p_NC_PCD, 'ci95': (lo_NC_PCD, hi_NC_PCD)},
    'GD  | control NC (MD~GD)' : {'r': pc_MD_GD,  'p': p_MD_GD,  'ci95': (lo_MD_GD,  hi_MD_GD)},
    'GD  | control MD (NC~GD)' : {'r': pc_NC_GD,  'p': p_NC_GD,  'ci95': (lo_NC_GD,  hi_NC_GD)},
}

# ---- 打印查看 ----
def pretty(d):
    for k, v in d.items():
        if 'pearson_r' in v:
            print(f"{k:>22s}: Pearson r={v['pearson_r']:.3f} (95% CI {v['ci95'][0]:.3f},{v['ci95'][1]:.3f}), "
                  f"p={v['p_value']:.3g};  Spearman ρ={v['spearman_rho']:.3f}, p={v['p_value_spearman']:.3g}, n={v['n']}")
        else:
            print(f"{k:>22s}: partial r={v['r']:.3f} (95% CI {v['ci95'][0]:.3f},{v['ci95'][1]:.3f}), p={v['p']:.3g}")

print("=== Correlations (pooled across objects) ===")
pretty(results)
print("\n=== Partial correlations (optional) ===")
pretty(results_partial)











import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# =========================
# 计算部分：生成 res_pcd / res_gd
# =========================

def zscore(v):
    v = np.asarray(v, float)
    mu = np.nanmean(v); sd = np.nanstd(v) + 1e-12
    return (v - mu) / sd

def collect_rows(dict_performs, obj_list, y_key):
    """收集 X1=Modulation_depth_mean, X2=Neighbor_correlation_mean, Y=y_key(PCD/GD)"""
    rows = []
    for obj in obj_list:
        if obj not in dict_performs:
            continue
        for _, rec in dict_performs[obj].items():
            try:
                md = float(rec['Modulation_depth_mean'])
                nc = float(rec['Neighbor_correlation_mean'])
                y  = float(rec[y_key])
                if np.isfinite(md) and np.isfinite(nc) and np.isfinite(y):
                    rows.append((md, nc, y))
            except Exception:
                pass
    if len(rows) < 10:
        raise RuntimeError(f"{y_key}: 有效样本过少。")
    return np.array(rows, float)

def partial_r2_core(x, y, z):
    """偏 R²：控制 z 后 x 对 y 的独立解释力"""
    Z = np.column_stack([np.ones_like(z), z])
    by = np.linalg.pinv(Z) @ y
    y_res = y - Z @ by
    bx = np.linalg.pinv(Z) @ x
    x_res = x - Z @ bx
    num = np.nansum((y_res - y_res.mean())*(x_res - x_res.mean()))
    den = np.sqrt(np.nansum((y_res - y_res.mean())**2) * np.nansum((x_res - x_res.mean())**2)) + 1e-12
    r = num / den
    return float(r**2)

def fit_and_bootstrap_beta_pr2(X1, X2, Y, B=1000, seed=0):
    """标准化 OLS 点估计 + bootstrap 95% CI（β 与 partial R²）"""
    Z1, Z2, Zy = zscore(X1), zscore(X2), zscore(Y)

    # β 点估计（闭式解）
    X = np.column_stack([np.ones_like(Z1), Z1, Z2])
    beta = np.linalg.pinv(X) @ Zy
    b1_hat, b2_hat = float(beta[1]), float(beta[2])

    # partial R² 点估计
    pr2_md_hat = partial_r2_core(Z1, Zy, Z2)
    pr2_nc_hat = partial_r2_core(Z2, Zy, Z1)

    # bootstrap
    rng = np.random.default_rng(seed)
    n = len(Z1)
    b1s, b2s, pr2_mds, pr2_ncs = [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        z1, z2, zy = Z1[idx], Z2[idx], Zy[idx]
        Xb = np.column_stack([np.ones_like(z1), z1, z2])
        bb = np.linalg.pinv(Xb) @ zy
        b1s.append(float(bb[1])); b2s.append(float(bb[2]))
        pr2_mds.append(partial_r2_core(z1, zy, z2))
        pr2_ncs.append(partial_r2_core(z2, zy, z1))

    b1s = np.array(b1s); b2s = np.array(b2s)
    pr2_mds = np.array(pr2_mds); pr2_ncs = np.array(pr2_ncs)

    # 95% CI
    b1_ci = np.percentile(b1s, [2.5, 97.5])
    b2_ci = np.percentile(b2s, [2.5, 97.5])
    pr2_md_ci = np.percentile(pr2_mds, [2.5, 97.5])
    pr2_nc_ci = np.percentile(pr2_ncs, [2.5, 97.5])

    return {
        'beta_point':  [b1_hat, b2_hat],          # [β_MD, β_NC]（signed）
        'beta_ci':     [b1_ci, b2_ci],            # [ (lo,hi)_MD, (lo,hi)_NC ]
        'pr2_point':   [pr2_md_hat, pr2_nc_hat],  # [partialR2_MD, partialR2_NC]
        'pr2_ci':      [pr2_md_ci, pr2_nc_ci],    # [ (lo,hi)_MD, (lo,hi)_NC ]
    }

# === 由 dict_performs / obj_list 计算 res_pcd 与 res_gd ===
# 这里假设你前面已经构建好了 dict_performs 和 obj_list
rows_pcd = collect_rows(dict_performs, obj_list, y_key='PCD')
rows_gd  = collect_rows(dict_performs, obj_list, y_key='GD')

res_pcd = fit_and_bootstrap_beta_pr2(rows_pcd[:,0], rows_pcd[:,1], rows_pcd[:,2], B=1000, seed=0)
res_gd  = fit_and_bootstrap_beta_pr2(rows_gd[:,0],  rows_gd[:,1],  rows_gd[:,2],  B=1000, seed=1)

# =========================
# 绘图部分（按你的样式）
# =========================

# ===== 样式 =====
fontsize_ = 8
linewidth_ax = 0.5
fontname_ = 'Arial'
fontweight_ = 'normal'
colors = ['#4C78A8','#F58518']  # [MD, NC]
labels = ['Modulation depth', 'Neighbor corr.']

def style_axis(ax):
    ax.tick_params(labelsize=fontsize_)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.spines['bottom'].set_linewidth(linewidth_ax)

def bar_with_ci(ax, vals, ci, ylabel, flas_=False, put_xaxis_at_zero=False, y_from_zero=False):
    """
    vals: [v1, v2], ci: [(lo1,hi1),(lo2,hi2)]
    put_xaxis_at_zero=True -> 把 x 轴放到 y=0（适合 β 有正负）
    y_from_zero=True -> y 轴从 0 起（适合 partial R²）
    """
    xpos = np.arange(len(vals))
    bars = ax.bar(xpos, vals, width=0.4, align='center',
                  color=colors, edgecolor='k', linewidth=0.6)
    centers = np.array([b.get_x() + b.get_width()/2 for b in bars])
    yerr = np.array([[vals[0]-ci[0][0], vals[1]-ci[1][0]],
                     [ci[0][1]-vals[0],  ci[1][1]-vals[1]]])
    ax.errorbar(centers, vals, yerr=yerr, fmt='none',
                ecolor='k', elinewidth=0.8, capsize=2)
    if flas_:
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize_, fontname=fontname_)
    else:
        ax.set_xticks(centers)
        ax.set_xticklabels([])

    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)

    # 纵轴范围 & x 轴位置
    if put_xaxis_at_zero:
        lo = min(ci[0][0], ci[1][0], 0.0)
        hi = max(ci[0][1], ci[1][1], 0.0)
        pad = 0.08 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.spines['bottom'].set_position(('data', 0.0))  # x轴移至 y=0
        ax.xaxis.set_ticks_position('bottom')
    else:
        if y_from_zero:
            hi = max(ci[0][1], ci[1][1], 0.0)
            ax.set_ylim(0.0, max(0.01, hi * 1.15))       # 从 0 起并留顶部空间

    style_axis(ax)
    ax.margins(x=0.05)

# ===== 2×2 绘图 =====
fig = plt.figure(figsize=(2.6, 3.8))
gs = gridspec.GridSpec(2, 2)
# (1,1) PCD: β 柱状图（x 轴放到 y=0）
ax = fig.add_subplot(gs[0, 0])
vals = res_pcd['beta_point']      # [β_MD, β_NC]
ci   = res_pcd['beta_ci']
bar_with_ci(ax, vals, ci, ylabel='β (PCD, signed)', flas_=False, put_xaxis_at_zero=True)
# (1,2) GD: β 柱状图（x 轴放到 y=0）
ax = fig.add_subplot(gs[0, 1])
vals = res_gd['beta_point']
ci   = res_gd['beta_ci']
fa=False
bar_with_ci(ax, vals, ci, ylabel='β (GD, signed)', flas_=False, put_xaxis_at_zero=True)
# (2,1) PCD: partial R^2 柱状图（从 0 起，完整显示柱子与误差线）
ax = fig.add_subplot(gs[1, 0])
vals = res_pcd['pr2_point']       # [partialR2_MD, partialR2_NC]
ci   = res_pcd['pr2_ci']
bar_with_ci(ax, vals, ci, ylabel='Partial $R^2$ (PCD)', flas_=True, put_xaxis_at_zero=False, y_from_zero=True)
# (2,2) GD: partial R^2 柱状图（从 0 起）
ax = fig.add_subplot(gs[1, 1])
vals = res_gd['pr2_point']
ci   = res_gd['pr2_ci']
bar_with_ci(ax, vals, ci, ylabel='Partial $R^2$ (GD)', flas_=True, put_xaxis_at_zero=False, y_from_zero=True)

plt.subplots_adjust(wspace=1.2, hspace=0.15, right=0.95, left=0.25, top=0.95, bottom=0.22)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'figure5-2.eps'), dpi=600,
            format='eps')















# ===== 统计检验（β 的 t 检验；partial R² 的 F 检验）=====
def zscore(v):
    v = np.asarray(v, float)
    mu = np.nanmean(v); sd = np.nanstd(v) + 1e-12
    return (v - mu) / sd

def t_test_betas(Z1, Z2, Zy):
    """两自变量+截距 OLS，返回 beta(带符号, 2维)、SE、t、p、df"""
    X = np.column_stack([np.ones_like(Z1), Z1, Z2])         # [1, Z1, Z2]
    pinv = np.linalg.pinv(X)
    beta_all = pinv @ Zy                                     # b0, b1, b2
    yhat = X @ beta_all
    resid = Zy - yhat
    n = len(Z1); p = 2                                       # 两个自变量
    df = max(n - (p + 1), 1)
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * (pinv @ pinv.T)
    se_all = np.sqrt(np.clip(np.diag(cov), 0, None))
    b = beta_all[1:]                                         # 仅 b1,b2
    se = se_all[1:]
    t = b / (se + 1e-12)
    try:
        from scipy.stats import t as student_t
        p = 2 * (1 - student_t.cdf(np.abs(t), df=df))
    except Exception:
        # 正态近似
        from math import erf, sqrt
        def norm_cdf(z): return 0.5*(1+erf(z/np.sqrt(2)))
        p = 2 * (1 - norm_cdf(np.abs(t)))
    return b, se, t, p, df

def pF_from_partialR2(partial_r2, df2):
    """单个参数的部分F检验 p 值（df1=1, df2=n-p-1）"""
    pvals = np.ones_like(partial_r2, dtype=float)
    try:
        from scipy.stats import f as fdist
        F = (partial_r2 / np.clip(1 - partial_r2, 1e-12, None)) * df2
        pvals = 1 - fdist.cdf(F, 1, df2)
    except Exception:
        # 粗略近似：由 pr2 -> t^2，再走正态近似
        import math
        def norm_cdf(z): return 0.5*(1+math.erf(z/np.sqrt(2)))
        F = (partial_r2 / np.clip(1 - partial_r2, 1e-12, None)) * df2
        t_abs = np.sqrt(np.clip(F, 0, None))                 # |t|
        pvals = 2*(1 - norm_cdf(t_abs))
    return pvals

# ---- 标准化数据用于检验 ----
Z1_pcd, Z2_pcd, Zy_pcd = zscore(rows_pcd[:,0]), zscore(rows_pcd[:,1]), zscore(rows_pcd[:,2])
Z1_gd,  Z2_gd,  Zy_gd  = zscore(rows_gd[:,0]),  zscore(rows_gd[:,1]),  zscore(rows_gd[:,2])

# β 的 t 检验
b_pcd, se_pcd, t_pcd, p_t_pcd, df_pcd = t_test_betas(Z1_pcd, Z2_pcd, Zy_pcd)
b_gd,  se_gd,  t_gd,  p_t_gd,  df_gd  = t_test_betas(Z1_gd,  Z2_gd,  Zy_gd)

# partial R² 的 F 检验（用你已经算好的 pr2 点估计）
# pr2 顺序与 beta 顺序一致： [MD, NC]
pF_pcd = pF_from_partialR2(np.asarray(res_pcd['pr2_point']), df2=df_pcd)
pF_gd  = pF_from_partialR2(np.asarray(res_gd['pr2_point']),  df2=df_gd)

# ===== 画图（与原图同样布局，但加入显著性星号）=====
labels = ['Modulation depth', 'Neighbor corr.']
colors = ['#4C78A8','#F58518']  # [MD, NC]

def style_axis(ax):
    ax.tick_params(labelsize=fontsize_)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth_ax)
    ax.spines['bottom'].set_linewidth(linewidth_ax)

def stars(p):
    return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 5e-2 else ''))

def bar_with_ci_and_sig(ax, vals, ci, pvals, ylabel, show_xtick=False,
                        put_xaxis_at_zero=False, y_from_zero=False):
    xpos = np.arange(len(vals))
    bars = ax.bar(xpos, vals, width=0.4, align='center',
                  color=colors, edgecolor='k', linewidth=0.6)
    centers = np.array([b.get_x() + b.get_width()/2 for b in bars])
    # ci: [ (lo,hi)_MD, (lo,hi)_NC ]  or shape(2,2)
    ci = np.asarray(ci)
    if ci.shape == (2,):
        ci = np.vstack([ci, ci]).T
    if ci.shape[0] == 2:   # (2,2) -> [lo,hi] x 2
        yerr = np.vstack([vals - ci[0], ci[1] - vals])
    else:                  # [(lo,hi),(lo,hi)]
        yerr = np.array([[vals[0]-ci[0][0], vals[1]-ci[1][0]],
                         [ci[0][1]-vals[0],  ci[1][1]-vals[1]]])
    ax.errorbar(centers, vals, yerr=yerr, fmt='none', ecolor='k', elinewidth=0.8, capsize=2)

    # 纵轴范围
    if put_xaxis_at_zero:
        lo = min(ci[0][0], ci[1][0], 0.0)
        hi = max(ci[0][1], ci[1][1], 0.0)
        pad = 0.08 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.spines['bottom'].set_position(('data', 0.0))
        ax.xaxis.set_ticks_position('bottom')
    elif y_from_zero:
        hi = max(ci[0][1], ci[1][1], 0.0)
        ax.set_ylim(0.0, max(0.01, hi * 1.15))

    # 显著性
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, (x, v) in enumerate(zip(centers, vals)):
        s = stars(pvals[i])
        if s:
            if put_xaxis_at_zero and v < 0:
                ytxt = min(v, ci[0][i]) - 0.04*yspan
                va = 'top'
            else:
                ytxt = max(v, ci[1][i]) + 0.04*yspan
                va = 'bottom'
            ax.text(x, ytxt, s, ha='center', va=va, fontsize=fontsize_+1, fontweight='bold')

    # x 轴标签
    ax.set_xticks(centers)
    if show_xtick:
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize_, fontname=fontname_)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel(ylabel, fontsize=fontsize_, fontname=fontname_, fontweight=fontweight_)
    style_axis(ax); ax.margins(x=0.05)

# ===== 与原图同尺寸/排版一致 =====
fig = plt.figure(figsize=(2.6, 3.8))
gs = gridspec.GridSpec(2, 2)

# (1,1) PCD: β（带签名）+ t 检验星号
ax = fig.add_subplot(gs[0, 0])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_pcd['beta_point']),
    ci=np.asarray(res_pcd['beta_ci']),
    pvals=p_t_pcd,
    ylabel='β (PCD, signed)',
    show_xtick=False,
    put_xaxis_at_zero=True)

# (1,2) GD: β + t 检验星号
ax = fig.add_subplot(gs[0, 1])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_gd['beta_point']),
    ci=np.asarray(res_gd['beta_ci']),
    pvals=p_t_gd,
    ylabel='β (GD, signed)',
    show_xtick=False,
    put_xaxis_at_zero=True)

# (2,1) PCD: partial R² + F 检验星号
ax = fig.add_subplot(gs[1, 0])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_pcd['pr2_point']),
    ci=np.asarray(res_pcd['pr2_ci']),
    pvals=pF_pcd,
    ylabel='Partial $R^2$ (PCD)',
    show_xtick=True,
    y_from_zero=True)

# (2,2) GD: partial R² + F 检验星号
ax = fig.add_subplot(gs[1, 1])
bar_with_ci_and_sig(ax,
    vals=np.asarray(res_gd['pr2_point']),
    ci=np.asarray(res_gd['pr2_ci']),
    pvals=pF_gd,
    ylabel='Partial $R^2$ (GD)',
    show_xtick=True,
    y_from_zero=True)

plt.subplots_adjust(wspace=1.2, hspace=0.15, right=0.95, left=0.25, top=0.95, bottom=0.22)
plt.savefig(os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'figure5-2-with-stats.eps'),
            dpi=600, format='eps')
plt.show()
