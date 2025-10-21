import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score


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

def compute_class_distance(rdm, labels):
    #计算平均类内距离，和平均类间距离
    intra, inter = [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                intra.append(rdm[i, j])
            else:
                inter.append(rdm[i, j])
    return np.mean(intra), np.mean(inter)

def cal_class_performs(resp_split, label_fine):
    dict_performs={}

    resp_=np.transpose(resp_split)
    #解码准确率
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    model = SVC(kernel='linear', decision_function_shape='ovr')        #linear
    y_pred = cross_val_predict(model, resp_, label_fine, cv=cv)
    accuracy = accuracy_score(label_fine, y_pred)  # 准确率

    #降维及rdm矩阵
    X_reduced, rdm = analyse_reducedim(resp_, 'MDS', 3)
    #类内距离、类间距离
    dist_intra, dist_inter=compute_class_distance(rdm, label_fine)
    #分离比
    Separability = dist_inter / (dist_intra + 1e-9)  # 计算类分离比

    dict_performs['Accuracy']=accuracy
    dict_performs['Intra-class dist.'] = dist_intra
    dict_performs['Inter-class dist.'] = dist_inter
    dict_performs['Separability'] = Separability

    return dict_performs

import numpy as np
from scipy.linalg import eigh   # 兼容 python3.6

def mi_gaussian_lda_approx(X, y, ridge=1e-3, base=2):
    """
    估算神经元集群对类别刺激的互信息 I(C;R)，
    高斯-LDA 近似，兼容 Python 3.6
    ----------
    X : ndarray, shape (N, D)
        N 个样本 × D 维神经元响应
    y : ndarray, shape (N,)
        样本的类别标签，0~K-1
    ridge : float
        协方差正则化系数
    base : int or float
        对数底。2 为比特，np.e 为nat。
    """
    N, D = X.shape
    classes = np.unique(y)
    K = len(classes)
    mu = np.mean(X, axis=0)

    # 类均值与类内协方差
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        muc = np.mean(Xc, axis=0)
        Xc0 = Xc - muc
        Sw += np.dot(Xc0.T, Xc0)
        diff = (muc - mu).reshape(-1, 1)
        Sb += nc * np.dot(diff, diff.T)
    Sw /= (N - K)
    Sb /= (K - 1)

    # 正则化以避免奇异
    Sw += ridge * np.eye(D)

    # 计算广义特征值：解 Sb v = λ Sw v
    vals, _ = eigh(Sb, Sw)
    vals = np.clip(vals, 0, None)   # 去除负数的数值误差

    log = np.log2 if base == 2 else np.log
    I = 0.5 * np.sum(log(1.0 + vals))
    return I

import pandas as pd
df = pd.DataFrame.from_dict(dict_perform, orient='index')

# 可选：只保留需要的列（避免后续名字里有空格/点导致不便）
keep_cols = [c for c in df.columns if any(k in c for k in [
    'Accuracy','Intra-class dist.','Inter-class dist.','Separability','MI',
    'Color','Shape','Texture','V1-like','V2-like','Alexnet Conv1','Alexnet Conv2','Alexnet Conv3','Alexnet Conv4','Alexnet Conv5',
    'Selectivity index','Separation ratio'
])]
df = df[keep_cols].copy()

# 看看缺失/异常
print(df.describe().T)
print(df.isna().sum().sort_values(ascending=False).head(10))

import numpy as np
from scipy.stats import spearmanr

targets = ['Accuracy']
features = [c for c in df.columns if c not in targets and 'dist' not in c]

corr_rows = []
for y in targets:
    for x in features:
        ok = df[[x,y]].dropna()
        r, p = spearmanr(ok[x], ok[y])
        corr_rows.append({'target':y,'feature':x,'rho':r,'p':p})

corr_df = pd.DataFrame(corr_rows)

# Benjamini–Hochberg (FDR)
corr_df = corr_df.sort_values('p')
m = len(corr_df)
corr_df['p_fdr'] = corr_df['p'] * (np.arange(1, m+1)/m)
corr_df['p_fdr'] = corr_df['p_fdr'].clip(upper=1.0)

# 看看对 Accuracy/Separability/MI 贡献最大的前几项
print(corr_df.query("target=='Accuracy'").nsmallest(10,'p_fdr'))
print(corr_df.query("target=='Separability'").nsmallest(10,'p_fdr'))
print(corr_df.query("target=='MI'").nsmallest(10,'p_fdr'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

# ===== 1) 配置：因变量与自变量 =====
features = ['Color_mean','Shape_mean','Texture_mean','V1-like_mean','V2-like_mean',
            'Alexnet Conv1_mean','Alexnet Conv2_mean','Alexnet Conv3_mean','Alexnet Conv4_mean','Alexnet Conv5_mean']
target = 'MI'

# 只保留需要的列并去掉缺失
cols_needed = features + [target]
ok = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
X = ok[features].values
Y = ok[target].values

# ===== 2) 岭回归（5折CV） =====
ridge_pipe = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5)
)
ridge_pipe.fit(X, Y)

best_alpha = ridge_pipe.named_steps['ridgecv'].alpha_
coefs = ridge_pipe.named_steps['ridgecv'].coef_
coef_s = pd.Series(coefs, index=features).sort_values(key=np.abs, ascending=False)

# 交叉验证 R^2（更稳健的泛化度量）
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(ridge_pipe, X, Y, cv=cv, scoring='r2')
print(f"[Ridge] best alpha = {best_alpha:.4g}; CV R^2 = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print("\n[Ridge] 标准化系数（按绝对值排序，越大表示独立贡献越强）")
print(coef_s.round(4).head(12))

# ===== 3) 偏相关（Spearman，控制其余特征） =====
def partial_corr_spearman(df_xyc, x_name, y_name, control_names):
    """y~controls 残差 与 x~controls 残差 的 Spearman 相关"""
    sub = df_xyc[[x_name, y_name] + control_names].dropna()
    if sub.shape[0] < 10:
        return np.nan, np.nan
    Xc = sub[control_names].values

    # y 残差
    lin_y = LinearRegression().fit(Xc, sub[y_name].values)
    ry = sub[y_name].values - lin_y.predict(Xc)

    # x 残差
    lin_x = LinearRegression().fit(Xc, sub[x_name].values)
    rx = sub[x_name].values - lin_x.predict(Xc)

    rho, p = spearmanr(rx, ry)
    return rho, p

rows = []
for x in features:
    controls = [c for c in features if c != x]
    rho, p = partial_corr_spearman(ok, x, target, controls)
    rows.append({'feature': x, 'rho_partial': rho, 'p': p})

pc_df = pd.DataFrame(rows).sort_values('p')

# Benjamini–Hochberg FDR
m = pc_df['p'].notna().sum()
ranks = np.arange(1, m+1)
p_sorted = pc_df['p'].dropna().values
p_fdr_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]  # 保序
pc_df.loc[pc_df['p'].notna(), 'p_fdr'] = p_fdr_sorted

print("\n[偏相关] Accuracy ~ 每个特征 | 其余特征（Spearman，FDR校正后按 p 排序）")
print(pc_df[['feature','rho_partial','p','p_fdr']].round(4))
