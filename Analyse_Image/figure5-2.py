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


import cv2
import numpy as np
import os


def scatter_2d_image_with_path(
        X_reduced, name_class, savename, img_size=60, figure_size=360,
        circle_color=(230, 230, 230), circle_radius=None,
        poly_color=(80, 80, 80), poly_thickness=2,
        map_bounds=None,  # <<< 新增：((xmin, xmax), (ymin, ymax)) or None
        flip_x=True  # 与之前保持一致：水平翻转
):
    """
    map_bounds: 若为 None，则按本图数据自适应；否则用给定全局范围做线性映射，保证多图同尺度。
                形如 ((xmin, xmax), (ymin, ymax))，对应 X_reduced 的列顺序。
    """
    X_reduced = np.asarray(X_reduced, float)
    assert X_reduced.ndim == 2 and X_reduced.shape[1] == 2, "X_reduced shape must be (N,2)"
    N = X_reduced.shape[0]
    assert len(name_class) == N, "name_class length must match X_reduced"

    # --- 映射到画布：每轴映射到 [img_size/2, figure_size - img_size/2] ---
    pad_low = img_size / 2.0
    pad_high = figure_size - img_size / 2.0
    mapped = np.zeros_like(X_reduced, dtype=float)

    for ax in (0, 1):
        if map_bounds is not None:
            vmin, vmax = map_bounds[ax]
        else:
            vmin, vmax = np.min(X_reduced[:, ax]), np.max(X_reduced[:, ax])
        if np.isclose(vmax, vmin):
            mapped[:, ax] = (pad_low + pad_high) / 2.0
        else:
            mapped[:, ax] = np.interp(X_reduced[:, ax], (vmin, vmax), (pad_low, pad_high))

    # OpenCV坐标：(col=x, row=y)。你之前对 x 轴做了水平翻转，这里保留可选 flip_x。
    pts = []
    for i in range(N):
        my, mx = mapped[i, 0], mapped[i, 1]  # 注意这里 my是列方向，mx是行方向
        if flip_x:
            mx = figure_size - mx
        pts.append([int(round(my)), int(round(mx))])
    pts = np.array(pts, dtype=int)  # (N,2) -> (col, row)

    # 画布
    image_bg = np.ones((figure_size, figure_size, 3), dtype=np.uint8) * 255

    # 圆半径
    if circle_radius is None:
        circle_radius = int(round(img_size * 0.6 * 0.5))

    # 先画灰色圆
    for (cx, cy) in pts:
        cv2.circle(image_bg, (cx, cy), circle_radius, circle_color, thickness=-1, lineType=cv2.LINE_AA)

    # 再画首尾相接的折线
    cv2.polylines(image_bg, [pts.reshape(-1, 1, 2)], isClosed=True,
                  color=poly_color, thickness=poly_thickness, lineType=cv2.LINE_AA)

    # 贴图（最上层）
    for i_index, fname in enumerate(name_class):
        imgpath = os.path.join(r'Acute_Image\image\img_rgb', fname.split('.')[0][0:-3], fname)
        image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if image is None:
            continue
        imgpath_mask = imgpath.replace('img_rgb', 'img_mask')
        img_mask = cv2.imread(imgpath_mask, cv2.IMREAD_GRAYSCALE)
        if img_mask is None:
            img_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        resized_image = cv2.resize(image, (img_size, img_size))
        resized_mask = cv2.resize(img_mask, (img_size, img_size))
        m = resized_mask.copy()
        m[m > 0] = 1
        m = 1 - m  # 前景=1
        m3 = np.stack([m] * 3, axis=-1).astype(np.uint8)
        image_fg = resized_image * m3

        cx, cy = pts[i_index]  # (col, row)
        x0 = int(round(cx - img_size / 2));
        y0 = int(round(cy - img_size / 2))
        x1 = x0 + img_size;
        y1 = y0 + img_size

        x0c = max(x0, 0);
        y0c = max(y0, 0)
        x1c = min(x1, figure_size);
        y1c = min(y1, figure_size)
        if x1c <= x0c or y1c <= y0c:
            continue

        fx0 = x0c - x0;
        fy0 = y0c - y0
        fx1 = fx0 + (x1c - x0c);
        fy1 = fy0 + (y1c - y0c)

        roi = image_bg[y0c:y1c, x0c:x1c]
        fg = image_fg[fy0:fy1, fx0:fx1]
        m3c = m3[fy0:fy1, fx0:fx1]
        inv = np.where(m3c == 0, 1, 0).astype(np.uint8)
        tmp = roi * inv + fg
        image_bg[y0c:y1c, x0c:x1c] = tmp

    # 黑色外边框（描边、不改尺寸）
    border_thickness = 2
    cv2.rectangle(image_bg, (0, 0), (figure_size - 1, figure_size - 1), (0, 0, 0), thickness=border_thickness)

    # 保存
    cv2.imwrite(savename, image_bg)

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
        dict_performs[obj][idx]['resp'] = resp_split
        dict_performs[obj][idx]['X_reduced'] = X_reduced
        dict_performs[obj][idx]['rdm'] = rdm
        print(obj, idx, pd, global_distance)

with open(os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'dict_performs.pkl'), 'wb') as file:
    pickle.dump(dict_performs, file)

neuron_path = os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'dict_performs.pkl')
with open(neuron_path, "rb") as f:
    dict_performs= pickle.load(f)  # 获取全部神经元响应

for obj in obj_list:
    for metric_key in ['Modulation_depth_mean',  'Neighbor_correlation_mean']:
        k = 4  # 均匀选取的数量
        # 1) 收集并排序（升序）
        pairs = []  # (idx, metric_value)
        for idx, rec in dict_performs.get(obj, {}).items():
            val = rec.get(metric_key, None)
            if val is not None and np.isfinite(val):
                pairs.append((idx, float(val)))
        pairs.sort(key=lambda t: t[1])  # 按 metric 升序

        # 2) 在排序后的索引上等间距选择 k 个位置
        k_eff = min(k, len(pairs))
        pos = np.linspace(0, len(pairs)-1, k_eff)
        sel_ids = np.unique(np.round(pos).astype(int))  # 去重防越界
        # 若去重后少于 k_eff，则从未选过的位置补齐
        if sel_ids.size < k_eff:
            remaining = [i for i in range(len(pairs)) if i not in sel_ids]
            sel_ids = np.r_[sel_ids, remaining[:(k_eff - sel_ids.size)]]

        # 3) 取出选中的 X_reduced（以及对应的 idx 和 metric 值）
        selected = []
        for i in sel_ids:
            idx, val = pairs[i]
            xr = dict_performs[obj][idx]['X_reduced']
            selected.append({
                'idx': idx,
                metric_key: val,
                'X_reduced': xr
            })

        # 也可以单独拿到一个列表：4 个 X_reduced
        selected_X_reduced = [s['X_reduced'] for s in selected]



        # 计算全局坐标范围（按列：0轴、1轴分别）
        all_points = np.vstack([np.asarray(xr, float) for xr in selected_X_reduced])  # 4个拼起来
        xmin, xmax = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        ymin, ymax = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        global_bounds = ((xmin, xmax), (ymin, ymax))  # 传给 map_bounds

        # 绘制四张图，使用同一 map_bounds 确保尺度一致
        for idx in range(4):
            X_reduced = selected_X_reduced[idx]
            name_class = dict_view['ento_exp15_ch20'][obj]['name']
            savename = os.path.join(r'E:\Image_paper\Project\plot', 'figure5', 'image', f'{metric_key}_{obj}_{idx}.jpg')

            scatter_2d_image_with_path(
                X_reduced, name_class, savename,
                img_size=60,
                figure_size=360,
                circle_color=(230,230,230),
                circle_radius=None,
                poly_color=(80,80,80),
                poly_thickness=1,
                map_bounds=global_bounds,   # <<< 统一尺度
                flip_x=True                 # 与之前一致
            )
