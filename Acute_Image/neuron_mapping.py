# 环境为tf1.13.1
import os.path
import pickle
import h5py
import numpy as np
import tqdm
from sklearn.metrics import  mean_absolute_error,mean_squared_error, r2_score,explained_variance_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVR
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

def plot_heatmap(neuron_dict, img_type, savename,neuron_name):
    feature_lists = list(neuron_dict['decode'].keys())
    resp_lists = list(neuron_dict['decode'][feature_lists[0]][img_type].keys())
    model_list = list(neuron_dict['decode'][feature_lists[0]][img_type][resp_lists[0]].keys())
    for model_name in model_list:
        df = pd.DataFrame(index=resp_lists, columns=feature_lists, dtype=float)
        for f in feature_lists:
            for r in resp_lists:
                df.at[r, f] = round(neuron_dict['decode'][f][img_type][r][model_name]['correlation'], 2)
        plt.figure(figsize=(16, 8))
        sns.heatmap(df, annot=True, cmap='YlGnBu', linewidths=0.5)
        plt.title(neuron_name + '-' + model_name + '_{}-corr'.format(img_type))
        plt.tight_layout()
        savepath = savename + '_' + model_name + '.png'
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

def plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name):
    resp_lists = list(neuron_dict['decode'][feature_plot][img_type].keys())
    model_list = list(neuron_dict['decode'][feature_plot][img_type][resp_lists[0]].keys())
    fig, axes = plt.subplots(len(resp_lists), len(model_list),figsize=(20, 9))
    for findex,decoder in enumerate(model_list):
        for rindex,res in enumerate(resp_lists):
            y_test=neuron_dict['decode'][feature_plot][img_type][res][decoder]['Y_test']
            y_pred= neuron_dict['decode'][feature_plot][img_type][res][decoder]['Y_pred']

            label_corr='corr '+str(round(neuron_dict['decode'][feature_plot][img_type][res][decoder]['correlation'],2))
            label_r2='r2 '+str(round(neuron_dict['decode'][feature_plot][img_type][res][decoder]['r2'],2))
            label_evs='evs '+str(round(neuron_dict['decode'][feature_plot][img_type][res][decoder]['evs'],2))
            axes[rindex, findex].plot(y_test,label=label_corr,linestyle='-',color='k',linewidth=1 )
            axes[rindex, findex].plot(y_pred,label=label_r2,linestyle='-',color='r',linewidth=1 )
            axes[rindex, findex].plot(y_pred,label=label_evs,linestyle='-',color='r',linewidth=1 )

            axes[rindex, findex].xaxis.set_visible(False)
            if findex==0:
                axes[rindex, findex].set_ylabel(res,fontsize=10)
            if rindex==0:
                axes[rindex, findex].set_title(decoder + '-{}-'.format(img_type) + feature_plot,fontsize=14)
            axes[rindex, findex].legend(loc='upper left')
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.95, wspace=0, hspace=0.2)
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    # plt.show()

def truncate_outliers_iqr(data):
    """
    使用 IQR 方法检测并截断异常值。
    """
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 截断异常值
    data_truncated = np.copy(data)
    data_truncated[data < lower_bound] = lower_bound
    data_truncated[data > upper_bound] = upper_bound
    return data_truncated

def fit_performance(y_test,y_pred):
    # 评价拟合性能
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # correlation = np.corrcoef(y_test, y_pred)[0, 1]
    correlation, p_value = pearsonr(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    # print(f"mae: {mae}", f"mse: {mse}", f"R^2 Score: {r2}", f"corr: {correlation}", f"evs: {evs}",f"p_value: {p_value}")
    return [mae,mse,r2,correlation,evs,p_value]

def extract_neuron_feature(neuron,task):
    #提取神经元指定任务的数据和标签
    data = []
    label = []
    name = []
    if task=='class':
        for t in list(neuron.keys()):
            if t.split('_')[0]==task:
                data.append(neuron[t])
                label.append(t.split('_')[2])
                name.append(t)
    if task=='view' or task=='tolerate':
        for t in list(neuron.keys()):
            if t.split('_')[0]==task:
                data.append(neuron[t])
                label.append(t.split('_')[1])
                name.append(t)
    data=np.array(data)
    return data,label,name

def assembly_neuron_resp(ndata_class):
    # 确定神经元响应特征组合形式
    resp = {}
    resp['trial_1']=ndata_class[:,0]                        # 取第一个试次
    resp['trial_2']=ndata_class[:,1]                        # 取第二个试次
    resp['trial_3']=ndata_class[:,2]                        # 取第三个试次
    resp['mean']=np.mean(ndata_class, 1)                    # 取平均
    resp['max']=np.max(ndata_class, 1)                      # 取最大
    resp['trial_1_2']=np.mean(ndata_class[:,[0, 1]], 1)     # 取1、2试次
    resp['trial_1_3']=np.mean(ndata_class[:,[0, 2]], 1)     # 取1、3试次
    resp['trial_2_3']=np.mean(ndata_class[:,[1, 2]], 1)     # 取2、3试次

    # 取3个数据中相近的两个的平均值
    b = np.sort(ndata_class, 1)
    a = np.diff(b, axis=1)
    c = np.argmin(a, 1)
    d = (b[:,0] + b[:,1]) / 2
    e = (b[:,1] + b[:,2]) / 2
    f = d * (1 - c) + e * c
    resp['trial_corr']=f
    return resp

def decode_analyse(x_train,y_train,x_test,model_name):
    Y_pred=[]
    # 训练线性回归模型
    if model_name=='regressor':
        model_regressor = LinearRegression()
        model_regressor.fit(x_train, y_train)
        Y_pred=model_regressor.predict(x_test)

    # bp网络
    if model_name=='bp':
        model_bp = Sequential()
        model_bp.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
        model_bp.add(Dense(1,))  # 输出层
        model_bp.compile(optimizer='adam', loss='mse')
        model_bp.fit(x_train, y_train, epochs=200, batch_size=32,verbose=0)
        temp = model_bp.predict(x_test,verbose=0)
        Y_pred=temp.reshape(-1)

    # svr
    if model_name=='svr':
        svr = SVR(kernel='rbf', C=100, epsilon=0.1)
        svr.fit(x_train, y_train)
        Y_pred=svr.predict(x_test)

    # xgb
    if model_name=='xgb':
        xg_reg = xgb.XGBRegressor(objective='count:poisson', n_estimators=300, learning_rate=0.01, max_depth=5)
        xg_reg.fit(x_train, y_train)
        Y_pred=xg_reg.predict(x_test)

    return Y_pred

if __name__ == "__main__":

    expname='ento_exp1'

    # 1.加载图像特征
    feat_path = r'E:\Image_paper\Project\Acute_Image\image\img_dict_pca.pkl'
    with open(feat_path, "rb") as f:
        img_dict = pickle.load(f)                           # 图像特征字典，每张图像是一个键

    # 2.加载神经元响应
    neuron_path = r'results/neuron_response'
    neuron_data = h5py.File(os.path.join(neuron_path,expname,expname+'_resp.mat'), 'r')
    neuron_data = neuron_data['resp_trial']         #读出维度为32*308*3
    select_channels=[2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,20,21,22,23,25,26,27,28,29,31,32]   #去除坏道
    df = pd.read_csv(os.path.join(neuron_path,expname,expname+'_classname.csv'),header=None)  # 读取图像名称，csv文件
    classnames= df.squeeze().tolist()

    # 循环每一有效通道
    for ch in select_channels:
        neuron_dict={}
        neuron_name=expname+'_ch'+str(ch)
        savepath = r'results/neuron_mapping'
        os.makedirs(os.path.join(savepath, expname), exist_ok=True)
        os.makedirs(os.path.join(savepath, expname+'_onlymapping'), exist_ok=True)
        # 3.神经元响应的规范化处理
        data_ch=neuron_data[ch-1]       #通道编号需要减1，Python从0开始
        data_reshaped = data_ch.reshape(-1, 1)  # 变为 (308 * 3, 1)，适合用于 fit_transform
        data_truncated = truncate_outliers_iqr(data_reshaped)  # 处理异常值
        data_scaled = scaler.fit_transform(data_truncated)
        data_scaled = (data_scaled - np.min(data_scaled)) / (np.max(data_scaled) - np.min(data_scaled)) #最大最小归一化
        data_scaled = np.nan_to_num(data_scaled)     #将nan值置为0
        data_scaled = data_scaled.reshape(data_ch.shape)
        neuron_dict['resp_class']=data_scaled[:200,:]        # 保存规范后的神经元数据,图像类别
        neuron_dict['resp_view'] = data_scaled[200:, :]     # 保存规范后的神经元数据，图像视角
        neuron_dict['name_class']=classnames[:200]        #保存规范后的神经元数据,图像类别
        neuron_dict['name_view']=classnames[200:]        #保存规范后的神经元数据,图像类别

        # 4.计算试次间的相关系数
        corr_matrix = np.corrcoef(neuron_dict['resp_class'], rowvar=False)  # rowvar=False表示列代表变量（试次）
        trial_corr_class=np.mean([corr_matrix[0, 1],corr_matrix[0, 2],corr_matrix[1, 2]])
        corr_matrix = np.corrcoef(neuron_dict['resp_view'], rowvar=False)  # rowvar=False表示列代表变量（试次）
        trial_corr_view=np.mean([corr_matrix[0, 1],corr_matrix[0, 2],corr_matrix[1, 2]])
        neuron_dict['trial_corr_class']=trial_corr_class
        neuron_dict['trial_corr_view'] = trial_corr_view

        # 5.确定响应的组合
        resp_assembly=assembly_neuron_resp(neuron_dict['resp_class'])
        resp_view_assembly = assembly_neuron_resp(neuron_dict['resp_view'])

        # 6.确定图像特征
        feature_names = ['color_features_pca', 'shape_features_pca', 'texture_features_pca', 'cst_features',
                         'v1like_features_pca', 'v2like_features_pca', 'hmax_c1_pca', 'hmax_c2_pca',
                         'alexnet_features.2_pca', 'alexnet_features.5_pca', 'alexnet_features.7_pca', 'alexnet_features.9_pca','alexnet_features.12_pca',
                         'vgg16_features.4_pca', 'vgg16_features.9_pca', 'vgg16_features.16_pca', 'vgg16_features.23_pca','vgg16_features.30_pca',
                         'resnet50_layer1_pca', 'resnet50_layer2_pca', 'resnet50_layer3_pca', 'resnet50_layer4_pca',
                         'inceptionv3_maxpool2_pca', 'inceptionv3_Mixed_5d_pca', 'inceptionv3_Mixed_6e_pca','inceptionv3_Mixed_7c_pca']

        # 7.映射
        neuron_dict['decode']={}
        for feature in feature_names:
            neuron_dict['decode'][feature] = {}
            neuron_dict['decode'][feature]['class']={}
            neuron_dict['decode'][feature]['view']={}
            # 取类别图像特征
            X_data = []
            for name in neuron_dict['name_class']:
                X_data.append(img_dict[name][feature])
            X_data = np.array(X_data)         # neuron number * feature number
            # 取视角图像特征
            X_view_test = []
            for name in neuron_dict['name_view']:
                X_view_test.append(img_dict[name][feature])
            X_view_test = np.array(X_view_test)  # neuron number * feature number

            for resp_name in list(resp_assembly.keys()):
                neuron_dict['decode'][feature]['class'][resp_name] = {}
                neuron_dict['decode'][feature]['view'][resp_name] = {}
                # 取神经元响应
                Y_data=resp_assembly[resp_name]      # 只选择类别图像
                model_list = ['regressor', 'svr', 'xgb']
                for model_name in model_list:
                    neuron_dict['decode'][feature]['class'][resp_name][model_name] = {}
                    neuron_dict['decode'][feature]['view'][resp_name][model_name] = {}
                    # 取解码器
                    Y_test=[]
                    Y_index=[]
                    Y_pred=[]
                    for k in range(5):
                        # 五折交叉验证
                        test_indices = np.arange(k, Y_data.shape[0], 5)  # 获取测试集索引
                        train_indices = np.setdiff1d(np.arange(Y_data.shape[0]), test_indices)  # 获取训练集的索引（除去测试集的索引）
                        x_train = X_data[train_indices, :]
                        y_train = Y_data[train_indices]
                        x_test = X_data[test_indices, :]
                        y_test = Y_data[test_indices]
                        Y_test.extend(y_test)
                        Y_index.extend(list(test_indices))
                        #解码
                        Y_pred.extend(decode_analyse(x_train,y_train,x_test,model_name))
                    #排序
                    sorted_indices = np.argsort(Y_index)  # 从小到大排序
                    Y_index_sorted = np.array(Y_index)[sorted_indices]
                    Y_test_sorted = np.array(Y_test)[sorted_indices]
                    Y_pred_sorted = np.array(Y_pred)[sorted_indices]
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['Y_test'] = Y_test_sorted
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['Y_pred'] = Y_pred_sorted
                    #计算性能指标
                    mae, mse, r2, correlation, evs, p_value=fit_performance(Y_test_sorted, Y_pred_sorted)
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['mae']  = mae
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['mse'] = mse
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['r2'] = r2
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['correlation'] = correlation
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['evs'] = evs
                    neuron_dict['decode'][feature]['class'][resp_name][model_name]['p_value'] = p_value
                    print('class: {},{},{},{},{:.2f}'.format(neuron_name,feature,resp_name,model_name,correlation))

                    # 预测视角响应
                    Y_view_test=resp_view_assembly[resp_name]      # 只选择类别图像
                    Y_view_pred=decode_analyse(X_data, Y_data, X_view_test, model_name)
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['Y_test'] = Y_view_test
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['Y_pred'] = Y_view_pred
                    mae, mse, r2, correlation, evs, p_value=fit_performance(Y_view_test, Y_view_pred)
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['mae']  = mae
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['mse'] = mse
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['r2'] = r2
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['correlation'] = correlation
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['evs'] = evs
                    neuron_dict['decode'][feature]['view'][resp_name][model_name]['p_value'] = p_value
                    print('view: {},{},{},{},{:.2f}'.format(neuron_name, feature, resp_name, model_name, correlation))

        #保存结果
        with open(os.path.join(savepath,expname,neuron_name+'.pkl'), 'wb') as file:
            pickle.dump(neuron_dict, file)

        #可视化,热力图，图像类别
        img_type='class'
        savename=os.path.join(savepath, expname, neuron_name + '_{}'.format(img_type))
        plot_heatmap(neuron_dict, img_type, savename,neuron_name)
        savename = os.path.join(savepath, expname+'_onlymapping', neuron_name + '_{}'.format(img_type))
        plot_heatmap(neuron_dict, img_type, savename, neuron_name)

        #可视化,热力图，图像视角
        img_type='view'
        savename=os.path.join(savepath, expname, neuron_name + '_{}'.format(img_type))
        plot_heatmap(neuron_dict, img_type, savename,neuron_name)


        #可视化,调谐曲线相关性
        img_type = 'class'
        feature_plot='cst_features'
        savename=os.path.join(savepath, expname, neuron_name + '_{}_{}.png'.format(img_type,feature_plot))
        plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name)
        feature_plot='alexnet_features.12_pca'
        savename=os.path.join(savepath, expname, neuron_name + '_{}_{}.png'.format(img_type,feature_plot))
        plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name)
        feature_plot='color_features_pca'
        savename=os.path.join(savepath, expname, neuron_name + '_{}_{}.png'.format(img_type,feature_plot))
        plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name)


        img_type = 'view'
        feature_plot='cst_features'
        savename=os.path.join(savepath, expname, neuron_name + '_{}_{}.png'.format(img_type,feature_plot))
        plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name)
        feature_plot='alexnet_features.12_pca'
        savename=os.path.join(savepath, expname, neuron_name + '_{}_{}.png'.format(img_type,feature_plot))
        plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name)
        feature_plot='color_features_pca'
        savename=os.path.join(savepath, expname, neuron_name + '_{}_{}.png'.format(img_type,feature_plot))
        plot_mapping(neuron_dict, img_type,feature_plot,savename,neuron_name)

        # with open(os.path.join(savepath,expname,neuron_name+'.pkl'), 'rb') as file:
        #     neuron_dict= pickle.load(file)

