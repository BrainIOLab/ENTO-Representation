# 环境为pt1.12.1
import cv2
import os
import numpy as np
import pickle
from tools_img.extract_color_features import *
from tools_img.extract_shape_features import *
from tools_img.extract_texture_features import *
from tools_img.extract_v1like_features import *
from tools_img.extract_v2like_features import *
from tools_img.extract_hmax_features import *
from tools_img.extract_cnn_features import *
from sklearn.decomposition import PCA

def pca_image_dict(img_dict_norm):
    pca_feature_dict={'color_features':100,
                           'shape_features': 100,  # 需要pca降维的特征和降维后的维度
                           'texture_features':100,
                           'v1like_features':384,
                           'v2like_features':384,
                           'hmax':384,
                           'alexnet':384,
                           'vgg16':384,
                           'resnet50':384,
                           'inceptionv3':384,
                           'mobilenetv2':384,
                           'densenet121':384}

    feature_list=get_feature_names(img_dict_norm)                       #获取字典中的全部特征名称,90个特征
    for pca_feature, pca_n in pca_feature_dict.items():
        if pca_feature in feature_list:                                 #判断特征名称是单一的还是代指的
            feature_name=[pca_feature]
        else:
            feature_name=[]
            for s in feature_list:                                      #将代指的一类特征取出
                if pca_feature in s:
                    feature_name.append(s)
        #对数据进行pca降维
        for f in tqdm.tqdm(feature_name):
            value=return_feature_value(img_dict_norm, f)                #获取单个特征的所有图像
            pca = PCA(n_components=0.95)  # 保留95%方差
            reduced_data = pca.fit_transform(value)
            print(f,'降维后维度：',reduced_data.shape[1])    #降维后的可解释方差
            new_feature_name = f + '_pca'
            img_dict_norm = assignment_imgdict_feature(img_dict_norm, reduced_data, new_feature_name)   #将数据赋予字典中，新的变量

    return img_dict_norm

def assignment_imgdict_feature(img_dict_norm,value,new_feature_name):
    # 将一个数组中的变量赋予到img_dict中
    if value.shape[0] == len(img_dict_norm):  # 如果字典数量和特征数组维度不等，则不能进行赋值
        for index, key in enumerate(list(img_dict_norm.keys())):
            img_dict_norm[key][new_feature_name] = value[index,:]
    else:
        raise ValueError("输入维度和图像数目不匹配！")
    return img_dict_norm

def return_feature_value(img_dict, feature_name):
    # 返回所有图像的某个特征，n_image×n_dinm
    value = []
    for key in list(img_dict.keys()):
        value.append(img_dict[key][feature_name])
    value = np.array(value)
    return value

def get_feature_names(img_dict):
    # 返回字典里所有特征的名称
    feature_list=['color','shape','texture','v1like','v2like','hmax','alexnet','vgg16',
                   'resnet50','inceptionv3','mobilenetv2','densenet121','inceptionv3']  #需要规范化的图像参数
    key = list(img_dict.keys())[0]
    feature_names = []
    for feature_key, feature_value in img_dict[key].items():
        if feature_key.split('_')[0] in feature_list and feature_key.split('_')[-1] != 'names':
            feature_names.append(feature_key)
    return feature_names

def built_image_struct(rootpath):
    #创建图像字典
    img_dict = {}
    for cla in os.listdir(rootpath):
        idict = {}
        print(cla, ':', len(os.listdir(os.path.join(rootpath, cla))))
        for name in os.listdir(os.path.join(rootpath, cla)):
            idict_=idict.copy()
            idict_['imgpath'] = os.path.join(rootpath, cla, name)
            idict_['task'] = cla.split('_')[0]
            idict_['object'] = cla.split('_')[1]
            idict_['class'] = cla.split('_')[2]
            img_dict[name] = idict_
    return img_dict


def norm_image_dict(img_dict):
    from sklearn.preprocessing import StandardScaler
    feature_names=['color','shape','texture','v1like','v2like','hmax','alexnet','vgg16',
                   'resnet50','inceptionv3','mobilenetv2','densenet121']  #需要规范化的图像参数
    #索引所有特征名称，创建空字典
    key =list(img_dict.keys())[0]
    feature_dict={}
    for feature_key, feature_value in img_dict[key].items():
        if feature_key.split('_')[0] in feature_names and feature_key.split('_')[-1] != 'names':
            feature_dict[feature_key]=[]
    #所有图像的特征添加到字典
    for key in list(img_dict.keys()):
        for feature_key, feature_value in img_dict[key].items():
            if feature_key.split('_')[0] in feature_names and feature_key.split('_')[-1] != 'names':
                feature_dict[feature_key].append(feature_value)
    #所有特征计算规范化参数，并保存到字典
    scaler_dict={}
    for key in tqdm.tqdm(list(feature_dict.keys())):
        value=feature_dict[key]
        value=np.array(value)                               #特征维度为2维，nimages*nfeatures
        value = np.squeeze(value)                           #防止特征有多的维度
        if len(value.shape)==1:                             #特征维度为1时，reshape
            value=value.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(value)
        scaler_dict[key]=scaler                             #保存规范化结果
    #所有特征执行规范化
    for key in tqdm.tqdm(list(img_dict.keys())):
        for feature_key, feature_value in img_dict[key].items():
            if feature_key.split('_')[0] in feature_names and feature_key.split('_')[-1] != 'names':
                # print(feature_key)
                value=np.array(img_dict[key][feature_key])
                value=np.transpose(value.reshape(-1,1))
                value_norm=scaler_dict[feature_key].transform(value)
                img_dict[key][feature_key]=value_norm.reshape(-1)
    return img_dict

if __name__ == "__main__":
    rootpath = r'image\img_rgb'
    img_dict = built_image_struct(rootpath)
    img_dict = extract_color_features(img_dict)
    img_dict = extract_shape_features(img_dict)
    img_dict = extract_texture_features(img_dict)
    img_dict = extract_v1like_features(img_dict,[224,224,3])        #输入图像的尺寸
    img_dict = extract_v2like_features(img_dict)
    img_dict = extract_hmax_features(img_dict)
    img_dict = extract_cnn_features(img_dict, 'alexnet')            #输入模型的名称，alexnet，vgg16，resnet50，inceptionv3，mobilenetv2，densenet121
    img_dict = extract_cnn_features(img_dict, 'vgg16')
    img_dict = extract_cnn_features(img_dict, 'resnet50')
    img_dict = extract_cnn_features(img_dict, 'inceptionv3')
    img_dict = extract_cnn_features(img_dict, 'mobilenetv2')
    img_dict = extract_cnn_features(img_dict, 'densenet121')

    with open('image\img_features\img_dict.pkl', 'wb') as file:
        pickle.dump(img_dict, file)

    #数据规范化处理
    img_dict_norm=norm_image_dict(img_dict)
    with open('image\img_features\img_dict_norm.pkl', 'wb') as file:
        pickle.dump(img_dict_norm, file)

    # 将一些变量pca降维
    img_dict_pca=pca_image_dict(img_dict_norm)

    #将颜色、形状和纹理特征合并
    color_feature=return_feature_value(img_dict_pca, 'color_features_pca')
    shape_features=return_feature_value(img_dict_pca, 'shape_features')
    texture_features=return_feature_value(img_dict_pca, 'texture_features_pca')
    cst_features=np.concatenate([color_feature,shape_features,texture_features],axis=1)
    img_dict_pca=assignment_imgdict_feature(img_dict_pca,cst_features,'cst_features')

    with open('image\img_features\img_dict_pca.pkl', 'wb') as file:
        pickle.dump(img_dict_pca, file)


