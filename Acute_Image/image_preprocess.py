#图像预处理函数
import cv2
import os
import numpy as np

def resize_img_keep_ratio(img,target_size,bg_color):
    old_size= img.shape[0:2]
    #ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv2.resize(img,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,bg_color)
    return img_new
#

def image_rename(rootpath):
    #将rootpath里所有文件夹（二级目录）中的图像的名称改为“文件夹名称_编号”的形式
    for i in os.listdir(rootpath):
        print(i)
        for index,j in enumerate(os.listdir(os.path.join(rootpath,i))):
            print(os.path.join(rootpath,i,i+'_{:0>2d}.png'.format(index)))
            os.rename(os.path.join(rootpath,i,j), os.path.join(rootpath,i,i+'_{:0>2d}.png'.format(index)))


def image_resize_padding(rootpath,target_size,bg_color):
    #rootpath同级目录下根据img_init创建img_rgb,img_gray和img_mask三个文件夹
    bg_value=bg_color[0]
    init_rootpath=os.path.join(rootpath,'img_init')
    for i in os.listdir(init_rootpath):
        for j in os.listdir(os.path.join(init_rootpath, i)):
            imgfile=os.path.join(init_rootpath,i,j)                          #图像地址
            print(imgfile)

            #读取图像，得到mask
            img = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
            assert img.shape[2]==4,  imgfile      #图像必须有4个通道，最后一个通道表示mask
            mask=img[:,:,3]
            mask[mask<200]=0                                            #mask阈值化
            mask[mask>=200]=1
            mask=np.int32(mask)

            #根据mask取出rgb图像前景
            temp=np.expand_dims(mask, 2)
            temp=np.repeat(temp,3,2)
            temp=np.int32(temp)
            img_color=img[:,:,:3]*temp+np.abs(temp*bg_value-bg_value)
            #根据mask取出灰度图像前景
            gray=cv2.imread(imgfile,0)
            img_gray=gray*mask+np.abs(mask*bg_value-bg_value)
            #根据mask取出二值图像前景
            img_bina=mask*(250-bg_value)+bg_value
            # img_bina=mask*255

            mask=np.uint8(mask)
            #寻找最面积最大的外接矩形
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>1:
                print('error!')
            area=[cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3] for c in contours]
            cindex=np.argmax(area)
            x, y, w, h = cv2.boundingRect(contours[cindex])  # 计算点集最外面的矩形边界

            #根据最小外接边缘对图像进行裁剪
            img_color=img_color[y:y+h,x:x+w,:]
            img_gray=img_gray[y:y+h,x:x+w]
            img_bina=img_bina[y:y+h,x:x+w]

            #resize和padding
            img_color=np.uint8(img_color)
            img_gray = np.uint8(img_gray)
            img_bina = np.uint8(img_bina)
            img_color= resize_img_keep_ratio(img_color, target_size,bg_color)
            img_gray = resize_img_keep_ratio(img_gray, target_size,bg_color)
            img_bina = resize_img_keep_ratio(img_bina, target_size,bg_color)

            #写入文件
            try:
                os.mkdir(os.path.join(rootpath,'img_rgb',i))
                os.mkdir(os.path.join(rootpath, 'img_gray', i))
                os.mkdir(os.path.join(rootpath, 'img_mask', i))
            except:
                pass
            cv2.imwrite(os.path.join(rootpath,'img_rgb',i,j[:-4]+'.jpg'),img_color)
            cv2.imwrite(os.path.join(rootpath,'img_gray',i,j[:-4]+'.jpg'),img_gray)
            cv2.imwrite(os.path.join(rootpath,'img_mask',i,j[:-4]+'.jpg'),img_bina)

            gray_image_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            binary_image_3ch = cv2.cvtColor(img_bina, cv2.COLOR_GRAY2BGR)
            concatenated_img = cv2.hconcat([img_color, gray_image_3ch, binary_image_3ch])
            cv2.imshow('Concatenated Image', concatenated_img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

def image_resize_padding_view(rootpath,savepath,target_size,bg_color):
    #rootpath同级目录下根据img_init创建img_rgb,img_gray和img_mask三个文件夹
    init_rootpath=rootpath
    bg_value=bg_color[0]
    for i in os.listdir(init_rootpath):
        for j in os.listdir(os.path.join(init_rootpath, i)):
            imgfile=os.path.join(init_rootpath,i,j)                          #图像地址
            #读取图像，得到mask
            img = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
            mask=img[:,:,3]
            mask[mask<200]=0                                            #mask阈值化
            mask[mask>=200]=1
            mask = np.int32(mask)

            #根据mask取出rgb图像前景
            temp=np.expand_dims(mask, 2)
            temp=np.repeat(temp,3,2)
            temp = np.int32(temp)
            img_color=img[:,:,:3]*temp+np.abs(temp*bg_value-bg_value)
            #根据mask取出灰度图像前景
            gray=cv2.imread(imgfile,0)
            img_gray=gray*mask+np.abs(mask*bg_value-bg_value)
            img_bina=mask*(250-bg_value)+bg_value

            # img_bina=mask*255
            img_color=np.uint8(img_color)
            img_gray = np.uint8(img_gray)
            img_bina = np.uint8(img_bina)
            img_color=resize_img_keep_ratio(img_color,target_size,bg_color)
            img_gray = resize_img_keep_ratio(img_gray, target_size,bg_color)
            img_bina = resize_img_keep_ratio(img_bina, target_size,bg_color)

            #写入文件
            try:
                os.mkdir(os.path.join(savepath,'img_rgb',i))
                os.mkdir(os.path.join(savepath, 'img_gray', i))
                os.mkdir(os.path.join(savepath, 'img_mask', i))
            except:
                pass
            cv2.imwrite(os.path.join(savepath,'img_rgb',i,j),img_color)
            cv2.imwrite(os.path.join(savepath,'img_gray',i,j),img_gray)
            cv2.imwrite(os.path.join(savepath,'img_mask',i,j),img_bina)

            gray_image_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            binary_image_3ch = cv2.cvtColor(img_bina, cv2.COLOR_GRAY2BGR)
            concatenated_img = cv2.hconcat([img_color, gray_image_3ch, binary_image_3ch])
            cv2.imshow('Concatenated Image', concatenated_img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

def image_show_classes(rootpath,flag):
    savename = r'image\image_all_{}.jpg'.format(flag)
    #可视化图像，每个类别一行
    for iindex, i in enumerate(os.listdir(rootpath)):
        print(i)
        if flag in i:
            sorted_files = sorted(os.listdir(os.path.join(rootpath, i)))
            for jindex, j in enumerate(sorted_files):
                imgfile = os.path.join(rootpath, i, j)
                img = cv2.imread(imgfile)
                if jindex==0 :
                    img_h=img.copy()
                else:
                    img_h = cv2.hconcat([img_h, img])
            if iindex==0 or iindex==8:
                img_v=img_h.copy()
            else:
                img_v = cv2.vconcat([img_v, img_h])
    cv2.imshow("img_class", img_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(savename,img_v)

def image_show_views(rootpath,savename):
    #可视化图像，每个类别一行
    for iindex, i in enumerate(os.listdir(rootpath)):
        print(i)
        # img_v=[]
        sorted_files = sorted(os.listdir(os.path.join(rootpath, i)))
        f=0
        for jindex, j in enumerate(sorted_files):
            # img_h=[]
            imgfile = os.path.join(rootpath, i, j)
            img = cv2.imread(imgfile)
            if jindex==0 or jindex==18:
                img_h=img.copy()
            else:
                img_h = cv2.hconcat([img_h, img])
            if jindex==17 or jindex==35:
                if iindex==0:
                    if f==0:
                        img_v=img_h.copy()
                        f=1
                    else:
                        img_v = cv2.vconcat([img_v, img_h])
                else:
                    img_v = cv2.vconcat([img_v, img_h])
    cv2.imshow("img_class", img_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(savename,img_v)

def image_tolerate_trans(rootpath,maskpath,savepath,bg_color):
    bg_value = bg_color[0]
    for i in os.listdir(rootpath):
        imgfile = os.path.join(rootpath, i)
        img = cv2.imread(imgfile)
        img=cv2.resize(img,None,fx=2, fy=2)

        img=img[:,:,0:3]
        (h, w) = img.shape[:2]  # 获取图像的高度和宽度
        # 创建文件夹
        imgname=i.split('.')[0]
        folder_path=os.path.join(savepath,imgname)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # 0. 保存原始图像，2倍大小
        cv2.imwrite(os.path.join(savepath,imgname,'tolerate_'+imgname+'_init.png'),img)

        # 1. 尺度变化
        resize_list=[0.5,0.75,1.5]
        for r in resize_list:
            img_resize = cv2.resize(img, None, fx=r, fy=r)
            cv2.imwrite(os.path.join(savepath,imgname,'tolerate_'+imgname+'_resize_{}.png'.format(r)),img_resize)

        # 2. 位置变化
        background_ = np.full((img.shape[0]*2, img.shape[0]*2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
        background_h, background_w = background_.shape[:2]
        x_offset = 0  # x 方向起始位置
        y_offset = (background_h - h) // 2  # y 方向起始位置
        background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
        cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+imgname + '_location_{}.png'.format('left')), background_)

        background_ = np.full((img.shape[0]*2, img.shape[0]*2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
        background_h, background_w = background_.shape[:2]
        x_offset = w  # x 方向起始位置
        y_offset = (background_h - h) // 2  # y 方向起始位置
        background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
        cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+imgname + '_location_{}.png'.format('right')), background_)

        background_ = np.full((img.shape[0]*2, img.shape[0]*2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
        background_h, background_w = background_.shape[:2]
        x_offset = (background_w - w) // 2  # x 方向起始位置
        y_offset = 0  # y 方向起始位置
        background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
        cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+imgname + '_location_{}.png'.format('up')), background_)

        background_ = np.full((img.shape[0]*2, img.shape[0]*2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
        background_h, background_w = background_.shape[:2]
        x_offset = (background_w - w) // 2  # x 方向起始位置
        y_offset = h  # y 方向起始位置
        background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
        cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+imgname + '_location_{}.png'.format('down')), background_)

        # 3. 旋转变化
        rotation_list=[45,90,135,180,225,270,315]
        for r in rotation_list:
            background_ = np.full((img.shape[0] * 2, img.shape[0] * 2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
            background_h, background_w = background_.shape[:2]
            center = (background_w // 2, background_h// 2)  # 旋转中心点（图像中心）
            angle = r  # 旋转角度（逆时针）
            scale = 1.0  # 缩放比例
            x_offset = (background_w - w) // 2  # x 方向起始位置
            y_offset = (background_h - h) // 2  # y 方向起始位置
            background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_image = cv2.warpAffine(background_, rotation_matrix, (background_w, background_h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+imgname + '_rotation_{}.png'.format(r)), rotated_image)

        # 4. 模糊变化
        blurred_list=[51]
        for b in blurred_list:
            gaussian_blurred = cv2.GaussianBlur(img, (b, b), 100)
            cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+imgname + '_blurred_{}.png'.format(b)), gaussian_blurred)

        #5. 灰度化
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(savepath,imgname,'tolerate_'+imgname+'_gray.png'),gray_image)

        #6. 轮廓化
        maskfile = os.path.join(maskpath,i.split('_')[1],i)
        mask = cv2.imread(maskfile)
        mask = cv2.resize(mask, None, fx=2, fy=2)
        cv2.imwrite(os.path.join(savepath,imgname,'tolerate_'+imgname+'_bina.png'),mask)

        #7. 干扰
        for j in os.listdir(rootpath):
            img_jc=cv2.imread(os.path.join(rootpath,j))
            img_jc = cv2.resize(img_jc, None, fx=2, fy=2)

            background_ = np.full((img.shape[0] * 2, img.shape[0] * 2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
            background_h, background_w = background_.shape[:2]
            x_offset = 0  # x 方向起始位置
            y_offset = (background_h - h) // 2  # y 方向起始位置
            background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
            x_offset = w  # x 方向起始位置
            y_offset = (background_h - h) // 2  # y 方向起始位置
            background_[y_offset:y_offset + h, x_offset:x_offset + w] = img_jc
            cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+'lr_'+imgname +'+'+ j), background_)

            background_ = np.full((img.shape[0] * 2, img.shape[0] * 2, 3), bg_value, dtype=np.uint8)  # 128 表示灰色
            background_h, background_w = background_.shape[:2]
            x_offset = (background_w - w) // 2  # x 方向起始位置
            y_offset = 0  # y 方向起始位置
            background_[y_offset:y_offset + h, x_offset:x_offset + w] = img
            x_offset = (background_w - w) // 2  # x 方向起始位置
            y_offset = h  # y 方向起始位置
            background_[y_offset:y_offset + h, x_offset:x_offset + w] = img_jc
            cv2.imwrite(os.path.join(savepath, imgname, 'tolerate_'+'ud_'+imgname +'+'+ j), background_)

def cat_minboundingRect(rootpath, savepath):
    for i in os.listdir(rootpath):
        imgfile = os.path.join(rootpath, i)
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
        mask = img[:, :, 3].copy()
        mask[mask < 200] = 0  # mask阈值化
        mask[mask >= 200] = 1

        # 寻找最面积最大的外接矩形
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            print('error!')
        area = [cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] for c in contours]
        cindex = np.argmax(area)
        x, y, w, h = cv2.boundingRect(contours[cindex])  # 计算点集最外面的矩形边界

        img_color = img[y:y + h, x:x + w, :]
        cv2.imwrite(os.path.join(savepath, i[0:-4] + '.png'), img_color)
def cat_minboundingRect(rootpath, i):

    x1,x2,y1,y2=[],[],[],[]
    for j in os.listdir(os.path.join(rootpath,i)):
        imgfile = os.path.join(rootpath, i,j)
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
        mask = img[:, :, 3].copy()
        mask[mask < 200] = 0  # mask阈值化
        mask[mask >= 200] = 1

        # 寻找最面积最大的外接矩形
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            print('error!')
        area = [cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] for c in contours]
        cindex = np.argmax(area)
        x, y, w, h = cv2.boundingRect(contours[cindex])  # 计算点集最外面的矩形边界
        x1.append(x)
        x2.append(x+w)
        y1.append(y)
        y2.append(y+h)
    x1=min(x1)
    x2 = max(x2)
    y1=min(y1)
    y2=max(y2)
    for j in os.listdir(os.path.join(rootpath,i)):
        imgfile = os.path.join(rootpath, i,j)
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
        img = img[y1:y2, x1:x2, :]
        cv2.imwrite(os.path.join(rootpath, i,j), img)


if __name__ == "__main__":

    bg_color=(128,128,128)
#################################################图像类别和属性#########################################
    #重命名
    rootpath=r'image\img_rgb'
    image_rename(rootpath)

    #将mask中前景变为黑颜色
    rootpath = r'image\img_mask'
    savepath=r'image\img_mask1'
    for i in os.listdir(rootpath):
        try:
            os.makedirs(os.path.join(savepath,i))
        except:
            pass
        for j in os.listdir(os.path.join(rootpath,i)):
            img = cv2.imread(os.path.join(rootpath,i,j))
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_image[gray_image>200]=0
            cv2.imwrite(os.path.join(savepath, i, j), gray_image)


    #对一个文件夹中的图像，根据mask（最小边界框）裁剪
    # rootpath=r'H:\Code-Acute\config\image\image\img_init\inanimate_warship'
    # savepath=r'C:\Users\pt\Desktop\a'
    # cat_minboundingRect(rootpath, savepath)

    #将一个视角类别的图像缩放到同一尺度，保持视角中心不变
    rootpath = r'H:\Image_Paper1\image\img\img_view\img_init\a'
    cat_minboundingRect(rootpath,'cowboyhat')

    # 将图像转为rgb、gray和mask，resize+padding，保存
    rootpath=r'image\img_init'
    target_size=[224,224]
    image_resize_padding(rootpath,target_size,bg_color)

    # 可视化类别图像
    rootpath = r'image\img_rgb'
    image_show_classes(rootpath, 'class')
##############################################视角不变性##################################################
    # 重命名
    # rootpath = r'C:\Users\pt\Desktop\3d_image'
    # image_rename(rootpath)

    #将一个类别的视角图像统一到最小尺寸
    # rootpath = r'C:\Users\pt\Desktop\3d_image'
    # savepath = r'C:\Users\pt\Desktop\img_init'
    # cat_minboundingRect(rootpath, savepath)

    # 将所有视角图像resize到224*224
    rootpath = r'H:\Image_Paper1\image\img\img_view\img_init\a'
    savepath= r'H:\Image_Paper1\image\img\img_view\img_init\b'
    target_size = [224, 224]
    image_resize_padding_view(rootpath,savepath, target_size,bg_color)

    # 可视化视角图像
    # rootpath = r'H:\Code_Acute_Image\image\img_data_2\img_view'
    # savename = r'results\image_all_views.jpg'
    # image_show_classes(rootpath, savename)
##############################################图像容忍性##################################################
    # 对图像进行一系列身份保留的变换
    rootpath=r'G:\Code_Acute_Image\image\img\img_tolerate\img_init'
    maskpath=r'G:\Code_Acute_Image\image\img\img_view\img_mask'
    savepath=r'G:\Code_Acute_Image\image\img\img_tolerate'
    image_tolerate_trans(rootpath,maskpath,savepath,bg_color)

################################################改变图像背景颜色#############################################
    #
    # rootpath=r'H:\Code_Acute_Image\image\img_data_4\img_class'
    # for i in os.listdir(rootpath):
    #     for j in os.listdir(os.path.join(rootpath,i)):
    #         imgfile = os.path.join(rootpath, i,j)
    #         print(imgfile)
    #         img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
    #         mask = np.all(img == np.array([128, 128, 128]), axis=2)
    #         # 将掩码中为True的像素设置为64, 64, 64
    #         img[mask] = [64, 64, 64]
    #         cv2.imwrite(os.path.join(rootpath, i,j), img)
