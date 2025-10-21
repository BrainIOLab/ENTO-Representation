%% 播放ONOFF刺激,打标通过pci-e上升沿触发（从0跳变为1）
clear;close all;clc
addpath(genpath('tools'));                                                              %子函数
%% 设置实验参数
image_config.experiment = 'ento_exp1_20250325_scale_2';                                 %实验名称
image_config.imagepath ='image\img_rgb';                                                %按照图像的名称依次读取
image_config.color_index = load('config/color.mat').color_map;                          %颜色索引
image_config.size=224;                                                                  %图像的原始尺寸
image_config.scale= [1, 1.5, 2, 2.5, 3];                                                %控制图像尺度
image_config.xcenter = 1920;                                                          %x坐标，中心点
image_config.ycenter = 1080;                                                          %y坐标，中心点
image_config.stimulatetime = 500;                                                       %ms，每个刺激持续的时间。
image_config.darktime = 29500;                                                          %ms，每个休息持续的时间。
image_config.stimulatefps = 100;                                                        %Hz,屏幕显示频率。
image_config.bgcolor_list={'gray'};                                                     %背景颜色
image_config.shuffle=1;                                                                 %是否打乱
[image_config.classname,image_config.classdata,image_config.stimulateorder,...          %index是y,x
    image_config.indexs] = s_order_imgscale(image_config);
image_config.makernum=length(image_config.stimulateorder)*2;                            %打标数量
image_config.ioflag = 1;                                                                %是否使用io，用来测试程序
%计算实验时间
time = ((image_config.darktime+image_config.stimulatetime)/1000)...
            *length(image_config.stimulateorder)/60;
disp(strcat('刺激耗时：',num2str(time),'分钟'))
pause(1)
%% 检测文件是否存在，不存在创建，存在终止运行
if ~exist(fullfile('data',image_config.experiment))
    mkdir(fullfile('data',image_config.experiment));                                  %创建文件夹
else
    error('已存在该实验编号')
end
save(fullfile('data',image_config.experiment,'config.mat'),'image_config');           %保存配置文件
%% 播放刺激
w=s_exp_init(image_config);                                                           %初始化测试程序
s_exp_imgscale(image_config,w);                                                       %播放刺激图像
s_exp_end(w);                                                                         %结束测试程序,鼠标右键退出
