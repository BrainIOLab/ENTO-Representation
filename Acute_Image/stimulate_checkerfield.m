%% 播放ONOFF刺激,打标通过pci-e上升沿触发（从0跳变为1）
clear;close all;clc
addpath(genpath('tools'));                                                                      %子函数
%% 设置实验参数
checkerfield_config.experiment='ento_exp1_20250325_checkerfield_1';                                   %单个颜色实验，实验名称
checkerfield_config.color_index = load('config/color.mat').color_map;                           %颜色索引
checkerfield_config.region =[1 1 3840 2160];                                                    %x1、y1、x2、y2
checkerfield_config.bgcolor_list={'gray'};                                                      %背景颜色
checkerfield_config.stcolor_list={'fcolor'};                                                    %刺激颜色
checkerfield_config.objsize=40;                                                                 %刺激尺寸
checkerfield_config.stimulatevel = 120;                                                         %刺激间隔
checkerfield_config.scale=1;                                                                    %刺激尺度
checkerfield_config.angle=0;                                                                    %刺激角度
checkerfield_config.stimulatetime =500;                                                        %ms，每个刺激持续的时间。
checkerfield_config.darktime = 500;                                                             %ms，每个休息持续的时间。
checkerfield_config.stimulatefps = 100;                                                         %Hz,屏幕显示频率。
checkerfield_config.ioflag=1;                                                                   %是否使用io，用来测试程序
[checkerfield_config.coords,checkerfield_config.stimulateorder]...                              %播放的坐标和次序 
    = s_order_checkerfield(checkerfield_config);
checkerfield_config.makernum=length(checkerfield_config.stimulateorder)*2;                      %打标数量   
%计算刺激时间
time=length(checkerfield_config.bgcolor_list)*length(checkerfield_config.stcolor_list)*...
    length(checkerfield_config.stimulateorder)*(checkerfield_config.stimulatetime+checkerfield_config.darktime)/1000/60;
disp(strcat('刺激耗时：',num2str(time),'分钟'))
pause(1)
%% 检测文件是否存在，不存在创建，存在终止运行
if ~exist(fullfile('data',checkerfield_config.experiment))
    mkdir(fullfile('data',checkerfield_config.experiment));                                      %创建文件夹
else
    error('已存在该实验编号')
end
save(fullfile('data',checkerfield_config.experiment,'config.mat'),'checkerfield_config');        %保存配置文件
%% 棋盘格感受野实验
w=s_exp_init(checkerfield_config);                                                              
for i = 1:length(checkerfield_config.bgcolor_list)
    for j = 1:length(checkerfield_config.stcolor_list)
        checkerfield_config.bgcolor=checkerfield_config.bgcolor_list{i};
        checkerfield_config.stcolor=checkerfield_config.stcolor_list{j};
        s_exp_checkerfield(checkerfield_config,w);         
%         WaitSecs(30)
    end
end
s_exp_end(w);%结束测试程序,鼠标右键退出