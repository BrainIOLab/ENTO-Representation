%% 播放ONOFF刺激,打标通过pci-e上升沿触发（从0跳变为1）
clear;close all;clc
addpath(genpath('tools'));                                                              %子函数
%% 设置实验参数
movefield_config.experiment = '20241121_ento_reexp1_movefield_1';                       %单个颜色实验，实验名称
movefield_config.color_index = load('config/color.mat').color_map;                      %颜色索引
movefield_config.region =[1 1 3840 2160];                                               %x1、y1、x2、y2
movefield_config.bgcolor_list={'gray'};                                                 %背景颜色
movefield_config.stcolor_list={'fcolor'};                                               %刺激颜色.
movefield_config.objsize = 40;                                                          %刺激尺寸
movefield_config.interval = 45;                                                         %间隔
movefield_config.stimulatevel = 15;                                                     %像素/帧，控制刺激的速度
movefield_config.darktime = 500;                                                      	%ms，每个休息持续的时间。
movefield_config.stimulatefps = 100;                                                    %Hz,屏幕显示频率。
movefield_config.ioflag=1;                                                              %是否使用io，用来测试程序
[movefield_config.coord1,movefield_config.stimulateorder1,...
 movefield_config.coord2,movefield_config.stimulateorder2] ...                        	%播放的坐标和次序
        = s_order_movefield(movefield_config);
movefield_config.makernum=(length(movefield_config.stimulateorder1)+...               	%打标数量
    length(movefield_config.stimulateorder2))*2;      
% 计算实验时间
time=length(movefield_config.bgcolor_list)*length(movefield_config.stcolor_list)*...
    (size(movefield_config.coord1,1)*size(movefield_config.coord1,2)*4*1/movefield_config.stimulatefps+...
    movefield_config.makernum/2*movefield_config.darktime/1000)/60;
disp(strcat('刺激耗时：',num2str(time),'分钟'))
pause(1)
%% 检测文件是否存在，不存在创建，存在终止运行
if ~exist(fullfile('data',movefield_config.experiment))
    mkdir(fullfile('data',movefield_config.experiment));                                %创建文件夹
else
    error('已存在该实验编号')
end
save(fullfile('data',movefield_config.experiment,'config.mat'),'movefield_config');     %保存配置文件
%% 运动感受野实验
w=s_exp_init(movefield_config);                                                         %初始化测试程序
movefield_config.bgcolor=movefield_config.bgcolor_list{1};
movefield_config.stcolor=movefield_config.stcolor_list{1};
s_exp_movefield(movefield_config,w)                                                     %播放刺激
s_exp_end(w);                                                                           %结束测试程序,鼠标右键退出
