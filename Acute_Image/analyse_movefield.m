clc;clear;close all
addpath(genpath('tools'));
%% 选择数据
[nevfile,nevpath,~] = uigetfile('*.nev','select the date file','data'); 
analyse_config=load(fullfile(nevpath,'config.mat')).movefield_config;                                   % 加载配置文件; 
analyse_config.delay=0.1;                                                                               % 刺激到响应的延迟时间
analyse_config.spk_inter = [-1*analyse_config.darktime/1000 max(size(analyse_config.coord1,2),size(analyse_config.coord2,1))*1 ...
    /analyse_config.stimulatefps+analyse_config.darktime/1000]+analyse_config.delay;                    % s，取刺激后若干秒内的spike
analyse_config.channels = 32;
analyse_config.electrode=[31 27 23 19 15 11 7 3 ...
                          29 25 21 17 13 9 5 1 ...
                          32 28 24 20 16 12 8 4 ...
                          30 26 22 18 14 10 6 2];
analyse_config.figflag = 1;                                                                             % 是否绘制图像
%% 读取spike
[spikedata,refs] = f_spk_readwire(fullfile(nevpath,nevfile));     
if analyse_config.makernum*length(analyse_config.stcolor_list)~=length(refs)
    disp(analyse_config.makernum*length(analyse_config.stcolor_list))
    df=diff(refs);
    error('打标错误.');
end
bin_spike = f_spk_raster(spikedata,refs,analyse_config);  
%% 可视化
para_mrf=a_movefield(bin_spike,analyse_config);


