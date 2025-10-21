clc;clear;close all
addpath(genpath('tools'));
%% 选择数据
[nevfile,nevpath,~] = uigetfile('*.nev','select the date file','data'); 
imgclass_config=load(fullfile(nevpath,'config.mat')).image_config;                       % 加载配置文件; 
imgclass_config.spk_inter = [-3 1];                                                      % s，取刺激后若干秒内的spike
imgclass_config.channels = 32;                                                           % 通道数
imgclass_config.electrode=[2 6 10 14 18 22 26 30 ...
                          4 8 12 16 20 24 28 32 ...
                          1 5 9  13 17 21 25 29 ...
                          3 7 11 15 19 23 27 31];
imgclass_config.windowon = 3050:3500;                                                    % on响应时间区间
imgclass_config.windowoff = 1:3000;                                                      % off响应时间区间
%% 读取spike
[spikedata,refs] = f_spk_readwire(fullfile(nevpath,nevfile));     
if imgclass_config.makernum~=length(refs)
    df=diff(refs);
    disp(imgclass_config.makernum)
    error('打标错误.');
end
bin_spike_all = f_spk_raster(spikedata,refs,imgclass_config);  
%% 创建文件夹，保存实验结果
expname=imgclass_config.experiment;                                                     % 实验名称
parts = split(expname, '_');
imgclass_config.expname = [parts{1} '_' parts{2}];
mkdir(fullfile('results','neuron_response',imgclass_config.expname));                   % 重建文件夹，保存实验结果
imgclass_config.savepath=fullfile('results','neuron_response',imgclass_config.expname);
save(fullfile(imgclass_config.savepath,[imgclass_config.expname,'_config.mat']),'imgclass_config'); % 保存bin_spike_all
%% 可视化
a_imgclass(bin_spike_all,imgclass_config);