clc;clear;close all
addpath(genpath('tools'));
%% 选择数据
[nevfile,nevpath,~] = uigetfile('*.nev','select the date file','data'); 
analyse_config = load(fullfile(nevpath,'config.mat')).checkerfield_config;              % 加载配置文件; 
analyse_config.spk_inter = [0 1];                                                       % s，取刺激后若干秒内的spike
analyse_config.channels = 32;                                                           % 通道数
analyse_config.electrode=[2 6 10 14 18 22 26 30 ...
                          4 8 12 16 20 24 28 32 ...
                          1 5 9  13 17 21 25 29 ...
                          3 7 11 15 19 23 27 31];
analyse_config.windowon = 50:500;                                                       % on响应时间区间
analyse_config.windowoff = 550:1000;                                                    % off响应时间区间
%% 读取spike
[spikedata,refs] = f_spk_readwire(fullfile(nevpath,nevfile));     
if analyse_config.makernum*length(analyse_config.stcolor_list)~=length(refs)
    disp(analyse_config.makernum*length(analyse_config.stcolor_list))
    df = diff(refs);
    error('打标错误.');
end
bin_spike_all = f_spk_raster(spikedata,refs,analyse_config);  
%% 可视化
para_checkerfield = a_checkerfield(bin_spike_all,analyse_config);


