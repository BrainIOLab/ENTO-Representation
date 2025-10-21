clc;clear;close all
addpath(genpath('tools'));
%% ѡ������
[nevfile,nevpath,~] = uigetfile('*.nev','select the date file','data'); 
imgclass_config=load(fullfile(nevpath,'config.mat')).image_config;                       % ���������ļ�; 
imgclass_config.spk_inter = [-3 1];                                                      % s��ȡ�̼����������ڵ�spike
imgclass_config.channels = 32;                                                           % ͨ����
imgclass_config.electrode=[2 6 10 14 18 22 26 30 ...
                          4 8 12 16 20 24 28 32 ...
                          1 5 9  13 17 21 25 29 ...
                          3 7 11 15 19 23 27 31];
imgclass_config.windowon = 3050:3500;                                                    % on��Ӧʱ������
imgclass_config.windowoff = 1:3000;                                                      % off��Ӧʱ������
%% ��ȡspike
[spikedata,refs] = f_spk_readwire(fullfile(nevpath,nevfile));     
if imgclass_config.makernum~=length(refs)
    df=diff(refs);
    disp(imgclass_config.makernum)
    error('������.');
end
bin_spike_all = f_spk_raster(spikedata,refs,imgclass_config);  
%% �����ļ��У�����ʵ����
expname=imgclass_config.experiment;                                                     % ʵ������
parts = split(expname, '_');
imgclass_config.expname = [parts{1} '_' parts{2}];
mkdir(fullfile('results','neuron_response',imgclass_config.expname));                   % �ؽ��ļ��У�����ʵ����
imgclass_config.savepath=fullfile('results','neuron_response',imgclass_config.expname);
save(fullfile(imgclass_config.savepath,[imgclass_config.expname,'_config.mat']),'imgclass_config'); % ����bin_spike_all
%% ���ӻ�
a_imgclass(bin_spike_all,imgclass_config);