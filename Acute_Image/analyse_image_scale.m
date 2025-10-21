clc;clear;close all
addpath(genpath('tools'));
%% 选择数据
[nevfile,nevpath,~] = uigetfile('*.nev','select the date file','data'); 
analyse_config=load(fullfile(nevpath,'config.mat')).image_config;                       % 加载配置文件; 
analyse_config.spk_inter = [-3 1];                                                      % s，取刺激后若干秒内的spike
analyse_config.channels = 32;        
analyse_config.electrode=[2 6 10 14 18 22 26 30 ...
                          4 8 12 16 20 24 28 32 ...
                          1 5 9  13 17 21 25 29 ...
                          3 7 11 15 19 23 27 31];
analyse_config.windowon = 3050:3500;                                                    % on响应时间区间
analyse_config.windowoff = 1:3000;                                                      % off响应
%% 读取spike
[spikedata,refs] = f_spk_readwire(fullfile(nevpath,nevfile));     
if analyse_config.makernum~=length(refs)
    disp(analyse_config.makernum*length(analyse_config.scale_list));
    df=diff(refs);
    error('打标错误.');
end
bin_spike_all = f_spk_raster(spikedata,refs,analyse_config);  
%%
expname=analyse_config.experiment;                                                     % 实验名称
parts = split(expname, '_');
expname = [parts{1} '_' parts{2}];

bin_spike  = bin_spike_all(:,1:2:end,:);                     % 取出刺激帧
[label,index] = sort(analyse_config.stimulateorder);
raster=bin_spike(:,index,:);
resp_on=squeeze(mean(raster(analyse_config.windowon(1):analyse_config.windowon(end),:,:),1))*1000;    
resp_off=squeeze(mean(raster(analyse_config.windowoff(1):analyse_config.windowoff(end),:,:),1))*1000;    
resp=resp_on-resp_off;
resp=reshape(resp,length(analyse_config.scale),length(analyse_config.classname),analyse_config.channels);
raster_trial=reshape(raster,size(raster,1),length(analyse_config.scale),length(analyse_config.classname),analyse_config.channels);
save(fullfile('results','pre_experiment','scale',[expname,'_scale_resp.mat']), 'resp');  
save(fullfile('results','pre_experiment','scale',[expname,'_scale_raster.mat']), 'raster_trial');  
%% Raster
figure('units','normalized','outerposition',[0 0.1 1 0.8]); 
t = tiledlayout(4,8,'TileSpacing','Compact','Padding','Compact');
for ch = 1:analyse_config.channels
    nexttile
    raster_trial=raster(:,:,analyse_config.electrode(ch));
    yyaxis left     %激活左坐标
    imagesc(~raster_trial')
    colormap('gray')
    xlabel('时间（ms）')
    ylabel('位置')
    
    yyaxis right     %激活左坐标
    raster_trial=f_sl_binsl(raster_trial,50,1);
    raster_trial=mean(raster_trial,2);
    raster_trial=smooth(raster_trial);
    plot(raster_trial)
    title(strcat('ch-',num2str(analyse_config.electrode(ch))))
end

%% 尺度比较
figure('units','normalized','outerposition',[0 0.1 1 0.8]); 
t = tiledlayout(4,8,'TileSpacing','Compact','Padding','Compact');
for ch = 1:analyse_config.channels
    nexttile
    resp_=resp(:,:,analyse_config.electrode(ch));
    hold on
    for i = 1:length(analyse_config.classname)
        plot(resp_(:,i))
    end
    title(strcat('ch-',num2str(analyse_config.electrode(ch))))
end

%% 绘制箱线图
figure('units','normalized','outerposition',[0 0.1 1 0.8]); 
t = tiledlayout(4,8,'TileSpacing','Compact','Padding','Compact');
for ch = 1:analyse_config.channels
    nexttile
    resp_=resp(:,:,analyse_config.electrode(ch));
    hold on
    boxplot(resp_')
    title(strcat('ch-',num2str(analyse_config.electrode(ch))))
end

