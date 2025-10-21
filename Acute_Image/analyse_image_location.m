clc;clear;close all
addpath(genpath('tools'));
%% 选择数据
[nevfile,nevpath,~] = uigetfile('*.nev','select the date file','data'); 
analyse_config=load(fullfile(nevpath,'config.mat')).image_config;                       % 加载配置文件; 
analyse_config.spk_inter = [0 1];                                                       % s，取刺激后若干秒内的spike
analyse_config.channels = 32;        
analyse_config.electrode=[2 6 10 14 18 22 26 30 ...
                          4 8 12 16 20 24 28 32 ...
                          1 5 9  13 17 21 25 29 ...
                          3 7 11 15 19 23 27 31];
analyse_config.windowon = 50:500;           %150:500                                    % on响应时间区间
analyse_config.windowoff = 550:1000;                                                     % off响应
%% 读取spike
[spikedata,refs] = f_spk_readwire(fullfile(nevpath,nevfile));     
if analyse_config.makernum~=length(refs)
    disp(analyse_config.makernum);
    df=diff(refs);
    error('打标错误.');
end
bin_spike_all = f_spk_raster(spikedata,refs,analyse_config);  
%%
bin_spike  = bin_spike_all(:,1:2:end,:);                     % 取出刺激帧
[label,index] = sortrows(analyse_config.stimulateorder, [2, 1]);
raster=bin_spike(:,index,:);
resp=squeeze(mean(raster(analyse_config.windowon(1):analyse_config.windowon(end),:,:),1))*1000;    
resp=reshape(resp,length(analyse_config.indexs),length(analyse_config.classname),analyse_config.channels);

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

%% 位置比较
figure('units','normalized','outerposition',[0 0.1 1 0.8]); 
t = tiledlayout(4,8,'TileSpacing','Compact','Padding','Compact');
for ch = 1:analyse_config.channels
    nexttile
    resp_loc=resp(:,:,analyse_config.electrode(ch));
    resp_loc=mean(resp_loc,2);
    resp_loc=reshape(resp_loc,5,7);
    imagesc(resp_loc)
    colorbar
    title(strcat('ch-',num2str(analyse_config.electrode(ch))))
end
%% 类别比较
figure('units','normalized','outerposition',[0 0.1 1 0.8]); 
t = tiledlayout(4,8,'TileSpacing','Compact','Padding','Compact');
for ch = 1:analyse_config.channels
    nexttile
    resp_loc=resp(:,:,analyse_config.electrode(ch));
    hold on
    for i = 1:length(analyse_config.indexs)
        plot(resp_loc(i,:))
    end
    title(strcat('ch-',num2str(analyse_config.electrode(ch))))
end





