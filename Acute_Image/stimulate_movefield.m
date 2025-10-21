%% ����ONOFF�̼�,���ͨ��pci-e�����ش�������0����Ϊ1��
clear;close all;clc
addpath(genpath('tools'));                                                              %�Ӻ���
%% ����ʵ�����
movefield_config.experiment = '20241121_ento_reexp1_movefield_1';                       %������ɫʵ�飬ʵ������
movefield_config.color_index = load('config/color.mat').color_map;                      %��ɫ����
movefield_config.region =[1 1 3840 2160];                                               %x1��y1��x2��y2
movefield_config.bgcolor_list={'gray'};                                                 %������ɫ
movefield_config.stcolor_list={'fcolor'};                                               %�̼���ɫ.
movefield_config.objsize = 40;                                                          %�̼��ߴ�
movefield_config.interval = 45;                                                         %���
movefield_config.stimulatevel = 15;                                                     %����/֡�����ƴ̼����ٶ�
movefield_config.darktime = 500;                                                      	%ms��ÿ����Ϣ������ʱ�䡣
movefield_config.stimulatefps = 100;                                                    %Hz,��Ļ��ʾƵ�ʡ�
movefield_config.ioflag=1;                                                              %�Ƿ�ʹ��io���������Գ���
[movefield_config.coord1,movefield_config.stimulateorder1,...
 movefield_config.coord2,movefield_config.stimulateorder2] ...                        	%���ŵ�����ʹ���
        = s_order_movefield(movefield_config);
movefield_config.makernum=(length(movefield_config.stimulateorder1)+...               	%�������
    length(movefield_config.stimulateorder2))*2;      
% ����ʵ��ʱ��
time=length(movefield_config.bgcolor_list)*length(movefield_config.stcolor_list)*...
    (size(movefield_config.coord1,1)*size(movefield_config.coord1,2)*4*1/movefield_config.stimulatefps+...
    movefield_config.makernum/2*movefield_config.darktime/1000)/60;
disp(strcat('�̼���ʱ��',num2str(time),'����'))
pause(1)
%% ����ļ��Ƿ���ڣ������ڴ�����������ֹ����
if ~exist(fullfile('data',movefield_config.experiment))
    mkdir(fullfile('data',movefield_config.experiment));                                %�����ļ���
else
    error('�Ѵ��ڸ�ʵ����')
end
save(fullfile('data',movefield_config.experiment,'config.mat'),'movefield_config');     %���������ļ�
%% �˶�����Ұʵ��
w=s_exp_init(movefield_config);                                                         %��ʼ�����Գ���
movefield_config.bgcolor=movefield_config.bgcolor_list{1};
movefield_config.stcolor=movefield_config.stcolor_list{1};
s_exp_movefield(movefield_config,w)                                                     %���Ŵ̼�
s_exp_end(w);                                                                           %�������Գ���,����Ҽ��˳�
