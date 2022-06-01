clear all
clc
close all
% scene = 'static'
% scene = 'motion'
% scene = 'los'
% scene = 'nlos'


d1 = load(['static.mat']);
d2 = load(['motion.mat']);
d3 = load(['los.mat']);
d4 = load(['nlos.mat']);
% d4 = load(['static.mat']);

% T = data.newT;


% T=[d1.newT;d2.newT;d3.newT;d4.newT];

T=[d3.newT];
devlist = T.devices;


devices= unique(devlist);

n1 = 5;
n2 =2;
% tiledlayout(10,1)
% ax1 = nexttile;
% ax2 = nexttile;

for i=1:size(devices,1)
    
    
    curdev = devices(i)
    
    valid_rows = strcmp(devlist,curdev);
    
    
    curT = T(valid_rows,:);
    A = curT.rssi;
    TF = isoutlier(A,'mean');
    
    sanT = curT(TF==0,:);
    
    tx = sanT.trX;
    ty = sanT.trY;
    tz = sanT.trZ;
    rssi = sanT.rssi;
    
    markerSize=20;
%     subplot(1,1,i)
% ax = nexttile;
    scatter3(tx,ty,tz,markerSize,rssi)
%    view(2)
    colorbar
    title(curdev);
    saveas(gcf,['plots/' curdev{1} '_los.png'])
%  
%     keyboard
    
    
%     break
    
end

% suptitle (scene)
% saveas(gcf,['plots/' scene '.png'])

