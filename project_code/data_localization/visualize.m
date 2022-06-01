clear all
clc
close all
% scene = 'static'
% scene = 'motion'
% scene = 'los'
% scene = 'nlos'
scene = 'behind_thewall'


data = load([scene '.mat']);

T = data.newT;



devlist = T.devices;


devices= unique(devlist);

n1 = 5;
n2 =2;
for i=1:size(devices,1)
    
    
    curdev = devices(i)
    
    valid_rows = strcmp(devlist,curdev);
    
    
    curT = T(valid_rows,:);
    A = curT.rssi;
    TF = isoutlier(A,'mean');
    sanA = A(TF ==0);
    
    subplot(2,5,i)
    histogram(A);
    title(curdev);
%     break
    
end
suptitle (scene)
saveas(gcf,['plots/' scene '.png'])

