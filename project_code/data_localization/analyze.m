clear all
clc
close all

% scene='static'
% scene='nlos'
scene='los'
% 
% scene='behind_thewall'
% scene='motion'


real_file = [scene '.txt'];
real_data = importdata(real_file);
get_trans=real_data(:,1:3) ;
get_times = real_data(:,end);

% et = 1581617931060
e1 = datetime(get_times,'ConvertFrom','epochtime','TicksPerSecond',1000,'Format', 'MM/dd/yy HH:mm:ss.SSS','TimeZone','UTC');

e1.TimeZone = 'America/New_York';

e1;

read_pcap = [scene '.csv'];

pcap_data = importdata(read_pcap);

devices = pcap_data.textdata;
T = array2table(string(devices));

darr2=datetime( pcap_data.data(:,1), 'convertfrom','posixtime','TimeZone','America/New_York');


T.times =darr2;

T.rssi = pcap_data.data(:,2);

T(1,:)
T.Properties.VariableNames = {'devices' 'time' 'rssi'};
devs = unique(T.devices(:));
devs

devlist = T.devices;

newT = T;



newT(1,:)
todel=[];
trX=zeros(1,size(newT,1));
trY=zeros(1,size(newT,1));
trZ=zeros(1,size(newT,1));

poseX=zeros(1,size(newT,1));
poseY=zeros(1,size(newT,1));
poseZ=zeros(1,size(newT,1));
poseW=zeros(1,size(newT,1));


for j = 1:size(newT,1)
    
    if mod(j,100) ==0
        j
        size(newT)
    end
    
    newT(j,:);
    curtime  = newT.time(j);
    
    [minval,ind1] = min(abs(datenum(e1)-datenum(curtime)));
    
    cli  = e1(ind1);
    %         cli
    %         curtime
    ms1 = milliseconds ( abs (cli - curtime ) );
    
    if ms1 > 100
        todel=[todel;j];
    end
    trX(j) = real_data(ind1,1);
    trY(j) = real_data(ind1,2);
    trZ(j) = real_data(ind1,3);
    poseX(j) = real_data(ind1,4);
    poseY(j) = real_data(ind1,5);
    poseZ(j) = real_data(ind1,6);
    poseW(j) = real_data(ind1,7);
    
    
    %         abd = abs(datenum(e1)-datenum(curtime));
    %         abd
    %         cli
    
    %         break
end

newT.trX = trX';

newT.trY = trY';
newT.trZ = trZ';
newT.poseX = poseX';
newT.poseY = poseY';
newT.poseZ = poseZ';
newT.poseW = poseW';

newT(todel,:)=[];

save([scene '.mat'],'newT');

