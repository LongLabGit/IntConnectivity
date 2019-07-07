addpath('../Clustering/IO');
addpath('../common');
reboot;
% F='S:\Robert\INT_connectivity\SiProbe\PracticeBird_061917\SiProbe\Continuous_500_170619_141301\';
F='F:\sorting\ProbeBird_050219\';
% F='S:\Vigi\Datasets\SiliconProbe\Masmanadis\ucla2\';
ampCutFilename = 'amplifier_cut.dat';
analogCutFilename = 'analogin_cut.dat';
digitalCutFilename = 'digitalin_cut.dat';
maxPiece=10*60;%analyze X seconds at a time. here 10 minutes. use for RAM control
%% Auto detect imaging noise
% read_Intan_RHD2000_file([F,'info.rhd'])
read_Intan_RHD2000_file_v2_01([F,'info.rhd'])
samplingRate = frequency_parameters.amplifier_sample_rate; % in Hz
fileinfo = dir([F,'amplifier.dat']);
% analogfileinfo = dir([F,'analogin.dat']);
% analogfileinfo = dir([F,'analoginToDigitalin.dat']);
total_duration = (fileinfo.bytes/(length(amplifier_channels) * 2))/samplingRate; % int16 = 2 bytes, samples-->seconds
stop=0;indCut=0;
d = designfilt('bandpassfir', ...       % Response type
       'FilterOrder',300, ...            % Filter order
       'StopbandFrequency1',500, ...    % Frequency constraints
       'PassbandFrequency1',600, ...
       'PassbandFrequency2',800, ...
       'StopbandFrequency2',1000, ...
       'DesignMethod','ls', ...         % Design method
       'StopbandWeight1',1, ...         % Design method options
       'PassbandWeight', 2, ...
       'StopbandWeight2',3, ...
       'SampleRate',samplingRate);               % Sample rate
yA=[];
tA=[];
% ttlPulseTimes = [];
while stop<total_duration% 
% while length(ttlPulseTimes) == 0
    start=indCut*maxPiece;
    stop=min(total_duration,start+maxPiece);
    v=LoadBinary([F,'amplifier.dat'], 'frequency', samplingRate, 'nChannels',length(amplifier_channels),'channels',1,...
        'start',start,'duration',stop-start);%SWAP TO ALL SHANKS?
    v=double(v)*.195;
    v2=filtfilt(d,v);
    indCut=indCut+1;
    
    [yupper,ylower] = envelope(v2,1e4,'rms');
    t=(start+((1:length(v))/samplingRate))/60;
%     figure(indCut);clf; hold on;plot(t,v2);plot(t,yupper);drawnow;
    yA=[yA;decimate(yupper,samplingRate/10)];%.1 hZ resolution
    tA=[tA,downsample(t,samplingRate/10)];
end
save([F,'Scanning_Times.mat'],'tA','yA');

% [SamplingInterval, AnalogVoltage] = loadAnalogSignalIntan(F, 'analogin.dat');
% ttlPulseTimes = detectThresholdCrossings((1:length(AnalogVoltage))/samplingRate, AnalogVoltage, 1.0);
figure;
plot(tA,yA);
% hold on;
% plot(ttlPulseTimes, repmat([10], 1, length(ttlPulseTimes)), 'r+');
axis tight;
xlabel('time (minutes)')
ylabel('RMS @ 600-800 Hz')
title('Imaging Detection')
%% load manual imaging noise
% load times.mat tA yA
nums = xlsread([F,'ScanningPeriods.xlsx']);
manualNoisePeriods = zeros(length(nums(:,1)),2); % in samples
for i = 1:length(nums(:,1))
    tmpStart = round((nums(i,1)*60 + nums(i,2) + nums(i,3)/1000)*samplingRate);
    tmpStop = round((nums(i,4)*60 + nums(i,5) + nums(i,6)/1000)*samplingRate);
    manualNoisePeriods(i,1) = tmpStart;
    manualNoisePeriods(i,2) = tmpStop;
end
% figure(1);clf;
% plot(tA, smooth(yA,20));
% hold on;
% id=zeros(size(tA));
% for i = 1:length(manualNoisePeriods)
%     line(manualNoisePeriods(i,:)/samplingRate/60, 3*[1 1], 'color','r');
%     plot(manualNoisePeriods(i,:)/samplingRate/60, 3*[1 1],'ko');
%     l=find(tA>manualNoisePeriods(i,1)/samplingRate/60,1,'first');
%     r=find(tA<manualNoisePeriods(i,2)/samplingRate/60,1,'last');
%     id(l:r)=1;
% end
% id=logical(id);
% plot(tA,id*3)
% 
% figure(2);clf;hold on;
% yAs=smooth(yA,50);
% % histogram(yAs(id),0:5:100)
% % histogram(yAs(~id),0:5:100)
% % plot data
% [X,Y,T,AUC,OPTROCPT] = perfcurve(id,yAs,1);
% [AUC,OPTROCPT];
% thresh=prctile(yAs(~id),1-OPTROCPT(1));
% plot(X,Y)
% xlabel('False positive rate')
% ylabel('True positive rate')
% title('ROC for Classification by BandPass Power')
% figure(3);clf;hold on;
% plot(tA,yAs-median(yAs))
% plot(tA,id)
% line(xlim,thresh*[1,1],'LineStyle',':')
%% apply cutting
fileinfo = dir([F,'amplifier.dat']);
tot_samples = (fileinfo.bytes/(length(amplifier_channels) * 2)); % int16 = 2 bytes, samples-->seconds
edgeLeft=[0;manualNoisePeriods(:,2)];
edgeRight=[manualNoisePeriods(:,1);tot_samples];
% edgeLeft=0;
% edgeRight=55.5*60*2e4;
keptPerc=sum(edgeRight-edgeLeft)/tot_samples
%%
ampFileCut = fopen([F,ampCutFilename], 'a');
for e = 1:length(edgeLeft)
% for e = 1:2
    start = edgeLeft(e);
    stop = edgeRight(e);
    indCut=0;
    right = start;
    while right < stop
        left = start+indCut*maxPiece*samplingRate;
        right = min(stop,left+maxPiece*samplingRate);
        dataChunk = LoadBinary([F,'amplifier.dat'],'frequency',samplingRate,'nChannels',length(amplifier_channels),'channels',1:length(amplifier_channels),...
            'start',left/samplingRate,'duration',(right-left)/samplingRate);%SWAP TO ALL SHANKS?
        fwrite(ampFileCut, dataChunk', 'int16');
        indCut=indCut+1;
    end
    fprintf([num2str(e) '/' num2str(length(edgeLeft)) ', '])
end
fclose(ampFileCut);
disp('done')

analogFileCut = fopen([F,analogCutFilename], 'a');
for e = 1:length(edgeLeft)
    start = edgeLeft(e);
    stop = edgeRight(e);
    indCut=0;
    right = start;
    while right < stop
        left = start+indCut*maxPiece*samplingRate;
        right = min(stop,left+maxPiece*samplingRate);
        dataChunk = LoadBinary([F,'analogin.dat'],'frequency',samplingRate,'nChannels',1,'channels',1,'start',left/samplingRate,'duration',(right-left)/samplingRate);
        fwrite(analogFileCut, dataChunk', 'int16');
        indCut=indCut+1;
    end
    fprintf([num2str(e) '/' num2str(length(edgeLeft)) ', '])
end
fclose(analogFileCut);
disp('done')

digitalFileCut = fopen([F,digitalCutFilename], 'a');
for e = 1:length(edgeLeft)
    start = edgeLeft(e);
    stop = edgeRight(e);
    indCut=0;
    right = start;
    while right < stop
        left = start+indCut*maxPiece*samplingRate;
        right = min(stop,left+maxPiece*samplingRate);
        dataChunk = LoadBinary([F,'digitalin.dat'],'frequency',samplingRate,'nChannels',1,'channels',1,'start',left/samplingRate,'duration',(right-left)/samplingRate);
        fwrite(digitalFileCut, dataChunk', 'int16');
        indCut=indCut+1;
    end
    fprintf([num2str(e) '/' num2str(length(edgeLeft)) ', '])
end
fclose(digitalFileCut);
disp('done')