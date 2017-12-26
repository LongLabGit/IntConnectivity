addpath('../Clustering/IO');
addpath('../common');
reboot;
% F='Z:\Robert\INT_connectivity\SiProbe\ProbeBird_101917\SiProbe\Continuous_400_171019_160417\';
F='Z:\Robert\INT_connectivity\SiProbe\ProbeBird_101917\SiProbe\Continuous_400_cut_v1\';
% F='Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\SiProbe\Continuous_500_170725_155349\';
% F='Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\SiProbe\Continuous_500_cut\';
% F='Z:\Robert\INT_connectivity\SiProbe\ProbeBird_102817\SiProbe\Continuous_475_171028_145233\';
% F='Z:\Robert\INT_connectivity\SiProbe\ProbeBird_102817\SiProbe\Continuous_475_cut\';
% F='Z:\Robert\INT_connectivity\SiProbe\ProbeBird_110217\SiProbe\Baseline_500_171102_144503\';
% analogFilename = 'analogin.dat';
analogFilename = 'analogin_cut.dat';
digitalOutFilename = 'analoginToDigitalin.dat';
maxPiece=10*60;%analyze X seconds at a time. here 10 minutes. use for RAM control
samplingRate = 20000; % in Hz
%% Auto detect imaging noise
read_Intan_RHD2000_file([F,'info.rhd'])
% channelInfo = read_Intan_RHD2000_channel_numbers([F,'info.rhd']);
nAdcChannels = length(board_adc_channels);
channelNumber = nAdcChannels;
analogfileinfo = dir([F,analogFilename]);
total_duration = (analogfileinfo.bytes/(nAdcChannels*2))/samplingRate; % int16 = 2 bytes, samples-->seconds
stop=0;indCut=0;
direction = +1; % up: +1; down: -1
threshold = 0.7;
digitalFile = fopen([F,digitalOutFilename], 'a');
sumAbove = 0;
while stop<total_duration
% while indCut < 1% 
% while length(ttlPulseTimes) == 0
    start=indCut*maxPiece;
    stop=min(total_duration,start+maxPiece);
    indCut=indCut+1;
    ttlPulses = direction*double(LoadBinary([F,analogFilename],'nChannels',nAdcChannels,'channels',channelNumber,'start',start,'duration',stop-start))*0.000050354;
%     ttlPulses = direction*double(LoadBinary([F,analogFilename],'nChannels',nAdcChannels,'channels',channelNumber,'start',start,'duration',stop-start));
    ttlDigital = double(ttlPulses > threshold);
    sumAbove = sumAbove + sum(ttlDigital);
    fwrite(digitalFile, ttlDigital', 'int16');
end
fclose(digitalFile);
disp('Done converting to digital file');
disp(['Converted ', num2str(sumAbove), ' samples to signal']);
