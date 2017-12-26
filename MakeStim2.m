addpath(genpath('../common'));
addpath(genpath('../Utilities'));
reboot;
nTrials = 200;%number of trials
len_iti = 6;%intersound interval

baseFolder = 'Z:\Robert\INT_connectivity\SiProbe\BOS\surgery_110917_b1\111317\';
fileNames = dir([baseFolder '*.dat']);
for i = 1:length(fileNames)
    fname = [baseFolder fileNames(i).name];
    [audioFileRaw, samplingRate, dateandtime, label, props] = LoadEGUI_daq(fname, 1);
    % [wav, samplingRate]=audioread('S:\Bird_334-013.wav');%get data
    % wav=wav(9.03e5:1.052e6);
    audioFileRaw = audioFileRaw/max(abs(audioFileRaw));
    figure(i);
    vigiSpec(audioFileRaw,samplingRate);
end
% save wavFile.mat wav

%% generate motif and stimulus file
% 102717_b1
% selectedSong = 15;
% motifBounds = samplingRate*[10.9 11.57];
% 103017_b1
% selectedSong = 4;
% motifBounds = samplingRate*[5.984 6.602];
% 110917_b1
selectedSong = 5;
motifBounds = samplingRate*[9.504 10.38];
fname = [baseFolder fileNames(selectedSong).name];
[audioFileRaw, samplingRate, dateandtime, label, props] = LoadEGUI_daq(fname, 1);
audioFileRaw = audioFileRaw/max(abs(audioFileRaw));
croppedMotif = audioFileRaw(motifBounds(1):motifBounds(2));
figure(99);
vigiSpec(croppedMotif,samplingRate);
%%
% Freq=8000;%hz of signal
% len=.5;%length in time
% t=linspace(0,len,len*samplingRate);
% localizer=sin(2*pi*t*samplingRate*Freq);
% sound(localizer,fs)%test it 
stim=[];
for i=1:nTrials
    gap(i)=max(len_iti+rand,1);
    iti=zeros(round(gap(i)*samplingRate),1);
    % if the background noise in the selected motif is too high
    % maybe add white noise instead of zeros here
    stim=[stim;iti;croppedMotif];
end
gap(i+1)=len_iti+randn;
% stim=[stim;zeros(round(gap(i+1)*samplingRate),1);localizer'];
stim=[stim;zeros(round(gap(i+1)*samplingRate),1)];
audiowrite([baseFolder 'stim.wav'], stim, double(round(samplingRate)));
% audiowrite([baseFolder 'stim.wav'], stim, 40000);
% save gaps.mat gap stim
disp('done')
disp(length(stim)/samplingRate/60)

