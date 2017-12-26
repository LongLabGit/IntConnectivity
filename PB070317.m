function params = PB070317()
% experiment parameters

disp('Loading parameters for experiment PB070317...')
params.clusterFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_070317\SiProbe\Continuous_400_cut\clusters\';
params.SiProbeFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_070317\SiProbe\Continuous_400_cut\';
params.WCFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\WC\';
WCName1 = [params.WCFolder 'c2_0000.abf'];
WCName2 = [params.WCFolder 'c2_0001.abf'];
params.WCNames{1} = WCName1;
params.WCNames{2} = WCName2;
params.STAFolder = [params.clusterFolder 'STA_INH\cell2\'];
params.SiProbeSamplingRate = 2e4;
params.SiProbeSamplingInterval = 1/params.SiProbeSamplingRate;
params.WCSamplingRate = 5e4;
params.WCSamplingInterval = 1/params.WCSamplingRate;
params.WCChannels = [1 4];
params.WCROIsOnProbe = [2850 3150; 3150 3200];
params.WCBounds = [50 inf; 23 inf]; % periods of "good" recording quality in each file (in s)

end