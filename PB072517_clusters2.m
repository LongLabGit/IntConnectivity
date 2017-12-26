function params = PB072517()
% experiment parameters

disp('Loading parameters for experiment PB0702517...')
params.clusterFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\SiProbe\Continuous_500_cut\clusters_run2\';
params.SiProbeFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\SiProbe\Continuous_500_cut\';
params.WCFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\WC\';
WCName1 = [params.WCFolder 'c3_0001.abf'];
WCName2 = [params.WCFolder 'c3_0002.abf'];
params.WCNames{1} = WCName1;
params.WCNames{2} = WCName2;
params.STAFolder = 'Z:\Robert\INT_connectivity\SiProbe\PracticeBird_072517\SiProbe\Continuous_500_cut\clusters_run2\STA_INH\cell3\';
params.SiProbeSamplingRate = 2e4;
params.SiProbeSamplingInterval = 1/params.SiProbeSamplingRate;
params.WCSamplingRate = 5e4;
params.WCSamplingInterval = 1/params.WCSamplingRate;
params.WCChannels = [1 4];
params.WCROIsOnProbe = [4650 4865; 5000 5300];
params.WCBounds = [50 inf; 43 190]; % periods of "good" recording quality in each file (in s)

end