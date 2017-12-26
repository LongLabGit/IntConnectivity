function params = PB101917()
% experiment parameters

disp('Loading parameters for experiment PB101917...')
params.clusterFolder = 'Z:\Robert\INT_connectivity\SiProbe\ProbeBird_101917\SiProbe\Continuous_400_cut\clusters_run1\';
% params.clusterFolder = 'C:\Users\User\Desktop\Continuous_400_cut_v1\clusters_run1\';
params.SiProbeFolder = 'Z:\Robert\INT_connectivity\SiProbe\ProbeBird_101917\SiProbe\Continuous_400_cut\';
% params.SiProbeFolder = 'C:\Users\User\Desktop\Continuous_400_cut_v1\';
params.WCFolder = 'Z:\Robert\INT_connectivity\SiProbe\ProbeBird_101917\WC\';
WCName1 = [params.WCFolder 'c1_0001.abf'];
params.WCNames{1} = WCName1;
params.STAFolder = 'Z:\Robert\INT_connectivity\SiProbe\ProbeBird_101917\SiProbe\Continuous_400_cut\clusters_run1\STA_INH\cell1\';
params.SiProbeSamplingRate = 2e4;
params.SiProbeSamplingInterval = 1/params.SiProbeSamplingRate;
params.WCSamplingRate = 5e4;
params.WCSamplingInterval = 1/params.WCSamplingRate;
params.WCChannels = [1 4];
params.WCROIsOnProbe = [0 310];
params.WCBounds = [0 300]; % periods of "good" recording quality in each file (in s)

end