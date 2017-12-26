
addpath(genpath('../common'));
addpath(genpath('../Clustering'));
addpath(genpath('../sortingQuality'));
reboot;

Nfilt = 32;
th1Range = [4, 6, 8, 10];
th23Range = [11, 13, 15];
lam1Range = 10;
lam23Range = [30, 40, 50];
baseName = 'C:/Users/User/Desktop/Continuous_400_cut/';
disp('Starting parameter grid sorting quality');

for th1 = th1Range
    for th23 = th23Range
        for lam1 = lam1Range
            for lam23 = lam23Range
                % params = PB072517_clusters2();
                params = PB101917();
                % clusters = loadKiloSortClusters(params.SiProbeFolder, params.SiProbeSamplingInterval);
                
                currentFolderName = [baseName, num2str(Nfilt), '_filters/', ...
                    'Th_', num2str(th1), '_', num2str(th23), ...
                    '_', num2str(th23), '_Lam_', num2str(lam1), '_', ...
                    num2str(lam23), '_', num2str(lam23), '/'];
                params.SiProbeFolder = currentFolderName;
                
                disp('***************************')
                disp('***************************')
                disp(['Calculating cluster properties for parameter set ', currentFolderName])
                
                % write fake phy cluster group file; required for sorting
                % quality library
                clusters = loadKS(params.SiProbeFolder, 'dev', {'unsorted'}, params.SiProbeSamplingRate);
                clusterFile = fopen([params.SiProbeFolder, 'cluster_group.tsv'], 'w');
                for i = 0:length(clusters)
                    if i == 0
                        fprintf(clusterFile, 'cluster_id\tgroup\n');
                    else
                        fprintf(clusterFile, [num2str(clusters(i).clusterID), '\tgood\n']);
                    end
                end
                
                [clusterWaveforms, WFTimeAxes] = load_average_cluster_waveform(clusters, params, [baseName, 'amplifier_cut.dat'], 10);
                spikeWidthsBaseline = [];
                spikeWidthsMinMax = [];
                spikeWidthsMinMax2 = [];
                spikeAmpMin = [];
                spikeAmpMinMax = [];
                interpolationFactor = 100;
                % figure(1);
                % hold on;
                for i = 1:length(clusterWaveforms(:,1))
                    resampledWF = interp(clusterWaveforms(i,:), interpolationFactor);
                    resampledTime = interp(WFTimeAxes(i,:), interpolationFactor);
                    baseline = median(resampledWF(1:1000));
                    [minV, minBin] = min(resampledWF);
                    [maxV, maxBin] = max(resampledWF(minBin:end));
                    maxBin = min(maxBin + minBin, length(resampledWF));
                    halfMaxBaseline = 0.5*(baseline + minV);
                    halfMaxMinMax = 0.5*(maxV + minV);
                    halfMaxBaselineWindow = resampledWF <= halfMaxBaseline;
                    halfMaxMinMaxWindow = resampledWF <= halfMaxMinMax;
                    FWHMBaseline = max(resampledTime(halfMaxBaselineWindow)) - min(resampledTime(halfMaxBaselineWindow));
                    FWHMMinMax = max(resampledTime(halfMaxMinMaxWindow)) - min(resampledTime(halfMaxMinMaxWindow));
                    spikeWidthsBaseline = [spikeWidthsBaseline FWHMBaseline];
                    spikeWidthsMinMax = [spikeWidthsMinMax FWHMMinMax];
                    spikeWidthsMinMax2 = [spikeWidthsMinMax2 resampledTime(maxBin) - resampledTime(minBin)];
                    spikeAmpMin = [spikeAmpMin abs(min(clusterWaveforms(i,:)))];
                    spikeAmpMinMax = [spikeAmpMinMax (max(clusterWaveforms(i,:)) - min(clusterWaveforms(i,:)))];
                %     plot(resampledTime, resampledWF);
                end
                % figure(1);
                % hold on;
                % for i = 1:length(clusterWaveforms(:,1))
                %     plot(WFTimeAxes(i,:), clusterWaveforms(i,:));
                % end

%                 clust_id = readNPY(fullfile(params.SiProbeFolder,'spike_clusters.npy'));
%                 clust_group = importdata(fullfile(params.SiProbeFolder,'cluster_group.tsv'));
%                 clusterLabels = cell(length(clust_group) - 1, 2);
%                 for i = 1:length(clust_group) - 1
%                     line = textscan(clust_group{i+1},'%d %s');
%                     clusterLabels{i, 1} = line{2};
%                     clusterLabels{i, 2} = line{1};
%                 end
% 
%                 goodClusterNames = strcmp([clusterLabels{:,1}],'good');
%                 goodSpikeIndices = ismember(clust_id, [clusterLabels{goodClusterNames,2}]);
%                 goodClusterIDs = unique(clust_id(goodSpikeIndices));

                refractoryPeriod = [0.001 0.0015 0.002];
                minISI = params.SiProbeSamplingInterval;
                isiViolations = {};
                for i = 1:length(refractoryPeriod)
                    rp = refractoryPeriod(i);
                    isiViolations{i} = sqKilosort.isiViolations(params.SiProbeFolder, rp, minISI);
                end

                firingRates = [];
                ISIFiringRates = [];
                ISICVs = [];
                fprintf('Cluster ID\tFiring rate (Hz)\n');
                for i = 1:length(clusters)
                    nSpikes = length(clusters(i).spikeTimes);
                    tMin = min(clusters(i).spikeTimes);
                    tMax = max(clusters(i).spikeTimes);
                    firingRate = nSpikes/(tMax - tMin);
                    firingRates = [firingRates firingRate];
                    fprintf('%d\t%.2f\n', clusters(i).clusterID, firingRate);
                    ISIFiringRates = [ISIFiringRates 1.0/mean(diff(clusters(i).spikeTimes))];
                %     ISICVs = [ISICVs std(diff(clusters(i).spikeTimes))/mean(diff(clusters(i).spikeTimes))];
                    % robust version:
                    ISICVs = [ISICVs 1.4826*mad(diff(clusters(i).spikeTimes), 1)/median(diff(clusters(i).spikeTimes))];
                end

                [allClusterIDs, unitQuality, contaminationRate, LRatio] = sqKilosort.maskedClusterQuality(params.SiProbeFolder);
                allClusterIDs = allClusterIDs - 1; % KS vs. our convention
%                 unitQuality_goodclusters = unitQuality(find(ismember(allClusterIDs,goodClusterIDs))); % isolation distance
%                 contaminationRate_goodclusters = contaminationRate(find(ismember(allClusterIDs,goodClusterIDs)));
%                 LRatio_goodclusters = LRatio(find(ismember(allClusterIDs,goodClusterIDs)));

                outName = [params.SiProbeFolder, 'cluster_properties.csv'];
                outFile = fopen(outName, 'w');
                if outFile == -1
                    ME = MException('MyComponent:InvalidOutputName', 'Cannot open file %s', outFile);
                    throw(ME)
                end

                fprintf(outFile, 'Cluster ID\tshank\tISI viol (%.1fms)\tISI viol (%.1fms)\tISI viol (%.1fms)\tIsolation dist\tContamination rate\tL Ratio\tFiring rate (Hz)\tISI CV\tSpike width (half baseline)\tSpike width (half min-max)\tSpike width (tMax - tMin)\tSpike amp (min)\tSpike amp (max-min)\n', 1000*refractoryPeriod(1), 1000*refractoryPeriod(2), 1000*refractoryPeriod(3));
                for i = 1:length(clusters)
                    fprintf(outFile, '%d\t%d\t%.3f\t%.3f\t%.3f\t%.1f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n',...
                        clusters(i).clusterID, clusters(i).shank, isiViolations{1}(1,i), isiViolations{2}(1,i), isiViolations{3}(1,i),...
                        unitQuality(i), contaminationRate(i), LRatio(i),...
                        firingRates(i), ISICVs(i), spikeWidthsBaseline(i), spikeWidthsMinMax(i), spikeWidthsMinMax2(i),...
                        spikeAmpMin(i), spikeAmpMinMax(i));
                end
                fclose(outFile);
            end
        end
    end
end
