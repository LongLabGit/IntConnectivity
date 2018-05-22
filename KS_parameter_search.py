###################################
# (semi-) automated detection of
# "reference" unit in multiple runs
# of KiloSort.
# Used for analysis of systematic
# KiloSort parameter explorations
###################################

import numpy as np
import matplotlib.pyplot as plt
import os, os.path, sys, ast
import glob
import elephant as eph
import neo
import quantities as pq
import ClusterProcessing as clust
import ConnectionAnalyzer as ca
import spike_triggered_currents as STC

ClusteringSrcFolder = 'E:\\User\\project_src\\physiology\\Clustering'
# channelMapAllToShank4 = {56: 0, 58: 1, 63: 2, 61: 3, 59: 4, 44: 5, 46: 6, 40: 7}
channelMapAllToShank3Scaled = {56: 7, 58: 6, 63: 5, 61: 4, 59: 3, 44: 2, 46: 1, 40: 0}

def parse_KS_output_set(refPath, refClusterID, refTemplateName, baseFolder):
    # input:
    # reference unit: cluster object of reference unit
    # reference unit template: manually selected template
    # as good representation of reference unit (or avg. waveform?)
    # base folder: folder to iterate over, containing subfolders
    # which contain KS output for a specific parameter set each

    # load reference cluster and template/waveform
    refCluster = load_reference_cluster(refPath, refClusterID)
    # refClusterMaxChannel = channelMapAllToShank4[refCluster.maxChannel]
    refClusterMaxChannel = channelMapAllToShank3Scaled[refCluster.maxChannel]
    # refTemplate = refCluster.waveForm
    refTemplate = load_reference_cluster_template(refPath, refTemplateName)

    # iterate over folders within base folder:
    #   iterate over all KS clusters
    summaryData = {}
    folders_ = glob.glob(os.path.join(baseFolder, '*'))
    folders = []
    for folder in folders_:
        if not os.path.isdir(folder) or not 'Th' in folder or not 'Lam' in folder:
            continue
        else:
            folders.append(folder)

    # folders = ['C:\\Users\\User\\Desktop\\Continuous_400_cut\\32_filters\\Th_6_11_11_Lam_10_30_30']
    # folders = ['C:\\Users\\User\\Desktop\\Continuous_400_cut\\scale_1.00\\Th_6_13_13_Lam_10_30_30']
    # maxPrecision = 3
    allTemplateSimilarities = []
    for KSOutputFolder in folders:
        # if not os.path.isdir(KSOutputFolder):
        #     continue
        print 'Processing KiloSort output in folder %s' % KSOutputFolder
        summaryData[KSOutputFolder] = {}
        summaryData[KSOutputFolder]['clusters'] = []
        summaryData[KSOutputFolder]['templateMatch'] = []
        summaryData[KSOutputFolder]['spikeTrainMatch'] = []
        # load cluster waveform/template
        # load cluster spike times
        try:
            clusters = clust.reader.read_KS_clusters_unsorted(KSOutputFolder, ClusteringSrcFolder, 'dev', 20000.0)
        except:
            summaryData.pop(KSOutputFolder)
            continue
        for clusterID in clusters:
            cluster = clusters[clusterID]
            templateSimilarity = compute_template_similarity(cluster, refTemplate, refClusterMaxChannel)
            # spikeTrainDistance = compute_spiketrain_distance(cluster, refCluster)
            spikeTrainDistance = -1.0
            # dt = []
            # FP, FN, totalSpikes = compute_cluster_differences(cluster, refCluster, maxPrecision, dt)
            # nrMatchedSpikes = totalSpikes - FN
            # spikeTrainDistance = nrMatchedSpikes
            summaryData[KSOutputFolder]['clusters'].append(clusterID)
            summaryData[KSOutputFolder]['templateMatch'].append(templateSimilarity)
            summaryData[KSOutputFolder]['spikeTrainMatch'].append(spikeTrainDistance)
            allTemplateSimilarities.append(templateSimilarity)
        summaryData[KSOutputFolder]['clusters'] = np.array(summaryData[KSOutputFolder]['clusters'])
        summaryData[KSOutputFolder]['templateMatch'] = np.array(summaryData[KSOutputFolder]['templateMatch'])
        summaryData[KSOutputFolder]['spikeTrainMatch'] = np.array(summaryData[KSOutputFolder]['spikeTrainMatch'])

    plt.figure(1)
    bins = np.arange(np.min(allTemplateSimilarities), np.max(allTemplateSimilarities) + 5.0, 5.0)
    hist, _ = np.histogram(allTemplateSimilarities, bins)
    plt.bar(bins[:-1], hist, width=5.0)
    plt.show()

    # sort clusters from each KS output by template/spike train similarity
    bestClusters = {}
    nrBestClusters = 5
    for KSOutputFolder in summaryData:
        templateSimilaritySort = np.argsort(summaryData[KSOutputFolder]['templateMatch'])
        spikeTrainDistanceSort = np.argsort(summaryData[KSOutputFolder]['spikeTrainMatch'])
        nrClusters = len(summaryData[KSOutputFolder]['clusters'])
        bestTemplateSimilarityIndices = templateSimilaritySort[:nrBestClusters]
        bestTemplateSimilarity = summaryData[KSOutputFolder]['clusters'][bestTemplateSimilarityIndices]
        bestSpikeTrainDistanceIndices = spikeTrainDistanceSort[:nrBestClusters]
        # bestSpikeTrainDistanceIndices = spikeTrainDistanceSort[nrClusters - nrBestClusters:]
        bestSpikeTrainDistance = summaryData[KSOutputFolder]['clusters'][bestSpikeTrainDistanceIndices]
        bestClusters[KSOutputFolder] = bestTemplateSimilarity, bestSpikeTrainDistance

    allClusters = {}
    for KSOutputFolder in summaryData:
        templateSimilaritySort = np.argsort(summaryData[KSOutputFolder]['templateMatch'])
        spikeTrainDistanceSort = np.argsort(summaryData[KSOutputFolder]['spikeTrainMatch'])
        allTemplateSimilarity = summaryData[KSOutputFolder]['clusters'][templateSimilaritySort]
        allSpikeTrainDistance = summaryData[KSOutputFolder]['clusters'][spikeTrainDistanceSort]
        allClusters[KSOutputFolder] = allTemplateSimilarity, allSpikeTrainDistance
        paramSummary = KSOutputFolder.strip('\\') + '_summary.csv'
        paramSetOutName = os.path.join(baseFolder, paramSummary)
        with open(paramSetOutName, 'w') as paramSetOutFile:
            header = 'Cluster ID\ttemplate similarity\tspike train distance\n'
            paramSetOutFile.write(header)
            for i in range(len(summaryData[KSOutputFolder]['clusters'])):
                line = str(summaryData[KSOutputFolder]['clusters'][i])
                line += '\t'
                line += str(summaryData[KSOutputFolder]['templateMatch'][i])
                line += '\t'
                line += str(summaryData[KSOutputFolder]['spikeTrainMatch'][i])
                line += '\n'
                paramSetOutFile.write(line)

    # write summary file with best N clusters based on template similarity/spike train distance
    outputName = os.path.join(baseFolder, 'best_clusters_output.csv')
    with open(outputName, 'w') as outFile:
        header = 'Parameter set\t5 best template similarity\t5 best spiketrain distance\n'
        outFile.write(header)
        for paramSetName in bestClusters:
            templateSimilarity, spikeTrainDistance = bestClusters[paramSetName]
            line = paramSetName.split('\\')[-1]
            line += '\t'
            for clusterID in templateSimilarity:
                line += str(clusterID)
                line += ','
            line += '\t'
            for clusterID in spikeTrainDistance:
                line += str(clusterID)
                line += ','
            line += '\n'
            outFile.write(line)

    # write complete file with all clusters template similarity/spike train distances
    outputName2 = os.path.join(baseFolder, 'complete_clusters_output.csv')
    with open(outputName2, 'w') as outFile2:
        header = 'Parameter set\tsorted template similarity\tsorted spiketrain distance\n'
        outFile2.write(header)
        for paramSetName in allClusters:
            templateSimilarity, spikeTrainDistance = allClusters[paramSetName]
            line = paramSetName.split('\\')[-1]
            line += '\t'
            for clusterID in templateSimilarity:
                line += str(clusterID)
                line += ','
            line += '\t'
            for clusterID in spikeTrainDistance:
                line += str(clusterID)
                line += ','
            line += '\n'
            outFile2.write(line)

def parse_KS_parameter_summary_files(summaryFolder, baseFolder, refPath):
    '''
    load csv files; create summary statistics; check summary statistics;
    compute spike differences; create summary plots
    :param summaryFolder:
    :param baseFolder:
    :return:
    '''
    # load all csv summary files
    files = glob.glob(os.path.join(summaryFolder, '*_summary.csv'))
    summaryData = load_KS_cluster_properties_summary(files)

    # look at summary statistics across all parameter sets
    templateSimilarities = []
    minTemplateSimilarities = []
    spikeTrainDistances = []
    minSpikeTrainDistances = []
    for param in summaryData:
        paramSimilarities = []
        paramDistances = []
        for clusterID in summaryData[param]:
            similarity = summaryData[param][clusterID]['templateMatch']
            templateSimilarities.append(similarity)
            paramSimilarities.append(similarity)
            distance = summaryData[param][clusterID]['spikeTrainMatch']
            spikeTrainDistances.append(distance)
            paramDistances.append(distance)
        minTemplateSimilarities.append(np.min(paramSimilarities))
        minSpikeTrainDistances.append(np.min(paramDistances))

    plt.figure(1)
    plt.subplot(2,2,1)
    similarityBins = np.arange(0.0, np.max(templateSimilarities) + 5.0, 5.0)
    plt.hist(templateSimilarities, similarityBins)
    plt.xlabel('template similarities')
    plt.subplot(2,2,2)
    plt.hist(spikeTrainDistances)
    plt.xlabel('spike train dist.')
    plt.subplot(2,2,3)
    plt.hist(minTemplateSimilarities)
    plt.xlabel('min. template similarities')
    plt.subplot(2,2,4)
    plt.hist(minSpikeTrainDistances)
    plt.xlabel('min. spike train dist.')
    plt.savefig(os.path.join(summaryFolder, 'summary_histograms.pdf'))
    plt.show()

    folders_ = glob.glob(os.path.join(baseFolder, '*'))
    folders = []
    for folder in folders_:
        if not os.path.isdir(folder):
            continue
        else:
            folders.append(folder)

    templateScaleIndex = baseFolder.find('scale_')
    templateScale = float(baseFolder[templateScaleIndex + 6:templateScaleIndex + 10])
    # 979 parameters
    # for template similarity using only max channel:
    # templateThreshold979 = 200.0*templateScale
    # for template similarity using all channels, scale 1.0-0.6:
    templateThreshold979 = 40.0
    # for template similarity using all channels, scale 0.4:
    # templateThreshold979 = 25.0
    # for template similarity using all channels, scale 0.2-0.1:
    # templateThreshold979 = 20.0
    print 'Template threshold = %.2f' % templateThreshold979
    # refClusterID = 979
    refClusterID = 27 # 6 13 13 / 10 30 30

    # # 845 parameters
    # templateThreshold845 = 35.0
    # # refClusterID = 845
    # refClusterID = 15 # 6 13 13 / 10 30 30

    refCluster = load_reference_cluster(refPath, refClusterID)
    maxSpikeTime = refCluster.spiketrains[0].t_stop
    figureCount = 0
    for maxPrecision in range(2, 5):
        print 'Evaluating spike time recovery at a precision of %d samples' % maxPrecision
        mergedClusters = {}
        mergedClusterIDs ={}
        clusterDifferences = {}
        timingDifferences = {}
        for param in summaryData:
        # for param in ['Th_10_11_11_Lam_10_30_30']:
            KSOutputFolder = ''
            for folder in folders:
                if param in folder:
                    KSOutputFolder = folder
                    break
            clusterGroup = clust.ClusterGroup(clust.reader.read_KS_clusters_unsorted(KSOutputFolder, ClusteringSrcFolder, 'dev', 20000.0))
            currentMergeClusterIDs = []
            for clusterID in summaryData[param]:
                templateSimilarity = summaryData[param][clusterID]['templateMatch']
                if templateSimilarity < templateThreshold979:
                # if templateSimilarity < templateThreshold845:
                    currentMergeClusterIDs.append(clusterID)
            mergedClusterIDs[param] = currentMergeClusterIDs
            if len(currentMergeClusterIDs):
                mergedCluster = clusterGroup.merge_clusters_with_duplicates(currentMergeClusterIDs, 6.0e-4)
                # HACK for 845: instead of cutting all automtically found clusters, restrict comparison to duration of 845
                # oldSpikeTimes = mergedCluster.spiketrains[0]
                # selection = np.where(oldSpikeTimes <= maxSpikeTime)
                # if len(selection[0]):
                #     newSpikeTimes = oldSpikeTimes[np.where(oldSpikeTimes <= maxSpikeTime)]
                #     newSpikeTrain = neo.core.SpikeTrain(newSpikeTimes, t_start=np.min(newSpikeTimes), t_stop=np.max(newSpikeTimes),
                #                                         units=oldSpikeTimes.units)
                #     mergedCluster.spiketrains[0] = newSpikeTrain
                # END HACK for 845
                mergedClusters[param] = mergedCluster
                dt = []
                FP, FN, totalSpikes  = compute_cluster_differences(mergedCluster, refCluster, maxPrecision, dt)
                clusterDifferences[param] = FP, FN, totalSpikes
                timingDifferences[param] = dt
                print '\tFP = %d -- FN = %d -- Correct = %d' % (FP, FN, totalSpikes - FN)
                # plt.figure(figureCount)
                # bins = np.arange(-2.25e-4, 2.25e-4, 5e-5)
                # plt.hist(dt, bins)
                # n, bins, patches = plt.hist(dt)
                figureCount += 1
            else:
                clusterDifferences[param] = -1, -1, -1
        # plt.show()

        suffix = 'cluster_reference_differences_maxPrecision_' + str(maxPrecision - 1) + '.csv'
        outName = os.path.join(summaryFolder, suffix)
        with open(outName, 'w') as outFile:
            header = 'Th1\tTh2/3\tLambda1\tLambda2/3\tNr. merged clusters\tFP\tFN\tRecovered spikes\tTotal ref. spikes\n'
            outFile.write(header)
            params = mergedClusterIDs.keys()
            params.sort()
            for param in params:
                # line format: Th_[Th1]_[Th2]_[Th3]_Lam_[Lambda1]_[Lambda2]_[Lambda3]
                splitParam = param.split('_')
                line = splitParam[1]
                line += '\t'
                line += splitParam[2]
                line += '\t'
                line += splitParam[5]
                line += '\t'
                line += splitParam[6]
                line += '\t'
                nrClusters = len(mergedClusterIDs[param])
                line += str(nrClusters)
                line += '\t'
                if nrClusters:
                    differences = clusterDifferences[param]
                    line += str(differences[0])
                    line += '\t'
                    line += str(differences[1])
                    line += '\t'
                    line += str(differences[2] - differences[1])
                    line += '\t'
                    line += str(differences[2])
                    line += '\n'
                else:
                    line += 'N/A\tN/A\tN/A\tN/A\n'
                outFile.write(line)

def evaluate_KS_parameters_IPSC(experimentInfoName, baseFolder, summaryFolder, scaleSimilarityThreshold):
    '''
    evaluate IPSCs of different KS parameter clusters
    to allow comparison with ground-truth IPSC (avg., distribution, ...)
    :return:
    '''
    # load/filter WC recording
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    # constants for calculating spike-triggered stuff
    alignedWindow = np.array((-5.0, 10.0))*pq.ms
    maxBegin = 0.3*pq.ms
    maxWindow = np.array((1.2, 2.54))*pq.ms
    alignedWindowSamples_ = (alignedWindow*experimentInfo['WC']['SamplingRate']*pq.Hz).simplified
    maxBeginSamples_ = (maxBegin*experimentInfo['WC']['SamplingRate']*pq.Hz).simplified - alignedWindowSamples_[0]
    maxWindowSamples_ = (maxWindow*experimentInfo['WC']['SamplingRate']*pq.Hz).simplified - alignedWindowSamples_[0]
    maxBeginSamples = int(maxBeginSamples_ + 0.5)
    maxWindowSamples = (int(maxWindowSamples_[0] + 0.5), int(maxWindowSamples_[1] + 0.5))

    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    WCWindows = experimentInfo['WC']['RecordingPeriods']
    WCSignals = ca.reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])
    WCFilteredSignals = []
    for signal in WCSignals:
        WCFilteredSignals.append(STC.filter_current_trace(signal['current'].flatten(), experimentInfo))

    # compute alignment of spike times to WC recording
    alignments = STC.align_current_traces_probe_recordings(experimentInfo)

    # load all csv summary files
    files = glob.glob(os.path.join(summaryFolder, '*_summary.csv'))
    summaryData = load_KS_cluster_properties_summary(files)

    folders_ = glob.glob(os.path.join(baseFolder, '*'))
    folders = []
    for folder in folders_:
        if not os.path.isdir(folder) or not 'Th' in folder or not 'Lam' in folder:
            continue
        else:
            folders.append(folder)

    for param in summaryData:
    # for param in ['Th_6_13_13_Lam_10_30_30']:
        KSOutputFolder = ''
        for folder in folders:
            if param in folder:
                KSOutputFolder = folder
                break
        print 'Computing spike time-aligned traces for KS output %s' % KSOutputFolder
        # load clusters for parameter set
        clusterGroup = clust.ClusterGroup(clust.reader.read_KS_clusters_unsorted(KSOutputFolder, ClusteringSrcFolder, 'dev', 20000.0))
        currentMergeClusterIDs = []
        # select clusters below similarity threshold
        for clusterID in summaryData[param]:
            templateSimilarity = summaryData[param][clusterID]['templateMatch']
            if templateSimilarity < scaleSimilarityThreshold:
                currentMergeClusterIDs.append(clusterID)
        if len(currentMergeClusterIDs):
            mergedCluster = clusterGroup.merge_clusters_with_duplicates(currentMergeClusterIDs, 6.0e-4)
            # compute ST currents and ST IPSC amplitudes
            STA, STAlignedSnippets, nrSpikes = STC.compute_ST_traces_average(mergedCluster, WCFilteredSignals, experimentInfo['WC']['RecordingFilenames'],
                                                                   WCWindows, alignments, alignedWindow)
            # compute and store amplitudes (distribution?) for each parameter set
            if nrSpikes:
                STA_SE = np.std(STAlignedSnippets.snippets, axis=0)/np.sqrt(nrSpikes)
                beginValues = STAlignedSnippets.snippets[:, maxBeginSamples]
                maxValueRanges = STAlignedSnippets.snippets[:, maxWindowSamples[0]:maxWindowSamples[1]]
                maxValues = np.max(maxValueRanges, axis=1)
                amplitudes = maxValues - beginValues
            else:
                STA_SE = np.zeros(STA.shape)
                amplitudes = np.array([])
            suffix = param + '_STAmplitudes'
            amplitudesOutName = os.path.join(summaryFolder, suffix)
            np.save(amplitudesOutName, amplitudes)
            suffix = param + '_STA'
            STAOutName = os.path.join(summaryFolder, suffix)
            np.save(STAOutName, STA)
            suffix = param + '_STA_SE'
            STASEOutName = os.path.join(summaryFolder, suffix)
            np.save(STASEOutName, STA_SE)

def compute_template_similarity(cluster, refTemplate, refMaxChannel):
    '''
    Computes a measure of waveform similarity
    (RMSE, i.e., lower = more similar)
    :param cluster: cluster object
    :param refTemplate: waveform of reference cluster
    :return: similarity
    '''
    # comparisonWaveForm = cluster.template[refMaxChannel]
    # diff = comparisonWaveForm - refTemplate
    comparisonWaveForm = cluster.template
    diff = comparisonWaveForm.flatten() - refTemplate.flatten()
    RMSE = np.sqrt(np.dot(diff, diff)/len(diff))
    # if 20.0 < RMSE < 75.0:
    # if RMSE < 75.0:
    #     plt.figure(cluster.clusterID)
    #     plt.plot(refTemplate.flatten())
    #     plt.plot(comparisonWaveForm.flatten())
    #     titleStr = 'Cluster %d - RMSE = %.1f' % (cluster.clusterID, RMSE)
    #     plt.title(titleStr)
    #     plt.show()
    return RMSE

def compute_spiketrain_distance(cluster, refCluster):
    '''
    Computes spike train distance
    :param cluster: cluster object
    :param refCluster: cluster object of reference cluster
    :return: spike train distance metric
    '''
    costFactor = max(cluster.firingRate, refCluster.firingRate)
    distanceMatrix = eph.spike_train_dissimilarity.victor_purpura_dist((cluster.spiketrains[0], refCluster.spiketrains[0]), costFactor)
    return distanceMatrix[0][1]

def compute_cluster_differences(cluster, refCluster, maxPrecision, timingDifferences):
    '''

    :param maxPrecision: in sampling intervals; has to be at least 1
    :param cluster:
    :param refCluster:
    :return:
    '''
    samplingInterval = 1/20000.0
    samplingFrequency = 20000.0
    maxPrecision = max(1, maxPrecision)
    # greedy algorithm: iterate through all _reference_ spikes,
    # and look if spike in cluster present within +- spikeTimePrecision
    # if yes, remove from cluster spike train
    # if no, increment false negatives
    # at end, remaining spikes in cluster are defined as false positives
    clusterSpikes = list(cluster.spiketrains[0].magnitude)
    refClusterSpikes = list(refCluster.spiketrains[0].magnitude)
    FN = 0
    totalSpikes = len(refClusterSpikes)
    print 'Comparing %d spikes in cluster with %d spikes of reference cluster' % (len(clusterSpikes), totalSpikes)
    for tRef in refClusterSpikes:
        foundCorrespondingSpike = 0
        for n, t in enumerate(clusterSpikes):
            dSample = int(round(samplingFrequency*(t - tRef)))
            for i in range(maxPrecision):
                # if abs(t - tRef) <= i*samplingInterval:
                if abs(dSample) <= i:
                    # timingDifferences.append(t - tRef)
                    timingDifferences.append(dSample)
                    foundCorrespondingSpike = 1
                    # clusterSpikes.remove(t)
                    clusterSpikes[n] = -1.0e6
                if foundCorrespondingSpike:
                    break
            if foundCorrespondingSpike:
                break
        if not foundCorrespondingSpike:
            FN += 1
    clusterSpikes = np.array(clusterSpikes)
    remainingClusterSpikes = clusterSpikes[np.where(clusterSpikes > -1.0e6)]
    FP = len(remainingClusterSpikes)

    return FP, FN, totalSpikes

def load_reference_cluster(path, clusterID):
    '''
    Loads reference cluster
    :param path: path to KS output of reference cluster
    :param clusterID: ID of cluster
    :return: cluster object
    '''
    samplingRate = 20000.0
    # clusters = clust.reader.read_KS_clusters(path, ClusteringSrcFolder, 'dev', ('good',), samplingRate)
    clusters = clust.reader.read_KS_clusters_unsorted(path, ClusteringSrcFolder, 'dev', samplingRate)
    clusterGroup = clust.ClusterGroup(clusters)
    clusterGroup.remove_short_ISIs(clusterID, 6.0e-4)
    return clusterGroup.clusters[clusterID]

def load_reference_cluster_template(path, clusterName):
    '''
    Loads reference cluster template waveform
    :param path: path to KS output of reference cluster
    :param cluster: cluster object of reference cluster
    :return: numpay array of reference waveform
    '''
    return np.load(os.path.join(path, clusterName))

def load_KS_cluster_properties_summary(files):
    '''
    Loads summary csv files written by parse_KS_output_set
    :param files:
    :return:
    '''
    summaryData = {}
    for file in files:
        with open(file, 'r') as summaryFile:
            parameterName = os.path.split(file)[-1]
            parameterName = parameterName[:-12] # drop _summary.csv extension
            summaryData[parameterName] = {}
            for line in summaryFile:
                splitLine = line.split('\t')
                try:
                    clusterID = int(splitLine[0])
                    summaryData[parameterName][clusterID] = {}
                    templateSimilarity = float(splitLine[1])
                    spikeTrainDistance = float(splitLine[2])
                    summaryData[parameterName][clusterID]['templateMatch'] = templateSimilarity
                    summaryData[parameterName][clusterID]['spikeTrainMatch'] = spikeTrainDistance
                except:
                    continue

    return summaryData

if __name__ == '__main__':
    if len(sys.argv) == 5:
        refPath = sys.argv[1]
        refClusterID = int(sys.argv[2])
        refClusterName = sys.argv[3]
        baseFolder = sys.argv[4]
        parse_KS_output_set(refPath, refClusterID, refClusterName, baseFolder)
        # experimentInfoName = sys.argv[1]
        # baseFolder = sys.argv[2]
        # summaryFolder = sys.argv[3]
        # threshold = float(sys.argv[4])
        # evaluate_KS_parameters_IPSC(experimentInfoName, baseFolder, summaryFolder, threshold)
    elif len(sys.argv) == 4:
        summaryFolder = sys.argv[1]
        baseFolder = sys.argv[2]
        refPath = sys.argv[3]
        parse_KS_parameter_summary_files(summaryFolder, baseFolder, refPath)
    else:
        print 'Three arguments required: [refPath] [refClusterID] [refClusterName] [baseFolder]'
