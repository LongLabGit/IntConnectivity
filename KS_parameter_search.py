###################################
# (semi-) automated detection of
# "reference" unit in multiple runs
# of KiloSort.
# Used for analysis of systematic
# KiloSort parameter explorations
###################################

import numpy as np
import matplotlib.pyplot as plt
import os, os.path, sys
import glob
import elephant as eph
import neo
import ClusterProcessing as clust

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
    maxPrecision = 3
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
            spikeTrainDistance = compute_spiketrain_distance(cluster, refCluster)
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
    plt.hist(allTemplateSimilarities)
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
    summaryData = {}
    files = glob.glob(os.path.join(summaryFolder, '*_summary.csv'))
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

    # plt.figure(1)
    # plt.subplot(2,2,1)
    # similarityBins = np.arange(0.0, np.max(templateSimilarities) + 15.0, 15.0)
    # plt.hist(templateSimilarities, similarityBins)
    # plt.xlabel('template similarities')
    # plt.subplot(2,2,2)
    # plt.hist(spikeTrainDistances)
    # plt.xlabel('spike train dist.')
    # plt.subplot(2,2,3)
    # plt.hist(minTemplateSimilarities)
    # plt.xlabel('min. template similarities')
    # plt.subplot(2,2,4)
    # plt.hist(minSpikeTrainDistances)
    # plt.xlabel('min. spike train dist.')
    # # plt.savefig(os.path.join(summaryFolder, 'summary_histograms.pdf'))
    # # plt.show()

    folders_ = glob.glob(os.path.join(baseFolder, '*'))
    folders = []
    for folder in folders_:
        if not os.path.isdir(folder):
            continue
        else:
            folders.append(folder)

    # 979 parameters
    templateThreshold979 = 200.0
    # refClusterID = 979
    refClusterID = 27 # 6 13 13 / 10 30 30

    # # 845 parameters
    # templateThreshold845 = 35.0
    # # refClusterID = 845
    # refClusterID = 15 # 6 13 13 / 10 30 30

    refCluster = load_reference_cluster(refPath, refClusterID)
    maxSpikeTime = refCluster.spiketrains[0].t_stop
    figureCount = 0
    for maxPrecision in range(1, 5):
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
                clusterDifferences[param] = compute_cluster_differences(mergedCluster, refCluster, maxPrecision, dt)
                timingDifferences[param] = dt
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
            header = 'Th1\tTh2/3\tLambda1\tLambda2/3\tNr. merged clusters\tFP\tFN\tTotal ref. spikes\n'
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
                    line += str(differences[2])
                    line += '\n'
                else:
                    line += 'N/A\tN/A\tN/A\n'
                outFile.write(line)

def compute_template_similarity(cluster, refTemplate, refMaxChannel):
    '''
    Computes a measure of waveform similarity
    (RMSE, i.e., lower = more similar)
    :param cluster: cluster object
    :param refTemplate: waveform of reference cluster
    :return: similarity
    '''
    comparisonWaveForm = cluster.template[refMaxChannel]
    diff = comparisonWaveForm - refTemplate
    # plt.figure()
    # plt.plot(refTemplate)
    # plt.plot(comparisonWaveForm)
    # plt.show()
    return np.sqrt(np.dot(diff, diff)/len(diff))

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
    for tRef in refClusterSpikes:
        foundCorrespondingSpike = 0
        for t in clusterSpikes:
            for i in range(maxPrecision):
                dSample = int(round(samplingFrequency*(t - tRef)))
                # if abs(t - tRef) <= i*samplingInterval:
                if abs(dSample) <= i:
                    # timingDifferences.append(t - tRef)
                    timingDifferences.append(dSample)
                    foundCorrespondingSpike = 1
                    clusterSpikes.remove(t)
                if foundCorrespondingSpike:
                    break
            if foundCorrespondingSpike:
                break
        if not foundCorrespondingSpike:
            FN += 1
    FP = len(clusterSpikes)

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

if __name__ == '__main__':
    if len(sys.argv) == 5:
        refPath = sys.argv[1]
        refClusterID = int(sys.argv[2])
        refClusterName = sys.argv[3]
        baseFolder = sys.argv[4]
        parse_KS_output_set(refPath, refClusterID, refClusterName, baseFolder)
    elif len(sys.argv) == 4:
        summaryFolder = sys.argv[1]
        baseFolder = sys.argv[2]
        refPath = sys.argv[3]
        parse_KS_parameter_summary_files(summaryFolder, baseFolder, refPath)
    else:
        print 'Three arguments required: [refPath] [refClusterID] [refClusterName] [baseFolder]'
