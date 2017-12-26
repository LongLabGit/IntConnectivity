import numpy as np
import scipy.io
import os
import neo
from cluster import Cluster

def read_KS_clusters(dataFolder, ClusteringSrcFolder, version, keep_group, samplingRate):
    ''' folder: location of your data
        version: 'dev' or 'release'. This will tell the program where to look
        for the data
        keep: list of phy cluster types to be loaded. Options:
        'good', 'mua', 'noise', 'unsorted' ('unsorted' NOT in dev version)
        fs: sampling rate (in Hz)
        :return: dict of Cluster objects, where keys are phy cluster IDs
        '''
    if version == 'release':
        KiloSortData = scipy.io.loadmat(os.path.join(dataFolder, 'batches', 'KS_output.mat'), struct_as_record=False, squeeze_me=True)
        spike_clusters = np.load(os.path.join(dataFolder, 'batches', 'spike_clusters.npy')).flatten()
        amplitudes = np.load(os.path.join(dataFolder, 'batches', 'amplitudes.npy')).flatten()
        cluster_group_fname = os.path.join(dataFolder, 'batches', 'cluster_groups.csv')
    elif version == 'dev':
        KiloSortData = scipy.io.loadmat(os.path.join(dataFolder, 'KS_output.mat'), struct_as_record=False, squeeze_me=True)
        spike_clusters = np.load(os.path.join(dataFolder, 'spike_clusters.npy')).flatten()
        amplitudes = np.load(os.path.join(dataFolder, 'amplitudes.npy')).flatten()
        cluster_group_fname = os.path.join(dataFolder, 'cluster_group.tsv')
    else:
        errstr = 'Cluster loading for version %s not implemented' % version
        raise NotImplementedError(errstr)

    channelMapMatlab = scipy.io.loadmat(os.path.join(ClusteringSrcFolder,KiloSortData['ops'].chanMap), struct_as_record=False, squeeze_me=True)
    xcoords = channelMapMatlab['xcoords'][np.where(channelMapMatlab['connected'] == 1)]
    ycoords = channelMapMatlab['ycoords'][np.where(channelMapMatlab['connected'] == 1)]
    kcoords = channelMapMatlab['kcoords'][np.where(channelMapMatlab['connected'] == 1)]
    channelMap = channelMapMatlab['chanMap'][np.where(channelMapMatlab['connected'] == 1)]

    clusterIDs = []
    clusterGroup = []
    with open(cluster_group_fname, 'r') as f:
        lineCnt = 0
        for line in f:
            lineCnt += 1
            if lineCnt == 1:
                continue
            if version == 'release':
                delim = ' '
            elif version == 'dev':
                delim = '\t'
            splitLine = line.strip().split(delim)
            clusterIDs.append(int(splitLine[0]))
            clusterGroup.append(splitLine[1])
    clusterIDs = np.array(clusterIDs)
    clusterGroup = np.array(clusterGroup)

    selection = clusterGroup == keep_group[0]
    for i in range(1, len(keep_group)):
        selection *= clusterGroup == keep_group[i]
    keptClusters = clusterIDs[np.where(selection)]
    keptClusterGroups = clusterGroup[np.where(selection)]
    clusters = {}
    for i in range(len(keptClusters)):
        clusterID = keptClusters[i]
        spikeTimeIndices = np.where(spike_clusters == clusterID)
        # st3: 4 columns: sample time, original cluster, amplitude, useless
        spikeSamples = np.array(KiloSortData['rez'].st3)
        # remove erroneous identically duplicate spike times
        clusterSpikeSamples, uniqueIndices = np.unique(spikeSamples[spikeTimeIndices, 0], return_index=True)
        spikeTimes = clusterSpikeSamples/samplingRate
        spikeTrain = neo.core.SpikeTrain(spikeTimes, units='sec', t_stop=max(spikeTimes), t_start=min(0, min(spikeTimes)))
        templateAmplitudes = spikeSamples[spikeTimeIndices[0][uniqueIndices], 2]
        # find channel location of max. waveform
        # look up original clusters comprising this unit
        originalClusters = np.array(np.unique(spikeSamples[spikeTimeIndices, 1]), dtype='int') - 1  # Matlab vs. python indexing
        # compute mean template of these clusters
        meanTemplate = np.mean(np.array(KiloSortData['rez'].Wraw)[:, :, originalClusters], axis=2)
        # average power across time and find channel index with max power
        KS_channel = np.argmax(np.mean(np.abs(meanTemplate), axis=1))
        maxChannel = channelMap[KS_channel] - 1 # Matlab vs. python indexing
        maxWF = meanTemplate[KS_channel, :]
        shank = kcoords[KS_channel]
        coordinates = xcoords[KS_channel], ycoords[KS_channel]
        firingRate = len(spikeTrain.times)/(spikeTrain.t_stop - spikeTrain.t_start)
        thisCluster = Cluster(clusterID, keptClusterGroups[i], spikeTrain, maxWF, shank, maxChannel, coordinates, firingRate)
        thisCluster.template = meanTemplate
        thisCluster.templateAmplitudes = templateAmplitudes
        clusters[clusterID] = thisCluster

    return clusters

def read_KS_clusters_unsorted(dataFolder, ClusteringSrcFolder, version, samplingRate):
    ''' folder: location of your data
        version: 'dev' or 'release'. This will tell the program where to look
        for the data
        fs: sampling rate (in Hz)
        :return: dict of Cluster objects, where keys are phy cluster IDs
        '''
    if version == 'release':
        KiloSortData = scipy.io.loadmat(os.path.join(dataFolder, 'batches', 'KS_output.mat'), struct_as_record=False, squeeze_me=True)
        spike_clusters = np.load(os.path.join(dataFolder, 'batches', 'spike_clusters.npy')).flatten()
        amplitudes = np.load(os.path.join(dataFolder, 'batches', 'amplitudes.npy')).flatten()
    elif version == 'dev':
        KiloSortData = scipy.io.loadmat(os.path.join(dataFolder, 'KS_output.mat'), struct_as_record=False, squeeze_me=True)
        spike_clusters = np.load(os.path.join(dataFolder, 'spike_clusters.npy')).flatten()
        amplitudes = np.load(os.path.join(dataFolder, 'amplitudes.npy')).flatten()
    else:
        errstr = 'Cluster loading for version %s not implemented' % version
        raise NotImplementedError(errstr)

    channelMapMatlab = scipy.io.loadmat(os.path.join(ClusteringSrcFolder,KiloSortData['ops'].chanMap), struct_as_record=False, squeeze_me=True)
    xcoords = channelMapMatlab['xcoords'][np.where(channelMapMatlab['connected'] == 1)]
    ycoords = channelMapMatlab['ycoords'][np.where(channelMapMatlab['connected'] == 1)]
    kcoords = channelMapMatlab['kcoords'][np.where(channelMapMatlab['connected'] == 1)]
    channelMap = channelMapMatlab['chanMap'][np.where(channelMapMatlab['connected'] == 1)]

    clusterIDs = np.unique(spike_clusters)
    clusters = {}
    for clusterID in clusterIDs:
        spikeTimeIndices = np.where(spike_clusters == clusterID)
        # st3: 4 columns: sample time, original cluster, amplitude, useless
        spikeSamples = np.array(KiloSortData['rez'].st3)
        # remove erroneous identically duplicate spike times
        clusterSpikeSamples, uniqueIndices = np.unique(spikeSamples[spikeTimeIndices, 0], return_index=True)
        spikeTimes = clusterSpikeSamples/samplingRate
        spikeTrain = neo.core.SpikeTrain(spikeTimes, units='sec', t_stop=max(spikeTimes), t_start=min(0, min(spikeTimes)))
        templateAmplitudes = amplitudes[spikeTimeIndices[0][uniqueIndices]]
        # find channel location of max. waveform
        # look up original clusters comprising this unit
        originalClusters = np.array(np.unique(spikeSamples[spikeTimeIndices, 1]), dtype='int') - 1  # Matlab vs. python indexing
        # compute mean template of these clusters
        meanTemplate = np.mean(np.array(KiloSortData['rez'].Wraw)[:, :, originalClusters], axis=2)
        # average power across time and find channel index with max power
        KS_channel = np.argmax(np.mean(np.abs(meanTemplate), axis=1))
        maxChannel = channelMap[KS_channel] - 1 # Matlab vs. python indexing
        maxWF = meanTemplate[KS_channel, :]
        shank = kcoords[KS_channel]
        coordinates = xcoords[KS_channel], ycoords[KS_channel]
        firingRate = len(spikeTrain.times)/(spikeTrain.t_stop - spikeTrain.t_start)
        thisCluster = Cluster(clusterID, 'unsorted', spikeTrain, maxWF, shank, maxChannel, coordinates, firingRate)
        thisCluster.template = meanTemplate
        thisCluster.templateAmplitudes = templateAmplitudes
        clusters[clusterID] = thisCluster

    return clusters