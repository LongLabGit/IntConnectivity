from neo.core import Unit, SpikeTrain
import os
import numpy as np

class Cluster(Unit):
    '''lightweight object representing a single extracellular unit'''

    def __init__(self, clusterID, group, spikeTimes, waveForm=None, shank=None, maxChannel=None, coordinates=None, firingRate=None):
        super(Unit, self).__init__(name=str(clusterID))
        self.clusterID = clusterID
        self.group = group
        if hasattr(spikeTimes, 'times'):
            self.spiketrains.append(spikeTimes)
        else:
            newSpiketrain = SpikeTrain(spikeTimes, units='sec', t_stop=max(spikeTimes))
            self.spiketrains.append(newSpiketrain)
        self.waveForm = waveForm
        self.shank = shank
        self.maxChannel = maxChannel
        self.coordinates = coordinates
        self.firingRate = firingRate
        self.mergedDuplicates = False
        self.nrMergedDuplicateSpikes = 0


class ClusterGroup(object):
    '''
    Container for clusters supporting Long lab-specific
    postprocessing/manipulations methods
    '''

    def __init__(self, clusters):
        self.clusters = clusters

    def align_spike_times(self):
        '''
        :return:
        '''
        # TODO: Implement loading of waveforms
        # and alignment to (interpolated) minimum of waveform
        # on electrode with largest amplitude
        pass

    def remove_short_ISIs(self, clusterID, ISIThreshold):
        '''
        Removes second spike in cases where ISI is less than a given threshold.
        WARNING: Modifies cluster in-place.
        :param clusterID: ID of cluster to be modified
        :param ISIThreshold: spikes with ISIs less than threshold will be removed
        :return: N/A
        '''
        spikeTimes = self.clusters[clusterID].spiketrains[0].times.magnitude
        dt = np.zeros(len(spikeTimes))
        dt[:-1] = np.diff(spikeTimes)
        dt[-1] = 1e6
        duplicateSpikeDelays = spikeTimes[np.where(np.abs(dt) <= ISIThreshold)]
        keepSpikeTimes = spikeTimes[np.where(np.abs(dt) > ISIThreshold)]
        keepAmplitudes = self.clusters[clusterID].templateAmplitudes[np.where(np.abs(dt) > ISIThreshold)]
        spikeTrain = SpikeTrain(keepSpikeTimes, units='sec', t_stop=max(spikeTimes), t_start=min(0, min(spikeTimes)))
        self.clusters[clusterID].spiketrains[0] = spikeTrain
        self.clusters[clusterID].templateAmplitudes = keepAmplitudes
        self.clusters[clusterID].mergedDuplicates = True
        self.clusters[clusterID].nrMergedDuplicateSpikes = len(duplicateSpikeDelays)

    def merge_clusters_with_duplicates_post_hoc(self, dataFolder, ISIThreshold):
        '''
        For post-hoc merging of clusters with near-duplicate spike times.
        Modifies clusters in-place.
        :param dataFolder: path to folder containing misalignment file
        :param ISIThreshold: spikes with ISIs less than threshold will be removed
        :return: N/A
        '''
        # TODO: fix possible bug where doublets in main cluster are not removed
        # TODO: correct amplitudes, i.e., sort after merging and remove amplitudes belonging to removed spikes
        # load misalignment file if exists
        misalignmentFilename = os.path.join(dataFolder, 'misaligned_clusters.txt')
        if os.path.exists(misalignmentFilename):
            misalignedClusters = {}
            with open(misalignmentFilename, 'r') as misalignmentFile:
                for line in misalignmentFile:
                    splitLine = line.strip().split('\t')
                    try:
                        mainCluster = int(splitLine[0])
                        mergeClusterStr = splitLine[1].split(',')
                        mergeClusters = [int(cluster) for cluster in mergeClusterStr]
                        misalignedClusters[mainCluster] = mergeClusters
                    except ValueError:
                        continue
            # for each main cluster:
            for mainCluster in misalignedClusters:
                mainTimes = self.clusters[mainCluster].spiketrains[0].times
                dt = np.zeros(len(mainTimes))
                dt[:-1] = np.diff(mainTimes)
                dt[-1] = 1e6
                keepMainTimes = mainTimes[np.where(np.abs(dt) > ISIThreshold)]
                self.clusters[mainCluster].spiketrains[0] = keepMainTimes
                for mergeCluster in misalignedClusters[mainCluster]:
                    duplicateSpikes = []
                    duplicateSpikeDelays = []
                    mergeSpikeTimes = []
                    # iteratively look at ISIs between main cluster and merge clusters
                    for t in self.clusters[mergeCluster].spiketrains[0].times:
                        dt = t - mainTimes
                        # all ISIs where dt is few samples (up to +-3 samples):
                        # do not add the spikes from the merge cluster
                        duplicates = np.where(np.abs(dt) <= ISIThreshold)[0]
                        if len(duplicates):
                            duplicateSpikeDelays.append(dt[duplicates[:]])
                            duplicateSpikes.append(duplicates[:])
                        # else: add the spikes from the merge cluster
                        else:
                            mergeSpikeTimes.append(t.magnitude)
                            # clusters[mainCluster].spiketrains[0].times.append(t)
                    # determine systematic sample shift between main and merge cluster
                    # and shift merge spike times by that amount
                    meanDelay = np.mean(duplicateSpikeDelays)
                    mergeSpikeTimes = np.array(mergeSpikeTimes) - meanDelay
                    newSpikeTimes = self.clusters[mainCluster].spiketrains[0].magnitude
                    newSpikeTimes = np.append(newSpikeTimes, mergeSpikeTimes)
                    newSpikeTimes.sort()
                    newSpikeTrain = SpikeTrain(newSpikeTimes, units=self.clusters[mainCluster].spiketrains[0].units,
                                                   t_stop=np.max(newSpikeTimes), t_start=np.min(newSpikeTimes))
                    self.clusters[mainCluster].spiketrains[0] = newSpikeTrain
                    self.clusters[mainCluster].mergedDuplicates = True
                    self.clusters[mainCluster].nrMergedDuplicateSpikes = len(duplicateSpikeDelays)
                    self.clusters.pop(mergeCluster)
                    # var = 0 / 0

    def merge_clusters_with_duplicates(self, mergeClusters_, ISIThreshold):
        '''
        For merging of clusters. Near-duplicate spike times (< ISI threshold) are removed.
        Creates new cluster.
        :param mergeClusters: iterable of cluster IDs to be merged
        :param ISIThreshold: spikes with ISIs less than threshold will be removed
        :return: cluster object
        '''
        mainCluster = mergeClusters_[0]
        mergeClusters = mergeClusters_[1:]
        allSpikeTimes = list(self.clusters[mainCluster].spiketrains[0].magnitude)
        for mergeCluster in mergeClusters:
            mergeSpikeTimes = list(self.clusters[mergeCluster].spiketrains[0].magnitude)
            allSpikeTimes.extend(mergeSpikeTimes)
        newSpikeTimes = np.array(allSpikeTimes)
        newSpikeTimes.sort()
        newSpikeTrain = SpikeTrain(newSpikeTimes, units=self.clusters[mainCluster].spiketrains[0].units,
                                       t_stop=np.max(newSpikeTimes), t_start=np.min(newSpikeTimes))
        newCluster = Cluster('tmp', 'none', newSpikeTrain)
        self.clusters['tmp'] = newCluster
        self.remove_short_ISIs('tmp', ISIThreshold)
        return self.clusters.pop('tmp')