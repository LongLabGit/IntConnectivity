##############################################
# tool for visualizing units from SiProbe,
# raw extracellular voltage, intracellular currents.
# Converts KiloSort output to Neuroscope input.
# Aligns unit spike times and current traces,
# and also converts to Neuroscope input.
##############################################

import sys, os, ast, cPickle
import matplotlib.pyplot as plt
import numpy as np
import neo, elephant
import scipy.signal
import ConnectionAnalyzer as ca
import ClusterProcessing as clust
import quantities as pq

clusteringSrcFolder = 'E:\\User\\project_src\\physiology\\Clustering'

def write_neuroscope_extracellular_clusters(experimentInfoName):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    groups = ('good',)
    SiProbeDataFolder = experimentInfo['SiProbe']['DataBasePath']
    clusters = clust.reader.read_KS_clusters(SiProbeDataFolder, clusteringSrcFolder,
                                       'dev', groups, experimentInfo['SiProbe']['SamplingRate'])
    groupName = 'ClustersPerShank'
    for g in groups:
        groupName += '_'
        groupName += g
    nsClustersName = os.path.join(experimentInfo['SiProbe']['ClusterBasePath'], groupName)
    clust.writer.write_neuroscope_units_per_shank(nsClustersName, clusters, experimentInfo['SiProbe']['SamplingRate'])

    groups = ('good','mua')
    clustersMUA = clust.reader.read_KS_clusters(SiProbeDataFolder, clusteringSrcFolder,
                                       'dev', groups, experimentInfo['SiProbe']['SamplingRate'])
    groupName = 'ClustersPerShank'
    for g in groups:
        groupName += '_'
        groupName += g
    nsClustersName = os.path.join(experimentInfo['SiProbe']['ClusterBasePath'], groupName)
    clust.writer.write_neuroscope_units_per_shank(nsClustersName, clustersMUA, experimentInfo['SiProbe']['SamplingRate'])

def write_neuroscope_intracellular_clusters(experimentInfoName):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    WCWindows = experimentInfo['WC']['RecordingPeriods']
    WCSignals = ca.reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])
    WCFilteredSignals = []
    for signal in WCSignals:
        WCFilteredSignals.append(filter_current_trace(signal['current'].flatten(), experimentInfo))

    groups = ('good',)
    SiProbeDataFolder = experimentInfo['SiProbe']['DataBasePath']
    alignments = align_current_traces_probe_recordings(experimentInfo)
    clusters = clust.reader.read_KS_clusters(SiProbeDataFolder, clusteringSrcFolder,
                                       'dev', groups, experimentInfo['SiProbe']['SamplingRate'])
    # iterate over all WC recording files
    for i, signal in enumerate(WCFilteredSignals):
        # create aligned clusters
        newClusters = {}
        for clusterID in clusters:
            cluster = clusters[clusterID]
            newCluster = clust.Cluster(cluster.clusterID, cluster.group, cluster.spiketrains[0], cluster.shank,
                                            cluster.maxChannel, cluster.coordinates, cluster.firingRate)
            a, b = alignments[i]
            # b *= newCluster.spiketrains[0].units
            newSpikeTimes_ = a*newCluster.spiketrains[0].magnitude + b
            newSpikeTimes = []
            # throw out all spikes outside of WC windows
            # for t in newSpikeTimes_.magnitude:
            for t in newSpikeTimes_:
                if t >= WCWindows[i][0] and t <= WCWindows[i][1]:
                    newSpikeTimes.append(t)
            newSpikeTimes = neo.SpikeTrain(newSpikeTimes, t_stop=WCWindows[i][1], units=newCluster.spiketrains[0].units)
            newCluster.spiketrains[0] = newSpikeTimes
            newClusters[newCluster.clusterID] = newCluster

        # save clusters aligned to file
        groupName = 'ClustersPerShank_Aligned_' + experimentInfo['WC']['RecordingFilenames'][i]
        for g in groups:
            groupName += '_'
            groupName += g
        nsClustersName = os.path.join(experimentInfo['SiProbe']['ClusterBasePath'], groupName)
        # write clusters in samples of the aligned WC recording!!!
        clust.writer.write_neuroscope_units_per_shank(nsClustersName, newClusters, experimentInfo['WC']['SamplingRate'])
        # write WC .dat file for import into Neuroscope
        # minRange = np.min(signal)
        # maxRange = np.max(signal)
        # signal = signal/(maxRange - minRange)
        seg = neo.core.Segment()
        seg.analogsignals.append(WCSignals[i]['current'])
        seg.analogsignals.append(signal)
        nsSignalName = os.path.join(experimentInfo['SiProbe']['ClusterBasePath'], experimentInfo['WC']['RecordingFilenames'][i])
        nsSignalName += '_Neuroscope.dat'
        r = neo.io.RawBinarySignalIO(nsSignalName)
        r.write_segment(seg, dtype='int16', rangemin=np.min(signal), rangemax=np.max(signal))


def align_current_traces_probe_recordings(experimentInfo):
    '''
    simple wrapper around paired recording alignment
    :param experimentInfo:
    :return: dict of alignments:
     keys: filenames of WC recordings
     elements: alignment of cluster spike times to WC recordings
    '''
    pulseThreshold = 0.5
    syncChannels = 1
    ProbeAnalogDataName = os.path.join(experimentInfo['SiProbe']['DataBasePath'], 'analoginToDigitalin.dat')
    samplingRate = experimentInfo['SiProbe']['SamplingRate']
    probePulseSignal = ca.reader.read_Intan_digital_file(ProbeAnalogDataName, syncChannels, samplingRate)

    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    WCSignals = ca.reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])

    alignments = []
    for i, signal in enumerate(WCSignals):
        pulseAlignmentWindow = experimentInfo['WC']['PulsePeriodsOnProbe'][i]
        WCAlignmentWindow = (signal['pulseIn'].t_start.magnitude, signal['pulseIn'].t_stop.magnitude)
        alignmentPeriods = pulseAlignmentWindow, WCAlignmentWindow
        alignment = ca.recording_alignment.align_paired_recordings((probePulseSignal[0], signal['pulseIn']),
                                                                   alignmentPeriods, pulseThreshold,
                                                                   minimumInterval=0.1)
        # only align cluster spike times to current signal; WC sampling rate will stay fixed
        alignments.append(alignment[0])

    return alignments


# filter aligned current traces
def filter_current_trace(WCSignal, experimentInfo):
    '''
    Implements low-pass filtering of WC current trace
    :param WCSignal: neo AnalogSignal of current trace
    :return: FilteredWCSignal: neo AnalogSignal of filtered current trace
    '''
    filterOrder = 4
    cutoffFrequency = 2000.0*pq.Hz
    cutoffRad = cutoffFrequency/(WCSignal.sampling_rate*0.5)
    a, b = scipy.signal.butter(filterOrder, cutoffRad)
    print 'Filtering signal with order %d and cutoff frequency %.0f Hz' %(filterOrder, cutoffFrequency)
    filteredSignal = scipy.signal.filtfilt(a, b, WCSignal)

    return neo.core.AnalogSignal(filteredSignal, units=WCSignal.units, t_start=WCSignal.t_start,
                                 t_stop = WCSignal.t_stop, sampling_period=WCSignal.sampling_period)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        experimentInfoName = sys.argv[1]
        mode = sys.argv[2]
        if mode == 'extracellular':
            write_neuroscope_extracellular_clusters(experimentInfoName)
        elif mode == 'intracellular':
            write_neuroscope_intracellular_clusters(experimentInfoName)
        else:
            errstr = 'Mode %s not implented yet' % mode
            raise NotImplementedError(errstr)
    else:
        print 'Error: [experiment info name] and [mode (\'extracellular\'/\'intracellular\'] required as parameters'