##############################################
# example tool for loading units from SiProbe,
# intracellular currents, and synchronization
# signals.
# Aligns unit spike times and current traces,
# filters current traces and plots them along
# each other.
##############################################

import sys, os, ast, cPickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import neo
import ConnectionAnalyzer as ca
import ClusterProcessing as clust
import quantities as pq


def _filter_current_trace(WCSignal, experimentInfo):
    """
    Implements low-pass filtering of WC current trace
    :param WCSignal: neo AnalogSignal of current trace
    :return: FilteredWCSignal: neo AnalogSignal of filtered current trace
    """
    filterOrder = 4
    cutoffFrequency = 2500.0*pq.Hz
    cutoffRad = cutoffFrequency/(WCSignal.sampling_rate*0.5)
    a, b = scipy.signal.butter(filterOrder, cutoffRad)
    print 'Filtering signal with order %d and cutoff frequency %.0f Hz' %(filterOrder, cutoffFrequency)
    filteredSignal = scipy.signal.filtfilt(a, b, WCSignal)

    return neo.core.AnalogSignal(filteredSignal, units=WCSignal.units, t_start=WCSignal.t_start,
                                 t_stop = WCSignal.t_stop, sampling_period=WCSignal.sampling_period)


def _align_current_traces_probe_recordings(experimentInfo):
    """
    simple wrapper around paired recording alignment
    :param experimentInfo:
    :return: dict of alignments:
     keys: filenames of WC recordings
     elements: alignment of cluster spike times to WC recordings
    """
    pulseThreshold = 0.5
    syncChannels = 1
    ProbeAnalogDataName = os.path.join(experimentInfo['SiProbe']['DataBasePath'], experimentInfo['SiProbe']['PulseFileName'])
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
                                                                   alignmentPeriods, pulseThreshold, minimumInterval=0.05)
        # only align cluster spike times to current signal; WC sampling rate will stay fixed
        alignments.append(alignment[0])

    return alignments


def visualize_simultaneous_recordings(experiment_info_name):
    # load file with metadata
    with open(experiment_info_name, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    # load whole-cell recording data
    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    # these windows are the periods in which recordings are stable (i.e., no manipulation of holding potential)
    WCWindows = experimentInfo['WC']['RecordingPeriods']
    WCSignals = ca.reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])
    WCFilteredSignals = []
    for signal in WCSignals:
        WCFilteredSignals.append(_filter_current_trace(signal['current'].flatten(), experimentInfo))

    # load silicon probe data
    SiProbeDataFolder = experimentInfo['SiProbe']['DataBasePath']
    clusters = clust.reader.read_KS_clusters(SiProbeDataFolder, 'dev', ('good',),
                                             experimentInfo['SiProbe']['SamplingRate'])
    clustGroup = clust.ClusterGroup(clusters)
    for clusterID in clustGroup.clusters:
        clustGroup.remove_short_ISIs(clusterID, 2.0e-4)

    # compute linear transformation that aligns time axes of whole-cell and silicon probe recordings
    alignments = _align_current_traces_probe_recordings(experimentInfo)

    # for each whole-cell recording period, create a figure where we plot all spikes
    # of all recorded units as rasters aligned to the whole-cell current
    for i, signal in enumerate(WCFilteredSignals):
        fig = plt.figure(i + 1)
        ax1 = fig.add_subplot(2, 1, 1)
        t_start, t_stop = WCWindows[i]
        t_ = np.linspace(signal.t_start, signal.t_stop, len(signal))
        # plot whole-cell currents and rasters in stable period
        stable_signal = signal.magnitude[np.where((t_ >= t_start) * (t_ <= t_stop))]
        t_ = np.linspace(t_start, t_stop, len(stable_signal))
        ax1.plot(t_, stable_signal, 'k-', linewidth=0.5)
        ax1.set_ylabel('Current (pA)')
        alignedSpikeTrains = []
        for clusterID in clusters:
            cluster = clusters[clusterID]
            spikeTrain = cluster.spiketrains[0]
            a, b = alignments[i]
            t = ca.recording_alignment.linear_func(spikeTrain.magnitude, a, b)
            newSpikeTimes = t[np.where((t >= t_start) * (t <= t_stop))]
            if len(newSpikeTimes):
                newSpikeTrain = neo.core.SpikeTrain(newSpikeTimes, units=spikeTrain.units,
                                                    t_start=np.min(newSpikeTimes), t_stop=np.max(newSpikeTimes))
            else:
                newSpikeTrain = neo.core.SpikeTrain([], units=spikeTrain.units, t_start=0, t_stop=0)
            alignedSpikeTrains.append(newSpikeTrain)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.eventplot(alignedSpikeTrains, linewidths=0.5, colors='k')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Neuron #')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        experimentInfoName = sys.argv[1]
        visualize_simultaneous_recordings(experimentInfoName)