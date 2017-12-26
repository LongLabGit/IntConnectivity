##############################################
# tool for loading units from SiProbe,
# intracellular currents, and synchronization
# signals.
# Aligns unit spike times and current traces,
# filters current traces and generates
# spike-triggered traces and averages of
# raw and filtered current traces
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

def main(experimentInfoName):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    WCWindows = experimentInfo['WC']['RecordingPeriods']
    WCSignals = ca.reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])
    WCFilteredSignals = []
    for signal in WCSignals:
        WCFilteredSignals.append(filter_current_trace(signal['current'].flatten(), experimentInfo))

    SiProbeDataFolder = experimentInfo['SiProbe']['DataBasePath']
    alignments = align_current_traces_probe_recordings(experimentInfo)
    clusters = clust.reader.read_KS_clusters(SiProbeDataFolder, clusteringSrcFolder,
                                       'dev', ('good',), experimentInfo['SiProbe']['SamplingRate'])
    clustGroup = clust.ClusterGroup(clusters)
    for clusterID in clustGroup.clusters:
        clustGroup.remove_short_ISIs(clusterID, 2.0e-4)
    nsClustersName = os.path.join(experimentInfo['SiProbe']['ClusterBasePath'], 'ClustersPerShank')
    clust.writer.write_neuroscope_units_per_shank(nsClustersName, clustGroup.clusters, experimentInfo['SiProbe']['SamplingRate'])

    STAlignedTraces = {}
    STAlignedSnippets = {}
    alignedWindow = (-10.0*pq.ms, 25.0*pq.ms)

    # for clusterID in clusters.keys():
    # for clusterID in [0, 82]:
    for clusterID in [979]:
        cluster = clustGroup.clusters[clusterID]
        print 'Removed %d duplicate spike times due to misalignment' % cluster.nrMergedDuplicateSpikes
        # cluster = clusters[clusterID]
        print 'Collecting spike time-aligned traces for cluster %d' % clusterID
        STAlignedTraces[clusterID], STAlignedSnippets[clusterID], spikesUsed = compute_ST_traces_average(cluster, WCFilteredSignals, experimentInfo['WC']['RecordingFilenames'],
                                                               WCWindows, alignments, alignedWindow)
        timeAxis = np.linspace(alignedWindow[0].magnitude, alignedWindow[1].magnitude, len(STAlignedTraces[clusterID]))
        plt.figure(clusterID)
        SE = np.std(STAlignedSnippets[clusterID].snippets, axis=0)/np.sqrt(len(STAlignedSnippets[clusterID].snippets))
        plt.plot(timeAxis, STAlignedTraces[clusterID], 'k', linewidth=0.5)
        plt.plot(timeAxis, STAlignedTraces[clusterID].magnitude + SE, 'k--', linewidth=0.5)
        plt.plot(timeAxis, STAlignedTraces[clusterID].magnitude - SE, 'k--', linewidth=0.5)
        plt.plot([0, 0], plt.ylim(), 'r--', linewidth=0.5)
        plt.xlim([alignedWindow[0].magnitude, alignedWindow[1].magnitude])
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (pA)')
        titleStr = 'STA of cluster %d; using %d spikes' % (clusterID, spikesUsed)
        plt.title(titleStr)
        STAName = 'STA_Cluster_%d.pdf' % clusterID
        STAFigName = os.path.join(experimentInfo['STA']['DataBasePathI'], STAName)
        plt.savefig(STAFigName)
        STPickleName = 'STSnippets_Cluster_%d.pkl' % clusterID
        STSnippetName = os.path.join(experimentInfo['STA']['DataBasePathI'], STPickleName)
        with open(STSnippetName, 'wb') as snippetOutFile:
            cPickle.dump(STAlignedSnippets[clusterID], snippetOutFile, cPickle.HIGHEST_PROTOCOL)
    # plt.show()

def individual_spike_triggered_currents(experimentInfoName, clusterID):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    # STPickleName = 'STSnippets_Cluster_%d.pkl' % clusterID
    STPickleNameShuffled = 'STSnippets_shuffled_100x_Cluster_%d_95-300s.pkl' % clusterID
    STSnippetShuffledName = os.path.join(experimentInfo['STA']['DataBasePathI'], STPickleNameShuffled)
    STPickleName = 'STSnippets_Cluster_%d.pkl' % clusterID
    STSnippetName = os.path.join(experimentInfo['STA']['DataBasePathI'], STPickleName)
    with open(STSnippetShuffledName, 'rb') as STSnippetShuffledFile:
        clusterSnippetsShuffled = cPickle.load(STSnippetShuffledFile)
    with open(STSnippetName, 'rb') as STSnippetFile:
        clusterSnippets = cPickle.load(STSnippetFile)
    clusterSTA = np.mean(clusterSnippets.snippets, axis=0)
    peakBin = np.argmax(clusterSTA)
    tPeak = peakBin/experimentInfo['WC']['SamplingRate'] # s
    tOffset = 0.01 # s
    offsetBin = int(tOffset*experimentInfo['WC']['SamplingRate'])
    timeAxis = (np.array(range(len(clusterSTA)))/experimentInfo['WC']['SamplingRate'] - tOffset)*1000.0 # ms
    tOffsetPlot = 0.3 # ms
    tPeakPlot = (tPeak-tOffset)*1000.0 # ms
    tPeakPlot2 = 1.2 # ms

    tracesPerPage = 10
    # nrTraces = 40
    # traceStep = len(clusterSnippets.snippets)/nrTraces
    nrPages = len(clusterSnippets.snippets)/tracesPerPage + 1
    # nrPages = nrTraces/tracesPerPage
    # for i in range(nrPages):
    # # for i in range(2):
    #     spacing = 5.0*clusterSnippets.snippets[0].units
    #     plt.figure(i)
    #     lineMin = np.min(clusterSTA)
    #     plt.plot(timeAxis, clusterSTA, 'r')
    #     offset = np.max(clusterSTA)*clusterSnippets.snippets[0].units
    #     for j in range(tracesPerPage):
    #         traceNr = j + i*tracesPerPage
    #         # traceNr = (j + i*tracesPerPage)*traceStep
    #         if traceNr >= len(clusterSnippets.snippets):
    #             break
    #         plotTrace = clusterSnippets.snippets[traceNr]
    #         minShift = np.min(plotTrace)
    #         plotTrace += offset + spacing - minShift
    #         plt.plot(timeAxis, plotTrace, 'k')
    #         offset = np.max(plotTrace)
    #     lineMax = offset
    #     plt.plot([tOffsetPlot, tOffsetPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.plot([tPeakPlot, tPeakPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.plot([tPeakPlot2, tPeakPlot2], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.xlabel('Time relative to spike (ms)')
    #     plt.ylabel('Current (pA)')
    #     titleStr = 'Cluster %d - page %d - spikes %d - %d' % (clusterID, i + 1, i*tracesPerPage + 1, traceNr + 1)
    #     plt.title(titleStr)
    #     clusterPageName = 'Cluster_%d_page_%d.pdf' % (clusterID, i)
    #     pageName = os.path.join(experimentInfo['STA']['DataBasePathI'], clusterPageName)
    #     plt.savefig(pageName)
    #     # plt.show()
    lastFigure = nrPages

    tPeakOffset1 = 2.0
    tPeakOffset2 = 3.0
    ampOffsetBin = int((tOffsetPlot/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    ampPeakBin1 = int((tPeakOffset1/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    ampPeakBin2 = int((tPeakOffset2/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    # ampPeakBin = peakBin
    amplitudes = []
    amplitudes_0_95 = []
    amplitudes_95_end = []
    for i, snippet in enumerate(clusterSnippets.snippets):
        # amp = np.max(snippet[ampOffsetBin:ampPeakBin]) - snippet[ampOffsetBin]
        amp = np.max(snippet[ampPeakBin1:ampPeakBin2]) - snippet[ampOffsetBin]
        amplitudes.append(amp.magnitude)
        if clusterSnippets.snippetSpikeTimes[i] < 95.0:
            amplitudes_0_95.append(amp.magnitude)
        else:
            amplitudes_95_end.append(amp.magnitude)
    amplitudes = np.array(amplitudes)
    amplitudes_0_95 = np.array(amplitudes_0_95)
    amplitudes_95_end = np.array(amplitudes_95_end)

    amplitudesShuffled = []
    for i, snippet in enumerate(clusterSnippetsShuffled.snippets):
        # amp = np.max(snippet[ampOffsetBin:ampPeakBin]) - snippet[ampOffsetBin]
        amp = np.max(snippet[ampPeakBin1:ampPeakBin2]) - snippet[ampOffsetBin]
        amplitudesShuffled.append(amp.magnitude)
    amplitudesShuffled = np.array(amplitudesShuffled)

    # plt.figure(1)
    # plt.plot(clusterSnippets.snippetSpikeTimes, amplitudes, 'ko')
    print len(np.where(clusterSnippets.snippetSpikeTimes < 95)[0])/95.0
    print len(np.where(clusterSnippets.snippetSpikeTimes >= 95)[0])/205.0
    # plt.xlabel('Time during WC recording (s)')
    # plt.ylabel('Current amplitude (pA)')
    # plt.show()

    # plt.figure(nrPages + 1)
    binSize = 2.0
    # bins = np.arange(np.min(amplitudes), np.max(amplitudes)+binSize, binSize)
    # bins = np.arange(np.min(amplitudesShuffled), np.max(amplitudes_0_95)+binSize, binSize)
    bins = np.arange(np.min(amplitudesShuffled), np.max(amplitudes_95_end)+binSize, binSize)
    # hist, _ = np.histogram(amplitudes, bins)
    # # plt.hist(amplitudes, bins)
    # plt.bar(bins[:-1], hist, width=binSize)
    # plt.xlabel('Amplitude (pA)')
    # plt.ylabel('Events')

    # plt.figure(nrPages + 2)
    # hist_0_95, _ = np.histogram(amplitudes_0_95, bins)
    # hist_shuffled, _ = np.histogram(amplitudesShuffled, bins)
    # hist_shuffled = np.array(hist_shuffled, dtype='float64')
    # hist_shuffled *= 1.0*np.sum(hist_0_95)/np.sum(hist_shuffled)
    # # plt.hist(amplitudes, bins)
    # plt.bar(bins[:-1], hist_shuffled, width=binSize, linewidth=0, color='k', label='shuffled')
    # plt.bar(bins[:-1], hist_0_95, width=binSize, linewidth=0, color='r', label='spike times')
    # plt.xlabel('Amplitude (pA)')
    # plt.ylabel('Events (during 0-95 s)')
    # plt.legend()

    plt.figure(nrPages + 2)
    hist_95_end, _ = np.histogram(amplitudes_95_end, bins)
    hist_shuffled, _ = np.histogram(amplitudesShuffled, bins)
    hist_shuffled = np.array(hist_shuffled, dtype='float64')
    hist_shuffled *= 1.0*np.sum(hist_95_end)/np.sum(hist_shuffled)
    # plt.hist(amplitudes, bins)
    plt.bar(bins[:-1], hist_shuffled, width=binSize, linewidth=0, color='k', label='shuffled')
    plt.bar(bins[:-1], hist_95_end, width=binSize, linewidth=0, color='r', label='spike times')
    plt.xlabel('Amplitude (pA)')
    plt.ylabel('Events (during 95 s - end)')
    plt.legend()

    # # ISIs
    # earlySpikes = clusterSnippets.snippetSpikeTimes[np.where(clusterSnippets.snippetSpikeTimes < 95.0)]
    # earlyISIs = np.diff(earlySpikes)
    # earlyAmps = amplitudes_0_95[1:]
    # lateSpikes = clusterSnippets.snippetSpikeTimes[np.where(clusterSnippets.snippetSpikeTimes >= 95.0)]
    # lateISIs = np.diff(lateSpikes)
    # lateAmps = amplitudes_95_end[1:]
    # plt.figure(nrPages + 4)
    # plt.plot(earlyISIs, earlyAmps, 'ro', label='<95 s')
    # plt.plot(lateISIs, lateAmps, 'ko', label='>=95 s')
    # plt.legend()
    # plt.xlabel('ISI (s)')
    # plt.ylabel('Event amplitude (pA)')
    # plt.figure(nrPages + 5)
    # plt.semilogx(earlyISIs, earlyAmps, 'ro', label='<95 s')
    # plt.semilogx(lateISIs, lateAmps, 'ko', label='>=95 s')
    # plt.legend()
    # plt.xlabel('ISI (s)')
    # plt.ylabel('Event amplitude (pA)')
    plt.show()

    # amps_0_6 = np.where(amplitudes <= 6.0)[0]
    # nrPages = len(amps_0_6)/tracesPerPage
    # for i in range(nrPages):
    # # for i in range(2):
    #     spacing = 5.0*clusterSnippets.snippets[0].units
    #     plt.figure(lastFigure + i)
    #     lineMin = np.min(clusterSTA)
    #     plt.plot(timeAxis, clusterSTA, 'r')
    #     offset = np.max(clusterSTA)*clusterSnippets.snippets[0].units
    #     for j in range(tracesPerPage):
    #         traceNr = j + i*tracesPerPage
    #         if traceNr >= len(amps_0_6):
    #             break
    #         traceNr = amps_0_6[j + i*tracesPerPage]
    #         plotTrace = clusterSnippets.snippets[traceNr]
    #         minShift = np.min(plotTrace)
    #         plotTrace += offset + spacing - minShift
    #         plt.plot(timeAxis, plotTrace, 'k')
    #         offset = np.max(plotTrace)
    #     lineMax = offset
    #     plt.plot([tOffsetPlot, tOffsetPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.plot([tPeakPlot, tPeakPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.plot([tPeakPlot2, tPeakPlot2], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.xlabel('Time relative to spike (ms)')
    #     plt.ylabel('Current (pA)')
    #     clusterPageName = 'Cluster_%d_amps_0_6_page_%d.pdf' % (clusterID, i)
    #     pageName = os.path.join(experimentInfo['STA']['DataBasePathI'], clusterPageName)
    #     plt.savefig(pageName)
    #     # plt.show()
    # lastFigure += nrPages
    #
    # amps_6_12 = np.where((amplitudes > 6.0)*(amplitudes <=12.0))[0]
    # nrPages = len(amps_6_12)/tracesPerPage
    # for i in range(nrPages):
    # # for i in range(2):
    #     spacing = 5.0*clusterSnippets.snippets[0].units
    #     plt.figure(lastFigure + i)
    #     lineMin = np.min(clusterSTA)
    #     plt.plot(timeAxis, clusterSTA, 'r')
    #     offset = np.max(clusterSTA)*clusterSnippets.snippets[0].units
    #     for j in range(tracesPerPage):
    #         traceNr = j + i*tracesPerPage
    #         if traceNr >= len(amps_6_12):
    #             break
    #         traceNr = amps_6_12[j + i*tracesPerPage]
    #         plotTrace = clusterSnippets.snippets[traceNr]
    #         minShift = np.min(plotTrace)
    #         plotTrace += offset + spacing - minShift
    #         plt.plot(timeAxis, plotTrace, 'k')
    #         offset = np.max(plotTrace)
    #     lineMax = offset
    #     plt.plot([tOffsetPlot, tOffsetPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.plot([tPeakPlot, tPeakPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.plot([tPeakPlot2, tPeakPlot2], [lineMin, lineMax], 'r--', linewidth=0.5)
    #     plt.xlabel('Time relative to spike (ms)')
    #     plt.ylabel('Current (pA)')
    #     clusterPageName = 'Cluster_%d_amps_6_12_page_%d.pdf' % (clusterID, i)
    #     pageName = os.path.join(experimentInfo['STA']['DataBasePathI'], clusterPageName)
    #     plt.savefig(pageName)
    #     # plt.show()
    # # plt.show()

def generate_STA_shuffled_ISIs(experimentInfoName, shuffleClusterID, nShuffle):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    WCWindows = experimentInfo['WC']['RecordingPeriods']
    WCSignals = ca.reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])
    WCFilteredSignals = []
    for signal in WCSignals:
        WCFilteredSignals.append(filter_current_trace(signal['current'].flatten(), experimentInfo))

    SiProbeDataFolder = experimentInfo['SiProbe']['DataBasePath']
    alignments = align_current_traces_probe_recordings(experimentInfo)
    clusters = clust.reader.read_KS_clusters(SiProbeDataFolder, clusteringSrcFolder,
                                       'dev', ('good',), experimentInfo['SiProbe']['SamplingRate'])
    clustGroup = clust.ClusterGroup(clusters)
    for clusterID in clustGroup.clusters:
        clustGroup.remove_short_ISIs(clusterID, 2.0e-4)

    STAlignedTraces = {}
    alignedWindow = (-10.0*pq.ms, 25.0*pq.ms)
    # for clusterID in clustGroup.clusters.keys():
    for clusterID in [shuffleClusterID]:
        cluster = clustGroup.clusters[clusterID]
        print 'Collecting shuffled spike time-aligned traces for cluster %d' % clusterID
        STAlignedTraces[clusterID], currentSnippets, usedSpikes = compute_shuffled_ST_traces(cluster, WCFilteredSignals, experimentInfo['WC']['RecordingFilenames'],
                                                               WCWindows, alignments, alignedWindow, nShuffle)
        # STAlignedTraces[clusterID], _ = compute_shuffled_ST_traces_average(cluster, WCFilteredSignals, experimentInfo['WC']['RecordingFilenames'],
        #                                                        WCWindows, alignments, alignedWindow, nShuffle)
        timeAxis = np.linspace(alignedWindow[0].magnitude, alignedWindow[1].magnitude, len(STAlignedTraces[clusterID]))
        plt.figure(clusterID)
        plt.plot(timeAxis, STAlignedTraces[clusterID], 'k', linewidth=0.5)
        # plt.plot([0, 0], plt.ylim(), 'r--', linewidth=0.5)
        # plt.xlim([alignedWindow[0].magnitude, alignedWindow[1].magnitude])
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (pA)')
        titleStr = 'Shuffled STA of cluster %d; using %d shuffles' % (clusterID, nShuffle)
        plt.title(titleStr)
        plt.show()
        # STAName = 'STA_shuffled_Cluster_%d.pdf' % clusterID
        # STAFigName = os.path.join(experimentInfo['STA']['DataBasePathI'], STAName)
        # plt.savefig(STAFigName)
        # STPickleName = 'STA_shuffled_Cluster_%d.pkl' % clusterID
        # STSnippetName = os.path.join(experimentInfo['STA']['DataBasePathI'], STPickleName)
        # with open(STSnippetName, 'wb') as snippetOutFile:
        #     cPickle.dump(STAlignedTraces[clusterID], snippetOutFile, cPickle.HIGHEST_PROTOCOL)
        STSnippetsPickleName = 'STSnippets_shuffled_%dx_Cluster_%d_95-300s.pkl' % (nShuffles, clusterID)
        STSnippetName = os.path.join(experimentInfo['STA']['DataBasePathI'], STSnippetsPickleName)
        with open(STSnippetName, 'wb') as snippetOutFile:
            cPickle.dump(currentSnippets, snippetOutFile, cPickle.HIGHEST_PROTOCOL)


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
                                                                   alignmentPeriods, pulseThreshold, minimumInterval=0.1)
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
    # # GK: Savitzky-Golay Filter, 2nd order, 1.75ms window
    # filterOrder = 2
    # windowLengthTime = 0.00175
    # windowLength = int(round(windowLengthTime*experimentInfo['WC']['SamplingRate']))
    # if not windowLength%2:
    #     windowLength += 1
    # filteredSignal1 = scipy.signal.savgol_filter(WCSignal, windowLength, filterOrder)

    filterOrder = 4
    cutoffFrequency = 2500.0*pq.Hz
    cutoffRad = cutoffFrequency/(WCSignal.sampling_rate*0.5)
    a, b = scipy.signal.butter(filterOrder, cutoffRad)
    print 'Filtering signal with order %d and cutoff frequency %.0f Hz' %(filterOrder, cutoffFrequency)
    filteredSignal = scipy.signal.filtfilt(a, b, WCSignal)

    # timeAxis = np.linspace(WCSignal.t_start.magnitude, WCSignal.t_stop.magnitude, len(WCSignal))
    # plt.figure(3)
    # plt.plot(timeAxis, WCSignal, 'k', linewidth=0.5)
    # plt.plot(timeAxis, filteredSignal1, 'r', linewidth=0.5)
    # plt.plot(timeAxis, filteredSignal2, 'b', linewidth=0.5)
    # titleStr = 'LP frequency cutoff = %.0f Hz' % cutoffFrequency.magnitude
    # plt.title(titleStr)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Current (pA)')
    # plt.xlim([26.976, 26.990])
    # plt.ylim([180, 300])
    # plt.show()

    return neo.core.AnalogSignal(filteredSignal, units=WCSignal.units, t_start=WCSignal.t_start,
                                 t_stop = WCSignal.t_stop, sampling_period=WCSignal.sampling_period)

# compute ST traces and averages
def compute_ST_traces_average(cluster, WCSignals, WCSignalNames, WCWindows, WCAlignments, alignedWindow):
    '''
    aligns WC currents to spike times in cluster
    :param cluster: Cluster object
    :param WCSignals: array of neo AnalogSignals to be aligned
    :param WCWindows: array of time windows to use in each WC recordings
    :param WCAlignments: array of transformations aligning cluster spikes to WC recordings
    :param alignedWindow: array with time before/after each spike time to include in
    ST traces and average
    :return:
    '''
    spikeTrain = cluster.spiketrains[0]
    alignedSpikeTrains = []
    for j in range(len(WCAlignments)):
        a, b = WCAlignments[j]
        newSpikeTimes = ca.recording_alignment.linear_func(spikeTrain.magnitude, a, b)
        newSpikeTrain = neo.core.SpikeTrain(newSpikeTimes, units=spikeTrain.units, t_start=np.min(newSpikeTimes),
                                            t_stop=np.max(newSpikeTimes))
        alignedSpikeTrains.append(newSpikeTrain)
    cluster.alignedspiketrains = alignedSpikeTrains

    # determine window width in samples
    alignedWindowStart = alignedWindow[0].rescale(WCSignals[0].t_start.units)
    alignedWindowStop = alignedWindow[1].rescale(WCSignals[0].t_start.units)
    windowBins = int(np.ceil((alignedWindowStop - alignedWindowStart)*WCSignals[0].sampling_rate))

    # iterate through alignedspiketrains
    sta = neo.core.AnalogSignal(signal=np.zeros(windowBins), units=WCSignals[0].units, t_start=alignedWindowStart,
                                t_stop=alignedWindowStop, sampling_rate=WCSignals[0].sampling_rate)
    usedSpikes = 0
    unusedSpikes = 0
    snippets = []
    snippetFileNames =[]
    snippetSpikeTimes = []
    for i, signal in enumerate(WCSignals):
        spikeTrain = cluster.alignedspiketrains[i]
        print '\tSelecting spike time-aligned snippets from %d spikes in signal %s' % (len(spikeTrain), WCSignalNames[i])
        # find aligned spike times >= 0  and <= recording duration
        for t in spikeTrain:
            # if t + alignedWindowStart >= signal.t_start and t + alignedWindowStop <= signal.t_stop:
            if t + alignedWindowStart >= WCWindows[i][0] and t + alignedWindowStop <= WCWindows[i][1]:
                # copy snippets of analog signal of duration defined by aligned window
                # (if aligned window is completely contained within recording)
                startBin = int(np.floor((t + alignedWindowStart)*signal.sampling_rate))
                # subSignal = neo.AnalogSignal(signal[startBin:startBin+windowBins], units=signal.units,
                #                              t_start=alignedWindowStart, t_stop=alignedWindowStop,
                #                              sampling_rate=signal.sampling_rate)
                sta += signal[startBin:startBin+windowBins].reshape(sta.shape)
                usedSpikes += 1
                # currentSnippets.append(signal[startBin:startBin+windowBins].magnitude)
                snippets.append(signal[startBin:startBin+windowBins])
                snippetFileNames.append(WCSignalNames[i])
                snippetSpikeTimes.append(t)
                # snippets.append(snippet)
            else:
                unusedSpikes += 1

    # compute average in WCWindows
    if usedSpikes:
        sta /= usedSpikes
    # currentSnippets = np.array(currentSnippets)
    currentSnippets = ca.sts.SnippetArray(snippets, snippetFileNames, snippetSpikeTimes)
    print '\tComputed average from %d of %d spike times' % (usedSpikes, usedSpikes+unusedSpikes)
    return sta, currentSnippets, usedSpikes

# shuffle ISIs, compute ST traces and return average
def compute_shuffled_ST_traces_average(cluster, WCSignals, WCSignalNames, WCWindows, WCAlignments, alignedWindow, nShuffles):
    '''
    aligns WC currents to spike times in cluster
    :param cluster: Cluster object
    :param WCSignals: array of neo AnalogSignals to be aligned
    :param WCWindows: array of time windows to use in each WC recordings
    :param WCAlignments: array of transformations aligning cluster spikes to WC recordings
    :param alignedWindow: array with time before/after each spike time to include in
    ST traces and average
    :param nShuffles: number of surrogate spike trains to be generated
    :return:
    '''
    oldSpikeTrain = cluster.spiketrains[0]
    alignedSpikeTrains = []
    for j in range(len(WCAlignments)):
        a, b = WCAlignments[j]
        newSpikeTimes = ca.recording_alignment.linear_func(oldSpikeTrain.magnitude, a, b)
        newSpikeTrain = neo.core.SpikeTrain(newSpikeTimes, units=oldSpikeTrain.units, t_start=min(0, np.min(newSpikeTimes)),
                                            t_stop=np.max(newSpikeTimes))
        alignedSpikeTrains.append(newSpikeTrain)
    cluster.alignedspiketrains = alignedSpikeTrains

    # determine window width in samples
    alignedWindowStart = alignedWindow[0].rescale(WCSignals[0].t_start.units)
    alignedWindowStop = alignedWindow[1].rescale(WCSignals[0].t_start.units)
    windowBins = int(np.ceil((alignedWindowStop - alignedWindowStart)*WCSignals[0].sampling_rate))

    # iterate through alignedspiketrains
    sta = neo.core.AnalogSignal(signal=np.zeros(windowBins), units=WCSignals[0].units, t_start=alignedWindowStart,
                                t_stop=alignedWindowStop, sampling_rate=WCSignals[0].sampling_rate)
    usedSpikes = 0
    unusedSpikes = 0
    # snippets = []
    # snippetFileNames =[]
    # snippetSpikeTimes = []
    for i, signal in enumerate(WCSignals):
        spikeTrain = cluster.alignedspiketrains[i]
        print 'Generating %d surrogates of spike train in signal %s' % (nShuffles, WCSignalNames[i])
        spikesInWindowIndex = (spikeTrain >= WCWindows[i][0]) * (spikeTrain <= WCWindows[i][1])
        spikesInWindow = spikeTrain[np.where(spikesInWindowIndex)]
        surrogateSpikeTrains = elephant.spike_train_surrogates.shuffle_isis(spikesInWindow, nShuffles)
        for train in surrogateSpikeTrains:
            print '\tSelecting spike time-aligned snippets from %d spikes in signal %s' % (len(spikeTrain), WCSignalNames[i])
            # find aligned spike times >= 0  and <= recording duration
            for t in train:
                if t + alignedWindowStart >= WCWindows[i][0] and t + alignedWindowStop <= WCWindows[i][1]:
                    # copy snippets of analog signal of duration defined by aligned window
                    # (if aligned window is completely contained within recording)
                    startBin = int(np.floor((t + alignedWindowStart)*signal.sampling_rate))
                    sta += signal[startBin:startBin+windowBins].reshape(sta.shape)
                    usedSpikes += 1
                    # snippets.append(signal[startBin:startBin+windowBins])
                    # snippetFileNames.append(WCSignalNames[i])
                    # snippetSpikeTimes.append(t)
                else:
                    unusedSpikes += 1

    # compute average in WCWindows
    if usedSpikes:
        sta /= usedSpikes
    # currentSnippets = ca.sts.SnippetArray(snippets, snippetFileNames, snippetSpikeTimes)
    print '\tComputed shuffled average from %d of %d spike times' % (usedSpikes, usedSpikes+unusedSpikes)
    return sta, usedSpikes

def compute_shuffled_ST_traces(cluster, WCSignals, WCSignalNames, WCWindows, WCAlignments, alignedWindow, nShuffles):
    '''
    aligns WC currents to spike times in cluster
    :param cluster: Cluster object
    :param WCSignals: array of neo AnalogSignals to be aligned
    :param WCWindows: array of time windows to use in each WC recordings
    :param WCAlignments: array of transformations aligning cluster spikes to WC recordings
    :param alignedWindow: array with time before/after each spike time to include in
    ST traces and average
    :param nShuffles: number of surrogate spike trains to be generated
    :return:
    '''
    spikeTrain = cluster.spiketrains[0]
    alignedSpikeTrains = []
    for j in range(len(WCAlignments)):
        a, b = WCAlignments[j]
        newSpikeTimes = ca.recording_alignment.linear_func(spikeTrain.magnitude, a, b)
        newSpikeTrain = neo.core.SpikeTrain(newSpikeTimes, units=spikeTrain.units, t_start=np.min(newSpikeTimes),
                                            t_stop=np.max(newSpikeTimes))
        alignedSpikeTrains.append(newSpikeTrain)
    cluster.alignedspiketrains = alignedSpikeTrains

    # determine window width in samples
    alignedWindowStart = alignedWindow[0].rescale(WCSignals[0].t_start.units)
    alignedWindowStop = alignedWindow[1].rescale(WCSignals[0].t_start.units)
    windowBins = int(np.ceil((alignedWindowStop - alignedWindowStart)*WCSignals[0].sampling_rate))

    # iterate through alignedspiketrains
    sta = neo.core.AnalogSignal(signal=np.zeros(windowBins), units=WCSignals[0].units, t_start=alignedWindowStart,
                                t_stop=alignedWindowStop, sampling_rate=WCSignals[0].sampling_rate)
    usedSpikes = 0
    unusedSpikes = 0
    snippets = []
    snippetFileNames =[]
    snippetSpikeTimes = []
    for i, signal in enumerate(WCSignals):
        spikeTrain = cluster.alignedspiketrains[i]
        print 'Generating %d surrogates of spike train in signal %s' % (nShuffles, WCSignalNames[i])
        spikesInWindowIndex = (spikeTrain >= WCWindows[i][0]) * (spikeTrain <= WCWindows[i][1])
        spikesInWindow = spikeTrain[np.where(spikesInWindowIndex)]
        spikesInWindow.t_start = 0.0*spikesInWindow.units
        surrogateSpikeTrains = elephant.spike_train_surrogates.shuffle_isis(spikesInWindow, nShuffles)
        # find aligned spike times >= 0  and <= recording duration
        for train in surrogateSpikeTrains:
            print '\tSelecting spike time-aligned snippets from %d spikes in signal %s' % (len(spikeTrain), WCSignalNames[i])
            for t in train:
                # if t + alignedWindowStart >= signal.t_start and t + alignedWindowStop <= signal.t_stop:
                if t + alignedWindowStart >= WCWindows[i][0] and t + alignedWindowStop <= WCWindows[i][1]:
                    # copy snippets of analog signal of duration defined by aligned window
                    # (if aligned window is completely contained within recording)
                    startBin = int(np.floor((t + alignedWindowStart)*signal.sampling_rate))
                    # subSignal = neo.AnalogSignal(signal[startBin:startBin+windowBins], units=signal.units,
                    #                              t_start=alignedWindowStart, t_stop=alignedWindowStop,
                    #                              sampling_rate=signal.sampling_rate)
                    sta += signal[startBin:startBin+windowBins].reshape(sta.shape)
                    usedSpikes += 1
                    # currentSnippets.append(signal[startBin:startBin+windowBins].magnitude)
                    snippets.append(signal[startBin:startBin+windowBins])
                    snippetFileNames.append(WCSignalNames[i])
                    snippetSpikeTimes.append(t)
                    # snippets.append(snippet)
                else:
                    unusedSpikes += 1

    # compute average in WCWindows
    if usedSpikes:
        sta /= usedSpikes
    # currentSnippets = np.array(currentSnippets)
    currentSnippets = ca.sts.SnippetArray(snippets, snippetFileNames, snippetSpikeTimes)
    print '\tComputed average from %d of %d spike times' % (usedSpikes, usedSpikes+unusedSpikes)
    return sta, currentSnippets, usedSpikes

# save data and generate average plots (separate tool?)
def save_ST_traces_average(experimentInfo, STAlignedTraces):
    '''
    TBD
    :return: nothing
    '''
    pass

if __name__ == '__main__':
    if len(sys.argv) == 2:
        experimentInfoName = sys.argv[1]
        main(experimentInfoName)
    elif len(sys.argv) == 3:
        experimentInfoName = sys.argv[1]
        clusterID = int(sys.argv[2])
        individual_spike_triggered_currents(experimentInfoName, clusterID)
        # nShuffles = int(sys.argv[2])
        # generate_STA_shuffled_ISIs(experimentInfoName, nShuffles)
    elif len(sys.argv) == 4:
        experimentInfoName = sys.argv[1]
        clusterID = int(sys.argv[2])
        nShuffles = int(sys.argv[3])
        generate_STA_shuffled_ISIs(experimentInfoName, clusterID, nShuffles)
    else:
        print 'Error: Experiment info file required!'