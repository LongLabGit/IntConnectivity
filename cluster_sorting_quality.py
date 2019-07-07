#####################################
# Python implementation of some
# cluster sorting quality metrics
#####################################

import sys
import os, os.path
import ast
import numpy as np
from scipy import signal
import ClusterProcessing as cp
import ConnectionAnalyzer as ca

clusteringSrcFolder = 'E:\\User\\project_src\\physiology\\Clustering'

# BP filter parameters
filter_order = 3
high_pass = 500.0 # KiloSort default
low_pass = 0.475 # just below Nyquist

def _set_up_filter(order, highpass, lowpass, fs):
    return signal.butter(order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def compute_quality_metrics(experimentInfoName):
    '''

    :param experimentInfoName:
    :return:
    '''
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    print 'Loading clusters...'
    clusters = cp.reader.read_KS_clusters(experimentInfo['SiProbe']['DataBasePath'], clusteringSrcFolder,
                                       'dev', ('good',), experimentInfo['SiProbe']['SamplingRate'])
    datFilename = os.path.join(experimentInfo['SiProbe']['DataBasePath'], 'amplifier_cut.dat')
    compute_unit_SNR(clusters, datFilename, experimentInfo)


def compute_unit_SNR(clusters, datFilename, experimentInfo):
    '''

    :param clusters:
    :param datFilename:
    :param experimentInfo:
    :return:
    '''
    print 'Calculating effective SNR for %d units...' % len(clusters)

    # set up high-pass filter
    fs = experimentInfo['SiProbe']['SamplingRate']
    filter_b, filter_a = _set_up_filter(filter_order, high_pass, low_pass * fs, fs)
    def bp_filter(x):
        return signal.filtfilt(filter_b, filter_a, x, axis=0)

    alignments = ca.recording_alignment.align_current_traces_probe_recordings(experimentInfo, 0.5)
    datInfo = os.stat(datFilename)
    datSize = datInfo.st_size
    # template shape: NChannels x NTimePoints
    nChannels = experimentInfo['SiProbe']['Channels']
    binaryDType = np.dtype('int16')
    datSamples = datSize/(binaryDType.itemsize*nChannels)
    # write hybrid ground truth signal: load file and only add new spike times
    originalData = np.memmap(datFilename, dtype=binaryDType.name, mode='r', shape=(nChannels, datSamples), order='F')

    madPerChannel = {}
    madPerChannelAligned = {}
    clusterSNR = {}
    clusterSNRAligned = {}
    for clusterID in clusters:
        cluster = clusters[clusterID]
        # clusterTemplate = cluster.template
        # templateMaxChannel = np.argmax(np.mean(np.abs(clusterTemplate), axis=1))
        templateMaxChannel = cluster.maxChannel
        # scale template amplitude: use fitted amplitudes in reference run
        # clusterTemplateAmplitudes = cluster.templateAmplitudes
        # get raw amplitudes (converison to muV not necessary)
        # +- 2.5ms around spike time
        sampleWindow = int(2.5*experimentInfo['SiProbe']['SamplingRate']/1000.0 + 0.5)
        medianEndBin = int(2*sampleWindow/3.0)
        rawAmplitudes = []
        rawAmplitudesAligned = []
        spikeSamples_ = np.array(cluster.spiketrains[0].magnitude * experimentInfo['SiProbe']['SamplingRate'] + 0.5,
                                   dtype='int64')
        spikeSamples = spikeSamples_[np.where((spikeSamples_ >= sampleWindow) * (spikeSamples_ <= datSamples - sampleWindow))]
        spikeTimesAligned = []
        recordingPeriods = experimentInfo['WC']['RecordingPeriods']
        for i in range(len(alignments)):
            a, b = alignments[i]
            newSpikeTimes = ca.recording_alignment.linear_func(cluster.spiketrains[0].magnitude, a, b)
            spikeTimesAligned.extend(cluster.spiketrains[0].magnitude[np.where((newSpikeTimes >= recordingPeriods[i][0]) * (newSpikeTimes <= recordingPeriods[i][1]))])
        spikeSamplesAligned = np.array(np.array(spikeTimesAligned) * experimentInfo['SiProbe']['SamplingRate'] + 0.5, dtype='int64')
        # allWaveForms = []
        for tSample in spikeSamples:
            tmpWF = originalData[templateMaxChannel, tSample-sampleWindow:tSample+sampleWindow]
            tmpWF_filtered = bp_filter(tmpWF)
            tmpWFOffset = np.median(tmpWF_filtered[:medianEndBin])
            rawAmplitudes.append(np.min(tmpWF_filtered) - tmpWFOffset)
            # allWaveForms.append(tmpWF)
        for tSample in spikeSamplesAligned:
            tmpWF = originalData[templateMaxChannel, tSample-sampleWindow:tSample+sampleWindow]
            tmpWF_filtered = bp_filter(tmpWF)
            tmpWFOffset = np.median(tmpWF_filtered[:medianEndBin])
            rawAmplitudesAligned.append(np.min(tmpWF_filtered) - tmpWFOffset)
        # allWaveForms = np.array(allWaveForms)
        # medianWaveForm = np.median(allWaveForms, axis=0)
        # medianWaveForm -= np.median(medianWaveForm[:medianEndBin])
        rawAmplitudes = -1.0*np.array(rawAmplitudes)
        rawAmplitudesAligned = -1.0*np.array(rawAmplitudesAligned)

        # magicNumber = 1/0.195 # multiply by this number to convert from (micro-)volts to Intan binary format
        # # multiply by this number to convert template amplitude to (micro-)volts (approx.)
        # # peak-to-peak amp: ~1 mV; template peak-to-peak: 3339
        # # average template amp.: 94
        # # => factor ~ 1000 muV / 3339 / 94
        # magicAmplitude = 0.003186*0.67 # incl. heuristic correction 2/3
        # scaledClusterTemplate = magicNumber*magicAmplitude*clusterTemplate

        # meanAmp = np.mean(clusterTemplateAmplitudes)
        # maxTemplateAmp = abs(np.min(scaledClusterTemplate[templateMaxChannel,:]))
        if not madPerChannel.has_key(templateMaxChannel):
            print '\tComputing MAD for channel %d...' % templateMaxChannel
            print '\tFiltering...'
            filtered_channel_trace = bp_filter(originalData[templateMaxChannel, :])
            medianMaxChannel = np.median(filtered_channel_trace)
            mad = np.median(np.abs(filtered_channel_trace - medianMaxChannel))
            madPerChannel[templateMaxChannel] = mad
            alignedMaxSample = int(experimentInfo['WC']['PulsePeriodsOnProbe'][-1][1]*experimentInfo['SiProbe']['SamplingRate'] + 0.5)
            medianMaxChannelAligned = np.median(originalData[templateMaxChannel, :alignedMaxSample])
            madAligned = np.median(np.abs(originalData[templateMaxChannel, :alignedMaxSample] - medianMaxChannelAligned))
            madPerChannelAligned[templateMaxChannel] = madAligned
        else:
            mad = madPerChannel[templateMaxChannel]
            madAligned = madPerChannelAligned[templateMaxChannel]
        # SNREstimate = meanAmp*maxTemplateAmp/(mad*1.4862)
        SNREstimate = np.mean(rawAmplitudes)/(mad*1.4862)
        SNREstimateAligned = np.mean(rawAmplitudesAligned)/(madAligned*1.4862)
        clusterSNR[clusterID] = SNREstimate
        clusterSNRAligned[clusterID] = SNREstimateAligned
        print 'Cluster %d SNR = %.1f -- SNR aligned = %.1f' % (clusterID, SNREstimate, SNREstimateAligned)

    outName = os.path.join(experimentInfo['SiProbe']['ClusterBasePath'], 'clusterSNR.csv')
    with open(outName, 'w') as outFile:
        header = 'Cluster ID\tSNR\tSNR aligned\n'
        outFile.write(header)
        clusterIDs = clusters.keys()
        clusterIDs.sort()
        for clusterID in clusterIDs:
            line = str(clusterID)
            line += '\t'
            line += str(clusterSNR[clusterID])
            line += '\t'
            line += str(clusterSNRAligned[clusterID])
            line += '\n'
            outFile.write(line)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        experimentInfoName = sys.argv[1]
        compute_quality_metrics(experimentInfoName)
    else:
        print 'Wrong number of arguments'
