###################################
# generate "hybrid" ground-truth
# for multiple runs of KiloSort.
# Used for analysis of systematic
# KiloSort parameter explorations
###################################

import os, os.path, sys, ast
import matplotlib.pyplot as plt
import numpy as np
import neo
import ClusterProcessing as clust

ClusteringSrcFolder = 'E:\\User\\project_src\\physiology\\Clustering'

def generate_ground_truth(experimentInfoName, referenceFolder, referenceClusterID, scale):
    '''

    :param experimentInfoName:
    :param referenceFolder:
    :param referenceCluster:
    :param scale:
    :return:
    '''
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())
    # load reference template and reference spike times
    print 'Loading reference cluster %d' % referenceClusterID
    refCluster = load_reference_cluster_amplitudes(referenceFolder, referenceClusterID)
    refSpikeTimes = refCluster.spiketrains[0]
    refSpikeSamples = np.array(refSpikeTimes.magnitude*experimentInfo['SiProbe']['SamplingRate'] + 0.5, dtype='int64')
    # template: NChannels x NTimePoints
    refTemplate = refCluster.template
    # identify mapping of reference template channels to target channels
    # channelShankMap = np.load(os.path.join(experimentInfo['SiProbe']['DataBasePath'], 'channel_shank_map.npy'))
    # targetChannels = np.where(channelShankMap == targetShank)[0]
    # targetChannelOrder = [48, 62, 55, 54, 50, 60, 57, 52]
    # sourceChannelOrder = [56, 40, 58, 46, 63, 44, 61, 59]
    # sourceTargetChannelMapping = dict(zip(sourceChannelOrder, targetChannelOrder))
    # targetKSChannels = range(16, 24)
    # using 979 for reference
    # sourceKSChannels = range(24, 32)
    # using 27 (6/13/13 - 10/30/30) as reference
    sourceKSChannels = range(8)
    # KSSourceChannelOrder = [56, 58, 63, 61, 59, 44, 46, 40]
    KSTargetChannelOrder = [48, 55, 50, 57, 52, 60, 54, 62]
    spatialChannelOrder = [52, 57, 60, 50, 54, 55, 62, 48]
    # from one shank to the next: "identity" for neighboring electrodes
    # KSSourceTargetMap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    # KSToDatTargetChannelMap = dict(zip(range(8), KSTargetChannelOrder))
    sourceTemplate = refTemplate[sourceKSChannels]
    targetTemplate_ = dict(zip(KSTargetChannelOrder, sourceTemplate))
    # scale template amplitude: use fitted amplitudes in reference run
    refTemplateAmplitudes = refCluster.templateAmplitudes

    # load original information
    originalDatName = os.path.join(experimentInfo['SiProbe']['DataBasePath'], 'amplifier_cut.dat')
    originalDatInfo = os.stat(originalDatName)
    originalDatSize = originalDatInfo.st_size
    nChannels = 64
    nNewChannels = 8
    magicNumber = 1/0.195 # multiply by this number to convert from (micro-)volts to Intan binary format
    # multiply by this number to convert template amplitude to (micro-)volts (approx.)
    # peak-to-peak amp: ~1 mV; template peak-to-peak: 3339
    # average template amp.: 94
    # => factor ~ 1000 muV / 3339 / 94
    magicAmplitude = 0.003186*0.67 # incl. heuristic correction 2/3
    templateOffset = 21 # KS template padding before actual template
    templateSamples = 61
    spikeOffset = 40 - templateOffset # sample number in template used for spike time
    targetChannels = targetTemplate_.keys()
    targetTemplate = np.zeros((nNewChannels, len(targetTemplate_[targetChannels[0]][templateOffset:])))
    maxAmpChannelNewTemplate = None
    for i, targetChannel in enumerate(spatialChannelOrder):
        targetTemplate[i,:] = targetTemplate_[targetChannel][templateOffset:]
        if targetChannel == 48:
            maxAmpChannelNewTemplate = i
    targetTemplate *= scale*magicAmplitude*magicNumber
    # store this template (incl. KS offset) on its max channel to find it again in the KS output...
    pastedTemplate = scale*targetTemplate_[targetChannel][:]
    pastedTemplateName = 'referenceTemplate_scale_%.2f.npy' % scale
    np.save(os.path.join(referenceFolder, pastedTemplateName), pastedTemplate)

    # dtype = np.dtype('int16')
    # newDatName = 'C:\\Users\\User\\Desktop\\Continuous_400_cut\\amplifier_cut_scale_%.2f.dat' % scale
    # oldDatSamples = originalDatSize/(dtype.itemsize*nChannels)
    # # write hybrid ground truth signal: load file and only add new spike times
    # originalData = np.memmap(originalDatName, dtype=dtype.name, mode='r', shape=(nChannels, oldDatSamples), order='F')
    #
    # print 'Calculating effective SNR...'
    # maxAmpChannelOriginal = 48
    # meanAmp = np.mean(refTemplateAmplitudes)
    # maxTemplateAmp = abs(np.min(targetTemplate[maxAmpChannelNewTemplate,:]))
    # medianMaxChannel = np.median(originalData[maxAmpChannelOriginal,:])
    # mad = np.median(np.abs(originalData[maxAmpChannelOriginal,:] - medianMaxChannel))
    # SNREstimate = meanAmp*maxTemplateAmp/(mad*1.4862)
    # print 'SNR = %.2f' % SNREstimate
    # SNROutName = 'C:\\Users\\User\\Desktop\\Continuous_400_cut\\SNR_scale_%.2f.txt' % scale
    # with open(SNROutName, 'w') as SNRFile:
    #     line = 'SNR estimate = '
    #     line += str(SNREstimate)
    #     SNRFile.write(line)
    #
    # print 'Copying original data into new dat file %s' % newDatName
    # newData = np.memmap(newDatName, dtype=dtype.name, mode='w+', shape=(nNewChannels, oldDatSamples), order='F')
    # newData[:,:] = originalData[spatialChannelOrder,:]
    # print 'Writing %d scaled spikes into new dat file %s' % (len(refSpikeTimes), newDatName)
    # for n, newSample in enumerate(refSpikeSamples):
    #     startSample = newSample - spikeOffset
    #     endSample = startSample + templateSamples
    #     spikeChunk = range(startSample, endSample)
    #     amp = refTemplateAmplitudes[n]
    #     wf = np.array(amp*targetTemplate, dtype=dtype.name)
    #     newData[:,spikeChunk] += wf
    # newData.flush()
    # del newData
    # del originalData


def load_reference_cluster_amplitudes(path, clusterID):
    '''
    Loads reference cluster
    :param path: path to KS output of reference cluster
    :param clusterID: ID of cluster
    :return: cluster object
    '''
    samplingRate = 20000.0
    clusters = clust.reader.read_KS_clusters(path, ClusteringSrcFolder, 'dev', ('good',), samplingRate)
    clusterGroup = clust.ClusterGroup(clusters)
    # clusterGroup.remove_short_ISIs(clusterID, 2.0e-4)
    return clusterGroup.clusters[clusterID]


if __name__ == '__main__':
    if len(sys.argv) == 5:
        experimentInfoName = sys.argv[1]
        referenceFolder = sys.argv[2]
        referenceCluster = int(sys.argv[3])
        scale = float(sys.argv[4])
        scaleRange = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        for scaleSample in scaleRange:
            generate_ground_truth(experimentInfoName, referenceFolder, referenceCluster, scaleSample)
    else:
        print 'Wrong number of arguments'