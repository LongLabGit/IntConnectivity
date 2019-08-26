'''
collection of write functions
'''
import numpy as np
import scipy.io.wavfile


def write_wav_file(fname, fs, audio):
    scipy.io.wavfile.write(fname, fs, audio)


def write_neuroscope_spikes(fname, cluster, samplingRate):
    '''
    Writes neuroscope cluster and spike time file
    :param fname: output base name (appendix generated automatically)
    :param cluster: Cluster object containing discrete event times
    :param samplingRate: sampling rate in Hz
    :return: N/A
    '''
    clusterFilename = fname + '.clu.' + str(cluster.clusterID)
    spikeTimeFilename = fname + '.res.' + str(cluster.clusterID)

    with open(clusterFilename, 'w') as nsClusterFile:
        header = '1\n'
        nsClusterFile.write(header)
        content = ''
        clusterIDStr = str(cluster.clusterID)
        for i in range(len(cluster.spiketrains[0])):
            content += clusterIDStr
            content += '\n'
        nsClusterFile.write(content)

    with open(spikeTimeFilename, 'w') as nsSpikeTimeFile:
        content = ''
        for t in cluster.spiketrains[0]:
            spikeSample = int(round(t.magnitude*samplingRate))
            content += str(spikeSample)
            content += '\n'
        nsSpikeTimeFile.write(content)


def write_neuroscope_units_per_shank(fname, clusters, samplingRate):
    '''
    Writes neuroscope cluster and spike time file
    :param fname: output base name (appendix generated automatically)
    :param clusters: dict of Cluster objects containing discrete event times
    :param samplingRate: sampling rate in Hz
    :return: N/A
    '''
    clustersPerShank = {}
    spikeTimesPerShank = {}
    clusterIDsPerSpikePerShank = {}
    for clusterID in clusters:
        shank = clusters[clusterID].shank
        if shank not in clustersPerShank.keys():
            clustersPerShank[shank] = [clusterID]
            spikeTimesPerShank[shank] = list(clusters[clusterID].spiketrains[0].magnitude[:])
            clusterIDsPerSpikePerShank[shank] = [clusterID for i in range(len(clusters[clusterID].spiketrains[0]))]
        else:
            clustersPerShank[shank].append(clusterID)
            spikeTimesPerShank[shank] += list(clusters[clusterID].spiketrains[0].magnitude[:])
            clusterIDsPerSpikePerShank[shank] += [clusterID for i in range(len(clusters[clusterID].spiketrains[0]))]

    for shank in clustersPerShank:
        allSpikeTimes = np.array(spikeTimesPerShank[shank])
        allSpikeClusterIDs = np.array(clusterIDsPerSpikePerShank[shank])
        sortedSpikeIndices = np.argsort(allSpikeTimes)
        clusterFilename = fname + '.clu.' + str(shank)
        spikeTimeFilename = fname + '.res.' + str(shank)

        with open(clusterFilename, 'w') as nsClusterFile:
            nrOfUnits = len(clustersPerShank[shank])
            header = str(nrOfUnits) + '\n'
            nsClusterFile.write(header)
            content = ''
            for sortedIndex in sortedSpikeIndices:
                clusterIDStr = str(allSpikeClusterIDs[sortedIndex])
                content += clusterIDStr
                content += '\n'
            nsClusterFile.write(content)

        with open(spikeTimeFilename, 'w') as nsSpikeTimeFile:
            content = ''
            # for t in allSpikeTimes:
            for sortedIndex in sortedSpikeIndices:
                t = allSpikeTimes[sortedIndex]
                spikeSample = int(round(t*samplingRate))
                content += str(spikeSample)
                content += '\n'
            nsSpikeTimeFile.write(content)

