import numpy as np
import scipy.io, scipy.io.wavfile, scipy.signal
import os
import neo
from cluster import Cluster


intan_constant = 0.195


def _set_up_filter(highpass, lowpass, fs):
    filter_order = 3
    return scipy.signal.butter(filter_order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def _unwhiten(wmi, x, channel_ids=None):
    mat = wmi
    if channel_ids is not None:
        mat = mat[np.ix_(channel_ids, channel_ids)]
        assert mat.shape == (len(channel_ids),) * 2
    assert x.shape[1] == mat.shape[0]
    return np.dot(np.ascontiguousarray(x),
                  np.ascontiguousarray(mat))


def read_KS_clusters(dataFolder, version, keep_group, samplingRate):
    ''' folder: location of your data
        version: 'dev' or 'release'. This will tell the program where to look
        for the data
        keep: list of phy cluster types to be loaded. Options:
        'good', 'mua', 'noise', 'unsorted' ('unsorted' NOT in dev version)
        fs: sampling rate (in Hz)
        :return: dict of Cluster objects, where keys are phy cluster IDs
        '''

    if version == 'release':
        spike_clusters = np.load(os.path.join(dataFolder, 'batches', 'spike_clusters.npy')).flatten()
        spike_templates = np.load(os.path.join(dataFolder, 'batches', 'spike_templates.npy')).flatten()
        spike_times_numpy = np.load(os.path.join(dataFolder, 'batches', 'spike_times.npy')).flatten()
        amplitudes_numpy = np.load(os.path.join(dataFolder, 'batches', 'amplitudes.npy')).flatten()
        templates_numpy = np.load(os.path.join(dataFolder, 'batches', 'templates.npy'))
        channel_map_numpy = np.load(os.path.join(dataFolder, 'batches', 'channel_map.npy')).flatten()
        channel_coordinates_numpy = np.load(os.path.join(dataFolder, 'batches', 'channel_positions.npy'))
        channel_shank_map_numpy = np.load(os.path.join(dataFolder, 'batches', 'channel_shank_map.npy')).flatten()
        whitening_matrix_inv = np.load(os.path.join(dataFolder, 'batches', 'whitening_mat_inv.npy'))
        # amplitudes = np.load(os.path.join(dataFolder, 'batches', 'amplitudes.npy')).flatten()
        cluster_group_fname = os.path.join(dataFolder, 'batches', 'cluster_groups.csv')
    elif version == 'dev':
        spike_clusters = np.load(os.path.join(dataFolder, 'spike_clusters.npy')).flatten()
        spike_templates = np.load(os.path.join(dataFolder, 'spike_templates.npy')).flatten()
        spike_times_numpy = np.load(os.path.join(dataFolder, 'spike_times.npy')).flatten()
        amplitudes_numpy = np.load(os.path.join(dataFolder, 'amplitudes.npy')).flatten()
        templates_numpy = np.load(os.path.join(dataFolder, 'templates.npy'))
        channel_map_numpy = np.load(os.path.join(dataFolder, 'channel_map.npy')).flatten()
        channel_coordinates_numpy = np.load(os.path.join(dataFolder, 'channel_positions.npy'))
        channel_shank_map_numpy = np.load(os.path.join(dataFolder, 'channel_shank_map.npy')).flatten()
        whitening_matrix_inv = np.load(os.path.join(dataFolder, 'whitening_mat_inv.npy'))
        # amplitudes = np.load(os.path.join(dataFolder, 'amplitudes.npy')).flatten()
        cluster_group_fname = os.path.join(dataFolder, 'cluster_group.tsv')
    else:
        errstr = 'Cluster loading for version %s not implemented' % version
        raise NotImplementedError(errstr)

    xcoords = channel_coordinates_numpy[:, 0]
    ycoords = channel_coordinates_numpy[:, 1]
    channelMap = channel_map_numpy

    # unwhiten templates
    templates_unwhitened = templates_numpy[:, :, :]
    for i in range(templates_unwhitened.shape[0]):
        tmp = templates_unwhitened[i, :, :]
        tmp_unwhitened = _unwhiten(whitening_matrix_inv, tmp)
        templates_unwhitened[i, :, :] = tmp_unwhitened

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
        selection += clusterGroup == keep_group[i]
    keptClusters = clusterIDs[np.where(selection)]
    keptClusterGroups = clusterGroup[np.where(selection)]
    clusters = {}
    for i in range(len(keptClusters)):
        clusterID = keptClusters[i]
        spikeTimeIndices = np.where(spike_clusters == clusterID)
        spikeSamples = spike_times_numpy
        # remove erroneous identically duplicate spike times
        clusterSpikeSamples, uniqueIndices = np.unique(spikeSamples[spikeTimeIndices], return_index=True)
        spikeTimes = clusterSpikeSamples/samplingRate
        spikeTrain = neo.core.SpikeTrain(spikeTimes, units='sec', t_stop=max(spikeTimes), t_start=min(0, min(spikeTimes)))
        templateAmplitudes = amplitudes_numpy[spikeTimeIndices[0][uniqueIndices]]
        # find channel location of max. waveform
        # look up original clusters comprising this unit
        originalClusters = np.array(np.unique(spike_templates[spikeTimeIndices]), dtype='int')
        # compute mean template of these clusters
        meanTemplate = np.mean(templates_unwhitened[originalClusters, :, :], axis=0)
        # average power across time and find channel index with max power
        amplitude = meanTemplate.max(axis=0) - meanTemplate.min(axis=0)
        KS_channel = np.argmax(amplitude)
        maxChannel = channelMap[KS_channel]
        maxWF = meanTemplate[:, KS_channel]
        shank = channel_shank_map_numpy[KS_channel]
        coordinates = xcoords[KS_channel], ycoords[KS_channel]
        firingRate = len(spikeTrain.times)/(spikeTrain.t_stop - spikeTrain.t_start)
        thisCluster = Cluster(clusterID, keptClusterGroups[i], spikeTrain, maxWF, shank, maxChannel, coordinates, firingRate)
        thisCluster.template = meanTemplate
        thisCluster.templateAmplitudes = templateAmplitudes
        clusters[clusterID] = thisCluster

    return clusters

def read_KS_clusters_unsorted(dataFolder, version, samplingRate):
    ''' folder: location of your data
        version: 'dev' or 'release'. This will tell the program where to look
        for the data
        fs: sampling rate (in Hz)
        :return: dict of Cluster objects, where keys are phy cluster IDs
        '''
    if version == 'release':
        spike_clusters = np.load(os.path.join(dataFolder, 'batches', 'spike_clusters.npy')).flatten()
        spike_templates = np.load(os.path.join(dataFolder, 'batches', 'spike_templates.npy')).flatten()
        spike_times_numpy = np.load(os.path.join(dataFolder, 'batches', 'spike_times.npy')).flatten()
        amplitudes_numpy = np.load(os.path.join(dataFolder, 'batches', 'amplitudes.npy')).flatten()
        templates_numpy = np.load(os.path.join(dataFolder, 'batches', 'templates.npy'))
        channel_map_numpy = np.load(os.path.join(dataFolder, 'batches', 'channel_map.npy')).flatten()
        channel_coordinates_numpy = np.load(os.path.join(dataFolder, 'batches', 'channel_positions.npy'))
        channel_shank_map_numpy = np.load(os.path.join(dataFolder, 'batches', 'channel_shank_map.npy')).flatten()
        whitening_matrix_inv = np.load(os.path.join(dataFolder, 'batches', 'whitening_mat_inv.npy'))
        # amplitudes = np.load(os.path.join(dataFolder, 'batches', 'amplitudes.npy')).flatten()
    elif version == 'dev':
        spike_clusters = np.load(os.path.join(dataFolder, 'spike_clusters.npy')).flatten()
        spike_templates = np.load(os.path.join(dataFolder, 'spike_templates.npy')).flatten()
        spike_times_numpy = np.load(os.path.join(dataFolder, 'spike_times.npy')).flatten()
        amplitudes_numpy = np.load(os.path.join(dataFolder, 'amplitudes.npy')).flatten()
        templates_numpy = np.load(os.path.join(dataFolder, 'templates.npy'))
        channel_map_numpy = np.load(os.path.join(dataFolder, 'channel_map.npy')).flatten()
        channel_coordinates_numpy = np.load(os.path.join(dataFolder, 'channel_positions.npy'))
        channel_shank_map_numpy = np.load(os.path.join(dataFolder, 'channel_shank_map.npy')).flatten()
        whitening_matrix_inv = np.load(os.path.join(dataFolder, 'whitening_mat_inv.npy'))
        # amplitudes = np.load(os.path.join(dataFolder, 'amplitudes.npy')).flatten()
    else:
        errstr = 'Cluster loading for version %s not implemented' % version
        raise NotImplementedError(errstr)

    xcoords = channel_coordinates_numpy[:, 0]
    ycoords = channel_coordinates_numpy[:, 1]
    channelMap = channel_map_numpy

    # unwhiten templates
    templates_unwhitened = templates_numpy[:, :, :]
    for i in range(templates_unwhitened.shape[0]):
        tmp = templates_unwhitened[i, :, :]
        tmp_unwhitened = _unwhiten(whitening_matrix_inv, tmp)
        templates_unwhitened[i, :, :] = tmp_unwhitened

    clusterIDs = np.unique(spike_clusters)
    clusters = {}
    for clusterID in clusterIDs:
        spikeTimeIndices = np.where(spike_clusters == clusterID)
        spikeSamples = spike_times_numpy
        # remove erroneous identically duplicate spike times
        clusterSpikeSamples, uniqueIndices = np.unique(spikeSamples[spikeTimeIndices], return_index=True)
        spikeTimes = clusterSpikeSamples/samplingRate
        spikeTrain = neo.core.SpikeTrain(spikeTimes, units='sec', t_stop=max(spikeTimes), t_start=min(0, min(spikeTimes)))
        templateAmplitudes = amplitudes_numpy[spikeTimeIndices[0][uniqueIndices]]
        # find channel location of max. waveform
        # look up original clusters comprising this unit
        originalClusters = np.array(np.unique(spike_templates[spikeTimeIndices]), dtype='int')
        # compute mean template of these clusters
        meanTemplate = np.mean(templates_unwhitened[originalClusters, :, :], axis=0)
        # average power across time and find channel index with max power
        amplitude = meanTemplate.max(axis=0) - meanTemplate.min(axis=0)
        KS_channel = np.argmax(amplitude)
        maxChannel = channelMap[KS_channel]
        maxWF = meanTemplate[:, KS_channel]
        shank = channel_shank_map_numpy[KS_channel]
        coordinates = xcoords[KS_channel], ycoords[KS_channel]
        firingRate = len(spikeTrain.times)/(spikeTrain.t_stop - spikeTrain.t_start)
        thisCluster = Cluster(clusterID, 'unsorted', spikeTrain, maxWF, shank, maxChannel, coordinates, firingRate)
        thisCluster.template = meanTemplate
        thisCluster.templateAmplitudes = templateAmplitudes
        clusters[clusterID] = thisCluster

    return clusters


def read_all_clusters_except_noise(data_folder, version, fs):
    clusters = read_KS_clusters_unsorted(data_folder, '', version, fs)

    if version == 'release':
        cluster_group_fname = os.path.join(data_folder, 'batches', 'cluster_groups.csv')
    elif version == 'dev':
        cluster_group_fname = os.path.join(data_folder, 'cluster_group.tsv')
    else:
        errstr = 'Cluster loading for version %s not implemented' % version
        raise NotImplementedError(errstr)

    cluster_groups = {}
    with open(cluster_group_fname, 'r') as f:
        line_cnt = 0
        for line in f:
            line_cnt += 1
            if line_cnt == 1:
                continue
            if version == 'release':
                delim = ' '
            elif version == 'dev':
                delim = '\t'
            split_line = line.strip().split(delim)
            cluster_id = int(split_line[0])
            cluster_groups[cluster_id] = split_line[1]

    for cluster_id in cluster_groups:
        if cluster_id not in clusters:
            continue
        if cluster_groups[cluster_id] == 'noise':
            print 'removing noise cluster %d' % cluster_id
            clusters.pop(cluster_id)

    return clusters


def read_audiofile(fname):
    fs, data = scipy.io.wavfile.read(fname)
    return fs, data


def read_motifs(motif_fname):
    motif_data = scipy.io.loadmat(motif_fname, struct_as_record=False, squeeze_me=True)
    return motif_data['Motif']


def load_recording(fname, nchannels, dtype=np.dtype('int16')):
    """returns pointer to binary file
    rows: channel numbers
    columns: samples
    """
    file_info = os.stat(fname)
    file_size = file_info.st_size
    nsamples = int(file_size / (dtype.itemsize * nchannels))
    return np.memmap(fname, dtype=dtype.name, mode='r', shape=(int(nchannels), int(nsamples)), order='F')


def load_recording_chunk(recording, channels, chunk_start, chunk_stop, fs):
    """copy binary file from time chunk_start to chunk_stop"""
    sample_start = int(chunk_start * fs)
    sample_stop = int(chunk_stop * fs)

    if sample_stop <= sample_start:
        raise ValueError

    return recording[channels, sample_start:sample_stop]


def _get_waveforms_from_spike_times(spike_times, channels, recording_file, fs):
    # set up high-pass filter
    b, a = _set_up_filter(300.0, 0.49*fs, fs)
    def bp_filter(x, axis=0):
        return scipy.signal.filtfilt(b, a, x, axis=axis)

    # wf_offset: spike time at center when wf_samples = 61.
    # Otherwise KiloSort pads the template with zeros
    # starting from the beginning. So we have to move
    # the center of the extracted waveform accordingly
    wf_samples = 45
    sample_diff = 61 - wf_samples
    wf_offset_begin = (wf_samples - sample_diff) // 2
    wf_offset_end = (wf_samples + sample_diff) // 2

    channels_per_shank = len(channels)
    motif_waveforms = np.zeros((len(spike_times), channels_per_shank, wf_samples))
    for i, spike_time in enumerate(spike_times):
        spike_sample = int(fs*spike_time)
        wf = np.zeros((channels_per_shank, wf_samples))
        # careful with the edge cases - zero-padding
        # uint64 converted silently to float64 when adding an int - cast to int64
        start_index = np.int64(spike_sample) - wf_offset_begin - 1
        # uint64 converted silently to float64 when adding an int - cast to int64
        stop_index = np.int64(spike_sample) + wf_offset_end
        # now copy the appropriately sized snippet from channels on same clusters
        wf[:, :] = intan_constant*recording_file[channels, start_index:stop_index]
        motif_waveforms[i, :, :] = bp_filter(wf, axis=1)

    return motif_waveforms


def load_cluster_waveforms_random_sample(experiment_info, channel_shank_map, clusters, n_spikes=1000):
    """

    :param experiment_info: parameter file
    :param channel_shank_map: channel shank map
    :param clusters: dict of clusters
    :param n_spikes: number of (randomly selected) spikes to return. Default: 1000
    :return: dict with (cluster_id, waveform_array), where waveform_array is a (n_spikes, n_samples) numpy array
    """
    fs = experiment_info['SiProbe']['SamplingRate']
    recording_name = os.path.join(experiment_info['SiProbe']['DataBasePath'], experiment_info['SiProbe']['AmplifierName'])
    recording_file = load_recording(recording_name, experiment_info['SiProbe']['Channels'])

    cluster_waveforms = {}
    for cluster_id in clusters:
        spike_times = clusters[cluster_id].spiketrains[0].magnitude
        n_selected = min(len(spike_times), n_spikes)
        spike_times_ = np.random.permutation(spike_times)
        spike_times_selected = spike_times_[:n_selected]
        # extract all spike waveforms
        shank = clusters[cluster_id].shank
        channels = np.where(channel_shank_map == shank)[0]
        cluster_waveforms[cluster_id] = _get_waveforms_from_spike_times(spike_times_selected, channels, recording_file, fs)

    return cluster_waveforms


def load_cluster_waveforms_from_spike_times(experiment_info, channel_shank_map, cluster, spike_times):
    """
    :param experiment_info: parameter file
    :param channel_shank_map: channel shank map
    :param cluster: cluster
    :param spike_times: spike times
    :return: dict with (cluster_id, waveform_array), where waveform_array is a (n_spikes, n_samples) numpy array
    """
    fs = experiment_info['SiProbe']['SamplingRate']
    recording_name = os.path.join(experiment_info['SiProbe']['DataBasePath'], experiment_info['SiProbe']['AmplifierName'])
    recording_file = load_recording(recording_name, experiment_info['SiProbe']['Channels'])

    # extract all spike waveforms
    shank = cluster.shank
    channels = np.where(channel_shank_map == shank)[0]
    return _get_waveforms_from_spike_times(spike_times, channels, recording_file, fs)
