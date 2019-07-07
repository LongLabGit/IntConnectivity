import os
import ast
import sys
import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
import utilities
import ClusterProcessing as cp

# NEED:
# Loader for stimulus-aligned waveforms
# takes care of stimulus-aligned waveform loading, list of threshold crossing loading, aligning waveform by minimum
# on max channel and extracting waveform on shank
#
# Loader for cluster waveforms
# takes care of loading clusters and motif times, and extracts waveforms of spikes within motifs
#
# Compute function for waveform similarity
#
# Main comparison function
# allows comparison of motif-aligned spikes with stimulus-aligned waveform, comparison of stimulus-aligned waveforms,
# and comparison of motif-aligned spikes
#
# Visualization function
# generates appropriate plots for waveform comparisons (i.e., overlays of waveforms)


intan_constant = 0.195


def _set_up_filter(highpass, lowpass, fs):
    filter_order = 3
    return signal.butter(filter_order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def _load_stimulus_aligned_waveforms(experiment_info):
    """

    :param crossing_info_name: parameter file
    :return: Tuple: Stimulus-aligned waveforms, identified threshold crossings
            each crossing is a tuple t_crossing, channel_crossing, where t_crossing is in ms
    """
    antidromic_waveforms = {}
    antidromic_crossings = {}
    for shank in experiment_info['Antidromic']['ShankWaveforms']:
        tmp_waveform_name = os.path.join(experiment_info['Antidromic']['CrossingBasePath'],
                                         experiment_info['Antidromic']['ShankWaveforms'][shank])
        antidromic_waveforms[shank] = np.load(tmp_waveform_name)
        tmp_crossing_name = os.path.join(experiment_info['Antidromic']['CrossingBasePath'],
                                         experiment_info['Antidromic']['ShankCrossings'][shank])
        antidromic_crossings[shank] = np.load(tmp_crossing_name)

    return antidromic_waveforms, antidromic_crossings


def _load_motif_data(experiment_info):
    return cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                              experiment_info['Motifs']['MotifFilename']))


def _load_cluster_waveforms(selection_str, experiment_info, channel_shank_map):
    """

    :param selection_str: 'good' or 'all'
    :param experiment_info: parameter file
    :return: dict with (cluster_id, waveform_array), where waveform_array is a (n_spikes, n_samples) numpy array
    """
    fs = experiment_info['SiProbe']['SamplingRate']
    if selection_str == 'good':
        data_folder = experiment_info['SiProbe']['ClusterBasePath']
        clusters = cp.reader.read_KS_clusters(data_folder, '', 'dev', 'good', fs)
    elif selection_str == 'all':
        data_folder = experiment_info['SiProbe']['ClusterBasePath']
        clusters = cp.reader.read_KS_clusters_unsorted(data_folder, '', 'dev', fs)
    else:
        e = 'Unknown option "%s"; has to be one of "good" or "all"' % selection_str
        raise ValueError(e)

    # set up high-pass filter
    b, a = _set_up_filter(300.0, 0.49*fs, fs)
    def bp_filter(x, axis=0):
        return signal.filtfilt(b, a, x, axis=axis)

    motif_data = _load_motif_data(experiment_info)
    recording_name = os.path.join(experiment_info['SiProbe']['DataBasePath'], experiment_info['SiProbe']['AmplifierName'])
    recording_file = utilities.load_recording(recording_name, experiment_info['SiProbe']['Channels'])

    # wf_offset: spike time at center when wf_samples = 61.
    # Otherwise KiloSort pads the template with zeros
    # starting from the beginning. So we have to move
    # the center of the extracted waveform accordingly
    wf_samples = 45
    sample_diff = 61 - wf_samples
    wf_offset_begin = (wf_samples - sample_diff) // 2
    wf_offset_end = (wf_samples + sample_diff) // 2
    cluster_waveforms = {}
    for cluster_id in clusters:
        spike_times = clusters[cluster_id].spiketrains[0]
        # for each spike time, determine if within motif
        motif_spike_times = []
        # motif object with attributes start, stop (and more not relevant here)
        for i in range(len(motif_data.start)):
            motif_start = motif_data.start[i]
            motif_stop = motif_data.stop[i]
            # motif_warp = motif_data.warp[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
            if np.sum(selection):
                motif_spike_times.extend(spike_times.magnitude[selection])

        if not len(motif_spike_times):
            continue

        # extract all spike waveforms
        shank = clusters[cluster_id].shank
        channels = np.where(channel_shank_map == shank)[0]
        channels_per_shank = len(channels)
        motif_waveforms = np.zeros((len(motif_spike_times), channels_per_shank, wf_samples))
        for i, spike_time in enumerate(motif_spike_times):
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

        cluster_waveforms[cluster_id] = motif_waveforms

    return cluster_waveforms


def _waveform_overlay_plots():
    # show overlayed waveforms and color them according to similarity score
    # also show motif-aligned raster plot and color them according to similarity score
    pass


def _waveform_similarity():
    pass


def compare_motif_antidromic():
    pass


def compare_antidromic_antidromic():
    pass


def compare_motif_motif():
    pass


if __name__ == '__main__':
    experiment_info_name = sys.argv[1]
    with open(experiment_info_name, 'r') as experiment_info_file:
        experiment_info = ast.literal_eval(experiment_info_file.read())

    channel_shank_map = np.load(os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'channel_shank_map.npy'))
    cluster_waveforms = _load_cluster_waveforms('all', experiment_info, channel_shank_map)
    antidromic_waveforms, antidromic_crossings = _load_stimulus_aligned_waveforms(experiment_info)
