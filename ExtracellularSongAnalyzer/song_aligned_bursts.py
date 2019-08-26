import os
import ast
import sys
import cPickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import ClusterProcessing as cp
import utilities as utils


class Syllable(object):
    def __init__(self, label, motifs, onsets, offsets):
        self.label = label # string
        self.motifs = motifs # motif IDs
        self.onsets = onsets # onset times within motifs (sorted by motif IDs)
        self.offsets = offsets # offset times within motifs (sorted by motif IDs)


def _load_syllables_from_egui(fname):
    """
    :param fname: filename for eGUI MATLAB data structure
    :return: dict with keys syllable ID and elements of class Syllable
    """
    egui_data_ = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
    egui_data = egui_data_['dbase']

    # get motif order from file names
    # format: motif_N.wav -> extract N
    motif_ids = []
    for i in range(len(egui_data.SoundFiles)):
        motif_name = egui_data.SoundFiles[i].name
        split_name = motif_name.split('_')
        motif_id = int(split_name[1][:-4])
        motif_ids.append(motif_id)

    # get all syllable labels
    max_n_segments = 0
    motif_for_labels = None
    for i in range(len(egui_data.SegmentIsSelected)):
        if np.sum(egui_data.SegmentIsSelected[i]) > max_n_segments:
            max_n_segments = np.sum(egui_data.SegmentIsSelected[i])
            motif_for_labels = i
    syllable_labels = egui_data.SegmentTitles[motif_for_labels]

    egui_syllables = {}
    for label in syllable_labels:
        tmp_motif_ids = []
        tmp_onsets = []
        tmp_offsets = []
        for i in range(len(motif_ids)):
            motif_syllables = egui_data.SegmentTitles[i]
            good_syllables = egui_data.SegmentIsSelected[i]
            if label in motif_syllables:
                syllable_index = np.where(motif_syllables == label)[0]
                if good_syllables[syllable_index]:
                    tmp_motif_ids.append(motif_ids[i])
                    tmp_onsets.append(egui_data.SegmentTimes[i][syllable_index, 0] * 1.0 / egui_data.Fs)
                    tmp_offsets.append(egui_data.SegmentTimes[i][syllable_index, 1] * 1.0 / egui_data.Fs)

        tmp_motif_ids = np.array(tmp_motif_ids)
        tmp_onsets = np.array(tmp_onsets)
        tmp_offsets = np.array(tmp_offsets)
        motif_order = np.argsort(tmp_motif_ids)
        new_syllable = Syllable(label, tmp_motif_ids[motif_order], tmp_onsets[motif_order], tmp_offsets[motif_order])
        egui_syllables[label] = new_syllable

    return egui_syllables


def _save_for_matlab(experiment_info, burst_onset_times, syllable_onset_times, syllable_offset_times):
    print 'Saving in matlab format...'
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    summary_suffix = 'motif_syllable_aligned_burst_onset_times.mat'
    summary_fname = os.path.join(cluster_folder, 'burst_identity', summary_suffix)
    spacetime = {} #Vigi format

    # syllable on-/offset times
    motif = []
    for i in range(len(syllable_onset_times)):
        motif.append(syllable_onset_times[i])
        motif.append(syllable_offset_times[i])
    spacetime['Motif'] = motif

    # burst onset times in motif
    bt = np.full((len(burst_onset_times), 9), np.nan)
    bt[:, 0] = burst_onset_times[:]
    spacetime['bT'] = bt

    # syllable ID during which each burst occurs; NaN if during gap
    # burst onset times relative to syllable onset; NaN if during gap
    syllable_ids = np.full((len(burst_onset_times), 9), np.nan)
    syllable_times = np.full((len(burst_onset_times), 9), np.nan)
    for i, t in enumerate(burst_onset_times):
        syllable = np.nan
        for j in range(len(syllable_onset_times)):
            if syllable_onset_times[j] <= t <= syllable_offset_times[j]:
                syllable_ids[i, 0] = j + 1
                syllable_times[i, 0] = t - syllable_onset_times[j]
                break
    spacetime['sylID'] = syllable_ids
    spacetime['syl_T'] = syllable_times

    scipy.io.savemat(summary_fname, {'SpaceTime': spacetime})


def syllable_aligned_bursts(experiment_info_name):
    """
    burst raster plot aligned to syllables
    :param experiment_info_name: parameter file name
    :return: nothing
    """
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get full audio
    audio_name = os.path.join(experiment_info['Motifs']['DataBasePath'], experiment_info['Motifs']['AudioFilename'])
    audio_fs, audio_data = cp.reader.read_audiofile(audio_name)
    # get syllables from eGUI
    egui_syllables = _load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])

    # motif object with attributes start, stop, and more not relevant here
    motif_audio_traces = []
    n_motifs = len(motif_finder_data.start)
    for i in range(n_motifs):
        motif_start = motif_finder_data.start[i]
        motif_stop = motif_finder_data.stop[i]
        motif_start_sample = int(motif_start*audio_fs)
        motif_stop_sample = int(motif_stop*audio_fs)
        motif_audio = audio_data[motif_start_sample:motif_stop_sample]
        motif_audio_traces.append(motif_audio)

    # check syllable alignment
    mean_duration = {}
    for syllable in egui_syllables:
        syllable_duration = 0.0
        for i in range(len(egui_syllables[syllable].onsets)):
            dt = egui_syllables[syllable].offsets[i] - egui_syllables[syllable].onsets[i]
            syllable_duration += dt
        syllable_duration /= len(egui_syllables[syllable].onsets)
        mean_duration[syllable] = syllable_duration
    # buffer = int(0.03 * audio_fs) # 30 ms in samples
    # for i, syllable in enumerate(egui_syllables):
    #     fig = plt.figure(i)
    #     ax = plt.subplot(1, 1, 1)
    #     for j, motif_index in enumerate(egui_syllables[syllable].motifs):
    #     # for j in range(1):
    #         onset_index = int(egui_syllables[syllable].onsets[j] * audio_fs)
    #         offset_index = int(egui_syllables[syllable].offsets[j] * audio_fs)
    #         snippet_ = motif_audio_traces[motif_index][onset_index:offset_index]
    #         snippet = utils.normalize_trace(snippet_, -0.95, 0.95)
    #         t_snippet = np.linspace(0.0, mean_duration[syllable], len(snippet))
    #         ax.plot(t_snippet, snippet + 2*j, 'k', linewidth=0.5)
    #         title_str = 'Syllable %s' % syllable
    #         ax.set_title(title_str)
    #         # if j == 0:
    #         #     dummy = 1
    # plt.show()

    # # get clusters
    # data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    # clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)
    # get bursts, burst spike times and spontaneous spike times
    clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929]
    burst_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0]
    assert len(clusters_of_interest) == len(burst_ids)
    # cluster_bursts = {}
    for i, cluster_id in enumerate(clusters_of_interest):
        summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
        summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
        with open(summary_burst_fname, 'rb') as summary_burst_file:
            # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
            cluster_bursts = cPickle.load(summary_burst_file)
        # select burst ID
        burst = cluster_bursts[burst_ids[i]]
        # determine nearest syllable (i.e., nearest on/offset to burst onset time)
        syllable_distance = 1e6
        nearest_syllable = None
        for j, label in enumerate(egui_syllables):
            ref_motif_id = None
            motif_id_cnt = 0
            # determine first motif where syllable is segmented and make sure there are burst spikes
            while ref_motif_id is None and motif_id_cnt < len(burst):
                tmp_motif_id = egui_syllables[label].motifs[motif_id_cnt]
                if len(burst[tmp_motif_id][0]):
                    ref_motif_id = tmp_motif_id
                    break
                motif_id_cnt += 1
            tmp_offset = motif_finder_data.start[ref_motif_id]
            tmp_syllable_onset = egui_syllables[label].onsets[motif_id_cnt]
            tmp_syllable_distance = abs(burst[ref_motif_id][0][0] - tmp_offset - tmp_syllable_onset)
            if tmp_syllable_distance < syllable_distance:
                syllable_distance = tmp_syllable_distance
                nearest_syllable = label
        # select all motifs where syllable is segmented
        # and align burst times to aligned syllables
        aligned_burst_times = []
        for j, motif_id in enumerate(egui_syllables[nearest_syllable].motifs):
            trial_duration = egui_syllables[nearest_syllable].offsets[j] - egui_syllables[nearest_syllable].onsets[j]
            trial_scale = mean_duration[nearest_syllable] / trial_duration
            trial_onset = egui_syllables[nearest_syllable].onsets[j]
            burst_times_syllable = (burst[motif_id][0] - motif_finder_data.start[motif_id] - trial_onset) * trial_scale
            aligned_burst_times.append(burst_times_syllable)
        fig = plt.figure(cluster_id)
        ax = plt.subplot(1, 1, 1)
        ax.eventplot(aligned_burst_times, colors='k', linewidths=0.5)
        tmp_ylim = ax.get_ylim()
        ax.plot([0, 0], tmp_ylim, 'r')
        ax.plot([mean_duration[nearest_syllable], mean_duration[nearest_syllable]], tmp_ylim, 'r')
        ax.set_ylim(tmp_ylim)
        title_str = 'Cluster %d aligned to syllable %s' % (cluster_id, nearest_syllable)
        ax.set_title(title_str)
        plt.show()


def motif_aligned_bursts(experiment_info_name):
    '''
    Alignment of all bursts in individual trials.
    Possible because they have been recorded simultaneously.
    '''
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get full audio
    audio_name = os.path.join(experiment_info['Motifs']['DataBasePath'], experiment_info['Motifs']['AudioFilename'])
    audio_fs, audio_data = cp.reader.read_audiofile(audio_name)
    # get syllables from eGUI
    egui_syllables = _load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])

    # motif object with attributes start, stop, and more not relevant here
    motif_audio_traces = []
    n_motifs = len(motif_finder_data.start)
    for i in range(n_motifs):
        motif_start = motif_finder_data.start[i]
        motif_stop = motif_finder_data.stop[i]
        motif_start_sample = int(motif_start * audio_fs)
        motif_stop_sample = int(motif_stop * audio_fs)
        motif_audio = audio_data[motif_start_sample:motif_stop_sample]
        motif_audio_traces.append(motif_audio)

    # check syllable alignment
    mean_duration = {}
    for syllable in egui_syllables:
        syllable_duration = 0.0
        for i in range(len(egui_syllables[syllable].onsets)):
            dt = egui_syllables[syllable].offsets[i] - egui_syllables[syllable].onsets[i]
            syllable_duration += dt
        syllable_duration /= len(egui_syllables[syllable].onsets)
        mean_duration[syllable] = syllable_duration
    # buffer = int(0.03 * audio_fs) # 30 ms in samples
    # for i, syllable in enumerate(egui_syllables):
    #     fig = plt.figure(i)
    #     ax = plt.subplot(1, 1, 1)
    #     for j, motif_index in enumerate(egui_syllables[syllable].motifs):
    #     # for j in range(1):
    #         onset_index = int(egui_syllables[syllable].onsets[j] * audio_fs)
    #         offset_index = int(egui_syllables[syllable].offsets[j] * audio_fs)
    #         snippet_ = motif_audio_traces[motif_index][onset_index:offset_index]
    #         snippet = utils.normalize_trace(snippet_, -0.95, 0.95)
    #         t_snippet = np.linspace(0.0, mean_duration[syllable], len(snippet))
    #         ax.plot(t_snippet, snippet + 2*j, 'k', linewidth=0.5)
    #         title_str = 'Syllable %s' % syllable
    #         ax.set_title(title_str)
    #         # if j == 0:
    #         #     dummy = 1
    # plt.show()

    # # get clusters
    # data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    # clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)
    # get bursts, burst spike times and spontaneous spike times
    # C21
    clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929]
    burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0]
    assert len(clusters_of_interest) == len(burst_ids)
    # load all bursts
    cluster_bursts = {}
    for i, cluster_id in enumerate(clusters_of_interest):
        summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
        summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
        with open(summary_burst_fname, 'rb') as summary_burst_file:
            # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
            tmp_bursts = cPickle.load(summary_burst_file)
        # select burst ID
        cluster_bursts[cluster_id] = tmp_bursts[burst_ids[i]]

    # plot burst onset times in each motif
    # for i in range(n_motifs):
    for i in [n_motifs - 1]:
        fig = plt.figure(i)
        motif_start = motif_finder_data.start[i]
        motif_stop = motif_finder_data.stop[i]
        motif_spikes = []
        motif_burst_onsets = []
        for j, cluster_id in enumerate(clusters_of_interest):
            burst = cluster_bursts[cluster_id]
            burst_times_motif = burst[i][0] - motif_start
            if len(burst_times_motif):
                motif_burst_onsets.append(burst_times_motif[0])
                motif_spikes.append(burst_times_motif)
            # else:
            #     motif_burst_onsets.append([])
        sorted_indices = np.argsort(motif_burst_onsets)
        motif_spikes_sorted = []
        motif_burst_onsets_sorted = []
        for index in sorted_indices:
            motif_spikes_sorted.append(motif_spikes[index])
            motif_burst_onsets_sorted.append(motif_burst_onsets[index])
        ax = plt.subplot(1, 1, 1)
        ax.eventplot(motif_spikes_sorted, colors='k', linewidths=0.5)
        # ax.eventplot(motif_burst_onsets_sorted, colors='r', linewidths=1.0)
        t_audio = np.linspace(0.0, motif_stop - motif_start, len(motif_audio_traces[i]))
        plot_audio = utils.normalize_audio_trace(motif_audio_traces[i])
        ax.plot(t_audio, plot_audio + len(clusters_of_interest) + 2, 'k', linewidth=0.5)
        syllable_onsets = []
        syllable_offsets = []
        for syllable in egui_syllables:
            if i in egui_syllables[syllable].motifs:
                motif_index = np.where(egui_syllables[syllable].motifs == i)[0]
                onset, offset = np.squeeze(egui_syllables[syllable].onsets[motif_index]), \
                                np.squeeze(egui_syllables[syllable].offsets[motif_index])
                syllable_onsets.append(onset)
                syllable_offsets.append(offset)
                tmp_ylim = ax.get_ylim()
                ax.plot([onset, onset], tmp_ylim, 'r--', linewidth=0.5)
                ax.plot([offset, offset], tmp_ylim, 'r--', linewidth=0.5)
                ax.set_ylim(tmp_ylim)
        title_str = 'Bursts in motif %d' % (i)
        ax.set_title(title_str)
        plt.show()

        _save_for_matlab(experiment_info, motif_burst_onsets_sorted, syllable_onsets, syllable_offsets)


# def manual_burst_spike(experiment_info_name):
#     # load stuff
#     with open(experiment_info_name, 'r') as data_file:
#         experiment_info = ast.literal_eval(data_file.read())
#     # get motif times
#     motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
#                                                            experiment_info['Motifs']['MotifFilename']))
#     # get motif template
#     audio_fs, audio_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
#     plot_audio = utils.normalize_trace(audio_data, -1.0, 1.0)
#     # get clusters
#     data_folder = experiment_info['SiProbe']['DataBasePath']
#     cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
#     fs = experiment_info['SiProbe']['SamplingRate']
#     clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
#     # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)
#     # burst_cluster_ids, burst_cluster_nr = np.loadtxt(os.path.join(cluster_folder, 'cluster_burst.tsv'), skiprows=1,
#     #                                                  unpack=True)
#
#     # channel_shank_map = np.load(os.path.join(cluster_folder, 'channel_shank_map.npy'))
#     # channel_positions = np.load(os.path.join(cluster_folder, 'channel_positions.npy'))
#
#     clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929]
#     burst_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0]
#     assert len(clusters_of_interest) == len(burst_ids)
#
#     # got through all clusters
#     # for cluster_id in clusters_of_interest:
#     # for cluster_id in [670, 786, 883, 918, 1073, 941, 776, 777, 841, 842, 938, 1093, 387, 807, 897, 976, 979, 980]:
#     # for cluster_id in [883, 918, 1073, 941, 776, 777, 841, 842, 938, 1093, 387, 807, 897, 976, 979, 980]:
#     # for cluster_id in [841, 842, 1093, 387, 807, 897, 976, 979, 980]:
#     # C23 shanks 1/2
#     # for cluster_id in [670, 786, 883, 918, 983, 1073, 941, 776, 777, 841, 842, 938, 1092, 1093, 387, 807, 897, 976, 979, 980]:
#     # C23 shanks 3/4
#     # for cluster_id in [1116, 1129, 1154, 1158, 1166, 1169, 1175, 1205, 1220, 1236, 1247, 1257, 1267, 1268, 1283, 1288,
#     #                    1298, 1302, 1303, 1309, 1314, 1330, 1340, 1346, 1367, 1374, 1376]:
#     # for cluster_id in [1205]:
#     # for cluster_id in burst_cluster_ids:
#     for i, cluster_id in enumerate(clusters_of_interest):
#         summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
#         summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
#         with open(summary_burst_fname, 'rb') as summary_burst_file:
#             # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
#             cluster_bursts = cPickle.load(summary_burst_file)
#         # select burst ID
#         burst = cluster_bursts[burst_ids[i]]
#         cluster = clusters[cluster_id]
#
#         recording_file = cp.reader.load_recording(os.path.join(data_folder, experiment_info['SiProbe']['AmplifierName']),
#                                                   experiment_info['SiProbe']['Channels'])
#         b, a = _set_up_filter(300.0, 0.49*fs, fs)
#         max_channel = cluster.maxChannel
#         burst_window = 15.0 # +- 5 ms around spike time
#         burst_window_index = int(burst_window*1e-3*fs)
#         burst_intervals_manual = dict(zip(range(len(burst_onsets)), [[] for i in range(len(burst_onsets))]))
#         for burst_id in burst_indices:
#             trial_cnt = 0
#             spike_cnt = 0
#             while spike_cnt < min(10, len(burst_indices[burst_id])) and trial_cnt <len(burst_indices[burst_id]):
#                 if not len(burst_indices[burst_id][trial_cnt]):
#                     trial_cnt += 1
#                     continue
#                 indices = burst_indices[burst_id][trial_cnt]
#                 spike_time_indices = np.where(motif_spikes > 0)[0][indices]
#                 burst_spike_times = spike_times[spike_time_indices].magnitude
#                 t_spike = burst_spike_times[0]
#                 t_spike_index = int(t_spike*fs)
#                 start_index = t_spike_index - burst_window_index
#                 stop_index = t_spike_index + burst_window_index
#                 snippet = recording_file[max_channel, start_index:stop_index]
#                 filtered_snippet = signal.filtfilt(b, a, snippet)
#                 fig = plt.figure()
#                 ax = plt.subplot(1, 1, 1)
#                 ax.plot(np.array(range(len(filtered_snippet)))*1.0e3/fs, filtered_snippet, 'k', linewidth=0.5)
#                 for t_spike_tmp in burst_spike_times:
#                     t_spike_tmp_shift = t_spike_tmp*1.0e3 - (t_spike*1.0e3 - burst_window)
#                     # t_spike_index_tmp = int(t_spike_tmp * fs) - (t_spike_index - burst_window_index)
#                     y_min, y_max = ax.get_ylim()
#                     ax.plot((t_spike_tmp_shift, t_spike_tmp_shift), (y_min, y_max), 'r--', linewidth=0.5)
#                     ax.set_ylim((y_min, y_max))
#                 title_str = 'Cluster %d; burst %d; trial %d (spike %d)' % (cluster_id, burst_id, trial_cnt, spike_cnt)
#                 ax.set_title(title_str)
#                 spike_times_picked = []
#                 sp = utils.SpikePicker(ax, spike_times_picked)
#                 sp.connect()
#                 plt.show()
#                 plt.close(fig)
#                 trial_cnt += 1
#                 if len(spike_times_picked) > 1:
#                     spike_times_picked.sort()
#                     intervals = np.diff(spike_times_picked)
#                     burst_intervals_manual[burst_id].extend(intervals)
#                     spike_cnt += 1
#
#         summary_suffix = 'burst_firing_rates_cluster_%d.csv' % cluster_id
#         summary_fname = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'burst_identity', summary_suffix)
#         with open(summary_fname, 'w') as summary_file:
#             for burst_id in burst_intervals_manual:
#                 all_intervals = burst_intervals_manual[burst_id]
#                 mean_fr = 1.0e3/np.mean(all_intervals)
#                 line = str(burst_id)
#                 line += '\t'
#                 line += str(mean_fr)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        info_name = sys.argv[1]
        # syllable_aligned_bursts(info_name)
        motif_aligned_bursts(info_name)
