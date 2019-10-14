import os
import ast
import sys
import cPickle
import numpy as np
import scipy.io, scipy.signal
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import ClusterProcessing as cp
import utilities as utils

# C21
# clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929]
# burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929, 767]
# burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0, 0]
# C22
# clusters_of_interest = [30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833, 938]
# burst_ids = [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833, 938,
#                         622, 639, 703, 738, 791, 832, 942]
# burst_ids = [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0,
#              0, 0, 0, 0, 1, 0, 0]
# clusters_of_interest = [547]
# burst_ids = [0]
# C23
# clusters_of_interest = [776, 776, 842, 842, 1092, 1092, 1267, 1267, 1302, 1302, 1303, 1303, 1330, 1330, 1376, 1154,
#                         1154, 1166, 1166, 1205, 1205, 1220, 1220, 1268, 1268, 1340, 1340]
# burst_ids = [2, 3, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 2, 2, 3, 0, 2, 1, 3, 2, 3]
# C23 - lower variance of burst pairs (i.e., AB or BA? -> BA it is!)
# clusters_of_interest = [776,  842, 1092, 1154, 1166, 1205, 1220, 1267, 1268, 1302, 1303, 1330, 1340, 1376]
# burst_ids = [3, 2, 1, 2, 2, 3, 2, 1, 3, 1, 1, 1, 3, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [776,  842, 1092, 1154, 1166, 1205, 1220, 1267, 1268, 1302, 1303, 1330, 1340, 1376,
#                         670, 786, 941, 777, 938, 1093, 1330, 1154]
# burst_ids = [3, 2, 1, 2, 2, 3, 2, 1, 3, 1, 1, 1, 3, 0,
#              1, 1, 1, 3, 3, 1, 1, 3]
# C24
# clusters_of_interest = [77, 89, 91, 264, 563, 743, 753, 813, 853]
# burst_ids = [2, 1, 2, 0, 1, 1, 1, 1, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [77, 89, 91, 264, 563, 743, 753, 813, 853,
#                         360, 751, 867, 904]
# burst_ids = [2, 1, 2, 0, 1, 1, 1, 1, 0,
#              0, 0, 0, 0]
# C25
# clusters_of_interest = [110, 130, 159, 189, 521, 240, 289, 310, 346, 366, 412, 432]
# burst_ids = [0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1]
# incl. 22 low-frequency spontaneous
clusters_of_interest = [110, 130, 159, 189, 521, 240, 289, 310, 346, 366, 412, 432]
burst_ids = [0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1]
assert len(clusters_of_interest) == len(burst_ids)


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
    syllable_labels = np.unique(egui_data.SegmentTitles[motif_for_labels])

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
                for index in syllable_index:
                    if good_syllables[index]:
                        tmp_motif_ids.append(motif_ids[i])
                        tmp_onsets.append(egui_data.SegmentTimes[i][index, 0] * 1.0 / egui_data.Fs)
                        tmp_offsets.append(egui_data.SegmentTimes[i][index, 1] * 1.0 / egui_data.Fs)

        tmp_motif_ids = np.array(tmp_motif_ids)
        tmp_onsets = np.array(tmp_onsets)
        tmp_offsets = np.array(tmp_offsets)
        motif_order = np.argsort(tmp_motif_ids)
        new_syllable = Syllable(label, tmp_motif_ids[motif_order], tmp_onsets[motif_order], tmp_offsets[motif_order])
        egui_syllables[label] = new_syllable

    return egui_syllables


def _calculate_reference_syllables(egui_syllables):
    # calculate mean on-/offset per syllable
    reference_syllables = {}
    for label in egui_syllables:
        syllable = egui_syllables[label]
        mean_onset = np.mean(syllable.onsets)
        mean_offset = np.mean(syllable.offsets)
        reference_syllables[label] = mean_onset, mean_offset

    return reference_syllables


def _map_trial_time_to_reference_syllable(t_trial, trial_nr, egui_syllables):
    reference_syllables = _calculate_reference_syllables(egui_syllables)

    for label in egui_syllables:
        syllable = egui_syllables[label]
        if trial_nr not in syllable.motifs:
            continue
        trial_index = np.where(syllable.motifs == trial_nr)[0]
        trial_onset = syllable.onsets[trial_index]
        trial_offset = syllable.offsets[trial_index]
        ref_onset = reference_syllables[label][0]
        ref_offset = reference_syllables[label][1]
        t_ = t_trial - trial_onset
        mapped_t_trial = ref_onset + t_ * (ref_offset - ref_onset) / (trial_offset - trial_onset)
        if ref_onset <= mapped_t_trial <= ref_offset:
            return label, mapped_t_trial
        # DIRTY HACK for C23:
        # for t__ in mapped_t_trial:
        #     if ref_onset <= t__ <= ref_offset:
        #         return label, t__

    return None, None


def _save_individual_syllables_for_matlab(experiment_info, motif_ids, burst_onset_times, syllable_onset_times, syllable_offset_times,
                                          syllable_motifs):
    print 'Saving burst onset times in individual syllables in matlab format...'
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    summary_suffix = 'motif_syllable_aligned_burst_onset_times.mat'
    summary_fname = os.path.join(cluster_folder, 'burst_identity', summary_suffix)
    spacetime = {} # Vigi format

    # syllable on-/offset times
    motif = np.zeros((len(syllable_onset_times), 2))
    for i in range(len(syllable_onset_times)):
        motif[i, 0] = syllable_onset_times[i]
        motif[i, 1] = syllable_offset_times[i]
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
        trial_nr = motif_ids[i]
        for j in range(len(syllable_onset_times)):
            if not syllable_motifs[j] == trial_nr:
                continue
            if syllable_onset_times[j] <= t <= syllable_offset_times[j]:
                syllable_ids[i, 0] = j + 1
                syllable_times[i, 0] = t - syllable_onset_times[j]
                break
    spacetime['sylID'] = syllable_ids
    spacetime['syl_T'] = syllable_times

    # keep track of motif number during which the syllable occurred
    motif_ids_ = np.full((len(burst_onset_times), 9), np.nan)
    motif_ids_[:, 0] = motif_ids[:]
    spacetime['motif_nr'] = motif_ids_

    # dummy variable for uncertainty of burst onset times; small (1e-4 s)
    burst_variance = np.full((len(burst_onset_times), 9), np.nan)
    burst_variance[:, 0] = 1e-4 * np.ones(len(burst_onset_times))
    spacetime['bS'] = burst_variance

    scipy.io.savemat(summary_fname, {'SpaceTime': spacetime})


def _save_mean_syllables_for_matlab(experiment_info, syllable_burst_onsets, syllable_burst_variances,
                                    motif_burst_onsets, syllable_burst_labels, reference_syllables):
    print 'Saving mean burst onset times in syllables in matlab format...'
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    summary_suffix = 'motif_syllable_aligned_mean_burst_onset_times.mat'
    summary_fname = os.path.join(cluster_folder, 'burst_identity', summary_suffix)
    spacetime = {} # Vigi format

    syllable_labels = reference_syllables.keys()
    syllable_labels.sort()

    # syllable on-/offset times
    motif = np.zeros((len(syllable_labels), 2))
    for i, label in enumerate(syllable_labels):
        motif[i, 0] = reference_syllables[label][0]
        motif[i, 1] = reference_syllables[label][1]
    spacetime['Motif'] = motif

    # burst onset times in motif
    bt = np.full((len(motif_burst_onsets), 9), np.nan)
    bt[:, 0] = motif_burst_onsets[:]
    spacetime['bT'] = bt

    # syllable ID during which each burst occurs; NaN if during gap
    # burst onset times relative to syllable onset; NaN if during gap
    syllable_ids = np.full((len(syllable_burst_labels), 9), np.nan)
    syllable_times = np.full((len(syllable_burst_onsets), 9), np.nan)
    for i, label in enumerate(syllable_burst_labels):
        label_index = syllable_labels.index(label)
        syllable_ids[i, 0] = label_index + 1
    syllable_times[:, 0] = syllable_burst_onsets[:]
    spacetime['sylID'] = syllable_ids
    spacetime['syl_T'] = syllable_times

    # dummy variable for uncertainty of burst onset times; small (1e-4 s)
    burst_variance = np.full((len(syllable_burst_variances), 9), np.nan)
    burst_variance[:, 0] = syllable_burst_variances[:]
    spacetime['bS'] = burst_variance

    scipy.io.savemat(summary_fname, {'SpaceTime': spacetime})


def _save_motif_for_matlab(experiment_info, burst_onset_times, burst_onset_variances, motif_onset_times,
                           motif_offset_times):
    print 'Saving mean burst onset times in motif in matlab format...'
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    summary_suffix = 'motif_aligned_mean_burst_onset_times.mat'
    summary_fname = os.path.join(cluster_folder, 'burst_identity', summary_suffix)
    spacetime = {} # Vigi format

    # motif on-/offset times
    motif = np.zeros((len(motif_onset_times), 2))
    for i in range(len(motif_onset_times)):
        motif[i, 0] = motif_onset_times[i]
        motif[i, 1] = motif_offset_times[i]
    spacetime['Motif'] = motif

    # burst onset times in motif
    bt = np.full((len(burst_onset_times), 9), np.nan)
    bt[:, 0] = burst_onset_times[:]
    spacetime['bT'] = bt

    # syllable ID during which each burst occurs (i.e., always 1 for entire motif)
    # burst onset times relative to motif onset (i.e., same as bT)
    syllable_ids = np.full((len(burst_onset_times), 9), np.nan)
    syllable_times = np.full((len(burst_onset_times), 9), np.nan)
    syllable_ids[:, 0] = 1
    syllable_times[:, :] = bt[:, :]
    spacetime['sylID'] = syllable_ids
    spacetime['syl_T'] = syllable_times

    # uncertainty of burst onset times
    burst_variance = np.full((len(burst_onset_times), 9), np.nan)
    burst_variance[:, 0] = burst_onset_variances[:]
    spacetime['bS'] = burst_variance

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
    # get template audio
    template_fs, template_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_audio_trace(template_data, -1.0, 1.0)
    # get syllables from eGUI
    egui_syllables = _load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])

    # # get clusters
    # data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    # clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)
    # get bursts, burst spike times and spontaneous spike times
    # load all bursts
    # cluster_bursts = {}
    cluster_bursts = []
    for i, cluster_id in enumerate(clusters_of_interest):
        summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
        summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
        with open(summary_burst_fname, 'rb') as summary_burst_file:
            # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
            tmp_bursts = cPickle.load(summary_burst_file)
        # select burst ID
        # cluster_bursts[cluster_id] = tmp_bursts[burst_ids[i]]
        cluster_bursts.append(tmp_bursts[burst_ids[i]])

    # E-GUI SYLLABLE ALIGNMENT - INDIVIDUAL SYLLABLES
    # # for i in [n_motifs - 1]:
    # # for i in [best_motif]:
    # motif_burst_onsets_sorted = []
    # syllable_onsets = []
    # syllable_offsets = []
    # syllable_motifs = []
    # motif_ids = []
    # for i in range(n_motifs):
    #     fig = plt.figure(i)
    #     motif_start = motif_finder_data.start[i]
    #     motif_stop = motif_finder_data.stop[i]
    #     motif_spikes = []
    #     motif_burst_onsets = []
    #     for j, cluster_id in enumerate(clusters_of_interest):
    #         # burst = cluster_bursts[cluster_id]
    #         burst = cluster_bursts[j]
    #         burst_times_motif = burst[i][0] - motif_start
    #         if len(burst_times_motif):
    #             motif_burst_onsets.append(burst_times_motif[0])
    #             motif_spikes.append(burst_times_motif)
    #         # else:
    #         #     motif_burst_onsets.append([])
    #     sorted_indices = np.argsort(motif_burst_onsets)
    #     motif_spikes_sorted = []
    #     # motif_burst_onsets_sorted = []
    #     for index in sorted_indices:
    #         motif_spikes_sorted.append(motif_spikes[index])
    #         motif_burst_onsets_sorted.append(motif_burst_onsets[index])
    #         motif_ids.append(i)
    #     ax = plt.subplot(1, 1, 1)
    #     ax.eventplot(motif_spikes_sorted, colors='k', linewidths=0.5)
    #     # ax.eventplot(motif_burst_onsets_sorted, colors='r', linewidths=1.0)
    #     t_audio = np.linspace(0.0, motif_stop - motif_start, len(motif_audio_traces[i]))
    #     plot_audio = utils.normalize_audio_trace(motif_audio_traces[i])
    #     ax.plot(t_audio, plot_audio + len(motif_burst_onsets) + 2, 'k', linewidth=0.5)
    #     # syllable_onsets = []
    #     # syllable_offsets = []
    #     for syllable in egui_syllables:
    #         if i in egui_syllables[syllable].motifs:
    #             motif_index = np.where(egui_syllables[syllable].motifs == i)[0]
    #             onset, offset = np.squeeze(egui_syllables[syllable].onsets[motif_index]), \
    #                             np.squeeze(egui_syllables[syllable].offsets[motif_index])
    #             try:
    #                 syllable_onsets.extend(onset)
    #                 syllable_motifs.extend([i for n in range(len(onset))])
    #             except TypeError:
    #                 syllable_onsets.append(onset)
    #                 syllable_motifs.append(i)
    #             try:
    #                 syllable_offsets.extend(offset)
    #             except TypeError:
    #                 syllable_offsets.append(offset)
    #             tmp_ylim = ax.get_ylim()
    #             ax.plot([onset, onset], tmp_ylim, 'r--', linewidth=0.5)
    #             ax.plot([offset, offset], tmp_ylim, 'r--', linewidth=0.5)
    #             ax.set_ylim(tmp_ylim)
    #     title_str = 'Bursts in motif %d' % (i)
    #     ax.set_title(title_str)
    #     plt.show()
    #
    # _save_individual_syllables_for_matlab(experiment_info, motif_ids, motif_burst_onsets_sorted, syllable_onsets,
    #                                       syllable_offsets, syllable_motifs)

    # E-GUI SYLLABLE ALIGNMENT - MEAN ONSET TIMES IN SYLLABLES
    # for i in [n_motifs - 1]:
    # for i in [best_motif]:
    n_motifs = len(motif_finder_data.start)
    reference_syllables = _calculate_reference_syllables(egui_syllables)
    syllable_burst_onsets = []
    syllable_burst_variances = []
    motif_burst_onsets = []
    syllable_burst_labels = []
    syllable_onsets = []
    syllable_offsets = []
    print 'Time in syllable\tOnset variance (ms)'
    for j, cluster_id in enumerate(clusters_of_interest):
        fig = plt.figure(j)
        # burst = cluster_bursts[cluster_id]
        burst = cluster_bursts[j]
        cluster_syllables = []
        cluster_burst_onsets = []
        for i in range(n_motifs):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            burst_times_motif = burst[i][0] - motif_start
            if len(burst_times_motif):
                syllable, ref_time = _map_trial_time_to_reference_syllable(burst_times_motif[0], i, egui_syllables)
                if syllable is None:
                    continue
                cluster_syllables.append(syllable)
                cluster_burst_onsets.append(ref_time)
        ax = plt.subplot(1, 1, 1)
        title_str = 'Cluster %d burst onset times' % (cluster_id)
        ax.set_title(title_str)
        for i in range(len(cluster_burst_onsets)):
            ax.plot([cluster_burst_onsets[i], cluster_burst_onsets[i]], [i-0.5, i+0.5], 'k-', linewidth=0.5)
        t_audio = np.linspace(0.0, motif_finder_data.stop[0] - motif_finder_data.start[0], len(plot_audio))
        ax.plot(t_audio, plot_audio + len(cluster_burst_onsets) + 2, 'k', linewidth=0.5)
        syllable_ = np.unique(cluster_syllables)
        if len(syllable_) > 1:
            e = 'Burst onset time in more than one syllables'
            raise RuntimeError(e)
        if not len(syllable_): # gap burst
            continue
        syllable = syllable_[0]
        onset = reference_syllables[syllable][0]
        offset = reference_syllables[syllable][1]
        tmp_ylim = ax.get_ylim()
        ax.plot([onset, onset], tmp_ylim, 'r--', linewidth=0.5)
        ax.plot([offset, offset], tmp_ylim, 'r--', linewidth=0.5)
        ax.set_ylim(tmp_ylim)
        plt.show()
        # print 'Cluster %d, mean burst onset time = %.3f s' % (cluster_id, np.mean(cluster_burst_onsets) - reference_syllables[syllable][0])
        onset_var = np.std(cluster_burst_onsets) * 1e3
        print '%.3f\t%.1f' % (np.mean(cluster_burst_onsets) - reference_syllables[syllable][0], onset_var)
        syllable_burst_onsets.append(np.mean(cluster_burst_onsets) - reference_syllables[syllable][0])
        syllable_burst_variances.append(np.std(cluster_burst_onsets))
        syllable_burst_labels.append(syllable)
        motif_burst_onsets.append(np.mean(cluster_burst_onsets))

    _save_mean_syllables_for_matlab(experiment_info, syllable_burst_onsets, syllable_burst_variances,
                                    motif_burst_onsets, syllable_burst_labels, reference_syllables)


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
    # get template audio
    template_fs, template_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_audio_trace(template_data, -1.0, 1.0)

    # # get clusters
    # data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    channel_shank_map = np.load(os.path.join(cluster_folder, 'channel_shank_map.npy'))
    channel_locations = np.load(os.path.join(cluster_folder, 'channel_positions.npy'))
    clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)
    # get bursts, burst spike times and spontaneous spike times
    # load all bursts
    # cluster_bursts = {}
    cluster_bursts = []
    cluster_bursts_proofread = []
    proofread = False
    for i, cluster_id in enumerate(clusters_of_interest):
        summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
        summary_burst_suffix_proofed = 'burst_times_waveforms_cluster_%d_proofread.pkl' % cluster_id
        summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
        summary_burst_fname_proofed = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix_proofed)
        with open(summary_burst_fname, 'rb') as summary_burst_file:
            # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
            tmp_bursts = cPickle.load(summary_burst_file)
        if proofread:
            with open(summary_burst_fname_proofed, 'rb') as summary_burst_file_proofed:
            #     cluster_bursts[cluster_id] = cPickle.load(summary_burst_file_proofed)
                tmp_bursts_proofed = cPickle.load(summary_burst_file_proofed)
        # select burst ID
        # cluster_bursts[cluster_id] = tmp_bursts[burst_ids[i]]
        # not proofread
        cluster_bursts.append(tmp_bursts[burst_ids[i]])
        # proofread
        if proofread:
            cluster_bursts_proofread.append(tmp_bursts_proofed)

    # MOTIF FINDER ALIGNMENT
    n_motifs = len(motif_finder_data.start)
    # print 'Cluster\tBurst ID\tOnset variance (ms)'
    print 'Time in motif(s)\tOnset variance (ms)'
    burst_onset_times = []
    burst_onset_variances = []
    burst_max_channel_locs = []
    cmap_name = 'gist_ncar'
    cmap = cm.get_cmap(cmap_name)
    color_norm = colors.Normalize(0, len(clusters_of_interest) - 1)
    for j, cluster_id in enumerate(clusters_of_interest):
        fig = plt.figure(2*j)
        if proofread:
            burst_proofed = cluster_bursts_proofread[j]
        else:
            burst = cluster_bursts[j]
        cluster_burst_onsets = []
        cluster_spikes = []
        spike_times_flattened = []
        for i in range(n_motifs):
            if len(burst[i][0]):
                motif_start = motif_finder_data.start[i]
                motif_warp = motif_finder_data.warp[i]
                if proofread:
                    # reverse this: aligned_times = burst_spikes_proofed[trial] + burst_spike_times[0] - pre_window * 1.0e-3
                    # pre_window = 5.0
                    # tmp1 = burst_proofed[i][0] - burst[i][0][0] + pre_window * 1.0e-3
                    # tmp1 *= 1.0e-3
                    # tmp1 += burst[i][0][0] - pre_window * 1.0e-3
                    # proofread
                    # burst_times_motif = (tmp1 - motif_start) / motif_warp
                    burst_times_motif = (burst_proofed[i][0] - motif_start) / motif_warp
                # not proofread
                else:
                    burst_times_motif = (burst[i][0] - motif_start) / motif_warp
            # if len(burst_times_motif):
                cluster_burst_onsets.append(burst_times_motif[0])
                cluster_spikes.append(burst_times_motif)
                # spike_times_flattened.extend(tmp1)
                if proofread:
                    spike_times_flattened.extend(burst_proofed[i][0])
                else:
                    spike_times_flattened.extend(burst[i][0])
            else:
                cluster_spikes.append([])

        ax = plt.subplot(1, 1, 1)
        # ax.eventplot(cluster_spikes, colors='k', linewidths=0.5)
        ax.eventplot(cluster_spikes, colors=cmap(color_norm(j)), linewidths=0.5)
        t_audio = np.linspace(0.0, motif_finder_data.stop[0] - motif_finder_data.start[0], len(plot_audio))
        # not proofread
        # ax.plot(t_audio, plot_audio + len(cluster_burst_onsets) + 2, 'k', linewidth=0.5)
        # proofread
        ax.plot(t_audio, plot_audio + n_motifs + 2, 'k', linewidth=0.5)
        onset_var = np.std(cluster_burst_onsets) * 1e3
        # if np.mean(cluster_burst_onsets) > 0.4:
        #     burst_onset_times.append(np.mean(cluster_burst_onsets))
        #     burst_onset_variances.append(onset_var)
        burst_onset_times.append(np.mean(cluster_burst_onsets))
        burst_onset_variances.append(onset_var)
        title_str = 'Burst onset %d of cluster %d - var = %.1f ms' % (burst_ids[j], cluster_id, onset_var)
        # print '%s\t%d\t%.1f' % (cluster_id, burst_ids[j], onset_var)
        print '%.3f\t%.1f' % (np.mean(cluster_burst_onsets), onset_var)
        ax.set_title(title_str)
        fig_suffix = 'Cluster_%d_burst_%d_motif_aligned.pdf' % (cluster_id, burst_ids[j])
        fig_name = os.path.join(cluster_folder, 'burst_identity', fig_suffix)
        plt.savefig(fig_name)
        plt.show()

        cluster = clusters[cluster_id]
        raw_burst_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info, channel_shank_map,
                                                                                cluster, spike_times_flattened)
        # spike times, channels, samples
        mean_wf = np.mean(raw_burst_waveforms, axis=0)
        max_channel = np.argmax(np.max(mean_wf, axis=1) - np.min(mean_wf, axis=1))
        shank = cluster.shank
        channels = np.where(channel_shank_map == shank)[0]
        max_channel += (shank - 1) * len(channels)
        max_channel = int(max_channel + 0.5)
        burst_max_channel_locs.append(channel_locations[max_channel])

    fig2 = plt.figure(1)
    ax2 = plt.subplot(1, 1, 1)
    burst_max_channel_locs = np.array(burst_max_channel_locs)
    ax2.plot(channel_locations[:, 0], channel_locations[:, 1], 'ko', markersize=1)
    ax2.scatter(burst_max_channel_locs[:, 0], burst_max_channel_locs[:, 1], c=range(burst_max_channel_locs.shape[0]),
                cmap=cmap_name)

    fig3 = plt.figure(2)
    ax3 = plt.subplot(1, 1, 1)
    for t in burst_onset_times:
        ax3.plot([t, t], [0, 1], 'k-', linewidth=1)
    t_audio = np.linspace(0.0, motif_finder_data.stop[0] - motif_finder_data.start[0], len(plot_audio))
    ax3.plot(t_audio, plot_audio + 3, 'k', linewidth=0.5)
    ax3.set_title('Mean burst onset times')
    ax3.set_xlabel('Time (s)')

    plt.show()

    motif_times = 0.0, motif_finder_data.stop[0] - motif_finder_data.start[0]
    _save_motif_for_matlab(experiment_info, burst_onset_times, burst_onset_variances, [motif_times[0]],
                           [motif_times[1]])


def manual_burst_proofing(experiment_info_name):
    '''
    Manually check burst spikes and add missing spike times
    :param experiment_info_name:
    :return:
    '''
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get full audio
    audio_name = os.path.join(experiment_info['Motifs']['DataBasePath'], experiment_info['Motifs']['AudioFilename'])
    audio_fs, audio_data = cp.reader.read_audiofile(audio_name)
    # get template audio
    template_fs, template_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_audio_trace(template_data, -1.0, 1.0)

    # # get clusters
    data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)

    intan_constant = 0.195
    recording_file = cp.reader.load_recording(os.path.join(data_folder, experiment_info['SiProbe']['AmplifierName']),
                                              experiment_info['SiProbe']['Channels'])
    b, a = utils.set_up_bp_filter(300.0, 0.49*fs, fs)

    # get bursts, burst spike times and spontaneous spike times
    # load all bursts
    # cluster_bursts = {}
    n_motifs = len(motif_finder_data.start)
    cluster_bursts = []
    for i, cluster_id in enumerate(clusters_of_interest):
        summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
        summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
        with open(summary_burst_fname, 'rb') as summary_burst_file:
            # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
            tmp_bursts = cPickle.load(summary_burst_file)
        # select burst ID
        cluster_burst = tmp_bursts[burst_ids[i]]
        cluster = clusters[cluster_id]

        cluster_burst_onsets = []
        cluster_spikes = []
        spike_times_flattened = []
        for j in range(n_motifs):
            motif_start = motif_finder_data.start[j]
            motif_warp = motif_finder_data.warp[j]
            burst_times_motif = (cluster_burst[j][0] - motif_start) / motif_warp
            if len(burst_times_motif):
                cluster_burst_onsets.append(burst_times_motif[0])
                cluster_spikes.append(burst_times_motif)
                spike_times_flattened.extend(cluster_burst[j][0])

        # template max channel not necessarily reliable for sparse units split off more active units
        # max_channel = cluster.maxChannel
        # hence, determine max channel from raw waveforms
        channel_shank_map = np.load(os.path.join(cluster_folder, 'channel_shank_map.npy'))
        raw_burst_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info, channel_shank_map,
                                                                                cluster, spike_times_flattened)
        # spike times, channels, samples
        mean_wf = np.mean(raw_burst_waveforms, axis=0)
        max_channel = np.argmax(np.max(mean_wf, axis=1) - np.min(mean_wf, axis=1))
        shank = cluster.shank
        channels = np.where(channel_shank_map == shank)[0]
        max_channel += (shank - 1) * len(channels)
        max_channel = int(max_channel + 0.5)
        # pre_window = 5.0 # ms before first spike time
        pre_window = 50.0 # ms before first spike time
        # post_window = 15.0 # ms after first spike time
        post_window = 50.0 # ms after first spike time
        pre_window_index = int(pre_window*1e-3*fs)
        post_window_index = int(post_window*1e-3*fs)
        trial_cnt = 0
        spike_cnt = 0
        burst_spikes_proofed = {}
        while trial_cnt < len(cluster_burst):
            if not len(cluster_burst[trial_cnt][0]):
                trial_cnt += 1
                continue

            # ugly hack: re-plot this figure every time
            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
            ax.eventplot(cluster_spikes, colors='k', linewidths=0.5)
            t_audio = np.linspace(0.0, motif_finder_data.stop[0] - motif_finder_data.start[0], len(plot_audio))
            ax.plot(t_audio, plot_audio + len(cluster_burst_onsets) + 2, 'k', linewidth=0.5)
            onset_var = np.std(cluster_burst_onsets) * 1e3
            title_str = 'Burst onset %d of cluster %d - var = %.1f ms' % (burst_ids[i], cluster_id, onset_var)
            ax.set_title(title_str)

            # new plot the interactive figure
            burst_spike_times = cluster_burst[trial_cnt][0]
            t_spike = burst_spike_times[0]
            t_spike_index = int(t_spike*fs)
            start_index = t_spike_index - pre_window_index
            stop_index = t_spike_index + post_window_index
            snippet = intan_constant * recording_file[max_channel, start_index:stop_index]
            filtered_snippet = scipy.signal.filtfilt(b, a, snippet)
            fig_interactive = plt.figure()
            ax = plt.subplot(1, 1, 1)
            ax.plot(np.array(range(len(filtered_snippet)))*1.0e3/fs, filtered_snippet, 'k', linewidth=0.5)
            for t_spike_tmp in burst_spike_times:
                t_spike_tmp_shift = t_spike_tmp*1.0e3 - (t_spike*1.0e3 - pre_window)
                # t_spike_index_tmp = int(t_spike_tmp * fs) - (t_spike_index - burst_window_index)
                y_min, y_max = ax.get_ylim()
                ax.plot((t_spike_tmp_shift, t_spike_tmp_shift), (y_min, y_max), 'r--', linewidth=0.5)
                ax.set_ylim((y_min, y_max))
            title_str = 'Cluster %d; trial %d (spike %d)' % (cluster_id, trial_cnt, spike_cnt)
            ax.set_title(title_str)
            spike_times_picked = []
            sp = utils.SpikePicker(ax, spike_times_picked)
            sp.connect()
            plt.show()
            plt.close(fig_interactive)
            burst_spikes_proofed[trial_cnt] = np.array(spike_times_picked)
            if len(spike_times_picked):
                spike_cnt += 1
            trial_cnt += 1

        # append manually added burst spikes to existing spikes
        for trial in burst_spikes_proofed:
            burst_spike_times = cluster_burst[trial][0]
            # WRONG UNITS!!! TO BE FIXED:
            # aligned_times = burst_spikes_proofed[trial] + burst_spike_times[0] - pre_window * 1.0e-3
            aligned_times = burst_spikes_proofed[trial] * 1.0e-3 + burst_spike_times[0] - pre_window * 1.0e-3
            tmp_wf = cluster_burst[trial][1]
            cluster_burst[trial] = (aligned_times, tmp_wf)
            # new_times = np.zeros(burst_spike_times.shape)
            # new_times[:len(burst_spike_times)] = burst_spike_times[:]
            # new_times[len(burst_spike_times):] = aligned_times[:]
            # cluster_burst[trial][0] = new_timesq
        # save
        summary_burst_suffix_out = 'burst_times_waveforms_cluster_%d_proofread.pkl' % cluster_id
        summary_burst_fname_out = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix_out)
        with open(summary_burst_fname_out, 'wb') as summary_burst_file_out:
            cPickle.dump(cluster_burst, summary_burst_file_out, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        info_name = sys.argv[1]
        syllable_aligned_bursts(info_name)
        # motif_aligned_bursts(info_name)
        # manual_burst_proofing(info_name)
