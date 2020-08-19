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

bird_bursts = dict()
bird_info = dict()
cell_ids = []
clusters_of_interest = []
burst_ids = []

bird_info['C21'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C21\clustered\experiment_C21_d1_alignment_reducedTemp.info'
bird_info['C22'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C22\d2_afternoon_song_stim\experiment_C22_d2_afternoon_song_alignment.info'
# bird_info['C22'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C22\d2_afternoon_song_stim\experiment_C22_d2_afternoon_song_alignment_non-RA.info'
# use the following for motif-level variability
bird_info['C23'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C23\C23_190611_131550\experiment_C23_song_alignment_BAonly.info'
# and this one for syllable-level variability
# bird_info['C23'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C23\C23_190611_131550\experiment_C23_song_alignment.info'
bird_info['C24'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C24\experiment_C24_alignment.info'
bird_info['C25'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C25\experiment_C25_alignment.info'

# C21
# clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929]
# burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929, 767]
# burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0, 0]
# all non-HVC(RA) PNs
bird_bursts['C21'] = ([58, 388, 741, 108, 209, 46, 46, 128, 128, 128, 266, 266, 353, 353, 454, 454, 685, 685,
                       728, 728, 733, 733, 733, 738, 738, 738, 917, 917, 1, 1, 9, 9, 9, 696, 696, 696, 696, 732, 732,
                       759, 759, 764, 764, 772, 772],
                      [0, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1,
                       0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2, 1, 2, 3, 0, 1, 2, 4, 0, 2,
                       0, 2, 0, 1, 0, 1])
# remove unstable bursts
# bird_bursts['C21'] = ([55, 304, 309, 695, 701, 702, 761, 779, 108, 696, 732, 759, 764, 772, 767],
#                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 1, 2, 2, 0])
# C22
# clusters_of_interest = [30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833, 938]
# burst_ids = [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0]
# incl. 22 low-frequency spontaneous
# bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833, 938,
#                        622, 639, 703, 738, 791, 832, 942],
#                       [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0,
#                        0, 0, 0, 0, 1, 0, 0])
# pick common motifs (HVC(RA) PNs)
# bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 622, 639, 650, 685, 703, 738, 791, 832, 833,
#                        938, 942],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                        0, 0])
# pick common motifs (old version non-VC(RA) PNs)
# dummy_ids = [58, 60, 61, 76, 97, 136, 151, 162, 177, 180, 183, 209, 216, 243, 299, 323, 359, 368, 377, 402,
#                        613, 616, 638, 649, 677, 678, 692, 693, 724, 728, 735, 824]
# bird_bursts['C22'] = (dummy_ids, [0 for i in range(len(dummy_ids))])
# forgot this one above
bird_bursts['C22'] = ([124], [0])
# all non-HVC(RA) PNs
# bird_bursts['C22'] = ([30, 69, 71, 103, 126, 205, 225, 258, 354, 370, 497, 546, 734, 765, 786, 803, 252, 252, 252, 252,
#                        288, 288, 288, 305, 305, 318, 318, 318, 318, 318, 339, 339, 339, 532, 532, 532, 532,
#                        603, 603, 603, 604, 604, 607, 607, 607, 607, 623, 623, 650, 650, 650, 657, 657, 657, 657,
#                        694, 694, 694, 694, 719, 719, 723, 723, 741, 741, 761, 761, 804, 804, 810, 810, 817, 817, 817,
#                        833, 833, 945, 945, 945],
#                       [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
#                        0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3,
#                        0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 0, 3, 4, 0, 1, 2, 3,
#                        0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 1, 3, 0, 1, 0, 2, 3,
#                        1, 2, 0, 1, 2])
# bird_bursts['C22'] = ([547],
#                       [0])
# remove unstable bursts
# bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833,
#                        622, 639, 703, 791, 832],
#                       [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0,
#                        0, 0, 0, 1, 0])
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
# bird_bursts['C23'] = ([776,  842, 1092, 1154, 1166, 1205, 1220, 1267, 1268, 1302, 1303, 1330, 1340, 1376,
#                         670, 786, 941, 777, 938, 1093, 1330, 1154],
#                        [3, 2, 1, 2, 2, 3, 2, 1, 3, 1, 1, 1, 3, 0,
#                         1, 1, 1, 3, 3, 1, 1, 3])
# all non-HVC(RA) PNs
bird_bursts['C23'] = ([883, 918, 1073, 777, 841, 1288, 1298, 1220, 387, 983, 983, 807, 807, 1116, 1116, 1129, 1129,
                       1175, 1175, 1247, 1283, 1283, 1205, 1205, 1257, 1257, 1340, 1340, 1374, 1374],
                       [0, 0, 0, 2, 2, 0, 0, 3, 3, 2, 3, 3, 4, 2, 3, 2, 3,
                        1, 2, 2, 2, 3, 4, 5, 3, 4, 4, 5, 2, 3])
# remove unstable bursts
# bird_bursts['C23'] = ([776, 842, 1092, 1154, 1166, 1205, 1220, 1267, 1303, 1330, 1340,
#                        670, 786, 777, 1330, 1154],
#                       [3, 2, 1, 2, 2, 3, 2, 1, 1, 1, 3,
#                        1, 1, 3, 1, 3])
# C24
# clusters_of_interest = [77, 89, 91, 264, 563, 743, 753, 813, 853]
# burst_ids = [2, 1, 2, 0, 1, 1, 1, 1, 0]
# incl. 22 low-frequency spontaneous
# bird_bursts['C24'] = ([77, 89, 91, 264, 563, 743, 753, 813, 853,
#                         360, 751, 867, 904],
#                       [2, 1, 2, 0, 1, 1, 1, 1, 0,
#                         0, 0, 0, 0])
# all non-HVC(RA) PNs
bird_bursts['C24'] = ([5, 14, 23, 31, 55, 89, 148, 159, 196, 200, 231, 249, 264, 273, 330, 336, 349, 459, 478, 558, 563,
                       564, 720, 725, 743, 812, 813, 824, 827, 848, 853, 858, 867, 871, 883, 884, 903, 77, 77,
                       91, 91, 91, 190, 190, 221, 221, 286, 286, 308, 308, 308, 311, 311, 311, 388, 388, 451, 451, 451,
                       616, 616, 653, 653, 743, 743, 753, 753, 767, 767, 767, 801, 801, 801, 804, 804, 804, 880, 880,
                       908, 908],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                       0, 1, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2,
                       0, 1, 0, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
                       0, 1])
# remove unstable bursts
# bird_bursts['C24'] = ([77, 89, 91, 264, 563, 743, 753, 813, 853,
#                        751, 867, 904],
#                       [2, 1, 2, 0, 1, 1, 1, 1, 0,
#                        0, 0, 0])
# C25
# clusters_of_interest = [110, 130, 159, 189, 521, 240, 289, 310, 346, 366, 412, 432]
# burst_ids = [0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1]
# incl. 22 low-frequency spontaneous
# bird_bursts['C25'] = ([110, 130, 159, 189, 521, 240, 289, 310, 346, 366, 412, 432],
#                       [0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1])
# all non-HVC(RA) PNs
bird_bursts['C25'] = ([16, 50, 77, 79, 95, 104, 119, 139, 159, 187, 189, 194, 208, 229, 234, 339, 378, 386, 391, 412,
                       418, 432, 442, 456, 465, 469, 476, 72, 72, 163, 163, 216, 216, 216, 240, 240, 289, 289, 310, 310,
                       320, 320, 325, 325, 346, 346, 384, 384, 443, 443, 453, 453, 471, 471],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 2, 0, 2, 1, 2,
                       0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# remove unstable bursts
# bird_bursts['C25'] = ([110, 130, 189, 521, 240, 289, 310, 346, 366, 412, 432],
#                       [0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1])


def _load_common_data(experiment_info_name):
    # load all bursts in all trials
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))

    # # get clusters
    # data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    # get bursts, burst spike times and spontaneous spike times
    # load all bursts
    cluster_bursts = dict()
    cluster_celltypes = dict()
    cluster_bursts_proofread = dict()
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
        clean_burst = _clean_up_bursts(tmp_bursts[burst_ids[i]])
        checksum = 0
        for trial_burst in clean_burst:
            if len(trial_burst):
                checksum = 1
        if checksum:
            cluster_bursts[i] = clean_burst
            cluster_celltypes[i] = celltypes[i]
        # proofread
        if proofread:
            clean_burst_proofread = _clean_up_bursts(tmp_bursts_proofed)
            checksum = 0
            for trial_burst in clean_burst_proofread:
                if len(trial_burst):
                    checksum = 1
            if len(clean_burst_proofread):
                cluster_bursts_proofread[i] = clean_burst_proofread

    # now sort this by cell id
    # types_per_cell = []
    # types_per_cell.append(cluster_celltypes[0])
    # for i in range(1, len(cell_ids)):
    #     if cell_ids[i] == cell_ids[i - 1]:
    #         continue
    #     types_per_cell.append(celltypes[i])
    cell_bursts = dict()
    types_per_cell = dict()
    unique_cell_ids = np.unique(cell_ids)
    for cell_id in unique_cell_ids:
        burst_indices = [i for i, x in enumerate(cell_ids) if x == cell_id]
        types_per_cell[cell_id] = celltypes[burst_indices[0]]
        joint_bursts = []
        n_trials = len(cluster_bursts[cluster_bursts.keys()[0]])
        for i in range(n_trials):
            common_spikes = []
            for burst_index in burst_indices:
                if not cluster_bursts.has_key(burst_index):
                    continue
                common_spikes.extend(cluster_bursts[burst_index][i])
            common_spikes.sort()
            joint_bursts.append(np.array(common_spikes))
        cell_bursts[cell_id] = joint_bursts

    common_data = dict()
    common_data['motif'] = motif_finder_data
    common_data['bursts'] = cell_bursts
    common_data['celltypes'] = types_per_cell
    common_data['bursts_proofread'] = cluster_bursts_proofread

    return common_data


def _clean_up_bursts(bursts):
    # ugh I should have taken care of this during burst sorting... remove FP at ISIs <= 1 ms
    # also only keep all spikes with ISIs < 10 ms
    clean_bursts = []
    trials = len(bursts)
    for trial in range(trials):
        spikes = bursts[trial]
        if len(spikes[0]) < 2:
            clean_bursts.append([])
            # clean_bursts.append(spikes[0])
            continue
        # tmp = spikes[0]
        tmp = []
        for i in range(len(spikes[0]) - 1):
            if spikes[0][i + 1] - spikes[0][i] >= 0.001:
                tmp.append(spikes[0][i + 1])
                tmp.append(spikes[0][i])
            else:
                continue
        tmp = np.unique(tmp)
        tmp_burst = []
        for i in range(len(tmp) - 1):
            if tmp[i + 1] - tmp[i] < 0.01:
                tmp_burst.append(tmp[i + 1])
                tmp_burst.append(tmp[i])
            else:
                break
        cleaned_tmp_burst = np.unique(tmp_burst)
        clean_bursts.append(cleaned_tmp_burst)

    return clean_bursts


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
    summary_suffix = 'motif_syllable_aligned_mean_burst_onset_times_non-RA.mat'
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
    summary_suffix = 'motif_aligned_mean_burst_onset_times_non-RA.mat'
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


def _save_motif_spikes_for_matlab(experiment_info, cell_types, spike_times):
    print 'Saving spike times in motif in matlab format...'
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    summary_suffix = 'motif_aligned_spike_times_ME.mat'
    summary_fname = os.path.join(cluster_folder, 'burst_identity', summary_suffix)
    spacetime = {}  # Vigi format

    # cell types
    ct = np.array(cell_types, dtype=np.object)
    spacetime['celltype'] = ct

    # spike times
    st = np.array(spike_times, dtype=np.object)
    spacetime['spiketimes'] = st

    scipy.io.savemat(summary_fname, {'MotifSpikes': spacetime})


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
    # audio_name = os.path.join(experiment_info['Motifs']['DataBasePath'], experiment_info['Motifs']['AudioFilename'])
    # audio_fs, audio_data = cp.reader.read_audiofile(audio_name)
    # get template audio
    template_fs, template_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_audio_trace(template_data, -1.0, 1.0)
    # get syllables from eGUI
    egui_syllables = utils.load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])

    # UGLY HACK for C22 2nd alignment for non-RA
    # C22_nonRA_motifs = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]

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
    reference_syllables = utils.calculate_reference_syllables(egui_syllables)
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
        # UGLY HACK for C22 non-RA
        # if len(burst) == n_motifs:
        #     motif_iter = range(n_motifs)
        # else:
        #     motif_iter = C22_nonRA_motifs
        # END UGLY HACK for C22 non-RA
        for i in range(n_motifs):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            # UGLY HACK for C22 non-RA
            # burst_times_motif = burst[motif_iter[i]][0] - motif_start
            # END UGLY HACK for C22 non-RA
            # following line for normal version
            burst_times_motif = burst[i][0] - motif_start
            if len(burst_times_motif):
                # UGLY HACK for C22 non-RA
                # syllable, ref_time = utils.map_trial_time_to_reference_syllable(burst_times_motif[0], C22_nonRA_motifs[i], egui_syllables)
                # END UGLY HACK for C22 non-RA
                # following line for normal version
                syllable, ref_time = utils.map_trial_time_to_reference_syllable(burst_times_motif[0], i, egui_syllables)
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

    # _save_mean_syllables_for_matlab(experiment_info, syllable_burst_onsets, syllable_burst_variances,
    #                                 motif_burst_onsets, syllable_burst_labels, reference_syllables)


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
    # audio_name = os.path.join(experiment_info['Motifs']['DataBasePath'], experiment_info['Motifs']['AudioFilename'])
    # audio_fs, audio_data = cp.reader.read_audiofile(audio_name)
    # get template audio
    template_fs, template_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_audio_trace(template_data, -1.0, 1.0)

    # UGLY HACK for C22 2nd alignment for non-RA
    C22_nonRA_motifs = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]

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
    fig = plt.figure(0)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 1])
    all_cluster_spikes = {} # dict of all spike times in all motifs; keys are mean burst onset time
    for j, cluster_id in enumerate(clusters_of_interest):
        # fig = plt.figure(2*j)
        if proofread:
            burst_proofed = cluster_bursts_proofread[j]
        else:
            burst = cluster_bursts[j]
        cluster_burst_onsets = []
        cluster_spikes = {}
        spike_times_flattened = []
        # UGLY HACK for C22 non-RA
        if len(burst) == n_motifs:
            motif_iter = range(n_motifs)
        else:
            motif_iter = C22_nonRA_motifs
        # END UGLY HACK for C22 non-RA
        for i in range(n_motifs):
            # if len(burst_proofed[i][0]):
            # UGLY HACK for C22 non-RA
            if len(burst[motif_iter[i]][0]):
            # END UGLY HACK for C22 non-RA
            # following line for normal version
            # if len(burst[i][0]):
                motif_start = motif_finder_data.start[i]
                motif_warp = motif_finder_data.warp[i]
                # motif_warp = 1.0 # Show slowest and longest sequence in figure to illustrate global sequence variance
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
                    # UGLY HACK for C22 non-RA
                    burst_times_motif = (burst[motif_iter[i]][0] - motif_start) / motif_warp
                    # END UGLY HACK for C22 non-RA
                    # following line for normal version
                    # burst_times_motif = (burst[i][0] - motif_start) / motif_warp
            # if len(burst_times_motif):
                cluster_burst_onsets.append(burst_times_motif[0])
                ax.plot(burst_times_motif[0], i, color=cmap(color_norm(j)), linewidth=0.5)
                cluster_spikes[i] = burst_times_motif
                # spike_times_flattened.extend(tmp1)
                if proofread:
                    spike_times_flattened.extend(burst_proofed[i][0])
                else:
                    # UGLY HACK for C22 non-RA
                    spike_times_flattened.extend(burst[motif_iter[i]][0])
                    # END UGLY HACK for C22 non-RA
                    # following line for normal version
                    # spike_times_flattened.extend(burst[i][0])
            else:
                cluster_spikes[i] = []

        # ax = plt.subplot(1, 1, 1)
        # ax.eventplot(cluster_spikes, colors='k', linewidths=0.5)
        # ax.eventplot(cluster_spikes, colors=cmap(color_norm(j)), linewidths=0.5)
        t_audio = np.linspace(0.0, motif_finder_data.stop[0] - motif_finder_data.start[0], len(plot_audio))
        # not proofread
        # ax.plot(t_audio, plot_audio + len(cluster_burst_onsets) + 2, 'k', linewidth=0.5)
        # proofread
        # ax.plot(t_audio, plot_audio + n_motifs + 2, 'k', linewidth=0.5)
        onset_var = np.std(cluster_burst_onsets) * 1e3
        # if np.mean(cluster_burst_onsets) > 0.4:
        #     burst_onset_times.append(np.mean(cluster_burst_onsets))
        #     burst_onset_variances.append(onset_var)
        burst_onset_times.append(np.mean(cluster_burst_onsets))
        burst_onset_variances.append(onset_var)
        # title_str = 'Burst onset %d of cluster %d - var = %.1f ms' % (burst_ids[j], cluster_id, onset_var)
        # print '%s\t%d\t%.1f' % (cluster_id, burst_ids[j], onset_var)
        print '%.3f\t%.1f' % (np.mean(cluster_burst_onsets), onset_var)
        all_cluster_spikes[np.mean(cluster_burst_onsets)] = cluster_spikes
        # ax.set_title(title_str)
        # fig_suffix = 'Cluster_%d_burst_%d_motif_aligned_no_warping.pdf' % (cluster_id, burst_ids[j])
        # fig_name = os.path.join(cluster_folder, 'burst_identity', fig_suffix)
        # plt.savefig(fig_name)
        # plt.show()

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

    # bursts = all_cluster_spikes.keys()
    # bursts.sort()
    # for i in range(n_motifs):
    #     fig_i = plt.figure(i + 3)
    #     ax_i = fig_i.add_subplot(1, 1, 1)
    #     for j, b in enumerate(bursts):
    #         try:
    #             t_spike = all_cluster_spikes[b][i]
    #             for t in t_spike:
    #                 ax_i.plot([t, t], [j, j + 1], 'k-', linewidth=1)
    #         except KeyError:
    #             continue
    #     ax_i.set_xlabel('Time (s)')
    #     ax_i.set_ylabel('Neuron')
    #     title_str = 'Sequence motif %d' % (i + 1)
    #     ax_i.set_title(title_str)
    #     fig_suffix = 'motif_%d_non-RA_neurons_sequence.pdf' % (i + 1)
    #     fig_name = os.path.join(cluster_folder, 'burst_identity', fig_suffix)
    #     plt.savefig(fig_name)

    # motif_times = 0.0, motif_finder_data.stop[0] - motif_finder_data.start[0]
    # _save_motif_for_matlab(experiment_info, burst_onset_times, burst_onset_variances, [motif_times[0]],
    #                        [motif_times[1]])

    # # ugly HACK for C23: because we're using the second part (BA), shift motif onset to first burst time
    # tmp_offset = 0.428
    # motif_times = 0.0, motif_finder_data.stop[0] - motif_finder_data.start[0] - tmp_offset
    # _save_motif_for_matlab(experiment_info, np.array(burst_onset_times) - tmp_offset, burst_onset_variances,
    #                        [motif_times[0]], [motif_times[1]])


def motif_aligned_cell_bursts(experiment_info_name):
    '''
    Alignment of all bursts sorted by cell in individual trials.
    Possible because they have been recorded simultaneously.
    '''
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cell_bursts = common_data['bursts']
    cluster_bursts_proofread = common_data['bursts_proofread']
    cluster_celltypes = common_data['celltypes']
    proofread = False

    # MOTIF FINDER ALIGNMENT
    n_motifs = len(motif_finder_data.start)
    # for Margot: find motif that has most active cells (she doesn't want average aligned to core motif)
    max_cells, max_motif = 0, 0
    for i in range(n_motifs):
        motif_cells = 0
        for cluster_id, burst in cell_bursts.iteritems():
            motif_cells += len(burst[i]) > 0
        if motif_cells > max_cells:
            max_cells = motif_cells
            max_motif = i

    spike_times_aligned = []
    cell_types_aligned = []
    for cluster_id, burst in cell_bursts.iteritems():
        if len(burst[max_motif]):
            motif_start = motif_finder_data.start[max_motif]
            motif_warp = motif_finder_data.warp[max_motif]
            burst_times_motif = (burst[max_motif] - motif_start) / motif_warp
            spike_times_aligned.append(burst_times_motif)
            cell_types_aligned.append(cluster_celltypes[cluster_id])

    fig3 = plt.figure(2)
    ax3 = fig3.add_subplot(1, 1, 1)
    print 'Cell ID\tCell type\tSpike times'
    for i, t_vec in enumerate(spike_times_aligned):
        print '%d\t%s\t%s' % (i, cell_types_aligned[i], str(t_vec))
        for t in t_vec:
            ax3.plot([t, t], [i, i + 1], 'k-', linewidth=1)
    ax3.set_title('Mean burst onset times')
    ax3.set_xlabel('Time (s)')

    plt.show()

    # motif_times = 0.0, motif_finder_data.stop[max_motif] - motif_finder_data.start[max_motif]
    _save_motif_spikes_for_matlab(experiment_info, cell_types_aligned, spike_times_aligned)

    # # ugly HACK for C23: because we're using the second part (BA), shift motif onset to first burst time
    # tmp_offset = 0.428
    # motif_times = 0.0, motif_finder_data.stop[0] - motif_finder_data.start[0] - tmp_offset
    # _save_motif_for_matlab(experiment_info, np.array(burst_onset_times) - tmp_offset, burst_onset_variances,
    #                        [motif_times[0]], [motif_times[1]])


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


def fix_motifs(experiment_info_name):
    '''
    Stupid fix to remove trials from already clicked burst
    because we re-aligned motifs for C22 (better alignment though)
    '''
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    # # get clusters
    # data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    # C22 - joining it down to the 11 common motifs
    keep_motifs = [0, 2, 3, 4, 7, 10, 11, 13, 14, 15, 16] # non-RA style
    # keep_motifs = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12] # RA style
    load_clusters = np.unique(clusters_of_interest)
    for i, cluster_id in enumerate(load_clusters):
        summary_burst_suffix = 'burst_times_waveforms_cluster_%d.pkl' % cluster_id
        summary_burst_fname = os.path.join(cluster_folder, 'burst_identity', summary_burst_suffix)
        with open(summary_burst_fname, 'rb') as summary_burst_file:
            # cluster_bursts[cluster_id] = cPickle.load(summary_burst_file)
            old_bursts = cPickle.load(summary_burst_file)
        nr_old_trials = len(old_bursts[0])
        # if nr_old_trials == 14:
        #     continue
        new_bursts = dict()
        for burst_id in old_bursts:
            new_bursts[burst_id] = []
            for keep_id in keep_motifs:
                new_bursts[burst_id].append(old_bursts[burst_id][keep_id])

        backup_burst_suffix = 'burst_times_waveforms_cluster_%d_old_motifs.pkl' % cluster_id
        backup_burst_fname = os.path.join(cluster_folder, 'burst_identity', backup_burst_suffix)
        with open(backup_burst_fname, 'wb') as backup_burst_file_out:
            cPickle.dump(old_bursts, backup_burst_file_out, cPickle.HIGHEST_PROTOCOL)

        new_burst_fname = summary_burst_fname
        with open(new_burst_fname, 'wb') as new_burst_file_out:
            cPickle.dump(new_bursts, new_burst_file_out, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # info_name = sys.argv[1]
        valid_bird = False
        while not valid_bird:
            bird_id = raw_input('Please enter a bird ID (C21-25): ')
            try:
                # clusters_of_interest, burst_ids = bird_bursts[bird_id]
                info_name = bird_info[bird_id]
                cell_ids, clusters_of_interest, burst_ids, celltypes = utils.load_burst_info(info_name, cells=True)
                valid_bird = True
            except KeyError:
                print 'Please enter a valid bird ID (C21-25)'
        assert len(clusters_of_interest) == len(burst_ids)

        # syllable_aligned_bursts(info_name)
        # motif_aligned_bursts(info_name)
        motif_aligned_cell_bursts(info_name)
        # manual_burst_proofing(info_name)
        # fix_motifs(info_name)
