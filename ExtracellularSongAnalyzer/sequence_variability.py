import os
import ast
import sys
import cPickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm, matplotlib.colors
import ClusterProcessing as cp
import utilities as utils

bird_bursts = dict()
bird_info = dict()
clusters_of_interest = ()
burst_ids = ()
celltypes = ()

bird_info['C21'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C21\clustered\experiment_C21_d1_alignment_reducedTemp.info'
bird_info['C22'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C22\d2_afternoon_song_stim\experiment_C22_d2_afternoon_song_alignment.info'
# bird_info['C22'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C22\d2_afternoon_song_stim\experiment_C22_d2_afternoon_song_alignment_non-RA.info'
# use the following for motif-level variability
# bird_info['C23'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C23\C23_190611_131550\experiment_C23_song_alignment_BAonly.info'
# and this one for syllable-level variability
bird_info['C23'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C23\C23_190611_131550\experiment_C23_song_alignment.info'
bird_info['C24'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C24\experiment_C24_alignment.info'
bird_info['C25'] = r'Z:\Robert\PolychronousProject\HVC_recordings\C25\experiment_C25_alignment.info'

# C21
# clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929]
# burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [55, 304, 309, 522, 695, 701, 702, 761, 779, 1, 108, 209, 696, 710, 732, 759, 764, 772, 929, 767]
# burst_ids = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 3, 0, 1, 1, 2, 2, 0, 0]
# remove unstable bursts
bird_bursts['C21'] = ([55, 304, 309, 695, 701, 702, 761, 779, 108, 696, 732, 759, 764, 772, 767],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 1, 2, 2, 0])
# C22
# clusters_of_interest = [30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833, 938]
# burst_ids = [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0]
# incl. 22 low-frequency spontaneous
# bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833, 938,
#                        622, 639, 703, 738, 791, 832, 942],
#                       [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0,
#                        0, 0, 0, 0, 1, 0, 0])
# without 685/0 (end of motif; not beginning; due to MotifFinder template
bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 791, 833, 938,
                       622, 639, 703, 738, 791, 832, 942],
                      [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 1, 2, 3, 0, 0,
                       0, 0, 0, 0, 1, 0, 0])
# remove unstable bursts
# bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 685, 791, 833,
#                        622, 639, 703, 791, 832],
#                       [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 2, 3, 0,
#                        0, 0, 0, 1, 0])
# without 685/0 (end of motif; not beginning; due to MotifFinder template
# bird_bursts['C22'] = ([30, 33, 211, 225, 343, 364, 370, 547, 609, 650, 685, 685, 791, 833,
#                        622, 639, 703, 791, 832],
#                       [0, 1, 0, 1, 0, 3, 0, 0, 0, 2, 1, 2, 3, 0,
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
# bird_bursts['C23'] =  ([776,  842, 1092, 1154, 1166, 1205, 1220, 1267, 1268, 1302, 1303, 1330, 1340, 1376,
#                         670, 786, 941, 777, 938, 1093, 1330, 1154],
#                        [3, 2, 1, 2, 2, 3, 2, 1, 3, 1, 1, 1, 3, 0,
#                         1, 1, 1, 3, 3, 1, 1, 3])
# remove unstable bursts
bird_bursts['C23'] = ([776, 842, 1092, 1154, 1166, 1205, 1220, 1267, 1303, 1330, 1340,
                       670, 786, 777, 1330, 1154],
                      [3, 2, 1, 2, 2, 3, 2, 1, 1, 1, 3,
                       1, 1, 3, 1, 3])
# C24
# clusters_of_interest = [77, 89, 91, 264, 563, 743, 753, 813, 853]
# burst_ids = [2, 1, 2, 0, 1, 1, 1, 1, 0]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [77, 89, 91, 264, 563, 743, 753, 813, 853,
#                         360, 751, 867, 904]
# burst_ids = [2, 1, 2, 0, 1, 1, 1, 1, 0,
#              0, 0, 0, 0]
# remove unstable bursts
bird_bursts['C24'] = ([77, 89, 91, 264, 563, 743, 753, 813, 853,
                       751, 867, 904],
                      [2, 1, 2, 0, 1, 1, 1, 1, 0,
                       0, 0, 0])
# C25
# clusters_of_interest = [110, 130, 159, 189, 521, 240, 289, 310, 346, 366, 412, 432]
# burst_ids = [0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1]
# incl. 22 low-frequency spontaneous
# clusters_of_interest = [110, 130, 159, 189, 521, 240, 289, 310, 346, 366, 412, 432]
# burst_ids = [0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1]
# remove unstable bursts
bird_bursts['C25'] = ([110, 130, 189, 521, 240, 289, 310, 346, 366, 412, 432],
                      [0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1])


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
    cluster_bursts = []
    cluster_celltypes = []
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
        clean_burst = _clean_up_bursts(tmp_bursts[burst_ids[i]])
        checksum = 0
        for trial_burst in clean_burst:
            if len(trial_burst):
                checksum = 1
        if checksum:
            cluster_bursts.append(clean_burst)
            cluster_celltypes.append(celltypes[i])
        # proofread
        if proofread:
            clean_burst_proofread = _clean_up_bursts(tmp_bursts_proofed)
            checksum = 0
            for trial_burst in clean_burst_proofread:
                if len(trial_burst):
                    checksum = 1
            if len(clean_burst_proofread):
                cluster_bursts_proofread.append(clean_burst_proofread)

    common_data = dict()
    common_data['motif'] = motif_finder_data
    common_data['bursts'] = cluster_bursts
    common_data['celltypes'] = cluster_celltypes
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
            # clean_bursts.append([])
            clean_bursts.append(spikes[0])
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


def _align_burst_sequence_pair(sequence1, sequence2):
    '''align sequence2 to sequence1 using simple linear fit
    :returns alignment, residuals
    alignment: tuple (a, b) where a is slope and b is offset
    residuals: dict with (burst_id, residual) pairs; residual is squared difference of burst time after alignment
    '''
    seq1_times = []
    seq2_times = []
    burst_ids = sequence1.keys()
    common_burst_ids = []
    for burst_id in burst_ids:
        if burst_id in sequence2:
            common_burst_ids.append(burst_id)
            seq1_times.append(sequence1[burst_id])
            seq2_times.append(sequence2[burst_id])
    seq1_times = np.array(seq1_times)
    seq2_times = np.array(seq2_times)
    if len(seq1_times) < 2:
        return None, None
    # p_opt = seq2_times[0] - seq1_times[0]
    # p_opt_, p_cov = curve_fit(lambda x, a: a * x, seq2_times - seq1_times[0], seq1_times)
    # p_opt = p_opt_, - seq1_times[0] * p_opt_

    # Alignment to first burst for Dezhe
    # offset1 = np.min(seq1_times)
    # offset2 = np.min(seq2_times)
    # p_opt_, p_cov = curve_fit(lambda x, a: a * x, seq2_times - offset2, seq1_times - offset1)
    # p_opt = (p_opt_[0], offset1, offset2)
    # print 'offset1 = %.3f, p_opt[2] = %.3f, p_opt[0] = %.3f' % (offset1, p_opt[2], p_opt[0])

    p_opt, p_cov = curve_fit(lambda x, a, b: a * x + b, seq2_times, seq1_times)
    # p_opt, p_cov = curve_fit(lambda x, a, b, c: a * x * x + b * x + c, seq2_times, seq1_times)

    # seq2_times_aligned = seq2_times - p_opt
    # seq2_times_aligned = (seq2_times - seq1_times[0]) * p_opt
    seq2_times_aligned = p_opt[0] * seq2_times + p_opt[1]
    # seq2_times_aligned = p_opt[0] * seq2_times * seq2_times + p_opt[1] * seq2_times + p_opt[2]
    res = (seq1_times - seq2_times_aligned) * (seq1_times - seq2_times_aligned)
    residuals = {}
    for i, burst_id in enumerate(common_burst_ids):
        residuals[burst_id] = res[i]

    return p_opt, residuals


def _get_common_bursts(sequence1, sequence2):
    common_bursts = []
    for burst1 in sequence1:
        if burst1 in sequence2:
            common_bursts.append(burst1)

    return common_bursts


def _shuffle_syllable_times(eGUI_syllables, motif_finder_data):
    """
    take syllable times and move them around randomly; preserve motif length
    :param eGUI_syllables:
    :param motif_finder_data:
    :return: new eGUI syllables object with shuffled start/stop times
    """
    new_syllables = dict()
    n_motifs = len(motif_finder_data.start)

    for syllable in eGUI_syllables:
        onsets = []
        offsets = []
        valid_motifs = eGUI_syllables[syllable].motifs
        shift = np.random.rand()
        for i in range(n_motifs):
            if i in valid_motifs:
                # stupid hack for C23
                index_ = np.where(valid_motifs == i)[0]
                if len(index_) == 1: # I am too stupid to do proper indexing
                    index = index_[0]
                    tmp_onset = eGUI_syllables[syllable].onsets[index]
                    tmp_offset = eGUI_syllables[syllable].offsets[index]
                    motif_onset = motif_finder_data.start[i]
                    motif_offset = motif_finder_data.stop[i]
                    shift_range = -tmp_onset, motif_offset - motif_onset - tmp_offset
                    syllable_shift = shift_range[0] + shift * (shift_range[1] - shift_range[0])
                    onsets.append(tmp_onset + syllable_shift)
                    offsets.append(tmp_offset + syllable_shift)
                else: # C23
                    for index in index_:
                        tmp_onset = eGUI_syllables[syllable].onsets[index]
                        tmp_offset = eGUI_syllables[syllable].offsets[index]
                        motif_onset = motif_finder_data.start[i]
                        motif_offset = motif_finder_data.stop[i]
                        shift_range = -tmp_onset, motif_offset - motif_onset - tmp_offset
                        syllable_shift = shift_range[0] + shift * (shift_range[1] - shift_range[0])
                        onsets.append(tmp_onset + syllable_shift)
                        offsets.append(tmp_offset + syllable_shift)
        new_syllable = utils.Syllable(syllable, valid_motifs, np.array(onsets), np.array(offsets))
        new_syllables[syllable] = new_syllable

    return new_syllables


def burst_interval_scaling_per_trial(experiment_info_name):
    '''
    Alignment of all bursts in individual trials.
    Possible because they have been recorded simultaneously.
    '''
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cluster_bursts = common_data['bursts']
    cluster_bursts_proofread = common_data['bursts_proofread']
    proofread = False

    # TODO: get distribution of burst durations - our interval time scale should be some multiple of this

    n_motifs = len(motif_finder_data.start)
    # first, get all burst times in each trial aligned to motif onset
    # then collect all intervals in a dict
    # sorted by (burst 1 ID, burst 2 ID), pointing to a list with intervals
    min_interval = 0.02 # 20 ms - i.e., multiple of typical burst duration
    burst_intervals = {}
    for i, cluster_id1 in enumerate(clusters_of_interest):
        for j, cluster_id2 in enumerate(clusters_of_interest):
            # if i > 2:
            #     continue
            if j <= i:
                continue
            # fig = plt.figure(2 * i)
            burst_intervals[i, j] = {}
            if proofread:
                burst1_proofed = _clean_up_bursts(cluster_bursts_proofread[i])
                burst2_proofed = _clean_up_bursts(cluster_bursts_proofread[j])
            else:
                burst1 = _clean_up_bursts(cluster_bursts[i])
                burst2 = _clean_up_bursts(cluster_bursts[j])
            for k in range(n_motifs):
                if len(burst1[k]) > 1 and len(burst2[k]) > 1:
                    if proofread:
                        tmp1_burst = burst1_proofed[k]
                        tmp2_burst = burst2_proofed[k]
                    # not proofread
                    else:
                        tmp1_burst = burst1[k]
                        tmp2_burst = burst2[k]

                    # Vallentin & Long 2015: center point
                    # burst1_time = 0.5 * (tmp1_burst[-1] + tmp1_burst[0])
                    # burst2_time = 0.5 * (tmp2_burst[-1] + tmp2_burst[0])
                    # try out mean burst time (more robust?)
                    burst1_time = np.mean(tmp1_burst)
                    burst2_time = np.mean(tmp2_burst)
                    interval = abs(burst1_time - burst2_time)
                    if interval > min_interval:
                        burst_intervals[i, j][k] = interval

    # first, determine linearity
    fig1 = plt.figure(1)
    ax1_1 = plt.subplot(1, 2, 1)
    interval1_list = []
    interval2_list = []
    for burst_pair in burst_intervals:
        for n in burst_intervals[burst_pair]:
            interval1 = burst_intervals[burst_pair][n]
            for m in burst_intervals[burst_pair]:
                if n <= m:
                    continue
                interval2 = burst_intervals[burst_pair][m]
                interval1_list.append(interval1)
                interval2_list.append(interval2)
    interval1_list = np.array(interval1_list)
    interval2_list = np.array(interval2_list)
    ax1_1.plot(interval1_list, interval2_list, 'ko', markersize=0.5)
    ax1_1.set_xlabel('Interval duration i (s)')
    ax1_1.set_ylabel('Interval duration j (s)')

    def polynomial_2nd(x, a, b):
        return a * x + b * x * x
    p_opt, p_cov = curve_fit(polynomial_2nd, interval1_list, interval2_list)
    p_err = np.sqrt(np.diag(p_cov))
    ax1_1.plot(interval1_list, polynomial_2nd(interval1_list, p_opt[0], p_opt[1]), 'r-')
    fit_result_str = 'L = %.3f +- %.3f - NL = %.3f +- %.3f' % (p_opt[0], 2*p_err[0], p_opt[1], 2*p_err[1])
    ax1_1.set_title(fit_result_str)

    interval_diff = interval1_list - interval2_list
    ax1_2 = plt.subplot(1, 2, 2)
    ax1_2.set_xlabel('Interval difference (s)')
    ax1_2.hist(interval_diff, bins=20)

    print '%d intervals' % len(interval1_list)
    print 'mean interval: %.1f +- %.1f' % (1e3 * np.mean(interval1_list), 1e3 * np.std(interval1_list))

    # second, compute interval ratios
    eps = 1e-6
    outlier_intervals = set()
    outlier_ratiolist1 = []
    outlier_ratiolist2 = []
    fig2 = plt.figure(2)
    ax2_1 = plt.subplot(1, 3, 1)
    intervalratio1_list = []
    intervalratio2_list = []
    burst_pairs = burst_intervals.keys()
    for i in range(len(burst_pairs)):
        burst_trials_i = burst_intervals[burst_pairs[i]].keys()
        burst_trials_i.sort()
        for j in range(i + 1, len(burst_pairs)):
            burst_trials_j = burst_intervals[burst_pairs[j]].keys()
            for n in range(len(burst_trials_i)):
                for m in range(n + 1, len(burst_trials_i)):
                    if burst_trials_i[n] in burst_trials_j and burst_trials_i[m] in burst_trials_j:
                        # trial_i_n = burst_trials_i[n]
                        # trial_i_m = burst_trials_i[m]
                        # trial_j_n = burst_trials_j.index(trial_i_n)
                        # trial_j_m = burst_trials_j.index(trial_i_m)
                        trial_n = burst_trials_i[n]
                        trial_m = burst_trials_i[m]
                        # intervalratio1 = burst_intervals[burst_pairs[i]][trial_i_n] / \
                        #                  burst_intervals[burst_pairs[j]][trial_j_n]
                        # intervalratio2 = burst_intervals[burst_pairs[i]][trial_i_m] / \
                        #                  burst_intervals[burst_pairs[j]][trial_j_m]
                        # intervalratio1 = burst_intervals[burst_pairs[i]][trial_n] / \
                        #                  (burst_intervals[burst_pairs[j]][trial_n] + eps)
                        # intervalratio2 = burst_intervals[burst_pairs[i]][trial_m] / \
                        #                  (burst_intervals[burst_pairs[j]][trial_m] + eps)
                        interval_i_n = burst_intervals[burst_pairs[i]][trial_n]
                        interval_j_n = burst_intervals[burst_pairs[j]][trial_n]
                        interval_i_m = burst_intervals[burst_pairs[i]][trial_m]
                        interval_j_m = burst_intervals[burst_pairs[j]][trial_m]
                        if interval_i_n < interval_j_n:
                            intervalratio1 = interval_i_n / (interval_j_n + eps)
                            intervalratio2 = interval_i_m / (interval_j_m + eps)
                        else:
                            intervalratio1 = interval_j_n / (interval_i_n + eps)
                            intervalratio2 = interval_j_m / (interval_i_m + eps)
                        intervalratio1_list.append(intervalratio1)
                        intervalratio2_list.append(intervalratio2)
                        if intervalratio2 / intervalratio1 < 0.6 and intervalratio1 < 0.1:
                            outlier_intervals.add(burst_pairs[i])
                            outlier_intervals.add(burst_pairs[j])
                            outlier_ratiolist1.append(intervalratio1)
                            outlier_ratiolist2.append(intervalratio2)
    intervalratio1_list = np.array(intervalratio1_list)
    intervalratio2_list = np.array(intervalratio2_list)
    ax2_1.plot(intervalratio1_list, intervalratio2_list, 'ko', markersize=0.5)
    ax2_1.plot(outlier_ratiolist1, outlier_ratiolist2, 'ro', markersize=0.5)
    ax2_1.set_xlabel('Interval ratio trial i')
    ax2_1.set_ylabel('Interval ratio trial j')

    intervalratio_diff = 1 / np.sqrt(2) * (intervalratio1_list - intervalratio2_list)
    ax2_2 = plt.subplot(1, 3, 2)
    ax2_2.set_xlabel('Interval ratio difference')
    ax2_2.hist(intervalratio_diff, bins=100)
    intervalratio_ratio = intervalratio2_list / intervalratio1_list
    ax2_3 = plt.subplot(1, 3, 3)
    ax2_3.set_xlabel('Interval ratio ratio')
    ax2_3.hist(intervalratio_ratio, bins=100)

    print '%d interval ratios' % len(intervalratio1_list)

    ratio_diff_med = np.median(intervalratio_diff)
    ratio_diff_5th = np.percentile(intervalratio_diff, 5)
    ratio_diff_95th = np.percentile(intervalratio_diff, 95)
    print '%.3f median interval ratio diff (%.3f - %.3f)' % (ratio_diff_med, ratio_diff_5th, ratio_diff_95th)

    plt.show()


def pairwise_burst_distance_jitter(experiment_info_name):
    '''
    Implements analysis of jitter of inter-burst intervals as a function of interval duration
    (see Fig. 3 in Leonardo & Fee 2005)
    :param experiment_info_name: experiment info file
    :return: None
    '''
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cluster_bursts = common_data['bursts']

    # implement this similar to burst interval scaling, but now simply compute mean interval
    # and STD of interval and plot STD interval as function of mean interval
    n_motifs = len(motif_finder_data.start)
    burst_intervals = {}
    for i in range(len(cluster_bursts)):
        for j in range(i + 1, len(cluster_bursts)):
            tmp_intervals = []
            for k in range(n_motifs):
                burst1 = cluster_bursts[i][k]
                burst2 = cluster_bursts[j][k]
                if len(burst1) > 1 and len(burst2) > 1:
                    # Vallentin & Long 2015: center point
                    # burst1_time = 0.5 * (burst1[-1] + burst1[0])
                    # burst2_time = 0.5 * (burst2[-1] + burst2[0])
                    burst1_time = np.mean(burst1)
                    burst2_time = np.mean(burst2)
                    interval = abs(burst1_time - burst2_time)
                    tmp_intervals.append(interval)
            interval_mean, interval_std = 1e3 * np.mean(tmp_intervals), 1e3 * np.std(tmp_intervals) # in ms
            burst_intervals[i, j] = interval_mean, interval_std

    fig1 = plt.figure(1)
    ax1 = plt.subplot(1, 1, 1)
    for pair in burst_intervals:
        interval_mean, interval_std = burst_intervals[pair]
        ax1.plot(interval_mean, interval_std, 'ko', markersize=0.5)
    ax1.set_xlabel('Burst interval (ms)')
    ax1.set_ylabel('Interval variability (ms; no scaling)')
    plt.show()


def burst_sequence_alignment_per_trial(experiment_info_name):
    '''
    align entire sequences in each trial by (non-)linear fit
    also, as zero-order alternative, just aling by first spike
    Fit residuals are a direct measure of burst timing variability
    :param experiment_info_name: experiment info file
    :return: None
    '''
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cluster_bursts = common_data['bursts']
    cluster_celltypes = common_data['celltypes']

    # generate 'residual sequence' (i.e., data structure storing residuals for each burst)
    residual_sequence = []
    for i in range(len(cluster_bursts)):
        residual_sequence.append([])
    # generate sequence for each trial
    trial_sequences = []
    sequence_durations = []
    n_motifs = len(motif_finder_data.start)
    for trial_nr in range(n_motifs):
        sequence = {}
        tmp_seq = []
        for burst_id in range(len(cluster_bursts)):
            burst = cluster_bursts[burst_id][trial_nr]
            if len(burst):
                burst_time = burst[0]
                # burst_time = 0.5 * (burst[0] + burst[-1])
                # burst_time = np.mean(burst)
                sequence[burst_id] = burst_time
                tmp_seq.append(burst_time)
        trial_sequences.append(sequence)
        tmp_seq.sort()
        sequence_durations.append(tmp_seq[-1] - tmp_seq[0])

    # generate sequence ALIGNED TO FIRST SPIKE for each trial with first spike
    # trial_sequences = dict()
    # sequence_durations = []
    # n_motifs = len(motif_finder_data.start)
    # # find first burst
    # first_burst_id = []
    # for trial_nr in range(n_motifs):
    #     tmp_seq = []
    #     tmp_seq_burst_ids = []
    #     for burst_id in range(len(cluster_bursts)):
    #         burst = cluster_bursts[burst_id][trial_nr]
    #         if len(burst):
    #             # burst_time = burst[0]
    #             # burst_time = 0.5 * (burst[0] + burst[-1])
    #             burst_time = np.mean(burst)
    #             tmp_seq.append(burst_time)
    #             tmp_seq_burst_ids.append(burst_id)
    #     if trial_nr == 1:
    #         dummy = 1
    #     first_burst_id.append(tmp_seq_burst_ids[np.argmin(tmp_seq)])
    # first_burst = np.bincount(first_burst_id).argmax()
    #
    motif_durations = []
    # first_burst_trials = []
    for trial_nr in range(n_motifs):
        motif_duration = motif_finder_data.stop[trial_nr] - motif_finder_data.start[trial_nr]
        motif_durations.append(motif_duration)
    #     sequence = {}
    #     tmp_seq = []
    #     first_burst_in_seq = False
    #     for burst_id in range(len(cluster_bursts)):
    #         burst = cluster_bursts[burst_id][trial_nr]
    #         if len(burst):
    #             if burst_id == first_burst:
    #                 first_burst_in_seq = True
    #             # burst_time = burst[0]
    #             # burst_time = 0.5 * (burst[0] + burst[-1])
    #             burst_time = np.mean(burst)
    #             sequence[burst_id] = burst_time
    #             tmp_seq.append(burst_time)
    #     if first_burst_in_seq:
    #         # trial_sequences.append(sequence)
    #         trial_sequences[trial_nr] = sequence
    #         tmp_seq.sort()
    #         sequence_durations.append(tmp_seq[-1] - tmp_seq[0])
    #         first_burst_trials.append(trial_nr)

    # calculate all pairwise sequence alignments
    # for each pairwise alignment, add residuals for each burst to residual sequence
    sequence_alignments = {}
    for trial1_nr in range(n_motifs):
    # for trial1_nr in first_burst_trials:
        # for trial2_nr in range(trial1_nr + 1, n_motifs):
        for trial2_nr in range(n_motifs):
        # for trial2_nr in first_burst_trials:
            alignment, residuals = _align_burst_sequence_pair(trial_sequences[trial1_nr], trial_sequences[trial2_nr])
            sequence_alignments[trial1_nr, trial2_nr] = alignment
            if residuals is None:
                continue
            else:
                for burst_id in residuals:
                    res = residuals[burst_id]
                    residual_sequence[burst_id].append(res)

    # calculate residuals per burst and normalize by dof = n - 2 (linear regression: 2 parameters estimated)
    residuals = []
    for i, res_list in enumerate(residual_sequence):
        if res_list is None:
            continue
        if len(res_list) < 3:
            print 'Warning: less than three trials for residuals of burst %d; cannot use!' % i
            continue
        res = np.sqrt(np.sum(res_list) / (len(res_list) - 2.0))
        residuals.append(res)
    mean_residual = np.mean(residuals)
    print 'Mean residual = %.1f ms' % (1e3 * mean_residual)

    # plot all trials aligned to a reference trial
    # motif_durations = []
    # for i in range(n_motifs):
    #     motif_duration = motif_finder_data.stop[i] - motif_finder_data.start[i]
    #     motif_durations.append(motif_duration)
    mean_motif_duration = np.mean(motif_durations)
    ref_trial = None
    min_difference = 1e6
    for i in range(n_motifs):
    # for i in first_burst_trials:
        tmp_difference = abs(motif_durations[i] - mean_motif_duration)
        if tmp_difference < min_difference:
            min_difference = tmp_difference
            ref_trial = i

    # ref_trial = 0
    print 'Reference trial %d duration: %.0f ms; mean motif duration: %.0f ms' % (ref_trial, 1e3 * motif_durations[ref_trial],
                                                                                  1e3 * mean_motif_duration)
    # connect burst times in each trial with a transparent line, line density good visualization?
    # get sequence order from ref_trial
    ref_trial_burst_times = []
    ref_trial_burst_ids = np.array(trial_sequences[ref_trial].keys())
    ref_trial_burst_ids.sort()
    for burst_id in ref_trial_burst_ids:
        ref_trial_burst_times.append(trial_sequences[ref_trial][burst_id])
    burst_order = np.zeros(len(cluster_bursts)) - 1
    burst_order[:len(ref_trial_burst_ids)] = ref_trial_burst_ids[np.argsort(ref_trial_burst_times)]
    offset = 0
    all_burst_times = {}
    for burst_id in range(len(cluster_bursts)):
        all_burst_times[burst_id] = []
        if burst_id not in burst_order:
            burst_order[len(ref_trial_burst_ids) + offset] = burst_id
            offset += 1

    fig1 = plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    debug_plot = dict()
    sequence_ref_scales = []
    highlight_trials = {0: 'r|', 7: 'b|', 11: 'g|'} # alignment visualization for figure
    for i in range(n_motifs):
    # for i in first_burst_trials:
        # if i == ref_trial or i == 0:
        #     debug_plot[i] = []
        if i == ref_trial:
            # alignment = (1.0, 0.0)
            # alignment = (1.0, 0.0, 0.0)
            alignment = sequence_alignments[ref_trial, i]
            sequence_ref_scales.append(1.0)
        else:
            try:
                alignment_ = sequence_alignments[i, ref_trial]
                if alignment_ is None:
                    sequence_ref_scales.append(None)
                    continue
                alignment = []
                # we're inverting the fitted line
                alignment.append(1.0 / alignment_[0])
                alignment.append(-1.0 * alignment_[1] / alignment_[0])
                # for first burst alignment
                # alignment.append(1.0 / alignment_[0])
                # alignment.append(alignment_[1])
                # alignment.append(alignment_[2])
                sequence_ref_scales.append(alignment[0])
            except KeyError:
                alignment = sequence_alignments[ref_trial, i]
                if alignment is None:
                    sequence_ref_scales.append(None)
                    continue
                sequence_ref_scales.append(alignment[0])
        trial_sequence = trial_sequences[i]
        raw_times = []
        sequence_id = []
        for j, burst_id in enumerate(burst_order):
            try:
                raw_times.append(trial_sequence[burst_id])
                sequence_id.append(j)
            except KeyError:
                continue
        raw_times = np.array(raw_times)
        aligned_times = alignment[0] * raw_times + alignment[1]
        # aligned_times = alignment[0] * (raw_times - alignment[1])
        # aligned_times = alignment[0] * raw_times * raw_times + alignment[1] * raw_times + alignment[2]
        # keep track of all aligned burst times here for variability calculation later
        burst_count = 0
        for burst_id in burst_order:
            if burst_id in trial_sequence:
                all_burst_times[burst_id].append(aligned_times[burst_count])
                burst_count += 1
                # if i == ref_trial or i == 0:
                #     if burst_id in trial_sequences[ref_trial] and burst_id in trial_sequences[0]:
                #         debug_plot[i].append(aligned_times[burst_count - 1])
        # print '--------------------'
        # print 'Trial %d' % i
        # tmp_keys = trial_sequence.keys()
        # tmp_keys.sort()
        # print tmp_keys
        if i in highlight_trials:
            ax1.plot(aligned_times, sequence_id, highlight_trials[i], fillstyle='none', alpha=0.5)
        else:
            ax1.plot(aligned_times, sequence_id, 'k|', fillstyle='none', alpha=0.5)
    # ax1.plot(debug_plot[0], debug_plot[1], 'ko')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cell ID')

    mean_burst_times = []
    burst_variabilities = []
    aligned_burst_types = []
    for burst_id in all_burst_times:
        t_vec = all_burst_times[burst_id]
        if len(t_vec) > 1:
            mean_t_vec = np.mean(t_vec)
            rmse = np.sqrt(np.dot(t_vec - mean_t_vec, t_vec - mean_t_vec) / len(t_vec))
            mean_burst_times.append(mean_t_vec)
            burst_variabilities.append(rmse)
            aligned_burst_types.append(cluster_celltypes[burst_id])
    mean_burst_times = np.array(mean_burst_times)
    burst_variabilities = np.array(burst_variabilities)
    mean_aligned_variability = 1e3 * np.mean(burst_variabilities)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(mean_burst_times - np.min(mean_burst_times), 1e3 * burst_variabilities, 'ko')
    title_str = 'Mean aligned variability = %.1f ms' % mean_aligned_variability
    ax2.set_title(title_str)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Burst time variability (ms)')

    print 'Motif ref duration = %.3f ms' % (1e3 * motif_durations[ref_trial])
    print 'Mean burst time (ms)\tBurst time RMSE (ms)\tCell type'
    motif_offset = motif_finder_data.start[ref_trial]
    for i in range(len(mean_burst_times)):
        print '%.3f\t%.3f\t%s' % (1e3 * (mean_burst_times[i] - motif_offset), 1e3 * burst_variabilities[i],
                                  aligned_burst_types[i])

    # are there any that were not aligned?
    print 'Nr. of bursts: %d --- Nr. of aligned bursts: %d' % (len(cluster_bursts), len(mean_burst_times))

    # plot sequence scale factors vs. motif scale factors
    motif_ref_scales = []
    motif_ref_duration = motif_durations[ref_trial]
    print 'Trial nr.\tMotif scale\tSequence scale'
    for i in range(n_motifs):
    # for i in first_burst_trials:
        motif_duration = motif_durations[i]
        motif_scale = motif_ref_duration / motif_duration
        motif_ref_scales.append(motif_scale)
        if sequence_ref_scales[i] is None:
            continue
        print '%d\t%.3f\t%.3f' % (i, motif_scale, sequence_ref_scales[i])
        # if motif_scale < 1.0 and sequence_ref_scales[i] > 1.0:
        #     print 'Trial %d, motif scale = %.3f, sequence scale = %.3f' % (i, motif_scale, sequence_ref_scales[i])
    fig3 = plt.figure(3)
    # ax3 = plt.subplot(1, 2, 1)
    ax3 = plt.subplot(1, 1, 1)
    ax3.plot(sequence_ref_scales, motif_ref_scales, 'ko')
    ax3.set_xlabel('Sequence scale (a.u.)')
    ax3.set_ylabel('Motif scale (a.u.)')
    # ax3 = plt.subplot(1, 2, 2)
    # ax3.plot(1e3 * np.array(sequence_durations), 1e3 * np.array(motif_durations), 'ko')
    # ax3.set_xlabel('Sequence duration (ms)')
    # ax3.set_ylabel('Motif duration (ms)')

    plt.show()


def burst_sequence_syllable_alignment(experiment_info_name):
    '''
    align sequences in each trial in individual syllables by linear fit
    Fit residuals are a direct measure of burst timing variability
    :param experiment_info_name:
    :return:
    '''
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cluster_bursts = common_data['bursts']

    egui_syllables = utils.load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])

    # data structures for syllable-level sequences
    trial_syl_sequences = dict()
    for syl in egui_syllables:
        trial_syl_sequences[syl] = []
    n_motifs = len(motif_finder_data.start)
    for trial_nr in range(n_motifs):
        for syl in egui_syllables:
            sequence = {}
            for burst_id in range(len(cluster_bursts)):
                burst = cluster_bursts[burst_id][trial_nr]
                if len(burst):
                    burst_time = burst[0]
                    # burst_time = 0.5 * (burst[0] + burst[-1])
                    # burst_time = np.mean(burst)
                    burst_time_motif_aligned = burst_time - motif_finder_data.start[trial_nr]
                    burst_syl, burst_syl_time = utils.map_trial_time_to_trial_syllable(burst_time_motif_aligned,
                                                                                       trial_nr, egui_syllables)
                    if burst_syl is None or burst_syl != syl:
                        continue
                    sequence[burst_id] = burst_syl_time
            # print 'Added sequence %d of length %d for syllable %s' % (trial_nr, len(sequence), syl)
            tmp_burst_ids = sequence.keys()
            tmp_burst_ids.sort()
            # print tmp_burst_ids
            trial_syl_sequences[syl].append(sequence)

    # calculate all pairwise sequence alignments
    # for linear scaling, we need at least three bursts to estimate remaining variance
    syl_sequence_alignments = dict()
    alignments_per_trial = dict()
    for syl in egui_syllables:
        syl_sequence_alignments[syl] = dict()
        alignments_per_trial[syl] = dict()
        for trial1_nr in range(n_motifs):
            sequence1 = trial_syl_sequences[syl][trial1_nr]
            alignment_count = 0
            # for trial2_nr in range(trial1_nr + 1, n_motifs):
            for trial2_nr in range(n_motifs):
                sequence2 = trial_syl_sequences[syl][trial2_nr]
                common_bursts = _get_common_bursts(sequence1, sequence2)
                if len(common_bursts) < 3:
                    syl_sequence_alignments[syl][trial1_nr, trial2_nr] = None
                    continue
                alignment, residuals = _align_burst_sequence_pair(sequence1, sequence2)
                syl_sequence_alignments[syl][trial1_nr, trial2_nr] = alignment
                alignment_count += 1
                # if residuals is None:
                #     continue
                # else:
                #     for burst_id in residuals:
                #         res = residuals[burst_id]
                #         residual_sequence[burst_id].append(res)
            alignments_per_trial[syl][trial1_nr] = alignment_count

    motif_durations = []
    for i in range(len(motif_finder_data.start)):
        motif_durations.append(motif_finder_data.stop[i] - motif_finder_data.start[i])
    mean_motif_duration = np.mean(motif_durations)
    # here, we do the reference trial differently: we make sure we have valid alignments
    syl_ref_trial = dict()
    for syl in egui_syllables:
        tmp_trial = 0
        tmp_trial_count = 0
        for trial1_nr in range(n_motifs):
            if alignments_per_trial[syl][trial1_nr] > tmp_trial_count:
                tmp_trial_count = alignments_per_trial[syl][trial1_nr]
                tmp_trial = trial1_nr
        syl_ref_trial[syl] = tmp_trial
        print 'Syllable %s ref trial: %d' % (syl, tmp_trial)


    ref_trial = 0
    min_difference = abs(motif_durations[ref_trial] - mean_motif_duration)
    for i in range(1, n_motifs):
        tmp_difference = abs(motif_durations[i] - mean_motif_duration)
        if tmp_difference < min_difference:
            min_difference = tmp_difference
            ref_trial = i
    # for C23: choose ref trial = 2
    # ref_trial = 2

    print 'Reference trial %d duration: %.0f ms; mean motif duration: %.0f ms' % (ref_trial, 1e3 * motif_durations[ref_trial],
                                                                                  1e3 * mean_motif_duration)
    # don't choose any particular order for now
    burst_order = np.array(range(len(cluster_bursts)))

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    syllables = egui_syllables.keys()
    syllables.sort()
    nr_syl = len(syllables)
    for n, syl in enumerate(syllables):
        ax1 = fig1.add_subplot(2, nr_syl, n + 1)
        ax3 = fig2.add_subplot(1, nr_syl, n + 1)
        sequence_ref_scales = []
        all_burst_times = {}
        for burst_id in range(len(cluster_bursts)):
            all_burst_times[burst_id] = []
        for i in range(n_motifs):
            # if i == ref_trial or i == 0:
            #     debug_plot[i] = []
            if i == syl_ref_trial[syl]:
                alignment = (1.0, 0.0)
                # alignment = (0.0, 1.0, 0.0)
                sequence_ref_scales.append(1.0)
            else:
                try:
                    alignment_ = syl_sequence_alignments[syl][i, syl_ref_trial[syl]]
                    if alignment_ is None:
                        continue
                    alignment = []
                    # we're inverting the fitted line
                    alignment.append(1.0 / alignment_[0])
                    alignment.append(-1.0 * alignment_[1] / alignment_[0])
                    sequence_ref_scales.append(alignment[0])
                except KeyError:
                    alignment = syl_sequence_alignments[syl][syl_ref_trial[syl], i]
                    if alignment is None:
                        sequence_ref_scales.append(None)
                        continue
                    sequence_ref_scales.append(alignment[0])
            trial_sequence = trial_syl_sequences[syl][i]
            raw_times = []
            sequence_id = []
            for j, burst_id in enumerate(burst_order):
                try:
                    raw_times.append(trial_sequence[burst_id])
                    sequence_id.append(j)
                except KeyError:
                    continue
            raw_times = np.array(raw_times)
            aligned_times = alignment[0] * raw_times + alignment[1]
            # aligned_times = alignment[0] * raw_times * raw_times + alignment[1] * raw_times + alignment[2]
            # keep track of all aligned burst times here for variability calculation later
            burst_count = 0
            for burst_id in burst_order:
                if burst_id in trial_sequence:
                    all_burst_times[burst_id].append(aligned_times[burst_count])
                    burst_count += 1
            ax1.plot(aligned_times, sequence_id, 'ko', fillstyle='none', alpha=0.5)
            ax3.plot(aligned_times, [i + 1] * len(aligned_times), 'k|')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cell ID')
        title_str = 'Syllable %s' % syl
        ax1.set_title(title_str)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Trial')
        ax3.set_title(title_str)

        # now plot variability of aligned bursts in this syllable
        ax2 = fig1.add_subplot(2, nr_syl, n + nr_syl + 1)
        mean_burst_times = []
        burst_variabilities = []
        for burst_id in all_burst_times:
            t_vec = np.array(all_burst_times[burst_id])
            if len(t_vec) > 1:
                mean_t_vec = np.mean(t_vec)
                rmse = np.sqrt(np.dot(t_vec - mean_t_vec, t_vec - mean_t_vec) / len(t_vec))
                mean_burst_times.append(mean_t_vec)
                burst_variabilities.append(rmse)
        if len(mean_burst_times):
            mean_burst_times = np.array(mean_burst_times)
            burst_variabilities = np.array(burst_variabilities)
            mean_aligned_variability = 1e3 * np.mean(burst_variabilities)
            ax2.plot(mean_burst_times - np.min(mean_burst_times), 1e3 * burst_variabilities, 'ko')
            title_str = 'Mean aligned variability = %.1f ms' % mean_aligned_variability
            ax2.set_title(title_str)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Burst time variability (ms)')

        syl_index = np.where(egui_syllables[syl].motifs == syl_ref_trial[syl])[0]
        if not len(syl_index):
            syl_index = 0
        ref_syl_duration = egui_syllables[syl].offsets[syl_index] - egui_syllables[syl].onsets[syl_index]
        print 'Syllable %s ref duration = %.3f ms' % (syl, ref_syl_duration)
        # print 'Syllable %s ref duration = %.3f ms' % (syl, ref_syl_duration[1]) # C23
        print 'Mean burst time (ms)\tBurst time RMSE (ms)'
        for i in range(len(mean_burst_times)):
            print '%.3f\t%.3f' % (1e3 * mean_burst_times[i], 1e3 * burst_variabilities[i])

        # # for C22: visualize aligned burst onset time histogram for last syllable
        # if syl == u'e':
        #     all_burst_times_ = []
        #     for burst_id in all_burst_times:
        #         t_vec = np.array(all_burst_times[burst_id])
        #         all_burst_times_.extend(t_vec)
        #     fig3 = plt.figure(111)
        #     ax111 = fig3.add_subplot(1, 1, 1)
        #     bin_size = 0.5e-3 # 0.2 ms
        #     bins = np.arange(np.min(all_burst_times_) - bin_size, np.max(all_burst_times_) + bin_size, bin_size)
        #     hist, _ = np.histogram(all_burst_times_, bins=bins)
        #     ax111.step(bins[1:], hist)

    plt.show()


def burst_sequence_syllable_alignment_shuffle(experiment_info_name):
    '''
    align sequences in each trial in individual syllables by linear fit
    Fit residuals are a direct measure of burst timing variability
    :param experiment_info_name:
    :return:
    '''
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cluster_bursts = common_data['bursts']

    egui_syllables = utils.load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])

    shuffled_burst_times = dict()
    shuffled_burst_variabilities = dict()
    n_shuffle = 10000
    for shuffle in range(n_shuffle):
        shuffled_syllables = _shuffle_syllable_times(egui_syllables, motif_finder_data)
        shuffled_burst_times[shuffle] = []
        shuffled_burst_variabilities[shuffle] = []
        # data structures for syllable-level sequences
        trial_syl_sequences = dict()
        for syl in shuffled_syllables:
            trial_syl_sequences[syl] = []
        n_motifs = len(motif_finder_data.start)
        for trial_nr in range(n_motifs):
            for syl in shuffled_syllables:
                sequence = {}
                for burst_id in range(len(cluster_bursts)):
                    burst = cluster_bursts[burst_id][trial_nr]
                    if len(burst):
                        burst_time = burst[0]
                        # burst_time = 0.5 * (burst[0] + burst[-1])
                        # burst_time = np.mean(burst)
                        burst_time_motif_aligned = burst_time - motif_finder_data.start[trial_nr]
                        burst_syl, burst_syl_time = utils.map_trial_time_to_trial_syllable(burst_time_motif_aligned,
                                                                                           trial_nr, shuffled_syllables)
                        if burst_syl is None or burst_syl != syl:
                            continue
                        sequence[burst_id] = burst_syl_time
                # print 'Added sequence %d of length %d for syllable %s' % (trial_nr, len(sequence), syl)
                tmp_burst_ids = sequence.keys()
                tmp_burst_ids.sort()
                # print tmp_burst_ids
                trial_syl_sequences[syl].append(sequence)

        # calculate all pairwise sequence alignments
        # for linear scaling, we need at least three bursts to estimate remaining variance
        syl_sequence_alignments = dict()
        alignments_per_trial = dict()
        for syl in shuffled_syllables:
            syl_sequence_alignments[syl] = dict()
            alignments_per_trial[syl] = dict()
            for trial1_nr in range(n_motifs):
                sequence1 = trial_syl_sequences[syl][trial1_nr]
                alignment_count = 0
                # for trial2_nr in range(trial1_nr + 1, n_motifs):
                for trial2_nr in range(n_motifs):
                    sequence2 = trial_syl_sequences[syl][trial2_nr]
                    common_bursts = _get_common_bursts(sequence1, sequence2)
                    if len(common_bursts) < 3:
                        syl_sequence_alignments[syl][trial1_nr, trial2_nr] = None
                        continue
                    alignment, residuals = _align_burst_sequence_pair(sequence1, sequence2)
                    syl_sequence_alignments[syl][trial1_nr, trial2_nr] = alignment
                    alignment_count += 1
                    # if residuals is None:
                    #     continue
                    # else:
                    #     for burst_id in residuals:
                    #         res = residuals[burst_id]
                    #         residual_sequence[burst_id].append(res)
                alignments_per_trial[syl][trial1_nr] = alignment_count

        motif_durations = []
        for i in range(len(motif_finder_data.start)):
            motif_durations.append(motif_finder_data.stop[i] - motif_finder_data.start[i])
        mean_motif_duration = np.mean(motif_durations)
        # here, we do the reference trial differently: we make sure we have valid alignments
        syl_ref_trial = dict()
        for syl in shuffled_syllables:
            tmp_trial = 0
            tmp_trial_count = 0
            for trial1_nr in range(n_motifs):
                if alignments_per_trial[syl][trial1_nr] > tmp_trial_count:
                    tmp_trial_count = alignments_per_trial[syl][trial1_nr]
                    tmp_trial = trial1_nr
            syl_ref_trial[syl] = tmp_trial
            print 'Syllable %s ref trial: %d' % (syl, tmp_trial)

        ref_trial = 0
        min_difference = abs(motif_durations[ref_trial] - mean_motif_duration)
        for i in range(1, n_motifs):
            tmp_difference = abs(motif_durations[i] - mean_motif_duration)
            if tmp_difference < min_difference:
                min_difference = tmp_difference
                ref_trial = i
        # for C23: choose ref trial = 2
        # ref_trial = 2

        print 'Reference trial %d duration: %.0f ms; mean motif duration: %.0f ms' % (ref_trial, 1e3 * motif_durations[ref_trial],
                                                                                      1e3 * mean_motif_duration)
        # don't choose any particular order for now
        burst_order = np.array(range(len(cluster_bursts)))

        # fig1 = plt.figure(1)
        # fig2 = plt.figure(2)
        syllables = shuffled_syllables.keys()
        syllables.sort()
        nr_syl = len(syllables)
        for n, syl in enumerate(syllables):
            # ax1 = fig1.add_subplot(2, nr_syl, n + 1)
            sequence_ref_scales = []
            all_burst_times = {}
            for burst_id in range(len(cluster_bursts)):
                all_burst_times[burst_id] = []
            for i in range(n_motifs):
                # if i == ref_trial or i == 0:
                #     debug_plot[i] = []
                if i == syl_ref_trial[syl]:
                    alignment = (1.0, 0.0)
                    # alignment = (0.0, 1.0, 0.0)
                    sequence_ref_scales.append(1.0)
                else:
                    try:
                        alignment_ = syl_sequence_alignments[syl][i, syl_ref_trial[syl]]
                        if alignment_ is None:
                            continue
                        alignment = []
                        # we're inverting the fitted line
                        alignment.append(1.0 / alignment_[0])
                        alignment.append(-1.0 * alignment_[1] / alignment_[0])
                        sequence_ref_scales.append(alignment[0])
                    except KeyError:
                        alignment = syl_sequence_alignments[syl][syl_ref_trial[syl], i]
                        if alignment is None:
                            sequence_ref_scales.append(None)
                            continue
                        sequence_ref_scales.append(alignment[0])
                trial_sequence = trial_syl_sequences[syl][i]
                raw_times = []
                sequence_id = []
                for j, burst_id in enumerate(burst_order):
                    try:
                        raw_times.append(trial_sequence[burst_id])
                        sequence_id.append(j)
                    except KeyError:
                        continue
                raw_times = np.array(raw_times)
                aligned_times = alignment[0] * raw_times + alignment[1]
                # aligned_times = alignment[0] * raw_times * raw_times + alignment[1] * raw_times + alignment[2]
                # keep track of all aligned burst times here for variability calculation later
                burst_count = 0
                for burst_id in burst_order:
                    if burst_id in trial_sequence:
                        all_burst_times[burst_id].append(aligned_times[burst_count])
                        burst_count += 1
                # ax1.plot(aligned_times, sequence_id, 'ko', fillstyle='none', alpha=0.5)
            # ax1.set_xlabel('Time (s)')
            # ax1.set_ylabel('Cell ID')
            # title_str = 'Syllable %s' % syl
            # ax1.set_title(title_str)

            # now plot variability of aligned bursts in this syllable
            # ax2 = fig1.add_subplot(2, nr_syl, n + nr_syl + 1)
            mean_burst_times = []
            burst_variabilities = []
            for burst_id in all_burst_times:
                t_vec = np.array(all_burst_times[burst_id])
                if len(t_vec) > 1:
                    mean_t_vec = np.mean(t_vec)
                    rmse = np.sqrt(np.dot(t_vec - mean_t_vec, t_vec - mean_t_vec) / len(t_vec))
                    mean_burst_times.append(mean_t_vec)
                    burst_variabilities.append(rmse)
            if len(mean_burst_times):
                mean_burst_times = np.array(mean_burst_times)
                burst_variabilities = np.array(burst_variabilities)
                mean_aligned_variability = 1e3 * np.mean(burst_variabilities)
                shuffled_burst_times[shuffle].extend(1e3 * (mean_burst_times - np.min(mean_burst_times)))
                shuffled_burst_variabilities[shuffle].extend(1e3 * burst_variabilities)
                # ax2.plot(mean_burst_times - np.min(mean_burst_times), 1e3 * burst_variabilities, 'ko')
                # title_str = 'Mean aligned variability = %.1f ms' % mean_aligned_variability
                # ax2.set_title(title_str)
            # ax2.set_xlabel('Time (s)')
            # ax2.set_ylabel('Burst time variability (ms)')

            # syl_index = np.where(shuffled_syllables[syl].motifs == syl_ref_trial[syl])[0]
            # if not len(syl_index):
            #     syl_index = 0
            # ref_syl_duration = shuffled_syllables[syl].offsets[syl_index] - shuffled_syllables[syl].onsets[syl_index]
            # print 'Syllable %s ref duration = %.3f ms' % (syl, ref_syl_duration[1])
            # print 'Mean burst time (ms)\tBurst time RMSE (ms)'
            # for i in range(len(mean_burst_times)):
            #     print '%.3f\t%.3f' % (1e3 * mean_burst_times[i], 1e3 * burst_variabilities[i])

    # plt.show()
    out_name = os.path.join(experiment_info['Motifs']['DataBasePath'], 'shuffled_syllable_alignment.csv')
    with open(out_name, 'w') as out_file:
        header = 'Mean burst time (ms),Burst time RMSE (ms),Shuffle ID\n'
        out_file.write(header)
        for shuffle_id in shuffled_burst_times:
            times = shuffled_burst_times[shuffle_id]
            vars = shuffled_burst_variabilities[shuffle_id]
            for t, v in zip(times, vars):
                line = '%.2f,%.2f,%d\n' % (t, v, shuffle_id)
                out_file.write(line)


def burst_sequence_visualization(experiment_info_name):
    '''
    visualize entire sequences in each trial by aligning to first burst
    Fit residuals are a direct measure of burst timing variability
    :param experiment_info_name: experiment info file
    :return: None
    '''
    common_data = _load_common_data(experiment_info_name)
    motif_finder_data = common_data['motif']
    cluster_bursts = common_data['bursts']

    # generate sequence for each trial
    n_motifs = len(motif_finder_data.start)
    # trial_sequences_array = np.nan * np.ones((n_motifs, len(cluster_bursts) - 1)) # HACK for C23 - fix
    trial_sequences_array = np.nan * np.ones((n_motifs, len(cluster_bursts))) # HACK for C23 - fix
    trial_sequences = []
    sequence_durations = []
    motif_durations = []
    for trial_nr in range(trial_sequences_array.shape[0]):
        sequence = {}
        tmp_seq = []
        # for burst_id in range(1, len(cluster_bursts)): # HACK for C23 - fix
        for burst_id in range(len(cluster_bursts)): # HACK for C23 - fix
            burst = cluster_bursts[burst_id][trial_nr]
            if len(burst):
                # burst_time = burst[0]
                # burst_time = 0.5 * (burst[0] + burst[-1])
                burst_time = np.mean(burst)
                sequence[burst_id] = burst_time
                tmp_seq.append(burst_time)
                # trial_sequences_array[trial_nr, burst_id-1] = burst_time # HACK for C23 - fix
                trial_sequences_array[trial_nr, burst_id] = burst_time # HACK for C23 - fix
        trial_sequences.append(sequence)
        tmp_seq.sort()

    trial_sequences_shifted = []
    for trial_nr in range(trial_sequences_array.shape[0]):
        if trial_sequences_array[trial_nr, 0] > 0:
            offset = trial_sequences_array[trial_nr, 0]
            trial_sequences_shifted.append(trial_sequences_array[trial_nr, :] - offset)
    trial_sequences_shifted = np.array(trial_sequences_shifted)
    mean_sequence = np.nanmean(trial_sequences_shifted, axis=0)
    sequence_order = np.argsort(mean_sequence)
    mean_sequence = mean_sequence[sequence_order]
    mean_sequence -= mean_sequence[0]
    trial_sequences_shifted = trial_sequences_shifted[:, sequence_order]
    burst_sequences_plotted = np.isfinite(trial_sequences_shifted[:, 0])
    trial_sequences_shifted = trial_sequences_shifted[burst_sequences_plotted, :]
    trial_sequences_shifted = (trial_sequences_shifted.transpose() - trial_sequences_shifted[:, 0]).transpose()
    mean_sequence = np.nanmean(trial_sequences_shifted, axis=0)

    # plot deviations from mean sequence as in Glaze & Troyer, 2006
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    for i in range(trial_sequences_shifted.shape[0]):
        deviation_sequence = 1e3 * (trial_sequences_shifted[i, :] - mean_sequence)
        sel = np.isfinite(deviation_sequence)
        ax1.plot(1e3 * mean_sequence[sel], deviation_sequence[sel], 'ko-')
    ax1.set_xlabel('Mean sequence time (ms)')
    ax1.set_ylabel('Deviation (ms)')

    deviation_sequences = 1e3 * (trial_sequences_shifted - mean_sequence)
    var_sequence = np.nanstd(deviation_sequences, axis=0)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(1e3 * mean_sequence, var_sequence, 'ko-')
    ax2.set_xlabel('Mean sequence time (ms)')
    ax2.set_ylabel('Timing variability (ms)')

    # plot relationship between sequence duration and motif duration
    # C22: first burst - really from previous motif; last two bursts: outside annotated motif (within premotor delay,
    # but this gets complicated...)
    sequence_durations.append(tmp_seq[-3] - tmp_seq[1])
    motif_durations.append(motif_finder_data.stop[trial_nr] - motif_finder_data.start[trial_nr])
    sequence_durations = 1e3 * np.array(sequence_durations)
    motif_durations = 1e3 * np.array(motif_durations)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(sequence_durations, motif_durations, 'ko')
    ax3.set_xlabel('Sequence duration (ms)')
    ax3.set_ylabel('Motif duration (ms)')

    # # also plot syllable durations, sorted by overall length,
    # # and syllable deviation plots (as in Glaze & Troyer, 2006),
    # # as well as relationship between burst SD slopes
    # # and syllable SD slopes
    # with open(experiment_info_name, 'r') as data_file:
    #     experiment_info = ast.literal_eval(data_file.read())
    # egui_syllables = utils.load_syllables_from_egui(experiment_info['Motifs']['eGUIFilename'])
    # syllable_labels = egui_syllables.keys()
    # syllable_labels.sort()
    # syllable_sequences_array = np.nan * np.ones((n_motifs, 2 * len(egui_syllables))) # store syllable on-/offset
    # for i, label in enumerate(syllable_labels):
    #     syllable = egui_syllables[label]
    #     for trial_nr in range(n_motifs):
    #         if trial_nr not in syllable.motifs:
    #             continue
    #         trial_index = np.where(syllable.motifs == trial_nr)[0]
    #         syllable_sequences_array[trial_nr, 2 * i] = syllable.onsets[trial_index]
    #         syllable_sequences_array[trial_nr, 2 * i + 1] = syllable.offsets[trial_index]
    #
    # good_syllable_trials = np.isfinite(np.sum(syllable_sequences_array, axis=1))
    # combined_trials = good_syllable_trials * burst_sequences_plotted
    # # print combined_trials
    # plot_syllable_sequences = syllable_sequences_array[good_syllable_trials, :]
    # # align all first syllable onsets
    # for i in range(plot_syllable_sequences.shape[0]):
    #     plot_syllable_sequences[i, :] -= plot_syllable_sequences[i, 0]
    # mean_syllable_sequence = np.mean(plot_syllable_sequences, axis=0)
    # fig3 = plt.figure(3)
    # ax3_1 = fig3.add_subplot(1, 2, 1)
    # ax3_2 = fig3.add_subplot(1, 2, 2)
    # for i in range(plot_syllable_sequences.shape[0]):
    #     ax3_1.vlines(plot_syllable_sequences[i], i - 0.5, i + 0.5, 'k')
    #     deviation_sequence = 1e3 * (plot_syllable_sequences[i, :] - mean_syllable_sequence)
    #     sel = np.isfinite(deviation_sequence)
    #     ax3_2.plot(1e3 * mean_syllable_sequence[sel], deviation_sequence[sel], 'ko-')

    plt.show()


if __name__ == '__main__':
    # if len(sys.argv) == 2:
    if len(sys.argv) == 1:
        # info_name = sys.argv[1]
        valid_bird = False
        while not valid_bird:
            bird_id = raw_input('Please enter a bird ID (C21-25): ')
            try:
                # clusters_of_interest, burst_ids = bird_bursts[bird_id]
                info_name = bird_info[bird_id]
                clusters_of_interest, burst_ids, celltypes = utils.load_burst_info(info_name)
                valid_bird = True
            except KeyError:
                print 'Please enter a valid bird ID (C21-25)'
        assert len(clusters_of_interest) == len(burst_ids)

        # burst_interval_scaling_per_trial(info_name)
        burst_sequence_alignment_per_trial(info_name)
        # burst_sequence_syllable_alignment(info_name)
        # burst_sequence_syllable_alignment_shuffle(info_name)
        # pairwise_burst_distance_jitter(info_name)
        # burst_sequence_visualization(info_name)