import os
import ast
import sys
import cPickle
import numpy as np
import scipy.io, scipy.signal
from scipy.optimize import curve_fit
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


def _clean_up_bursts(bursts):
    # ugh I should have taken care of this during burst sorting... remove FP at ISIs <= 1 ms
    # also only keep all spikes with ISIs < 5 ms
    clean_bursts = []
    trials = len(bursts)
    for trial in range(trials):
        spikes = bursts[trial]
        if len(spikes[0]) < 2:
            clean_bursts.append([])
            continue
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
            if tmp[i + 1] - tmp[i] < 0.005:
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
    p_opt, p_cov = curve_fit(lambda x, a, b: a * x + b, seq2_times, seq1_times)
    # p_opt, p_cov = curve_fit(lambda x, a, b, c: a * x * x + b * x + c, seq2_times, seq1_times)

    seq2_times_aligned = p_opt[0] * seq2_times + p_opt[1]
    # seq2_times_aligned = p_opt[0] * seq2_times * seq2_times + p_opt[1] * seq2_times + p_opt[2]
    res = (seq1_times - seq2_times_aligned) * (seq1_times - seq2_times_aligned)
    residuals = {}
    for i, burst_id in enumerate(common_burst_ids):
        residuals[burst_id] = res[i]

    return p_opt, residuals


def burst_interval_scaling_per_trial(experiment_info_name):
    '''
    Alignment of all bursts in individual trials.
    Possible because they have been recorded simultaneously.
    '''
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


def burst_sequence_alignment_per_trial(experiment_info_name):
    '''
    align entire sequences in each trial by (non-)linear fit
    Fit residuals are a direct measure of burst timing variability
    :param experiment_info_name: experiment info file
    :return: None
    '''
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
        cluster_bursts.append(_clean_up_bursts(tmp_bursts[burst_ids[i]]))
        # proofread
        if proofread:
            cluster_bursts_proofread.append(_clean_up_bursts(tmp_bursts_proofed))

    # generate 'residual sequence' (i.e., data structure storing residuals for each burst)
    residual_sequence = []
    for i in range(len(cluster_bursts)):
        residual_sequence.append([])
    # generate sequence for each trial
    trial_sequences = []
    n_motifs = len(motif_finder_data.start)
    for trial_nr in range(n_motifs):
        sequence = {}
        for burst_id in range(len(cluster_bursts)):
            burst = cluster_bursts[burst_id][trial_nr]
            if len(burst):
                burst_time = 0.5 * (burst[0] + burst[-1])
                # burst_time = np.mean(burst)
                sequence[burst_id] = burst_time
        trial_sequences.append(sequence)

    # calculate all pairwise sequence alignments
    # for each pairwise alignment, add residuals for each burst to residual sequence
    sequence_alignments = {}
    for trial1_nr in range(n_motifs):
        for trial2_nr in range(trial1_nr + 1, n_motifs):
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
    # connect burst times in each trial with a transparent line, line density good visualization?
    ref_trial = 0
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
    for i in range(n_motifs):
        if i == ref_trial:
            alignment = (1.0, 0.0)
            # alignment = (0.0, 1.0, 0.0)
        else:
            try:
                alignment_ = sequence_alignments[i, ref_trial]
                if alignment is None:
                    continue
                alignment = []
                alignment[0], alignment[1] = 1.0 / alignment_[0], -1.0 * alignment[1] / alignment_[0]
            except KeyError:
                alignment = sequence_alignments[ref_trial, i]
                if alignment is None:
                    continue
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
        # aligned_times = alignment[0] * raw_times * raw_times + alignment[1] * raw_times + alignment[2]
        # keep track of all aligned burst times here for variability calculation later
        burst_count = 0
        for burst_id in burst_order:
            if burst_id in trial_sequence:
                all_burst_times[burst_id].append(aligned_times[burst_count])
                burst_count += 1
        ax1.plot(aligned_times, sequence_id, 'ko', fillstyle='none', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cell ID')

    mean_burst_times = []
    burst_variabilities = []
    for burst_id in all_burst_times:
        t_vec = all_burst_times[burst_id]
        if len(t_vec) > 1:
            mean_burst_times.append(np.mean(all_burst_times[burst_id]))
            burst_variabilities.append(np.std(all_burst_times[burst_id]))
    mean_burst_times = np.array(mean_burst_times)
    burst_variabilities = np.array(burst_variabilities)
    mean_aligned_variability = 1e3 * np.mean(burst_variabilities)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(mean_burst_times - mean_burst_times[0], 1e3 * burst_variabilities, 'ko')
    title_str = 'Mean aligned variability = %.1f ms' % mean_aligned_variability
    ax2.set_title(title_str)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Burst time variability (ms)')

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        info_name = sys.argv[1]
        # burst_interval_scaling_per_trial(info_name)
        burst_sequence_alignment_per_trial(info_name)