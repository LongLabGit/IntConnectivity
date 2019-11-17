import sys, os, ast, cPickle
import matplotlib.pyplot as plt
import numpy as np
import neo, elephant
import scipy.signal, scipy.stats
import ConnectionAnalyzer as ca
import ClusterProcessing as clust
import quantities as pq

clustering_src_folder = 'E:\\User\\project_src\\physiology\\Clustering'


def _load_snippets(experiment_info, cluster_id, snippet_identifier=''):
    if snippet_identifier:
        ST_pickle_name = 'STSnippets_' + snippet_identifier + '_Cluster_' + str(cluster_id) + '.pkl'
    else:
        ST_pickle_name = 'STSnippets_Cluster_%d.pkl' % cluster_id
    ST_snippetName = os.path.join(experiment_info['STA']['DataBasePathI'], ST_pickle_name)
    with open(ST_snippetName, 'rb') as ST_snippet_file:
        cluster_snippets = cPickle.load(ST_snippet_file)

    return cluster_snippets


def _individual_currents_from_snippets(experiment_info, cluster_snippets):

    t_offset = 0.01 # s
    t_baseline1 = -1.2 # ms
    t_baseline2 = -0.2 # ms
    t_peak1 = 1.0 # ms
    t_peak2 = 10.0 # ms
    peak_window_time = 1.0 # ms
    fs = experiment_info['WC']['SamplingRate']
    offset_bin = int(t_offset*fs)
    amp_baseline = (int((t_baseline1/1000.0 + t_offset)*fs),  int((t_baseline2/1000.0 + t_offset)*fs))
    amp_peak1 = int((t_peak1/1000.0 + t_offset)*fs)
    amp_peak2 = int((t_peak2/1000.0 + t_offset)*fs)

    # determine peak window by searching for peak of STA in 10 ms post-spike
    sta = np.mean(cluster_snippets.snippets, axis=0)
    sta_peak_bin = np.argmax(sta[offset_bin:amp_peak2]) + offset_bin
    peak_window = (sta_peak_bin - int(0.5*peak_window_time/1000.0*fs), sta_peak_bin + int(0.5*peak_window_time/1000.0*fs))

    print 'Determining amplitudes in %.1f ms window around STA peak at %.1f ms post-spike' % (peak_window_time,
                                                                                             sta_peak_bin/fs*1000.0 - 10.0)

    amplitudes = []
    for i, snippet_ in enumerate(cluster_snippets.snippets):
        try:
            snippet = snippet_.flatten().magnitude
        except:
            snippet = snippet_.flatten()
        amp = np.median(snippet[peak_window[0]:peak_window[1]]) - np.median(snippet[amp_baseline[0]:amp_baseline[1]])
        amplitudes.append(amp)

    return np.array(amplitudes)


def _estimate_population_firing(clusters):
    all_spike_times = []
    clustGroup = clust.ClusterGroup(clusters)
    for clusterID in clustGroup.clusters:
        cluster = clustGroup.clusters[clusterID]
        all_spike_times.extend(cluster.spiketrains[0].magnitude)

    all_spike_times = np.array(all_spike_times)*1000.0 # in ms
    t_min = np.min(all_spike_times)
    t_max = np.max(all_spike_times)
    bin_size = 10.0 # 10 ms resolution
    bins = np.arange(t_min, t_max + bin_size, bin_size)
    fr, _ = np.histogram(all_spike_times, bins)
    return fr


def _sort_spikes_by_population_rate(snippets, population_rate):
    # population_rate is a vector with 10 ms resolution
    spike_bins = [int(t*100) for t in snippets.snippet_spike_times]
    spike_population_rates = population_rate[spike_bins]
    sorted_spike_time_indices = np.argsort(spike_population_rates)
    return sorted_spike_time_indices, spike_population_rates[np.argsort(spike_population_rates)]


def individual_spike_triggered_currents(experiment_info_name, cluster_id):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    control_snippets = _load_snippets(experiment_info, cluster_id)
    shuffled_snippets = _load_snippets(experiment_info, cluster_id, 'shuffled_1x')

    control_amplitudes = _individual_currents_from_snippets(experiment_info, control_snippets)
    shuffled_amplitudes = _individual_currents_from_snippets(experiment_info, shuffled_snippets)

    all_amplitudes = np.zeros(len(control_amplitudes) + len(shuffled_amplitudes))
    all_amplitudes[:len(control_amplitudes)] = control_amplitudes[:]
    all_amplitudes[len(control_amplitudes):] = shuffled_amplitudes[:]

    print 'Control amplitude: %.1f +- %.1f pA' % (np.mean(control_amplitudes), np.std(control_amplitudes))
    print 'Shuffled amplitude: %.1f +- %.1f pA' % (np.mean(shuffled_amplitudes), np.std(shuffled_amplitudes))

    bin_size = 2.0
    bins = np.arange(np.min(all_amplitudes), np.max(all_amplitudes) + bin_size, bin_size)
    control_hist, _ = np.histogram(control_amplitudes, bins)
    control_hist = control_hist/(1.0*np.sum(control_hist))
    shuffled_hist, _ = np.histogram(shuffled_amplitudes, bins)
    shuffled_hist = shuffled_hist/(1.0*np.sum(shuffled_hist))

    plt.figure(1)
    plt.step(bins[:-1], shuffled_hist, 'k', linewidth=0.5)
    plt.step(bins[:-1], control_hist, 'r', linewidth=0.5)
    plt.xlabel('Amplitude (pA)')
    plt.ylabel('Rel. frequency')
    plt.title('Amplitudes cluster %d' % cluster_id)
    plt.show()


def individual_currents_network_coupling(experiment_info_name):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    # load good clusters and mua
    groups = ('good',)
    si_probe_data_folder = experiment_info['SiProbe']['DataBasePath']
    clusters = clust.reader.read_KS_clusters(si_probe_data_folder, clustering_src_folder,
                                       'dev', groups, experiment_info['SiProbe']['SamplingRate'])
    all_groups = ('good', 'mua')
    all_clusters = clust.reader.read_KS_clusters(si_probe_data_folder, clustering_src_folder,
                                       'dev', all_groups, experiment_info['SiProbe']['SamplingRate'])

    # compute smoothed population firing rate as a first proxy for network correlations
    population_rate = _estimate_population_firing(all_clusters)

    # test this - does it matter which one we use? highly correlated
    # population_rate_good = _estimate_population_firing(clusters)
    # plt.figure(0)
    # max_time_fr = len(population_rate)/100.0 # in s
    # time_axis_fr = np.linspace(0.0, max_time_fr, len(population_rate))
    # plt.plot(time_axis_fr, population_rate*100.0, 'k', linewidth=0.5)
    # plt.show()
    # 0/0
    # plt.plot(population_rate_good, 'r')

    # for cluster_id in clusters:
    for cluster_id in [14, 616, 635]:
        cluster = clusters[cluster_id]
        # if cluster_id != 942:
        #     continue
        print 'Sorting individual events of cluster %d by degree of network coupling' % cluster_id

        control_snippets = _load_snippets(experiment_info, cluster_id)
        shuffled_snippets = _load_snippets(experiment_info, cluster_id, 'shuffled_1x')

        # sort spike times and snippets for cluster of interest by population firing rate
        snippet_order, snippet_firing_rates = _sort_spikes_by_population_rate(control_snippets, population_rate)
        # compute STA and amplitude histogram for those spike times and compare to shuffled histogram
        # (or do same thing with shuffled spike times?)
        # number of subdivisions of the firing rate distribution
        n_percentiles = 10
        sample_step = len(control_snippets.snippets)//n_percentiles
        if not sample_step:
            continue
        # control curves
        st_bin = int(10.0*experiment_info['WC']['SamplingRate']/1000.0)
        sta_control = np.mean(control_snippets.snippets, axis=0)
        sta_shuffled = np.mean(shuffled_snippets.snippets, axis=0)
        time_axis = np.linspace(-10.0, 25.0, len(sta_control))

        ordered_sta_curves = []
        ordered_st_snippets = []
        last_start = 0
        for i in range(n_percentiles):
            start = last_start
            end = (i + 1)*sample_step
            if i == n_percentiles - 1:
                end = len(snippet_order)
            snippet_indices = snippet_order[start:end]
            snippets_percentile = ca.sts.SnippetArray(control_snippets.snippets[snippet_indices],
                                               control_snippets.signal_names[snippet_indices],
                                               control_snippets.snippet_timepoints[snippet_indices],
                                               np.array(control_snippets.snippet_spike_times)[snippet_indices])
            firing_rates = snippet_firing_rates[start:end]
            ordered_sta_curves.append((np.mean(snippets_percentile.snippets, axis=0), np.mean(firing_rates)))
            ordered_st_snippets.append(snippets_percentile)
            last_start = end

        plt.figure(2*cluster_id)
        plt.plot(time_axis, sta_shuffled - sta_shuffled[st_bin], 'k--', label='shuffled')
        plt.plot(time_axis, sta_control - sta_control[st_bin], 'k', label='control')
        for i in range(n_percentiles):
            tmp_sta = ordered_sta_curves[i][0]
            tmp_fr = ordered_sta_curves[i][1]
            label_str = '%.2f kHz' % (0.1*tmp_fr)
            plt.plot(time_axis, tmp_sta - tmp_sta[st_bin], label=label_str)
        plt.legend()
        plt.xlabel('Time relative to spike (ms)')
        plt.ylabel('Current (pA)')
        plt.title('STA of cluster %d (%d spikes)' % (cluster_id, len(control_snippets.snippets)))

        sta_name = 'STA_NetworkCoupling_%d_subdivisions_Cluster_%d.pdf' % (n_percentiles, cluster_id)
        sta_figure_name = os.path.join(experiment_info['STA']['DataBasePathI'], sta_name)
        # plt.savefig(sta_figure_name)

        # plot synaptic current amplitude as function of percentile
        # this graph contains information about whether a synapse is real or a result of network correlations
        plt.figure(2*cluster_id + 1)
        ax = plt.subplot(1, 1, 1)
        for i in range(n_percentiles):
            percentile_amps = _individual_currents_from_snippets(experiment_info, ordered_st_snippets[i])
            mean_amp = np.mean(percentile_amps)
            se_amp = np.std(percentile_amps)/np.sqrt(len(percentile_amps))
            percentile = (i + 1) * 100.0 / n_percentiles
            ax.errorbar(percentile, mean_amp, 2*se_amp, fmt='ko')
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Amplitude')
        ax.set_title('STA percentile amplitudes of cluster %d (%d spikes)' % (cluster_id, len(control_snippets.snippets)))

        sta_amplitudes_name = 'STA_NetworkCoupling_%d_subdivisions_amplitudes_Cluster_%d.pdf' % (n_percentiles, cluster_id)
        sta_amplitudes_figure_name = os.path.join(experiment_info['STA']['DataBasePathI'], sta_amplitudes_name)
        # plt.savefig(sta_amplitudes_figure_name)
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        experimentInfoName = sys.argv[1]
        individual_currents_network_coupling(experimentInfoName)
    if len(sys.argv) == 3:
        experimentInfoName = sys.argv[1]
        clusterID = int(sys.argv[2])
        individual_spike_triggered_currents(experimentInfoName, clusterID)


