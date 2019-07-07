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

    for cluster_id, cluster in clusters.iteritems():
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
        last_start = 0
        for i in range(n_percentiles):
            start = last_start
            end = (i + 1)*sample_step
            if i == n_percentiles - 1:
                end = len(snippet_order)
            snippet_indices = snippet_order[start:end]
            snippets_ordered = ca.sts.SnippetArray(control_snippets.snippets[snippet_indices],
                                               control_snippets.signal_names[snippet_indices],
                                               control_snippets.snippet_timepoints[snippet_indices],
                                               np.array(control_snippets.snippet_spike_times)[snippet_indices])
            firing_rates = snippet_firing_rates[start:end]
            ordered_sta_curves.append((np.mean(snippets_ordered.snippets, axis=0), np.mean(firing_rates)))
            last_start = end

        plt.figure(cluster_id)
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
        plt.savefig(sta_figure_name)

        # n_sample = min(len(control_snippets.snippets), 100)
        # if not n_sample:
        #     continue
        # snippet_indices_low_fr = snippet_order[:n_sample]
        # ordered_snippets_low_fr = ca.sts.SnippetArray(control_snippets.snippets[snippet_indices_low_fr],
        #                                        control_snippets.signal_names[snippet_indices_low_fr],
        #                                        control_snippets.snippet_timepoints[snippet_indices_low_fr],
        #                                        np.array(control_snippets.snippet_spike_times)[snippet_indices_low_fr])
        #
        # snippet_indices_high_fr = snippet_order[-n_sample:]
        # ordered_snippets_high_fr = ca.sts.SnippetArray(control_snippets.snippets[snippet_indices_high_fr],
        #                                        control_snippets.signal_names[snippet_indices_high_fr],
        #                                        control_snippets.snippet_timepoints[snippet_indices_high_fr],
        #                                        np.array(control_snippets.snippet_spike_times)[snippet_indices_high_fr])
        #
        # st_bin = int(10.0*experiment_info['WC']['SamplingRate']/1000.0)
        # sta_control = np.mean(control_snippets.snippets, axis=0)
        # sta_ordered_low_fr = np.mean(ordered_snippets_low_fr.snippets, axis=0)
        # sta_ordered_high_fr = np.mean(ordered_snippets_high_fr.snippets, axis=0)
        # sta_shuffled = np.mean(shuffled_snippets.snippets, axis=0)
        # time_axis = np.linspace(-10.0, 25.0, len(sta_control))
        # plt.figure(2*cluster_id)
        # plt.plot(time_axis, sta_shuffled - sta_shuffled[st_bin], 'k', label='shuffled')
        # plt.plot(time_axis, sta_control - sta_control[st_bin], 'r', label='control')
        # plt.plot(time_axis, sta_ordered_low_fr - sta_ordered_low_fr[st_bin], 'b', label='low network')
        # plt.plot(time_axis, sta_ordered_high_fr - sta_ordered_high_fr[st_bin], 'g', label='high network')
        # plt.legend()
        # plt.xlabel('Time relative to spike (ms)')
        # plt.ylabel('Current (pA)')
        # plt.title('STA of cluster %d (%d spikes)' % (cluster_id, len(control_snippets.snippets)))
        #
        # sta_name = 'STA_NetworkCoupling_Cluster_%d.pdf' % cluster_id
        # sta_figure_name = os.path.join(experiment_info['STA']['DataBasePathI'], sta_name)
        # plt.savefig(sta_figure_name)
        #
        # # tmp = sta_shuffled - sta_shuffled[st_bin]
        # # sta_shuffled_norm = tmp/np.max(tmp[st_bin:])
        # tmp_control = sta_control - sta_control[st_bin]
        # # sta_control_norm = tmp/np.max(tmp[st_bin:])
        # tmp = sta_ordered_low_fr - sta_ordered_low_fr[st_bin]
        # sta_ordered_low_fr_norm = tmp/np.max(tmp[st_bin:])*np.max(tmp_control[st_bin:])
        # tmp = sta_ordered_high_fr - sta_ordered_high_fr[st_bin]
        # sta_ordered_high_fr_norm = tmp/np.max(tmp[st_bin:])*np.max(tmp_control[st_bin:])
        # plt.figure(2*cluster_id + 1)
        # plt.plot(time_axis, sta_shuffled - sta_shuffled[st_bin], 'k', label='shuffled')
        # plt.plot(time_axis, sta_control - sta_control[st_bin], 'r', label='control')
        # plt.plot(time_axis, sta_ordered_low_fr_norm, 'b', label='low network')
        # plt.plot(time_axis, sta_ordered_high_fr_norm, 'g', label='high network')
        # plt.legend()
        # plt.xlabel('Time relative to spike (ms)')
        # plt.ylabel('Current (pA)')
        # plt.title('STA of cluster %d' % cluster_id)
        #
        # sta_name = 'STA_NetworkCoupling_Normalized_Cluster_%d.pdf' % cluster_id
        # sta_figure_name = os.path.join(experiment_info['STA']['DataBasePathI'], sta_name)
        # plt.savefig(sta_figure_name)
        #
        # control_amplitudes = _individual_currents_from_snippets(experiment_info, control_snippets)
        # ordered_amplitudes_low_fr = _individual_currents_from_snippets(experiment_info, ordered_snippets_low_fr)
        # ordered_amplitudes_high_fr = _individual_currents_from_snippets(experiment_info, ordered_snippets_high_fr)
        # shuffled_amplitudes = _individual_currents_from_snippets(experiment_info, shuffled_snippets)
        #
        # # min_amplitude = np.min((np.min(control_amplitudes), np.min(shuffled_amplitudes),
        # #                         np.min(ordered_amplitudes_low_fr), np.min(ordered_amplitudes_high_fr)))
        # # max_amplitude = np.max((np.max(control_amplitudes), np.max(shuffled_amplitudes),
        # #                         np.max(ordered_amplitudes_low_fr), np.max(ordered_amplitudes_high_fr)))
        #
        # print 'Control amplitude: %.1f +- %.1f pA' % (np.mean(control_amplitudes), np.std(control_amplitudes))
        # print 'Ordered amplitude (low FR): %.1f +- %.1f pA' % (np.mean(ordered_amplitudes_low_fr), np.std(ordered_amplitudes_low_fr))
        # print 'Ordered amplitude (high FR): %.1f +- %.1f pA' % (np.mean(ordered_amplitudes_high_fr), np.std(ordered_amplitudes_high_fr))
        # print 'Shuffled amplitude: %.1f +- %.1f pA' % (np.mean(shuffled_amplitudes), np.std(shuffled_amplitudes))
        #
        # # bin_size = 2.0
        # # bins = np.arange(min_amplitude, max_amplitude + bin_size, bin_size)
        # # control_hist, _ = np.histogram(control_amplitudes, bins)
        # # control_hist = control_hist/(1.0*np.sum(control_hist))
        # # ordered_hist_low_fr, _ = np.histogram(ordered_amplitudes_low_fr, bins)
        # # ordered_hist_low_fr = ordered_hist_low_fr/(1.0*np.sum(ordered_hist_low_fr))
        # # ordered_hist_high_fr, _ = np.histogram(ordered_amplitudes_high_fr, bins)
        # # ordered_hist_high_fr = ordered_hist_high_fr/(1.0*np.sum(ordered_hist_high_fr))
        # # shuffled_hist, _ = np.histogram(shuffled_amplitudes, bins)
        # # shuffled_hist = shuffled_hist/(1.0*np.sum(shuffled_hist))
        # #
        # # plt.figure(2)
        # # plt.step(bins[:-1], shuffled_hist, 'k', linewidth=0.5)
        # # plt.step(bins[:-1], control_hist, 'r', linewidth=0.5)
        # # plt.step(bins[:-1], ordered_hist_low_fr, 'b', linewidth=0.5)
        # # plt.step(bins[:-1], ordered_hist_high_fr, 'g', linewidth=0.5)
        # # plt.xlabel('Amplitude (pA)')
        # # plt.ylabel('Rel. frequency')
        # # plt.title('Amplitudes cluster %d' % cluster_id)
        #
        # # plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        experimentInfoName = sys.argv[1]
        individual_currents_network_coupling(experimentInfoName)
    if len(sys.argv) == 3:
        experimentInfoName = sys.argv[1]
        clusterID = int(sys.argv[2])
        individual_spike_triggered_currents(experimentInfoName, clusterID)


## Random stuff that ive used before
    # # STPickleName = 'STSnippets_Cluster_%d.pkl' % clusterID
    # # STPickleNameShuffled = 'STSnippets_shuffled_100x_Cluster_%d_95-300s.pkl' % clusterID
    # STPickleNameShuffled = 'STSnippets_shuffled_10x_Cluster_%d.pkl' % clusterID
    # STSnippetShuffledName = os.path.join(experimentInfo['STA']['DataBasePathI'], STPickleNameShuffled)
    # STPickleName = 'STSnippets_Cluster_%d.pkl' % clusterID
    # STSnippetName = os.path.join(experimentInfo['STA']['DataBasePathI'], STPickleName)
    # with open(STSnippetShuffledName, 'rb') as STSnippetShuffledFile:
    #     clusterSnippetsShuffled = cPickle.load(STSnippetShuffledFile)
    # with open(STSnippetName, 'rb') as STSnippetFile:
    #     clusterSnippets = cPickle.load(STSnippetFile)
    # clusterSTA = np.mean(clusterSnippets.snippets, axis=0)
    # peakBin = np.argmax(clusterSTA)
    # # tPeak = peakBin/experimentInfo['WC']['SamplingRate'] # s
    # tOffset = 0.01 # s
    # offsetBin = int(tOffset*experimentInfo['WC']['SamplingRate'])
    # timeAxis = (np.array(range(len(clusterSTA)))/experimentInfo['WC']['SamplingRate'] - tOffset)*1000.0 # ms
    # # tOffsetPlot1 = 0.3 # ms
    # tOffsetPlot1 = -1.2 # ms
    # tOffsetPlot2 = -0.2 # ms
    # # tPeakPlot = (tPeak-tOffset)*1000.0 # ms
    # tPeakPlot = 3.0 # ms
    # # tPeakPlot2 = 1.2 # ms
    # tPeakPlot2 = 2.0 # ms
    #
    # tracesPerPage = 10
    # # nrTraces = 40
    # # traceStep = len(clusterSnippets.snippets)/nrTraces
    # nrPages = len(clusterSnippets.snippets)/tracesPerPage + 1
    # # nrPages = nrTraces/tracesPerPage
    # # for i in range(nrPages):
    # # # for i in range(2):
    # #     spacing = 5.0*clusterSnippets.snippets[0].units
    # #     plt.figure(i)
    # #     lineMin = np.min(clusterSTA)
    # #     plt.plot(timeAxis, clusterSTA, 'r')
    # #     offset = np.max(clusterSTA)*clusterSnippets.snippets[0].units
    # #     for j in range(tracesPerPage):
    # #         traceNr = j + i*tracesPerPage
    # #         # traceNr = (j + i*tracesPerPage)*traceStep
    # #         if traceNr >= len(clusterSnippets.snippets):
    # #             break
    # #         plotTrace = clusterSnippets.snippets[traceNr]
    # #         minShift = np.min(plotTrace)
    # #         plotTrace += offset + spacing - minShift
    # #         plt.plot(timeAxis, plotTrace, 'k')
    # #         offset = np.max(plotTrace)
    # #     lineMax = offset
    # #     plt.plot([tOffsetPlot, tOffsetPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.plot([tPeakPlot, tPeakPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.plot([tPeakPlot2, tPeakPlot2], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.xlabel('Time relative to spike (ms)')
    # #     plt.ylabel('Current (pA)')
    # #     titleStr = 'Cluster %d - page %d - spikes %d - %d' % (clusterID, i + 1, i*tracesPerPage + 1, traceNr + 1)
    # #     plt.title(titleStr)
    # #     clusterPageName = 'Cluster_%d_page_%d.pdf' % (clusterID, i)
    # #     pageName = os.path.join(experimentInfo['STA']['DataBasePathI'], clusterPageName)
    # #     plt.savefig(pageName)
    # #     # plt.show()
    # lastFigure = nrPages
    #
    # tPeakOffset1 = 2.0
    # tPeakOffset2 = 3.0
    # # ampOffsetBin = int((tOffsetPlot/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    # ampOffsetBin1 = int((tOffsetPlot1/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    # ampOffsetBin2 = int((tOffsetPlot2/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    # ampPeakBin1 = int((tPeakOffset1/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    # ampPeakBin2 = int((tPeakOffset2/1000.0 + tOffset)*experimentInfo['WC']['SamplingRate'])
    # # ampPeakBin = peakBin
    # amplitudes = []
    # amplitudes_0_95 = []
    # amplitudes_95_end = []
    # spikeTimes_0_95 = []
    # spikeTimes_95_end = []
    # for i, snippet in enumerate(clusterSnippets.snippets):
    #     # amp = np.max(snippet[ampOffsetBin:ampPeakBin]) - snippet[ampOffsetBin]
    #     # amp = np.max(snippet[ampPeakBin1:ampPeakBin2]) - snippet[ampOffsetBin]
    #     amp = np.median(snippet[ampPeakBin1:ampPeakBin2]) - np.median(snippet[ampOffsetBin1:ampOffsetBin2])
    #     # amplitudes.append(amp.magnitude)
    #     amplitudes.append(amp)
    #     if clusterSnippets.snippetSpikeTimes[i] < 100.0:
    #         # amplitudes_0_95.append(amp.magnitude)
    #         amplitudes_0_95.append(amp)
    #         spikeTimes_0_95.append(clusterSnippets.snippetSpikeTimes[i])
    #     else:
    #         # amplitudes_95_end.append(amp.magnitude)
    #         amplitudes_95_end.append(amp)
    #         spikeTimes_95_end.append(clusterSnippets.snippetSpikeTimes[i])
    # amplitudes = np.array(amplitudes).flatten()
    # amplitudes_0_95 = np.array(amplitudes_0_95).flatten()
    # amplitudes_95_end = np.array(amplitudes_95_end).flatten()
    #
    # ISIs = np.diff(clusterSnippets.snippetSpikeTimes)
    # ISIs_0_95 = np.diff(spikeTimes_0_95)
    # ISIs_95_end = np.diff(spikeTimes_95_end)
    # # plt.plot(ISIs, amplitudes[1:])
    # bins = np.geomspace(np.min(ISIs), np.max(ISIs), 8)
    # # bins = np.arange(0.0, np.max(ISIs) + 0.1, 0.1)
    # meanAmp, _, _ = scipy.stats.binned_statistic(ISIs, amplitudes[1:], statistic='mean', bins=bins)
    # stdAmp, _, _ = scipy.stats.binned_statistic(ISIs, amplitudes[1:], statistic=(lambda x: np.sqrt(np.dot(x - np.mean(x), x - np.mean(x))/len(x))), bins=bins)
    # meanAmp_0_95, _, _ = scipy.stats.binned_statistic(ISIs_0_95, amplitudes_0_95[1:], statistic='mean', bins=bins)
    # stdAmp_0_95, _, _ = scipy.stats.binned_statistic(ISIs_0_95, amplitudes_0_95[1:], statistic=(lambda x: np.sqrt(np.dot(x - np.mean(x), x - np.mean(x))/len(x))), bins=bins)
    # meanAmp_95_end, _, _ = scipy.stats.binned_statistic(ISIs_95_end, amplitudes_95_end[1:], statistic='mean', bins=bins)
    # stdAmp_95_end, _, _ = scipy.stats.binned_statistic(ISIs_95_end, amplitudes_95_end[1:], statistic=(lambda x: np.sqrt(np.dot(x - np.mean(x), x - np.mean(x))/len(x))), bins=bins)
    # plt.figure(0)
    # # plt.loglog(ISIs, amplitudes[1:], 'ko')
    # # plt.semilogx(ISIs, amplitudes[1:], 'ko')
    # # plt.semilogx(ISIs_0_95, amplitudes_0_95[1:], 'ro', label='0 - 100 s')
    # plt.plot(ISIs_0_95, amplitudes_0_95[1:], 'ro', label='0 - 100 s')
    # # plt.xlabel('ISI (s)')
    # # plt.ylabel('IPSC amplitude (pA)')
    # # plt.title('0-100 s')
    # # plt.figure(1)
    # # plt.loglog(ISIs, amplitudes[1:], 'ko')
    # # plt.semilogx(ISIs, amplitudes[1:], 'ko')
    # # plt.semilogx(ISIs_95_end, amplitudes_95_end[1:], 'bo', label='100 s - end')
    # fit = np.polyfit(ISIs_0_95, amplitudes_0_95[1:], 1)
    # fit_fn = np.poly1d(fit)
    # x = np.min(ISIs_0_95), np.max(ISIs_0_95)
    # plt.plot(x, fit_fn(x), 'k-')
    # plt.xlabel('ISI (s)')
    # plt.ylabel('IPSC amplitude (pA)')
    # plt.legend()
    # # plt.title('100 s - end')
    # fig = plt.figure(2)
    # ax = fig.add_subplot(111)
    # # plotBins = bins[:-1][np.logical_not(np.isnan(meanAmp))] + 0.5*np.diff(bins)[np.logical_not(np.isnan(meanAmp))]
    # # plotMean = meanAmp[np.logical_not(np.isnan(meanAmp))]
    # # plotStd = stdAmp[np.logical_not(np.isnan(meanAmp))]
    # # plt.errorbar(plotBins, plotMean, yerr=plotStd, fmt='ko-', label='all')
    # plotBins_0_95 = bins[:-1][np.logical_not(np.isnan(meanAmp_0_95))] + 0.5*np.diff(bins)[np.logical_not(np.isnan(meanAmp_0_95))]
    # plotMean_0_95 = meanAmp_0_95[np.logical_not(np.isnan(meanAmp_0_95))]
    # plotStd_0_95 = stdAmp_0_95[np.logical_not(np.isnan(meanAmp_0_95))]
    # plt.errorbar(plotBins_0_95, plotMean_0_95, yerr=plotStd_0_95, fmt='ro-', label='0 - 100 s')
    # plotBins_95_end = bins[:-1][np.logical_not(np.isnan(meanAmp_95_end))] + 0.5*np.diff(bins)[np.logical_not(np.isnan(meanAmp_95_end))]
    # plotMean_95_end = meanAmp_95_end[np.logical_not(np.isnan(meanAmp_95_end))]
    # plotStd_95_end = stdAmp_95_end[np.logical_not(np.isnan(meanAmp_95_end))]
    # plt.errorbar(plotBins_95_end, plotMean_95_end, yerr=plotStd_95_end, fmt='bo-', label='100 s - end')
    # ax.set_xscale('log')
    # plt.xlabel('ISI (s)')
    # plt.ylabel('IPSC amplitude (pA)')
    # plt.legend()
    # plt.ylim([0, 150.0])
    # # plt.show()
    #
    # amplitudesShuffled = []
    # for i, snippet in enumerate(clusterSnippetsShuffled.snippets):
    #     # amp = np.max(snippet[ampOffsetBin:ampPeakBin]) - snippet[ampOffsetBin]
    #     # amp = np.max(snippet[ampPeakBin1:ampPeakBin2]) - snippet[ampOffsetBin]
    #     amp = np.median(snippet[ampPeakBin1:ampPeakBin2].magnitude) - np.median(snippet[ampOffsetBin1:ampOffsetBin2].magnitude)
    #     amplitudesShuffled.append(amp)
    # amplitudesShuffled = np.array(amplitudesShuffled)
    #
    # # ampName = 'STSnippets_Cluster_%d_amplitudes' % clusterID
    # # ampOutName = os.path.join(experimentInfo['STA']['DataBasePathI'], ampName)
    # # np.save(ampOutName, amplitudes)
    # # ampShuffledName = 'STSnippets_shuffled_10x_Cluster_%d_amplitudes' % clusterID
    # # ampShuffledOutName = os.path.join(experimentInfo['STA']['DataBasePathI'], ampShuffledName)
    # # np.save(ampShuffledOutName, amplitudesShuffled)
    #
    # # plt.figure(1)
    # # plt.plot(clusterSnippets.snippetSpikeTimes, amplitudes, 'ko')
    # print len(np.where(clusterSnippets.snippetSpikeTimes < 95)[0])/95.0
    # print len(np.where(clusterSnippets.snippetSpikeTimes >= 95)[0])/205.0
    # # plt.xlabel('Time during WC recording (s)')
    # # plt.ylabel('Current amplitude (pA)')
    # # plt.show()
    #
    # # plt.figure(nrPages + 1)
    # binSize = 2.0
    # bins = np.arange(np.min(amplitudesShuffled), np.max(amplitudes)+binSize, binSize)
    # # bins = np.arange(np.min(amplitudesShuffled), np.max(amplitudes_0_95)+binSize, binSize)
    # # bins = np.arange(np.min(amplitudesShuffled), np.max(amplitudes_95_end)+binSize, binSize)
    # # hist, _ = np.histogram(amplitudes, bins)
    # # # plt.hist(amplitudes, bins)
    # # plt.bar(bins[:-1], hist, width=binSize)
    # # plt.xlabel('Amplitude (pA)')
    # # plt.ylabel('Events')
    #
    # # plt.figure(nrPages + 2)
    # # hist_0_95, _ = np.histogram(amplitudes_0_95, bins)
    # # hist_shuffled, _ = np.histogram(amplitudesShuffled, bins)
    # # hist_shuffled = np.array(hist_shuffled, dtype='float64')
    # # hist_shuffled *= 1.0*np.sum(hist_0_95)/np.sum(hist_shuffled)
    # # # plt.hist(amplitudes, bins)
    # # plt.bar(bins[:-1], hist_shuffled, width=binSize, linewidth=0, color='k', label='shuffled')
    # # plt.bar(bins[:-1], hist_0_95, width=binSize, linewidth=0, color='r', label='spike times')
    # # plt.xlabel('Amplitude (pA)')
    # # plt.ylabel('Events (during 0-95 s)')
    # # plt.legend()
    #
    # plt.figure(nrPages + 2)
    # # hist_95_end, _ = np.histogram(amplitudes_95_end, bins)
    # hist, _ = np.histogram(amplitudes, bins)
    # hist_shuffled, _ = np.histogram(amplitudesShuffled, bins)
    # hist_shuffled = np.array(hist_shuffled, dtype='float64')
    # hist_shuffled *= 1.0*np.sum(hist)/np.sum(hist_shuffled)
    # # plt.hist(amplitudes, bins)
    # # plt.bar(bins[:-1], hist_shuffled, width=binSize, linewidth=0, color='k', label='shuffled')
    # # plt.bar(bins[:-1], hist, width=binSize, linewidth=0, color='r', label='spike times')
    # plt.plot(bins[:-1] + 0.5*binSize, hist_shuffled, 'k', label='shuffled')
    # plt.plot(bins[:-1] + 0.5*binSize, hist, 'r', label='spike times')
    # plt.xlabel('Amplitude (pA)')
    # plt.ylabel('Events')
    # plt.legend()
    #
    # # # ISIs
    # # earlySpikes = clusterSnippets.snippetSpikeTimes[np.where(clusterSnippets.snippetSpikeTimes < 95.0)]
    # # earlyISIs = np.diff(earlySpikes)
    # # earlyAmps = amplitudes_0_95[1:]
    # # lateSpikes = clusterSnippets.snippetSpikeTimes[np.where(clusterSnippets.snippetSpikeTimes >= 95.0)]
    # # lateISIs = np.diff(lateSpikes)
    # # lateAmps = amplitudes_95_end[1:]
    # # plt.figure(nrPages + 4)
    # # plt.plot(earlyISIs, earlyAmps, 'ro', label='<95 s')
    # # plt.plot(lateISIs, lateAmps, 'ko', label='>=95 s')
    # # plt.legend()
    # # plt.xlabel('ISI (s)')
    # # plt.ylabel('Event amplitude (pA)')
    # # plt.figure(nrPages + 5)
    # # plt.semilogx(earlyISIs, earlyAmps, 'ro', label='<95 s')
    # # plt.semilogx(lateISIs, lateAmps, 'ko', label='>=95 s')
    # # plt.legend()
    # # plt.xlabel('ISI (s)')
    # # plt.ylabel('Event amplitude (pA)')
    # plt.show()
    #
    # # amps_0_6 = np.where(amplitudes <= 6.0)[0]
    # # nrPages = len(amps_0_6)/tracesPerPage
    # # for i in range(nrPages):
    # # # for i in range(2):
    # #     spacing = 5.0*clusterSnippets.snippets[0].units
    # #     plt.figure(lastFigure + i)
    # #     lineMin = np.min(clusterSTA)
    # #     plt.plot(timeAxis, clusterSTA, 'r')
    # #     offset = np.max(clusterSTA)*clusterSnippets.snippets[0].units
    # #     for j in range(tracesPerPage):
    # #         traceNr = j + i*tracesPerPage
    # #         if traceNr >= len(amps_0_6):
    # #             break
    # #         traceNr = amps_0_6[j + i*tracesPerPage]
    # #         plotTrace = clusterSnippets.snippets[traceNr]
    # #         minShift = np.min(plotTrace)
    # #         plotTrace += offset + spacing - minShift
    # #         plt.plot(timeAxis, plotTrace, 'k')
    # #         offset = np.max(plotTrace)
    # #     lineMax = offset
    # #     plt.plot([tOffsetPlot, tOffsetPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.plot([tPeakPlot, tPeakPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.plot([tPeakPlot2, tPeakPlot2], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.xlabel('Time relative to spike (ms)')
    # #     plt.ylabel('Current (pA)')
    # #     clusterPageName = 'Cluster_%d_amps_0_6_page_%d.pdf' % (clusterID, i)
    # #     pageName = os.path.join(experimentInfo['STA']['DataBasePathI'], clusterPageName)
    # #     plt.savefig(pageName)
    # #     # plt.show()
    # # lastFigure += nrPages
    # #
    # # amps_6_12 = np.where((amplitudes > 6.0)*(amplitudes <=12.0))[0]
    # # nrPages = len(amps_6_12)/tracesPerPage
    # # for i in range(nrPages):
    # # # for i in range(2):
    # #     spacing = 5.0*clusterSnippets.snippets[0].units
    # #     plt.figure(lastFigure + i)
    # #     lineMin = np.min(clusterSTA)
    # #     plt.plot(timeAxis, clusterSTA, 'r')
    # #     offset = np.max(clusterSTA)*clusterSnippets.snippets[0].units
    # #     for j in range(tracesPerPage):
    # #         traceNr = j + i*tracesPerPage
    # #         if traceNr >= len(amps_6_12):
    # #             break
    # #         traceNr = amps_6_12[j + i*tracesPerPage]
    # #         plotTrace = clusterSnippets.snippets[traceNr]
    # #         minShift = np.min(plotTrace)
    # #         plotTrace += offset + spacing - minShift
    # #         plt.plot(timeAxis, plotTrace, 'k')
    # #         offset = np.max(plotTrace)
    # #     lineMax = offset
    # #     plt.plot([tOffsetPlot, tOffsetPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.plot([tPeakPlot, tPeakPlot], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.plot([tPeakPlot2, tPeakPlot2], [lineMin, lineMax], 'r--', linewidth=0.5)
    # #     plt.xlabel('Time relative to spike (ms)')
    # #     plt.ylabel('Current (pA)')
    # #     clusterPageName = 'Cluster_%d_amps_6_12_page_%d.pdf' % (clusterID, i)
    # #     pageName = os.path.join(experimentInfo['STA']['DataBasePathI'], clusterPageName)
    # #     plt.savefig(pageName)
    # #     # plt.show()
    # # # plt.show()