import os
import ast
import sys
import copy
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import ClusterProcessing as cp
import utilities as utils

clustering_src_folder = 'E:\\User\\project_src\\physiology\\Clustering'
clusters_of_interest = [1, 9, 45, 46, 55, 58, 108, 128, 135, 209, 244, 266, 304, 309, 337, 353, 388, 454, 469, 578,
                        685, 701, 702, 705, 721, 733, 738, 759, 760, 761, 772, 779] # bursters that look the most promising


def _set_up_filter(highpass, lowpass, fs):
    filter_order = 3
    return signal.butter(filter_order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def _load_bursts_per_cluster_file(fname):
    cluster_id, burst_nr = np.loadtxt(fname, skiprows=1, unpack=True)
    return dict(zip(cluster_id, burst_nr))


def _amplitude_pairwise_similarity(burst1_amplitudes, burst2_amplitudes, n_electrodes):
    """
    compute inner pairwise burst similarity within bursts and across bursts
    different if K-S test between within and across burst comparisons significant
    :param burst1_amplitudes: array with shape (nr_trials, electrode_amplitudes)
    :param burst2_amplitudes: array with shape (nr_trials, electrode_amplitudes)
    :param n_electrodes: number of electrodes to use in similarity
    :return: list with all combinations of pairwise similarities
    """
    pairwise_similarities = []
    for i in range(len(burst1_amplitudes)):
        if burst1_amplitudes[i] is None:
            continue
        for j in range(len(burst2_amplitudes)):
            if burst2_amplitudes[j] is None:
                continue
            similarity = np.dot(burst1_amplitudes[i][:n_electrodes], burst2_amplitudes[j][:n_electrodes]) #/ \
                         # np.dot(burst1_amplitudes[i][:n_electrodes], burst1_amplitudes[i][:n_electrodes]) / \
                         # np.dot(burst2_amplitudes[j][:n_electrodes], burst2_amplitudes[j][:n_electrodes])
            pairwise_similarities.append(similarity)

    return np.array(pairwise_similarities)


def template_extent_spatial(experiment_info_name):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get motif template
    # audio_fs, audio_data = clust.reader.read_template_audiofile(experiment_info['Motifs']['TemplateFilename'])
    # plot_audio = _normalize_trace(audio_data, -1.0, 1.0)
    # get clusters
    data_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    # clusters = _read_all_clusters_except_noise(data_folder, 'dev', fs)
    clusters = cp.reader.read_KS_clusters(data_folder, clustering_src_folder, 'dev', ('good',), fs)

    # get probe geometry
    channel_shank_map = np.load(os.path.join(data_folder, 'channel_shank_map.npy'))
    channel_coordinates = np.load(os.path.join(data_folder, 'channel_positions.npy'))

    raw_traces = cp.reader.load_recording(os.path.join(data_folder, experiment_info['SiProbe']['AmplifierName']),
                                          experiment_info['SiProbe']['Channels'])
    intan_constant = 0.195

    # b, a = _set_up_filter(300, 0.49*fs, fs)
    # def bp_filter(x):
    #     return signal.filtfilt(b, a, x, axis=1)

    for cluster_id in clusters_of_interest:
    # for cluster_id in [761]:
        # load template waveform
        # load channel map spatial coordinates
        # get amplitudes on all channels
        # calculate euclidean distance of all channels to max channel
        # plot wf amplitude (defined how? min-max? normalized how?) as function of distance
        # use raw waveforms instead of template?
        # can this amplitude decay be described parametrically (e.g., exponential, gaussian, ...?)
        # also - 2D map additionally to collapsing onto 1D distance
        cluster = clusters[cluster_id]
        spike_times = cluster.spiketrains[0]
        motif_spike_times = []
        # motif object with attributes start, stop, center and warp (and more not relevant here)
        for i in range(len(motif_finder_data.start)):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
            motif_spike_times.append(spike_times.magnitude[selection])

        max_channel = cluster.maxChannel
        shank = channel_shank_map[max_channel]
        shank_channels = np.where(np.squeeze(channel_shank_map) == shank)
        nr_shank_channels = np.sum(channel_shank_map == shank)
        template_wf = cluster.template
        wf_samples = np.sum(template_wf[:, max_channel] != 0)
        # wf_offset: spike time at center when wf_samples = 61.
        # Otherwise KiloSort pads the template with zeros
        # starting from the beginning. So we have to move
        # the center of the extracted waveform accordingly
        sample_diff = 61 - wf_samples
        wf_offset_begin = (wf_samples - sample_diff) // 2
        wf_offset_end = (wf_samples + sample_diff) // 2
        nr_spikes = 0
        for motif in motif_spike_times:
            nr_spikes += len(motif)
        motif_waveforms = np.zeros((nr_spikes, nr_shank_channels, wf_samples))
        # main loop
        for motif in motif_spike_times:
            for i, t_spike in enumerate(motif):
                # careful with the edge cases - zero-padding
                # uint64 converted silently to float64 when adding an int - cast to int64
                spike_sample = int(t_spike*fs)
                start_index_ = spike_sample - wf_offset_begin - 1
                start_index, start_diff = start_index_, 0
                # uint64 converted silently to float64 when adding an int - cast to int64
                stop_index_ = np.int64(spike_sample) + wf_offset_end
                stop_index, stop_diff = stop_index_, 0
                # now copy the appropriately sized snippet from channels on same clusters
                motif_waveforms[i, :, :] = intan_constant*raw_traces[shank_channels, start_index: stop_index]
        mean_wf = np.mean(motif_waveforms, axis=0)
        channel_distances = []
        channel_template_amplitudes = []
        for channel_id in range(len(channel_shank_map)):
            if not channel_id in shank_channels[0]:
                continue
            diff = channel_coordinates[max_channel] - channel_coordinates[channel_id]
            dist = np.sqrt(np.dot(diff, diff))
            lookup_channel = channel_id % 32
            amp = np.max(mean_wf[lookup_channel, :]) - np.min(mean_wf[lookup_channel, :])
            channel_distances.append(dist)
            channel_template_amplitudes.append(amp)
        channel_distances = np.array(channel_distances)
        channel_template_amplitudes = np.array(channel_template_amplitudes)

        ax = []
        fig = plt.figure(cluster_id)
        ax.append(plt.subplot(1, 2, 1))
        ax[-1].semilogy(channel_distances, channel_template_amplitudes, 'ko')
        plt.xlabel('Distance ($\mu$m)')
        plt.ylabel('Waveform amplitude (a.u.)')
        ax.append(plt.subplot(1, 2, 2))
        ax[-1].scatter(np.squeeze(channel_coordinates[shank_channels, 0]), np.squeeze(channel_coordinates[shank_channels, 1]),
                       c=np.log(channel_template_amplitudes))
        plt.xlabel('X pos ($\mu$m)')
        plt.ylabel('Y pos ($\mu$m)')
        plt.subplots_adjust(top=0.9, bottom=0.13, left=0.097, right=0.971, hspace=0.2, wspace=0.306)
        fig.suptitle('Cluster %d' % cluster_id)
        pdf_name = 'ClusterMotifWaveformAmplitude_%d.pdf' % cluster_id
        out_name = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'cluster_properties', pdf_name)
        plt.savefig(out_name)
        # plt.show()


def individual_burst_shapes(experiment_info_name):
    # find out if individual bursts across motifs and different bursts within motif
    # have similar or different shapes (i.e., are multiple bursts coming from same
    # or different units?)
    # load stuff
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get motif template
    audio_fs, audio_data = cp.reader.read_template_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_trace(audio_data, -1.0, 1.0)
    # get clusters
    data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)
    burst_cluster_ids, burst_cluster_nr = np.loadtxt(os.path.join(cluster_folder, 'cluster_burst.tsv'), skiprows=1,
                                                     unpack=True)

    channel_shank_map = np.load(os.path.join(cluster_folder, 'channel_shank_map.npy'))
    channel_positions = np.load(os.path.join(cluster_folder, 'channel_positions.npy'))

    # got through all clusters
    # for cluster_id in clusters_of_interest:
    for cluster_id in [1072, 1073]:
    # for cluster_id in burst_cluster_ids:
        cluster = clusters[cluster_id]
        if cluster.shank == 3 or cluster.shank == 4:
            continue
        spike_times = cluster.spiketrains[0]
        # for each spike time, determine if within motif
        motif_spike_times = []
        motif_event_times = []
        motif_event_spikes = []
        motif_times = []
        motif_spikes = np.zeros(len(spike_times), dtype='int')
        # motif object with attributes start, stop, center and warp (and more not relevant here)
        for i in range(len(motif_finder_data.start)):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            motif_warp = motif_finder_data.warp[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
            # in case template boundaries are so large that they reach into beginning of next motif in bout
            # we only want to assign spikes once
            # does not influence bursts WITHIN syllables
            duplicate_spikes = motif_spikes * selection
            selection -= np.array(duplicate_spikes, dtype='bool')
            motif_spikes += selection
            # scale spike time within motif by warp factor
            if np.sum(selection):
                motif_spike_times_trial = (spike_times.magnitude[selection] - motif_start) / motif_warp
                spike_times_unwarped = spike_times.magnitude[selection] - motif_start
            else:
                motif_spike_times_trial = []
                spike_times_unwarped = []
            motif_spike_times.append(motif_spike_times_trial)
            motif_times_trial = [0, (motif_stop - motif_start) / motif_warp]
            motif_times.append(motif_times_trial)
            # for all spikes in a motif, get event times (event: all successive spikes with <= 10 ms ISI)
            print 'Getting event times from %d spikes' % (len(motif_spike_times_trial))
            event_times_trial, event_spikes_trial = utils.event_times_from_spikes(spike_times_unwarped, 10.0)
            motif_event_times.append(event_times_trial)
            motif_event_spikes.append(event_spikes_trial)

        ###########################
        # Manual burst selection
        ###########################
        fig = plt.figure(cluster_id)
        # left
        # motif-aligned raster plot
        ax1 = plt.subplot(1, 1, 1)
        title_str = 'Cluster %d raster plot; shank %d' % (cluster_id, cluster.shank)
        ax1.set_title(title_str)
        ax1.eventplot(motif_spike_times, colors='k', linewidths=0.5)
        ax1.eventplot(motif_times, colors='r', linewidths=1.0)
        t_audio = np.linspace(motif_times[0][0], motif_times[0][1], len(plot_audio))
        ax1.plot(t_audio, plot_audio + len(motif_times) + 2, 'k', linewidth=0.5)
        binsize = 0.005
        motif_spike_hist, motif_spike_bins = utils.mean_firing_rate_from_aligned_spikes(motif_spike_times, motif_times[0][0], motif_times[0][1],
                                                                binsize=binsize)
        hist_norm = utils.normalize_trace(motif_spike_hist, 0.0, 5.0)
        ax1.plot(0.5 * (motif_spike_bins[:-1] + motif_spike_bins[1:]), hist_norm - 6.0, 'k-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Motif nr.')
        # manually select bursts to check
        burst_times = []
        bp = utils.BurstPicker(ax1, burst_times)
        bp.connect()
        plt.show()
        plt.close(fig)

        burst_onsets = burst_times[::2]
        burst_offsets = burst_times[1::2]
        # for visualization
        motif_burst_spike_times = dict(zip(range(len(burst_onsets)), [[] for i in range(len(burst_onsets))]))
        spike_cnt = 0
        # use this to the indices of only the non-zero elements of motif_spikes
        burst_indices = dict(zip(range(len(burst_onsets)), [[] for i in range(len(burst_onsets))]))
        for trial, motif_spike_times_trial in enumerate(motif_spike_times):
            motif_burst_spike_times_trial = [[] for i in range(len(burst_onsets))]
            for i in range(len(burst_onsets)):
                burst_indices[i].append([])
            # sanity check
            assigned_spikes = []
            for t in motif_spike_times_trial:
                if t in assigned_spikes:
                    print 'WARNING! Burst on- and offsets seem to be overlapping...'
                    continue
                for i in range(len(burst_onsets)):
                    if burst_onsets[i] <= t <= burst_offsets[i]:
                        motif_burst_spike_times_trial[i].append(t)
                        burst_indices[i][trial].append(spike_cnt)
                        assigned_spikes.append(t)
                spike_cnt += 1
            for i in range(len(burst_onsets)):
                motif_burst_spike_times[i].append(motif_burst_spike_times_trial[i])

        burst_times_waveforms = dict(zip(range(len(burst_onsets)), [[] for i in range(len(burst_onsets))]))
        for burst_id in burst_times_waveforms:
            print 'Loading motif waveforms for burst %d...' % burst_id
            for trial in range(len(burst_indices[burst_id])):
                indices = burst_indices[burst_id][trial]
                spike_time_indices = np.where(motif_spikes > 0)[0][indices]
                burst_spike_times = spike_times[spike_time_indices].magnitude
                burst_spike_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info,
                                                                                          channel_shank_map,
                                                                                          cluster,
                                                                                          spike_times[spike_time_indices])
                burst_times_waveforms[burst_id].append((burst_spike_times, burst_spike_waveforms))

        # get waveforms of all spikes within motifs (to sort electrodes according to mean amplitude)
        all_motif_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info, channel_shank_map,
                                                                            cluster, spike_times[np.where(motif_spikes > 0)])
        mean_motif_waveform_old = np.mean(all_motif_waveforms, axis=0)
        amplitude_per_electrode_old = np.max(mean_motif_waveform_old, axis=1) - np.min(mean_motif_waveform_old, axis=1)
        sorted_electrodes_old = np.argsort(amplitude_per_electrode_old)
        sorted_electrodes_old = sorted_electrodes_old[::-1]
        # extract spike waveforms outside of motifs
        # also keep spike time so we can later only compare spikes
        # within certain time window around motifs (work around drift...)
        outside_motif_spikes = 1 - motif_spikes
        if np.sum(outside_motif_spikes):
            # print 'Loading outside motif waveforms...'
            tmp_cluster = copy.deepcopy(cluster)
            tmp_cluster.spiketrains[0] = tmp_cluster.spiketrains[0][np.where(outside_motif_spikes > 0)]
            # outside_motif_waveforms = cp.reader.load_cluster_waveforms_random_sample(experiment_info,
            #                                                                          channel_shank_map,
            #                                                                          {cluster_id: tmp_cluster},
            #                                                                          n_spikes=100)
            outside_motif_spike_times = tmp_cluster.spiketrains[0].magnitude
        else:
            # outside_motif_waveforms = None
            outside_motif_spike_times = None

        ###########################
        # compare burst waveforms within motif vs. waveform of same burst across motifs
        # within motifs:
        #   overlay burst waveforms
        #   overlay burst PC features
        # across motifs:
        #   overlay waveforms across trials
        #   overlay features across trials
        # see if across trial feature range matches within trial feature range
        # if not, we may have erroneously joined single bursts
        ###########################

        fig = plt.figure(cluster_id)
        colors = ('r', 'g', 'b', 'y', 'c', 'm', 'grey')
        # left
        # motif-aligned raster plot
        ax1 = plt.subplot(1, 2, 1)
        title_str = 'Cluster %d raster plot; shank %d' % (cluster_id, cluster.shank)
        ax1.set_title(title_str)
        ax1.eventplot(motif_spike_times, colors='k', linewidths=0.5)
        for burst_id in range(len(burst_onsets)):
            ax1.eventplot(motif_burst_spike_times[burst_id], colors=colors[burst_id], linewidths=0.5)
        ax1.eventplot(motif_times, colors='r', linewidths=1.0)
        t_audio = np.linspace(motif_times[0][0], motif_times[0][1], len(plot_audio))
        ax1.plot(t_audio, plot_audio + len(motif_times) + 2, 'k', linewidth=0.5)
        hist_norm = utils.normalize_trace(motif_spike_hist, 0.0, 5.0)
        ax1.plot(0.5*(motif_spike_bins[:-1] + motif_spike_bins[1:]), hist_norm - 6.0, 'k-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Motif nr.')

        # amplitude/amplitude similarity plots
        shank_channels = np.where(channel_shank_map == cluster.shank)[0]
        # top right
        # Amplitude distribution per burst per motif
        ax2 = plt.subplot(4, 4, 3)
        ax2.set_title('Amplitude dist. in motifs')
        burst_waveform_amplitudes = dict(zip(range(len(burst_onsets)), [[] for i in range(len(burst_onsets))]))
        # waveforms shape:
        # (len(spike_times), channels, wf_samples)
        # first, determine electrode sorting based on mean first spike in bursts
        tmp_first_waveforms = []
        for burst_id in burst_times_waveforms:
            for trial_nr, trial in enumerate(burst_times_waveforms[burst_id]):
                times, waveforms = trial
                if not len(times):
                    continue
                tmp_first_waveforms.append(waveforms[0, :, :])
        tmp_first_waveforms = np.array(tmp_first_waveforms)
        mean_first_waveform = np.mean(tmp_first_waveforms, axis=0)
        # amplitude_per_electrode = np.max(mean_first_waveform, axis=1) - np.min(mean_first_waveform, axis=1)
        amplitude_per_electrode = np.max(np.abs(mean_first_waveform), axis=1)
        amplitude_index = np.argmax(np.abs(mean_first_waveform), axis=1)
        sorted_electrodes = np.argsort(amplitude_per_electrode)
        sorted_electrodes = sorted_electrodes[::-1]

        for burst_id in burst_times_waveforms:
            for trial_nr, trial in enumerate(burst_times_waveforms[burst_id]):
                times, waveforms = trial
                if not len(times):
                    burst_waveform_amplitudes[burst_id].append(None)
                    continue
                # amplitudes = np.max(waveforms, axis=2) - np.min(waveforms, axis=2)
                # amplitudes = np.abs(waveforms[:, :, amplitude_index])
                amplitudes = np.zeros((waveforms.shape[0], waveforms.shape[1]))
                for i in range(waveforms.shape[0]):
                    for j in range(waveforms.shape[1]):
                        amplitudes[i, j] = np.abs(waveforms[i, j, amplitude_index[j]])
                # # all spikes
                # for i in range(amplitudes.shape[0]):
                #     plot_amplitudes = amplitudes[i, :]/amplitudes[i, sorted_electrodes[0]]
                #     plt.plot(range(len(sorted_electrodes)), plot_amplitudes[sorted_electrodes], colors[burst_id],
                #              linewidth=0.5)
                # first spike
                plot_amplitudes = amplitudes[0, :]/amplitudes[0, sorted_electrodes[0]]
                burst_waveform_amplitudes[burst_id].append(plot_amplitudes[sorted_electrodes])
                ax2.plot(range(len(sorted_electrodes)), plot_amplitudes[sorted_electrodes], colors[burst_id],
                         linewidth=0.5)
        ax2.set_xlabel('Electrode (sorted)')
        ax2.set_ylabel('Amplitude rel. to max channel')
        # top right right
        # motif waveforms
        ax21 = plt.subplot(4, 4, 4)
        ax21.set_title('Waveform per burst')
        for burst_id in burst_times_waveforms:
            all_waveforms = []
            max_channel = sorted_electrodes[0]
            for trial in burst_times_waveforms[burst_id]:
                times, waveforms = trial
                all_waveforms.extend(waveforms)
            all_waveforms = np.array(all_waveforms)
            wf_time_axis = np.arange(all_waveforms.shape[2])*1.0e3/fs
            mean_wf = np.mean(all_waveforms, axis=0)
            wf_5_percentile = np.percentile(all_waveforms, 5, axis=0)
            wf_95_percentile = np.percentile(all_waveforms, 95, axis=0)
            ax21.plot(wf_time_axis, mean_wf[max_channel, :], colors[burst_id], linewidth=0.5)
            ax21.plot(wf_time_axis, wf_5_percentile[max_channel, :], colors[burst_id], linewidth=0.5, linestyle='--')
            ax21.plot(wf_time_axis, wf_95_percentile[max_channel, :], colors[burst_id], linewidth=0.5, linestyle='--')
        ax21.set_xlabel('Time (ms)')
        ax21.set_ylabel('Amplitude')

        ax22 = plt.subplot(4, 2, 4)
        ax22.set_title('Amplitude dist. in motifs (avg. per burst)')
        for burst_id in burst_waveform_amplitudes:
            tmp_amplitudes = []
            for i in range(len(burst_waveform_amplitudes[burst_id])):
                if burst_waveform_amplitudes[burst_id][i] is not None:
                    tmp_amplitudes.append(burst_waveform_amplitudes[burst_id][i])
            tmp_amplitudes = np.array(tmp_amplitudes)
            burst_mean_amplitudes = np.mean(tmp_amplitudes, axis=0)
            burst_se_amplitudes = np.std(tmp_amplitudes, axis=0)/np.sqrt(tmp_amplitudes.shape[0])
            ax22.plot(range(len(sorted_electrodes)), burst_mean_amplitudes, colors[burst_id], linewidth=0.5)
            ax22.plot(range(len(sorted_electrodes)), burst_mean_amplitudes + burst_se_amplitudes, colors[burst_id],
                     linewidth=0.5, linestyle='--')
            ax22.plot(range(len(sorted_electrodes)), burst_mean_amplitudes - burst_se_amplitudes, colors[burst_id],
                     linewidth=0.5, linestyle='--')
        ax22.set_xlabel('Electrode (sorted)')
        ax22.set_ylabel('Amplitude rel. to max channel')

        # bottom right
        # outside motif waveforms
        outside_window = 5*60.0 # minutes to seconds conversion; look in +- this window (i.e. twice as long)
        nr_electrodes_similarity = 8
        ax3 = plt.subplot(4, 2, 6)
        ax3.set_title('Amplitude dist. outside motifs')
        # waveforms shape:
        # (len(spike_times), channels, wf_samples)
        burst_waveform_similarities = dict(zip(range(len(burst_onsets)), [([], []) for i in range(len(burst_onsets))]))
        if outside_motif_spike_times is not None:
            all_outside_spike_times = []
            print 'Loading outside motif waveforms...'
            # dict[burst_id]: (trial_id_list, similarity_list)
            for trial_nr in range(len(motif_finder_data.start)):
                motif_start = motif_finder_data.start[trial_nr]
                motif_stop = motif_finder_data.stop[trial_nr]
                selection = np.where((outside_motif_spike_times >= motif_start - outside_window)
                                     * (outside_motif_spike_times < motif_start)
                                     + (outside_motif_spike_times > motif_stop)
                                     * (outside_motif_spike_times <= motif_stop + outside_window))
                tmp_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info,
                                                                                  channel_shank_map,
                                                                                  cluster,
                                                                                  spike_times[selection])
                all_outside_spike_times.extend(spike_times[selection])
                if not len(tmp_waveforms):
                    continue

                # tmp_amplitudes = np.max(tmp_waveforms, axis=2) - np.min(tmp_waveforms, axis=2)
                # tmp_amplitudes = np.abs(tmp_waveforms[:, :, amplitude_index])
                tmp_amplitudes = np.zeros((tmp_waveforms.shape[0], tmp_waveforms.shape[1]))
                for i in range(tmp_waveforms.shape[0]):
                    for j in range(tmp_waveforms.shape[1]):
                        tmp_amplitudes[i, j] = np.abs(tmp_waveforms[i, j, amplitude_index[j]])
                norm_amplitudes = np.zeros(tmp_amplitudes.shape)
                for i in range(tmp_amplitudes.shape[0]):
                    norm_amplitudes[i, :] = tmp_amplitudes[i, :]/tmp_amplitudes[i, sorted_electrodes[0]]
                norm_amplitudes = norm_amplitudes[:, sorted_electrodes]
                for burst_id in burst_waveform_amplitudes:
                    trial_amplitude = burst_waveform_amplitudes[burst_id][trial_nr]
                    if trial_amplitude is None:
                        continue
                    amplitude_similarity = np.zeros(norm_amplitudes.shape[0])
                    for i in range((len(amplitude_similarity))):
                        amplitude_similarity[i] = np.dot(norm_amplitudes[i][:nr_electrodes_similarity], trial_amplitude[:nr_electrodes_similarity]) #/ \
                                                  # np.dot(norm_amplitudes[i][:nr_electrodes_similarity], norm_amplitudes[i][:nr_electrodes_similarity]) / \
                                                  # np.dot(trial_amplitude[:nr_electrodes_similarity], trial_amplitude[:nr_electrodes_similarity])
                    for i in range(len(amplitude_similarity)):
                        burst_waveform_similarities[burst_id][0].append(trial_nr)
                    burst_waveform_similarities[burst_id][1].extend(amplitude_similarity)
            # plot it
            for burst_id in burst_waveform_similarities:
                ax3.plot(burst_waveform_similarities[burst_id][0], burst_waveform_similarities[burst_id][1],
                         colors[burst_id], marker='o', markersize=0.5, linestyle='')

            # plot spont. waveforms
            spontaneous_spike_times = np.unique(all_outside_spike_times)
            spontaneous_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info,
                                                                                      channel_shank_map,
                                                                                      cluster,
                                                                                      spontaneous_spike_times)
            wf_time_axis = np.arange(spontaneous_waveforms.shape[2])*1.0e3/fs
            max_channel = sorted_electrodes[0]
            mean_wf = np.mean(spontaneous_waveforms, axis=0)
            wf_5_percentile = np.percentile(spontaneous_waveforms, 5, axis=0)
            wf_95_percentile = np.percentile(spontaneous_waveforms, 95, axis=0)
            ax21.plot(wf_time_axis, mean_wf[max_channel, :], 'k', linewidth=0.5)
            ax21.plot(wf_time_axis, wf_5_percentile[max_channel, :], 'k', linewidth=0.5, linestyle='--')
            ax21.plot(wf_time_axis, wf_95_percentile[max_channel, :], 'k', linewidth=0.5, linestyle='--')

        ax3.set_xlabel('Trial nr.')
        ax3.set_ylabel('Similarity')

        ax32 = plt.subplot(4, 2, 8)
        ax32.set_title('Amplitude dist. outside motifs (avg. per burst)')
        for burst_id in burst_waveform_similarities:
            tmp_all_trials = np.array(burst_waveform_similarities[burst_id][0])
            tmp_all_similarities = np.array(burst_waveform_similarities[burst_id][1])
            trial_ids = np.unique(burst_waveform_similarities[burst_id][0])
            similarity_mean = np.zeros(trial_ids.shape)
            similarity_se = np.zeros(trial_ids.shape)
            for i, tmp_id in enumerate(trial_ids):
                similarities = tmp_all_similarities[np.where(tmp_all_trials == tmp_id)]
                similarity_mean[i] = np.mean(similarities)
                similarity_se[i] = np.std(similarities)/np.sqrt(len(similarities))
            ax32.plot(trial_ids, similarity_mean, colors[burst_id], linewidth=0.5)
            ax32.plot(trial_ids, similarity_mean + similarity_se, colors[burst_id], linewidth=0.5, linestyle='--')
            ax32.plot(trial_ids, similarity_mean - similarity_se, colors[burst_id], linewidth=0.5, linestyle='--')
        ax32.set_xlabel('Trial nr.')
        ax32.set_ylabel('Similarity')

        summary_fig_suffix = 'burst_similarities_cluster_%d.pdf' % cluster_id
        summary_fig_fname = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'burst_identity', summary_fig_suffix)
        fig.set_size_inches(11, 8)
        plt.savefig(summary_fig_fname)
        plt.show()

        summary_suffix = 'burst_similarities_cluster_%d.csv' % cluster_id
        summary_fname = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'burst_identity', summary_suffix)
        with open(summary_fname, 'w') as summary_file:
            burst_ids = burst_waveform_amplitudes.keys()
            burst_ids.sort()
            header = 'Burst ID'
            for i in range(len(burst_ids)):
                header += '\t'
                header += str(burst_ids[i])
            header += '\t'
            header += 'Spont.'
            header += '\n'
            summary_file.write(header)
            for i, burst_id1 in enumerate(burst_ids):
                # compare with spontaneous
                spont_similarities = np.array(burst_waveform_similarities[burst_id1][1])
                # spont_threshold = np.percentile(spont_similarities_, 95)
                # spont_similarities = spont_similarities_[np.where(spont_similarities_ < spont_threshold)]
                burst1_amplitudes = burst_waveform_amplitudes[burst_id1]
                burst1_similarities = _amplitude_pairwise_similarity(burst1_amplitudes, burst1_amplitudes, nr_electrodes_similarity)
                # burst1_threshold = np.percentile(burst1_similarities_, 95)
                # burst1_similarities = burst1_similarities_[np.where(burst1_similarities_ < burst1_threshold)]
                t_spont, p_spont = stats.ttest_ind(spont_similarities, burst1_similarities, equal_var=False)
                # w_spont, p_spont = stats.wilcoxon(spont_similarities, burst1_similarities)
                line = str(burst_id1)
                # compare with other bursts
                for j, burst_id2 in enumerate(burst_ids):
                    if i == j:
                        line += '\t'
                        line += '%.2f +- %.2f' % (np.mean(burst1_similarities), np.std(burst1_similarities))
                        continue
                    burst2_amplitudes = burst_waveform_amplitudes[burst_id2]
                    burst12_similarities = _amplitude_pairwise_similarity(burst1_amplitudes, burst2_amplitudes, nr_electrodes_similarity)
                    # burst12_threshold = np.percentile(burst12_similarities_, 95)
                    # burst12_similarities = burst12_similarities_[np.where(burst12_similarities_ < burst12_threshold)]
                    t, p = stats.ttest_ind(burst12_similarities, burst1_similarities, equal_var=False)
                    # w, p = stats.wilcoxon(burst12_similarities, burst1_similarities)
                    line += '\t'
                    line += '%.2f +- %.2f; p=%.3f' % (np.mean(burst12_similarities), np.std(burst12_similarities), p)
                    # if i == 0 and j == 2 or i == 0 and j == 3:
                    #     dummy = 1
                line += '\t'
                line += '%.2f +- %.2f; p=%.3f' % (np.mean(spont_similarities), np.std(spont_similarities), p_spont)
                line += '\n'
                summary_file.write(line)
            if cluster_id == 777:
                burst1_amplitudes = []
                burst1_amplitudes.extend(burst_waveform_amplitudes[0])
                burst1_amplitudes.extend(burst_waveform_amplitudes[2])
                burst2_amplitudes = []
                burst2_amplitudes.extend(burst_waveform_amplitudes[1])
                burst2_amplitudes.extend(burst_waveform_amplitudes[3])
                burst1_similarities = _amplitude_pairwise_similarity(burst1_amplitudes, burst1_amplitudes,
                                                                     nr_electrodes_similarity)
                burst2_similarities = _amplitude_pairwise_similarity(burst2_amplitudes, burst2_amplitudes,
                                                                     nr_electrodes_similarity)
                burst12_similarities = _amplitude_pairwise_similarity(burst1_amplitudes, burst2_amplitudes,
                                                                      nr_electrodes_similarity)
                t, p = stats.ttest_ind(burst12_similarities, burst1_similarities)
                line = 'Burst0+2: %.2f +- %.2f' % (np.mean(burst1_similarities), np.std(burst1_similarities))
                line += '\n'
                summary_file.write(line)
                line = 'Burst1+3: %.2f +- %.2f' % (np.mean(burst2_similarities), np.std(burst2_similarities))
                line += '\n'
                summary_file.write(line)
                line = 'Burst0+2/1+3: %.2f +- %.2f; p=%.3f' % (np.mean(burst12_similarities), np.std(burst12_similarities), p)
                line += '\n'
                summary_file.write(line)


        plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        experiment_info = sys.argv[1]
        # template_extent_spatial(experiment_info)
        individual_burst_shapes(experiment_info)