import os
import ast
import sys
import copy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import ClusterProcessing as cp
import utilities as utils

clustering_src_folder = 'E:\\User\\project_src\\physiology\\Clustering'
clusters_of_interest = [1, 9, 45, 46, 55, 58, 108, 128, 135, 209, 244, 266, 304, 309, 337, 353, 388, 454, 469, 578,
                        685, 701, 702, 705, 721, 733, 738, 759, 760, 761, 772, 779] # bursters that look the most promising


def cut_motifs_from_full_audio(experiment_info_name, out_folder):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get motif template
    audio_name = os.path.join(experiment_info['Motifs']['DataBasePath'], experiment_info['Motifs']['AudioFilename'])
    audio_fs, audio_data = cp.reader.read_audiofile(audio_name)

    # motif object with attributes start, stop, and more not relevant here
    n_motifs = len(motif_finder_data.start)
    for i in range(n_motifs):
        motif_start = motif_finder_data.start[i]
        motif_stop = motif_finder_data.stop[i]
        motif_start_sample = int(motif_start*audio_fs)
        motif_stop_sample = int(motif_stop*audio_fs)
        motif_audio = audio_data[motif_start_sample:motif_stop_sample]
        motif_name_suffix = 'motif_%d.wav' % i
        motif_name = os.path.join(out_folder, motif_name_suffix)
        print 'Writing motif %d of %d...' % (i+1, n_motifs)
        cp.writer.write_wav_file(motif_name, audio_fs, motif_audio)


def motif_aligned_rasters(experiment_info_name):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # go through all clusters
    data_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    clusters = cp.reader.read_all_clusters_except_noise(data_folder, 'dev', fs)
    # clusters = clust.reader.read_KS_clusters(data_folder, clustering_src_folder, 'dev', ('good',), fs)
    # for cluster_id in clusters:
    for cluster_id in [870, 872, 873]:
        # go through all spike times
        spike_times = clusters[cluster_id].spiketrains[0]
        # for each spike time, determine if within motif
        motif_spike_times = []
        motif_times = []
        # motif object with attributes start, stop, center and warp (and more not relevant here)
        for i in range(len(motif_finder_data.start)):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            motif_warp = motif_finder_data.warp[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
            # scale spike time within motif by warp factor
            if np.sum(selection):
                motif_spike_times_trial = (spike_times.magnitude[selection] - motif_start)/motif_warp
            else:
                motif_spike_times_trial = []
            motif_spike_times.append(motif_spike_times_trial)
            motif_times_trial = [0, (motif_stop - motif_start)/motif_warp]
            motif_times.append(motif_times_trial)
        # create motif-aligned raster plot
        plt.figure(cluster_id)
        plt.eventplot(motif_spike_times, colors='k')
        plt.eventplot(motif_times, colors='r')
        plt.title(str(cluster_id))
        plt.xlabel('Time (s)')
        plt.ylabel('Motif nr.')
        # plt.show()
        pdf_name = 'BurstCluster_%d.pdf' % cluster_id
        out_name = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'motif_aligned_rasters', pdf_name)
        plt.savefig(out_name)


def motif_aligned_traces(experiment_info_name, cluster_id_):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get motif template
    audio_fs, audio_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_trace(audio_data, -1.0, 1.0)
    # get clusters
    data_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    # clusters = _read_all_clusters_except_noise(data_folder, 'dev', fs)
    clusters = cp.reader.read_KS_clusters(data_folder, clustering_src_folder, 'dev', ('good',), fs)

    raw_traces = cp.reader.load_recording(os.path.join(data_folder, experiment_info['SiProbe']['AmplifierName']),
                                          experiment_info['SiProbe']['Channels'])
    intan_constant = 0.195

    b, a = utils.set_up_bp_filter(300, 0.49*fs, fs)
    def bp_filter(x):
        return signal.filtfilt(b, a, x, axis=1)

    for cluster_id in clusters_of_interest:
    # for cluster_id in [701]:
        cluster = clusters[cluster_id]
        spike_times = clusters[cluster_id].spiketrains[0]
        # for each spike time, determine if within motif
        motif_spike_times = []
        motif_times = []
        motif_traces = []
        # motif object with attributes start, stop, center and warp (and more not relevant here)
        for i in range(len(motif_finder_data.start)):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            motif_warp = motif_finder_data.warp[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
            # scale spike time within motif by warp factor
            if np.sum(selection):
                motif_spike_times_trial = (spike_times.magnitude[selection] - motif_start) / motif_warp
            else:
                motif_spike_times_trial = []
            motif_spike_times.append(motif_spike_times_trial)
            motif_times_trial = [0, (motif_stop - motif_start) / motif_warp]
            motif_times.append(motif_times_trial)
            print 'Loading and filtering max channel (%d) trace in trial %d...' % (cluster.maxChannel, i)
            raw_snippet = intan_constant * cp.reader.load_recording_chunk(raw_traces, (cluster.maxChannel,), motif_start, motif_stop, fs)
            motif_trace = bp_filter(raw_snippet)
            motif_traces.append(motif_trace)
        # create motif-aligned raster plot
        plt.figure(cluster_id)
        for i in range(len(motif_traces)):
            plot_trace = utils.normalize_trace(motif_traces[i], -0.5, 0.5) + 2*i + 1
            plot_trace = plot_trace.flatten()
            t_i = np.linspace(motif_times[i][0], motif_times[i][1], len(plot_trace))
            plt.plot(t_i, plot_trace, 'k', linewidth=0.5)
        plt.eventplot(motif_spike_times, colors='b', lineoffsets=2.0, linelengths=0.5)
        plt.eventplot(motif_times, colors='r', lineoffsets=2.0, linelengths=2.0)
        t_audio = np.linspace(motif_times[0][0], motif_times[0][1], len(plot_audio))
        plt.plot(t_audio, plot_audio + 2*len(motif_times) + 2, 'k', linewidth=0.5)
        plt.title(str(cluster_id))
        plt.xlabel('Time (s)')
        plt.ylabel('Motif nr.')
        # plt.show()
        pdf_name = 'TraceCluster_%d.pdf' % cluster_id
        out_name = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'motif_aligned_rasters', pdf_name)
        plt.savefig(out_name)


def motif_aligned_spike_snippets(experiment_info_name):
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get motif template
    audio_fs, audio_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_trace(audio_data, -1.0, 1.0)
    # get clusters
    data_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    # clusters = _read_all_clusters_except_noise(data_folder, 'dev', fs)
    clusters = cp.reader.read_KS_clusters(data_folder, clustering_src_folder, 'dev', ('good',), fs)

    raw_traces = cp.reader.load_recording(os.path.join(data_folder, experiment_info['SiProbe']['AmplifierName']),
                                          experiment_info['SiProbe']['Channels'])
    intan_constant = 0.195

    b, a = utils.set_up_bp_filter(300, 0.49*fs, fs)
    def bp_filter(x):
        return signal.filtfilt(b, a, x, axis=1)

    snippet_padding = 0.001  # 1 ms

    for cluster_id in clusters_of_interest:
    # for cluster_id in [701]:
        cluster = clusters[cluster_id]
        spike_times = cluster.spiketrains[0]
        # for each spike time, determine if within motif
        motif_spike_times = []
        motif_event_times = []
        motif_event_spikes = []
        motif_times = []
        motif_event_traces = []
        # motif object with attributes start, stop, center and warp (and more not relevant here)
        for i in range(len(motif_finder_data.start)):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            motif_warp = motif_finder_data.warp[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
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
            # for each event, extract waveform  plus 1 ms before first / after last spike
            print 'Loading and filtering max channel (%d) trace in trial %d...' % (cluster.maxChannel, i)
            event_snippets = []
            for event in event_times_trial:
                center, start, stop = event
                snippet_start = motif_start + start - snippet_padding
                snippet_stop = motif_start + stop + snippet_padding
                raw_snippet = intan_constant * cp.reader.load_recording_chunk(raw_traces, (cluster.maxChannel,), snippet_start, snippet_stop, fs)
                snippet_trace = bp_filter(raw_snippet)
                event_snippets.append(snippet_trace)
            motif_event_traces.append(event_snippets)
        # create motif-aligned raster plot
        plt.figure(cluster_id)
        for i, event_snippets in enumerate(motif_event_traces):
            for j, event in enumerate(event_snippets):
                # plot at location of event, but blown up in time by ~ 10x
                event_center, event_start, event_stop = motif_event_times[i][j]
                event_duration = event_stop - event_start
                event_scale = 10.0
                motif_warp = motif_finder_data.warp[i]
                plot_start = event_center/motif_warp - event_scale*(0.5*event_duration + snippet_padding)
                plot_stop = event_center/motif_warp + event_scale*(0.5*event_duration + snippet_padding)
                plot_trace = utils.normalize_trace(event, -0.5, 0.5) + 2*i + 1
                plot_trace = plot_trace.flatten()
                t_i = np.linspace(plot_start, plot_stop, len(plot_trace))
                plt.plot(t_i, plot_trace, 'k', linewidth=0.5)
                spikes = np.array(motif_event_spikes[i][j])
                plot_spikes = event_scale*(spikes - event_center) + event_center/motif_warp
                plt.eventplot(plot_spikes, colors='b', lineoffsets=2.0*i, linelengths=0.5)
        plt.eventplot(motif_times, colors='r', lineoffsets=2.0, linelengths=2.0)
        t_audio = np.linspace(motif_times[0][0], motif_times[0][1], len(plot_audio))
        plt.plot(t_audio, plot_audio + 2*len(motif_times) + 2, 'k', linewidth=0.5)
        plt.title(str(cluster_id))
        plt.xlabel('Time (s)')
        plt.ylabel('Motif nr.')
        # plt.show()
        pdf_name = 'BurstZoomCluster_%d.pdf' % cluster_id
        out_name = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'motif_aligned_rasters', pdf_name)
        plt.savefig(out_name)


def motif_aligned_rasters_spike_waveforms(experiment_info_name):
    """
    Visualization of sorted and/or unsorted clusters
    spike raster plot aligned to motifs
    individual spike waveforms on shank for all spikes within motif
    mean spike waveform (+- 5th/95th percentiles) for all spikes outside of motif
    :param experiment_info_name: parameter file name
    :return: nothing
    """
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())
    # get motif times
    motif_finder_data = cp.reader.read_motifs(os.path.join(experiment_info['Motifs']['DataBasePath'],
                                                           experiment_info['Motifs']['MotifFilename']))
    # get motif template
    audio_fs, audio_data = cp.reader.read_audiofile(experiment_info['Motifs']['TemplateFilename'])
    plot_audio = utils.normalize_trace(audio_data, -1.0, 1.0)
    # get clusters
    data_folder = experiment_info['SiProbe']['DataBasePath']
    cluster_folder = experiment_info['SiProbe']['ClusterBasePath']
    fs = experiment_info['SiProbe']['SamplingRate']
    clusters = cp.reader.read_all_clusters_except_noise(cluster_folder, 'dev', fs)
    # clusters = cp.reader.read_KS_clusters(cluster_folder, clustering_src_folder, 'dev', ('good',), fs)

    burst_clusters = np.loadtxt(os.path.join(cluster_folder, 'cluster_burst.tsv'), skiprows=1, unpack=True)

    channel_shank_map = np.load(os.path.join(cluster_folder, 'channel_shank_map.npy'))
    channel_positions = np.load(os.path.join(cluster_folder, 'channel_positions.npy'))

    # for cluster_id in clusters_of_interest:
    # for cluster_id in [1116, 1129, 1154, 1158, 1166, 1169, 1175, 1205, 1220, 1236, 1247, 1257, 1267, 1268, 1283, 1288,
    #                    1298, 1302, 1303, 1309, 1314, 1330, 1340, 1346, 1367, 1374, 1376]:
    # for cluster_id in [795, 796]:
    for cluster_id in clusters:
        if cluster_id not in burst_clusters[0]:
            continue
        cluster = clusters[cluster_id]
        spike_times = cluster.spiketrains[0]
        # for each spike time, determine if within motif
        motif_spike_times = []
        motif_times = []
        motif_spikes = np.zeros(len(spike_times), dtype='int')
        # motif object with attributes start, stop, center and warp (and more not relevant here)
        for i in range(len(motif_finder_data.start)):
            motif_start = motif_finder_data.start[i]
            motif_stop = motif_finder_data.stop[i]
            motif_warp = motif_finder_data.warp[i]
            selection = (spike_times.magnitude >= motif_start) * (spike_times.magnitude <= motif_stop)
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
        # get waveforms of spikes within motifs
        if np.sum(motif_spikes):
            print 'Loading motif waveforms...'
            motif_waveforms = cp.reader.load_cluster_waveforms_from_spike_times(experiment_info, channel_shank_map,
                                                                                cluster, spike_times[np.where(motif_spikes > 0)])
        else:
            motif_waveforms = None
        # extract all spike waveforms outside of motifs
        outside_motif_spikes = 1 - motif_spikes
        if np.sum(outside_motif_spikes):
            print 'Loading outside motif waveforms...'
            tmp_cluster = copy.deepcopy(cluster)
            tmp_cluster.spiketrains[0] = tmp_cluster.spiketrains[0][np.where(outside_motif_spikes > 0)]
            outside_motif_waveforms = cp.reader.load_cluster_waveforms_random_sample(experiment_info,
                                                                                     channel_shank_map,
                                                                                     {cluster_id: tmp_cluster},
                                                                                     n_spikes=100)
        else:
            outside_motif_waveforms = None
        fig = plt.figure(cluster_id)
        # left
        # motif-aligned raster plot
        ax1 = plt.subplot(1, 2, 1)
        title_str = 'Cluster %d raster plot; shank %d' % (cluster_id, cluster.shank)
        ax1.set_title(title_str)
        ax1.eventplot(motif_spike_times, colors='k', linewidths=0.5)
        ax1.eventplot(motif_times, colors='r', linewidths=1.0)
        t_audio = np.linspace(motif_times[0][0], motif_times[0][1], len(plot_audio))
        ax1.plot(t_audio, plot_audio + len(motif_times) + 2, 'k', linewidth=0.5)
        hist, bins = utils.mean_firing_rate_from_aligned_spikes(motif_spike_times, motif_times[0][0], motif_times[0][1],
                                                                binsize=0.005)
        hist_norm = utils.normalize_trace(hist, 0.0, 5.0)
        ax1.plot(0.5*(bins[:-1] + bins[1:]), hist_norm - 6.0, 'k-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Motif nr.')
        # waveform plots
        shank_channels = np.where(channel_shank_map == cluster.shank)[0]
        # top right
        # motif waveforms
        wf_time_scale = 10.0
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('Waveforms in motifs')
        # waveforms shape:
        # (len(spike_times), channels, wf_samples)
        if motif_waveforms is not None:
            wf_time_axis = wf_time_scale*np.arange(motif_waveforms.shape[2])*1.0e3/fs
            mean_wf = np.mean(motif_waveforms, axis=0)
            wf_5_percentile = np.percentile(motif_waveforms, 5, axis=0)
            wf_95_percentile = np.percentile(motif_waveforms, 95, axis=0)
            for i, channel in enumerate(shank_channels):
                channel_loc = channel_positions[channel]
                ax2.plot(wf_time_axis + channel_loc[0], mean_wf[i, :] + 15.0 * channel_loc[1],
                         'k', linewidth=0.5)
                ax2.plot(wf_time_axis + channel_loc[0], wf_5_percentile[i, :] + 15.0 * channel_loc[1],
                         'k--', linewidth=0.5)
                ax2.plot(wf_time_axis + channel_loc[0], wf_95_percentile[i, :] + 15.0 * channel_loc[1],
                         'k--', linewidth=0.5)
                # for j in range(motif_waveforms.shape[0]):
                #     ax2.plot(wf_time_axis + channel_loc[0], motif_waveforms[j, i, :] + 15.0 * channel_loc[1],
                #              'k', linewidth=0.5)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        # bottom right
        # outside motif waveforms
        ax3 = plt.subplot(2, 2, 4)
        ax3.set_title('Waveforms outside motifs')
        # waveforms shape:
        # (len(spike_times), channels, wf_samples)
        if outside_motif_waveforms is not None:
            tmp_waveforms = outside_motif_waveforms[cluster_id]
            wf_time_axis = wf_time_scale*np.arange(tmp_waveforms.shape[2])*1.0e3/fs
            mean_wf = np.mean(tmp_waveforms, axis=0)
            wf_5_percentile = np.percentile(tmp_waveforms, 5, axis=0)
            wf_95_percentile = np.percentile(tmp_waveforms, 95, axis=0)
            for i, channel in enumerate(shank_channels):
                channel_loc = channel_positions[channel]
                ax3.plot(wf_time_axis + channel_loc[0], mean_wf[i, :] + 15.0 * channel_loc[1],
                         'k', linewidth=0.5)
                ax3.plot(wf_time_axis + channel_loc[0], wf_5_percentile[i, :] + 15.0 * channel_loc[1],
                         'k--', linewidth=0.5)
                ax3.plot(wf_time_axis + channel_loc[0], wf_95_percentile[i, :] + 15.0 * channel_loc[1],
                         'k--', linewidth=0.5)
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        # plt.show()
        pdf_name = 'BurstCandidates_Shank_%d_Cluster_%d.pdf' % (cluster.shank, cluster_id)
        out_folder_name = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'motif_aligned_rasters3')
        if not os.path.exists(out_folder_name):
            os.makedirs(out_folder_name)
        out_name = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], 'motif_aligned_rasters3', pdf_name)
        plt.savefig(out_name)
        plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        info_name = sys.argv[1]
        # motif_aligned_rasters(info_name)
        # motif_aligned_spike_snippets(info_name)
        motif_aligned_rasters_spike_waveforms(info_name)
    elif len(sys.argv) == 3:
        info_name = sys.argv[1]
        # cluster_id = int(sys.argv[2])
        # motif_aligned_traces(info_name, cluster_id)
        motif_audio_folder = sys.argv[2]
        cut_motifs_from_full_audio(info_name, motif_audio_folder)
