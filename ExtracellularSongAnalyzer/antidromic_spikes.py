import os
import ast
import sys
import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
import utilities as utils


stim_protocol = [1, 5, 10, 20, 50, 100, 20, 50, 100]
# data_folder = r'Z:\Vigi\Datasets\SiliconProbe\Masmanadis\stim'
# out_folder = r'Z:\Vigi\Datasets\SiliconProbe\Masmanadis\stim\stimulus_aligned_traces'
# C22 d1
# data_folder = r'F:\sorting\C22\C22_d1_song_stim_period\d1_evening_stim'
# out_folder = r'F:\sorting\C22\C22_d1_song_stim_period\d1_evening_stim'
stimulus_alignment_info_name = r'Z:\Vigi\Datasets\SiliconProbe\Masmanadis\UCLA\Data\stim_protocol_6_28_17.mat'
phy_info_name = r'Z:\Vigi\Datasets\SiliconProbe\Masmanadis\CutAmplifier\batches'
fs = 20000.0


def _set_up_filter(highpass, lowpass, fs):
    filter_order = 3
    return signal.butter(filter_order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def _sort_channels_by_distance(channel_locations):
    # simply greedy sorting:
    # start at bottom channel and iteratively look for
    # closest channel not visited yet
    channel_order = []
    n_channels = channel_locations.shape[0]
    start_channel = np.argsort(channel_locations[:, 1])[0]
    channel_order.append(start_channel)
    while len(channel_order) < n_channels:
        last_channel = channel_order[-1]
        tmp_loc = channel_locations[last_channel]
        min_dist = 1e6
        min_channel = -1
        for i in range(n_channels):
            diff = channel_locations[i] - tmp_loc
            dist2 = np.dot(diff, diff)
            if dist2 < min_dist and i not in channel_order:
                min_dist = dist2
                min_channel = i
        if min_channel == -1:
            e = 'Could not find channel closest to %d' % last_channel
            raise RuntimeError(e)
        channel_order.append(min_channel)

    return np.array(channel_order)


def _get_nearest_channels(channel_ids, channel_locations, center_channel, n_nearest_channels=None):
    """
    sort channels by distance to center_channel and return IDs of n nearest channels
    :param channel_locations: spatial channel map, shape (n_channels, n_spatial_dim)
    :param center_channel: channel from which to compute distances
    :param n_nearest_channels: optional; number of nearest channels to return (default: return all channels sorted)
    :return: channel IDs sorted by distance to center channel (including center channel)
    """
    center_location = channel_locations[center_channel, :]
    channel_distances = []
    n_channels = channel_locations.shape[0]
    for channel in channel_ids:
        diff = channel_locations[channel, :] - center_location
        dist2 = np.dot(diff, diff)
        channel_distances.append(dist2)

    channel_distances_sorted = np.argsort(channel_distances)
    channel_ids_sorted = channel_ids[channel_distances_sorted]
    if n_nearest_channels is not None:
        return channel_ids_sorted[:n_nearest_channels]
    return channel_ids_sorted


def _waveform_similarity(waveforms, channels1, channels2, waveform_window, sample_crossing1, sample_crossing2):
    if len(channels1) != len(channels2):
        e = 'channels1 and channels2 have to be of same length'
        raise RuntimeError(e)
    common_channels = channels1[np.where(channels1 == channels2)]
    if not len(common_channels):
        return 0.0, 0

    snippet1 = np.zeros((len(common_channels), len(waveform_window)))
    snippet2 = np.zeros((len(common_channels), len(waveform_window)))

    window1 = sample_crossing1 + waveform_window
    snippet1_samples = (max(np.min(window1), 0), min(np.max(window1) + 1, waveforms.shape[1]))
    start_diff1 = np.min(window1) if np.min(window1) < 0 else waveforms.shape[1]
    stop_diff1 = waveforms.shape[1] - np.max(window1) - 1 if np.max(window1) + 1 > waveforms.shape[1] else 0
    snippet1[stop_diff1:start_diff1] = waveforms[common_channels, snippet1_samples[0]:snippet1_samples[1]]
    snippet1 = snippet1.flatten()

    window2 = sample_crossing2 + waveform_window
    snippet2_samples = (max(np.min(window2), 0), min(np.max(window2) + 1, waveforms.shape[1]))
    start_diff2 = np.min(window2) if np.min(window2) < 0 else waveforms.shape[1]
    stop_diff2 = waveforms.shape[1] - np.max(window2) - 1 if np.max(window2) + 1 > waveforms.shape[1] else 0
    snippet2[stop_diff2:start_diff2] = waveforms[common_channels, snippet2_samples[0]:snippet2_samples[1]]
    snippet2 = snippet2.flatten()

    return np.dot(snippet1, snippet2)/np.sqrt(np.dot(snippet1, snippet1)*np.dot(snippet2, snippet2)), len(common_channels)


def _aligned_traces_per_channel(raw_traces, stimulus_indices, stimulus_amplitude, stimulus_levels, channel_locations,
                                channel_shank_map):
    intan_constant = 0.195
    window_indices = np.array([i for i in range(-30, 600)]) # 1 ms pre-stim, 20 ms post-stim
    time_axis = window_indices/30.0
    stim_blank_time = 2.5

    # C 22 afternoon
    data_folder = r'Z:\Margot\Experiments\SiProbe\ChronicRecordings\C22\C22_190531_093100\cut_daytime'
    out_folder = r'Z:\Margot\Experiments\SiProbe\ChronicRecordings\C22\C22_190531_093100\cut_daytime\stim'

    b, a = _set_up_filter(300.0, 0.49*fs, fs)
    def bp_filter(x):
        return signal.filtfilt(b, a, x)

    shanks = range(1, 5)
    # shanks = [4]
    for shank in shanks:
        channels = np.where(channel_shank_map == shank)[0]
        # channels = [1]
        # channels = range(96, 128)
        # channels = range(32)
        for level in stimulus_levels:
            plt.figure(level + shank)
            tmp_indices = stimulus_indices[stimulus_amplitude == level]
            print 'Analyzing %d stimuli at level %.0f on shank %d' % (len(tmp_indices), level, shank)
            for channel in channels:
                snippets = np.zeros((len(tmp_indices), len(window_indices)))
                for i, index in enumerate(tmp_indices):
                    snippet_indices = window_indices + index
                    raw_snippet = intan_constant*raw_traces[channel, snippet_indices]
                    filtered_snippet = np.zeros(len(raw_snippet))
                    # filtered_snippet[time_axis >= stim_blank_time] = raw_snippet[time_axis >= stim_blank_time]
                    filtered_snippet[:] = raw_snippet[:]
                    filtered_snippet = bp_filter(filtered_snippet)
                    filtered_snippet[time_axis < stim_blank_time] = 0.0
                    # filtered_snippet[time_axis < 0.0] = raw_snippet[time_axis < 0.0]
                    # filtered_snippet[(time_axis >= 0.0)*(time_axis < stim_blank_time)] = 0.0
                    snippets[i, :] = filtered_snippet
                stimulus_average = np.mean(snippets, axis=0)
                label_str = str(level) + ' muA'
                channel_loc = channel_locations[channel]
                plt.plot(time_axis + channel_loc[0], stimulus_average + 15.0*channel_loc[1], 'k', linewidth=0.5,
                         label=label_str)
                # plt.plot(time_axis, stimulus_average, linewidth=0.5, label=label_str)
            # plt.legend()
            plt.xlabel('Time (ms)')
            plt.ylabel('$\mu$V')
            # plt.xlim([-1.0, 15.0])
            # plt.ylim([-350.0, 250.0])
            plt.title('Channels %d-%d; level %.0f $\mu$A' % (np.min(channels), np.max(channels), level))
            pdf_name = 'Stimulus_%.0f_aligned_traces_all_channels_shank%d.pdf' % (level, shank)
            out_name = os.path.join(out_folder, pdf_name)
            plt.savefig(out_name)
        # plt.show()


def _aligned_traces_map(raw_traces, stimulus_indices, stimulus_amplitude, channel_order):
    intan_constant = 0.195
    window_indices = np.array([i for i in range(-20, 301)])  # 1 ms pre-stim, 15 ms post-stim
    time_axis = window_indices / 20.0

    artifact_cutoff = 2.0

    b, a = _set_up_filter(500.0, 0.49 * fs, fs)

    def bp_filter(x):
        return signal.filtfilt(b, a, x)

    stimulus_levels = [50]
    antidromic_map = np.zeros((len(channel_order), len(window_indices)))
    for i, channel in enumerate(channel_order):
        for level in stimulus_levels:
            tmp_indices = stimulus_indices[stimulus_amplitude == level]
            print 'Analyzing %d stimuli at level %.0f on channel %d' % (len(tmp_indices), level, channel)
            snippets = np.zeros((len(tmp_indices), len(window_indices)))
            for j, index in enumerate(tmp_indices):
                snippet_indices = window_indices + index
                raw_snippet = intan_constant * raw_traces[channel, snippet_indices]
                filtered_snippet = np.zeros(len(raw_snippet))
                filtered_snippet[time_axis >= artifact_cutoff] = raw_snippet[time_axis >= artifact_cutoff]
                filtered_snippet = bp_filter(filtered_snippet)
                filtered_snippet[time_axis < artifact_cutoff] = 0.0
                snippets[j, :] = filtered_snippet
            stimulus_average = np.mean(snippets, axis=0)
            antidromic_map[i, :] = np.clip(stimulus_average, -500.0, 100.0)

    plt.figure(1)
    dt = np.diff(time_axis)[0]
    dy = 1.0
    y, x = np.mgrid[slice(0, len(channel_order), dy),
                    slice(time_axis[0], time_axis[-1] + dt, dt)]
    plt.pcolormesh(x, y, antidromic_map)
    plt.xlabel('Time (ms)')
    plt.ylabel('channel #')
    # plt.xlim([-1.0, 15.0])
    # plt.ylim([-250.0, 100.0])
    plt.title('Channels %d-%d' % (np.min(channel_order), np.max(channel_order)))
    pdf_name = 'Stimulus_aligned_traces_map_shank1.pdf'
    # out_name = os.path.join(out_folder, pdf_name)
    # plt.savefig(out_name)
    plt.show()


def _detect_stimulus_times(recording, t_start, t_stop, fs):
    """
    detect stimulus artifacts between start and stop
    :param recording: recording file
    :param t_start: start of period in which to look (in s)
    :param t_stop: stop of period in which to look (in s)
    :param fs: sampling frequency (in Hz)
    :return: list of stimulus times (in samples)
    """
    stimulus_times = []
    chunk_duration = int(600.0*fs) # in s
    start = int(t_start*fs)
    stop = int(t_stop*fs)
    chunk_start = start
    chunk_stop = start
    count = 0
    while chunk_stop < stop:
        # plt.figure(count)
        chunk_stop = min(stop, chunk_start + chunk_duration)
        chunk = np.mean(recording[:, chunk_start:chunk_stop], axis=0)
        # chunk = recording[0, chunk_start:chunk_stop]
        peaks, properties = scipy.signal.find_peaks(chunk, height=1e3, distance=int(0.9*fs))
        stimulus_times.extend(peaks + chunk_start)
        plt.plot(chunk, 'k-', linewidth=0.5)
        plt.plot(peaks, chunk[peaks], 'ro')
        plt.show()
        count += 1

    stimulus_times = np.array(stimulus_times)
    # stimulus_times = stimulus_times*1.0/fs + t_start
    # stimulus_times += start

    return stimulus_times


def _get_crossing_waveforms_stim_aligned(stim_file, stimulus_times, t_crossing, channel_crossing, fs):
    """
    :param stim_file: binary file
    :param stimulus_times: as the name says
    :param t_crossing: crossing time post-stimulus
    :param channel_crossing: channel on which to extract waveforms
    :return: (n_stim, n_samples) array of waveforms
    """
    intan_constant = 0.195
    b, a = _set_up_filter(300.0, 0.49 * fs, fs)

    def bp_filter(x):
        return signal.filtfilt(b, a, x)

    sample_crossing = int(t_crossing*1e-3*fs) # convert from ms to samples
    waveform_window_indices = np.array([int(-0.5 * 1e-3 * fs) + i for i in range(int(1.5 * 1e-3 * fs))])
    waveforms = np.zeros((len(stimulus_times), len(waveform_window_indices)))
    for i, stim_sample in enumerate(stimulus_times):
        t_index = stim_sample + sample_crossing
        snippet_indices = t_index + waveform_window_indices
        tmp_wf = intan_constant*stim_file[channel_crossing, snippet_indices]
        waveforms[i, :] = bp_filter(tmp_wf)

    return waveforms


def _save_stimulus_aligned_waveforms_crossings(crossing_info, stimulus_indices, stimulus_amplitude, stimulus_levels):
    # with open(crossing_info_name, 'r') as data_file:
    #     crossing_info = ast.literal_eval(data_file.read())
    #
    # # detect stimulus times
    stim_data_folder = crossing_info['Antidromic']['DataBasePath']
    # t_start = 199.0
    # t_stop = 360.0
    fs = crossing_info['Antidromic']['SamplingRate']
    stim_file = utils.load_recording(os.path.join(stim_data_folder, crossing_info['Antidromic']['AmplifierName']),
                                         nchannels=crossing_info['Antidromic']['Channels'])
    # stimulus_times = _detect_stimulus_times(stim_file, t_start, t_stop, fs)
    # # 1-45: 50 muA
    # # 46-93: 100 muA
    # # 94-148: 200 muA
    # stimulus_amplitude = np.zeros(148)
    # stimulus_amplitude[:45] = 50
    # stimulus_amplitude[45:92] = 100
    # stimulus_amplitude[92:147] = 200

    song_data_folder = crossing_info['ContinuousRecording']['ClusterBasePath']
    channel_shank_map = np.load(os.path.join(song_data_folder, 'channel_shank_map.npy'))

    intan_constant = 0.195
    window_indices = np.array([i for i in range(-30, 600)])  # 20 ms post-stim
    window_time_axis = window_indices / 30.0
    threshold = -100.0 # muV
    savepath = crossing_info['Antidromic']['CrossingBasePath']
    stim_blank_time = 2.5 # ms

    b, a = _set_up_filter(300.0, 0.49 * fs, fs)

    def bp_filter(x):
        return signal.filtfilt(b, a, x)

    # stimulus_levels = [50, 100, 200]
    # stimulus_levels = [50]
    shanks = range(1, 5)
    for shank in shanks:
        channels = np.where(channel_shank_map == shank)[0]
        for level in stimulus_levels:
            tmp_indices = stimulus_indices[np.where(stimulus_amplitude == level)]
            shank_threshold_crossings = []
            shank_stimulus_average = np.zeros((len(channels), len(window_indices)))
            print 'Analyzing %d stimuli at level %.0f on shank %d' % (len(tmp_indices), level, shank)
            for i, channel in enumerate(channels):
                snippets = np.zeros((len(tmp_indices), len(window_indices)))
                for j, index in enumerate(tmp_indices):
                    snippet_indices = window_indices + index
                    raw_snippet = intan_constant*stim_file[channel, snippet_indices]
                    filtered_snippet = np.zeros(len(raw_snippet))
                    filtered_snippet[:] = raw_snippet[:]
                    filtered_snippet = bp_filter(filtered_snippet)
                    filtered_snippet[window_time_axis < stim_blank_time] = 0.0
                    snippets[j, :] = filtered_snippet
                stimulus_average = np.mean(snippets, axis=0)
                shank_stimulus_average[i, :] = stimulus_average
                threshold_crossings = utils.get_threshold_crossings(stimulus_average, threshold)
                shank_threshold_crossings.extend(zip(window_time_axis[threshold_crossings], [channel]*len(threshold_crossings)))

            # all crossing plots
            print 'Saving stimulus-aligned waveforms and %d threshold crossings' % len(shank_threshold_crossings)
            # sort list by crossing times
            shank_threshold_crossings_sorted = shank_threshold_crossings[:]
            shank_threshold_crossings_sorted.sort()

            shank_wf_name = 'stim_%d_average_wf_shank%d.npy' % (level, shank)
            np.save(os.path.join(savepath, shank_wf_name), shank_stimulus_average)
            shank_crossing_name = 'stim_%d_crossings_shank%d.npy' % (level, shank)
            np.save(os.path.join(savepath, shank_crossing_name), shank_threshold_crossings_sorted)


def view_stimulus_aligned_recording_from_matlab():
    stimulus_alignment_info = scipy.io.loadmat(stimulus_alignment_info_name, struct_as_record=False, squeeze_me=True)
    channel_locations = np.load(os.path.join(phy_info_name, 'channel_positions.npy'))
    channel_locations_shank1 = channel_locations[:32, :]
    channel_locations_shank2 = channel_locations[32:, :]
    # sorted_channels_shank1 = np.argsort(channel_locations_shank1[:, 1])
    # sorted_channels_shank2 = np.argsort(channel_locations_shank2[:, 1]) + 32
    sorted_channels_shank1 = _sort_channels_by_distance(channel_locations_shank1)
    sorted_channels_shank2 = _sort_channels_by_distance(channel_locations_shank2) + 32

    # order: channels, samples
    data_folder = 'FIXTHIS'
    raw_traces = utils.load_recording(os.path.join(data_folder, 'amplifier.dat'), nchannels=64)

    # stim_period = stimulus_alignment_info['stimPeriod']
    stimulus_indices = stimulus_alignment_info['trigs']
    stimulus_amplitude = stimulus_alignment_info['stim_lvl']

    # _aligned_traces_per_channel(raw_traces, stimulus_indices, stimulus_amplitude)
    _aligned_traces_map(raw_traces, stimulus_indices, stimulus_amplitude, sorted_channels_shank1)


def create_stimulus_aligned_recording_file(raw_traces, stimulus_indices, stimulus_amplitude):
    window_indices = np.array([i for i in range(30, 30 + 600)])  # 1 ms post-stim - 21 ms post-stim
    # time_axis = window_indices / 20.0
    # stim_blank_time = 2.5

    b, a = _set_up_filter(300.0, 0.49 * fs, fs)

    def bp_filter(x):
        return signal.filtfilt(b, a, x)

    stimulus_levels = [50, 100, 200]
    # channels = [1]
    # channels = range(96, 128)
    channels = range(128)
    for level in stimulus_levels:
        tmp_indices = stimulus_indices[stimulus_amplitude == level]
        print 'Analyzing %d stimuli at level %.0f' % (len(tmp_indices), level)
        snippets = np.zeros((len(channels), len(tmp_indices), len(window_indices)))
        for i, index in enumerate(tmp_indices):
            snippet_indices = window_indices + index
            raw_snippet = raw_traces[:, snippet_indices]
            filtered_snippet = bp_filter(raw_snippet)
            # filtered_snippet[time_axis < 0.0] = raw_snippet[time_axis < 0.0]
            # filtered_snippet[(time_axis >= 0.0)*(time_axis < stim_blank_time)] = 0.0
            snippets[:, i, :] = filtered_snippet
        stimulus_average_ = np.mean(snippets, axis=1)
        stimulus_average = np.int16(stimulus_average_)
        stimulus_average_rep = np.tile(stimulus_average, 100)
        out_dat_file = 'stim_average_%d_muA.dat' % level
        out_folder = 'FIXTHIS'
        out_name = os.path.join(out_folder, out_dat_file)
        new_file = np.memmap(out_name,
                             'int16',
                             mode='w+',
                             shape=(stimulus_average_rep.shape[0], stimulus_average_rep.shape[1]),
                             order='F')
        new_file[:, :] = stimulus_average_rep[:, :]


def select_antidromic_units_waveforms(raw_traces, stimulus_indices, stimulus_amplitude, channel_locations, channel_shank_map, fs):
    intan_constant = 0.195
    window_indices = np.array([i for i in range(-30, 600)])  # 20 ms post-stim
    pre_stim_offset = 30
    window_time_axis = window_indices / 30.0
    waveform_window_indices = np.array([int(-0.5*1e-3*fs) + i for i in range(int(1.5*1e-3*fs))])
    waveform_time_axis = waveform_window_indices/fs*1e3
    threshold = -100.0 # muV
    savepath = r'F:\sorting\C22\C22_d1_song_stim_period\waveform_matching\manual_waveform_selection'
    stim_blank_time = 2.5 # ms

    b, a = _set_up_filter(300.0, 0.49 * fs, fs)

    def bp_filter(x):
        return signal.filtfilt(b, a, x)

    # stimulus_levels = [50, 100, 200]
    stimulus_levels = [50]
    shanks = range(2, 5)
    # shanks = [1]
    figure_count = 0
    for shank in shanks:
        channels = np.where(channel_shank_map == shank)[0]
        shank_threshold_crossings = []
        shank_stimulus_average = np.zeros((len(channels), len(window_indices)))
        for level in stimulus_levels:
            tmp_indices = stimulus_indices[stimulus_amplitude == level]
            print 'Analyzing %d stimuli at level %.0f on shank %d' % (len(tmp_indices), level, shank)
            for i, channel in enumerate(channels):
                snippets = np.zeros((len(tmp_indices), len(window_indices)))
                for j, index in enumerate(tmp_indices):
                    snippet_indices = window_indices + index
                    raw_snippet = intan_constant*raw_traces[channel, snippet_indices]
                    filtered_snippet = np.zeros(len(raw_snippet))
                    filtered_snippet[:] = raw_snippet[:]
                    filtered_snippet = bp_filter(filtered_snippet)
                    filtered_snippet[window_time_axis < stim_blank_time] = 0.0
                    snippets[j, :] = filtered_snippet
                stimulus_average = np.mean(snippets, axis=0)
                shank_stimulus_average[i, :] = stimulus_average
                threshold_crossings = utils.get_threshold_crossings(stimulus_average, threshold)
                # if channel == 24:
                #     dummy = 1
                shank_threshold_crossings.extend(zip(window_time_axis[threshold_crossings], [channel]*len(threshold_crossings)))

        # all crossing plots
        print 'Creating summary plots for %d threshold crossings' % len(shank_threshold_crossings)
        # sort list by crossing times
        shank_threshold_crossings_sorted = shank_threshold_crossings[:]
        shank_threshold_crossings_sorted.sort()

        shank_wf_name = 'stim_%d_average_wf_shank%d.npy' % (level, shank)
        np.save(os.path.join(savepath, shank_wf_name), shank_stimulus_average)
        shank_crossing_name = 'stim_%d_crossings_shank%d.npy' % (level, shank)
        np.save(os.path.join(savepath, shank_crossing_name), shank_threshold_crossings_sorted)
        # np.save(os.path.join(savepath, shank_crossing_name), shank_threshold_crossings)

        for i, crossing in enumerate(shank_threshold_crossings_sorted):
            t_crossing, channel_crossing = crossing
            plt.figure(figure_count)
            for j, channel in enumerate(channels):
                channel_loc = channel_locations[channel]
                plt.plot(window_time_axis + channel_loc[0],
                         shank_stimulus_average[j, :] + 15.0*channel_loc[1],
                         'k', linewidth=0.5)
                if channel == channel_crossing:
                    highlight_color = 'r'
                    highlight_thickness = 1.0
                else:
                    highlight_color = 'b'
                    highlight_thickness = 0.5
                crossing_sample = int(t_crossing*1e-3*fs)
                highlight_window = crossing_sample + pre_stim_offset + waveform_window_indices
                plt.plot(waveform_time_axis + t_crossing + channel_loc[0],
                         shank_stimulus_average[j, highlight_window] + 15.0*channel_loc[1],
                         highlight_color, linewidth=highlight_thickness)
                # if channel == 24:
                #     dummy = 1
            plt.xlabel('Time (ms)')
            plt.ylabel('$\mu$V')
            plt.title('Channels %d-%d; level %.0f $\mu$A, crossing %d ' % (np.min(channels), np.max(channels), level, i))
            pdf_name = 'Stimulus_%.0f_aligned_traces_all_channels_shank%d_crossing%d.pdf' % (level, shank, i)
            out_name = os.path.join(savepath, pdf_name)
            plt.savefig(out_name)
            figure_count += 1
            # plt.show()


def view_stimulus_aligned_recording(crossing_info_name):
    # # C22 d1 evening
    # stim_data_folder = r'F:\sorting\C22\C22_d1_song_stim_period\d1_evening_stim'
    # song_data_folder = r'F:\sorting\C22\C22_d1_song_stim_period\d1_evening_song'
    # channel_locations = np.load(os.path.join(song_data_folder, 'channel_positions.npy'))
    # channel_shank_map = np.load(os.path.join(song_data_folder, 'channel_shank_map.npy'))
    # t_start = 199.0
    # t_stop = 360.0
    # fs = 3e4
    # recording = utilities.load_recording(os.path.join(stim_data_folder, 'stim_cropped.dat'), nchannels=128)
    # stimulus_times = _detect_stimulus_times(recording, t_start, t_stop, fs)
    # # 1-45: 50 muA
    # # 46-93: 100 muA
    # # 94-148: 200 muA
    # stimulus_amplitude = np.zeros(148)
    # stimulus_amplitude[:45] = 50
    # stimulus_amplitude[45:92] = 100
    # stimulus_amplitude[92:147] = 200
    # stimulus_levels = [50, 100, 200]

    # C22 d2 afternoon
    stim_data_folder = r'Z:\Margot\Experiments\SiProbe\ChronicRecordings\C22\C22_190531_093100\cut_daytime'
    song_data_folder = r'Z:\Margot\Experiments\SiProbe\ChronicRecordings\C22\C22_190531_093100\cut_daytime'
    channel_locations = np.load(os.path.join(song_data_folder, 'channel_positions.npy'))
    channel_shank_map = np.load(os.path.join(song_data_folder, 'channel_shank_map.npy'))
    t_start = 259*60.0
    t_stop = 263.8*60.0
    fs = 3e4
    recording = utils.load_recording(os.path.join(stim_data_folder, 'cut_amplifier.dat'), nchannels=128)
    stimulus_indices = _detect_stimulus_times(recording, t_start, t_stop, fs)
    stimulus_times = stimulus_indices * 1.0 / fs
    # find stimulus blocks
    # Note: in this sweep, 50 muA is broken up into two blocks
    stimulus_levels = [20, 50, 100, 200]
    stimulus_levels_ = [20, 50, 50, 100, 200]
    dt = np.diff(stimulus_times)
    block_end_indices = np.where(dt > 1.1)[0]
    stimulus_amplitude = np.zeros(len(stimulus_times))
    stimulus_amplitude[:block_end_indices[0] + 1] = stimulus_levels_[0]
    for i in range(len(block_end_indices)):
        if i < len(block_end_indices) - 1:
            start_index = block_end_indices[i]
            stop_index = block_end_indices[i + 1]
            stimulus_amplitude[start_index:stop_index] = stimulus_levels_[i + 1]
        else:
            start_index = block_end_indices[i]
            stimulus_amplitude[start_index:] = stimulus_levels_[i + 1]

    # create_stimulus_aligned_recording_file(recording, stimulus_times, stimulus_amplitude)
    _aligned_traces_per_channel(recording, stimulus_indices, stimulus_amplitude, stimulus_levels, channel_locations,
                                channel_shank_map)
    # antidromic_units_waveforms(recording, stimulus_times, stimulus_amplitude, channel_locations, channel_shank_map)
    # select_antidromic_units_waveforms(recording, stimulus_times, stimulus_amplitude, channel_locations, channel_shank_map, fs)


def stimulus_aligned_recording(crossing_info_name):
    with open(crossing_info_name, 'r') as data_file:
        crossing_info = ast.literal_eval(data_file.read())

    # detect stimulus times
    stim_data_folder = crossing_info['Antidromic']['DataBasePath']
    t_start = crossing_info['Antidromic']['StimStart']
    t_stop = crossing_info['Antidromic']['StimStop']
    fs = crossing_info['Antidromic']['SamplingRate']
    stim_file = utils.load_recording(os.path.join(stim_data_folder, crossing_info['Antidromic']['AmplifierName']),
                                         nchannels=crossing_info['Antidromic']['Channels'])
    stimulus_indices = _detect_stimulus_times(stim_file, t_start, t_stop, fs)
    stimulus_times = stimulus_indices * 1.0 / fs
    # find stimulus blocks
    stimulus_levels_ = crossing_info['Antidromic']['StimLevels']
    dt = np.diff(stimulus_times)
    block_end_indices = np.where(dt > 1.1)[0] # assume 1 Hz stim frequency!!!
    stimulus_amplitude = np.zeros(len(stimulus_times))
    stimulus_amplitude[:block_end_indices[0] + 1] = stimulus_levels_[0]
    for i in range(len(block_end_indices)):
        if i < len(block_end_indices) - 1:
            start_index = block_end_indices[i]
            stop_index = block_end_indices[i + 1]
            stimulus_amplitude[start_index:stop_index] = stimulus_levels_[i + 1]
        else:
            start_index = block_end_indices[i]
            stimulus_amplitude[start_index:] = stimulus_levels_[i + 1]

    # save stimulus indices and amplitudes
    stim_indices_name = os.path.join(crossing_info['Antidromic']['CrossingBasePath'], 'stimulus_indices.npy')
    stim_amplitude_name = os.path.join(crossing_info['Antidromic']['CrossingBasePath'], 'stimulus_amplitudes.npy')
    np.save(stim_indices_name, stimulus_indices)
    np.save(stim_amplitude_name, stimulus_amplitude)
    # first generate stimulus-aligned averages and detect crossings
    stimulus_levels = np.unique(stimulus_levels_)
    _save_stimulus_aligned_waveforms_crossings(crossing_info, stimulus_indices, stimulus_amplitude, stimulus_levels)


def threshold_crossing_waveform_visualization(crossing_info_name):
    with open(crossing_info_name, 'r') as data_file:
        crossing_info = ast.literal_eval(data_file.read())

    antidromic_crossings_ids = np.loadtxt(os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                                   (crossing_info['Antidromic']['PutativeAntidromicName'])),
                                      skiprows=1,
                                      unpack=True)
    antidromic_waveforms = {}
    antidromic_crossings = {}
    for shank in crossing_info['Antidromic']['ShankWaveforms']:
        tmp_waveform_name = os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                         crossing_info['Antidromic']['ShankWaveforms'][shank])
        antidromic_waveforms[shank] = np.load(tmp_waveform_name)
        tmp_crossing_name = os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                         crossing_info['Antidromic']['ShankCrossings'][shank])
        antidromic_crossings[shank] = np.load(tmp_crossing_name)

    # detect stimulus times
    stim_data_folder = crossing_info['Antidromic']['DataBasePath']
    t_start = 199.0
    t_stop = 360.0
    fs = crossing_info['Antidromic']['SamplingRate']
    stim_file = utils.load_recording(os.path.join(stim_data_folder, crossing_info['Antidromic']['AmplifierName']),
                                         nchannels=crossing_info['Antidromic']['Channels'])
    stimulus_times = _detect_stimulus_times(stim_file, t_start, t_stop, fs)
    # 1-45: 50 muA
    # 46-93: 100 muA
    # 94-148: 200 muA
    stimulus_amplitude = np.zeros(148)
    stimulus_amplitude[:45] = 50
    stimulus_amplitude[45:92] = 100
    stimulus_amplitude[92:147] = 200
    stimulus_times_50 = stimulus_times[np.where(stimulus_amplitude == 50)]
    # iterate over crossings
    for i in range(antidromic_crossings_ids.shape[1]):
        crossing_shank, crossing_id = antidromic_crossings_ids[0, i], antidromic_crossings_ids[1, i]
        t_crossing, channel_crossing = antidromic_crossings[crossing_shank][crossing_id] # in ms!
        stim_waveforms = _get_crossing_waveforms_stim_aligned(stim_file, stimulus_times_50,
                                                              t_crossing, channel_crossing, fs)
        min_window = range(int(0.5*1e-3*fs), int(1.0*1e-3*fs))
        spike_indices = np.argmin(stim_waveforms[:, min_window], axis=1)
        spike_times = spike_indices*1.0/fs
        antidromic_variability = np.std(spike_times)
        print 'Var = %.5f' % antidromic_variability
        plt.figure(i)
        for j in range(stim_waveforms.shape[0]):
            plt.plot(stim_waveforms[j, :], 'k', linewidth=0.5)
        plt.plot(np.mean(stim_waveforms, axis=0), 'r', linewidth=1)
        title_str = 'Shank %d, crossing %d, var = %.2f ms' % (crossing_shank, crossing_id, antidromic_variability*1e3)
        plt.title(title_str)
        pdf_name = 'shank_%d_crossing_%d.pdf' % (crossing_shank, crossing_id)
        plt.savefig(os.path.join(crossing_info['Antidromic']['CrossingBasePath'], pdf_name))
        # plt.show()
        # for each crossing, determine waveform, amplitude and spike time after each stimulus
    song_data_folder = crossing_info['ContinuousRecording']['DataBasePath']
    channel_locations = np.load(os.path.join(song_data_folder, 'channel_positions.npy'))
    channel_shank_map = np.load(os.path.join(song_data_folder, 'channel_shank_map.npy'))
    # next: compare to waveforms of clusters on same channel (during motifs)


def crossing_shank_waveform_variability_visualization(crossing_info_name):
    with open(crossing_info_name, 'r') as data_file:
        crossing_info = ast.literal_eval(data_file.read())

    # detect stimulus times
    stim_data_folder = crossing_info['Antidromic']['DataBasePath']
    t_start = crossing_info['Antidromic']['StimStart']
    t_stop = crossing_info['Antidromic']['StimStop']
    fs = crossing_info['Antidromic']['SamplingRate']
    stim_file = utils.load_recording(os.path.join(stim_data_folder, crossing_info['Antidromic']['AmplifierName']),
                                         nchannels=crossing_info['Antidromic']['Channels'])
    stimulus_indices = _detect_stimulus_times(stim_file, t_start, t_stop, fs)
    stimulus_times = stimulus_indices * 1.0 / fs
    # find stimulus blocks
    stimulus_levels_ = crossing_info['Antidromic']['StimLevels']
    dt = np.diff(stimulus_times)
    block_end_indices = np.where(dt > 1.1)[0] # assume 1 Hz stim frequency!!!
    stimulus_amplitude = np.zeros(len(stimulus_times))
    stimulus_amplitude[:block_end_indices[0] + 1] = stimulus_levels_[0]
    for i in range(len(block_end_indices)):
        if i < len(block_end_indices) - 1:
            start_index = block_end_indices[i]
            stop_index = block_end_indices[i + 1]
            stimulus_amplitude[start_index:stop_index] = stimulus_levels_[i + 1]
        else:
            start_index = block_end_indices[i]
            stimulus_amplitude[start_index:] = stimulus_levels_[i + 1]

    # first generate stimulus-aligned averages and detect crossings
    stimulus_levels = np.unique(stimulus_levels_)
    _save_stimulus_aligned_waveforms_crossings(crossing_info, stimulus_indices, stimulus_amplitude, stimulus_levels)
    # _save_stimulus_aligned_waveforms_crossings(crossing_info_name)

    # if 'PutativeAntidromicName' in crossing_info['Antidromic']:
    #     antidromic_crossings_ids = np.loadtxt(os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
    #                                                        (crossing_info['Antidromic']['PutativeAntidromicName'])),
    #                                           skiprows=1,
    #                                           unpack=True)
    # else:
    #     antidromic_crossings_ids = None
    antidromic_waveforms = {}
    antidromic_crossings = {}
    for stim in crossing_info['Antidromic']['ShankWaveforms']:
        antidromic_waveforms[stim] = {}
        antidromic_crossings[stim] = {}
        for shank in crossing_info['Antidromic']['ShankWaveforms'][stim]:
            tmp_waveform_name = os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                             crossing_info['Antidromic']['ShankWaveforms'][stim][shank])
            antidromic_waveforms[stim][shank] = np.load(tmp_waveform_name)
            tmp_crossing_name = os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                             crossing_info['Antidromic']['ShankCrossings'][stim][shank])
            antidromic_crossings[stim][shank] = np.load(tmp_crossing_name)

    song_data_folder = crossing_info['ContinuousRecording']['DataBasePath']
    channel_locations = np.load(os.path.join(song_data_folder, 'channel_positions.npy'))
    channel_shank_map = np.load(os.path.join(song_data_folder, 'channel_shank_map.npy'))

    window_indices = np.array([i for i in range(-30, 600)])  # 20 ms post-stim
    pre_stim_offset = 30
    window_time_axis = window_indices / 30.0
    waveform_window_indices = np.array([int(-0.5*1e-3*fs) + i for i in range(int(1.5*1e-3*fs))])
    waveform_time_axis = waveform_window_indices/fs*1e3

    # iterate over crossings
    # for i in range(antidromic_crossings_ids.shape[1]):
    figure_count = 0
    for stim_amplitude in antidromic_crossings:
        for crossing_shank in antidromic_crossings[stim_amplitude]:
            for crossing_id in range(len(antidromic_crossings[stim_amplitude][crossing_shank])):
            # crossing_shank, crossing_id = antidromic_crossings_ids[0, i], antidromic_crossings_ids[1, i]
                t_crossing, channel_crossing = antidromic_crossings[stim_amplitude][crossing_shank][crossing_id] # in ms
                stimulus_indices_ = stimulus_indices[np.where(stimulus_amplitude == stim_amplitude)]
                stim_waveforms = _get_crossing_waveforms_stim_aligned(stim_file, stimulus_indices_,
                                                                      t_crossing, channel_crossing, fs)
                min_window = range(int(0.5 * 1e-3 * fs), int(1.0 * 1e-3 * fs))
                spike_indices = np.argmin(stim_waveforms[:, min_window], axis=1)
                spike_times = spike_indices * 1.0 / fs
                antidromic_variability = np.std(spike_times)
                print 'Var = %.5f' % antidromic_variability

                fig = plt.figure(figure_count)
                figure_count += 1
                ax1 = plt.subplot(1, 2, 1)
                # plot average stim-aligned waveform on shank
                channels = np.where(channel_shank_map == crossing_shank)[0]
                shank_stimulus_average = antidromic_waveforms[stim_amplitude][crossing_shank]
                for j, channel in enumerate(channels):
                    channel_loc = channel_locations[channel]
                    ax1.plot(window_time_axis + channel_loc[0],
                             shank_stimulus_average[j, :] + 15.0*channel_loc[1],
                             'k', linewidth=0.5)
                    if channel == channel_crossing:
                        highlight_color = 'r'
                        highlight_thickness = 1.0
                    else:
                        highlight_color = 'b'
                        highlight_thickness = 0.5
                    crossing_sample = int(t_crossing*1e-3*fs)
                    highlight_window = crossing_sample + pre_stim_offset + waveform_window_indices
                    ax1.plot(waveform_time_axis + t_crossing + channel_loc[0],
                             shank_stimulus_average[j, highlight_window] + 15.0*channel_loc[1],
                             highlight_color, linewidth=highlight_thickness)
                ax1.set_xlabel('Time (ms)')
                ax1.set_ylabel('$\mu$V')
                ax1.set_title('Shank %d; amp %.1f; crossing %d' % (crossing_shank, stim_amplitude, crossing_id))
                ax2 = plt.subplot(1, 2, 2)

                # plot variability of stim-aligned waveform on max channel
                for j in range(stim_waveforms.shape[0]):
                    ax2.plot(np.array(range(len(stim_waveforms[j, :]))) * 1.0 / fs, stim_waveforms[j, :], 'k', linewidth=0.5)
                mean_wf = np.mean(stim_waveforms, axis=0)
                ax2.plot(np.array(range(len(mean_wf))) * 1.0 / fs, mean_wf, 'r', linewidth=1)
                title_str = 'SD of min. = %.2f ms' % (antidromic_variability * 1e3, )
                ax2.set_title(title_str)
                pdf_name = 'shank_%d_amp_%.1f_crossing_%d.pdf' % (crossing_shank, stim_amplitude, crossing_id)
                plt.savefig(os.path.join(crossing_info['Antidromic']['CrossingBasePath'], pdf_name))
                plt.close(fig)


def manual_crossing_selection(crossing_info_name):
    with open(crossing_info_name, 'r') as data_file:
        crossing_info = ast.literal_eval(data_file.read())

    # select which stim level / shank waveforms to load
    stim_levels = crossing_info['Antidromic']['ShankWaveforms'].keys()
    stim_levels.sort()
    shanks = crossing_info['Antidromic']['ShankWaveforms'][stim_levels[0]].keys()
    shanks.sort()
    print 'Stim levels:'
    print stim_levels
    stim_level = None
    while stim_level is None:
        try:
            stim_level = int(raw_input('Enter which stim level to analyze: '))
            if stim_level not in stim_levels:
                print 'Stim level %d not available' % stim_level
                stim_level = None
        except ValueError:
            print 'Please use an integer number (microAmps)'
    print 'Shanks:'
    print shanks
    shank = None
    while shank is None:
        try:
            shank = int(raw_input('Enter which shank to analyze: '))
            if shank not in shanks:
                print 'Shank %d not available' % shank
                shank = None
        except ValueError:
            print 'Please use an integer number'

    stim_aligned_waveform = np.load(os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                                 crossing_info['Antidromic']['ShankWaveforms'][stim_level][shank]))
    ap = utils.AntidromicPicker(stim_aligned_waveform, crossing_info, stim_level, shank)
    ap.pick_antidromic_units()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        view_stimulus_aligned_recording()
    if len(sys.argv) == 2:
        crossing_info_name = sys.argv[1]
        # stimulus_aligned_recording(crossing_info_name)
        manual_crossing_selection(crossing_info_name)
        # threshold_crossing_waveform_visualization(crossing_info_name)
        # crossing_shank_waveform_variability_visualization(crossing_info_name)






# def antidromic_units_waveforms(raw_traces, stimulus_indices, stimulus_amplitude, channel_locations, channel_shank_map):
#     # for each shank:
#     # for each channel:
#     # detect (negative) threshold crossings
#     # plot waveform across all channels at this time
#     # (next step: merging - similarity in time and shape?)
#     intan_constant = 0.195
#     # window_indices = np.array([i for i in range(-20, 301)]) # 1 ms pre-stim, 15 ms post-stim
#     window_indices = np.array([i for i in range(301)]) # 15 ms post-stim
#     time_axis = window_indices/20.0
#     stim_blank_time = 2.5
#     threshold = -100.0
#     waveform_window_indices = np.array([int(-0.5*1e-3*fs) + i for i in range(int(1.5*1e-3*fs))])
#     waveform_time_axis = waveform_window_indices/fs*1e3
#
#     n_nearest_channels = 8
#     similarity_threshold = 20.0
#     savepath = r'F:\sorting\C22\C22_d1_song_stim_period\waveform_matching'
#
#     b, a = _set_up_filter(300.0, 0.49*fs, fs)
#     def bp_filter(x):
#         return signal.filtfilt(b, a, x)
#
#     all_threshold_crossings = []
#     stimulus_levels = [50]
#     # shanks = range(1, 5)
#     shanks = [4]
#     for shank in shanks:
#         channels = np.where(channel_shank_map == shank)[0]
#         shank_threshold_crossings = []
#         shank_stimulus_average = np.zeros((len(channels), len(window_indices)))
#         for level in stimulus_levels:
#             # plt.figure(level + shank)
#             tmp_indices = stimulus_indices[stimulus_amplitude == level]
#             print 'Analyzing %d stimuli at level %.0f on shank %d' % (len(tmp_indices), level, shank)
#             for i, channel in enumerate(channels):
#                 snippets = np.zeros((len(tmp_indices), len(window_indices)))
#                 for j, index in enumerate(tmp_indices):
#                     snippet_indices = window_indices + index
#                     raw_snippet = intan_constant*raw_traces[channel, snippet_indices]
#                     filtered_snippet = np.zeros(len(raw_snippet))
#                     # filtered_snippet[time_axis >= stim_blank_time] = raw_snippet[time_axis >= stim_blank_time]
#                     filtered_snippet[:] = raw_snippet[:]
#                     filtered_snippet = bp_filter(filtered_snippet)
#                     filtered_snippet[time_axis < stim_blank_time] = 0.0
#                     # filtered_snippet[time_axis < 0.0] = raw_snippet[time_axis < 0.0]
#                     # filtered_snippet[(time_axis >= 0.0)*(time_axis < stim_blank_time)] = 0.0
#                     snippets[j, :] = filtered_snippet
#                 stimulus_average = np.mean(snippets, axis=0)
#                 shank_stimulus_average[i, :] = stimulus_average
#                 threshold_crossings = _get_threshold_crossings(stimulus_average, threshold)
#                 all_threshold_crossings.extend(zip(time_axis[threshold_crossings], [channel]*len(threshold_crossings)))
#                 shank_threshold_crossings.extend(zip(time_axis[threshold_crossings], [channel]*len(threshold_crossings)))
#                 # plt.plot(time_axis, stimulus_average, linewidth=0.5)
#                 # plt.plot(time_axis[threshold_crossings], stimulus_average[threshold_crossings], 'ro')
#             # plt.show()
#
#         # extract waveforms for each threshold crossing time
#         crossing_similarity_matrix = np.zeros((len(shank_threshold_crossings), len(shank_threshold_crossings))) + \
#                                      similarity_threshold*np.identity(len(shank_threshold_crossings))
#         for i, crossing in enumerate(shank_threshold_crossings):
#
#             # compute crossing time difference to all other crossings
#             # compute similarity (i.e., inner product) of waveform at crossing time
#             # (use waveform across n_nearest_channels)
#             t_crossing, channel_crossing = crossing
#             neighboring_channels = _get_nearest_channels(channels, channel_locations, channel_crossing, n_nearest_channels)
#             neighboring_channels_shank = neighboring_channels%32 # we need them to be indices into the average stimulus-aligned waveforms
#             sample_crossing = int(t_crossing*1e-3*fs)
#             for j in range(i + 1, len(shank_threshold_crossings)):
#                 crossing2 = shank_threshold_crossings[j]
#                 t_crossing2, channel_crossing2 = crossing2
#                 sample_crossing2 = int(t_crossing2*1e-3*fs)
#                 time_diff = abs(t_crossing - t_crossing2)/(np.max(waveform_time_axis) - np.min(waveform_time_axis)) # some reasonable normalization
#                 neighboring_channels2 = _get_nearest_channels(channels, channel_locations, channel_crossing2, n_nearest_channels)
#                 neighboring_channels2_shank = neighboring_channels2%32 # we need them to be indices into the average stimulus-aligned waveforms
#                 waveform_similarity, n_common_channels = _waveform_similarity(shank_stimulus_average,
#                                                                               neighboring_channels_shank,
#                                                                               neighboring_channels2_shank,
#                                                                               waveform_window_indices,
#                                                                               sample_crossing,
#                                                                               sample_crossing2)
#                 similarity_score = waveform_similarity*n_common_channels/(time_diff + 1e-3/fs) # regularization by 1 sample
#                 # if i == 34 and j == 61:
#                 #     dummy = 1
#                 # TODO: the problem is that there is no unambiguous assignment of neighboring channels around the two
#                 # crossing channels... crossing channel alone is missing the point...
#                 # maybe: determine which ones of the neighboring channels are identical (allow larger number),
#                 # discard other channels (penalize lack of overlap) and compute inner product only on identical channels
#                 crossing_similarity_matrix[i, j] = similarity_score
#
#             # if i not in (23, 58, 60, 61):
#             #     continue
#             # plt.figure(i)
#             # snippet_indices = sample_crossing + waveform_window_indices
#             # for j, channel in enumerate(channels):
#             #     tmp_indices = snippet_indices[np.where((snippet_indices < len(window_indices)*(snippet_indices >= 0)))]
#             #     raw_snippet = shank_stimulus_average[j, tmp_indices]
#             #     channel_loc = channel_locations[channel]
#             #     snippet_color = 'k'
#             #     if channel == channel_crossing:
#             #         snippet_color = 'r'
#             #     plt.plot(5*waveform_time_axis[0:len(tmp_indices)] + channel_loc[0], raw_snippet + 15.0*channel_loc[1],
#             #              snippet_color, linewidth=0.5)
#             #     plt.title('Crossing %d' % i)
#             #     pdf_name = 'threshold_crossing_NN_%d_%d.pdf' % (n_nearest_channels, i)
#             #     plt.savefig(os.path.join(savepath, pdf_name))
#
#         # merge crossings which have similarity scores > similarity_threshold
#         merge_matrix = crossing_similarity_matrix > similarity_threshold
#         crossing_clusters = dict(zip(range(merge_matrix.shape[0]), [[i] for i in range(merge_matrix.shape[0])]))
#         merged_crossings = []
#         for i in range(crossing_similarity_matrix.shape[0]):
#             if i not in merged_crossings:
#                 merged_crossings.append(i)
#             for j in range(i + 1, crossing_similarity_matrix.shape[1]):
#                 if merge_matrix[i, j] and j not in merged_crossings:
#                     try:
#                         crossing_clusters[i].append(j)
#                     except KeyError:
#                         crossing_clusters[i] = [j]
#                     merged_crossings.append(j)
#         for crossing in crossing_clusters.keys():
#             try:
#                 merged_ids = crossing_clusters[crossing]
#                 if len(merged_ids) == 1:
#                     continue
#                 for id in merged_ids:
#                     if id in crossing_clusters.keys():
#                         merged_ids.extend(crossing_clusters[id])
#                         crossing_clusters.pop(id)
#                 crossing_clusters[crossing] = merged_ids
#             except KeyError:
#                 continue
#         tmp = crossing_similarity_matrix[crossing_similarity_matrix < 100.0]
#         # plt.figure(0)
#         # plt.hist(tmp.flatten(), bins=50)
#         # plt.title('nearest channels = %d' % n_nearest_channels)
#         # plt.show()
#         csv_name = 'NN_%d.csv' % n_nearest_channels
#         with open(os.path.join(savepath, csv_name), 'w') as similarity_score_file:
#             for i in range(crossing_similarity_matrix.shape[0]):
#                 for j in range(crossing_similarity_matrix.shape[1]):
#                     similarity_score_file.write(str(crossing_similarity_matrix[i, j]))
#                     similarity_score_file.write('\t')
#                 similarity_score_file.write('\n')
#         # 0/0
#     # plt.figure(0)
#     # bins = np.arange(0.0, 15.0, 0.04999)
#     # plt.hist(all_threshold_crossings, bins)
#     # plt.show()