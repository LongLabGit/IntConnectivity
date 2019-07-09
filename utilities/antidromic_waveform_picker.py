import os, os.path, time
import numpy as np
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import utilities as utils


class AntidromicPicker(object):
    def __init__(self, stim_aligned_waveform, crossing_info, stim_level, shank):

        # sta waveform shape: n_channels, n_samples
        self.sta_waveform = stim_aligned_waveform
        self.crossing_info = crossing_info
        self.stim_level = stim_level
        self.shank = shank

        # electrode info
        channel_locations = np.load(os.path.join(crossing_info['ContinuousRecording']['ClusterBasePath'],
                                         'channel_positions.npy'))
        channel_shank_map = np.load(os.path.join(crossing_info['ContinuousRecording']['ClusterBasePath'],
                                    'channel_shank_map.npy'))
        self.channel_locations = channel_locations[np.where(channel_shank_map == shank)]
        self.channel_ids = np.where(channel_shank_map == shank)[0]

        # recording info
        self.fs = crossing_info['Antidromic']['SamplingRate']

        # antidromic info + data
        stimulus_indices = np.load(os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                                     'stimulus_indices.npy'))
        stimulus_amplitudes = np.load(os.path.join(crossing_info['Antidromic']['CrossingBasePath'],
                                                        'stimulus_amplitudes.npy'))
        self.stimulus_indices = stimulus_indices[np.where(stimulus_amplitudes == self.stim_level)]
        self.antidromic_file = utils.load_recording(os.path.join(crossing_info['Antidromic']['DataBasePath'],
                                                                 crossing_info['Antidromic']['AmplifierName']),
                                                    crossing_info['Antidromic']['Channels'])

        # state variable
        self._modify_antidromic = False
        self._modify_var = False
        self.antidromic_units = []
        self._current_var = None
        self._current_var_channel = None
        self._current_antidromic_highlights = []
        self._saved_antidromic_highlights = []
        self._threshold = []
        self._var_time_axis = None
        self._var_waveforms = None
        self._var_time_index = None
        self._good_var_waveforms = None

        pid = os.getpid()
        suffix = 'shank_%d_stim_%d_' % (self.shank, self.stim_level)
        suffix += time.strftime('%Y-%m-%d_%H-%M-%S')
        suffix += '_'
        suffix += str(pid)
        self.save_path = os.path.join(crossing_info['Antidromic']['CrossingBasePath'], suffix)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def connect(self):
        """
        connect to all the events we need
        keyboard press 'a' to define new crossing
        click on channel at specific time to add
        keyboard press 't' to add 2 threshold window values
        keyboard press 'w' to save
        keyboard press 'q' to quit
        """
        self.cidpress = self.ax_wf_var.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidpick = self.ax_sta.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidkey = self.ax_sta.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        self.ax_wf_var.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax_sta.figure.canvas.mpl_disconnect(self.cidpick)
        self.ax_sta.figure.canvas.mpl_disconnect(self.cidkey)

    def on_press(self, event):
        if not self._modify_var:
            return
        self._add_threshold(event.ydata)
        if len(self._threshold) == 2:
            self._threshold_electrode()
            self._threshold = []
            self._modify_var = False

    def on_pick(self, event):
        if not self._modify_antidromic:
            return
        self._add_antidromic_waveform(event)
        self._modify_antidromic = False

    def on_key(self, event):
        if event.key == 'a' or event.key == 'A':
            self._modify_antidromic = True
            print 'Click to add antidromic waveform'
        # if event.key == 'u' or event.key == 'U':
        #     self._modify_antidromic = True
        #     print 'Removing last antidromic waveform'
        #     self._remove_last_antidromic_waveform()
        #     self._modify_antidromic = False
        if event.key == 't' or event.key == 'T':
            self._modify_var = True
            print 'Choose two threshold values...'
        if event.key == 'w' or event.key == 'W':
            print 'Saving antidromic waveforms'
            self._save_antidromic_waveforms()
        if event.key == 'q' or event.key == 'Q':
            print 'Finished adding antidromic waveforms, quitting...'
            self.disconnect()

    def pick_antidromic_units(self):
        self._create_base_figure()
        self.connect()
        plt.show()

    def _create_base_figure(self):
        amp_scaling = 8.0
        window_indices = np.array([i for i in range(-30, 600)])  # 20 ms post-stim
        window_time_axis = window_indices / 30.0
        self.fig = plt.figure(1)
        self.ax_sta = plt.subplot(1, 2, 1)
        for i in range(self.sta_waveform.shape[0]):
            self.ax_sta.plot(window_time_axis + self.channel_locations[i][0],
                             self.sta_waveform[i, :] + amp_scaling*self.channel_locations[i][1],
                             'k', linewidth=0.5, picker=True)
        title_str = 'Shank %d; amp = %d muA' % (self.shank, self.stim_level)
        self.ax_sta.set_title(title_str)
        self.ax_wf_var = plt.subplot(1, 2, 2)
        self.ax_wf_var.set_title('Waveform variability')
        self.ax_wf_var.set_xlabel('Time (ms)')
        self.ax_wf_var.set_ylabel('Amplitude (muV)')
        self.ax_wf_var.set_xlim((-0.5, 1.5))

    def _add_antidromic_waveform(self, event):
        # get channel id and time from event xdata/ydata
        if isinstance(event.artist, Line2D):
            line = event.artist
            channel = None
            for i, line_ in enumerate(self.ax_sta.lines):
                if line_ == line:
                    channel = self.channel_ids[i]
                    break
            if channel is None:
                e = 'Could not find line %s in axes!' % line._label
                raise RuntimeError(e)

            # extract waveform from individual stim trials
            offset = 30
            t_index = event.ind[0] - offset
            self._var_time_index = t_index
            waveform_window_indices = np.array([int(-0.5 * 1e-3 * self.fs) + i for i in range(int(1.5 * 1e-3 * self.fs))])
            waveform_window_indices += t_index
            waveform_time_axis = waveform_window_indices / self.fs * 1e3
            wf_snippets = np.zeros((len(self.stimulus_indices), len(waveform_window_indices)))
            b, a = utils.set_up_bp_filter(300.0, 0.49*self.fs, self.fs)
            for i, stim_index in enumerate(self.stimulus_indices):
                snippet = self.antidromic_file[channel, stim_index + waveform_window_indices]
                wf_snippets[i, :] = filtfilt(b, a, snippet)
            mean_wf = np.mean(wf_snippets, axis=0)
            self.ax_wf_var.clear()
            for i in range(wf_snippets.shape[0]):
                self.ax_wf_var.plot(waveform_time_axis, wf_snippets[i, :], 'k', linewidth=0.5)
            self.ax_wf_var.plot(waveform_time_axis, mean_wf, 'r', linewidth=1.0)
            self._var_time_axis = waveform_time_axis
            self._var_waveforms = wf_snippets

            # compute variability and display in right window
            var = self._compute_antidromic_variability(wf_snippets)
            self._current_var = var
            self._current_var_channel = channel
            self._good_var_waveforms = range(len(self.channel_ids))
            title_str = 'Channel %d; peak STD = %.2f ms ' % (channel, var)
            self.ax_wf_var.set_xlim(np.min(self._var_time_axis), np.max(self._var_time_axis))
            self.ax_wf_var.set_title(title_str)
            self.ax_wf_var.set_xlabel('Time (ms)')
            self.ax_wf_var.set_ylabel('Amplitude (muV)')
            self.ax_wf_var.figure.canvas.draw()

            # highlight waveform in sta on all channels in left window
            # first remove any highlight lines if present
            self._remove_current_highlight()
            # now highlight current selection
            amp_scaling = 8.0
            for i in range(self.sta_waveform.shape[0]):
                line, = self.ax_sta.plot(waveform_time_axis + self.channel_locations[i][0],
                                        self.sta_waveform[i, waveform_window_indices + offset] +
                                        amp_scaling*self.channel_locations[i][1],
                                        'r', linewidth=1.0, picker=False)
                self._current_antidromic_highlights.append(line)
            self.ax_sta.figure.canvas.draw()

    def _add_threshold(self, threshold):
        self._threshold.append(threshold)
        x_min, x_max = self.ax_wf_var.get_xlim()
        self.ax_wf_var.plot((x_min, x_max), (threshold, threshold), color='grey', linestyle='--', linewidth=0.5)
        self.ax_wf_var.set_xlim((x_min, x_max))
        self.ax_wf_var.figure.canvas.draw()

    def _threshold_electrode(self):
        if not self._modify_var:
            return

        # compute threshold of peak deflection in +- 0.3 ms around peak of mean waveform
        min_threshold = np.min(self._threshold)
        max_threshold = np.max(self._threshold)
        if 0.5*(min_threshold + max_threshold) > 0:
            compare_func = np.max
        else:
            compare_func = np.min
        mean_wf = np.mean(self._var_waveforms, axis=0)
        peak_index = np.argmax(np.abs(mean_wf))
        window_width = int(0.3*1e-3*self.fs) # +- 0.3 ms
        start_index = max(peak_index - window_width, 0)
        stop_index = min(peak_index + window_width, len(mean_wf))
        window_indices = np.array([i for i in range(start_index, stop_index)])
        good_trials = []
        failure_trials = []
        for trial in range(self._var_waveforms.shape[0]):
            if min_threshold <= compare_func(self._var_waveforms[trial, window_indices]) <= max_threshold:
                good_trials.append(trial)
            else:
                failure_trials.append(trial)
        self._good_var_waveforms = good_trials

        mean_wf = np.mean(self._var_waveforms[good_trials, :], axis=0)
        self.ax_wf_var.clear()
        # plot waveforms from good/failure trials
        for i in failure_trials:
            self.ax_wf_var.plot(self._var_time_axis, self._var_waveforms[i, :], 'grey', linewidth=0.5)
        for i in good_trials:
            self.ax_wf_var.plot(self._var_time_axis, self._var_waveforms[i, :], 'k', linewidth=0.5)
        self.ax_wf_var.plot(self._var_time_axis, mean_wf, 'r', linewidth=1.0)
        self.ax_wf_var.set_xlim(np.min(self._var_time_axis), np.max(self._var_time_axis))
        # add thresholds back in
        x_min, x_max = self.ax_wf_var.get_xlim()
        self.ax_wf_var.plot((x_min, x_max), (self._threshold[0], self._threshold[0]),
                            color='grey', linestyle='--', linewidth=0.5)
        self.ax_wf_var.plot((x_min, x_max), (self._threshold[1], self._threshold[1]),
                            color='grey', linestyle='--', linewidth=0.5)
        self.ax_wf_var.set_xlim((x_min, x_max))

        # compute variability and display in right window
        var = self._compute_antidromic_variability(self._var_waveforms, good_trials)
        self._current_var = var
        title_str = 'Channel %d; peak STD = %.2f ms ' % (self._current_var_channel, var)
        self.ax_wf_var.set_title(title_str)
        self.ax_wf_var.set_xlabel('Time (ms)')
        self.ax_wf_var.set_ylabel('Amplitude (muV)')
        self.ax_wf_var.figure.canvas.draw()

    def _remove_last_antidromic_waveform(self):
        # TODO
        pass

    def _save_antidromic_waveforms(self):
        if not len(self._current_antidromic_highlights):
            print 'No antidromic unit selected'
            return

        # save antidromic highlight waveform on all channels
        tmpx, _ = self._current_antidromic_highlights[0].get_data()
        n_time_indices = len(tmpx)
        average_wf = np.zeros((len(self.channel_ids), n_time_indices))
        good_average_wf = np.zeros((len(self.channel_ids), n_time_indices))
        for i in range(len(self._current_antidromic_highlights)):
            x, y = self._current_antidromic_highlights[i].get_data()
            average_wf[i, :] = y[:]
            line, = self.ax_sta.plot(x, y, 'b', linewidth=1.0, picker=False)
            # add highlight waveforms to list of save antidromic waveforms
            self._saved_antidromic_highlights.append(line)
            # re-compute average waveform from good trials
            if len(self._good_var_waveforms) == len(self.channel_ids):
                good_average_wf = average_wf
            else:
                channel = self.channel_ids[i]
                waveform_window_indices = np.array([int(-0.5 * 1e-3 * self.fs) + j for j in range(int(1.5 * 1e-3 * self.fs))])
                waveform_window_indices += self._var_time_index
                wf_snippets = np.zeros((len(self.stimulus_indices), n_time_indices))
                good_wf_snippets = np.zeros((len(self._good_var_waveforms), n_time_indices))
                good_wf_cnt = 0
                b, a = utils.set_up_bp_filter(300.0, 0.49*self.fs, self.fs)
                for j, stim_index in enumerate(self.stimulus_indices):
                    snippet = self.antidromic_file[channel, stim_index + waveform_window_indices]
                    filtered_snippet = filtfilt(b, a, snippet)
                    wf_snippets[j, :] = filtered_snippet
                    if j in self._good_var_waveforms:
                        good_wf_snippets[good_wf_cnt, :] = filtered_snippet
                        good_wf_cnt += 1
                average_wf[i, :] = np.mean(wf_snippets, axis=0)
                good_average_wf[i, :] = np.mean(good_wf_snippets, axis=0)
        antidromic_id = int(len(self._saved_antidromic_highlights)//len(self.channel_ids)) - 1
        wf_outname = 'shank_%d_stim_%d_average_wf_%d.npy' % (self.shank, self.stim_level, antidromic_id)
        np.save(os.path.join(self.save_path, wf_outname), average_wf)
        good_wf_outname = 'shank_%d_stim_%d_good_average_wf_%d.npy' % (self.shank, self.stim_level, antidromic_id)
        np.save(os.path.join(self.save_path, good_wf_outname), good_average_wf)
        self._remove_current_highlight()
        self.ax_sta.figure.canvas.draw()

        # save separate figure of average waveform + only highlight waveform, plus variability panel
        # poor man's version: just copy all the lines...
        amp_scaling = 8.0
        window_indices = np.array([i for i in range(-30, 600)])  # 20 ms post-stim
        window_time_axis = window_indices / 30.0
        tmp_fig = plt.figure(antidromic_id + 10)
        tmp_ax_sta = plt.subplot(1, 2, 1)
        for i in range(self.sta_waveform.shape[0]):
            tmp_ax_sta.plot(window_time_axis + self.channel_locations[i][0],
                             self.sta_waveform[i, :] + amp_scaling*self.channel_locations[i][1],
                             'k', linewidth=0.5)
            highlight_index = i + antidromic_id*len(self.channel_ids)
            line = self._saved_antidromic_highlights[highlight_index]
            x, y = line.get_data()
            tmp_ax_sta.plot(x, y, 'r', linewidth=1.0)
        title_str = 'Shank %d; amp = %d muA' % (self.shank, self.stim_level)
        tmp_ax_sta.set_title(title_str)

        tmp_ax_wf_var = plt.subplot(1, 2, 2)
        for line in self.ax_wf_var.lines:
            x, y = line.get_data()
            tmp_ax_wf_var.plot(x, y, color=line.get_color(), linewidth=line.get_linewidth())
        quality = raw_input('Quality: ')
        title_str = 'Channel %d; peak STD = %.2f ms; quality: %s' % (self._current_var_channel, self._current_var,
                                                                     quality)
        tmp_ax_wf_var.set_title(title_str)
        tmp_ax_wf_var.set_xlabel('Time (ms)')
        tmp_ax_wf_var.set_ylabel('Amplitude (muV)')

        tmp_fig_name = 'shank_%d_stim_%d_average_wf_%d.pdf' % (self.shank, self.stim_level, antidromic_id)
        tmp_fig.set_size_inches(11, 8)
        plt.savefig(os.path.join(self.save_path, tmp_fig_name))
        plt.close(tmp_fig)

    def _compute_antidromic_variability(self, wf_snippets, trials=None):
        if trials is None:
            trials = range(wf_snippets.shape[0])
        # compute variability of peak deflection in +- 0.3 ms around peak of mean waveform
        mean_wf = np.mean(wf_snippets[trials], axis=0)
        peak_index = np.argmax(np.abs(mean_wf))
        window_width = int(0.3*1e-3*self.fs) # +- 0.3 ms
        start_index = max(peak_index - window_width, 0)
        stop_index = min(peak_index + window_width, len(mean_wf))
        window_indices = np.array([i for i in range(start_index, stop_index)])
        tmp_wf = np.abs(wf_snippets[:, window_indices])[trials]
        trial_peak_indices = np.argmax(tmp_wf, axis=1)
        peak_var = 1.0e3/self.fs*np.std(trial_peak_indices) # variability in ms
        return peak_var

    def _remove_current_highlight(self):
        while len(self._current_antidromic_highlights):
            line = self._current_antidromic_highlights.pop(-1)
            line.remove()
        self.ax_sta.figure.canvas.draw()

