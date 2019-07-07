import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import utilities as utils

# waveform_window_indices = np.array([int(-0.5 * 1e-3 * fs) + i for i in range(int(1.5 * 1e-3 * fs))])
# waveform_time_axis = waveform_window_indices / fs * 1e3


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
        self.antidromic_units = []

    def connect(self):
        """
        connect to all the events we need
        keyboard press 'a' to define new crossing
        click on channel at specific time to add
        keyboard press 'u' to undo (remove last crossing)
        keyboard press 's' to save
        keyboard press 'q' to quit
        """
        # self.cidpress = self.ax_sta.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidpress = self.ax_sta.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidkey = self.ax_sta.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def disconnect(self):
        self.ax_sta.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax_sta.figure.canvas.mpl_disconnect(self.cidkey)

    def on_pick(self, event):
        if not self._modify_antidromic:
            return
        self._add_antidromic_waveform(event)
        self._modify_antidromic = False

    def on_key(self, event):
        if event.key == 'a' or event.key == 'A':
            self._modify_antidromic = True
            print 'Click to add antidromic waveform'
        if event.key == 'u' or event.key == 'U':
            self._modify_antidromic = True
            print 'Removing last antidromic waveform'
            self._remove_last_antidromic_waveform()
            self._modify_antidromic = False
        if event.key == 's' or event.key == 'S':
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
        print 'Picked something!'
        if isinstance(event.artist, Line2D):
            print 'This something is a line...'
        # highlight waveform in sta on all channels in left window
        # extract waveform from individual stim trials,
        # compute variability and display in right window
        pass

    def _remove_last_antidromic_waveform(self):
        pass

    def _save_antidromic_waveforms(self):
        pass
