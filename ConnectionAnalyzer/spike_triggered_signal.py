import numpy as np
import neo
import quantities as pq

class SpikeTriggeredSignal(object):
    '''
    simple container holding SnippetArray,
    average signal, aligned window
    '''
    def __init__(self, sts, sta, window=None):
        self.sts = sts
        self.sta = sta
        if window is None:
            self.window = self.sta.t_start, self.sta.t_stop
        else:
            try:
                _ = self.window[0] + self.sta.t_start
                self.window = window
            except ValueError:
                self.window = window*pq.s

class Snippet(object):
    '''
    container for AnalogSignal snippet
    with attached information about original signal
    and time point within original signal
    '''
    def __init__(self, signal, originalSignalName, originalSignalTime, spike_time):
        self.signal = signal
        self.signal_name = originalSignalName
        self.snippet_timepoint = originalSignalTime
        self.snippet_t_start = self.snippet_timepoint + self.signal.t_start
        self.snippet_t_stop = self.snippet_timepoint + self.signal.t_stop
        self.snippet_spike_time = spike_time

class SnippetArray(object):
    '''
    contains array of individual spike time-aligned signals,
    as well as filenames and spike times within these files and unaligned spike times
    '''
    def __init__(self, snippets, signalNames, snippet_timepoints, spike_times):
        self.snippets = snippets
        self.signal_names = np.array(signalNames)
        self.snippet_timepoints = np.array(snippet_timepoints)
        self.snippet_spike_times = spike_times

    # def compute_average(self, files=None, times=None):
    #     '''
    #     compute average trace across snippets
    #     :param files: optional; array of files to be included in average; default: all
    #     :param times: optional; array of same length as files; determines snippetTimePoints
    #     in each file to be included in average trace; default: all
    #     :return: averageTrace: neo.AnalogSignal
    #     '''
    #     if files is None:
    #         selectedSnippets = self.snippets
    #     else:
    #         selectedArray = np.array([0 for s in range(len(self.snippets))])
    #         for i, file in enumerate(files):
    #             fileMatch = self.signalNames in files
    #             timeMatch = (self.snippetTimePoints >= times[i][0])*(self.snippetTimePoints <= times[i][1])
    #             selectedArray += fileMatch*timeMatch
    #         selectedSnippets = self.signals[np.where(selectedArray)]
    #
    #     result = np.mean(selectedSnippets, axis=0)
    #     exampleSignal = self.snippets[0].signal
    #     result = neo.AnalogSignal(result, units=exampleSignal.units, t_start=exampleSignal.t_start,
    #                               t_stop=exampleSignal.t_stop, sampling_rate=exampleSignal.sampling_rate)
    #
    #     return SpikeTriggeredSignal(SnippetArray(selectedSnippets), result)
    #
    # def compute_function(self, func, files=None, times=None, **kwargs):
    #     '''
    #     apply function across snippets
    #     :param func: function object taking array of snippets as input, and returning one trace
    #     :param files: optional; array of files to be included in average; default: all
    #     :param times: optional; array of same length as files; determines snippetTimePoints
    #     in each file to be included in average trace; default: all
    #     :param kwargs: optional arguments; passed into func
    #     :return: averageTrace: neo.AnalogSignal
    #     '''
    #     if files is None:
    #         selectedSnippets = self.snippets
    #     else:
    #         selectedArray = np.array([0 for s in range(len(self.snippets))])
    #         for i, file in enumerate(files):
    #             fileMatch = self.signalNames in files
    #             timeMatch = (self.snippetTimePoints >= times[i][0])*(self.snippetTimePoints <= times[i][1])
    #             selectedArray += fileMatch*timeMatch
    #         selectedSnippets = self.signals[np.where(selectedArray)]
    #
    #     result = func(selectedSnippets, kwargs)
    #     if not hasattr(result, 'units'):
    #         exampleSignal = self.snippets[0].signal
    #         result = neo.AnalogSignal(result, units=exampleSignal.units, t_start=exampleSignal.t_start,
    #                                   t_stop=exampleSignal.t_stop, sampling_rate=exampleSignal.sampling_rate)
    #
    #     return SpikeTriggeredSignal(SnippetArray(selectedSnippets), result)