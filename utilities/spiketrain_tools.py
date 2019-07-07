# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Spiketrain tools
# -----------------------------------------------------------------------------


def event_times_from_spikes(spike_times, interval):
    # each event consists of tuple (center, start, stop) in seconds
    # spike times: array of spike times (in s)
    # interval: max ISI in ms
    dt = 0.001*interval # in seconds
    event_spikes = []
    event_count = 0
    for i in range(len(spike_times)):
        if i == 0:
            event_spikes.append(event_count)
            continue
        if spike_times[i] - spike_times[i - 1] <= dt:
            event_spikes.append(event_count)
        if spike_times[i] - spike_times[i - 1] > dt:
            event_count += 1
            event_spikes.append(event_count)

    events = np.unique(event_spikes)
    event_times = []
    event_spike_times = []
    for event in events:
        tmp_times = spike_times[event_spikes == event]
        event_center = 0.5*(tmp_times[0] + tmp_times[-1])
        event_start = np.min(tmp_times)
        event_stop = np.max(tmp_times)
        event_times.append((event_center, event_start, event_stop))
        event_spike_times.append(tmp_times)

    return event_times, event_spike_times


def mean_firing_rate_from_aligned_spikes(trial_spike_times, trial_start, trial_end, binsize):
    """
    compute mean firing rate from aligned spike trains
    :param trial_spike_times: iterable of len(trials); each element contains iterable of spike times (in s)
    :param binsize: bin size for histogram (in s)
    :return: hist, bins (bins is len(hist) + 1 and contains start of each bin and end of last bin)
    """
    all_spike_times = []
    for i in range(len(trial_spike_times)):
        all_spike_times.extend(trial_spike_times[i])

    nr_bins = int(np.ceil((trial_end - trial_start)/binsize))
    bins = trial_start + np.array(range(nr_bins))*binsize
    hist, _ = np.histogram(all_spike_times, bins)
    hist = hist*1.0/len(trial_spike_times)

    return hist, bins
