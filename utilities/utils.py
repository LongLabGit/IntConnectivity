# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import ast
import numpy as np
from scipy import signal


intan_constant = 0.195


# -----------------------------------------------------------------------------
# Recording
# -----------------------------------------------------------------------------


def _get_recording_chunk(recording, chunk_start, chunk_stop, fs):
    sample_start = int(chunk_start * fs)
    sample_stop = int(chunk_stop * fs)
    if sample_stop <= sample_start:
        raise RuntimeError('start less than stop')

    return recording[:, sample_start:sample_stop]


def _get_n_samples(fname, nchannels, dtype=np.dtype('int16')):
    file_info = os.stat(fname)
    file_size = file_info.st_size
    nsamples = int(file_size / (dtype.itemsize * nchannels))
    return nsamples


def _crop_recording(recording_name, nchannels, out_name, nsamples, n_snippets, snippets_seconds, fs,
                    dtype=np.dtype('int16')):
    recording = load_recording(recording_name, nchannels)
    new_file = np.memmap(out_name, dtype=dtype.name, mode='w+', shape=(nchannels, nsamples), order='F')
    sample_offset = 0
    for i in range(n_snippets):
        t_start = snippets_seconds[i, 0]
        t_stop = snippets_seconds[i, 1]
        outstr = 'Copying %d channels from file %s to file %s from time %.1f to %.1f s' % \
                 (nchannels, recording_name, out_name, t_start, t_stop)
        print(outstr)
        chunk = _get_recording_chunk(recording, t_start, t_stop, fs)
        start_sample = sample_offset
        stop_sample = start_sample + chunk.shape[1]
        new_file[:, start_sample:stop_sample] = chunk[:, :]
        sample_offset = stop_sample


def load_recording(fname, nchannels, dtype=np.dtype('int16')):
    """returns pointer to binary file
    rows: channel numbers
    columns: samples
    """
    file_info = os.stat(fname)
    file_size = file_info.st_size
    nsamples = int(file_size / (dtype.itemsize * nchannels))
    return np.memmap(fname, dtype=dtype.name, mode='r', shape=(int(nchannels), int(nsamples)), order='F')


def copy_recording_chunk(fname, outname, nchannels, chunk_start, chunk_stop, fs, dtype=np.dtype('int16')):
    """copy binary file from time chunk_start to chunk_stop"""
    if fname == outname:
        e = 'Chunk output name cannot be same as original file'
        raise ValueError(e)

    sample_start = int(chunk_start * fs)
    sample_stop = int(chunk_stop * fs)
    if sample_stop <= sample_start:
        raise RuntimeError('start less than stop')
    nsamples = sample_stop - sample_start

    outstr = 'Copying %d channels from file %s to file %s from time %.1f to %.1f s; fs = %.1f' % \
             (nchannels, fname, outname, chunk_start, chunk_stop, fs)
    print(outstr)
    recording = load_recording(fname, nchannels, dtype)
    recording_chunk = np.memmap(outname, dtype=dtype.name, mode='w+', shape=(nchannels, nsamples), order='F')
    recording_chunk[:, :] = recording[:, sample_start:sample_stop]
    recording_chunk.flush()


def crop_recording_from_file(recording_name, out_name, snippets_name, nchannels, fs, dtype=np.dtype('int16')):
    """
    crop binary file into specified snippets
    snippets: csv file where first row is header, each other row is start end end time of a snippet in format
    start min   start sec   start ms    stop min    stop sec    stop ms
    """
    recording_samples = _get_n_samples(recording_name, nchannels)
    recording_duration = recording_samples*1.0/fs

    snippets_times = np.loadtxt(snippets_name, skiprows=1, delimiter='\t')
    n_snippets = len(snippets_times.shape)
    if n_snippets > 1:
        snippets_seconds = np.zeros((snippets_times.shape[0], 2))
        nsamples = 0
        for i in range(snippets_times.shape[0]):
            t_start = snippets_times[i, 0] * 60.0 + snippets_times[i, 1] + snippets_times[i, 2] * 1e-3
            t_stop = snippets_times[i, 3] * 60.0 + snippets_times[i, 4] + snippets_times[i, 5] * 1e-3
            if t_stop > recording_duration:
                print('Requested t_stop larger than recording duration; stopping at end of recording')
                t_stop = recording_duration
            snippets_seconds[i, 0] = t_start
            snippets_seconds[i, 1] = t_stop
            sample_start = int(t_start * fs)
            sample_stop = int(t_stop * fs)
            nsamples += sample_stop - sample_start
    else:
        snippets_seconds = np.zeros((1, 2))
        t_start = snippets_times[0] * 60.0 + snippets_times[1] + snippets_times[2] * 1e-3
        t_stop = snippets_times[3] * 60.0 + snippets_times[4] + snippets_times[5] * 1e-3
        if t_stop > recording_duration:
            print('Requested t_stop larger than recording duration; stopping at end of recording')
            t_stop = recording_duration
        snippets_seconds[0, 0] = t_start
        snippets_seconds[0, 1] = t_stop
        sample_start = int(t_start * fs)
        sample_stop = int(t_stop * fs)
        nsamples = sample_stop - sample_start

    _crop_recording(recording_name, nchannels, out_name, nsamples, n_snippets, snippets_seconds)


def crop_recording_from_times(recording_name, out_name, snippets_times, nchannels, fs, dtype=np.dtype('int16')):
    """
    snippets_times has to be of shape (n_snippets, 2) (i.e., row for each snippet, with t_start and t_stop in columns)
    units of t_start and t_stop: second
    """
    recording_samples = _get_n_samples(recording_name, nchannels)
    recording_duration = recording_samples*1.0/fs

    n_snippets = snippets_times.shape[0]
    nsamples = 0
    for i in range(n_snippets):
        t_start = snippets_times[i, 0]
        t_stop = snippets_times[i, 1]
        if t_stop > recording_duration:
            print('Requested t_stop larger than recording duration; stopping at end of recording')
            t_stop = recording_duration
            snippets_times[i, 1] = recording_duration
        sample_start = int(t_start * fs)
        sample_stop = int(t_stop * fs)
        nsamples += sample_stop - sample_start

    _crop_recording(recording_name, nchannels, out_name, nsamples, n_snippets, snippets_times)


def copy_channels(recording_name, out_name, nchannels, channels, fs, dtype=np.dtype('int16')):
    """
    copy specified channels from recording file
    """
    pass


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------


def _load_burst_info_csv(fname, skiprows=0, delimiter=','):
    """
    load burst info from csv file with columns cluster id, burst id, type
    :param fname: filename
    :param skiprows: number of rows to skip (default: 0)
    :param delimiter: column delimiter string (default: ',')
    :return: tuples cluster_ids, burst_ids, cell_types
    """
    cluster_ids, burst_ids, cell_types = [], [], []
    with open(fname, 'r') as cluster_file:
        line_cnt = 0
        for line in cluster_file:
            line_cnt += 1
            if line_cnt <= skiprows or not line:
                continue
            split_line = line.strip().split(delimiter)
            if len(split_line) != 3:
                e = 'Number of columns is %d; expected 3' % len(split_line)
                raise RuntimeError(e)
            cluster_ids.append(int(split_line[0]))
            burst_ids.append(int(split_line[1]))
            cell_types.append(split_line[2])

    return tuple(cluster_ids), tuple(burst_ids), tuple(cell_types)


def _load_cell_info_csv(fname, skiprows=0, delimiter=','):
    """
    load burst info from csv file with columns cell id, cluster id, burst id, type
    :param fname: filename
    :param skiprows: number of rows to skip (default: 0)
    :param delimiter: column delimiter string (default: ',')
    :return: tuples cell_ids, cluster_ids, burst_ids, cell_types
    """
    cell_ids, cluster_ids, burst_ids, cell_types = [], [], [], []
    with open(fname, 'r') as cluster_file:
        line_cnt = 0
        for line in cluster_file:
            line_cnt += 1
            if line_cnt <= skiprows or not line:
                continue
            split_line = line.strip().split(delimiter)
            if len(split_line) != 4:
                e = 'Number of columns is %d; expected 4' % len(split_line)
                raise RuntimeError(e)
            cell_ids.append(int(split_line[0]))
            cluster_ids.append(int(split_line[1]))
            burst_ids.append(int(split_line[2]))
            cell_types.append(split_line[3])

    return tuple(cell_ids), tuple(cluster_ids), tuple(burst_ids), tuple(cell_types)


def load_burst_info(experiment_info_name, cells=False):
    """
    Look up file name with burst information in experiment_info dictionary
    return: pair of tuples containing matched cluster IDs and burst IDs
    """
    with open(experiment_info_name, 'r') as data_file:
        experiment_info = ast.literal_eval(data_file.read())

    fname = os.path.join(experiment_info['SiProbe']['ClusterBasePath'], experiment_info['SiProbe']['BurstIdentity'])
    # cluster_ids_, burst_ids_ = np.loadtxt(fname, skiprows=1, unpack=True, delimiter=',')
    # cluster_ids = cluster_ids_.astype(int)
    # burst_ids = burst_ids_.astype(int)
    if cells == False:
        cluster_ids, burst_ids, cell_types = _load_burst_info_csv(fname, skiprows=1, delimiter=',')
        return cluster_ids, burst_ids, cell_types
    else:
        cell_ids, cluster_ids, burst_ids, cell_types = _load_cell_info_csv(fname, skiprows=1, delimiter=',')
        return cell_ids, cluster_ids, burst_ids, cell_types

# -----------------------------------------------------------------------------
# signal processing
# -----------------------------------------------------------------------------


def set_up_bp_filter(highpass, lowpass, fs):
    filter_order = 3
    return signal.butter(filter_order, (highpass / (fs / 2.), lowpass / (fs / 2.)), 'bandpass')


def normalize_trace(trace, new_min=-1.0, new_max=1.0):
    old_min = float(np.min(trace))
    old_max = float(np.max(trace))
    return (trace - old_min)*(new_max - new_min)/(old_max - old_min) + new_min


def normalize_audio_trace(trace, new_min=-1.0, new_max=1.0):
    new_trace = np.zeros(len(trace), dtype=np.dtype('float64'))
    new_trace[:] = trace[:]
    old_min = float(np.min(trace))
    old_max = float(np.max(trace))
    return (new_trace - old_min)*(new_max - new_min)/(old_max - old_min) + new_min


def get_threshold_crossings(trace, threshold, direction='down'):
    trace_ = trace - threshold
    tmp1 = trace_[:-1]
    tmp2 = trace_[1:]
    if direction == 'up':
        trace_ *= -1.0
    crossings = np.where((tmp1*tmp2 < 0) * (tmp2 < tmp1))
    return crossings[0] + 1 # add skipped index
