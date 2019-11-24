import numpy as np
import scipy.io


class Syllable(object):
    def __init__(self, label, motifs, onsets, offsets):
        self.label = label # string
        self.motifs = motifs # motif IDs
        self.onsets = onsets # onset times within motifs (sorted by motif IDs)
        self.offsets = offsets # offset times within motifs (sorted by motif IDs)


def load_syllables_from_egui(fname):
    """
    :param fname: filename for eGUI MATLAB data structure
    :return: dict with keys syllable ID and elements of class Syllable
    """
    egui_data_ = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
    egui_data = egui_data_['dbase']

    # get motif order from file names
    # format: motif_N.wav -> extract N
    motif_ids = []
    for i in range(len(egui_data.SoundFiles)):
        motif_name = egui_data.SoundFiles[i].name
        split_name = motif_name.split('_')
        motif_id = int(split_name[1][:-4])
        motif_ids.append(motif_id)

    # get all syllable labels
    max_n_segments = 0
    motif_for_labels = None
    for i in range(len(egui_data.SegmentIsSelected)):
        if np.sum(egui_data.SegmentIsSelected[i]) > max_n_segments:
            max_n_segments = np.sum(egui_data.SegmentIsSelected[i])
            motif_for_labels = i
    syllable_labels = np.unique(egui_data.SegmentTitles[motif_for_labels])

    egui_syllables = {}
    for label in syllable_labels:
        tmp_motif_ids = []
        tmp_onsets = []
        tmp_offsets = []
        for i in range(len(motif_ids)):
            motif_syllables = egui_data.SegmentTitles[i]
            good_syllables = egui_data.SegmentIsSelected[i]
            if label in motif_syllables:
                syllable_index = np.where(motif_syllables == label)[0]
                for index in syllable_index:
                    if good_syllables[index]:
                        tmp_motif_ids.append(motif_ids[i])
                        tmp_onsets.append(egui_data.SegmentTimes[i][index, 0] * 1.0 / egui_data.Fs)
                        tmp_offsets.append(egui_data.SegmentTimes[i][index, 1] * 1.0 / egui_data.Fs)

        tmp_motif_ids = np.array(tmp_motif_ids)
        tmp_onsets = np.array(tmp_onsets)
        tmp_offsets = np.array(tmp_offsets)
        motif_order = np.argsort(tmp_motif_ids)
        new_syllable = Syllable(label, tmp_motif_ids[motif_order], tmp_onsets[motif_order], tmp_offsets[motif_order])
        egui_syllables[label] = new_syllable

    return egui_syllables


def calculate_reference_syllables(egui_syllables):
    # calculate mean on-/offset per syllable
    reference_syllables = {}
    for label in egui_syllables:
        syllable = egui_syllables[label]
        mean_onset = np.mean(syllable.onsets)
        mean_offset = np.mean(syllable.offsets)
        reference_syllables[label] = mean_onset, mean_offset

    return reference_syllables


def map_trial_time_to_trial_syllable(t_trial, trial_nr, egui_syllables):
    '''
    return syllable within which t_trial falls. If t_trial falls in gap, returns None.
    :param t_trial:
    :param trial_nr:
    :param egui_syllables:
    :return:
    '''
    for label in egui_syllables:
        syllable = egui_syllables[label]
        if trial_nr not in syllable.motifs:
            continue
        trial_index = np.where(syllable.motifs == trial_nr)[0]
        trial_onset = syllable.onsets[trial_index]
        trial_offset = syllable.offsets[trial_index]
        # if trial_onset <= t_trial <= trial_offset:
        #     return label, (t_trial - trial_onset)[0] # dirty hack to return non-array but float
        # DIRTY HACK for C23:
        for i in range(len(trial_onset)):
            if trial_onset[i] <= t_trial <= trial_offset[i]:
                return label, t_trial - trial_onset[i]

    return None, None


def map_trial_time_to_trial_syllable_onset(t_trial, trial_nr, egui_syllables):
    '''
    return syllable DURING/AFTER which t_trial occurs (i.e., including gap after syllable).
    Assumes that t_trial does not go beyond motif end (no sanity checking done).
    :param t_trial:
    :param trial_nr:
    :param egui_syllables:
    :return:
    '''
    for label in egui_syllables:
        syllable = egui_syllables[label]
        if trial_nr not in syllable.motifs:
            continue
        trial_index = np.where(syllable.motifs == trial_nr)[0]
        trial_onset = syllable.onsets[trial_index]
        if trial_onset <= t_trial:
            return label, (t_trial - trial_onset)[0] # dirty hack to return non-array but float

    return None, None


def map_trial_time_to_reference_syllable(t_trial, trial_nr, egui_syllables):
    reference_syllables = calculate_reference_syllables(egui_syllables)

    for label in egui_syllables:
        syllable = egui_syllables[label]
        if trial_nr not in syllable.motifs:
            continue
        trial_index = np.where(syllable.motifs == trial_nr)[0]
        trial_onset = syllable.onsets[trial_index]
        trial_offset = syllable.offsets[trial_index]
        ref_onset = reference_syllables[label][0]
        ref_offset = reference_syllables[label][1]
        t_ = t_trial - trial_onset
        mapped_t_trial = ref_onset + t_ * (ref_offset - ref_onset) / (trial_offset - trial_onset)
        if ref_onset <= mapped_t_trial <= ref_offset:
            return label, mapped_t_trial
        # DIRTY HACK for C23:
        # for t__ in mapped_t_trial:
        #     if ref_onset <= t__ <= ref_offset:
        #         return label, t__

    return None, None
