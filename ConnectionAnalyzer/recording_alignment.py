import numpy as np
import scipy
# import matplotlib.pyplot as plt
# import quantities as pq

def align_paired_recordings(pulseSignals, alignmentPeriods, pulseThreshold, minimumInterval=0):
    '''
    takes two synchronizing signals (pulse signals) as input and calculates
    offset and scaling necessary to align pulse onsets
    :param pulseSignals: pair of neo AnalogSignals
    :param alignmentPeriods: pair of onset/offset for periods to consider for alignment for each signal
    :param pulseThreshold: threshold for pulse onset detection
    :param minimumInterval: in s; optional; can be used to avoid incorrect short-interval pulse onset times
    :return: alignmentParameters (tuple)
    alignmentParameters[i] = (a, b) where a is alignment scale, b is alignment offset
    '''
    time0 = np.linspace(pulseSignals[0].t_start.magnitude, pulseSignals[0].t_stop.magnitude, len(pulseSignals[0]))
    time1 = np.linspace(pulseSignals[1].t_start.magnitude, pulseSignals[1].t_stop.magnitude, len(pulseSignals[1]))
    pulseTimes0_ = detect_threshold_crossings(pulseSignals[0], time0, pulseThreshold)
    pulseTimes1_ = detect_threshold_crossings(pulseSignals[1], time1, pulseThreshold)
    pulseTimes0 = [pulseTimes0_[0]]
    pulseTimes1 = [pulseTimes1_[0]]
    for i in range(1, len(pulseTimes0_)):
        dt = pulseTimes0_[i] - pulseTimes0_[i-1]
        if dt > minimumInterval:
            pulseTimes0.append(pulseTimes0_[i])
    for i in range(1, len(pulseTimes1_)):
        dt = pulseTimes1_[i] - pulseTimes1_[i-1]
        if dt > minimumInterval:
            pulseTimes1.append(pulseTimes1_[i])

    pulseTimes0 = np.array(pulseTimes0)
    pulseTimes1 = np.array(pulseTimes1)
    alignmentPulses0 = pulseTimes0[np.where((pulseTimes0 >= alignmentPeriods[0][0]) * (pulseTimes0 <= alignmentPeriods[0][1]))]
    alignmentPulses1 = pulseTimes1[np.where((pulseTimes1 >= alignmentPeriods[1][0]) * (pulseTimes1 <= alignmentPeriods[1][1]))]

    alignment_0To1 = fit_linear_alignment((alignmentPulses0, alignmentPulses1))
    alignment_1To0 = fit_linear_alignment((alignmentPulses1, alignmentPulses0))

    print 'alignment 0 to 1: %.6f, %.2f' % (alignment_0To1[0], alignment_0To1[1])
    print 'alignment 1 to 0: %.6f, %.2f' % (alignment_1To0[0], alignment_1To0[1])
    # plt.figure(1)
    # plt.plot(alignmentPulses0 - alignmentPulses0[0], alignmentPulses1 - alignmentPulses1[0], 'kx')
    # plt.show()

    return alignment_0To1, alignment_1To0

def detect_threshold_crossings(signal, time, threshold, upward=True):
    '''
    takes analog signal and time and finds time points of threshold crossings
    :param signal: analog signal with pulses
    :param time: time array of analog signal
    :param threshold: same units of signal; crossings are strictly > threshold
    :param upward: optional; if False, detects downward crossings
    :return: array of threshold crossing times
    '''
    if not upward:
        signal *= -1.0
        threshold *= -1.0
    # try:
    #     dummy = signal[0] - threshold
    # except ValueError:
    #     threshold *= signal.units
    try:
        signal -= threshold
    except ValueError:
        threshold *= signal.units
        signal -= threshold

    tmpSignal = signal.magnitude
    sign = tmpSignal[:-1]*tmpSignal[1:]
    putative_threshold_crossings = np.where(sign < 0)[0]
    threshold_crossings = []
    for i in range(len(putative_threshold_crossings)):
        n = putative_threshold_crossings[i]
        if signal[n] < signal[n + 1]:
            threshold_crossings.append(time[n])

    # threshold_crossings = []
    # for i in range(len(signal) - 1):
    #     if signal[i] <= threshold and signal[i+1] > threshold:
    #         threshold_crossings.append(time[i+1])

    return np.array(threshold_crossings)

def fit_linear_alignment(timepoints):
    '''
    :param timepoints: pair of tuples t1, t2 to be aligned
    :return: tuple a, b of linear fit t2 = a*t1 + b
    '''
    slopeGuess = 1.0
    offsetGuess = timepoints[1][0] - timepoints[0][0]
    pOpt, pCov = scipy.optimize.curve_fit(linear_func, timepoints[0], timepoints[1], p0=(slopeGuess, offsetGuess))
    return pOpt

def linear_func(x, a, b):
    return a * x + b