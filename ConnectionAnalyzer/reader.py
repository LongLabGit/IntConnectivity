import neo


def read_wholecell_data(fnames, channels):
    '''
    load channels in channels dict from Axon abf files given in fname list
    :return: list of neo.core.AnalogSignal
    '''
    analogSignals = [{} for i in range(len(fnames))]
    for i in range(len(fnames)):
        readABF = neo.io.AxonIO(fnames[i])
        abfBlock = readABF.read_block(lazy=False, cascade=True)
        for c in channels.keys():
            # from neo documentation for segment/analogsignal list indexing:
            signal = abfBlock.segments[0].analogsignals[c]
            analogSignals[i][channels[c]] = signal.flatten()

    return analogSignals

def read_Intan_digital_file(fname, nChannels, samplingRate):
    '''
    load Intan binary file with digital signal
    :param fname: file name
    :param nChannels: number of channels
    :param samplingRate: sampling rate (Hz)
    :return: array of neo AnalogSignals
    '''
    r = neo.io.RawBinarySignalIO(fname)
    seg = r.read_segment(lazy=False, cascade=True, nbchannel=nChannels, dtype='uint16', sampling_rate=samplingRate,
                         rangemin=0, rangemax=2**16)  # avoid neo IO built-in unit conversion
    return seg.analogsignals

def read_Intan_analog_file(fname, nChannels, samplingRate):
    '''
    load Intan binary file with analog signal
    :param fname: file name
    :param nChannels: number of channels
    :param samplingRate: sampling rate (Hz)
    :return: array of neo AnalogSignals
    '''
    r = neo.io.RawBinarySignalIO(fname)
    seg = r.read_segment(lazy=False, cascade=True, nbchannel=nChannels, dtype='uint16', sampling_rate=samplingRate,
                         rangemin=0, rangemax=2**16)  # avoid neo IO built-in unit conversion
    seg.analogsignals *= 0.000050354  # Intan magic number: convert to volts
    return seg.analogsignals