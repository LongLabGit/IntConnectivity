##############################################
# simply view synchronization signals on probe
# and from recordings in separate figures
##############################################

import sys, os, ast
import matplotlib.pyplot as plt
import numpy as np
from ConnectionAnalyzer import reader

def main(experimentInfoName):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())

    ProbeAnalogDataName = os.path.join(experimentInfo['SiProbe']['DataBasePath'], 'analoginToDigitalin.dat')
    syncChannels = 1
    samplingRate = experimentInfo['SiProbe']['SamplingRate']
    probePulseSignal = reader.read_Intan_digital_file(ProbeAnalogDataName, syncChannels, samplingRate)
    tStart = probePulseSignal[0].t_start.magnitude
    tStop = probePulseSignal[0].t_stop.magnitude
    probePulseTime = np.linspace(tStart, tStop, len(probePulseSignal[0]))
    plt.figure(1)
    plt.plot(probePulseTime, probePulseSignal[0], 'k', linewidth=0.5)
    plt.title('Sync signal on SiProbe')

    WCDataFolder = experimentInfo['WC']['DataBasePath']
    WCFileNames = [os.path.join(WCDataFolder, fname) for fname in experimentInfo['WC']['RecordingFilenames']]
    WCSignals = reader.read_wholecell_data(WCFileNames, experimentInfo['WC']['Channels'])
    nSignals = len(WCSignals)
    axes = []
    plt.figure(2)
    plt.title('Sync signals during WC recordings')
    for i, signal in enumerate(WCSignals):
        tStart = signal['pulseIn'].t_start.magnitude
        tStop = signal['pulseIn'].t_stop.magnitude
        WCPulseTime = np.linspace(tStart, tStop, len(signal['pulseIn']))
        axes.append(plt.subplot(nSignals, 1, i+1))
        axes[-1].plot(WCPulseTime, signal['pulseIn'], 'k', linewidth=0.5)
        axes[-1].set_title('WC recording ' + experimentInfo['WC']['RecordingFilenames'][i])

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        experimentInfoName = sys.argv[1]
        main(experimentInfoName)
    else:
        print 'Error: Experiment info file required!'