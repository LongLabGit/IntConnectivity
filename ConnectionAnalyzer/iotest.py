import os
import reader
import recording_alignment
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo

# # test KiloSort cluster reading
# dataFolder = 'Z:\\Robert\\INT_connectivity\\SiProbe\\ProbeBird_101917\\SiProbe\\Continuous_400_cut'
# ClusteringSrcFolder = 'E:\\User\\project_src\\physiology\\Clustering'
# version = 'dev'
# keep_group = 'good'
# samplingRate = 20000.0
#
# clusters = reader.read_clusters(dataFolder, ClusteringSrcFolder, version, keep_group, samplingRate)

# # test abf analog file reading
# dataFolder = 'Z:\\Robert\\INT_connectivity\\SiProbe\\ProbeBird_101917\\WC'
# fileNames = [os.path.join(dataFolder, 'c1_0001.abf'), os.path.join(dataFolder, 'c1_0002.abf')]
# channels = 0, 3
# WCSignals = reader.read_wholecell_data(fileNames, channels)
# current = WCSignals[0][0]
# time = np.linspace(current.t_start.magnitude, current.t_stop.magnitude, len(current))
# plt.figure(1)
# plt.plot(time, current, linewidth=0.5)
# plt.show()

# test alignment of synchronization signals
ProbeAnalogDataName = 'Z:\\Robert\\INT_connectivity\\SiProbe\\ProbeBird_101917\\SiProbe\\Continuous_400_cut\\analoginToDigitalin.dat'
probePulseSignal = reader.read_Intan_digital_file(ProbeAnalogDataName, 1, 20000)
WCDataFolder = 'Z:\\Robert\\INT_connectivity\\SiProbe\\ProbeBird_101917\\WC'
WCFileNames = [os.path.join(WCDataFolder, 'c1_0001.abf')]
channels = {0: 'current', 3: 'pulseIn'}
WCSignals = reader.read_wholecell_data(WCFileNames, channels)
current = WCSignals[0]['current']
WCPulseSignal = WCSignals[0]['pulseIn']
# this experiment: probe and WC roughly starting at 0; hence, use same alignment boundaries
alignmentPeriods = (WCPulseSignal.t_start, WCPulseSignal.t_stop), (WCPulseSignal.t_start, WCPulseSignal.t_stop)
alignments = recording_alignment.align_paired_recordings((probePulseSignal[0], WCPulseSignal), alignmentPeriods, 0.5, 0.1)

probeToWC = alignments[0]
WCToProbe = alignments[1]

probeTStartAligned = recording_alignment.linear_func(probePulseSignal[0].t_start, probeToWC[0], probeToWC[1]*pq.s)
probeTStopAligned = recording_alignment.linear_func(probePulseSignal[0].t_stop, probeToWC[0], probeToWC[1]*pq.s)
probeTSampleAligned = probePulseSignal[0].sampling_period*probeToWC[0]
probeAlignedToWC = neo.core.AnalogSignal(probePulseSignal[0], t_start=probeTStartAligned, t_stop=probeTStopAligned, sampling_period=probeTSampleAligned)
probeAlignedToWCTime = np.linspace(probeAlignedToWC.t_start.magnitude, probeAlignedToWC.t_stop.magnitude, len(probeAlignedToWC))

WCTStartAligned = recording_alignment.linear_func(WCPulseSignal.t_start, WCToProbe[0], WCToProbe[1]*pq.s)
WCTStopAligned = recording_alignment.linear_func(WCPulseSignal.t_stop, WCToProbe[0], WCToProbe[1]*pq.s)
WCTSampleAligned = WCPulseSignal.sampling_period*WCToProbe[0]
WCAlignedToProbe = neo.core.AnalogSignal(WCPulseSignal, t_start=WCTStartAligned, t_stop=WCTStopAligned, sampling_period=WCTSampleAligned)
WCAlignedToProbeTime = np.linspace(WCAlignedToProbe.t_start.magnitude, WCAlignedToProbe.t_stop.magnitude, len(WCAlignedToProbe))

WCPulseTime = np.linspace(WCPulseSignal.t_start.magnitude, WCPulseSignal.t_stop.magnitude, len(WCPulseSignal))
probePulseTime = np.linspace(probePulseSignal[0].t_start.magnitude, probePulseSignal[0].t_stop.magnitude, len(probePulseSignal[0]))

plt.figure(1)
plt.plot(WCPulseTime, WCPulseSignal, 'k')
plt.plot(probeAlignedToWCTime, probeAlignedToWC, 'r')
plt.title('probe (red) aligned to WC (black)')
plt.figure(2)
plt.plot(probePulseTime, probePulseSignal[0], 'k')
plt.plot(WCAlignedToProbeTime, WCAlignedToProbe, 'r')
plt.title('probe (black) aligned to WC (red)')
plt.show()
