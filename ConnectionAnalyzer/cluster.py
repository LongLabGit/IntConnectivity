from neo.core import Unit, SpikeTrain

# class Cluster(Unit):
#     '''lightweight object representing a single extracellular unit'''
#     def __init__(self, clusterID, group, spikeTimes, shank, maxChannel, coordinates, firingRate):
#         super(Unit, self).__init__(name=str(clusterID))
#         self.clusterID = clusterID
#         self.group = group
#         if hasattr(spikeTimes, 'times'):
#             # super(Unit, self).spiketrains.append(spikeTimes)
#             self.spiketrains.append(spikeTimes)
#         else:
#             newSpiketrain = SpikeTrain(spikeTimes, units='sec', t_stop=max(spikeTimes))
#             # super(Unit, self).spiketrains.append(newSpiketrain)
#             self.spiketrains.append(newSpiketrain)
#         self.shank = shank
#         self.maxChannel = maxChannel
#         self.coordinates = coordinates
#         self.firingRate = firingRate
#         self.mergedDuplicates = False
#         self.nrMergedDuplicateSpikes = 0
