###################################
# Visualization of systematic
# KiloSort parameter explorations
###################################
import numpy as np
import matplotlib.pyplot as plt
import sys, os.path, ast
import quantities as pq

# summary file format definitions
refSpikeCol = 0
thresh1Col = 1
thresh2Col = 2
var1Col = 3
var2Col = 4
SNRCols = [5, 10, 15, 20, 25, 30]
clusterNrCols = [6, 11, 16, 21, 26, 31]
FPCols = [7, 12, 17, 22, 27, 32]
FNCols = [8, 13, 18, 23, 28, 33]
CorrectCols = [9, 14, 19, 24, 29, 34]

def visualize_spike_correspondence(summaryFile):
    summaryData = np.loadtxt(summaryFile, skiprows=1, unpack=True)
    refSpikeNr = summaryData[refSpikeCol][0]

    figureCount = 0
    # overview: how many do/do not have any matching cluster?
    plt.figure(figureCount)
    figureCount += 1
    for i, col in enumerate(SNRCols):
        noClusterFound = len(np.where(summaryData[clusterNrCols[i]] == 0)[0])
        refNr = len(summaryData[clusterNrCols[i]])
        plt.plot(summaryData[col][0], 1.0 - 1.0*noClusterFound/refNr, 'ko')
    plt.ylim([0, 1])
    plt.xlabel('SNR')
    plt.ylabel('Fraction of parameter sets with matching cluster')

    # # how many spikes can we recover on avg?
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     plt.semilogy(summaryData[col][sel], summaryData[CorrectCols[i]][sel]/refSpikeNr, 'ko')
    # plt.xlabel('SNR')
    # plt.ylabel('Recovered spikes')
    #
    # # what are our error rates?
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[FPCols[i]] > -1.0)
    #     if not i:
    #         plt.semilogy(summaryData[col][sel], summaryData[FPCols[i]][sel]/refSpikeNr, 'bo', label='FP')
    #         plt.semilogy(summaryData[col][sel], summaryData[FNCols[i]][sel]/refSpikeNr, 'ro', label='FN')
    #     else:
    #         plt.semilogy(summaryData[col][sel], summaryData[FPCols[i]][sel]/refSpikeNr, 'bo')
    #         plt.semilogy(summaryData[col][sel], summaryData[FNCols[i]][sel]/refSpikeNr, 'ro')
    # plt.legend()
    # plt.xlabel('SNR')
    # plt.ylabel('FP/FN rates')
    #
    # # let's look at this as a function of different threshold/variance parameters
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.plot(summaryData[thresh1Col][sel], summaryData[CorrectCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Th1')
    # plt.ylabel('Recovered spikes')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.plot(summaryData[thresh2Col][sel], summaryData[CorrectCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Th2/3')
    # plt.ylabel('Recovered spikes')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.plot(summaryData[var1Col][sel], summaryData[CorrectCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Var1')
    # plt.ylabel('Recovered spikes')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.plot(summaryData[var2Col][sel], summaryData[CorrectCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Var2/3')
    # plt.ylabel('Recovered spikes')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     ax1 = plt.subplot(1,2,1)
    #     ax1.semilogy(summaryData[thresh1Col][sel], summaryData[FPCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    #     ax2 = plt.subplot(1,2,2)
    #     ax2.semilogy(summaryData[thresh1Col][sel], summaryData[FNCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Th1')
    # plt.xlabel('Th1')
    # plt.ylabel('FP/FN')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.subplot(1,2,1)
    #     plt.semilogy(summaryData[thresh2Col][sel], summaryData[FPCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    #     plt.subplot(1,2,2)
    #     plt.semilogy(summaryData[thresh2Col][sel], summaryData[FNCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Th2/3')
    # plt.ylabel('FP/FN')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.subplot(1,2,1)
    #     plt.semilogy(summaryData[var1Col][sel], summaryData[FPCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    #     plt.subplot(1,2,2)
    #     plt.semilogy(summaryData[var1Col][sel], summaryData[FNCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Var1')
    # plt.ylabel('FP/FN')
    #
    # plt.figure(figureCount)
    # figureCount += 1
    # for i, col in enumerate(SNRCols):
    #     sel = np.where(summaryData[CorrectCols[i]] > -1.0)
    #     labelStr = 'SNR %.1f' % summaryData[col][0]
    #     plt.subplot(1,2,1)
    #     plt.semilogy(summaryData[var2Col][sel], summaryData[FPCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    #     plt.subplot(1,2,2)
    #     plt.semilogy(summaryData[var2Col][sel], summaryData[FNCols[i]][sel]/refSpikeNr, 'o', label=labelStr)
    # plt.legend()
    # plt.xlabel('Var2/3')
    # plt.ylabel('FP/FN')

    plt.show()

def visualize_IPSC_correspondence(summaryFile, experimentInfoName, basePath):
    with open(experimentInfoName, 'r') as dataFile:
        experimentInfo = ast.literal_eval(dataFile.read())    # constants for calculating spike-triggered stuff

    alignedWindow = np.array((-5.0, 10.0))*pq.ms
    maxBegin = 0.3*pq.ms
    maxWindow = np.array((1.2, 2.54))*pq.ms
    alignedWindowSamples_ = (alignedWindow*experimentInfo['WC']['SamplingRate']*pq.Hz).simplified
    maxBeginSamples_ = (maxBegin*experimentInfo['WC']['SamplingRate']*pq.Hz).simplified - alignedWindowSamples_[0]
    maxWindowSamples_ = (maxWindow*experimentInfo['WC']['SamplingRate']*pq.Hz).simplified - alignedWindowSamples_[0]
    maxBeginSamples = int(maxBeginSamples_ + 0.5)
    maxWindowSamples = (int(maxWindowSamples_[0] + 0.5), int(maxWindowSamples_[1] + 0.5))

    summaryData = np.loadtxt(summaryFile, skiprows=1, unpack=True)
    paramRowLUT = {}
    rowParamLUT = {}
    for rowNr in range(len(summaryData[0])):
        rowStr = 'Th_%d_%d_%d_Lam_%d_%d_%d' % (summaryData[thresh1Col][rowNr], summaryData[thresh2Col][rowNr], summaryData[thresh2Col][rowNr],
                                               summaryData[var1Col][rowNr], summaryData[var2Col][rowNr], summaryData[var2Col][rowNr])
        # rowStr = 'Th_'
        # rowStr += str(summaryData[thresh1Col][rowNr])
        # rowStr += '_'
        # rowStr += str(summaryData[thresh2Col][rowNr])
        # rowStr += '_'
        # rowStr += str(summaryData[thresh2Col][rowNr])
        # rowStr += '_Lam_'
        # rowStr += str(summaryData[var1Col][rowNr])
        # rowStr += '_'
        # rowStr += str(summaryData[var2Col][rowNr])
        # rowStr += '_'
        # rowStr += str(summaryData[var2Col][rowNr])
        paramRowLUT[rowStr] = rowNr
        rowParamLUT[rowNr] = rowStr

    refAmplitudesName = os.path.join(experimentInfo['STA']['DataBasePathI'], 'STSnippets_Cluster_979_amplitudes.npy')
    refAmplitudesShuffledName = os.path.join(experimentInfo['STA']['DataBasePathI'], 'STSnippets_shuffled_10x_Cluster_979_amplitudes.npy')
    refAmplitudes = np.load(refAmplitudesName)
    refAmplitudesShuffled = np.load(refAmplitudesShuffledName)
    histRange = -50.0, 150.0
    binSize = 2.0
    bins = np.arange(histRange[0], histRange[1], binSize)
    refHist, _ = np.histogram(refAmplitudes, bins)
    refNr = np.sum(refHist)
    refHist = 1.0/refNr*refHist
    refShuffledHist, _ = np.histogram(refAmplitudesShuffled, bins)
    refShuffledNr = np.sum(refShuffledHist)
    refShuffledHist = 1.0/refShuffledNr*refShuffledHist

    # automated analysis of all parameter sets
    figureCount = 0
    paramAmplitudeErrors = {}
    paramAmplitudeHeaders = ['SNR', 'Avg. IPSC amp.', 'Nr spikes used', 'RMSE reference', 'RMSE shuffled']
    for rowNr in range(len(summaryData[0])):
        paramStr = rowParamLUT[rowNr]
        paramAmplitudeErrors[paramStr] = []
        print 'Computing IPSC amplitude differences for parameter set %s' % paramStr

        # plot avg. IPSC as function of SNR
        # optParams = 'Th_4_15_15_Lam_20_50_50'
        scales = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        # scales = [1.0, 0.8, 0.6, 0.4, 0.2]
        # basePath = 'Z:\\Robert\\INT_connectivity\\SiProbe\\ProbeBird_101917\\SiProbe\\Continuous_400_cut\\KS_parameter_search\\'
        baseNames = [os.path.join(basePath, 'scale_%.02f', 'refCluster_979') % scale for scale in scales]
        plt.figure(figureCount)
        figureCount += 1
        cmap = plt.get_cmap('gist_rainbow')
        nrOfColors = len(scales)
        for i, baseName in enumerate(baseNames):
            if not summaryData[clusterNrCols[i], rowNr]:
                paramAmplitudeErrors[paramStr].append(summaryData[SNRCols[i]][0])
                paramAmplitudeErrors[paramStr].append(0)
                paramAmplitudeErrors[paramStr].append(0)
                paramAmplitudeErrors[paramStr].append(-1)
                paramAmplitudeErrors[paramStr].append(-1)
                continue
            STAName = paramStr + '_STA.npy'
            STASEName = paramStr + '_STA_SE.npy'
            STA = np.load(os.path.join(baseName, STAName)).flatten()
            STA_SE = np.load(os.path.join(baseName, STASEName)).flatten()
            timeAxis = np.array(range(len(STA)))/50000.0*1000.0 - 5.0
            labelStr = 'SNR %.1f' % summaryData[SNRCols[i]][0]
            plt.plot(timeAxis, STA, label=labelStr, color=cmap(1.0*i/nrOfColors))
            plt.plot(timeAxis, STA + 2*STA_SE, color=cmap(1.0*i/nrOfColors))
            plt.plot(timeAxis, STA - 2*STA_SE, color=cmap(1.0*i/nrOfColors))

            # compute nr. of used spikes for IPSC and RMSE with reference/shuffled amp. distributions
            STAmplitudesName = paramStr + '_STAmplitudes.npy'
            STAmplitudes = np.load(os.path.join(baseName, STAmplitudesName))
            if len(STAmplitudes):
                avgAmp = np.max(STA[maxWindowSamples[0]:maxWindowSamples[1]]) - STA[maxBeginSamples]
            else:
                avgAmp = 0.0
            hist_, _ = np.histogram(STAmplitudes, bins)
            histNr = np.sum(hist_)
            if histNr:
                hist = hist_*1.0/histNr

            diffRef = refHist - hist
            RMSERef = np.sqrt(np.dot(diffRef, diffRef))
            diffShuffled = refShuffledHist - hist
            RMSEShuffled = np.sqrt(np.dot(diffShuffled, diffShuffled))
            paramAmplitudeErrors[paramStr].append(summaryData[SNRCols[i]][0])
            paramAmplitudeErrors[paramStr].append(avgAmp)
            paramAmplitudeErrors[paramStr].append(histNr)
            paramAmplitudeErrors[paramStr].append(RMSERef)
            paramAmplitudeErrors[paramStr].append(RMSEShuffled)

        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (pA)')
        IPSCName = os.path.join(basePath, paramStr + '_IPSCs.pdf')
        plt.savefig(IPSCName)

    IPSCAmpSummaryName = os.path.join(basePath, 'IPSC_amplitude_comparison.csv')
    with open(IPSCAmpSummaryName, 'w') as IPSCFile:
        header = 'Th1\tTh2/3\tLam1\tLam2/3'
        for i in range(len(SNRCols)):
            for j in range(len(paramAmplitudeHeaders)):
                header += '\t'
                header += paramAmplitudeHeaders[j]
        header += '\n'
        IPSCFile.write(header)
        for paramStr in paramAmplitudeErrors:
            paramSplitStr = paramStr.split('_')
            line = paramSplitStr[1]
            line += '\t'
            line += paramSplitStr[2]
            line += '\t'
            line += paramSplitStr[5]
            line += '\t'
            line += paramSplitStr[6]
            for i in range(len(paramAmplitudeErrors[paramStr])):
                line += '\t'
                line += str(paramAmplitudeErrors[paramStr][i])
            line += '\n'
            IPSCFile.write(line)

        # for i, baseName in enumerate(baseNames):
        #     plt.figure(figureCount)
        #     figureCount += 1
        #     STAmplitudesName = paramStr + '_STAmplitudes.npy'
        #     STAmplitudes = np.load(os.path.join(baseName, STAmplitudesName))
        #     hist_, _ = np.histogram(STAmplitudes, bins)
        #     histNr = np.sum(hist_)
        #     hist = hist_*1.0/histNr
        #     # matplotlib bug: edgecolor has to be set for each bar individually...
        #     plt.bar(bins[:-1], refShuffledHist, label='Shuffled', width=binSize, facecolor='none',
        #             edgecolor=['k']*len(hist), linewidth=0.5, linestyle='-')
        #     refLabelStr = 'Ref (%d)' % refNr
        #     plt.bar(bins[:-1], refHist, label=refLabelStr, width=binSize, facecolor='none',
        #             edgecolor=['b']*len(hist), linewidth=0.5, linestyle='-')
        #     labelStr = 'SNR %.1f (%d)' % (summaryData[SNRCols[i]][0], histNr)
        #     plt.bar(bins[:-1], hist, label=labelStr, width=binSize, facecolor='none',
        #             edgecolor=['r']*len(hist), linewidth=0.5, linestyle='-')
        #     # plt.bar(bins[:-1], hist, label=str(summaryData[SNRCols[i]][0]), width=binSize, facecolor='none',
        #             # edgecolor=[cmap(1.0*i/nrOfColors)]*len(hist), lw=0.5, ls='-')
        #     # plt.bar(bins[:-1], hist, label=str(summaryData[SNRCols[i]][0]), width=binSize, facecolor=cmap(1.0*i/nrOfColors), alpha=0.5)
        #
        #     diffRef = refHist - hist
        #     RMSERef = np.sqrt(np.dot(diffRef, diffRef))
        #     diffShuffled = refShuffledHist - hist
        #     RMSEShuffled = np.sqrt(np.dot(diffShuffled, diffShuffled))
        #     titleStr = 'RMSE ref = %.2f -- RMSE shuffled = %.2f' % (RMSERef, RMSEShuffled)
        #     plt.title(titleStr)
        #     plt.legend()
        #     plt.xlabel('Current amplitude (pA)')
        #     plt.ylabel('Rel. frequency')
        #
        # plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        summaryName = sys.argv[1]
        visualize_spike_correspondence(summaryName)
    elif len(sys.argv) == 4:
        summaryFile = sys.argv[1]
        experimentInfo = sys.argv[2]
        basePath = sys.argv[3]
        # optParams = sys.argv[3]
        # basePath = sys.argv[4]
        # visualize_IPSC_correspondence(summaryFile, experimentInfo, optParams, basePath)
        visualize_IPSC_correspondence(summaryFile, experimentInfo, basePath)
