## Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math
from bisect import bisect_left 

#
#
## Functions that Read Data. In the Ipinyou case this data was created in DataExplorationSeason3
#
#

def CreateMatEdgesAndCTR(ctrMat, impWonLogMat, minImpressions):
    '''
    This function receives a CTR matrix of shape # impression x # campaign and
    returns a modified CTR matrix of the same shape and an incidence matrix. 
    The return matrix differs from the original only in pairs (impression, campaign) 
    in which the number of logs in the impression log is less than minImpressions.
    input:
        (1) original CTR matrix
        (2) matrix of shape # impression x # campaign with the number of times
        each pair appeared in the impression logs.
        (3) minimum number of times we require a pair (impression, campaign)
        to appear in the impression logs to consider its empirical average ctr
        as statistically significant.
    output:
        (1) binary matrix of shape # impression x # campaign with value of 1 
        for a pair (impression, campaign) if the pair has more impressions
        than minImpressions.
        (2) modified CTR matrix.
        (3) Refined impression matrix.
    '''
    edgesMatrix = np.array(impWonLogMat >= minImpressions,  dtype=int)
    ctrRefined = ctrMat * edgesMatrix
    impWonRefined = impWonLogMat * edgesMatrix
    return [edgesMatrix, ctrRefined, impWonRefined]

def CreateCtrVectorFromMatrices(numCampaigns, num_impressions, num_edges, matEdges, ctrMat):
    vector_ctr = np.zeros(num_edges)
    aux = 0
    for j in range(numCampaigns):
        for i in np.arange(num_impressions)[(matEdges[:, j]>0)]:
            vector_ctr[aux] = ctrMat[i, j]
            aux += 1
    return vector_ctr

def CreateGraph(edgesMatrix, ctrMat):
    '''
    This function creates some neccesary information, which we had in the DataExplorationSeason3 Jupyter Notebook, 
    and that is needed to run our optimization algorithm. 

    The info to be created is:
    input:
        (1) binary matrix of shape # impression x # campaign with value of 1 
        for a pair (impression, campaign) if the pair has more impressions
        than minImpressions.
        (2) modified CTR matrix.

    output:
        (1) num_edges (number of edges)
        (2) vector_ctr = The expected revenue per edge ($r_{ik}$)
        (3) index_Imps = Impression type associated to each edge.
        (4) index_sizeCamps = Number of impression types associated to each campaign. 
        (5) index_startCamp =  Position in which the first edge associated to a given campaign appears in the edges array. 
    '''
    num_impressions, numCampaigns = np.shape(edgesMatrix)
    num_edges = int(np.sum(edgesMatrix))
    index_sizeCamps = np.zeros(numCampaigns)
    index_startCamp = np.zeros(numCampaigns)
    index_Imps = np.zeros(num_edges)
    aux = 0 # General count index
    aux_startCamp = 0
    vector_ctr = CreateCtrVectorFromMatrices(numCampaigns, num_impressions, num_edges, edgesMatrix, ctrMat)
    for j in range(numCampaigns):
        size_camp = int(np.sum(edgesMatrix[:, j]))
        index_startCamp[j] = aux_startCamp
        index_sizeCamps[j] = size_camp
        for i in np.arange(num_impressions)[(edgesMatrix[:, j]>0)]:
            index_Imps[aux] = i
            aux += 1
        aux_startCamp += size_camp
    return [num_edges, index_Imps.astype(int), index_sizeCamps.astype(int), \
        index_startCamp.astype(int), vector_ctr]



# The functions NumericSecPrice, ForBetaDist, CalcRhoAndBetaVectors and BetaValue are used to calculate the 
# $\beta_i(\cdot)$ and $\rho_i(\cdot)$ functions.

# For $\rho_i(\cdot)$: We just need to calculate the c.d.f of the correct $\beta$ distribution at the given bid value. 

# For $\beta_i(\cdot)$ (much harder than the previous): If we call $C$ the associated $\beta$ distribution for the 
# impression type (which represents the distribution of the highest competing bid) we have $\beta_i(b) = \mathbb{E}[C|C<b]$ 
# (Ipinyou assumes that second price auctions are used). To approximate this conditional expectation wefirst made a partition
# of the [0, 1.0] set in intervals of large 0.01 (this is a parameter that can be changed). Then, 
# for each 0.01 * j with $j \in \{ 0, 1, \dots, 99 \}$ we sample 10, 000 i.i.d. uniform(0, 1) random variables (another parameter), 
# multiply each sample by 0.01 * j and then calculate the inverse of the cdf function ( .ppf(\cdot) in scipy). With this we 
# have approximate values for $\beta_i(\cdot)$ in function of the $\cdf$ distribution for all the 0.01 ''breakpoints'' from
# 0.0 to 1.0. Then, to calculate $\beta_i(b)$ for a given bid value $b$ we first calculate $\rho_i(b)$. Then, 
# find a $j \in \{ 0, 1, \dots, 99 \}$, such that $\rho_i(b) \ in  0.01*[j, j+1]$ or if $\rho_i(b)\ge 0.99$ choose $j = 99$. 
# Finally, we interpolate the approximate inverse cdf values found before to have an estimate of the $\beta_i(b)$ value.

def CreatePPFTable(distMP, tunning = 1000):
    allPPFs = []
    step = 1.0/tunning
    for dist in distMP:
        auxList = []
        for i in range(tunning):
            auxList.append(dist.ppf(step*i))
        allPPFs.append(auxList)
    return allPPFs

def NumericSecPrice(distMP, numUnifSamples):
    numSecPrice = np.zeros((len(distMP), 100))
    samples = np.random.uniform(0, 1.0, numUnifSamples)
    for j in range(100):
        samplesToUse = samples*0.01*(j+1.0)
        for i in range(len(distMP)):
            values = [ distMP[i].ppf(x) for x in samplesToUse]
            numSecPrice[i, j] = np.average(values)
        if j % 10 == 0:
            print("In numericSecPrice we have "+str(j)+" centiles ready")
    return numSecPrice

def ForBetaDist(betaParams, num_impressions):
    distToRet = []
    for i in range(num_impressions):
        a = betaParams[i, 0]
        b = betaParams[i, 1]
        loc = betaParams[i, 2]
        scale = betaParams[i, 3]
        distToRet.append(scipy.stats.beta(a = a, b = b, loc = loc, scale = scale))
    return distToRet

def ForLogNormalDist(logNormalParams, num_impressions):
    distToRet = []
    for i in range(num_impressions):
        s = logNormalParams[i, 0]
        loc =logNormalParams[i, 1]
        scale= logNormalParams[i, 2]
        distToRet.append(scipy.stats.lognorm(s = s, loc = loc, scale = scale))
    return distToRet

# def CalcRhoAndBetaVectors(bid_vec, num_edges, index_Imps, distMP, numericBeta) :
#     ## I will assume I want to evaluate the full vector.
#     rhoBetaMat = np.zeros((num_edges, 2))
#     for i in range(len(index_Imps)):
#         rhoBetaMat[i, :] = RhoBetaValue(bid_vec[i], index_Imps[i], distMP, numericBeta)
#     return [rhoBetaMat[:, 0], rhoBetaMat[:, 1]]

# def BetaValue(bid, impNumber, distMP, numericBeta):
#     aux = distMP[impNumber].cdf(bid)
#     rest = 0
#     intg = 0
#     if aux >=  0.0:
#         [rest, intg] = math.modf(aux*100)
#         intg = int(intg)
#     if intg>0 and intg<100:
#         return numericBeta[impNumber, intg-1]*(1.0-rest)+numericBeta[impNumber, intg]*rest
#     elif intg >= 100:
#         # Is equal to the expectation in this case, as the bid was high enough
#         # that it would always have been won.
#         return numericBeta[impNumber, 99]
#     else:
#         # This case is e.g. cdf(bid) = 0.004, which then is 0*0.6+ numericBeta[impNumber, 0]*0.4
#         return 0.0*(1.0-rest)+numericBeta[impNumber, 0]*rest

# def RhoBetaValue(bid, impNumber, distMP, numericBeta):
#     ## For rho_beta_Type = 0, args[0] = adv
#     rho = distMP[impNumber].cdf(bid)
#     beta = BetaValue(bid, impNumber, distMP, numericBeta)
#     return [rho, beta]


def CalcRhoAndBetaVectors(bid_vec, num_edges, index_Imps, PPFTable, numericBeta):

    ## I will assume I want to evaluate the full vector.
    lenPPF = len(PPFTable[0])
    rhoBetaMat = np.zeros((num_edges, 2))
    for i in range(len(index_Imps)):
        rhoBetaMat[i, :] = RhoBetaValue(bid_vec[i], index_Imps[i], PPFTable, numericBeta, lenPPF)
    return [rhoBetaMat[:, 0], rhoBetaMat[:, 1]]

def BetaValue(bid, impNumber, cdf, numericBeta):
    rest = 0
    intg = 0
    if cdf >=  0.0:
        [rest, intg] = math.modf(cdf*100)
        intg = int(intg)
    if intg>0 and intg<100:
        return numericBeta[impNumber, intg-1]*(1.0-rest)+numericBeta[impNumber, intg]*rest
    elif intg >= 100:
        # Is equal to the expectation in this case, as the bid was high enough
        # that it would always have been won.
        return numericBeta[impNumber, 99]
    else:
        # This case is e.g. cdf(bid) = 0.004, which then is 0*0.6+ numericBeta[impNumber, 0]*0.4
        return 0.0*(1.0-rest)+numericBeta[impNumber, 0]*rest

def RhoBetaValue(bid, impNumber, PPFTable, numericBeta, lenPPF):
    ## For rho_beta_Type = 0, args[0] = adv
    rho = bisect_left(PPFTable[impNumber], bid, lo = 0, hi = lenPPF) * (1.0/lenPPF)
    # rho = distMP[impNumber].cdf(bid)
    beta = BetaValue(bid, impNumber, rho, numericBeta)
    return [rho, beta]


#
#
## Create The Beta Params

def CreateBetaParams(pathMPLog, fixNameCluster, impNames, numCampaigns, numToAvg):
    betaParams = np.zeros((len(impNames), numCampaigns))
    for i in range(len(impNames)):
        alist = [line.rstrip() for line in open(pathMPLog+fixNameCluster+impNames[i])]
        numbers = [ float(x) for x in alist ]
        if len(numbers)>100000:
            betaParams[i, :] = scipy.stats.beta.fit(np.random.choice(numbers,  numToAvg,  replace = False), floc = 0.0)
        else:
            betaParams[i, :] = scipy.stats.beta.fit(numbers, floc = 0.0)                       
    return betaParams

## Table of Results for Long Run Ipinyou

# The following function saves the results of all simulations for the three methods. In order it first saves the Percentage 
# used and simulation number, and then saves in order for each method: 1.- Total Profit Obtained, 2.- Total Revenue, 
# 3.- Total Cost, 4.- Total Bids Made, 5.- Total Clicks Made, 6.- Average Budget Utilization

def CreateTableIndL2IndGr(dictToRetInd, dictToRetL2Ind, dictToRetGr, vector_mOrig, perVector_m, sim):
    tableToCsv = np.zeros((sim*len(perVector_m), 23))
    for z in range(len(perVector_m)):
        perc = perVector_m[z]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = perc
            tableToCsv[i+z*sim, 1] = i
            tableToCsv[i+z*sim, 2] = np.sum(dictToRetInd[perc][i][6])
            tableToCsv[i+z*sim, 3] = np.sum(dictToRetInd[perc][i][5])
            tableToCsv[i+z*sim, 4] = np.sum(dictToRetInd[perc][i][4])
            tableToCsv[i+z*sim, 5] = np.sum(dictToRetInd[perc][i][1])
            tableToCsv[i+z*sim, 6] = np.sum(dictToRetInd[perc][i][2])
            tableToCsv[i+z*sim, 7] = np.sum(dictToRetInd[perc][i][3])
            tableToCsv[i+z*sim, 8] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetInd[perc][i][0]))/(np.sum(vector_mOrig)*perc)
            tableToCsv[i+z*sim, 9] = np.sum(dictToRetL2Ind[perc][i][6])
            tableToCsv[i+z*sim, 10] = np.sum(dictToRetL2Ind[perc][i][5])
            tableToCsv[i+z*sim, 11] = np.sum(dictToRetL2Ind[perc][i][4])
            tableToCsv[i+z*sim, 12] = np.sum(dictToRetL2Ind[perc][i][1])
            tableToCsv[i+z*sim, 13] = np.sum(dictToRetL2Ind[perc][i][2])
            tableToCsv[i+z*sim, 14] = np.sum(dictToRetL2Ind[perc][i][3])
            tableToCsv[i+z*sim, 15] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetL2Ind[perc][i][0]))/(np.sum(vector_mOrig)*perc)
            tableToCsv[i+z*sim, 16] = np.sum(dictToRetGr[perc][i][6])
            tableToCsv[i+z*sim, 17] = np.sum(dictToRetGr[perc][i][5])
            tableToCsv[i+z*sim, 18] = np.sum(dictToRetGr[perc][i][4])
            tableToCsv[i+z*sim, 19] = np.sum(dictToRetGr[perc][i][1])
            tableToCsv[i+z*sim, 20] = np.sum(dictToRetGr[perc][i][2])
            tableToCsv[i+z*sim, 21] = np.sum(dictToRetGr[perc][i][3])
            tableToCsv[i+z*sim, 22] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetGr[perc][i][0]))/(np.sum(vector_mOrig)*perc)
    return tableToCsv

# The following function saves the results of all simulations for the three methods. In order it first saves the Percentage 
# used and simulation number, and then saves in order for each method: 1.- Total Profit Obtained, 2.- Total Revenue, 
# 3.- Total Cost, 4.- Total Bids Made, 5.- Total Clicks Made, 6.- Average Budget Utilization

def CreateTable_Ind_L2_L2Ind_Gr(dictToRetInd, dictToRetL2, dictToRetL2Ind, dictToRetGr, vector_mOrig, \
    perVector_m, sim):
    tableToCsv = np.zeros((sim*len(perVector_m), 30))
    for z in range(len(perVector_m)):
        perc = perVector_m[z]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = perc
            tableToCsv[i+z*sim, 1] = i
            tableToCsv[i+z*sim, 2] = np.sum(dictToRetInd[perc][i][6])
            tableToCsv[i+z*sim, 3] = np.sum(dictToRetInd[perc][i][5])
            tableToCsv[i+z*sim, 4] = np.sum(dictToRetInd[perc][i][4])
            tableToCsv[i+z*sim, 5] = np.sum(dictToRetInd[perc][i][1])
            tableToCsv[i+z*sim, 6] = np.sum(dictToRetInd[perc][i][2])
            tableToCsv[i+z*sim, 7] = np.sum(dictToRetInd[perc][i][3])
            tableToCsv[i+z*sim, 8] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetInd[perc][i][0]))/(np.sum(vector_mOrig)*perc)
            tableToCsv[i+z*sim, 9] = np.sum(dictToRetL2[perc][i][6])
            tableToCsv[i+z*sim, 10] = np.sum(dictToRetL2[perc][i][5])
            tableToCsv[i+z*sim, 11] = np.sum(dictToRetL2[perc][i][4])
            tableToCsv[i+z*sim, 12] = np.sum(dictToRetL2[perc][i][1])
            tableToCsv[i+z*sim, 13] = np.sum(dictToRetL2[perc][i][2])
            tableToCsv[i+z*sim, 14] = np.sum(dictToRetL2[perc][i][3])
            tableToCsv[i+z*sim, 15] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetL2[perc][i][0]))/(np.sum(vector_mOrig)*perc)
            tableToCsv[i+z*sim, 16] = np.sum(dictToRetL2Ind[perc][i][6])
            tableToCsv[i+z*sim, 17] = np.sum(dictToRetL2Ind[perc][i][5])
            tableToCsv[i+z*sim, 18] = np.sum(dictToRetL2Ind[perc][i][4])
            tableToCsv[i+z*sim, 19] = np.sum(dictToRetL2Ind[perc][i][1])
            tableToCsv[i+z*sim, 20] = np.sum(dictToRetL2Ind[perc][i][2])
            tableToCsv[i+z*sim, 21] = np.sum(dictToRetL2Ind[perc][i][3])
            tableToCsv[i+z*sim, 22] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetL2Ind[perc][i][0]))/(np.sum(vector_mOrig)*perc)
            tableToCsv[i+z*sim, 23] = np.sum(dictToRetGr[perc][i][6])
            tableToCsv[i+z*sim, 24] = np.sum(dictToRetGr[perc][i][5])
            tableToCsv[i+z*sim, 25] = np.sum(dictToRetGr[perc][i][4])
            tableToCsv[i+z*sim, 26] = np.sum(dictToRetGr[perc][i][1])
            tableToCsv[i+z*sim, 27] = np.sum(dictToRetGr[perc][i][2])
            tableToCsv[i+z*sim, 28] = np.sum(dictToRetGr[perc][i][3])
            tableToCsv[i+z*sim, 29] = (np.sum(vector_mOrig)*perc-np.sum(dictToRetGr[perc][i][0]))/(np.sum(vector_mOrig)*perc)
    return tableToCsv

## Table of Results for the Pareto Experiment

def createTableGreedy(listResultsGreedy, vector_mOrig, sim):
    tableToCsv = np.zeros((sim, 8))
    for i in range(sim):
        tableToCsv[i, 0] = i
        tableToCsv[i, 1] = np.sum(listResultsGreedy[i][6])
        tableToCsv[i, 2] = np.sum(listResultsGreedy[i][5])
        tableToCsv[i, 3] = np.sum(listResultsGreedy[i][4])
        tableToCsv[i, 4] = np.sum(listResultsGreedy[i][1])
        tableToCsv[i, 5] = np.sum(listResultsGreedy[i][2])
        tableToCsv[i, 6] = np.sum(listResultsGreedy[i][3])
        tableToCsv[i, 7] = (np.sum(vector_mOrig)-np.sum(listResultsGreedy[i][0]))/(np.sum(vector_mOrig))
    return tableToCsv

def createTableL2Pareto(dictToRetL2Ind, vector_mOrig, perMult, sim):
    tableToCsv = np.zeros((sim*len(perMult), 9))
    for z in range(len(perMult)):
        perc = perMult[z]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = perc
            tableToCsv[i+z*sim, 1] = i
            tableToCsv[i+z*sim, 2] = np.sum(dictToRetL2Ind[perc][i][6])
            tableToCsv[i+z*sim, 3] = np.sum(dictToRetL2Ind[perc][i][5])
            tableToCsv[i+z*sim, 4] = np.sum(dictToRetL2Ind[perc][i][4])
            tableToCsv[i+z*sim, 5] = np.sum(dictToRetL2Ind[perc][i][1])
            tableToCsv[i+z*sim, 6] = np.sum(dictToRetL2Ind[perc][i][2])
            tableToCsv[i+z*sim, 7] = np.sum(dictToRetL2Ind[perc][i][3])
            tableToCsv[i+z*sim, 8] = (np.sum(vector_mOrig)-np.sum(dictToRetL2Ind[perc][i][0]))/(np.sum(vector_mOrig))
    return tableToCsv

def CreateTable_L2_L2Ind_Gr(dictToRetL2, dictToRetL2Ind, dictToRetGr, vector_mOrig, \
    multL2, multGr, sim):
    tableToCsv = np.zeros((sim*len(multL2), 24))
    for z in range(len(multL2)):
        l2Mult = multL2[z]
        grMult = multGr[z]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = l2Mult
            tableToCsv[i+z*sim, 1] = grMult
            tableToCsv[i+z*sim, 2] = i
            tableToCsv[i+z*sim, 3] = np.sum(dictToRetL2[l2Mult][i][6])
            tableToCsv[i+z*sim, 4] = np.sum(dictToRetL2[l2Mult][i][5])
            tableToCsv[i+z*sim, 5] = np.sum(dictToRetL2[l2Mult][i][4])
            tableToCsv[i+z*sim, 6] = np.sum(dictToRetL2[l2Mult][i][1])
            tableToCsv[i+z*sim, 7] = np.sum(dictToRetL2[l2Mult][i][2])
            tableToCsv[i+z*sim, 8] = np.sum(dictToRetL2[l2Mult][i][3])
            tableToCsv[i+z*sim, 9] = (np.sum(vector_mOrig)-np.sum(dictToRetL2[l2Mult][i][0]))/(np.sum(vector_mOrig))
            tableToCsv[i+z*sim, 10] = np.sum(dictToRetL2Ind[l2Mult][i][6])
            tableToCsv[i+z*sim, 11] = np.sum(dictToRetL2Ind[l2Mult][i][5])
            tableToCsv[i+z*sim, 12] = np.sum(dictToRetL2Ind[l2Mult][i][4])
            tableToCsv[i+z*sim, 13] = np.sum(dictToRetL2Ind[l2Mult][i][1])
            tableToCsv[i+z*sim, 14] = np.sum(dictToRetL2Ind[l2Mult][i][2])
            tableToCsv[i+z*sim, 15] = np.sum(dictToRetL2Ind[l2Mult][i][3])
            tableToCsv[i+z*sim, 16] = (np.sum(vector_mOrig)-np.sum(dictToRetL2Ind[l2Mult][i][0]))/(np.sum(vector_mOrig))
            tableToCsv[i+z*sim, 17] = np.sum(dictToRetGr[grMult][i][6])
            tableToCsv[i+z*sim, 18] = np.sum(dictToRetGr[grMult][i][5])
            tableToCsv[i+z*sim, 19] = np.sum(dictToRetGr[grMult][i][4])
            tableToCsv[i+z*sim, 20] = np.sum(dictToRetGr[grMult][i][1])
            tableToCsv[i+z*sim, 21] = np.sum(dictToRetGr[grMult][i][2])
            tableToCsv[i+z*sim, 22] = np.sum(dictToRetGr[grMult][i][3])
            tableToCsv[i+z*sim, 23] = (np.sum(vector_mOrig)-np.sum(dictToRetGr[grMult][i][0]))/(np.sum(vector_mOrig))
    return tableToCsv