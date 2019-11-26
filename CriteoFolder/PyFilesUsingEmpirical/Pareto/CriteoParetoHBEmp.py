#!/usr/bin/env python

def RunForOnePerc(tupleData):
    print('process id: '+str(os.getpid()))
    cL2, cL2Ind, tauMult, multGr = tupleData

    alphasL2Ind = np.array([cL2 * np.sqrt(1.0/(i + 1)) for i in range(num_it)])
    alphasL2Ind = np.array([cL2Ind * np.sqrt(1.0/(i + 1)) for i in range(num_it)])

    return [ExpPareto(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_r, vector_mOrigTest, vector_sOrigTest, \
    vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, clusterIds, alphasL2Ind, \
    num_it, 1, init_lam, [tauMult], sim, listCampPerImp, seeds, addIndicator = False),\
    ExpPareto(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_r, vector_mOrigTest, vector_sOrigTest, \
    vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, clusterIds, alphasL2Ind, \
    num_it, 2, init_lam, [tauMult], sim, listCampPerImp, seeds, addIndicator = True), \
    RunOnlyGreedySeveralSeeds(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_r, vector_mOrigTest, vector_ctrTrain,\
    vector_ctrTest, ImpInOrder, MPInOrder, clusterIds, listCampPerImp, seeds, multGr)]

if __name__ == '__main__':
    import numpy as np
    import time
    import pickle
    import os
    import csv
    from gurobipy import *
    from Utilities import CreateGraph, CreateTable_L2_L2Ind_Gr, createTableGreedy, CreateMatEdgesAndCTR,\
        CreateCtrVectorFromMatrices, CreateBetaParams, ForBetaDist, NumericSecPrice
    from SimulationCode import ExpPareto, RunOnlyGreedyOneSeed, RunOnlyGreedySeveralSeeds

    suffix='DataCriteo/'
    current_directory = os.getcwd()
    results_directory = os.path.join(current_directory, suffix)
    vector_m = pickle.load(open(results_directory+'vector_m'+'.p',"rb"))
    vector_sTest = pickle.load(open(results_directory+'vector_s'+'.p',"rb"))
    vector_r = pickle.load(open(results_directory+'vector_r'+'.p',"rb"))
    impWonTrainRefined = pickle.load(open(results_directory+'impWonTrainRefined'+'.p',"rb"))
    impWonTestRefined = pickle.load(open(results_directory+ 'impWonTestRefined'+'.p',"rb"))
    ctrTrain = pickle.load(open(results_directory+'ctrTrain'+'.p',"rb"))
    ctrTest = pickle.load(open(results_directory+'ctrTest'+'.p',"rb"))
    usefulCampIds = pickle.load(open(results_directory+'usefulCampIds'+'.p',"rb"))
    clusterIds = list(pickle.load(open(results_directory+'clusterIds'+'.p',"rb")))
    distMP = pickle.load(open(results_directory+'distMP'+'.p',"rb"))
    ImpInOrder = pickle.load(open(results_directory+'ImpInOrder'+'.p',"rb"))
    MPInOrder = pickle.load(open(results_directory+'MPInOrder'+'.p',"rb"))
    parametersBetaT = pickle.load(open(results_directory+'parametersBetaT'+'.p',"rb"))
    edgesMatrix = pickle.load(open(results_directory+'edgesMatrix'+'.p',"rb"))
    avgCTRPerCamp = pickle.load(open(results_directory+'avgCTRPerCamp'+'.p',"rb"))
    vector_mTest = pickle.load(open(results_directory+'vector_mTest'+'.p',"rb"))
    numCampaigns = pickle.load(open(results_directory+'numCampaigns'+'.p',"rb"))
    num_impressions = pickle.load(open(results_directory+'num_impressions'+'.p',"rb"))

    vector_ctrTrain = pickle.load(open(results_directory+'vector_ctrTrain'+'.p',"rb"))
    vector_ctrTest = pickle.load(open(results_directory+'vector_ctrTest'+'.p',"rb"))

    numericBeta = pickle.load(open(results_directory+'numSecPricEmp.p',"rb"))
    PPFTable = pickle.load(open(results_directory+'PPFListsEmp.p',"rb"))

    listCampPerImp = []
    for i in range(num_impressions):
        listCampPerImp.append(np.arange(numCampaigns)[edgesMatrix[i,:] == 1])


    [num_edges, index_Imps, index_sizeCamps, index_startCamp, vector_ctrTrain]=\
        CreateGraph(edgesMatrix, ctrTrain)

    ## How Ipinyou paid and bid in the test log guides vector_m and vector_s
    vector_sOrigTest=np.sum(impWonTestRefined, axis=1)
    vector_mOrigTest=vector_mTest[:] * 0.5

    ## More common info.
    init_lam=np.zeros(numCampaigns)

    ## Particular Data For Ind
    ext_s = vector_sOrigTest[index_Imps]

    vector_ctrTrain = CreateCtrVectorFromMatrices(numCampaigns, num_impressions, \
        num_edges, edgesMatrix, ctrTrain)

    # vector_r = np.repeat(vector_r, index_sizeCamps)
    vector_rctr = vector_r * vector_ctrTrain

    ## More common info.
    init_lam = np.zeros(numCampaigns)
    sim = 100
    np.random.seed(98765)
    seeds = np.random.randint(low = 1, high = 100000, size = sim)

    vector_s = vector_sOrigTest[:]
    ## Number of subgradient iterations to run 
    ## for the indicator utility function.

    cL2List = [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, \
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, \
        10, 10, 10, 10, 10, 10]

    cL2IndList = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, \
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, \
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, \
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    num_it = 50000

    grMult =  [0.5]
    for i in range(1,50):
        grMult.append(0.5 + 0.02*i)

    tauMult = [0.018, 0.020, 0.022, 0.025, 0.027, 0.030, 0.033, 0.037, 0.041, \
    0.045, 0.050, 0.055, 0.061, 0.067, 0.074, 0.082, 0.091, 0.100, 0.111, \
    0.122, 0.135, 0.150, 0.165, 0.183, 0.202, 0.223, 0.247, 0.273, 0.301, \
    0.333, 0.368, 0.407, 0.449, 0.497, 0.549, 0.607, 0.670, 0.741, 0.819, \
    0.905, 1.000, 1.105, 1.221, 1.350, 1.492, 1.649, 1.822, 2.014, 2.226, \
    2.460]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help = "Index of Percentage To Used")
    args = parser.parse_args()

    indexToUse = int(args.id)

    initTime = time.time()

    dictToRetL2, dictToRetL2Ind, dictToRetGr = RunForOnePerc([cL2List[indexToUse], \
        cL2IndList[indexToUse], tauMult[indexToUse], grMult[indexToUse]])

    tauMult = [tauMult[indexToUse]]
    multGr =  [grMult[indexToUse]]

    
    suffix = 'Results/'
    current_directory = os.getcwd()
    results_directory = os.path.join(current_directory, suffix)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)


    tablePareto = CreateTable_L2_L2Ind_Gr(dictToRetL2, dictToRetL2Ind, dictToRetGr, \
        vector_mOrigTest, tauMult, multGr, sim)

    with open(suffix + 'TableCriteoHBEmp_' + str(indexToUse) + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['multL2', 'multGr', 'sim',\
            'L2-Profit', 'L2-Revenue', 'L2-Cost', 'L2-BidsMade',\
            'L2-BidsWon', 'L2-ClicksWon', 'L2-%BudgetUsed',\
            'L2Ind-Profit', 'L2Ind-Revenue', 'L2Ind-Cost', 'L2Ind-BidsMade',\
            'L2Ind-BidsWon', 'L2Ind-ClicksWon', 'L2Ind-%BudgetUsed',\
            'Gr-Profit', 'Gr-Revenue', 'Gr-Cost', 'Gr-BidsMade',\
            'Gr-BidsWon', 'Gr-ClicksWon', 'Gr-%BudgetUsed'])
        [writer.writerow(r) for r in tablePareto]

    print('IpinyouPareto.py took '+str(time.time()- initTime) +' seconds to run')
