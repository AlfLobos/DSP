#!/usr/bin/env python

def RunForOneC(pair):

    c, mult = pair

    vector_m = vector_mOrigTest[:] 
    alphasInd = np.array([c*np.sqrt(1.0/(i + 1)) for i in range(num_itInd)])

    tau = np.power(vector_m, -1) * mult
    initTime = time.time()
    [_, _, _, _, dual_ValInd, primalValueInd, _, dual_varsInd] = \
        SubgrAlgSavPrimDualObjInd(init_lam[:], num_itInd, alphasInd[:], vector_r, vector_ctrTrain, \
        vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
        PPFTable, numericBeta, index_sizeCamps, index_Imps, num_itInd-1, p_grad_TypeInd)

    alphasL2 = np.array([c*np.sqrt(1.0/(i + 1)) for i in range(num_itL2Ind)])

    [_, _, _, _, dual_ValL2, primalValueL2, _, dual_varsL2] = \
        SubgrAlgSavPrimDualObjFn_L2Ind(init_lam[:], num_itL2Ind, alphasL2[:], vector_r, vector_ctrTrain, \
    vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
    PPFTable, numericBeta, index_sizeCamps, index_Imps, num_itL2Ind-1, p_grad_TypeL2, tau, False)

    alphasL2Ind = np.array([c*np.sqrt(1.0/(i + 1)) for i in range(num_itL2Ind)])

    #[dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
    #            primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]
    [_, _, _, _, dual_ValL2Ind, primalValueL2Ind, _, dual_varsL2Ind] = \
        SubgrAlgSavPrimDualObjFn_L2Ind(init_lam[:], num_itL2Ind, alphasL2Ind[:], vector_r, vector_ctrTrain, \
    vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
    PPFTable, numericBeta, index_sizeCamps, index_Imps, num_itL2Ind-1, p_grad_TypeL2Ind, tau, True)

    print('Running RunForOneC took: '+str(time.time()-initTime)+' seconds, process id: '+str(os.getpid()))
    return [pair[0], pair[1], dual_ValInd[0], primalValueInd[0], dual_varsInd[0][0], dual_varsInd[0][1], dual_varsInd[0][2], dual_varsInd[0][3], \
        dual_ValL2[0], primalValueL2[0], dual_varsL2[0][0], dual_varsL2[0][1], dual_varsL2[0][2], dual_varsL2[0][3], \
        dual_ValL2Ind[0], primalValueL2Ind[0], dual_varsL2Ind[0][0], dual_varsL2Ind[0][1], dual_varsL2Ind[0][2], dual_varsL2Ind[0][3]]




if __name__ == '__main__':

    import numpy as np
    import time
    import pickle
    import os
    import csv
    from gurobipy import *
    from Utilities import CreateGraph, CreateTableIndL2IndGr, CreateMatEdgesAndCTR,\
        CreateCtrVectorFromMatrices, CreateBetaParams, ForBetaDist, NumericSecPrice
    from UtilitiesOptimization import SubgrAlgSavPrimDualObjInd, SubgrAlgSavPrimDualObjFn_L2Ind

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help = "Index of Percentage To Used")
    args = parser.parse_args() 


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
    numericBeta = pickle.load(open(results_directory+'numericBeta'+'.p',"rb"))
    vector_ctrTrain = pickle.load(open(results_directory+'vector_ctrTrain'+'.p',"rb"))
    vector_ctrTest = pickle.load(open(results_directory+'vector_ctrTest'+'.p',"rb"))
    PPFTable = pickle.load(open(results_directory+'PPFTable.p',"rb"))

    listCampPerImp = []
    for i in range(num_impressions):
        listCampPerImp.append(np.arange(numCampaigns)[edgesMatrix[i,:] == 1])


    [num_edges, index_Imps, index_sizeCamps, index_startCamp, vector_ctrTrain]=\
        CreateGraph(edgesMatrix, ctrTrain)

    ## How Ipinyou paid and bid in the test log guides vector_m and vector_s
    vector_sOrigTest=np.sum(impWonTestRefined, axis=1)
    vector_mOrigTest=vector_mTest[:] * 0.75

    ## More common info.
    init_lam=np.zeros(numCampaigns)

    auxMult = [-4.0]
    for i in range(1,50):
        auxMult.append(-4.0 + 0.1*i)
    tauMult = np.round(np.exp(np.array(auxMult)), decimals = 3)

    ## Particular Data For Ind
    ext_s = vector_sOrigTest[index_Imps]

    vector_ctrTrain = CreateCtrVectorFromMatrices(numCampaigns, num_impressions, \
        num_edges, edgesMatrix, ctrTrain)

    # vector_r = np.repeat(vector_r, index_sizeCamps)
    vector_rctr = vector_r * vector_ctrTrain
    vector_s = vector_sOrigTest[:]
    ## Number of subgradient iterations to run 
    ## for the indicator utility function.

    num_itInd = 5000
    num_itL2Ind = 5000
    p_grad_TypeInd = 0
    p_grad_TypeL2 = 1
    p_grad_TypeL2Ind = 2

    cToTry = np.array([100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    # cToTry = [0.1, 0.001]
    np.random.seed(12345)

    pairs = []
    for mult in tauMult:
        for c in cToTry:
            pairs.append([c, mult])

    allData =  []

    indexToUse = int(args.id)
    allData.append(RunForOneC(pairs[indexToUse]))


    # # results = [pool.apply_async(RunForOnePerc, args = (perc,)) for perc in perVector_m]

    # results  =  pool.map(RunForOneC, pairs)
    # pool.close()
    # pool.join()
    # for res in results:
    #     asArray = np.squeeze(np.array(res))
    #     allData.append(asArray)

    
    suffix = 'Results/'
    current_directory = os.getcwd()
    results_directory = os.path.join(current_directory, suffix)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    with open(suffix+'TableValCriteoQB_'+str(indexToUse)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['c', 'Multiplier', 'DualInd', 'PrimalInd', 'Ind_Lam0', 'Ind_Lam1', 'Ind_Lam2', 'Ind_Lam3', \
            'DualL2', 'PrimalL2', 'L2_Lam0', 'L2_Lam1', 'L2_Lam2', 'L2_Lam3', \
            'DualL2Ind', 'PrimalL2Ind', 'L2Ind_Lam0', 'L2Ind_Lam1', 'L2Ind_Lam2', 'L2Ind_Lam3'])
        count = 0
        for row in allData:
            writer.writerow(row)
