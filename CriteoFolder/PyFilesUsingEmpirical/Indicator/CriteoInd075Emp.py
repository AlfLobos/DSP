if __name__ == '__main__':
    import numpy as np
    import time
    import pickle
    import os
    import csv
    from gurobipy import *
    from Utilities import CreateGraph, CreateTable_L2_L2Ind_Gr, createTableGreedy, CreateMatEdgesAndCTR,\
        CreateCtrVectorFromMatrices, CreateBetaParams, ForBetaDist, NumericSecPrice, createTableGreedy
    from SimulationCode import RunOnlyInd

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
    vector_mOrigTest=vector_mTest[:]

    ## More common info.
    init_lam=np.zeros(numCampaigns)

    ## Particular Data For Ind
    ext_s = vector_sOrigTest[index_Imps]

    vector_ctrTrain = CreateCtrVectorFromMatrices(numCampaigns, num_impressions, \
        num_edges, edgesMatrix, ctrTrain)

    vector_r = vector_r * 0.75

    vector_rctr = vector_r * vector_ctrTrain

    ## More common info.
    init_lam = np.zeros(numCampaigns)
    sim = 100
    np.random.seed(98765)
    seeds = np.random.randint(low = 1, high = 100000, size = sim)

    vector_s = vector_sOrigTest[:]
    ## Number of subgradient iterations to run 
    ## for the indicator utility function.



    num_it = 50000

    initTime = time.time()

    alphasInd = np.array([np.sqrt(1.0/(i + 1)) for i in range(num_it)]) * 0.0001

    listInd = RunOnlyInd(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_r, vector_mOrigTest, vector_sOrigTest, \
    vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, clusterIds, alphasInd, \
    num_it, 0, init_lam, sim, listCampPerImp, seeds)

    
    suffix = 'Results/'
    current_directory = os.getcwd()
    results_directory = os.path.join(current_directory, suffix)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)


    tableInd = createTableGreedy(listInd, vector_mOrigTest, sim)

    with open(suffix + 'CriteoInd075Emp.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sim','Ind-Profit', 'Ind-Revenue', 'Ind-Cost', 'Ind-BidsMade',\
            'Ind-BidsWon', 'Ind-ClicksWon', 'Ind-%BudgetUsed'])
        [writer.writerow(r) for r in tableInd]

    print('The whole code took '+str(time.time()- initTime) +' seconds to run')
