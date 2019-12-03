import numpy as np
import sys
import time
from RhoAndBeta import CalcRhoAndBetaVectors
from UtilitiesOptimization import CalculateLPGurobi, CalculateQuadGurobi, \
    SubgrAlgSavPrimDualObjInd, SubgrAlgSavPrimDualObjFn_L2Ind, ExtendSizeCamps, OptimalBids, OptimalX

#
#
## Extra Function that was not needed for Ipinyou nor Criteo

# Create One Set of Simulated Impression Arrivals And Market Prices
def CreateImpAndMPInOrder(num_impressions, numCampaigns, vector_s, adverPerImp,\
    shuffle = True, numpy_seed = -1):
    if numpy_seed != -1:
        np.random.seed(numpy_seed)
    totalArrivals = np.sum(vector_s)
    impArrivals = 0
    if shuffle:
        impArrivals = np.repeat(np.arange(num_impressions), repeats = vector_s)
        np.random.shuffle(impArrivals)
    else:
        percVector = vector_s/totalArrivals
        impArrivals = np.random.choice(np.arange(num_impressions),size = int(totalArrivals), p = percVector)
    ## The following array is just to speed up the code of creating the maximum of uniforms
    auxUniforms = np.random.uniform(size = int(totalArrivals) * int(np.max(adverPerImp)))
    counter = 0
    mpArrivals = np.zeros(int(totalArrivals))
    for pos, impType in enumerate(impArrivals):
        maxUnif = auxUniforms[counter]
        counter  += 1
        if adverPerImp[impType]>1:
            for j in range(1,adverPerImp[impType]):
                if auxUniforms[counter] > maxUnif:
                    maxUnif = auxUniforms[counter]
                counter  += 1
        mpArrivals[pos] = maxUnif
    return impArrivals, mpArrivals

#
#
## Simulation Code

#  This function returns return two dictionaries. 'dictImpPerCamp' has as keys np.arange(numCampaigns)
#  and as values lists with the indexes of the impressions that can be used by each campaign. 'dictCampPerImp'
#  has as keys np.arange(num_impressions) and as values lists with the indexes of the campaigns that can be
#  served by each impression.
def CreateNiceDictionariesForImpAndCampPos(numCampaigns, num_impressions, num_edges, \
    index_Imps, index_sizeCamps):
    dictImpPerCamp = {}
    dictCampPerImp = {}
    count = 0
    ## The impressions that are interested in each camapign appear in order in index_Imps.
    for campNumber, impsInCamp in enumerate(index_sizeCamps):
        dictImpPerCamp[campNumber] = index_Imps[count:(count+impsInCamp)]
        count += impsInCamp

    for impType in range(num_impressions):
        dictCampPerImp[impType] = []

    for camp in range(numCampaigns):
        for impType in dictImpPerCamp[camp]:
            dictCampPerImp[impType].append(camp)

    return dictImpPerCamp, dictCampPerImp


## To make the simulation code faster and easier to read (in particular the greedy heuristic)
## we would like to change the vector of click-through rates 'vector_ctr', vector of 
## revenues 'vector_r', of revenue times the click trhough rate 'vector_qctr'
## into a matrix of size number of campaigns times number of impressions.
## That's done in the following function. We also return the optimal bids and ranking for 
## the greedy algorithm.
def CreateMatR_ctr_Rctr_AndRankings(numCampaigns, num_impressions, num_edges, \
            index_Imps, index_sizeCamps, vector_r, vector_ctr, \
            vector_qctr,  UB_bidShort, adverPerImp):
    _, dictCampPerImp = CreateNiceDictionariesForImpAndCampPos(numCampaigns,\
        num_impressions, num_edges, index_Imps, index_sizeCamps)

    mat_r_by_Imp = np.zeros((numCampaigns, num_impressions))
    mat_ctr = np.zeros((numCampaigns, num_impressions))
    mat_rctr_by_Imp = np.zeros((numCampaigns, num_impressions))
    ranking = {}
    optBidGreedySP = {}
    optBidGreedyFP = {}
    UB_bidLong = UB_bidShort[index_Imps]
    ## Rememeber that index_Imps is an array if size num_edges that has the impType for each edge.
    for impType in range(num_impressions):
        indexes = np.arange(num_edges)[(index_Imps == impType)]
        mat_r_by_Imp[dictCampPerImp[impType], impType] =  vector_r[indexes]
        mat_ctr[dictCampPerImp[impType], impType] =  vector_ctr[indexes]
        mat_rctr_by_Imp[dictCampPerImp[impType], impType] =  vector_qctr[indexes]
        rankingWrtIndexes = (np.argsort(-vector_qctr[indexes])).astype(int)
        ranking[impType] = [dictCampPerImp[impType][z] for z in rankingWrtIndexes]
        posRankingGlobal = [indexes[z] for z in rankingWrtIndexes]
        optBidGreedySP[impType] = np.minimum(vector_qctr[posRankingGlobal], UB_bidLong[posRankingGlobal])
        optBidGreedyFP[impType] =\
            np.minimum(vector_qctr[posRankingGlobal]*adverPerImp[impType]/(adverPerImp[impType]+1.0), UB_bidLong[posRankingGlobal])
    return [mat_r_by_Imp, mat_ctr, mat_rctr_by_Imp, ranking, optBidGreedySP, optBidGreedyFP]

# ### Greedy Heuristic Procedure

# When the greedy heuristic has the opportunity to bid for  given impression type
# it first check the budget to see which of the interested campaigns has enough 
# money to pay in case a click is done and then it bids for the campaign
# that maximizes the profit. Given that Ipinyou assumes second price auctions, 
# the greedy heuristic bids for the campaign with highest revenue times ctr
# that still has enough money to pay for the impression in case of winning. 

# 'CreateMatrixBidAndX' transforms bid and allocation vectors into matrices. This code
# will be used by all methods in the simulation step as we will obtain bidding and 
# allocation vectors for Indicator and Indicator + $\ell_2$ once we run our primal-dual 
# methodology and the greedy step has bidding prices equal to $r_{ik}$. Given that we run 
# our primal-dual methodogy only once per simulation (which is clearly sub-optimal), the 
# allocation vector is enough to decide in behalf of which campaign to bid in behalf of 
# for a whole simulation. 

def CreateMatrixBidAndX(numCampaigns, num_impressions, num_edges, \
    index_Imps, index_sizeCamps, bid_vector, x):
    _, dictCampPerImp = CreateNiceDictionariesForImpAndCampPos(numCampaigns,\
        num_impressions, num_edges, index_Imps, index_sizeCamps)

    mat_bid_by_Imp = np.zeros((numCampaigns, num_impressions))
    mat_x_by_Imp = np.zeros((numCampaigns, num_impressions))
    for impType in range(num_impressions):
        indexes = np.arange(num_edges)[(index_Imps == impType)]
        mat_bid_by_Imp[dictCampPerImp[impType], impType] =  bid_vector[indexes]
        mat_x_by_Imp[dictCampPerImp[impType], impType] =  x[indexes]
    return [mat_bid_by_Imp, mat_x_by_Imp]


# For each impression type $i$, the probability of not bidding for it is
#  $$1-\sum_{k \in \mathcal{K}_i} x_{ik}$$ We obtain the vector of probaility of not bidding 
#  for each impression type in CreateProbOfBidding. In Case we decide to bid for a given 
#  impression type, FastRandomChoice helps us to decide for which campaign to bid in behalf of. 
#  It receives as inputs the vector 'condProbVector' which represent the probability of bidding 
#  in behalf of the camapaigns that  have enough budget to bid (this vector entries are 
#  non-negative and sum up to one), and a number 'unif_value'  which we assumed was sampled 
#  for a uniform random variable. Then, it uses a standard trick to decide on behalf of 
#  which campaign to bid in behalf of. The campaign number which returns is relative to the 
#  campaigns that have enough budget to bid on behalf of.

def CreateProbOfBidding(mat_x_by_Imp):
    return np.sum(mat_x_by_Imp, axis = 0)

def FastRandomChoice(condProbVector, unif_value):
    auxPartSum = 0.0
    for i in range(len(condProbVector)):
        if auxPartSum+condProbVector[i]<unif_value:
            return i
        else:
            auxPartSum += condProbVector[i]
    return len(condProbVector)-1

### Initializing data for each method (Greedy and derived from our method)

# When we run a simulation we would like to save by campaign the amount of bids made, 
# won, and clicks made by each impression type. That info is saved in cartBids, cartWon, 
# and cartClicked resp. Also, as general statistics we would like to know the cost, revenue 
# and profit each impression type brough for the DSP. That info is saved in costBids, revenue, 
# and profit resp. 

# Function CreateDataForSimulation creates all the data needed to tart the simulation for the 
# Greedy and a non-Greedy method. 

def CreateIndicatorSimulation(numCampaigns, num_impressions, vector_m):
    budget = np.zeros(numCampaigns)
    budget[:] = vector_m
    cartBids = np.zeros((numCampaigns, num_impressions))
    cartWon = np.zeros((numCampaigns, num_impressions))
    cartClicked = np.zeros((numCampaigns, num_impressions))
    costBids = np.zeros(num_impressions)
    revenue = np.zeros(num_impressions)
    profit = np.zeros(num_impressions)
    return [budget, cartBids, cartWon, cartClicked, \
        costBids, revenue, profit]

def CreateDataForSimulation(bidFound, xFound, numCampaigns, \
    num_impressions, num_edges, index_Imps, index_sizeCamps, vector_r, \
    vector_ctr, vector_qctr, vector_m,  UB_bidShort, adverPerImp):
    [budgetLR, cartBidsLR, cartWonLR, cartClickedLR, costBidsLR, revenueLR, \
        profitLR] = CreateIndicatorSimulation(numCampaigns, \
        num_impressions, vector_m)
    [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, \
        profitGr] = CreateIndicatorSimulation(numCampaigns, \
        num_impressions, vector_m)
    [mat_r_by_Imp, mat_ctr, mat_rctr_by_Imp, ranking, optBidGreedySP,\
        optBidGreedyFP] = CreateMatR_ctr_Rctr_AndRankings(numCampaigns, num_impressions, num_edges, \
        index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_qctr,  UB_bidShort, adverPerImp)
    [mat_bid_by_ImpLR, mat_x_by_ImpLR] = CreateMatrixBidAndX(numCampaigns, \
        num_impressions, num_edges, index_Imps, index_sizeCamps, \
        bidFound, xFound)
    probBidLR = CreateProbOfBidding(mat_x_by_ImpLR)
    return [budgetLR, cartBidsLR, cartWonLR, cartClickedLR, costBidsLR, \
        revenueLR, profitLR, budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr,\
        mat_rctr_by_Imp, ranking, optBidGreedySP, optBidGreedyFP, \
        mat_bid_by_ImpLR, mat_x_by_ImpLR, probBidLR]

# ## Important Comment About Performance

# (This comment was for Ipinyou but is still mostly valid here)
# The data is not big enough to justify the use of a cluster or parallelization, 
# even though it would help. Even though, I quickly realize after writting a 
# first version of the code that using numpy functions which creates a new 
# array every time they are called can make the code painfully slow (for example 
# calling numpy.argmax when there is only 4 advertisers in the data). For that reason, 
# I sacrifice readability and decide to write 'runIndAndGreedy' and 'RunIndL2IndAndGreedy' 
# as long codes with few function calls and numpy usage.  For the same reason, I decide 
# to create all uniform random variables we need in the simulation at the beginning of 
# this one (so we don't need to call np.random.uniform 4.5 million times in each simulation).


# Comments About the Implementation
# - We win an auction only if the bid_amount is higher than the market price that appear in the Ipinyou Log. 
# In case of winning the auction we then need to check if a click occurs. We update the revenue, profit, 
# budget, cartWon, costBids, and cartClicked accordingly. 
# - For the indicator and indicator+$\ell_2$ case we only need to check the allocation vector to decide 
# the campaign to bid in behalf of (allocation vector that comes from running other primal-dual procedure).

# Simulation code for "\ell_2", Indicator + $\ell_2$, and Greedy 
# The first two methods could be any (Indicator, $\ell_2$, Indicator +$\ell_2$)
def Run_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr, bidsL2, xL2,\
    bidsL2Ind, xL2Ind, UB_bidShort, adverPerImp, ImpInOrder, MPInOrder, firstPrice,\
    multGr):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    [budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, \
        revenueL2, profitL2,  budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr,\
        _, ranking, optBidGreedySP, optBidGreedyFP, mat_bid_by_ImpL2, mat_x_by_ImpL2,\
        probBidL2] = CreateDataForSimulation(bidsL2,\
        xL2, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, vector_r, \
        vector_ctr, vector_qctr, vector_m,  UB_bidShort, adverPerImp)
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind,  _, _, _, _, _, _, _, _, _,_, _, _, _, \
        mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind] = \
        CreateDataForSimulation(bidsL2Ind, xL2Ind, numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_qctr, \
        vector_m,  UB_bidShort, adverPerImp)
    
    ## Now we simulate
    campaignsArange = np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs = allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value = MPInOrder[i]

        ## Update L2 
        indBuyerL2 = 0
        tryToBidL2 = False
        bidAmountL2 = 0.0
        if unifs[0] <= probBidL2[impType]:
            indInterested = [False]*numCampaigns
            bidUsingL2 = False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<= budgetL2[j] and mat_x_by_ImpL2[j, impType]>0:
                    indInterested[j] = True
                    bidUsingL2 = True
            if bidUsingL2: 
                posInt = campaignsArange[indInterested]
                condProbInterested = mat_x_by_ImpL2[:, impType][posInt]
                condProbInterested *= 1.0/np.sum(condProbInterested)
                auxPartSum = 0.0
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2 = numInterest-1
                z = 0
                while z<numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2 = z
                        z += numInterest
                    z += 1
                indBuyerL2 = posInt[auxPosForindBuyerL2]
                tryToBidL2 = True
                bidAmountL2 = mat_bid_by_ImpL2[indBuyerL2, impType]
        if(tryToBidL2):
            cartBidsL2[indBuyerL2, impType] += 1
            if bidAmountL2 >= mp_value:
                ## Impression Won.
                cartWonL2[indBuyerL2, impType] += 1
                if firstPrice:
                    costBidsL2[impType] -= bidAmountL2
                    profitL2[impType] -= bidAmountL2
                else:
                    costBidsL2[impType] -= mp_value
                    profitL2[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerL2, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2[indBuyerL2, impType] += 1
                    payment = mat_r_by_Imp[indBuyerL2, impType]
                    revenueL2[impType] += payment
                    profitL2[impType] += payment
                    budgetL2[indBuyerL2] -= payment

        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind = 0
        tryToBidL2Ind = False
        bidAmountL2Ind = 0.0
        if unifs[0] <= probBidL2Ind[impType]:
            indInterested = [False]*numCampaigns
            bidUsingL2Ind = False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<= budgetL2Ind[j] and mat_x_by_ImpL2Ind[j, impType]>0:
                    indInterested[j] = True
                    bidUsingL2Ind = True
            if bidUsingL2Ind: 
                posInt = campaignsArange[indInterested]
                condProbInterested = mat_x_by_ImpL2Ind[:, impType][posInt]
                condProbInterested *= 1.0/np.sum(condProbInterested)
                auxPartSum = 0.0
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2Ind = numInterest-1
                z = 0
                while z < numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2Ind = z
                        z += numInterest
                    z += 1
                indBuyerL2Ind = posInt[auxPosForindBuyerL2Ind]
                tryToBidL2Ind = True
                bidAmountL2Ind = mat_bid_by_ImpL2Ind[indBuyerL2Ind, impType]
        if(tryToBidL2Ind):
            cartBidsL2Ind[indBuyerL2Ind, impType] += 1
            if bidAmountL2Ind >= mp_value:
                ## Impression Won.
                cartWonL2Ind[indBuyerL2Ind, impType] += 1
                if firstPrice:
                    costBidsL2Ind[impType] -= bidAmountL2Ind
                    profitL2Ind[impType] -= bidAmountL2Ind
                else:
                    costBidsL2Ind[impType] -= mp_value
                    profitL2Ind[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerL2Ind, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2Ind[indBuyerL2Ind, impType] += 1
                    payment = mat_r_by_Imp[indBuyerL2Ind, impType]
                    revenueL2Ind[impType] += payment
                    profitL2Ind[impType] += payment
                    budgetL2Ind[indBuyerL2Ind] -= payment
        ### Now we update the Greedy Policy    
        ## The greedy heuristic bids for the campaign which stills have remaining
        ## budget and from thos bid for the one with highest r times ctr. 
        ## The previous is true as Ipinyou assumes second price auctions.
        indBuyerGr = -1
        bidAmountGr = 0.0
        tryToBidGr = False
        for auxPos,compNum in enumerate(ranking[impType]):
            if  mat_r_by_Imp[compNum, impType] <= budgetGr[compNum]:
                tryToBidGr = True
                indBuyerGr = compNum 
                bidAmountGr = optBidGreedySP[impType][auxPos] * multGr
                if firstPrice:
                    bidAmountGr = optBidGreedyFP[impType][auxPos] * multGr
                break

        ## If tryToBidGr == True, we will bid in behalf of campaign 'indBuyerGr'
        ## the amount 'bidAmountGr'
        if (tryToBidGr):
            ## Save that we are bidding in behalf of 'indBuyerGr' for an impression of 
            ## type 'impType'
            cartBidsGr[indBuyerGr, impType] += 1
            ## We win the auction only if the value we are bidding is higher 
            ## than the market price observed by Ipinyou.
            if bidAmountGr >= mp_value:
                ## Impression Won.
                cartWonGr[indBuyerGr, impType] += 1
                if firstPrice:
                    costBidsGr[impType] -= bidAmountGr
                    profitGr[impType] -= bidAmountGr
                else:
                    costBidsGr[impType] -= mp_value
                    profitGr[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerGr, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType] += 1
                    payment = mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType] += payment
                    profitGr[impType] += payment
                    budgetGr[indBuyerGr] -= payment
    return [budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2,\
            revenueL2, profitL2,\
            budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind,\
            revenueL2Ind, profitL2Ind,\
            budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr,\
            revenueGr, profitGr]

# Simulation code for Indicator, $\ell_2$, Indicator + $\ell_2$, and Greedy
def Run_Ind_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr, bidsInd, xInd,\
    bidsL2, xL2, bidsL2Ind, xL2Ind, tau, UB_bidShort, adverPerImp, ImpInOrder,\
    MPInOrder, firstPrice):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
        revenueInd, profitInd,  budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr,\
        _, ranking, optBidGreedySP, optBidGreedyFP, mat_bid_by_ImpInd, mat_x_by_ImpInd,\
        probBidInd] = CreateDataForSimulation(bidsInd,\
        xInd, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, vector_r, \
        vector_ctr, vector_qctr, vector_m,  UB_bidShort, adverPerImp)

    [budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, \
        revenueL2, profitL2,  _, _, _, _, _, _, _, _, _,_, _, _, _, \
        mat_bid_by_ImpL2, mat_x_by_ImpL2, probBidL2] = \
        CreateDataForSimulation(bidsL2, xL2, numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_qctr, \
        vector_m,  UB_bidShort, adverPerImp)
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind,  _, _, _, _, _, _, _, _, _,_, _, _, _, \
        mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind] = \
        CreateDataForSimulation(bidsL2Ind, xL2Ind, numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_qctr, \
        vector_m,  UB_bidShort, adverPerImp)
    
    ## Now we simulate
    campaignsArange = np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs = allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value = MPInOrder[i]
        
        ## Update Ind
        indBuyerInd = 0
        tryToBidInd = False
        bidAmountInd = 0.0
        ## First we check if the method would try to bid for the impression 
        ## or would just discard it immediately
        if unifs[0] <= probBidInd[impType]:
            ## There are only 4 advertisers, that's why I'm hardcoding
            ## indInterested = [False, False, False, False]
            indInterested = [False]*numCampaigns
            bidUsingInd = False
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<= budgetInd[j] and mat_x_by_ImpInd[j, impType]>0:
                    indInterested[j] = True
                    bidUsingInd = True
            if bidUsingInd: 
                ## There is at least one campaign that wants to bid.
                posInt = campaignsArange[indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested = mat_x_by_ImpInd[:, impType][posInt]
                condProbInterested *= 1.0/np.sum(condProbInterested)
                auxPartSum = 0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerInd = numInterest-1
                z = 0
                while z < numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerInd = z
                        z += numInterest
                    z += 1
                indBuyerInd = posInt[auxPosForindBuyerInd]
                tryToBidInd = True
                bidAmountInd = mat_bid_by_ImpInd[indBuyerInd, impType]
            ## If tryToBidInd == True, we will try to bid inbehalf of campaign indBuyerInd
            ## bidding an amount of bidAmountInd. 
            if(tryToBidInd):
                ## We first register that we are bidding on behalf of indBuyerInd for an 
                ## impression of type impType
                cartBidsInd[indBuyerInd, impType] += 1
                ## We win the auction only if the value we are bidding is higher 
                ## than the market price observed by Ipinyou
                if bidAmountInd >= mp_value:
                    ## Impression Won. Register that we won the impression and the change
                    ## in cost and profit.
                    cartWonInd[indBuyerInd, impType] += 1
                    if firstPrice:
                        costBidsInd[impType] -= bidAmountInd
                        profitInd[impType] -= bidAmountInd
                    else:
                        costBidsInd[impType] -= mp_value
                        profitInd[impType] -= mp_value
                    # Now we need to check if the ad was clicked.
                    probOfClick = mat_ctr[indBuyerInd, impType]
                    if (unifs[2]<= probOfClick):
                        ## User clicked, increase revenue and charge the campaign (i.e. DSP wins money).
                        cartClickedInd[indBuyerInd, impType] += 1
                        payment = mat_r_by_Imp[indBuyerInd, impType]
                        revenueInd[impType] += payment
                        profitInd[impType] += payment
                        budgetInd[indBuyerInd] -= payment

        ## Update L2 (Same code as done before for the pure indicator case)
        indBuyerL2 = 0
        tryToBidL2 = False
        bidAmountL2 = 0.0
        if unifs[0] <= probBidL2[impType]:
            indInterested = [False]*numCampaigns
            bidUsingL2 = False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<= budgetL2[j] and mat_x_by_ImpL2[j, impType]>0:
                    indInterested[j] = True
                    bidUsingL2 = True
            if bidUsingL2: 
                posInt = campaignsArange[indInterested]
                condProbInterested = mat_x_by_ImpL2[:, impType][posInt]
                condProbInterested *= 1.0/np.sum(condProbInterested)
                auxPartSum = 0.0
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2 = numInterest-1
                z = 0
                while z<numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2 = z
                        z += numInterest
                    z += 1
                indBuyerL2 = posInt[auxPosForindBuyerL2]
                tryToBidL2 = True
                bidAmountL2 = mat_bid_by_ImpL2[indBuyerL2, impType]
        if(tryToBidL2):
            cartBidsL2[indBuyerL2, impType] += 1
            if bidAmountL2 >= mp_value:
                ## Impression Won.
                cartWonL2[indBuyerL2, impType] += 1
                if firstPrice:
                    costBidsL2[impType] -= bidAmountL2
                    profitL2[impType] -= bidAmountL2
                else:
                    costBidsL2[impType] -= mp_value
                    profitL2[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerL2, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2[indBuyerL2, impType] += 1
                    payment = mat_r_by_Imp[indBuyerL2, impType]
                    revenueL2[impType] += payment
                    profitL2[impType] += payment
                    budgetL2[indBuyerL2] -= payment

        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind = 0
        tryToBidL2Ind = False
        bidAmountL2Ind = 0.0
        if unifs[0] <= probBidL2Ind[impType]:
            indInterested = [False]*numCampaigns
            bidUsingL2Ind = False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<= budgetL2Ind[j] and mat_x_by_ImpL2Ind[j, impType]>0:
                    indInterested[j] = True
                    bidUsingL2Ind = True
            if bidUsingL2Ind: 
                posInt = campaignsArange[indInterested]
                condProbInterested = mat_x_by_ImpL2Ind[:, impType][posInt]
                condProbInterested *= 1.0/np.sum(condProbInterested)
                auxPartSum = 0.0
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2Ind = numInterest-1
                z = 0
                while z < numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2Ind = z
                        z += numInterest
                    z += 1
                indBuyerL2Ind = posInt[auxPosForindBuyerL2Ind]
                tryToBidL2Ind = True
                bidAmountL2Ind = mat_bid_by_ImpL2Ind[indBuyerL2Ind, impType]
        if(tryToBidL2Ind):
            cartBidsL2Ind[indBuyerL2Ind, impType] += 1
            if bidAmountL2Ind >= mp_value:
                ## Impression Won.
                cartWonL2Ind[indBuyerL2Ind, impType] += 1
                if firstPrice:
                    costBidsL2Ind[impType] -= bidAmountL2Ind
                    profitL2Ind[impType] -= bidAmountL2Ind
                else:
                    costBidsL2Ind[impType] -= mp_value
                    profitL2Ind[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerL2Ind, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2Ind[indBuyerL2Ind, impType] += 1
                    payment = mat_r_by_Imp[indBuyerL2Ind, impType]
                    revenueL2Ind[impType] += payment
                    profitL2Ind[impType] += payment
                    budgetL2Ind[indBuyerL2Ind] -= payment
        ### Now we update the Greedy Policy    
        ## The greedy heuristic bids for the campaign which stills have remaining
        ## budget and from thos bid for the one with highest r times ctr. 
        ## The previous is true as Ipinyou assumes second price auctions.
        indBuyerGr = -1
        bidAmountGr = 0.0
        tryToBidGr = False
        for auxPos,compNum in enumerate(ranking[impType]):
            if  mat_r_by_Imp[compNum, impType] <= budgetGr[compNum]:
                tryToBidGr = True
                indBuyerGr = compNum 
                bidAmountGr = optBidGreedySP[impType][auxPos]
                if firstPrice:
                    bidAmountGr = optBidGreedyFP[impType][auxPos]
                break

        ## If tryToBidGr == True, we will bid in behalf of campaign 'indBuyerGr'
        ## the amount 'bidAmountGr'
        if (tryToBidGr):
            ## Save that we are bidding in behalf of 'indBuyerGr' for an impression of 
            ## type 'impType'
            cartBidsGr[indBuyerGr, impType] += 1
            ## We win the auction only if the value we are bidding is higher 
            ## than the market price observed by Ipinyou.
            if bidAmountGr >= mp_value:
                ## Impression Won.
                cartWonGr[indBuyerGr, impType] += 1
                if firstPrice:
                    costBidsGr[impType] -= bidAmountGr
                    profitGr[impType] -= bidAmountGr
                else:
                    costBidsGr[impType] -= mp_value
                    profitGr[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerGr, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType] += 1
                    payment = mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType] += payment
                    profitGr[impType] += payment
                    budgetGr[indBuyerGr] -= payment
    return [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
            revenueInd, profitInd,\
            budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2,\
            revenueL2, profitL2,\
            budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind,\
            revenueL2Ind, profitL2Ind,\
            budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr,\
            revenueGr, profitGr]


# ## Simulation for the profit maximization, profit maximization + L2 and Greedy

# Here we want to run the experiment change the budget values to be [(1.0/32.0), (1.0/8.0), .25, 0.5, 1.0]
#  of the budgets used by Ipinyou. The iteration over the percentage budget values is done in 'for perc in perVector_m:'


def ExperIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_s, vector_ctr, alphasInd, num_itInd, alphasL2Ind,\
    num_itL2Ind, p_grad_TypeInd, p_grad_TypeL2Ind, init_lam, seq_vectorM, \
    UB_bidShort, adverPerImp, firstPrice, sim = 100, addIndicator = True, shuffle = True):

    ext_s = vector_s[index_Imps]
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    global p_grad_Type
    vector_qctr = np.multiply(vector_r, vector_ctr)
    dictToRetInd = {}
    dictToRetL2Ind = {}
    dictToRetGr = {}
    
    for numOfVecM,vector_m in enumerate(seq_vectorM):
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        print("Number of Vector M: "+str(numOfVecM))
        dictToRetInd[numOfVecM] = []
        dictToRetL2Ind[numOfVecM] = []
        dictToRetGr[numOfVecM] = []
        p_grad_Type = p_grad_TypeInd
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg] = SubgrAlgSavPrimDualObjInd(init_lam, \
            num_itInd, alphasInd, vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, \
            num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,  UB_bidShort, firstPrice,\
            adverPerImp, (num_itInd-1), p_grad_Type)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal = dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal = ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsInd = OptimalBids(ext_LamFinal, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsInd, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xInd = CalculateLPGurobi(rho_eval, beta_eval, vector_qctr, vector_m, \
            ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
            index_sizeCamps, saveObjFn = False, tol = 0.00000001)
        
        p_grad_Type = p_grad_TypeL2Ind
        tau = np.power(vector_m, -1)
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg] = SubgrAlgSavPrimDualObjFn_L2Ind(init_lam,\
            num_itL2Ind, alphasL2Ind, vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, \
            num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,  UB_bidShort, firstPrice, \
            adverPerImp, (num_itInd-1), p_grad_Type, tau, True)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal = dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal = ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind = OptimalBids(ext_LamFinal, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsL2Ind, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2Ind = CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
            num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
            tau, addIndicator = addIndicator, saveObjFn = False, tol = 0.00000001)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('Finished running the Primal-Dual Algorithms')
        s = time.time()
        for i in range(sim): 
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle)
            [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
            revenueInd, profitInd, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
            cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr] = \
            RunIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
            index_sizeCamps,  vector_r, vector_m, vector_ctr, \
            vector_qctr, bidsInd, xInd, bidsL2Ind, \
            xL2Ind, tau, UB_bidShort, adverPerImp, ImpInOrder, MPInOrder, firstPrice)
            dictToRetInd[numOfVecM].append([budgetInd, cartBidsInd, cartWonInd, \
                cartClickedInd, costBidsInd, revenueInd, profitInd])
            dictToRetL2Ind[numOfVecM].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])
            dictToRetGr[numOfVecM].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, \
            costBidsGr, revenueGr, profitGr])
        print("Running the Whole Simulation Code took: "+str(time.time()-s)+"secs.")
    return [dictToRetInd, dictToRetL2Ind, dictToRetGr]

def Exper_Ind_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_s, vector_ctr, alphasInd, num_itInd, alphasL2, num_itL2,\
    alphasL2Ind, num_itL2Ind, p_grad_TypeInd, p_grad_TypeL2, p_grad_TypeL2Ind, init_lam, \
    seq_vectorM, UB_bidShort, adverPerImp, firstPrice, sim = 100, shuffle = True):

    ext_s = vector_s[index_Imps]
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    global p_grad_Type
    vector_qctr = np.multiply(vector_r, vector_ctr)
    dictToRetInd = {}
    dictToRetL2 = {}
    dictToRetL2Ind = {}
    dictToRetGr = {}
    
    for numOfVecM,vector_m in enumerate(seq_vectorM):
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        print("Budget per Campaign: "+str(vector_m[0]))
        dictToRetInd[numOfVecM] = []
        dictToRetL2[numOfVecM] = []
        dictToRetL2Ind[numOfVecM] = []
        dictToRetGr[numOfVecM] = []

        ## Two-Phase 
        ## Indicator Method
        startTime = time.time()
        p_grad_Type = p_grad_TypeInd 
        # The full return is [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        # primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]
        # If you want to plot the duality gap just do:
        # print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        print('Running Two-Phase: Indicator', end=', ')
        [_, _, _, _, _, _, _, dual_varsAvg] = SubgrAlgSavPrimDualObjInd(init_lam, \
            num_itInd, alphasInd, vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, \
            num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,  UB_bidShort, firstPrice,\
            adverPerImp, (num_itInd-1), p_grad_Type)
        
        lamFinal = dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal = ExtendSizeCamps(lamFinal, index_sizeCamps)

        bidsInd = OptimalBids(ext_LamFinal, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsInd, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xInd = CalculateLPGurobi(rho_eval, beta_eval, vector_qctr, vector_m, \
            ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
            index_sizeCamps, saveObjFn = False, tol = 0.00000001)

        ## L2
        p_grad_Type = p_grad_TypeL2
        tau = np.power(vector_m, -1)
        print('L2', end=', ')
        [_, _, _, _, _, _, _, dual_varsAvg] = SubgrAlgSavPrimDualObjFn_L2Ind(init_lam,\
            num_itL2, alphasL2, vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, \
            num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,  UB_bidShort, firstPrice, \
            adverPerImp, (num_itInd-1), p_grad_Type, tau, False)
        
        lamFinal = dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal = ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2 = OptimalBids(ext_LamFinal, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsL2, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2 = CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
                num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
                tau, addIndicator = False, saveObjFn = False, tol = 0.00000001)
        
        ## Indicator + L2
        p_grad_Type = p_grad_TypeL2Ind
        tau = np.power(vector_m, -1)
        print('L2 + Indicator')
        [_, _, _, _, _, _, _, dual_varsAvg] = SubgrAlgSavPrimDualObjFn_L2Ind(init_lam,\
            num_itL2Ind, alphasL2Ind, vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, \
            num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,  UB_bidShort, firstPrice, \
            adverPerImp, (num_itInd-1), p_grad_Type, tau, True)
        
        lamFinal = dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal = ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind = OptimalBids(ext_LamFinal, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsL2Ind, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2Ind = CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
                num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
                tau, addIndicator = True, saveObjFn = False, tol = 0.00000001)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('Running the Primal-Dual Algorithms took: '+str(time.time()-startTime))
        startTime = time.time()
        for i in range(sim): 
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle)
            [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
            revenueInd, profitInd, budgetL2, cartBidsL2, cartWonL2, cartClickedL2, \
            costBidsL2, revenueL2, profitL2, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
            cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr] = \
            Run_Ind_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
                index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr, bidsInd, xInd,\
                bidsL2, xL2, bidsL2Ind, xL2Ind, tau, UB_bidShort, adverPerImp, ImpInOrder,\
                MPInOrder, firstPrice)

            dictToRetInd[numOfVecM].append([budgetInd, cartBidsInd, cartWonInd, \
                cartClickedInd, costBidsInd, revenueInd, profitInd])

            dictToRetL2[numOfVecM].append([budgetL2, cartBidsL2, cartWonL2, \
                cartClickedL2, costBidsL2, revenueL2, profitL2])

            dictToRetL2Ind[numOfVecM].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])

            dictToRetGr[numOfVecM].append([budgetGr, cartBidsGr, cartWonGr, \
                cartClickedGr, costBidsGr, revenueGr, profitGr])

        print("Running the Whole Simulation Code took: "+str(time.time()-startTime)+"secs.")
    return [dictToRetInd, dictToRetL2, dictToRetL2Ind, dictToRetGr]



## For the Pareto Experiment we need to Run only the L2+Indicator for several values of \tau a
## number of simulations. We also need to run the greedy method 

def RunSimOnlyGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr, UB_bidShort,\
    adverPerImp, ImpInOrder, MPInOrder, firstPrice, multiplier = 1.0):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    [_, _, _, _, _, _, _,  budgetGr, cartBidsGr, cartWonGr, cartClickedGr,\
        costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr, _, ranking,\
        optBidGreedySP, optBidGreedyFP, _, _, _] = \
        CreateDataForSimulation(np.zeros(num_edges), np.zeros(num_edges),\
        numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps,\
        vector_r, vector_ctr, vector_qctr, vector_m,  UB_bidShort, adverPerImp)
    
    ## Now we simulate
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs = allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value = MPInOrder[i]
        
        ### Now we update the Greedy Policy    
        ## The greedy heuristic bids for the campaign which stills have remaining
        ## budget and from thos bid for the one with highest r times ctr. 
        ## The previous is true as Ipinyou assumes second price auctions.
        max_rctr = 0.0
        indBuyerGr = -1
        bidAmountGr = 0.0
        tryToBidGr = False
        for auxPos,compNum in enumerate(ranking[impType]):
            if  mat_r_by_Imp[compNum, impType] <= budgetGr[compNum]:
                tryToBidGr = True
                indBuyerGr = compNum 
                bidAmountGr = optBidGreedySP[impType][auxPos] * multiplier
                if firstPrice:
                    bidAmountGr = optBidGreedyFP[impType][auxPos] * multiplier
        ## If tryToBidGr == True, we will bid in behalf of campaign 'indBuyerGr'
        ## the amount 'bidAmountGr'
        if (tryToBidGr):
            ## Save that we are bidding in behalf of 'indBuyerGr' for an impression of 
            ## type 'impType'
            cartBidsGr[indBuyerGr, impType] += 1
            ## We win the auction only if the value we are bidding is higher 
            ## than the market price observed by Ipinyou.
            if bidAmountGr >= mp_value:
                ## Impression Won.
                cartWonGr[indBuyerGr, impType] += 1
                if firstPrice:
                    costBidsGr[impType] -= bidAmountGr
                    profitGr[impType] -= bidAmountGr
                else:
                    costBidsGr[impType] -= mp_value
                    profitGr[impType] -= mp_value
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerGr, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType] += 1
                    payment = mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType] += payment
                    profitGr[impType] += payment
                    budgetGr[indBuyerGr] -= payment
    return [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]


def RunOneSimL2Ind(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr,\
    bidsL2Ind, xL2Ind, UB_bidShort, adverPerImp, ImpInOrder, MPInOrder, firstPrice):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind,  _, _, _, _, _, _, _, mat_r_by_Imp, mat_ctr,\
        _, _, _, _, mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind] = \
        CreateDataForSimulation(bidsL2Ind, xL2Ind, numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_qctr, \
        vector_m,  UB_bidShort, adverPerImp)
    
    ## Now we simulate
    campaignsArange = np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs = allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value = MPInOrder[i]
        
        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind = 0
        tryToBidL2Ind = False
        bidAmountL2Ind = 0.0
        if unifs[0] <= probBidL2Ind[impType]:
            indInterested = [False]*numCampaigns
            bidUsingL2Ind = False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<= budgetL2Ind[j] and mat_x_by_ImpL2Ind[j, impType]>0:
                    indInterested[j] = True
                    bidUsingL2Ind = True
            if bidUsingL2Ind: 
                posInt = campaignsArange[indInterested]
                condProbInterested = mat_x_by_ImpL2Ind[:, impType][posInt]
                condProbInterested *= 1.0/np.sum(condProbInterested)
                auxPartSum = 0.0
                numInterest = len(condProbInterested)
                z = 0
                while z <numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2Ind=z
                        z+=numInterest
                    z += 1
                indBuyerL2Ind = posInt[auxPosForindBuyerL2Ind]
                tryToBidL2Ind = True
                bidAmountL2Ind = mat_bid_by_ImpL2Ind[indBuyerL2Ind, impType]
        if(tryToBidL2Ind):
            cartBidsL2Ind[indBuyerL2Ind, impType] += 1
            if bidAmountL2Ind >= mp_value:
                ## Impression Won.
                cartWonL2Ind[indBuyerL2Ind, impType] += 1
                if firstPrice:
                    costBidsL2Ind[impType] -= bidAmountL2Ind
                    profitL2Ind[impType] -= bidAmountL2Ind
                else:
                    costBidsL2Ind[impType] -= mp_value
                    profitL2Ind[impType] -= mp_value                   
                # Now we need to check if the ad was clicked.
                probOfClick = mat_ctr[indBuyerL2Ind, impType]
                if (unifs[2]<= probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2Ind[indBuyerL2Ind, impType] += 1
                    payment = mat_r_by_Imp[indBuyerL2Ind, impType]
                    revenueL2Ind[impType] += payment
                    profitL2Ind[impType] += payment
                    budgetL2Ind[indBuyerL2Ind] -= payment
    return [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind]


def RunOnlyGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_s, vector_r, vector_m, vector_ctr, UB_bidShort,\
    adverPerImp, sim, firstPrice, vecOfSeeds = None,\
    shuffle = True):

    vector_qctr = np.multiply(vector_r, vector_ctr)
    vector_m = vector_m
    toRetGreedy = []
    for i in range(sim): 
        ImpInOrder, MPInOrder = 0, 0
        if vecOfSeeds is not None:
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle, vecOfSeeds[i])
        else:
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle)
        [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, \
        revenueGr, profitGr] = RunSimOnlyGreedy(numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_m, vector_ctr,\
        vector_qctr, UB_bidShort,adverPerImp, ImpInOrder, MPInOrder, firstPrice)
        toRetGreedy.append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, \
        revenueGr, profitGr])
    return toRetGreedy

def ExpPareto(numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, vector_s, vector_r,\
    vector_m, vector_ctr, vector_qctr, UB_bidShort, adverPerImp, alphasInd, num_itInd, alphasL2, num_itL2,\
    alphasL2Ind, num_itL2Ind, p_grad_TypeInd, p_grad_TypeL2, p_grad_TypeL2Ind, multsGreedy, multsTwoPhase, \
    init_lam, sim, firstPrice, vecOfSeeds = None, shuffle = False):

    ext_s = vector_s[index_Imps]
    global p_grad_Type
    dictToRetL2 = {}
    dictToRetL2Ind = {}
    dictToRetGr = {}
    listToRetInd = []

    sizeMult = len(multsGreedy)

    if len(multsGreedy) != len(multsTwoPhase):
        print('Error: The size of the multipliers for the Greedy method and L2/L2Ind should be the same')
        sys.exit()

    
    print('We will first run the indicator case. ')

    startTime = time.time()

    ## Ind
    p_grad_Type = p_grad_TypeInd
    [_, _, _, _, _, _, _, dual_varsAvgInd] = SubgrAlgSavPrimDualObjInd(init_lam[:], \
        num_itInd, alphasInd, vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, \
        num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,  UB_bidShort, firstPrice,\
        adverPerImp, (num_itInd-1), p_grad_Type)
    
    lamFinalInd = dual_varsAvgInd[len(dual_varsAvgInd)-1]
    ext_LamFinalInd = ExtendSizeCamps(lamFinalInd, index_sizeCamps)

    bidsInd = OptimalBids(ext_LamFinalInd, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
    rho_evalInd, beta_evalInd = CalcRhoAndBetaVectors(bidsInd, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
    xInd = CalculateLPGurobi(rho_evalInd, beta_evalInd, vector_qctr, vector_m, \
        ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
        index_sizeCamps, saveObjFn = False, tol = 0.00000001)

    del lamFinalInd
    del ext_LamFinalInd

    for i in range(sim): 
        ImpInOrder, MPInOrder = 0, 0
        if vecOfSeeds is not None:
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle, vecOfSeeds[i])
        else:
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle)

        listToRetInd.append(RunOneSimL2Ind(numCampaigns, num_impressions, num_edges, index_Imps, \
            index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr,\
            bidsInd, xInd, UB_bidShort, adverPerImp, ImpInOrder, MPInOrder, firstPrice))

    print('Finishing running the Indicator part. It took: '+str(time.time()-startTime)+' secs.')



    for i in range(sizeMult):
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        startTime = time.time()
        multGr = multsGreedy[i]
        multTP = multsTwoPhase[i]
        print("Greedy multiplier: "+str(multGr) +', L2/L2Ind multiplier: '+str(multTP))

        dictToRetL2[multTP] = []
        dictToRetL2Ind[multTP] = []
        dictToRetGr[multGr] = []

        ## The same tau is used for L2/L2Ind
        tau = np.power(vector_m, -1) * multTP

        ## L2
        p_grad_Type = p_grad_TypeL2
        [_, _, _, _, _, _, _, dual_varsAvgL2] = SubgrAlgSavPrimDualObjFn_L2Ind(init_lam[:], num_itL2Ind, alphasL2Ind, \
            vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
            index_sizeCamps, index_Imps,  UB_bidShort, firstPrice, adverPerImp, (num_itL2Ind-1), p_grad_Type, tau, False)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinalL2 = dual_varsAvgL2[-1]
        ext_LamFinalL2 = ExtendSizeCamps(lamFinalL2, index_sizeCamps)
        bidsL2 = OptimalBids(ext_LamFinalL2, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsL2, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2 = CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
            num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
            tau, addIndicator = False, saveObjFn = False, tol = 0.00000001)


        ## L2Ind
        p_grad_Type = p_grad_TypeL2Ind
        [_, _, _, _, _, _, _, dual_varsAvgL2Ind] = SubgrAlgSavPrimDualObjFn_L2Ind(init_lam[:], num_itL2Ind, alphasL2Ind, \
            vector_r, vector_ctr, vector_qctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
            index_sizeCamps, index_Imps,  UB_bidShort, firstPrice, adverPerImp, (num_itL2Ind-1), p_grad_Type, tau, True)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinalL2Ind = dual_varsAvgL2Ind[-1]
        ext_LamFinalL2Ind = ExtendSizeCamps(lamFinalL2Ind, index_sizeCamps)
        bidsL2Ind = OptimalBids(ext_LamFinalL2Ind, vector_qctr, UB_bidShort[index_Imps], firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidsL2Ind, UB_bidShort, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2Ind = CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
            num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
            tau, addIndicator = True, saveObjFn = False, tol = 0.00000001)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('Finished running the Primal-Dual Algorithms, it took: '+str(time.time()-startTime)+' secs.')
        startTime = time.time()
        for i in range(sim): 
            ImpInOrder, MPInOrder = 0, 0
            if vecOfSeeds is not None:
                ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                    vector_s, adverPerImp, shuffle, vecOfSeeds[i])
            else:
                ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                    vector_s, adverPerImp, shuffle)

            [budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, \
            revenueL2, profitL2, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
            cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr] = \
            Run_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
                index_sizeCamps, vector_r, vector_m, vector_ctr, vector_qctr, bidsL2, xL2,\
                bidsL2Ind, xL2Ind, UB_bidShort, adverPerImp, ImpInOrder, MPInOrder, firstPrice,\
                multGr)

            dictToRetL2[multTP].append([budgetL2, cartBidsL2, cartWonL2, \
                cartClickedL2, costBidsL2, revenueL2, profitL2])

            dictToRetL2Ind[multTP].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])

            dictToRetGr[multGr].append([budgetGr, cartBidsGr, cartWonGr, \
                cartClickedGr, costBidsGr, revenueGr, profitGr])
        print('Running the simulations took '+str(time.time()-startTime)+" secs.")
    return [listToRetInd, dictToRetL2, dictToRetL2Ind, dictToRetGr]