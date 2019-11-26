import numpy as np
import sys
import time
from RhoAndBeta import CalcRhoAndBetaVectors
from UtilitiesOptimization import CalculateQuadGurobi, SubgrAlgSavPrimDualObjInd, \
    SubgrAlgSavPrimDualObjFn_L2Ind, ExtendSizeCamps, OptimalBids, OptimalX
from SubgradientAlgPart import SubgrAlgSavPrimDualObjInd, SubgrAlgSavPrimDualObjFn_L2Ind

#
#
## Extra Functions that weren't needed for Ipinyou nor Criteo

# Ranking For Greedy Method
def RankingAndOptBids(dictWithComp, dictWithRCTR,  UB_bid, adverPerImp):
    ranking = {}
    optBidGreedySP = {}
    optBidGreedyFP = {}
    for i in dictWithComp.keys():
        rctrAsArray = np.array(dictWithRCTR[i])
        compAsArray = np.array(dictWithComp[i])
        aux_order = (np.argsort(-rctrAsArray)).astype(int)
        ranking[i] = compAsArray[aux_order]
        optBidGreedySP[i] = rctrAsArray[aux_order]
        optBidGreedyFP[i] = (adverPerImp[i]/(adverPerImp[i]+1.0))*optBidGreedySP[i]
    return ranking, optBidGreedySP, optBidGreedyFP

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
        maxUnif =  auxUniforms[counter]
        counter += 1
        if adverPerImp[impType]>1:
            for j in range(1,adverPerImp[impType]):
                if auxUniforms[counter] > maxUnif:
                    maxUnif = auxUniforms[counter]
                counter += 1
        mpArrivals[pos] = maxUnif
    return impArrivals, mpArrivals

#
#
## Simulation Code

## To make the simulation code faster and easier to read (in particular the greedy heuristic)
## we would like to change the vector of click-through rates 'vector_ctr', vector of 
## revenues 'vector_r', of revenue times the click trhough rate 'vector_rctr', and others 
## into a matrix of size number of campaigns times number of impressions.
## That's done in the following function.
def CreateMatR_ctr_Rctr_AndRankings(numCampaigns, num_impressions, num_edges, \
    index_Imps, index_sizeCamps, vector_r, vector_ctr, \
    vector_rctr,  UB_bid, adverPerImp):
    ## mat_r_by_Imp is a matrix in which each column 'i'
    ## represents the valuations of all campaigns in order for 
    ## an impression of type 'i'. If the campaign is not interested 
    ## in the impression a zero value is entered in that position.
    mat_r_by_Imp = np.zeros((numCampaigns, num_impressions))
    mat_ctr = np.zeros((numCampaigns, num_impressions))
    mat_rctr_by_Imp = np.zeros((numCampaigns, num_impressions))
    auxRCTR = {}
    greedyCampOrdered = {}
    for i in range(num_impressions):
        count = 0
        aux = 0
        indexes = np.arange(num_edges)[(index_Imps == i)]
        sizeIndexes = len(indexes)
        compInImp = []
        rctrInOrder = []
        if(sizeIndexes!= 0):
            pos = indexes[aux]
            for j in range(numCampaigns):
                impInCamp = index_sizeCamps[j]
                if (pos<(count+impInCamp)):
                    compInImp.append(j)
                    mat_r_by_Imp[j, i] = vector_r[pos]
                    mat_ctr[j, i] = vector_ctr[pos]
                    mat_rctr_by_Imp[j, i] = vector_rctr[pos]
                    rctrInOrder.append(mat_rctr_by_Imp[j, i])
                    if(aux<sizeIndexes-1):
                        aux += 1
                        pos = indexes[aux]
                    else:
                        # No more campaigns use that impression
                        pos = num_edges
                count += impInCamp
        greedyCampOrdered[i] = compInImp
        auxRCTR[i] = rctrInOrder
    ranking, optBidGreedySP, optBidGreedyFP = RankingAndOptBids(greedyCampOrdered, \
        auxRCTR,  UB_bid, adverPerImp)
    return [mat_r_by_Imp, mat_ctr, mat_rctr_by_Imp, ranking, optBidGreedySP,\
        optBidGreedyFP]

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
    mat_bid_by_Imp=np.zeros((numCampaigns, num_impressions))
    mat_x_by_Imp=np.zeros((numCampaigns, num_impressions))
    for i in range(num_impressions):
        count=0
        aux=0
        indexes=np.arange(num_edges)[(index_Imps==i)]
        sizeIndexes=len(indexes)
        if(sizeIndexes!=0):
            pos=indexes[aux]
            for j in range(numCampaigns):
                impInCamp=index_sizeCamps[j]
                if (pos<(count+impInCamp)):
                    mat_bid_by_Imp[j, i]=bid_vector[pos]
                    mat_x_by_Imp[j, i]=x[pos]
                    if(aux<sizeIndexes-1):
                        aux+=1
                        pos=indexes[aux]
                    else:
                        # No more campaigns use that impression
                        # This should be done with a while.
                        pos=num_edges
                count+=impInCamp
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
    return np.sum(mat_x_by_Imp, axis=0)

def FastRandomChoice(condProbVector, unif_value):
    auxPartSum=0.0
    for i in range(len(condProbVector)):
        if auxPartSum+condProbVector[i]<unif_value:
            return i
        else:
            auxPartSum+=condProbVector[i]
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
    budget=np.zeros(numCampaigns)
    budget[:]=vector_m
    cartBids=np.zeros((numCampaigns, num_impressions))
    cartWon=np.zeros((numCampaigns, num_impressions))
    cartClicked=np.zeros((numCampaigns, num_impressions))
    costBids=np.zeros(num_impressions)
    revenue=np.zeros(num_impressions)
    profit=np.zeros(num_impressions)
    return [budget, cartBids, cartWon, cartClicked, \
        costBids, revenue, profit]

def CreateDataForSimulation(bidFound, xFound, numCampaigns, \
    num_impressions, num_edges, index_Imps, index_sizeCamps, vector_r, \
    vector_ctr, vector_rctr, vector_m,  UB_bid, adverPerImp):
    [budgetLR, cartBidsLR, cartWonLR, cartClickedLR, costBidsLR, revenueLR, \
        profitLR]=CreateIndicatorSimulation(numCampaigns, \
        num_impressions, vector_m)
    [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, \
        profitGr]=CreateIndicatorSimulation(numCampaigns, \
        num_impressions, vector_m)
    [mat_r_by_Imp, mat_ctr, mat_rctr_by_Imp, ranking, optBidGreedySP,\
        optBidGreedyFP]=CreateMatR_ctr_Rctr_AndRankings(numCampaigns, num_impressions, num_edges, \
        index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_rctr,  UB_bid, adverPerImp)
    [mat_bid_by_ImpLR, mat_x_by_ImpLR]=CreateMatrixBidAndX(numCampaigns, \
        num_impressions, num_edges, index_Imps, index_sizeCamps, \
        bidFound, xFound)
    probBidLR=CreateProbOfBidding(mat_x_by_ImpLR)
    return [budgetLR, cartBidsLR, cartWonLR, cartClickedLR, costBidsLR, \
        revenueLR, profitLR, budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr,\
        mat_rctr_by_Imp, ranking, optBidGreedySP, optBidGreedyFP, \
        mat_bid_by_ImpLR, mat_x_by_ImpLR, probBidLR]

# ## Important Comment About Performance

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

# Simulation code for Indicator, Indicator + $\ell_2$, and Greedy
def RunIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_rctr, bidsInd, xInd,\
    bidsL2Ind, xL2Ind, tau, UB_bid, adverPerImp, ImpInOrder, MPInOrder, firstPrice):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
        revenueInd, profitInd,  budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr,\
        _, ranking, optBidGreedySP, optBidGreedyFP, mat_bid_by_ImpInd, mat_x_by_ImpInd,\
        probBidInd]=CreateDataForSimulation(bidsInd,\
        xInd, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, vector_r, \
        vector_ctr, vector_rctr, vector_m,  UB_bid, adverPerImp)
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind,  _, _, _, _, _, _, _, _, _,_, _, _, _, \
        mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind]=\
        CreateDataForSimulation(bidsL2Ind, xL2Ind, numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_rctr, \
        vector_m,  UB_bid, adverPerImp)
    
    ## Now we simulate
    campaignsArange=np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse=np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs=allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value=MPInOrder[i]
        
        ## Update Ind
        indBuyerInd=0
        tryToBidInd=False
        bidAmountInd=0.0
        ## First we check if the method would try to bid for the impression 
        ## or would just discard it immediately
        if unifs[0] <= probBidInd[impType]:
            ## There are only 4 advertisers, that's why I'm hardcoding
            ## indInterested=[False, False, False, False]
            indInterested=[False]*numCampaigns
            bidUsingInd=False
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<=budgetInd[j] and mat_x_by_ImpInd[j, impType]>0:
                    indInterested[j]=True
                    bidUsingInd=True
            if bidUsingInd: 
                ## There is at least one campaign that wants to bid.
                posInt=campaignsArange[indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpInd[:, impType][posInt]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest=len(condProbInterested)
                auxPosForindBuyerInd=numInterest-1
                for z in range(numInterest):
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerInd=z
                        z+=numInterest
                indBuyerInd=posInt[auxPosForindBuyerInd]
                tryToBidInd=True
                bidAmountInd=mat_bid_by_ImpInd[indBuyerInd, impType]
            ## If tryToBidInd == True, we will try to bid inbehalf of campaign indBuyerInd
            ## bidding an amount of bidAmountInd. 
            if(tryToBidInd):
                ## We first register that we are bidding on behalf of indBuyerInd for an 
                ## impression of type impType
                cartBidsInd[indBuyerInd, impType]+=1
                ## We win the auction only if the value we are bidding is higher 
                ## than the market price observed by Ipinyou
                if bidAmountInd>= mp_value:
                    ## Impression Won. Register that we won the impression and the change
                    ## in cost and profit.
                    cartWonInd[indBuyerInd, impType]+=1
                    costBidsInd[impType]-=mp_value
                    profitInd[impType]-=mp_value
                    # Now we need to check if the ad was clicked.
                    probOfClick=mat_ctr[indBuyerInd, impType]
                    if (unifs[2]<=probOfClick):
                        ## User clicked, increase revenue and charge the campaign (i.e. DSP wins money).
                        cartClickedInd[indBuyerInd, impType]+=1
                        payment=mat_r_by_Imp[indBuyerInd, impType]
                        revenueInd[impType]+=payment
                        profitInd[impType]+=payment
                        budgetInd[indBuyerInd]-=payment
        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind=0
        tryToBidL2Ind=False
        bidAmountL2Ind=0.0
        if unifs[0] <= probBidL2Ind[impType]:
            indInterested=[False]*numCampaigns
            bidUsingL2Ind=False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<=budgetL2Ind[j] and mat_x_by_ImpL2Ind[j, impType]>0:
                    indInterested[j]=True
                    bidUsingL2Ind=True
            if bidUsingL2Ind: 
                posInt=campaignsArange[indInterested]
                condProbInterested=mat_x_by_ImpL2Ind[:, impType][posInt]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                numInterest=len(condProbInterested)
                auxPosForindBuyerL2Ind=numInterest-1
                for z in range(numInterest):
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        auxPosForindBuyerL2Ind=z
                        z+=numInterest
                indBuyerL2Ind=posInt[auxPosForindBuyerL2Ind]
                tryToBidL2Ind=True
                bidAmountL2Ind=mat_bid_by_ImpL2Ind[indBuyerL2Ind, impType]
        if(tryToBidL2Ind):
            cartBidsL2Ind[indBuyerL2Ind, impType]+=1
            if bidAmountL2Ind>= mp_value:
                ## Impression Won.
                cartWonL2Ind[indBuyerL2Ind, impType]+=1
                costBidsL2Ind[impType]-=mp_value
                profitL2Ind[impType]-=mp_value
                # Now we need to check if the ad was clicked.
                probOfClick=mat_ctr[indBuyerL2Ind, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2Ind[indBuyerL2Ind, impType]+=1
                    payment=mat_r_by_Imp[indBuyerL2Ind, impType]
                    revenueL2Ind[impType]+=payment
                    profitL2Ind[impType]+=payment
                    budgetL2Ind[indBuyerL2Ind]-=payment
        ### Now we update the Greedy Policy    
        ## The greedy heuristic bids for the campaign which stills have remaining
        ## budget and from thos bid for the one with highest r times ctr. 
        ## The previous is true as Ipinyou assumes second price auctions.
        indBuyerGr=-1
        bidAmountGr=0.0
        tryToBidGr = False
        for auxPos,compNum in enumerate(ranking[impType]):
            if  mat_r_by_Imp[compNum, impType] <= budgetGr[compNum]:
                tryToBidGr = True
                indBuyerGr = compNum 
                bidAmountGr = optBidGreedySP[impType][auxPos]
                if firstPrice:
                    bidAmountGr = optBidGreedyFP[impType][auxPos]
        ## Does ranking[impType] can be a scalar?
        # if np.isscalar(ranking[impType]):
        #     if  mat_r_by_Imp[ranking[impType], impType] <= budgetGr[ranking[impType]]:
        #         tryToBidGr = True
        #         indBuyerGr = ranking[impType] 
        #         bidAmountGr = optBidGreedySP[impType]
        #         if firstPrice:
        #             bidAmountGr = optBidGreedyFP[impType]
        # else:
        #     for auxPos,compNum in enumerate(ranking[impType]):
        #         if  mat_r_by_Imp[compNum, impType] <= budgetGr[compNum]:
        #             tryToBidGr = True
        #             indBuyerGr = compNum 
        #             bidAmountGr = optBidGreedySP[impType][auxPos]
        #             if firstPrice:
        #                 bidAmountGr = optBidGreedyFP[impType][auxPos]

        ## If tryToBidGr == True, we will bid in behalf of campaign 'indBuyerGr'
        ## the amount 'bidAmountGr'
        if (tryToBidGr):
            ## Save that we are bidding in behalf of 'indBuyerGr' for an impression of 
            ## type 'impType'
            cartBidsGr[indBuyerGr, impType]+=1
            ## We win the auction only if the value we are bidding is higher 
            ## than the market price observed by Ipinyou.
            if bidAmountGr>= mp_value:
                ## Impression Won.
                cartWonGr[indBuyerGr, impType]+=1
                costBidsGr[impType]-=mp_value
                profitGr[impType]-=mp_value
                # Now we need to check if the ad was clicked.
                probOfClick=mat_ctr[indBuyerGr, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType]+=1
                    payment=mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType]+=payment
                    profitGr[impType]+=payment
                    budgetGr[indBuyerGr]-=payment
    return [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
            revenueInd, profitInd, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
            cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]


# ## Simulation for the profit maximization, profit maximization + L2 and Greedy

# Here we want to run the experiment change the budget values to be [(1.0/32.0), (1.0/8.0), .25, 0.5, 1.0]
#  of the budgets used by Ipinyou. The iteration over the percentage budget values is done in 'for perc in perVector_m:'


def ExperIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_s, vector_ctr, alphasInd, num_itInd, alphasL2Ind,\
    num_itL2Ind, p_grad_TypeInd, p_grad_TypeL2Ind, init_lam, seq_vectorM, \
    UB_bid, adverPerImp, firstPrice, sim=100, shuffle =True):

    ext_s = vector_s[index_Imps]
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    global p_grad_Type
    vector_rctr = np.multiply(vector_r, vector_ctr)
    dictToRetInd = {}
    dictToRetL2Ind = {}
    dictToRetGr = {}
    
    for numOfVecM,vector_m in enumerate(seq_vectorM):
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        print("Number of Vector M: "+str(numOfVecM))
        dictToRetInd[numOfVecM]=[]
        dictToRetL2Ind[numOfVecM]=[]
        dictToRetGr[numOfVecM]=[]
        p_grad_Type=p_grad_TypeInd
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjInd(\
        init_lam, num_itInd, alphasInd, vector_r, vector_ctr, vector_rctr, vector_s, ext_s,\
        vector_m, num_impressions, numCampaigns, num_edges, index_sizeCamps, index_Imps,\
        (num_itInd-1), adverPerImp, p_grad_Type,UB_bid, firstPrice)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsInd=OptimalBids(ext_LamFinal, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidsInd, UB_bid, num_edges, index_Imps, adverPerImp, firstPrice)
        xInd=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctr, num_edges, numCampaigns, \
            num_impressions, index_Imps, index_sizeCamps)
        
        p_grad_Type=p_grad_TypeL2Ind
        tau=np.power(vector_m, -1)
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg] = \
        SubgrAlgSavPrimDualObjFn_L2Ind(init_lam, num_itL2Ind, alphasL2Ind, vector_r, vector_ctr, \
        vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
        index_sizeCamps, index_Imps, (num_itInd-1), adverPerImp, tau, True, p_grad_Type,\
        UB_bid, firstPrice)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidsL2Ind, UB_bid, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2Ind=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctr, num_edges, numCampaigns, \
            num_impressions, index_Imps, index_sizeCamps)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('Finished running the Primal-Dual Algorithms')
        s=time.time()
        for i in range(sim): 
            ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                vector_s, adverPerImp, shuffle)
            [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
            revenueInd, profitInd, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
            cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]=\
            RunIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
            index_sizeCamps,  vector_r, vector_m, vector_ctr, \
            vector_rctr, bidsInd, xInd, bidsL2Ind, \
            xL2Ind, tau, UB_bid, adverPerImp, ImpInOrder, MPInOrder, firstPrice)
            dictToRetInd[numOfVecM].append([budgetInd, cartBidsInd, cartWonInd, \
                cartClickedInd, costBidsInd, revenueInd, profitInd])
            dictToRetL2Ind[numOfVecM].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])
            dictToRetGr[numOfVecM].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, \
            costBidsGr, revenueGr, profitGr])

#             print("Profit Ind: "+str(np.sum(profitInd)))
#             print("Profit Gr: "+str(np.sum(profitGr)))
#             print("Ratio of Profits: "+str(np.sum(profitInd)/np.sum(profitGr)))
        print("Running the Whole Simulation Code took: "+str(time.time()-s)+"secs.")
    return [dictToRetInd, dictToRetL2Ind, dictToRetGr]

## For the Pareto Experiment we need to Run only the L2+Indicator for several values of \tau a
## number of simulations. We also need to run the greedy method 

def RunSimOnlyGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_rctr, UB_bid,\
    adverPerImp, ImpInOrder, MPInOrder, firstPrice):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    [_, _, _, _, _, _, _,  budgetGr, cartBidsGr, cartWonGr, cartClickedGr,\
        costBidsGr, revenueGr, profitGr, mat_r_by_Imp, mat_ctr, _, ranking,\
        optBidGreedySP, optBidGreedyFP, _, _, _] = \
        CreateDataForSimulation(np.zeros(num_edges), np.zeros(num_edges),\
        numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps,\
        vector_r, vector_ctr, vector_rctr, vector_m,  UB_bid, adverPerImp)
    
    ## Now we simulate
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse=np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs=allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value=MPInOrder[i]
        
        ### Now we update the Greedy Policy    
        ## The greedy heuristic bids for the campaign which stills have remaining
        ## budget and from thos bid for the one with highest r times ctr. 
        ## The previous is true as Ipinyou assumes second price auctions.
        max_rctr=0.0
        indBuyerGr=-1
        bidAmountGr=0.0
        tryToBidGr = False
        for auxPos,compNum in enumerate(ranking[impType]):
            if  mat_r_by_Imp[compNum, impType] <= budgetGr[compNum]:
                tryToBidGr = True
                indBuyerGr = compNum 
                bidAmountGr = optBidGreedySP[impType][auxPos]
                if firstPrice:
                    bidAmountGr = optBidGreedyFP[impType][auxPos]
        ## If tryToBidGr == True, we will bid in behalf of campaign 'indBuyerGr'
        ## the amount 'bidAmountGr'
        if (tryToBidGr):
            ## Save that we are bidding in behalf of 'indBuyerGr' for an impression of 
            ## type 'impType'
            cartBidsGr[indBuyerGr, impType]+=1
            ## We win the auction only if the value we are bidding is higher 
            ## than the market price observed by Ipinyou.
            if bidAmountGr>= mp_value:
                ## Impression Won.
                cartWonGr[indBuyerGr, impType]+=1
                costBidsGr[impType]-=mp_value
                profitGr[impType]-=mp_value
                # Now we need to check if the ad was clicked.
                probOfClick=mat_ctr[indBuyerGr, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType]+=1
                    payment=mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType]+=payment
                    profitGr[impType]+=payment
                    budgetGr[indBuyerGr]-=payment
    return [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]


def RunOneSimL2Ind(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_r, vector_m, vector_ctr, vector_rctr,\
    bidsL2Ind, xL2Ind, tau, UB_bid, adverPerImp, ImpInOrder, MPInOrder, firstPrice):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind,  _, _, _, _, _, _, _, mat_r_by_Imp, mat_ctr,\
        mat_rctr_by_Imp, _, _, _, mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind]=\
        CreateDataForSimulation(bidsL2Ind, xL2Ind, numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_ctr, vector_rctr, \
        vector_m,  UB_bid, adverPerImp)
    
    ## Now we simulate
    campaignsArange=np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse=np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    ## We read the test log in irder of how the impressions type appear.
    for i,impType in enumerate(ImpInOrder):
        ## Update the methods.
        unifs=allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value=MPInOrder[i]
        
        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind=0
        tryToBidL2Ind=False
        bidAmountL2Ind=0.0
        if unifs[0] <= probBidL2Ind[impType]:
            indInterested=[False]*numCampaigns
            bidUsingL2Ind=False
            for j in range(numCampaigns):
                if mat_r_by_Imp[j, impType]<=budgetL2Ind[j] and mat_x_by_ImpL2Ind[j, impType]>0:
                    indInterested[j]=True
                    bidUsingL2Ind=True
            if bidUsingL2Ind: 
                posInt=campaignsArange[indInterested]
                condProbInterested=mat_x_by_ImpL2Ind[:, impType][posInt]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                numInterest=len(condProbInterested)
                auxPosForindBuyerL2Ind=numInterest-1
                for z in range(numInterest):
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        auxPosForindBuyerL2Ind=z
                        z+=numInterest
                indBuyerL2Ind=posInt[auxPosForindBuyerL2Ind]
                tryToBidL2Ind=True
                bidAmountL2Ind=mat_bid_by_ImpL2Ind[indBuyerL2Ind, impType]
        if(tryToBidL2Ind):
            cartBidsL2Ind[indBuyerL2Ind, impType]+=1
            if bidAmountL2Ind>= mp_value:
                ## Impression Won.
                cartWonL2Ind[indBuyerL2Ind, impType]+=1
                costBidsL2Ind[impType]-=mp_value
                profitL2Ind[impType]-=mp_value
                # Now we need to check if the ad was clicked.
                probOfClick=mat_ctr[indBuyerL2Ind, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2Ind[indBuyerL2Ind, impType]+=1
                    payment=mat_r_by_Imp[indBuyerL2Ind, impType]
                    revenueL2Ind[impType]+=payment
                    profitL2Ind[impType]+=payment
                    budgetL2Ind[indBuyerL2Ind]-=payment
    return [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind]


def RunOnlyGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_s, vector_r, vector_m, vector_ctr, UB_bid,\
    adverPerImp, sim, firstPrice, vecOfSeeds = None,\
    shuffle = True):

    vector_rctr = np.multiply(vector_r, vector_ctr)
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
        [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind] = RunSimOnlyGreedy(numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, vector_r, vector_m, vector_ctr,\
        vector_rctr, UB_bid,adverPerImp, ImpInOrder, MPInOrder, firstPrice)
        toRetGreedy.append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind])
    return toRetGreedy

def ExpPareto(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, vector_s, vector_r, vector_m, vector_ctr, UB_bid,\
    adverPerImp, alphasL2Ind, num_itL2Ind, p_grad_TypeL2Ind,\
    multipliers, init_lam, sim, firstPrice, vecOfSeeds = None, shuffle = True):

    vector_rctr = np.multiply(vector_r, vector_ctr)
    ext_s = vector_s[index_Imps]
    vector_m = vector_m
    ext_s = vector_s[index_Imps]
    global p_grad_Type
    dictToRetL2Ind = {}

    for i,perc in enumerate(multipliers):
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        print("Tau multiplier: "+str(perc))
        dictToRetL2Ind[perc] = []
        p_grad_Type=p_grad_TypeL2Ind
        tau=np.power(vector_m, -1)*perc
        [_, _, _, _, _, _, _, dual_varsAvg] = \
        SubgrAlgSavPrimDualObjFn_L2Ind(init_lam, num_itL2Ind, alphasL2Ind, vector_r, vector_ctr, \
        vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
        index_sizeCamps, index_Imps, (num_itL2Ind-1), adverPerImp, tau, True, p_grad_Type,\
        UB_bid, firstPrice)
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp)
        [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidsL2Ind, UB_bid, num_edges, index_Imps, adverPerImp, firstPrice)
        xL2Ind=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctr, num_edges, numCampaigns, \
            num_impressions, index_Imps, index_sizeCamps)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('Finished running the Primal-Dual Algorithms')
        s=time.time()
        for i in range(sim): 
            ImpInOrder, MPInOrder = 0, 0
            if vecOfSeeds is not None:
                ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                    vector_s, adverPerImp, shuffle, vecOfSeeds[i])
            else:
                ImpInOrder, MPInOrder = CreateImpAndMPInOrder(num_impressions, numCampaigns,\
                    vector_s, adverPerImp, shuffle)
            [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind]=\
            RunOneSimL2Ind(numCampaigns, num_impressions, num_edges, index_Imps, \
                index_sizeCamps, vector_r, vector_m, vector_ctr, vector_rctr,\
                bidsL2Ind, xL2Ind, tau, UB_bid, adverPerImp, ImpInOrder, MPInOrder,\
                firstPrice)
            dictToRetL2Ind[perc].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])
        print('Running the Multiplier '+str(perc) +' took '+str(time.time()-s)+"secs.")
    return dictToRetL2Ind