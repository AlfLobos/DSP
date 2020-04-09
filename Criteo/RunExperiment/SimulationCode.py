import numpy as np
from gurobipy import *
import time
import os
import sys
from Utilities import CalcRhoAndBetaVectors
from UtilitiesOptimization import CalculateLPGurobi, CalculateQuadGurobi,\
    SubgrAlgSavPrimDualObjInd, SubgrAlgSavPrimDualObjFn_L2Ind, ExtendSizeCamps, OptimalBids, OptimalX

#
#
## Simulation Code

## To make the simulation code faster and easier to read (in particular the greedy heuristic)
## we would like to change the vector of click-through rates 'vector_ctr', vector of 
## revenues 'vector_q', of revenue times the click trhough rate 'vector_qctr', and others 
## into a matrix of size number of campaigns times number of impressions.
## That's done in the following function.
def CreateMatR_ctr_Rctr_Rhoctr(numCampaigns, num_impressions, num_edges, \
    index_Imps, index_sizeCamps, vector_q, vector_ctr, \
    vector_qctr, PPFTable, numericBeta):
    ## mat_r_by_Imp is a matrix in which each column 'i'
    ## represents the valuations of all campaigns in order for 
    ## an impression of type 'i'. If the campaign is not interested 
    ## in the impression a zero value is entered in that position.
    mat_r_by_Imp=np.zeros((numCampaigns, num_impressions))
    mat_ctr=np.zeros((numCampaigns, num_impressions))
    mat_rctr_by_Imp=np.zeros((numCampaigns, num_impressions))
    mat_rhoctr_by_Imp=np.zeros((numCampaigns, num_impressions))
    mat_rctrBetarho_by_Imp=np.zeros((numCampaigns, num_impressions))
    [rho_rctr, beta_rctr]=CalcRhoAndBetaVectors(vector_qctr, num_edges, \
        index_Imps, PPFTable, numericBeta) 
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
                    mat_r_by_Imp[j, i]=vector_q[pos]
                    mat_ctr[j, i]=vector_ctr[pos]
                    mat_rctr_by_Imp[j, i]=vector_qctr[pos]
                    mat_rhoctr_by_Imp[j, i]=rho_rctr[pos]
                    mat_rctrBetarho_by_Imp[j, i] =(vector_qctr[pos]-\
                    beta_rctr[pos])*rho_rctr[pos]
                    if(aux<sizeIndexes-1):
                        aux+=1
                        pos=indexes[aux]
                    else:
                        # No more campaigns use that impression
                        pos=num_edges
                count+=impInCamp
    return [mat_r_by_Imp, mat_ctr, mat_rctr_by_Imp, mat_rhoctr_by_Imp, \
            mat_rctrBetarho_by_Imp]

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
    num_impressions, num_edges, index_Imps, index_sizeCamps, vector_q, \
    vector_ctr, vector_qctr, vector_m, PPFTable, numericBeta):
    [budgetLR, cartBidsLR, cartWonLR, cartClickedLR, costBidsLR, revenueLR, \
        profitLR]=CreateIndicatorSimulation(numCampaigns, \
        num_impressions, vector_m)
    [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, \
        profitGr]=CreateIndicatorSimulation(numCampaigns, \
        num_impressions, vector_m)
    [mat_r_by_Imp, mat_ctrTest, mat_rctr_by_Imp, mat_rhoctr_by_Imp, mat_rctrBetarho_by_Imp]=\
        CreateMatR_ctr_Rctr_Rhoctr(numCampaigns, num_impressions, num_edges, \
        index_Imps, index_sizeCamps, vector_q, vector_ctr, vector_qctr, PPFTable, numericBeta)
    [mat_bid_by_ImpLR, mat_x_by_ImpLR]=CreateMatrixBidAndX(numCampaigns, \
        num_impressions, num_edges, index_Imps, index_sizeCamps, \
        bidFound, xFound)
    probBidLR=CreateProbOfBidding(mat_x_by_ImpLR)
    return [budgetLR, cartBidsLR, cartWonLR, cartClickedLR, costBidsLR, \
        revenueLR, profitLR, budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, \
        mat_ctrTest, mat_rctr_by_Imp, mat_rhoctr_by_Imp, mat_rctrBetarho_by_Imp, \
        mat_bid_by_ImpLR, mat_x_by_ImpLR, probBidLR]

# Comments About the Implementation
# - We win an auction only if the bid_amount is higher than the market price that appear in the Ipinyou Log. 
# In case of winning the auction we then need to check if a click occurs. We update the revenue, profit, 
# budget, cartWon, costBids, and cartClicked accordingly. 
# - For the indicator and indicator+$\ell_2$ case we only need to check the allocation vector to decide 
# the campaign to bid in behalf of (allocation vector that comes from running other primal-dual procedure).

# Simulation code for Indicator, Indicator + $\ell_2$, and Greedy

def RunIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, vector_rctrTrain, \
    vector_ctrTest, vector_rctrTest, bidsInd, xInd, bidsL2Ind, xL2Ind, tau, ImpInOrder, MPInOrder,\
    impNames, listCampPerImp):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    
    [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
        revenueInd, profitInd, _, _, _, _, _, _, _, _, _, mat_rctr_by_ImpTrain, _, _, \
        mat_bid_by_ImpInd, mat_x_by_ImpInd, probBidInd]=CreateDataForSimulation(bidsInd, \
    xInd, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTrain, vector_rctrTrain, vector_m, PPFTable, numericBeta)
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind, budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, \
        mat_ctrTest, _, _, _, \
        mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind]=CreateDataForSimulation(bidsL2Ind, \
    xL2Ind, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTest, vector_rctrTest, vector_m, PPFTable, numericBeta)
    
    ## Now we simulate
    # campaignsArange=np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    # uniOnline=False
    ## We read the test log in irder of how the impressions type appear.
    for i,clusterId in enumerate(ImpInOrder):
        impType=impNames.index(clusterId)
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
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            bidUsingInd=False
            # print('budgetInd[listCampPerImp[impType]]: '+str(budgetInd[listCampPerImp[impType]]))
            # aux53 =(mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetInd[listCampPerImp[impType]])
            indInterested = (mat_x_by_ImpInd[listCampPerImp[impType],impType]>0) *\
                (mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetInd[listCampPerImp[impType]])
            if np.sum(indInterested) >0:
                bidUsingInd= True
            if bidUsingInd: 
                ## There is at least one campaign that wants to bid.
                posInt=listCampPerImp[impType][indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpInd[posInt, impType]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerInd = numInterest-1
                z = 0
                while z<numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerInd=z
                        z+=numInterest
                    z += 1
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
                probOfClick=mat_ctrTest[indBuyerInd, impType]
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
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            bidUsingL2Ind=False
            indInterested =\
                (mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetL2Ind[listCampPerImp[impType]]) * \
                    (mat_x_by_ImpL2Ind[listCampPerImp[impType],impType]>0)
            if np.sum(indInterested) >0:
                bidUsingL2Ind= True
            if bidUsingL2Ind: 
                ## There is at least one campaign that wants to bid.
                posInt=listCampPerImp[impType][indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpL2Ind[posInt, impType]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2Ind = numInterest-1
                z = 0
                while z <numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2Ind=z
                        z+=numInterest
                    z += 1
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
                probOfClick=mat_ctrTest[indBuyerL2Ind, impType]
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
        tryToBidGr=False
        indInterested =\
            mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetGr[listCampPerImp[impType]]
        if np.sum(indInterested) > 0:
            posInt=listCampPerImp[impType][indInterested]
            indBuyerGr = posInt[np.argmax(mat_rctr_by_ImpTrain[posInt,impType])]
            bidAmountGr=mat_rctr_by_ImpTrain[indBuyerGr, impType]
            tryToBidGr=True
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
                probOfClick=mat_ctrTest[indBuyerGr, impType]
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

def RunInd_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, vector_rctrTrain, \
    vector_ctrTest, vector_rctrTest, bidsInd, xInd, bidsL2, xL2, bidsL2Ind, xL2Ind, tau,\
    ImpInOrder, MPInOrder, impNames, listCampPerImp):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.
    
    [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
        revenueInd, profitInd, _, _, _, _, _, _, _, _, _, mat_rctr_by_ImpTrain, _, _, \
        mat_bid_by_ImpInd, mat_x_by_ImpInd, probBidInd]=CreateDataForSimulation(bidsInd, \
    xInd, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTrain, vector_rctrTrain, vector_m, PPFTable, numericBeta)
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind, budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, \
        mat_ctrTest, _, _, _, \
        mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind]=CreateDataForSimulation(bidsL2Ind, \
    xL2Ind, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTest, vector_rctrTest, vector_m, PPFTable, numericBeta)

    [budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, \
        revenueL2, profitL2, _, _, _, _, _, _, _, _, _, _, _, _, \
        mat_bid_by_ImpL2, mat_x_by_ImpL2, probBidL2]=CreateDataForSimulation(bidsL2, \
    xL2, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTest, vector_rctrTest, vector_m, PPFTable, numericBeta)
    
    ## Now we simulate
    # campaignsArange=np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    # uniOnline=False
    ## We read the test log in irder of how the impressions type appear.
    for i,clusterId in enumerate(ImpInOrder):
        impType=impNames.index(clusterId)
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
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            bidUsingInd=False
            # print('budgetInd[listCampPerImp[impType]]: '+str(budgetInd[listCampPerImp[impType]]))
            # aux53 =(mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetInd[listCampPerImp[impType]])
            indInterested = (mat_x_by_ImpInd[listCampPerImp[impType],impType]>0) *\
                (mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetInd[listCampPerImp[impType]])
            if np.sum(indInterested) >0:
                bidUsingInd= True
            if bidUsingInd: 
                ## There is at least one campaign that wants to bid.
                posInt=listCampPerImp[impType][indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpInd[posInt, impType]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerInd = numInterest-1
                z = 0
                while z<numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerInd=z
                        z+=numInterest
                    z += 1
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
                probOfClick=mat_ctrTest[indBuyerInd, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign (i.e. DSP wins money).
                    cartClickedInd[indBuyerInd, impType]+=1
                    payment=mat_r_by_Imp[indBuyerInd, impType]
                    revenueInd[impType]+=payment
                    profitInd[impType]+=payment
                    budgetInd[indBuyerInd]-=payment
        ## Update L2 (Same code as done before for the pure indicator case)
        indBuyerL2=0
        tryToBidL2=False
        bidAmountL2=0.0
        if unifs[0] <= probBidL2[impType]:
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            bidUsingL2=False
            indInterested =\
                (mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetL2[listCampPerImp[impType]]) * \
                    (mat_x_by_ImpL2[listCampPerImp[impType],impType]>0)
            if np.sum(indInterested) >0:
                bidUsingL2= True
            if bidUsingL2: 
                ## There is at least one campaign that wants to bid.
                posInt=listCampPerImp[impType][indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpL2[posInt, impType]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2 = numInterest-1
                z = 0
                while z <numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2=z
                        z+=numInterest
                    z += 1
                indBuyerL2=posInt[auxPosForindBuyerL2]
                tryToBidL2=True
                bidAmountL2=mat_bid_by_ImpL2[indBuyerL2, impType]
        if(tryToBidL2):
            cartBidsL2[indBuyerL2, impType]+=1
            if bidAmountL2>= mp_value:
                ## Impression Won.
                cartWonL2[indBuyerL2, impType]+=1
                costBidsL2[impType]-=mp_value
                profitL2[impType]-=mp_value
                # Now we need to check if the ad was clicked.
                probOfClick=mat_ctrTest[indBuyerL2, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2[indBuyerL2, impType]+=1
                    payment=mat_r_by_Imp[indBuyerL2, impType]
                    revenueL2[impType]+=payment
                    profitL2[impType]+=payment
                    budgetL2[indBuyerL2]-=payment

        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind=0
        tryToBidL2Ind=False
        bidAmountL2Ind=0.0
        if unifs[0] <= probBidL2Ind[impType]:
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            bidUsingL2Ind=False
            indInterested =\
                (mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetL2Ind[listCampPerImp[impType]]) * \
                    (mat_x_by_ImpL2Ind[listCampPerImp[impType],impType]>0)
            if np.sum(indInterested) >0:
                bidUsingL2Ind= True
            if bidUsingL2Ind: 
                ## There is at least one campaign that wants to bid.
                posInt=listCampPerImp[impType][indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpL2Ind[posInt, impType]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2Ind = numInterest-1
                z = 0
                while z < numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2Ind=z
                        z+=numInterest
                    z += 1
                indBuyerL2Ind=posInt[auxPosForindBuyerL2Ind]
                tryToBidL2Ind=True
                bidAmountL2Ind = mat_bid_by_ImpL2Ind[indBuyerL2Ind, impType]
        if(tryToBidL2Ind):
            cartBidsL2Ind[indBuyerL2Ind, impType]+=1
            if bidAmountL2Ind>= mp_value:
                ## Impression Won.
                cartWonL2Ind[indBuyerL2Ind, impType]+=1
                costBidsL2Ind[impType]-=mp_value
                profitL2Ind[impType]-=mp_value
                # Now we need to check if the ad was clicked.
                probOfClick=mat_ctrTest[indBuyerL2Ind, impType]
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
        tryToBidGr=False
        indInterested =\
            mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetGr[listCampPerImp[impType]]
        if np.sum(indInterested) > 0:
            posInt=listCampPerImp[impType][indInterested]
            indBuyerGr = posInt[np.argmax(mat_rctr_by_ImpTrain[posInt,impType])]
            bidAmountGr = mat_rctr_by_ImpTrain[indBuyerGr, impType]
            tryToBidGr = True
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
                probOfClick=mat_ctrTest[indBuyerGr, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType]+=1
                    payment=mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType]+=payment
                    profitGr[impType]+=payment
                    budgetGr[indBuyerGr]-=payment
    return [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, revenueInd, \
        profitInd, budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, revenueL2, \
        profitL2, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind, budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr,\
        revenueGr, profitGr]

# ## Simulation for the profit maximization, profit maximization + L2 and Greedy

# Here we want to run the experiment change the budget values to be [(1.0/32.0), (1.0/8.0), .25, 0.5, 1.0]
#  of the budgets used by Ipinyou. The iteration over the percentage budget values is done in 'for perc in perVector_m:'

def ExperIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrigTest, \
    vector_sTest, vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, impNames, \
    alphasInd, num_itInd, alphasL2Ind, num_itL2Ind, p_grad_TypeInd, p_grad_TypeL2Ind, \
    tau, init_lam, listCampPerImp, perVector_m=[(1.0/32.0), (1.0/8.0), .25, 0.5, 1.0], sim=100):

    print('Starting ExperIndL2IndAndGreedy')
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    global p_grad_Type
    vector_rctrTrain=np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest=np.multiply(vector_q, vector_ctrTest)
    dictToRetInd={}
    dictToRetL2Ind={}
    dictToRetGr={}
    
    for perc in perVector_m:
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        print("Percentage: "+str(perc))
        vector_m = vector_mOrigTest*perc
        vector_s = vector_sTest
        ext_s = vector_s[index_Imps]
        dictToRetInd[perc]=[]
        dictToRetL2Ind[perc]=[]
        dictToRetGr[perc]=[]
        p_grad_Type=p_grad_TypeInd

        print('About to Run the SubgrAlgSavPrimDualObjInd using '+str(num_itInd)+' iterations')
        initTime =time.time()
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjInd(\
        init_lam, num_itInd, alphasInd, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
        vector_m, num_impressions, numCampaigns, num_edges, \
        PPFTable, numericBeta, index_sizeCamps, index_Imps, (num_itInd-1), p_grad_Type)
        print("Took: "+str( time.time()-initTime)+' seconds')
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsInd=OptimalBids(ext_LamFinal, vector_rctrTrain)
        [rho_eval_Ind, beta_eval_Ind]=CalcRhoAndBetaVectors(bidsInd, num_edges, index_Imps, PPFTable, numericBeta)

        xInd = CalculateLPGurobi(rho_eval_Ind, beta_eval_Ind, vector_rctrTrain, vector_m, \
        ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
        index_sizeCamps)
        # xInd=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
        #     num_impressions, index_Imps, index_sizeCamps)

        
        print('')
        print('')
        print('About to Run the SubgrAlgSavPrimDualObjFn_L2Ind using '+str(num_itL2Ind)+' iterations')
        initTime =time.time()
        p_grad_Type=p_grad_TypeL2Ind
        tau=np.power(vector_m, -1)
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjFn_L2Ind(\
        init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
        vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
        index_Imps, (num_itL2Ind-1), p_grad_Type, tau, True)
        print("Took: "+str( time.time()-initTime)+' seconds')
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctrTrain)
        [rho_eval_L2Ind, beta_eval_L2Ind]=CalcRhoAndBetaVectors(bidsL2Ind, num_edges, index_Imps, PPFTable, numericBeta)

        xL2Ind = CalculateQuadGurobi(rho_eval_L2Ind, beta_eval_L2Ind, vector_rctrTrain, vector_m, ext_s, \
            num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau)

        # xL2Ind=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
        #     num_impressions, index_Imps, index_sizeCamps)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('')
        print('')
        print('Finished running the Primal-Dual Algorithms')
        print('Starting RunIndL2IndAndGreedy using '+str(perc)+' percentage of the Test budgets')
        initTime =time.time()
        for i in range(sim): 
            [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
            revenueInd, profitInd, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
            cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]=\
            RunIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
            index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, \
            vector_rctrTrain, vector_ctrTest, vector_rctrTest, bidsInd, xInd, bidsL2Ind, \
            xL2Ind, tau, ImpInOrder, MPInOrder, impNames,listCampPerImp)
            dictToRetInd[perc].append([budgetInd, cartBidsInd, cartWonInd, \
                cartClickedInd, costBidsInd, revenueInd, profitInd])
            dictToRetL2Ind[perc].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])
            dictToRetGr[perc].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, \
            costBidsGr, revenueGr, profitGr])
#             print("Profit Ind: "+str(np.sum(profitInd)))
#             print("Profit Gr: "+str(np.sum(profitGr)))
#             print("Ratio of Profits: "+str(np.sum(profitInd)/np.sum(profitGr)))
        print("Took: "+str(time.time()-initTime)+' seconds')
    return [dictToRetInd, dictToRetL2Ind, dictToRetGr]

def Exper_Ind_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps,\
    PPFTable, numericBeta, vector_q, vector_mOrigTest, vector_sTest, vector_ctrTrain, vector_ctrTest, \
    ImpInOrder, MPInOrder, impNames, alphasInd, num_itInd, alphasL2, num_itL2, alphasL2Ind, num_itL2Ind,\
    p_grad_TypeInd, p_grad_TypeL2, p_grad_TypeL2Ind, tau, init_lam, listCampPerImp,\
    perVector_m=[(1.0/32.0), (1.0/8.0), .25, 0.5, 1.0], sim=100):

    print('Starting Exper_Ind_L2_L2Ind_Greedy')
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    global p_grad_Type
    vector_rctrTrain=np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest=np.multiply(vector_q, vector_ctrTest)
    dictToRetInd={}
    dictToRetL2={}
    dictToRetL2Ind={}
    dictToRetGr={}
    
    for perc in perVector_m:
        ## We first run the primal dual-subgradient method using the pure indicator utility function first
        ## and then the indicator plus l2 penalization.
        print("Percentage: "+str(perc))
        vector_m = vector_mOrigTest*perc
        vector_s = vector_sTest
        ext_s = vector_s[index_Imps]
        dictToRetInd[perc] = []
        dictToRetL2[perc] = []
        dictToRetL2Ind[perc] = []
        dictToRetGr[perc] = []
        p_grad_Type=p_grad_TypeInd

        print('About to Run the SubgrAlgSavPrimDualObjInd using '+str(num_itInd)+' iterations')
        initTime =time.time()
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjInd(\
        init_lam, num_itInd, alphasInd, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
        vector_m, num_impressions, numCampaigns, num_edges, \
        PPFTable, numericBeta, index_sizeCamps, index_Imps, (num_itInd-1), p_grad_Type)
        print("Took: "+str( time.time()-initTime)+' seconds')
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsInd=OptimalBids(ext_LamFinal, vector_rctrTrain)
        [rho_eval_Ind, beta_eval_Ind]=CalcRhoAndBetaVectors(bidsInd, num_edges, index_Imps, PPFTable, numericBeta)

        xInd = CalculateLPGurobi(rho_eval_Ind, beta_eval_Ind, vector_rctrTrain, vector_m, \
        ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
        index_sizeCamps)
        # xInd=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
        #     num_impressions, index_Imps, index_sizeCamps)

        print('')
        print('')
        print('About to Run the SubgrAlgSavPrimDualObjFn_L2Ind using '+str(num_itL2Ind)+' iterations without Indicator')
        initTime =time.time()
        p_grad_Type=p_grad_TypeL2
        tau=np.power(vector_m, -1)
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjFn_L2Ind(\
        init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
        vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
        index_Imps, (num_itL2Ind-1), p_grad_Type, tau, False)
        print("Took: "+str( time.time()-initTime)+' seconds')
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2=OptimalBids(ext_LamFinal, vector_rctrTrain)
        [rho_eval_L2, beta_eval_L2]=CalcRhoAndBetaVectors(bidsL2, num_edges, index_Imps, PPFTable, numericBeta)

        xL2 = CalculateQuadGurobi(rho_eval_L2, beta_eval_L2, vector_rctrTrain, vector_m, ext_s, \
            num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau, addIndicator = False)
        
        print('')
        print('')
        print('About to Run the SubgrAlgSavPrimDualObjFn_L2Ind using '+str(num_itL2Ind)+' iterations')
        initTime =time.time()
        p_grad_Type=p_grad_TypeL2Ind
        tau=np.power(vector_m, -1)
        [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
        primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjFn_L2Ind(\
        init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
        vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
        index_Imps, (num_itL2Ind-1), p_grad_Type, tau, True)
        print("Took: "+str( time.time()-initTime)+' seconds')
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctrTrain)
        [rho_eval_L2Ind, beta_eval_L2Ind]=CalcRhoAndBetaVectors(bidsL2Ind, num_edges, index_Imps, PPFTable, numericBeta)

        xL2Ind = CalculateQuadGurobi(rho_eval_L2Ind, beta_eval_L2Ind, vector_rctrTrain, vector_m, ext_s, \
            num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau, addIndicator = True)

        # xL2Ind=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
        #     num_impressions, index_Imps, index_sizeCamps)
        
        
        ## Now that we have run the primal-dual subgradient methods we run simulations of 
        ## how they would perform in the test log as explained in the paper. The nuber of simulations to
        ## run is equal to the parameter sim. 
        print('')
        print('')
        print('Finished running the Primal-Dual Algorithms')
        print('Starting RunInd_L2_L2Ind_Greedy using '+str(perc)+' percentage of the Test budgets')
        initTime =time.time()
        for i in range(sim): 
            [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, revenueInd, \
            profitInd, budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, revenueL2, \
            profitL2, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
            revenueL2Ind, profitL2Ind, budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr,\
            revenueGr, profitGr] = RunInd_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
            index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, \
            vector_rctrTrain, vector_ctrTest, vector_rctrTest, bidsInd, xInd, bidsL2, xL2, bidsL2Ind, \
            xL2Ind, tau, ImpInOrder, MPInOrder, impNames,listCampPerImp)

            dictToRetInd[perc].append([budgetInd, cartBidsInd, cartWonInd, \
                cartClickedInd, costBidsInd, revenueInd, profitInd])

            dictToRetL2[perc].append([budgetL2, cartBidsL2, cartWonL2, \
                cartClickedL2, costBidsL2, revenueL2, profitL2])

            dictToRetL2Ind[perc].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])

            dictToRetGr[perc].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, \
            costBidsGr, revenueGr, profitGr])
        print("Took: "+str(time.time()-initTime)+' seconds')
    return [dictToRetInd, dictToRetL2, dictToRetL2Ind, dictToRetGr]

def ExperIndL2IndAndGreedyOnePerc(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrigTest, \
    vector_sTest, vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, impNames, \
    alphasInd, num_itInd, alphasL2Ind, num_itL2Ind, p_grad_TypeInd, p_grad_TypeL2Ind, \
    init_lam, listCampPerImp, perc, sim, seeds):

    print('Starting ExperIndL2IndAndGreedy')
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    np.random.seed(12345)
    global p_grad_Type
    vector_rctrTrain=np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest=np.multiply(vector_q, vector_ctrTest)
    dictToRetInd = {}
    dictToRetL2Ind = {}
    dictToRetGr = {}
    
    ## We first run the primal dual-subgradient method using the pure indicator utility function first
    ## and then the indicator plus l2 penalization.
    print('')
    print("Percentage: "+str(perc)+ ', process id: '+str(os.getpid()))
    vector_m = vector_mOrigTest[:]*perc
    vector_s = vector_sTest[:]
    ext_s = vector_s[index_Imps]
    dictToRetInd[perc]=[]
    dictToRetL2Ind[perc]=[]
    dictToRetGr[perc]=[]
    p_grad_Type=p_grad_TypeInd

    print('About to Run the SubgrAlgSavPrimDualObjInd using '+str(num_itInd)+' iterations, '+'process id: '+str(os.getpid()))
    initTime =time.time()
    [_, _, _, dual_vars, _, _, _, dual_varsAvg] = SubgrAlgSavPrimDualObjInd(\
    init_lam, num_itInd, alphasInd, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
    vector_m, num_impressions, numCampaigns, num_edges, \
    PPFTable, numericBeta, index_sizeCamps, index_Imps, (num_itInd-1), p_grad_Type)
    print("Took: "+str( time.time()-initTime)+' seconds, '+'process id: '+str(os.getpid()))
    
    #print("Duality Gap Last Iteration")
    #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
    lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
    ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
    bidsInd=OptimalBids(ext_LamFinal, vector_rctrTrain)
    [rho_eval_Ind, beta_eval_Ind]=CalcRhoAndBetaVectors(bidsInd, num_edges, index_Imps, PPFTable, numericBeta)

    xInd = CalculateLPGurobi(rho_eval_Ind, beta_eval_Ind, vector_rctrTrain, vector_m, \
    ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
    index_sizeCamps)
    # xInd=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
    #     num_impressions, index_Imps, index_sizeCamps)

    print('')
    print('About to Run the SubgrAlgSavPrimDualObjFn_L2Ind using '+str(num_itL2Ind)+' iterations, '+'process id: '+str(os.getpid()))
    initTime =time.time()
    p_grad_Type=p_grad_TypeL2Ind
    tau = np.power(vector_m, -1)
    [_, _, _, dual_vars, _, _, _, dual_varsAvg] = SubgrAlgSavPrimDualObjFn_L2Ind(\
    init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
    vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
    index_Imps, (num_itL2Ind-1), p_grad_Type, tau, True)
    print("Took: "+str( time.time()-initTime)+' seconds, '+'process id: '+str(os.getpid()))
    
    #print("Duality Gap Last Iteration")
    #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
    lamFinal=dual_varsAvg[len(dual_varsAvg) - 1]
    ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
    bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctrTrain)
    [rho_eval_L2Ind, beta_eval_L2Ind] = CalcRhoAndBetaVectors(bidsL2Ind, num_edges, index_Imps, PPFTable, numericBeta)

    xL2Ind = CalculateQuadGurobi(rho_eval_L2Ind, beta_eval_L2Ind, vector_rctrTrain, vector_m, ext_s, \
        num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau)
    
    ## Now that we have run the primal-dual subgradient methods we run simulations of 
    ## how they would perform in the test log as explained in the paper. The nuber of simulations to
    ## run is equal to the parameter sim. 
    print('Finished running the Primal-Dual Algorithms, '+'process id: '+str(os.getpid()))
    print('About to Run the RunIndL2IndAndGreedy using '+str(perc)+' percentage of the Test budgets, '+'process id: '+str(os.getpid()))
    initTime = time.time()
    for i in range(sim): 
        np.random.seed(seeds[i])
        [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
        revenueInd, profitInd, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
        cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind, budgetGr, \
        cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]=\
        RunIndL2IndAndGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
        index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, \
        vector_rctrTrain, vector_ctrTest, vector_rctrTest, bidsInd, xInd, bidsL2Ind, \
        xL2Ind, tau, ImpInOrder, MPInOrder, impNames,listCampPerImp)
        dictToRetInd[perc].append([budgetInd, cartBidsInd, cartWonInd, \
            cartClickedInd, costBidsInd, revenueInd, profitInd])
        dictToRetL2Ind[perc].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])
        dictToRetGr[perc].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, \
        costBidsGr, revenueGr, profitGr])
#             print("Profit Ind: "+str(np.sum(profitInd)))
#             print("Profit Gr: "+str(np.sum(profitGr)))
#             print("Ratio of Profits: "+str(np.sum(profitInd)/np.sum(profitGr)))
    print("Took: "+str(time.time()-initTime)+' seconds')
    return [dictToRetInd, dictToRetL2Ind, dictToRetGr]

def Exper_Ind_L2_L2Ind_GreedyOnePerc(numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps,\
    PPFTable, numericBeta, vector_q, vector_mOrigTest, vector_sTest, vector_ctrTrain, vector_ctrTest, \
    ImpInOrder, MPInOrder, impNames, alphasInd, num_itInd, alphasL2, num_itL2, alphasL2Ind, num_itL2Ind,\
    p_grad_TypeInd, p_grad_TypeL2, p_grad_TypeL2Ind, init_lam, listCampPerImp,\
    perc, sim, seeds, tauMult = -1):

    print('Starting Exper_Ind_L2_L2Ind_Greedy')
    ## The gradient type is needed as the different utility functions have different forms 
    ## for p'(\cdot), and we want to use the right subgradient depending on the method we are using.
    np.random.seed(12345)
    global p_grad_Type
    vector_rctrTrain = np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest = np.multiply(vector_q, vector_ctrTest)
    dictToRetInd = {}
    dictToRetL2 = {}
    dictToRetL2Ind = {}
    dictToRetGr = {}

    print('')
    print("Percentage: "+str(perc)+ ', process id: '+str(os.getpid()))
    vector_m = vector_mOrigTest[:]*perc
    vector_s = vector_sTest[:]
    ext_s = vector_s[index_Imps]
    dictToRetInd[perc] = []
    dictToRetL2[perc] = []
    dictToRetL2Ind[perc] = []
    dictToRetGr[perc] = []

    p_grad_Type=p_grad_TypeInd

    print('About to Run the SubgrAlgSavPrimDualObjInd using '+str(num_itInd)+' iterations')
    initTime =time.time()
    [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
    primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjInd(\
    init_lam, num_itInd, alphasInd, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
    vector_m, num_impressions, numCampaigns, num_edges, \
    PPFTable, numericBeta, index_sizeCamps, index_Imps, (num_itInd-1), p_grad_Type)
    print("Took: "+str( time.time()-initTime)+' seconds')
    
    #print("Duality Gap Last Iteration")
    #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
    lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
    ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
    bidsInd=OptimalBids(ext_LamFinal, vector_rctrTrain)
    [rho_eval_Ind, beta_eval_Ind]=CalcRhoAndBetaVectors(bidsInd, num_edges, index_Imps, PPFTable, numericBeta)

    xInd = CalculateLPGurobi(rho_eval_Ind, beta_eval_Ind, vector_rctrTrain, vector_m, \
    ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
    index_sizeCamps)
    # xInd=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
    #     num_impressions, index_Imps, index_sizeCamps)

    print('')
    print('')
    print('About to Run the SubgrAlgSavPrimDualObjFn_L2Ind using '+str(num_itL2Ind)+' iterations without Indicator')
    initTime =time.time()
    p_grad_Type=p_grad_TypeL2
    tau=np.power(vector_m, -1) * tauMult
    [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
    primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjFn_L2Ind(\
    init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
    vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
    index_Imps, (num_itL2Ind-1), p_grad_Type, tau, False)
    print("Took: "+str( time.time()-initTime)+' seconds')
    
    #print("Duality Gap Last Iteration")
    #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
    lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
    ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
    bidsL2=OptimalBids(ext_LamFinal, vector_rctrTrain)
    [rho_eval_L2, beta_eval_L2]=CalcRhoAndBetaVectors(bidsL2, num_edges, index_Imps, PPFTable, numericBeta)

    xL2 = CalculateQuadGurobi(rho_eval_L2, beta_eval_L2, vector_rctrTrain, vector_m, ext_s, \
        num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau, addIndicator = False)
    
    print('')
    print('')
    print('About to Run the SubgrAlgSavPrimDualObjFn_L2Ind using '+str(num_itL2Ind)+' iterations')
    initTime =time.time()
    p_grad_Type=p_grad_TypeL2Ind
    [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
    primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]=SubgrAlgSavPrimDualObjFn_L2Ind(\
    init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
    vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
    index_Imps, (num_itL2Ind-1), p_grad_Type, tau, True)
    print("Took: "+str( time.time()-initTime)+' seconds')
    
    #print("Duality Gap Last Iteration")
    #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
    lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
    ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
    bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctrTrain)
    [rho_eval_L2Ind, beta_eval_L2Ind]=CalcRhoAndBetaVectors(bidsL2Ind, num_edges, index_Imps, PPFTable, numericBeta)

    xL2Ind = CalculateQuadGurobi(rho_eval_L2Ind, beta_eval_L2Ind, vector_rctrTrain, vector_m, ext_s, \
        num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau, addIndicator = True)

    # xL2Ind=OptimalX(beta_eval, rho_eval, ext_LamFinal, ext_s, vector_rctrTrain, num_edges, numCampaigns, \
    #     num_impressions, index_Imps, index_sizeCamps)
    
    
    ## Now that we have run the primal-dual subgradient methods we run simulations of 
    ## how they would perform in the test log as explained in the paper. The nuber of simulations to
    ## run is equal to the parameter sim. 
    print('')
    print('')
    print('Finished running the Primal-Dual Algorithms')
    print('Starting RunInd_L2_L2Ind_Greedy using '+str(perc)+' percentage of the Test budgets')
    initTime =time.time()
    for i in range(sim): 
        np.random.seed(seeds[i])
        [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, revenueInd, \
        profitInd, budgetL2, cartBidsL2, cartWonL2, cartClickedL2, costBidsL2, revenueL2, \
        profitL2, budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind, budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr,\
        revenueGr, profitGr] = RunInd_L2_L2Ind_Greedy(numCampaigns, num_impressions, num_edges, index_Imps, \
        index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, \
        vector_rctrTrain, vector_ctrTest, vector_rctrTest, bidsInd, xInd, bidsL2, xL2, bidsL2Ind, \
        xL2Ind, tau, ImpInOrder, MPInOrder, impNames,listCampPerImp)

        dictToRetInd[perc].append([budgetInd, cartBidsInd, cartWonInd, \
            cartClickedInd, costBidsInd, revenueInd, profitInd])

        dictToRetL2[perc].append([budgetL2, cartBidsL2, cartWonL2, \
            cartClickedL2, costBidsL2, revenueL2, profitL2])

        dictToRetL2Ind[perc].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
            cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])

        dictToRetGr[perc].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, \
        costBidsGr, revenueGr, profitGr])
    print("Took: "+str(time.time()-initTime)+' seconds')
    return [dictToRetInd, dictToRetL2, dictToRetL2Ind, dictToRetGr]

## For the Pareto Experiment we need to Run only the L2+Indicator for several values of \tau a
## number of simulations. We also need to run the greedy method 

def RunSimOnlyGreedy(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain,\
    vector_rctrTrain, vector_ctrTest, vector_rctrTest, ImpInOrder, MPInOrder, impNames,\
    listCampPerImp, mult = 1.0):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked for 
    ## three methods.

    [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, mat_rctr_by_ImpTrain, _, _, \
        _, _, _]=CreateDataForSimulation(np.zeros(num_edges), np.zeros(num_edges), \
        numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
        vector_q, vector_ctrTrain, vector_rctrTrain, vector_m, PPFTable, numericBeta)
    
    [_, _, _, _, _, _, _, budgetGr, cartBidsGr, cartWonGr, \
        cartClickedGr, costBidsGr, revenueGr, profitGr, mat_r_by_Imp, \
        mat_ctrTest, _, _, _, _, _, _]=CreateDataForSimulation(np.zeros(num_edges), \
    np.zeros(num_edges), numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTest, vector_rctrTest, vector_m, PPFTable, numericBeta)

    ## Now we simulate
    # campaignsArange=np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    # uniOnline=False
    ## We read the test log in irder of how the impressions type appear.
    for i,clusterId in enumerate(ImpInOrder):
        impType=impNames.index(clusterId)
        unifs=allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value=MPInOrder[i]  
        ### Now we update the Greedy Policy    
        ## The greedy heuristic bids for the campaign which stills have remaining
        ## budget and from thos bid for the one with highest r times ctr. 
        ## The previous is true as Ipinyou assumes second price auctions.
        indBuyerGr=-1
        bidAmountGr=0.0
        tryToBidGr=False
        indInterested =\
            mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetGr[listCampPerImp[impType]]
        if np.sum(indInterested) > 0:
            posInt = listCampPerImp[impType][indInterested]
            indBuyerGr = posInt[np.argmax(mat_rctr_by_ImpTrain[posInt,impType])]
            bidAmountGr = mat_rctr_by_ImpTrain[indBuyerGr, impType] * mult
            tryToBidGr = True
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
                probOfClick=mat_ctrTest[indBuyerGr, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedGr[indBuyerGr, impType]+=1
                    payment=mat_r_by_Imp[indBuyerGr, impType]
                    revenueGr[impType]+=payment
                    profitGr[impType]+=payment
                    budgetGr[indBuyerGr]-=payment
    return [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr]

def RunOneSimL2Ind(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, vector_ctrTrain, vector_rctrTrain, \
    vector_ctrTest, vector_rctrTest, bidsL2Ind, xL2Ind, tau, ImpInOrder, MPInOrder, impNames,\
    listCampPerImp):
    
    ## We first initialize the budgets used, matrices of bids  made, won, and clicked.
    
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
        revenueL2Ind, profitL2Ind, _, _, _, _, _, _, _, mat_r_by_Imp, mat_ctrTest, _, _, _, \
        mat_bid_by_ImpL2Ind, mat_x_by_ImpL2Ind, probBidL2Ind] = CreateDataForSimulation(bidsL2Ind, \
    xL2Ind, numCampaigns, num_impressions, num_edges, index_Imps, index_sizeCamps, \
    vector_q, vector_ctrTest, vector_rctrTest, vector_m, PPFTable, numericBeta)
    
    ## Now we simulate
    # campaignsArange=np.arange(numCampaigns)
    ## Instead of np.random.uniform every time we need a random uniform we call
    ## the method at the beginnning of the simulation and save all uniform 
    ## samples we need. 
    allUnifToUse = np.random.uniform(0.0, 1.0, (len(ImpInOrder)*3))
    # uniOnline=False
    ## We read the test log in irder of how the impressions type appear.
    for i,clusterId in enumerate(ImpInOrder):
        impType=impNames.index(clusterId)
        unifs=allUnifToUse[(3*i):(3*(i+1))]
        ## Market Price that appears in the test log. 
        mp_value=MPInOrder[i]
        ## Update L2Ind (Same code as done before for the pure indicator case)
        indBuyerL2Ind=0
        tryToBidL2Ind=False
        bidAmountL2Ind=0.0
        if unifs[0] <= probBidL2Ind[impType]:
            ## For each campaign we check if there is any that has enough budget to bid and that 
            ## also wants to do so. 
            bidUsingL2Ind=False
            indInterested =\
                (mat_r_by_Imp[listCampPerImp[impType],impType] <= budgetL2Ind[listCampPerImp[impType]]) * \
                    (mat_x_by_ImpL2Ind[listCampPerImp[impType],impType]>0)
            if np.sum(indInterested) >0:
                bidUsingL2Ind= True
            if bidUsingL2Ind: 
                ## There is at least one campaign that wants to bid.
                posInt=listCampPerImp[impType][indInterested]
                ## Conditional probability assuming that the method is going to bid.
                ## This conditional probability excludes all those campaigns
                ## that do not want to bid
                condProbInterested=mat_x_by_ImpL2Ind[posInt, impType]
                condProbInterested*=1.0/np.sum(condProbInterested)
                auxPartSum=0.0
                ## Now we will choose in behalf of which campaign to bid for.
                numInterest = len(condProbInterested)
                auxPosForindBuyerL2Ind = numInterest-1
                z = 0
                while z <numInterest:
                    auxPartSum += condProbInterested[z]
                    if auxPartSum >= unifs[1]:
                        ## If we exceed unifs[1] go out of the loop
                        auxPosForindBuyerL2Ind=z
                        z+=numInterest
                    z += 1
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
                probOfClick=mat_ctrTest[indBuyerL2Ind, impType]
                if (unifs[2]<=probOfClick):
                    ## User clicked, increase revenue and charge the campaign.
                    cartClickedL2Ind[indBuyerL2Ind, impType]+=1
                    payment=mat_r_by_Imp[indBuyerL2Ind, impType]
                    revenueL2Ind[impType]+=payment
                    profitL2Ind[impType]+=payment
                    budgetL2Ind[indBuyerL2Ind]-=payment
    return [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
            revenueL2Ind, profitL2Ind]

def RunOnlyGreedyOneSeed(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrig, vector_ctrTrain,\
    vector_ctrTest, ImpInOrder, MPInOrder, impNames, listCampPerImp, seed, mult = 1.0):

    vector_rctrTrain = np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest = np.multiply(vector_q, vector_ctrTest)
    vector_m = vector_mOrig[:]

    np.random.seed(seed)
    [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
    revenueL2Ind, profitL2Ind] = RunSimOnlyGreedy(numCampaigns, num_impressions, \
    num_edges, index_Imps, index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m,\
    vector_ctrTrain, vector_rctrTrain, vector_ctrTest, vector_rctrTest, ImpInOrder, \
    MPInOrder, impNames, listCampPerImp, mult)
    return [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
    revenueL2Ind, profitL2Ind]

def RunOnlyGreedySeveralSeeds(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrig, vector_ctrTrain,\
    vector_ctrTest, ImpInOrder, MPInOrder, impNames, listCampPerImp, seeds, mult):

    dictToRet={}
    dictToRet[mult] = []
    print('Running Simulations Greedy')
    initTime = time.time()
    for seed in seeds:
        [budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr, revenueGr, profitGr] =\
        RunOnlyGreedyOneSeed(numCampaigns, num_impressions, num_edges, index_Imps, \
        index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrig, vector_ctrTrain,\
        vector_ctrTest, ImpInOrder, MPInOrder, impNames, listCampPerImp, seed, mult)
        dictToRet[mult].append([budgetGr, cartBidsGr, cartWonGr, cartClickedGr, costBidsGr,\
            revenueGr, profitGr])
    print(" 'Running Simulations Greedy took: "+str( time.time()-initTime)+' seconds')
    return dictToRet


def ExpPareto(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrig, vector_sOrig, \
    vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, impNames, alphasL2Ind, \
    num_itL2Ind, p_grad_Type, init_lam, tauMult, sim, listCampPerImp, seeds, addIndicator =True):

    vector_s=vector_sOrig
    ext_s = vector_s[index_Imps]
    vector_rctrTrain=np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest=np.multiply(vector_q, vector_ctrTest)
    vector_m=vector_mOrig
    dictToRetL2Ind={}
    
    for perc in tauMult:
        # initTime = time.time()
        dictToRetL2Ind[perc]=[]
        p_grad_Type=p_grad_Type
        tau=np.power(vector_m, -1)*perc
        # initTime = time.time()
        np.random.seed(12345)
        [_, _, _, _, _, _, _, dual_varsAvg] = SubgrAlgSavPrimDualObjFn_L2Ind(\
        init_lam, num_itL2Ind, alphasL2Ind, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
        vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
        index_Imps, (num_itL2Ind-1), p_grad_Type, tau, addIndicator = addIndicator)
        # print("Took: "+str( time.time()-initTime)+' seconds')
        
        #print("Duality Gap Last Iteration")
        #print(str(dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]))
        lamFinal=dual_varsAvg[len(dual_varsAvg)-1]
        ext_LamFinal=ExtendSizeCamps(lamFinal, index_sizeCamps)
        bidsL2Ind=OptimalBids(ext_LamFinal, vector_rctrTrain)
        [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidsL2Ind, num_edges, index_Imps, PPFTable, numericBeta)

        xL2Ind = CalculateQuadGurobi(rho_eval, beta_eval, vector_rctrTrain, vector_m, ext_s, \
        num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, tau, addIndicator = addIndicator)

        # print('Running Simulations')
        # initTime = time.time()
        for i in range(sim): 
            np.random.seed(seeds[i])
            [budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, cartClickedL2Ind, costBidsL2Ind, \
            revenueL2Ind, profitL2Ind]=RunOneSimL2Ind(numCampaigns, num_impressions, \
            num_edges, index_Imps, index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, \
            vector_ctrTrain, vector_rctrTrain, vector_ctrTest, vector_rctrTest, \
            bidsL2Ind, xL2Ind, tau, ImpInOrder, MPInOrder, impNames, listCampPerImp)
            dictToRetL2Ind[perc].append([budgetL2Ind, cartBidsL2Ind, cartWonL2Ind, \
                cartClickedL2Ind, costBidsL2Ind, revenueL2Ind, profitL2Ind])
        # print("Took: "+str( time.time()-initTime)+' seconds')
    return dictToRetL2Ind

def RunOnlyInd(numCampaigns, num_impressions, num_edges, index_Imps, \
    index_sizeCamps, PPFTable, numericBeta, vector_q, vector_mOrig, vector_sOrig, \
    vector_ctrTrain, vector_ctrTest, ImpInOrder, MPInOrder, impNames, alphasInd, \
    num_itInd, p_grad_Type, init_lam, sim, listCampPerImp, seeds):

    vector_s=vector_sOrig
    ext_s = vector_s[index_Imps]
    vector_rctrTrain=np.multiply(vector_q, vector_ctrTrain)
    vector_rctrTest=np.multiply(vector_q, vector_ctrTest)
    vector_m=vector_mOrig
    
    listToRet = []
    p_grad_Type = p_grad_Type
    initTime = time.time()
    np.random.seed(12345)
    [_, _, _, _, dual_AvgLamFnValues, primal_AvgLamGivenMu, _, dual_varsAvg] = SubgrAlgSavPrimDualObjInd(\
    init_lam, num_itInd, alphasInd, vector_q, vector_ctrTrain, vector_rctrTrain, vector_s, ext_s, \
    vector_m, num_impressions, numCampaigns, num_edges, PPFTable, numericBeta, index_sizeCamps, \
    index_Imps, (num_itInd-1), p_grad_Type)

    print("Running SubgrAlgSavPrimDualObjInd took: "+str( time.time()-initTime)+' seconds')
    
    dGap = (dual_AvgLamFnValues[len(dual_AvgLamFnValues)-1]-primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1])/\
        primal_AvgLamGivenMu[len(primal_AvgLamGivenMu)-1]
    print('Duality Gap Last Iteration: '+str(dGap*100)+'%')

    lamFinal = dual_varsAvg[len(dual_varsAvg)-1]
    ext_LamFinal = ExtendSizeCamps(lamFinal, index_sizeCamps)
    bidsInd = OptimalBids(ext_LamFinal, vector_rctrTrain)
    [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidsInd, num_edges, index_Imps, PPFTable, numericBeta)

    xInd = CalculateLPGurobi(rho_eval, beta_eval, vector_rctrTrain, vector_m, ext_s, \
    num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps)

    print('Running Simulations')
    initTime = time.time()
    for i in range(sim): 
        np.random.seed(seeds[i])
        [budgetInd, cartBidsInd, cartWonInd, cartClickedInd, costBidsInd, \
        revenueInd, profitInd]=RunOneSimL2Ind(numCampaigns, num_impressions, \
        num_edges, index_Imps, index_sizeCamps, PPFTable, numericBeta, vector_q, vector_m, \
        vector_ctrTrain, vector_rctrTrain, vector_ctrTest, vector_rctrTest, \
        bidsInd, xInd, 0.0, ImpInOrder, MPInOrder, impNames, listCampPerImp)
        listToRet.append([budgetInd, cartBidsInd, cartWonInd, \
            cartClickedInd, costBidsInd, revenueInd, profitInd])
    print("Took: "+str( time.time()-initTime)+' seconds')
    return listToRet