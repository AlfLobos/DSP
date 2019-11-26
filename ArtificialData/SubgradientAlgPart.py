import numpy as np
from RhoAndBeta import CalcRhoAndBetaVectors
from UtilitiesOptimization import ExtendSizeCamps, OptimalBids, OptimalX,\
    CalculateDualFnValueInd, CalculateDualFnValueL2Ind, CalculateLPGurobi,\
    CalculateQuadGurobi, CalculateBudgetUsed, LamSubgr

### Subgradient Algorithm for the pure Indicator Case
def SubgrAlgSavPrimDualObjInd(init_lam, num_it, alphas, vector_r, vector_ctr, \
    vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
    index_sizeCamps, index_Imps, it_per_cal, adverPerImp, p_grad_Type, UB_bid, firstPrice):
    ## Initialize the lambdas at zero 
    lam=np.zeros(numCampaigns)
    lam_average=np.zeros(numCampaigns)
    ## Sum steps is used to calculate the average subgradient
    sum_steps=0.0
    ## The following are recalculated at each step of the 
    x=np.zeros(num_edges)
    x_dual=np.zeros(num_edges)
    bid=np.zeros(num_edges)
    ## We saved the primal and dual F'n values of the averaged and non-averaged methods
    ## Same with the budget used by the solution implied by the dual problem and 
    ## the dual variables. 
    dual_FnValues=[]
    primal_GivenMu=[]
    budget_used=[]
    dual_vars=[]
    primal_AvgLamGivenMu=[]
    dual_AvgLamFnValues=[]
    budget_LamAvgUse=[]
    dual_varsAvg=[]
    for t in range(num_it):
        ext_lam=ExtendSizeCamps(lam, index_sizeCamps)
        ## Step 1: Obtain Optimal bids
        bid[:]=OptimalBids(ext_lam, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp)
        ## Step 2: Obtain optimal x
        #[rho_eval, beta_eval]=rhoAndBetaEvaluated(bid, num_edges, \
        #    index_Imps, lb_bid_others_vec, ub_bid_others_vec, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bid, UB_bid, num_edges,\
            index_Imps, adverPerImp, firstPrice) 
        x[:]=OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_rctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
        ## Step 3: Obtain a subgradient
        subgr_lam=LamSubgr(beta_eval, rho_eval, x, vector_rctr, ext_s, \
            numCampaigns, index_sizeCamps, vector_m, lam, p_grad_Type)
        ## Step 4&5: Subgradient step and projection to [0, 1]
        #lam[:]=np.maximum(lam-alphas[t]*subgr_lam, 1.0, 0)
        lam[:]=np.maximum(np.minimum(lam-alphas[t]*subgr_lam, 1), 0)
        
        ## See if we need to calculate the average Lambda
        if t>=num_it*0.5:        
            lam_average=(alphas[t]/(sum_steps+alphas[t]))*lam+\
            (sum_steps/(sum_steps+alphas[t]))*lam_average
            sum_steps+=alphas[t]
        ## See if we need to save the primal, dual F'n values, dual variables, and budgets used.
        if (t%it_per_cal==(it_per_cal-1)):
            ## First for non-weighted lambda
            ext_lam=ExtendSizeCamps(lam, index_sizeCamps)
            bidFound=np.maximum(OptimalBids(ext_lam, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp), 0)
            [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidFound, UB_bid, num_edges,\
                index_Imps, adverPerImp, firstPrice) 
            x_dual[:]=OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_rctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
            
            dualFnValue=CalculateDualFnValueInd(beta_eval, rho_eval, vector_rctr, \
            ext_lam, ext_s, vector_m, lam, x_dual)
            
            [optSol, primalFnValue]=CalculateLPGurobi(rho_eval, beta_eval, \
            vector_rctr, vector_m, ext_s, num_impressions, numCampaigns, num_edges, \
            index_Imps, index_sizeCamps, True)
            dual_FnValues.append(dualFnValue)
            primal_GivenMu.append(primalFnValue)
            dual_vars.append(lam)
            
            budget_used.append(CalculateBudgetUsed(rho_eval, vector_rctr, \
            num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSol))

#             print('Iteration number ', t)
#             print('Dual Function Value ', dualFnValue)
#             print('Primal Function Value', primalFnValue)
            
            ## For weighted Lambda
            if t>=num_it*0.5:
                ext_lam=ExtendSizeCamps(lam_average, index_sizeCamps)
                bidFound=np.maximum(OptimalBids(ext_lam, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp), 0)
                [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidFound, UB_bid, num_edges, index_Imps,\
                    adverPerImp, firstPrice) 
                x_dual[:]=OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_rctr, \
                num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
                dualAvgLamFnValue=CalculateDualFnValueInd(beta_eval, rho_eval, vector_rctr, \
                    ext_lam, ext_s, vector_m, lam_average, x_dual)
                [optSolAvg, primalAvgLamFnValue]=CalculateLPGurobi(rho_eval, beta_eval, \
                vector_rctr, vector_m, ext_s, num_impressions, numCampaigns, num_edges, \
                index_Imps, index_sizeCamps, True)
                budget_LamAvgUse.append(CalculateBudgetUsed(rho_eval, vector_rctr, \
                num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSolAvg))
                dual_AvgLamFnValues.append(dualAvgLamFnValue)
                primal_AvgLamGivenMu.append(primalAvgLamFnValue)
#                 print('Dual Avg Function Value ', dualAvgLamFnValue)
#                 print('Primal Avg Function Value', primalAvgLamFnValue)
                dual_varsAvg.append(lam_average)
            else:
                dual_AvgLamFnValues.append(0)
                primal_AvgLamGivenMu.append(0)
    return [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
            primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]


### Subgradient Algorithm for the L2+  Indicator Case
def SubgrAlgSavPrimDualObjFn_L2Ind(init_lam, num_it, alphas, vector_r, vector_ctr, \
    vector_rctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
    index_sizeCamps, index_Imps, it_per_cal, adverPerImp, tau, addIndicator, p_grad_Type,\
    UB_bid, firstPrice):
    ## Initialize the lambdas at zero 
    lam=np.zeros(numCampaigns)
    lam_average=np.zeros(numCampaigns)
    ## Sum steps is used to calculate the average subgradient
    sum_steps=0.0
    ## The following are recalculated at each step of the 
    x=np.zeros(num_edges)
    x_dual=np.zeros(num_edges)
    bid=np.zeros(num_edges)
    ## We saved the primal and dual F'n values of the averaged and non-averaged methods
    ## Same with the budget used by the solution implied by the dual problem and 
    ## the dual variables. 
    dual_FnValues=[]
    primal_GivenMu=[]
    budget_used=[]
    dual_vars=[]
    primal_AvgLamGivenMu=[]
    dual_AvgLamFnValues=[]
    budget_LamAvgUse=[]
    dual_varsAvg=[]
    for t in range(num_it):
        ext_lam=ExtendSizeCamps(lam, index_sizeCamps)
        ## Step 1: Obtain Optimal bids
        ext_lam_for_bid=np.maximum(ext_lam, 0)
        bid[:]=OptimalBids(ext_lam_for_bid, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp)
        ## Step 2: Obtain optimal x
        [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bid, UB_bid, num_edges,\
            index_Imps, adverPerImp, firstPrice) 
        x[:]=OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_rctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
        ## Step 3: Obtain a subgradient
        subgr_lam=LamSubgr(beta_eval, rho_eval, x, vector_rctr, ext_s, \
            numCampaigns, index_sizeCamps, vector_m, lam, p_grad_Type, tau)[0]
        ## Step 4&5: Subgradient step and projection to <=1
        normSubg=0.0
#         print(subgr_lam)
        for elem in  subgr_lam:
            normSubg+=elem*elem
        if normSubg>0:
            lam[:]=np.minimum(lam-alphas[t]*(subgr_lam/np.sqrt(normSubg)), 1)
        # else nothing happens

        
        ## See if we need to calculate the average Lambda
        if t>=num_it*0.5:        
            lam_average=(alphas[t]/(sum_steps+alphas[t]))*lam+\
            (sum_steps/(sum_steps+alphas[t]))*lam_average
            sum_steps+=alphas[t]
            
        ## See if we need to save the primal, dual F'n values, dual variables, and budgets used.
        if (t%it_per_cal==(it_per_cal-1)):
            ## First for non-weighted lambda
            ext_lam=ExtendSizeCamps(lam, index_sizeCamps)
            ext_lam_for_bid=np.maximum(ext_lam, 0)
            bidFound=np.maximum(OptimalBids(ext_lam_for_bid, vector_rctr, UB_bid, firstPrice, index_Imps, adverPerImp), 0)
            [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bidFound, UB_bid, num_edges,\
                index_Imps, adverPerImp, firstPrice) 
            x_dual[:]=OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_rctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
            dualFnValue=CalculateDualFnValueL2Ind(beta_eval, rho_eval, vector_rctr, \
            ext_lam, ext_s, vector_m, lam, tau, x_dual, addIndicator)
            [optSol, primalFnValue]=CalculateQuadGurobi(rho_eval, beta_eval, vector_rctr, \
            vector_m, ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
            index_sizeCamps, tau, addIndicator, True)
            dual_FnValues.append(dualFnValue)
            primal_GivenMu.append(primalFnValue)
            dual_vars.append(lam)
            
            budget_used.append(CalculateBudgetUsed(rho_eval, vector_rctr, \
            num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSol))

            
#             print('Iteration number ', t)
#             print('Dual Function Value ', dualFnValue)
#             print('Primal Function Value', primalFnValue)
            
            ## For weighted Lambda
            if t>=num_it*0.5:
                ext_lam=ExtendSizeCamps(lam_average, index_sizeCamps)
                ext_lamAvg_for_bid=np.maximum(ext_lam, 0)
                bidFound=np.maximum(OptimalBids(ext_lamAvg_for_bid, vector_rctr, UB_bid, firstPrice,\
                    index_Imps, adverPerImp), 0)
                [rho_eval, beta_eval]=CalcRhoAndBetaVectors(bid, UB_bid, num_edges, index_Imps,\
                    adverPerImp, firstPrice) 
                x_dual[:]=OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_rctr, \
                num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
                dualAvgLamFnValue=CalculateDualFnValueL2Ind(beta_eval, rho_eval, vector_rctr, \
                ext_lam, ext_s, vector_m, lam_average, tau, x_dual, addIndicator)
                [optSolAvg, primalAvgLamFnValue]=CalculateQuadGurobi(rho_eval, beta_eval, 
                vector_rctr, vector_m, ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
                index_sizeCamps, tau, addIndicator, True)
                budget_LamAvgUse.append(CalculateBudgetUsed(rho_eval, vector_rctr, \
                num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSolAvg))
                dual_AvgLamFnValues.append(dualAvgLamFnValue)
                primal_AvgLamGivenMu.append(primalAvgLamFnValue)
#                 print('Dual Avg Function Value ', dualAvgLamFnValue)
#                 print('Primal Avg Function Value', primalAvgLamFnValue)
                dual_varsAvg.append(lam_average)
            else:
                dual_AvgLamFnValues.append(0)
                primal_AvgLamGivenMu.append(0)
    return [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
            primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]