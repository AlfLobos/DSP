import numpy as np
from gurobipy import *
import sys
import time
import os

from Utilities import NumericSecPrice, ForBetaDist, CalcRhoAndBetaVectors, BetaValue

## Subgradient for the Indicator case
def P_gradient_0(lam, vector_m):
    return vector_m

## Subgradient for l_2 case -0.5*tau*(v-m)^2 without indicator
def P_gradient_1(lam, vector_m, tau):
    aux = 1.0/np.asarray(tau)
    aux2 = np.multiply(aux, lam)
    return vector_m+aux2

## This is  l_2 case -0.5*tau*(v-m)^2 plus 
## indicator 1{v <=  m}
def P_gradient_2(lam, vector_m, tau):
    aux = 1.0/np.asarray(tau)
    aux2 = np.multiply(aux, lam)
    indicatorNeg = -np.minimum(np.sign(lam), 0)
    return vector_m+np.multiply(indicatorNeg, aux2)

## General function that calls any of the 3 gradient types
def P_gradient(lam, vector_m, p_grad_Type, *args, **kwargs):
    ## p_grad_Type = 0 is the only budget constraints problem
    ## p_grad_Type = 1 is when only a quadratic penalization is used.
    ## p_grad_Type = 2 is when we mix a quadratic penalization with 
    ## budget constraints.
    if p_grad_Type == 0:
        return P_gradient_0(lam, vector_m)
    elif p_grad_Type == 1:
        return P_gradient_1(lam, vector_m, args[0])
    elif p_grad_Type == 2:
        return P_gradient_2(lam, vector_m, args[0])
    else:
        sys.exit("Not a valid P gradient Type Option")
        
def ExtendSizeCamps(vecToExtend, vector_rep):
    '''
    Takes a vector [a b c ..] and returns
    [a a .. a b b .. b c c .. c ..] where the number of times each
    element gets repeated equals vector_rep.
    '''
    return np.repeat(vecToExtend, vector_rep)

## For second price auctions is optimal to bid truthfully
## or the maximum  bid allowed (if the truthful value exceeds it).
def OptimalBids(ext_lam, vector_qctr):
    truthVal = np.multiply((1-ext_lam), vector_qctr) 
    return truthVal

## For given bid values, we calculte an optimal allocation vector 
## which is a vector full of zeroes except in one coordinate 
## which takes the value of 1 for a campaign that maximizes the profit. 
## If no campaign achieves a strictly positive profit it returns a full
## zero vector. The code can be simplified be comparing only the
##  $r_{ik}(1-\lambda_k)$ values.
def OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
    num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps):
    optX = np.zeros(num_edges)
    insideBracket = np.multiply((1-ext_lam), vector_qctr)-beta_eval
    outsideBracket = np.multiply(ext_s, rho_eval)
    toBeCompared = np.multiply(insideBracket, outsideBracket)
    for i in range(num_impressions):
        indexes = np.arange(num_edges, dtype = int)[(index_Imps == i)]
        indMax = np.argmax(toBeCompared[indexes])
        pos = indexes[indMax]
        if(toBeCompared[pos]>0):
            optX[pos] = 1
    return optX        

## Subgradient of the dual function, i.e., subgradient of $p(\cdot)$
## minus the expected budget used ($v(x, b)$).
def LamSubgr(beta_eval, rho_eval, x, vector_qctr, ext_s, numCampaigns, \
          index_sizeCamps, vector_m, lam, p_grad_Type, *args, **kwargs):
    ## First I will multiply everything and then I will reduce
    exp_for_lam = -np.multiply(np.multiply(vector_qctr, ext_s), \
                    np.multiply(rho_eval, x))
    subg_lam = np.zeros(numCampaigns)
    # Could be faster, but this is good enough
    aux = 0
    for i in range(numCampaigns):
        sizeCamp = index_sizeCamps[i]
        # subg_lam[i] = np.sum(exp_for_lam[np.arange(aux, aux+sizeCamp)])
        subg_lam[i] = np.sum(exp_for_lam[aux:(aux+sizeCamp)])
        aux+= sizeCamp
    return (subg_lam+ P_gradient(lam, vector_m, p_grad_Type, args))



### Subgradient Algorithm for the pure Indicator Case
def SubgrAlgSavPrimDualObjInd(init_lam, num_it, alphas, vector_q, vector_ctr, \
    vector_qctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
    PPFTable, numericBeta, index_sizeCamps, index_Imps, it_per_cal, p_grad_Type):
    ## Initialize the lambdas at zero 
    lam = np.zeros(numCampaigns)
    lam_average = np.zeros(numCampaigns)
    ## Sum steps is used to calculate the average subgradient
    sum_steps = 0.0
    ## The following are recalculated at each step of the 
    x = np.zeros(num_edges)
    x_dual = np.zeros(num_edges)
    bid = np.zeros(num_edges)
    ## We saved the primal and dual F'n values of the averaged and non-averaged methods
    ## Same with the budget used by the solution implied by the dual problem and 
    ## the dual variables. 
    dual_FnValues = []
    primal_GivenMu = []
    budget_used = []
    dual_vars = []
    primal_AvgLamGivenMu = []
    dual_AvgLamFnValues = []
    budget_LamAvgUse = []
    dual_varsAvg = []

    for t in range(num_it):
        ext_lam = ExtendSizeCamps(lam, index_sizeCamps)
        ## Step 1: Obtain Optimal bids
        bid[:] = OptimalBids(ext_lam, vector_qctr)
        ## Step 2: Obtain optimal x
        #[rho_eval, beta_eval] = rhoAndBetaEvaluated(bid, num_edges, \
        #    index_Imps, lb_bid_others_vec, ub_bid_others_vec, adverPerImp)
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bid, num_edges, \
                            index_Imps, PPFTable, numericBeta)
        x[:] = OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
        ## Step 3: Obtain a subgradient
        subgr_lam = LamSubgr(beta_eval, rho_eval, x, vector_qctr, ext_s, \
            numCampaigns, index_sizeCamps, vector_m, lam, p_grad_Type)
        ## Step 4&5: Subgradient step and projection to [0, 1]
        #lam[:] = np.maximum(lam-alphas[t]*subgr_lam, 1.0, 0)
        lam[:] = np.maximum(np.minimum(lam-alphas[t]*subgr_lam, 1), 0)
        
        ## See if we need to calculate the average Lambda
        if t >= num_it*0.5:        
            lam_average = (alphas[t]/(sum_steps+alphas[t]))*lam+\
            (sum_steps/(sum_steps+alphas[t]))*lam_average
            sum_steps += alphas[t]
        ## See if we need to save the primal, dual F'n values, dual variables, and budgets used.
        if (t%it_per_cal == (it_per_cal-1)):
            # print('It took: '+str(time.time() -initTime)+' seconds the part when we still have not called Gurobi, '+'process id: '+str(os.getpid()))
            
            ## First for non-weighted lambda
            ext_lam = ExtendSizeCamps(lam, index_sizeCamps)
            bidFound = OptimalBids(ext_lam, vector_qctr)
            [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidFound, num_edges, \
                            index_Imps, PPFTable, numericBeta)
            x_dual[:] = OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
            
            # print('About to run CalculateDualFnValueInd, '+'process id: '+str(os.getpid()))
            # initTime = time.time()
            dualFnValue = CalculateDualFnValueInd(beta_eval, rho_eval, vector_qctr, \
            ext_lam, ext_s, vector_m, lam, x_dual)
            # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

            # print('About to run CalculateLPGurobi, '+'process id: '+str(os.getpid()))
            # initTime = time.time()
            [optSol, primalFnValue] = CalculateLPGurobi(rho_eval, beta_eval, \
            vector_qctr, vector_m, ext_s, num_impressions, numCampaigns, num_edges, \
            index_Imps, index_sizeCamps, saveObjFn = True)
            # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))
            dual_FnValues.append(dualFnValue)
            primal_GivenMu.append(primalFnValue)
            dual_vars.append(lam)
            
            budget_used.append(CalculateBudgetUsed(rho_eval, vector_qctr, \
            num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSol))

#             print('Iteration number ', t)
#             print('Dual Function Value ', dualFnValue)
#             print('Primal Function Value', primalFnValue)
            
            ## For weighted Lambda
            if t >= num_it*0.5:
                ext_lam = ExtendSizeCamps(lam_average, index_sizeCamps)
                bidFound = OptimalBids(ext_lam, vector_qctr)
                [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidFound, num_edges, \
                            index_Imps, PPFTable, numericBeta)
                x_dual[:] = OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
                num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)

                # print('About to run CalculateDualFnValueInd, '+'process id: '+str(os.getpid()))
                # initTime = time.time()
                dualAvgLamFnValue = CalculateDualFnValueInd(beta_eval, rho_eval, vector_qctr, \
                    ext_lam, ext_s, vector_m, lam_average, x_dual)
                # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

                # print('About to run CalculateLPGurobi, '+'process id: '+str(os.getpid()))
                # initTime = time.time()
                [optSolAvg, primalAvgLamFnValue] = CalculateLPGurobi(rho_eval, beta_eval, \
                vector_qctr, vector_m, ext_s, num_impressions, numCampaigns, num_edges, \
                index_Imps, index_sizeCamps, saveObjFn = True)
                # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

                budget_LamAvgUse.append(CalculateBudgetUsed(rho_eval, vector_qctr, \
                num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSolAvg))
                dual_AvgLamFnValues.append(dualAvgLamFnValue)
                primal_AvgLamGivenMu.append(primalAvgLamFnValue)
                dual_varsAvg.append(lam_average)
            else:
                dual_AvgLamFnValues.append(0)
                primal_AvgLamGivenMu.append(0)
    return [dual_FnValues, primal_GivenMu, budget_used, dual_vars, dual_AvgLamFnValues, \
            primal_AvgLamGivenMu, budget_LamAvgUse, dual_varsAvg]

### Subgradient Algorithm for the L2+  Indicator Case
def SubgrAlgSavPrimDualObjFn_L2Ind(init_lam, num_it, alphas, vector_q, vector_ctr, \
    vector_qctr, vector_s, ext_s, vector_m, num_impressions, numCampaigns, num_edges, \
    PPFTable, numericBeta, index_sizeCamps, index_Imps, it_per_cal, p_grad_Type, tau, addIndicator):
    ## Initialize the lambdas at zero 
    lam = np.zeros(numCampaigns)
    lam_average = np.zeros(numCampaigns)
    ## Sum steps is used to calculate the average subgradient
    sum_steps = 0.0
    ## The following are recalculated at each step of the 
    x = np.zeros(num_edges)
    x_dual = np.zeros(num_edges)
    bid = np.zeros(num_edges)
    ## We saved the primal and dual F'n values of the averaged and non-averaged methods
    ## Same with the budget used by hte solution implied by the dual problem and 
    ## the dual variables. 
    dual_FnValues = []
    primal_GivenMu = []
    budget_used = []
    dual_vars = []
    primal_AvgLamGivenMu = []
    dual_AvgLamFnValues = []
    budget_LamAvgUse = []
    dual_varsAvg = []
    # initTime = time.time()

    ## Obtain Upper Bound for bidding variables
    UB_bids = np.zeros(len(PPFTable))
    for i, listInPPFTable in enumerate(PPFTable):
        UB_bids[i] = listInPPFTable[-1]
    UB_bids = UB_bids[index_Imps]

    for t in range(num_it):
        ext_lam = ExtendSizeCamps(lam, index_sizeCamps)
        ## Step 1: Obtain Optimal bids
        bid[:] = np.minimum(OptimalBids(ext_lam, vector_qctr), UB_bids)
        ## Step 2: Obtain optimal x
        [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bid, num_edges, \
                            index_Imps, PPFTable, numericBeta)
        x[:] = OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)
        ## Step 3: Obtain a subgradient
        subgr_lam = LamSubgr(beta_eval, rho_eval, x, vector_qctr, ext_s, \
            numCampaigns, index_sizeCamps, vector_m, lam, p_grad_Type, tau)[0]
        ## Step 4&5: Subgradient step and projection to  <=  1
        normSubg = 0.0
#         print(subgr_lam)
        for elem in  subgr_lam:
            normSubg+= elem*elem
        if normSubg>0:
            lam[:] = np.minimum(lam-alphas[t]*(subgr_lam/np.sqrt(normSubg)), 1)
        # else nothing happens

        
        ## See if we need to calculate the average Lambda
        if t >=  num_it*0.5:        
            lam_average = (alphas[t]/(sum_steps+alphas[t]))*lam + (sum_steps/(sum_steps+alphas[t]))*lam_average
            sum_steps += alphas[t]
            
        ## See if we need to save the primal, dual F'n values, dual variables, and budgets used.
        if (t%it_per_cal == (it_per_cal-1)):
            # print('It took: '+str(time.time() -initTime)+' seconds the part when we still have not called Gurobi')
            ## First for non-weighted lambda
            ext_lam = ExtendSizeCamps(lam, index_sizeCamps)
            bidFound = np.minimum(OptimalBids(ext_lam, vector_qctr), UB_bids)
            [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidFound, num_edges, \
                            index_Imps, PPFTable, numericBeta)
            x_dual[:] = OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
            num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)

            # print('About to run CalculateDualFnValueL2Ind, '+'process id: '+str(os.getpid()))
            # initTime = time.time()
            dualFnValue = CalculateDualFnValueL2Ind(beta_eval, rho_eval, vector_qctr, \
            ext_lam, ext_s, vector_m, lam, tau, x_dual, addIndicator)
            # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

            # print('About to run CalculateQuadGurobi, '+'process id: '+str(os.getpid()))
            # initTime = time.time()
            [optSol, primalFnValue] = CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, \
            vector_m, ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
            index_sizeCamps, tau, addIndicator, saveObjFn = True)
            # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

            dual_FnValues.append(dualFnValue)
            primal_GivenMu.append(primalFnValue)
            dual_vars.append(lam)
            
            budget_used.append(CalculateBudgetUsed(rho_eval, vector_qctr, \
            num_impressions, numCampaigns, index_sizeCamps, ext_s, vector_m, optSol))

            
#             print('Iteration number ', t)
#             print('Dual Function Value ', dualFnValue)
#             print('Primal Function Value', primalFnValue)
            
            ## For weighted Lambda
            if t >=  num_it*0.5:
                ext_lam = ExtendSizeCamps(lam_average, index_sizeCamps)
                bidFound = np.minimum(np.maximum(OptimalBids(ext_lam, vector_qctr), 0), UB_bids)
                [rho_eval, beta_eval] = CalcRhoAndBetaVectors(bidFound, num_edges, \
                            index_Imps, PPFTable, numericBeta)
                x_dual[:] = OptimalX(beta_eval, rho_eval, ext_lam, ext_s, vector_qctr, \
                num_edges, numCampaigns, num_impressions, index_Imps, index_sizeCamps)

                # print('About to run CalculateDualFnValueL2Ind, '+'process id: '+str(os.getpid()))
                # initTime = time.time()
                dualAvgLamFnValue = CalculateDualFnValueL2Ind(beta_eval, rho_eval, vector_qctr, \
                ext_lam, ext_s, vector_m, lam_average, tau, x_dual, addIndicator)
                # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

                # print('About to run CalculateQuadGurobi, '+'process id: '+str(os.getpid()))
                # initTime = time.time()
                [optSolAvg, primalAvgLamFnValue] = CalculateQuadGurobi(rho_eval, beta_eval, 
                vector_qctr, vector_m, ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
                index_sizeCamps, tau, addIndicator, saveObjFn = True)
                # print('It took: '+str(time.time() -initTime)+' seconds, '+'process id: '+str(os.getpid()))

                budget_LamAvgUse.append(CalculateBudgetUsed(rho_eval, vector_qctr, \
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

# ## Function to Calculate the Dual Fn Value and Budget Used

# 1) 'CalculateDualFnValueInd' and 'CalculateDualFnValueL2Ind' are the functions to calculate the dual function value 
# for the utility function that uses the indicator function and indicator plus the $\ell_2$ penalty.

# 2) 'CalculateBudgetUsed' calculated the budget used given the bids and allocation obtained by the dual function (it 
# may spend more the budget allowed for some campaigns)

def CalculateDualFnValueInd(beta_eval, rho_eval, vector_qctr, \
            ext_lam, ext_s, vector_m, lam, x):
    ## Let's the terms that will be later multiplied by x in the profit part.
    insideBracket = np.multiply(vector_qctr, (1-ext_lam))-beta_eval
    outsideBracket = np.multiply(ext_s, rho_eval)
    coeffPnotP = np.multiply(insideBracket, outsideBracket)
    ## Let's simple multiply those terms by x and sum the p(\cdot) part.
    return np.dot(coeffPnotP, x)+np.dot(lam, vector_m)

def CalculateDualFnValueL2Ind(beta_eval, rho_eval, vector_qctr, \
    ext_lam, ext_s,  vector_m, lam, tau, x, addIndicator = False):
    ## Let's the terms that will be later multiplied by x in the profit part.
    insideBracket = np.multiply(vector_qctr, (1-ext_lam))-beta_eval
    outsideBracket = np.multiply(ext_s, rho_eval)
    coeffPnotP = np.multiply(insideBracket, outsideBracket)
    ## Let's now calculate the p(\cdot) part.
    lamSq = np.multiply(lam, lam)
    if addIndicator:
        lamSq = lamSq * (lam <= 0)
        # indicatorNeg = -np.minimum(np.sign(lam), 0)
        # lamSq = np.multiply(indicatorNeg, lamSq)
    return np.dot(coeffPnotP, x)+np.dot(lam, vector_m)\
        +np.dot((0.5/np.asarray(tau)), lamSq)

def CalculateBudgetUsed(rho_eval, vector_qctr, \
    num_impressions, numCampaigns, index_sizeCamps, ext_s, \
    vector_m, x):
    
    budget_spend = np.zeros(numCampaigns)
    aux = 0
    for k in range(numCampaigns):
        numImp = index_sizeCamps[k]
        for j in range(numImp):
            budget_spend[k] += vector_qctr[aux]*ext_s[aux]*rho_eval[aux]*x[aux]
            aux+= 1
    return budget_spend


#     ### Solve Primal Optimization Problem Given A Bid Value

# 1) In the case we use the  <=  budget utility function, the indicator alone case, the primal problem is a linear problem
#  and is solved in CalculateLPGurobi.

# 2) In the case we use the  <=  budget plus the l2 penalization the primal problem is a quadratic optimization problem and
#  is solved in CalculateQuadGurobi

# The previous functions will always return a feasible allocation for the primal problem, as the vector
#  $\textbf{x} = \vec{0}$ is always a feasible allocation.

def CalculateLPGurobi(rho_eval, beta_eval, vector_qctr, budget, \
    ext_s, num_impressions, numCampaigns, num_edges, index_Imps, \
    index_sizeCamps, saveObjFn = False, tol = 0.00000001):
   
    #Obj Func Coeff
    insideBracket = vector_qctr-beta_eval
    outsideBracket = np.multiply(ext_s, rho_eval)
    objFnCoeff = np.multiply(insideBracket, outsideBracket)
    # Right hand side. First the budget vector, then a vector 
    # of ones for the sum over the allocation constraints.
    rhs = np.concatenate((budget, np.ones(num_impressions)))
    
    # Matrix A. A is np.concatenate((A_1, A_2))
    # A_1 is the LHS for the budget constraints, and A_2
    # for the sum over allocation constraints.
    A_1 = np.zeros((numCampaigns, num_edges))
    aux = 0
    for k in range(numCampaigns):
        numImp = index_sizeCamps[k]
        for j in range(numImp):
            A_1[k, aux] = vector_qctr[aux]*ext_s[aux]*rho_eval[aux]
            aux+= 1

    A_2 = np.zeros((num_impressions, num_edges))
    for i in range(num_impressions):
        A_2[i, :] = (index_Imps == i).astype(int)
    A = np.concatenate((A_1, A_2))
    
    ### Translation to Gurobi.
    ## Create a model
    model = Model('LP for x')
    model.setParam( 'OutputFlag', False )
    
    ## Variables
    x = [model.addVar(lb = 0.0, ub = 1.0, name = "x" + str(v)) for v in range(num_edges)]
    x = np.array(x).reshape(num_edges)
    
    ## Add the constraints
    for i in range(numCampaigns+num_impressions):
        model.addConstr(lhs = LinExpr(A[i, :].tolist(), x), sense = GRB.LESS_EQUAL, \
        rhs = rhs[i])
        
    ## For stability purposes whenever rho(b_{i, k}) < tol we obligate x_{ik} = 0
    ## (This can be removed if needed)
    to_remove = (rho_eval<tol)
    model.addConstr(lhs = LinExpr((to_remove.astype(int)).tolist(), x), \
        sense = GRB.LESS_EQUAL, rhs = 0)
    
    ## objective Function
    
    model.setObjective(expr = LinExpr(objFnCoeff.tolist(), x), sense = GRB.MAXIMIZE)
    
    ## Solve
    model.optimize()
    
    ## Return variables
    optimalSolution = np.zeros(num_edges)
    for i in range(num_edges):
        optimalSolution[i] = x[i].x
    if saveObjFn:
        toReturn = []
        toReturn.append(optimalSolution)
        toReturn.append(model.objVal)
        return toReturn
    return optimalSolution


def CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
    num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
    tau, addIndicator = False, saveObjFn = False, tol = 0.00000001):
   
    ## First everything done with numpy

    # print('Are all beta_evalRound coordinates greater or equal to zero?')
    if np.sum(beta_eval<0) >0:
        print('When using addIndicator '+str(addIndicator)+' we got some negative beta_evalRound coordinates'+', process id: '+str(os.getpid()))
    if np.sum(rho_eval<0) >0:
        print('When using addIndicator '+str(addIndicator)+' we got some negative rho_evalRound coordinates'+', process id: '+str(os.getpid()))

    ## For numerical issues here the variables will be y[edge] = rctr[edge] s[impType] \rho[impType](b[edge]) x[edge]
    ## Instead of just using 'x'. Also here my rctr is what in the paper is called r.

    vecBudEdg = np.multiply(vector_qctr, np.multiply(ext_s, rho_eval))
    coefProfitPart = np.ones(num_edges) - beta_eval/vector_qctr

    ## The second line is just to avoid division by zero cases
    auxVector  = np.copy(vecBudEdg)
    auxVector[auxVector  <=  tol] = 1.0
    divVecBudEdg = 1.0/auxVector
    

    ### Translation to Gurobi.
    ## Create a model
    model = Model('Quad for x')
    model.setParam( 'OutputFlag', False )
    # model.setParam('PSDTol', 0.00001)
    

    ## Variables
    ## I'll add all variables. I assume that for those which lb = ub = 0.0 Gurobi will eliminate them
    for i in range(num_edges):
        if vecBudEdg[i]  >=  tol:
            model.addVar(lb = 0.0, ub = vecBudEdg[i])
        else:
            model.addVar(lb = 0.0, ub = 0.0)
    model.update()
    vars = model.getVars()    
    
    A_2 = np.zeros((num_impressions, num_edges))
    for i in range(num_impressions):
        A_2[i, :] = ((index_Imps == i).astype(int)) * divVecBudEdg
    for i in range(num_impressions):
        model.addConstr(lhs = LinExpr(A_2[i, :].tolist(), vars), sense = GRB.LESS_EQUAL, rhs = 1.0)
    model.update()
    
    ## list_ExprObjFun will have the different components that are summed together in
    ## objective function. Let's start by adding the profit term. 
    list_ExprObjFun = []
    list_ExprObjFun.append(LinExpr(coefProfitPart.tolist(), vars))
    
    ## Now I create the quadratic part of the objective function and I check if I need 
    ## to add the budget constraints. The latter are added if addIndicator = True.
    aux = 0
    for k in range(numCampaigns):
        numImp = index_sizeCamps[k]
        listV_k = []
        listV_kSquared = []
        for j in range(numImp):
            listV_k.append(vars[aux + j])
            for j2 in range(numImp):
                listV_kSquared.append(vars[aux + j] * vars[aux + j2])
        aux += numImp
        exprV_k = quicksum(listV_k)
        exprV_k2 = quicksum(listV_kSquared)
        consLinear = tau[k] * vector_m[k]
        consQuad = tau[k] * (-0.5)
        list_ExprObjFun.append(consLinear * exprV_k)
        list_ExprObjFun.append(consQuad * exprV_k2)
        ## Add the budget constraints if needed
        if addIndicator:
            model.addConstr(lhs = exprV_k, sense = GRB.LESS_EQUAL, rhs = vector_m[k])
    
    ## objective Function
    
    model.setObjective(expr = quicksum(list_ExprObjFun), sense = GRB.MAXIMIZE)
    # print('All in  CalculateQuadGurobi except .optimize(), '+'process id: '+str(os.getpid()))
    ## Solve
    try:
        model.optimize()
    except GurobiError as e:
        print('model.optimize() raised an exception'+', process id: '+str(os.getpid()))
        print('Error code ' + str(e.errno) + ": " + str(e)+', process id: '+str(os.getpid()))
        print('I will return a vector of zeroes as optimal solution and -9999999999 as the objective function')
        optimalSolution = np.zeros(num_edges)
        if saveObjFn:
            toReturn = []
            toReturn.append(optimalSolution)
            toReturn.append(-9999999999)
            return toReturn
        return optimalSolution
        
    # print('In CalculateQuadGurobi after .optimize(), '+'process id: '+str(os.getpid()))
    if model.status != GRB.OPTIMAL:
        print('The status of the model is', int(model.status))
        print('I will return a vector of zeroes as optimal solution and -99999999 as the objective function')
        optimalSolution = np.zeros(num_edges)
        if saveObjFn:
            toReturn = []
            toReturn.append(optimalSolution)
            toReturn.append(-99999999)
            return toReturn
        return optimalSolution
        
    
    ## Return variables
    optimalSolution = np.zeros(num_edges)
    for i in range(num_edges):
        optimalSolution[i] = vars[i].x * divVecBudEdg[i]
    if saveObjFn:
        toReturn = []
        toReturn.append(optimalSolution)
        ## The following constant appears from the quadratic part
        extraCons = np.sum(-0.5*tau*vector_m*vector_m)
        toReturn.append(float(model.objVal)+extraCons)
        return toReturn
    return optimalSolution

## I wrote the quadratic part as 
## -0.5*sum_{k \in K} (tau[k]*(vector_m[k]^2))*(v_k/vector_m[k]-1)^2

# def CalculateQuadGurobi(rho_eval, beta_eval, vector_qctr, vector_m, ext_s, \
#     num_impressions, numCampaigns, num_edges, index_Imps, index_sizeCamps, \
#     tau, addIndicator = False, saveObjFn = False, tol = 0.000001):

#     rho_evalRound = np.copy(rho_eval)
#     np.round(rho_evalRound, decimals = 5)
#     vector_rctrRound = np.copy(vector_qctr)
#     np.round(vector_rctrRound, decimals = 5)
#     beta_evalRound = np.copy(beta_eval)
#     np.round(beta_evalRound, decimals = 5)
   
#     ## First everything done with numpy

#     # print('Are all beta_evalRound coordinates greater or equal to zero?')
#     if np.sum(beta_evalRound<0) >0:
#         print('When using addIndicator '+str(addIndicator)+' we got some negative beta_evalRound coordinates'+', process id: '+str(os.getpid()))
#     if np.sum(rho_evalRound<0) >0:
#         print('When using addIndicator '+str(addIndicator)+' we got some negative rho_evalRound coordinates'+', process id: '+str(os.getpid()))
#     insideBracket = vector_rctrRound - beta_evalRound
#     outsideBracket = np.multiply(ext_s, rho_evalRound)
#     firstPartObjFn = np.multiply(insideBracket, outsideBracket)
    
#     ## I always need to add the simplex allocation constraints, but the 
#     ## budget constraints are added only if addIndicator = True.
#     ## (which is done late in the code)
#     rhs = np.ones(num_impressions)
    
#     ### Translation to Gurobi.
#     ## Create a model
#     model = Model('Quad for x')
#     model.setParam( 'OutputFlag', False )
#     # model.setParam('PSDTol', 0.00001)
    
#     ## Variables
#     for i in range(num_edges):
#         model.addVar(lb = 0.0, ub = 1.0)
#     model.update()
#     vars = model.getVars()
    
#     A_2 = np.zeros((num_impressions, num_edges))
#     for i in range(num_impressions):
#         A_2[i, :] = (index_Imps == i).astype(int)
#     for i in range(num_impressions):
#         model.addConstr(lhs = LinExpr(A_2[i, :].tolist(), vars), sense = GRB.LESS_EQUAL, rhs = rhs[i])
#     model.update()

#     ## For stability purposes whenever rho(b_{i, k}) < tol we obligate x_{ik} = 0
#     ## (This can be removed if needed)
#     to_remove = (rho_evalRound<tol)
#     model.addConstr(lhs = LinExpr((to_remove.astype(int)).tolist(), vars), \
#         sense = GRB.LESS_EQUAL, rhs = 0)
#     model.update()
#     toNotRemove = (rho_evalRound >=  tol)
    
#     ## list_ExprObjFun will have the different components that are summed together in
#     ## objective function. Let's start by adding the profit term. 
#     list_ExprObjFun = []
#     list_ExprObjFun.append(LinExpr(firstPartObjFn.tolist(), vars))
    
#     ## Now I create the quadratic part of the objective function and I check if I need 
#     ## to add the budget constraints. The latter are added if addIndicator = True.
#     aux = 0
#     for k in range(numCampaigns):
#         numImp = index_sizeCamps[k]
#         listV_k = []
#         for j in range(numImp):
#             if toNotRemove[aux]:
#                 listV_k.append((vector_rctrRound[aux]*ext_s[aux]*rho_evalRound[aux])*vars[aux])
#             aux += 1
#         if len(listV_k)>0:
#             exprV_k = quicksum(listV_k)
#             listV_kSquared = []
#             for j in range(len(listV_k)):
#                 for j2 in range(len(listV_k)):
#                     listV_kSquared.append(listV_k[j]*listV_k[j2])
#             exprV_k2 = quicksum(listV_kSquared)
#             # list_ExprObjFun.append(-0.5 * (np.sqrt(tau[k]) * vector_m[k] - np.sqrt(tau[k]) * exprV_k) \
#             #     * (np.sqrt(tau[k]) * vector_m[k] - np.sqrt(tau[k]) * exprV_k ))
#             consLinear = tau[k] * vector_m[k]
#             consQuad = tau[k] * (-0.5)
#             # list_ExprObjFun.append(cte * (vector_m[k] - exprV_k) * (vector_m[k] - exprV_k))
#             list_ExprObjFun.append(consLinear * exprV_k)
#             list_ExprObjFun.append(consQuad * exprV_k2)
#             ## Add the budget constraints if needed
#             if addIndicator:
#                 ## The multiplier *0.0001 is used for stability.
#                 model.addConstr(lhs = exprV_k, sense = GRB.LESS_EQUAL, rhs = vector_m[k])
    
#     ## objective Function
    
#     model.setObjective(expr = quicksum(list_ExprObjFun), sense = GRB.MAXIMIZE)
#     print('All in  CalculateQuadGurobi except .optimize(), '+'process id: '+str(os.getpid()))
#     ## Solve
#     try:
#         model.optimize()
#     except GurobiError as e:
#         print('model.optimize() raised an exception'+', process id: '+str(os.getpid()))
#         print('Error code ' + str(e.errno) + ": " + str(e)+', process id: '+str(os.getpid()))
#         print('I will return a vector of zeroes as optimal solution and -9999999999 as the objective function')
#         optimalSolution = np.zeros(num_edges)
#         if saveObjFn:
#             toReturn = []
#             toReturn.append(optimalSolution)
#             toReturn.append(-9999999999)
#             return toReturn
#         return optimalSolution
        
#     print('In CalculateQuadGurobi after .optimize(), '+'process id: '+str(os.getpid()))
#     if model.status != GRB.OPTIMAL:
#         print('The status of the model is', int(model.status))
#         print('I will return a vector of zeroes as optimal solution and 99999999 as the objective function')
#         optimalSolution = np.zeros(num_edges)
#         if saveObjFn:
#             toReturn = []
#             toReturn.append(optimalSolution)
#             toReturn.append(99999999)
#             return toReturn
#         return optimalSolution
        
    
#     ## Return variables
#     optimalSolution = np.zeros(num_edges)
#     for i in range(num_edges):
#         optimalSolution[i] = vars[i].x
#     if saveObjFn:
#         toReturn = []
#         toReturn.append(optimalSolution)
#         toReturn.append(model.objVal)
#         return toReturn
#     return optimalSolution
