import numpy as np

def CalcRhoAndBetaVectors(bid_vec, UB_bid, num_edges, index_Imps, adverPerImp, firstPrice):
    ## I will assume I want to evaluate the full vector.
    rhoBetaMat=np.zeros((num_edges,2))
    for edge_num,impType in enumerate(index_Imps):
        rhoBetaMat[edge_num,:]=RhoBetaValue(bid_vec[edge_num], UB_bid[impType],\
            adverPerImp[impType], firstPrice)
    return [rhoBetaMat[:,0],rhoBetaMat[:,1]]

def CalcBeta(bid, num_adv, firstPrice):
    if firstPrice:
        return bid
    else:
        return (num_adv/(num_adv+1.0)) * bid

def RhoBetaValue(bid, ub,  n, firstPrice):
    ## For rho_beta_Type=0, args[0]=adv
    rho = np.power((bid/ub),n)
    beta = CalcBeta(bid, n, firstPrice)
    return [rho, beta]