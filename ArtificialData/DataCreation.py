import numpy as np
import sys

def createGraph_0(maxCampaign,scoresImp,num_impressions):
    ## Paul works with the order i,k and I work with
    ## k,i so I need to make extra steps. 
    num_edges=0
    numCampaigns=0
    numCampPerImp=np.zeros(num_impressions)
    campPerImp=[]
    ## First we say for each impression which campaigns are associated to it.
    for i in range(num_impressions):
        numCampPerImp[i]=np.maximum(np.random.binomial(maxCampaign,\
            scoresImp[i]),1)
        campPerImp.append(np.sort(np.random.choice(maxCampaign,\
            int(numCampPerImp[i]),replace=False)))
    
    ## Now I need to say which impressions are associated to which campaign
    ## First we create a list of MaxCampaigns number of lists. 
    tempCampImpDic={}
    for i in range(maxCampaign):
        tempCampImpDic[i]=[]
        
    for i in range(num_impressions):
        for number in campPerImp[i]:
            tempCampImpDic[number].append(i)
    # Now build the vectors we need.
    # I start by counting the campaigns that have associated
    # at least one impression.
    non_nullCamp=maxCampaign
    for i in range(maxCampaign):
        if (len(tempCampImpDic[i])==0):
            non_nullCamp-=1
    num_edges=int(np.sum(numCampPerImp))
    numCampaigns=non_nullCamp
    index_sizeCamps=np.zeros(numCampaigns)
    index_startCamp=np.zeros(numCampaigns)
    index_Imps=np.zeros(num_edges)
    aux=0 # General count index
    aux_camp=0 # Count of the non Null Campaigns
    for i in range(maxCampaign):
        size_camp=len(tempCampImpDic[i])
        if size_camp!=0:
            index_sizeCamps[aux_camp]=size_camp
            for j in range(size_camp):
                index_Imps[aux+j]=tempCampImpDic[i][j]
            if(aux_camp<(numCampaigns-1)):
                index_startCamp[aux_camp+1]=index_startCamp[aux_camp]+size_camp
            aux_camp+=1
            aux+=size_camp
    return [numCampaigns,num_edges,index_Imps.astype(int),\
            index_sizeCamps.astype(int),index_startCamp.astype(int)]


def createGraph(maxCampaign,num_impressions,graph_type, *args, **kwargs):
    # For option 0
    # args[0] is scoresImp
    if graph_type==0:
        return createGraph_0(maxCampaign,args[0],num_impressions)
    else:
        sys.exit("Not a valid Graph Type Creation Option")


def createBudgets_0(numCampaigns,constant):
    return np.ones(numCampaigns)*constant

def createValuations_0(num_edges,constant):
    return np.ones(num_edges)*constant

def createNumImp_0(num_impressions,constant):
    return np.ones(num_impressions)*constant
    
def createNumAdvPaul(num_impressions,scoresImp,uB_adv):
    adverPerImp=np.zeros(num_impressions)
    for i in range(num_impressions):
        adverPerImp[i]=np.maximum(np.random.binomial(uB_adv,\
            scoresImp[i]),1)
    return adverPerImp
    
def createOtherPlayersData_0(num_impressions,scoresImp,max_adversaries, maxBid):
    return [np.ones(num_impressions)* maxBid,\
    createNumAdvPaul(num_impressions,scoresImp,max_adversaries)]

def createCtr_0(numCampaigns,num_edges,index_Imps,index_sizeCamps,\
             scoresImp,scoresCam):
    matCtr=np.multiply(np.expand_dims(scoresCam,axis=1),\
                  np.expand_dims(scoresImp,axis=0))
    ctr=np.zeros(num_edges)
    aux=0
    for i in range(numCampaigns):
        impInCamp=index_sizeCamps[i]
        for j in range(impInCamp):
            ctr[aux+j]=matCtr[i,index_Imps[aux+j]]
        aux+=impInCamp
    return ctr

def createCtr_1(numCampaigns,num_edges,index_Imps,index_sizeCamps,\
             scoresImp,scoresCam):
    ctr=np.zeros(num_edges)
    aux=0
    for i in range(numCampaigns):
        impInCamp=index_sizeCamps[i]
        for j in range(impInCamp):
            imp=index_Imps[aux+j]
            ctr[aux+j]=(scoresCam[i]+scoresImp[imp])*0.5
        aux+=impInCamp
    return ctr

def createVector_m(numCampaigns, budget_Type, *args, **kwargs):
    # For option 0 
    ## args[0] is the constant
    if budget_Type==0:
        return createBudgets_0(numCampaigns,args[0])
    else:
        sys.exit("Not a valid Budget Creation Type")
        
def createVector_r(num_edges, valuation_Type, *args, **kwargs):
    # For option 0
    ## args[0] is a constant
    if valuation_Type ==0:
        return createValuations_0(num_edges,args[0])
    else:
        sys.exit("Not a valid Valuation Creation Type")

def createVector_s(num_impressions, numImp_Type, *args, **kwargs):
    # For option 0 
    ## args[0] is the constant
    if numImp_Type==0:
        return createNumImp_0(num_impressions,args[0])
    else:
        sys.exit("Not a valid Num Impression Creation Type")

### First auxiliary fn's

def createScores(numCamps,num_impressions):
    scoresImp=np.random.uniform(size=num_impressions)
    scoresCam=np.random.uniform(size=numCamps)
    return [scoresImp,scoresCam]

def extendSizeCamps(vecToExtend,vector_rep):
    return np.repeat(vecToExtend,vector_rep)

## Args here is a list of lists.
# args[0] is for the graph, args[1] for ctr and then for vectors m r s
def createInitialData(maxCampaign, num_impressions, graph_type, ctr_Type, \
    othersData, budget_Type, valuation_Type, numImp_Type, *args, **kwargs):
    ## Both maxCampaign and Num_impression defined outside.

    ## I start by giving everything a 0 default value
    [numCampaigns,num_edges,index_Imps,index_sizeCamps,index_startCamps,\
    vector_m,vector_r,vector_s,ext_s,adverPerImp,lb_bid_others_vec,\
    ub_bid_others_vec,vector_ctr,vector_rctr]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    ## I first create the scoresImp because I may need it for the graph
    ## I will put a default value to scoresCam for now
    scoresImp=createScores(maxCampaign,num_impressions)[0]
    scoresCam=0
    
    ## Sadly for creating the graph in Paul's way, ctr and others data i Need 
    ## scoresImp that I cannot pass through the *args
    ## so I need to put the if and else here. 
    ## Graph Creation
    if graph_type==0:
        [numCampaigns,num_edges,index_Imps,index_sizeCamps,index_startCamps]=\
        createGraph(maxCampaign,num_impressions,graph_type,scoresImp)
        scoresCam=createScores(numCampaigns,num_impressions)[1]
    else:
        sys.exit('Graph Type not supported')
    ## Ctr Creation
    if ctr_Type==0:
        vector_ctr = createCtr_0(numCampaigns,num_edges,index_Imps,\
                    index_sizeCamps,scoresImp,scoresCam)
    elif ctr_Type==1:
        vector_ctr = createCtr_1(numCampaigns,num_edges,index_Imps,\
                    index_sizeCamps,scoresImp,scoresCam)
    else:
        sys.exit("Not a valid Ctr Creation Type")
    
    # Let's create the vector r, s and then the ctr vector. 
    vector_m = createVector_m(numCampaigns, budget_Type, *args[0][2])
    vector_r = createVector_r(num_edges, valuation_Type, *args[0][3])
    vector_s = createVector_s(num_impressions, numImp_Type, *args[0][4])
    ext_s = vector_s[index_Imps] 
    vector_rctr=np.multiply(vector_r,vector_ctr)
    
    if othersData==0:
        [UB_bids,adverPerImp]=\
        createOtherPlayersData_0(num_impressions,scoresImp,*args[0][5])
    else:
        sys.exit("Not a valid Others Data Creation Type")

    return [numCampaigns,num_edges,index_Imps,index_sizeCamps,index_startCamps,\
           vector_m,vector_r,vector_s,ext_s,adverPerImp,UB_bids,vector_ctr,vector_rctr]