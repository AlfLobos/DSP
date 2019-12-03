import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math


def CreateTableSensitivityWoutL2(dictInd, dictL2Ind, dictGr, seq_vectorM, sim):
    tableToCsv = np.zeros((sim*len(seq_vectorM), 23))
    for z, vector_m in enumerate(seq_vectorM):
        budPerNode = vector_m[0]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = budPerNode
            tableToCsv[i+z*sim, 1] = i
            tableToCsv[i+z*sim, 2] = np.sum(dictInd[budPerNode][i][6])
            tableToCsv[i+z*sim, 3] = np.sum(dictInd[budPerNode][i][5])
            tableToCsv[i+z*sim, 4] = np.sum(dictInd[budPerNode][i][4])
            tableToCsv[i+z*sim, 5] = np.sum(dictInd[budPerNode][i][1])
            tableToCsv[i+z*sim, 6] = np.sum(dictInd[budPerNode][i][2])
            tableToCsv[i+z*sim, 7] = np.sum(dictInd[budPerNode][i][3])
            tableToCsv[i+z*sim, 8] = (np.sum(vector_m)-np.sum(dictInd[budPerNode][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 9] = np.sum(dictL2Ind[budPerNode][i][6])
            tableToCsv[i+z*sim, 10] = np.sum(dictL2Ind[budPerNode][i][5])
            tableToCsv[i+z*sim, 11] = np.sum(dictL2Ind[budPerNode][i][4])
            tableToCsv[i+z*sim, 12] = np.sum(dictL2Ind[budPerNode][i][1])
            tableToCsv[i+z*sim, 13] = np.sum(dictL2Ind[budPerNode][i][2])
            tableToCsv[i+z*sim, 14] = np.sum(dictL2Ind[budPerNode][i][3])
            tableToCsv[i+z*sim, 15] = (np.sum(vector_m)-np.sum(dictL2Ind[budPerNode][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 16] = np.sum(dictGr[budPerNode][i][6])
            tableToCsv[i+z*sim, 17] = np.sum(dictGr[budPerNode][i][5])
            tableToCsv[i+z*sim, 18] = np.sum(dictGr[budPerNode][i][4])
            tableToCsv[i+z*sim, 19] = np.sum(dictGr[budPerNode][i][1])
            tableToCsv[i+z*sim, 20] = np.sum(dictGr[budPerNode][i][2])
            tableToCsv[i+z*sim, 21] = np.sum(dictGr[budPerNode][i][3])
            tableToCsv[i+z*sim, 22] = (np.sum(vector_m)-np.sum(dictGr[budPerNode][i][0]))/(np.sum(vector_m))
    return tableToCsv

def CreateTableSensitivity(dictInd, dictL2, dictL2Ind, dictGr, seq_vectorM, sim):
    tableToCsv = np.zeros((sim*len(seq_vectorM), 30))
    for z, vector_m in enumerate(seq_vectorM):
        budPerNode = vector_m[0]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = budPerNode
            tableToCsv[i+z*sim, 1] = i
            tableToCsv[i+z*sim, 2] = np.sum(dictInd[z][i][6])
            tableToCsv[i+z*sim, 3] = np.sum(dictInd[z][i][5])
            tableToCsv[i+z*sim, 4] = np.sum(dictInd[z][i][4])
            tableToCsv[i+z*sim, 5] = np.sum(dictInd[z][i][1])
            tableToCsv[i+z*sim, 6] = np.sum(dictInd[z][i][2])
            tableToCsv[i+z*sim, 7] = np.sum(dictInd[z][i][3])
            tableToCsv[i+z*sim, 8] = (np.sum(vector_m)-np.sum(dictInd[z][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 9] = np.sum(dictL2[z][i][6])
            tableToCsv[i+z*sim, 10] = np.sum(dictL2[z][i][5])
            tableToCsv[i+z*sim, 11] = np.sum(dictL2[z][i][4])
            tableToCsv[i+z*sim, 12] = np.sum(dictL2[z][i][1])
            tableToCsv[i+z*sim, 13] = np.sum(dictL2[z][i][2])
            tableToCsv[i+z*sim, 14] = np.sum(dictL2[z][i][3])
            tableToCsv[i+z*sim, 15] = (np.sum(vector_m)-np.sum(dictL2[z][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 16] = np.sum(dictL2Ind[z][i][6])
            tableToCsv[i+z*sim, 17] = np.sum(dictL2Ind[z][i][5])
            tableToCsv[i+z*sim, 18] = np.sum(dictL2Ind[z][i][4])
            tableToCsv[i+z*sim, 19] = np.sum(dictL2Ind[z][i][1])
            tableToCsv[i+z*sim, 20] = np.sum(dictL2Ind[z][i][2])
            tableToCsv[i+z*sim, 21] = np.sum(dictL2Ind[z][i][3])
            tableToCsv[i+z*sim, 22] = (np.sum(vector_m)-np.sum(dictL2Ind[z][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 23] = np.sum(dictGr[z][i][6])
            tableToCsv[i+z*sim, 24] = np.sum(dictGr[z][i][5])
            tableToCsv[i+z*sim, 25] = np.sum(dictGr[z][i][4])
            tableToCsv[i+z*sim, 26] = np.sum(dictGr[z][i][1])
            tableToCsv[i+z*sim, 27] = np.sum(dictGr[z][i][2])
            tableToCsv[i+z*sim, 28] = np.sum(dictGr[z][i][3])
            tableToCsv[i+z*sim, 29] = (np.sum(vector_m)-np.sum(dictGr[z][i][0]))/(np.sum(vector_m))
    return tableToCsv

def CreateTableParetoL2_L2Ind_Gr(dictL2, dictL2Ind, dictGr, vector_m, multL2, multGr, sim):
    tableToCsv = np.zeros((sim*len(multL2), 24))
    for z in range(len(multL2)):
        l2Mult = multL2[z]
        grMult = multGr[z]
        for i in range(sim):
            tableToCsv[i+z*sim, 0] = l2Mult
            tableToCsv[i+z*sim, 1] = grMult
            tableToCsv[i+z*sim, 2] = i
            tableToCsv[i+z*sim, 3] = np.sum(dictL2[l2Mult][i][6])
            tableToCsv[i+z*sim, 4] = np.sum(dictL2[l2Mult][i][5])
            tableToCsv[i+z*sim, 5] = np.sum(dictL2[l2Mult][i][4])
            tableToCsv[i+z*sim, 6] = np.sum(dictL2[l2Mult][i][1])
            tableToCsv[i+z*sim, 7] = np.sum(dictL2[l2Mult][i][2])
            tableToCsv[i+z*sim, 8] = np.sum(dictL2[l2Mult][i][3])
            tableToCsv[i+z*sim, 9] = (np.sum(vector_m)-np.sum(dictL2[l2Mult][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 10] = np.sum(dictL2Ind[l2Mult][i][6])
            tableToCsv[i+z*sim, 11] = np.sum(dictL2Ind[l2Mult][i][5])
            tableToCsv[i+z*sim, 12] = np.sum(dictL2Ind[l2Mult][i][4])
            tableToCsv[i+z*sim, 13] = np.sum(dictL2Ind[l2Mult][i][1])
            tableToCsv[i+z*sim, 14] = np.sum(dictL2Ind[l2Mult][i][2])
            tableToCsv[i+z*sim, 15] = np.sum(dictL2Ind[l2Mult][i][3])
            tableToCsv[i+z*sim, 16] = (np.sum(vector_m)-np.sum(dictL2Ind[l2Mult][i][0]))/(np.sum(vector_m))
            tableToCsv[i+z*sim, 17] = np.sum(dictGr[grMult][i][6])
            tableToCsv[i+z*sim, 18] = np.sum(dictGr[grMult][i][5])
            tableToCsv[i+z*sim, 19] = np.sum(dictGr[grMult][i][4])
            tableToCsv[i+z*sim, 20] = np.sum(dictGr[grMult][i][1])
            tableToCsv[i+z*sim, 21] = np.sum(dictGr[grMult][i][2])
            tableToCsv[i+z*sim, 22] = np.sum(dictGr[grMult][i][3])
            tableToCsv[i+z*sim, 23] = (np.sum(vector_m)-np.sum(dictGr[grMult][i][0]))/(np.sum(vector_m))
    return tableToCsv


# def CreateTableParetoInd(dataSP, dataFP, vector_m, sim):
#     toRet = []
#     for i in range(sim):
#         toRet.append([i, 'SP', np.sum(dataSP[i][6]), np.sum(dataSP[i][5]), np.sum(dataSP[i][4]), \
#         np.sum(dataSP[i][1]), np.sum(dataSP[i][2]), np.sum(dataSP[i][3]), \
#         ((np.sum(vector_m)-np.sum(dataSP[i][0]))/(np.sum(vector_m)))])
#     for i in range(sim):
#         toRet.append([i, 'FP', np.sum(dataFP[i][6]), np.sum(dataFP[i][5]), np.sum(dataFP[i][4]), \
#         np.sum(dataFP[i][1]), np.sum(dataFP[i][2]), np.sum(dataFP[i][3]), \
#         ((np.sum(vector_m)-np.sum(dataFP[i][0]))/(np.sum(vector_m)))])
#     return toRet

def CreateTableParetoInd(data, vector_m, sim, name = "FP"):
    toRet = []
    for i in range(sim):
        toRet.append([i, np.sum(data[i][6]), np.sum(data[i][5]), np.sum(data[i][4]), \
        np.sum(data[i][1]), np.sum(data[i][2]), np.sum(data[i][3]), \
        ((np.sum(vector_m)-np.sum(data[i][0]))/(np.sum(vector_m)))])
    return toRet

