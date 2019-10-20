'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import scipy.sparse as sp
import gc
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_num_users = None
_num_items = None
_path_ulul = None
_path_ulll = None
_path_uuul = None
_path_uull = None
_path_nums = None
_jumps = None
_length = None

_user_feature = None
_item_feature = None
_type_feature = None
_features = None
def evaluate_model(model, user_feature, item_feature, num_users, num_items, path_ulul, path_ulll, path_uuul, path_uull, path_nums, jumps, length, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _num_users
    global _num_items
    global _path_ulul
    global _path_ulll
    global _path_uuul
    global _path_uull
    global _path_nums
    global _jumps
    global _length

    global _user_feature
    global _item_feature
    global _features

    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _num_users = num_users
    _num_items = num_items
    _path_ulul = path_ulul
    _path_ulll = path_ulll
    _path_uuul = path_uuul
    _path_uull = path_uull
    _path_nums = path_nums
    _jumps = jumps
    _length = length

    _user_feature = user_feature
    _item_feature = item_feature
    _features = [user_feature, item_feature]
    hrs, ndcgs, hrs1, ndcgs1, hrs2, ndcgs2 = [], [], [] , [], [], []
    
    for idx in range(len(_testRatings)):
        if idx % 1 == 0:
            (hr, ndcg, hr1, ndcg1, hr2 ,ndcg2) = eval_one_rating(idx)
            hrs.append(hr)
            ndcgs.append(ndcg)    
            hrs1.append(hr1)
            ndcgs1.append(ndcg1)    
            hrs2.append(hr2)
            ndcgs2.append(ndcg2)    
    '''
        
    if np.array(ps).mean() > 0.4:
        print('wrong predication')
        for idx in range(len(_testRatings)):
            if idx < 10:
                (p, r, ndcg) = eval_one_rating1(idx)
               
    
    ps1 = np.array(ps).mean()
    if ps1 > 0.3:
        print('wrong prediction:')
        for idx in range(len(_testRatings)):
             p = eval_one_rating1(idx)
        return 0
    '''    
    return (ps, ndcgs, ps1, ndcgs1, ps2, ndcgs2)



def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItems = rating[1:]
    
    # Get prediction scores
    map_item_score = {}
    user_input = []
    item_input = []
    user_fe = np.zeros((len(items),  8))
    item_fe = np.zeros((len(items),  8))
    ulul_input = np.zeros((len(items), _path_nums[0], _jumps[0], _length))
    ulll_input = np.zeros((len(items), _path_nums[1], _jumps[1], _length))
    uuul_input = np.zeros((len(items), _path_nums[2], _jumps[2], _length)) 
    uull_input = np.zeros((len(items), _path_nums[3], _jumps[3], _length)) 
    k = 0
 
    for i in items:
       
        user_input.append(u)
        item_input.append(i)
        user_fe[k] = _user_feature[u]
        item_fe[k] = _item_feature[i]
        if (u, i) in _path_ulul: 
            for p_i in range(len(_path_ulul[(u, i)])):
                for p_j in range(len(_path_ulul[(u, i)][p_i])):
                    type_id = _path_ulul[(u, i)][p_i][p_j][0]
                    index = _path_ulul[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        ulul_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        ulul_input[k][p_i][p_j] = _item_feature[index]
            
        if (u, i) in _path_ulll:
            for p_i in range(len(_path_ulll[(u, i)])):
                for p_j in range(len(_path_ulll[(u, i)][p_i])):
                    type_id = _path_ulll[(u, i)][p_i][p_j][0]
                    index = _path_ulll[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        ulll_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        ulll_input[k][p_i][p_j] = _item_feature[index]
                   
        if (u, i) in _path_uuul:
            for p_i in range(len(_path_uuul[(u, i)])):
                for p_j in range(len(_path_uuul[(u, i)][p_i])):
                    type_id = _path_uuul[(u, i)][p_i][p_j][0]
                    index = _path_uuul[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        uuul_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        uuul_input[k][p_i][p_j] = _item_feature[index]
                        
        if (u, i) in _path_uull:
            for p_i in range(len(_path_uull[(u, i)])):
                for p_j in range(len(_path_uull[(u, i)][p_i])):
                    type_id = _path_uull[(u, i)][p_i][p_j][0]
                    index = _path_uull[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        uuul_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        uuul_input[k][p_i][p_j] = _item_feature[index]
        k += 1

    
    predictions = _model.predict([np.array(user_input), np.array(item_input), ulul_input, ulll_input, uuul_input, uull_input], 
                                 batch_size = 256, verbose = 0)
    
    
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i] 
    '''
    ii = 0
    for i in items:
        map_item_score[i] = predictions[ii]
        ii += 1
    '''
    #items.pop()
    # Evaluate top rank list
    
    ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
    hr = getHR(ranklist[:3], gtItems,items)
    ndcg = getNDCG(ranklist[:3], gtItems)
    hr1 = getHR(ranklist[:5], gtItems,items)
    ndcg1 = getNDCG(ranklist[:5], gtItems)
    hr2 = getHR(ranklist, gtItems,items)   
    ndcg2 = getNDCG(ranklist[:10], gtItems)
    return (hr, ndcg, hr1, ndcg1, hr2 ,ndcg2)



def getHR(ranklist, gtItems,items):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
      
    return p * 1.0

def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)


def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return  dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg 

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg
