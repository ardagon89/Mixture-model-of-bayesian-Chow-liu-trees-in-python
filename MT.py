#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
import copy
import time
import sys
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def loadfile(filename1, filename2=None):
    ds1 = np.loadtxt(filename1, delimiter=",", dtype=int)
    if filename2:
        ds2 = np.loadtxt(filename2, delimiter=",", dtype=int)
        ds = np.vstack((ds1, ds2))
    else:
        ds = ds1
    return ds, ds.shape[0], ds.shape[1]

def count_matrix(ds, m, n):
    prob_xy = np.zeros((n, n, 4))
    for i in range(n):
        subds = ds[ds[:, i] == 0]
        for j in range(n):
            if prob_xy[i, j, 0] == 0:
                prob_xy[i, j, 0] = (subds[subds[:, j] == 0].shape[0]+1)/(m+4)
            if prob_xy[j, i, 0] == 0:
                prob_xy[j, i, 0] = prob_xy[i, j, 0]
            if prob_xy[i, j, 1] == 0:
                prob_xy[i, j, 1] = (subds[subds[:, j] == 1].shape[0]+1)/(m+4)
            if prob_xy[j, i, 2] == 0:
                prob_xy[j, i, 2] = prob_xy[i, j, 1]
            
        subds = ds[ds[:, i] == 1]
        for j in range(n):
            if prob_xy[i, j, 2] == 0:
                prob_xy[i, j, 2] = (subds[subds[:, j] == 0].shape[0]+1)/(m+4)
            if prob_xy[j, i, 1] == 0:
                prob_xy[j, i, 1] = prob_xy[i, j, 2]
            if prob_xy[i, j, 3] == 0:
                prob_xy[i, j, 3] = (subds[subds[:, j] == 1].shape[0]+1)/(m+4)
            if prob_xy[j, i, 3] == 0:
                prob_xy[j, i, 3] = prob_xy[i, j, 3]
    return prob_xy

def prob_matrix(ds, m, n, k=0):
    prob_xy = np.zeros((n, n, 4))
    l = 1
    for i in range(n):
        subds = ds[ds[:, i] == 0]
        for j in range(n):
            if prob_xy[i, j, 0] == 0:
                prob_xy[i, j, 0] = (np.sum(subds[subds[:, j] == 0][:, n+k]))
                cnt = subds[subds[:, j] == 0].shape[0]
                l = prob_xy[i, j, 0]/cnt if cnt>0 and prob_xy[i, j, 0]/cnt < l else l
            if prob_xy[j, i, 0] == 0:
                prob_xy[j, i, 0] = prob_xy[i, j, 0]
            if prob_xy[i, j, 1] == 0:
                prob_xy[i, j, 1] = (np.sum(subds[subds[:, j] == 1][:, n+k]))
                cnt = subds[subds[:, j] == 1].shape[0]
                l = prob_xy[i, j, 1]/cnt if cnt>0 and prob_xy[i, j, 1]/cnt < l else l
            if prob_xy[j, i, 2] == 0:
                prob_xy[j, i, 2] = prob_xy[i, j, 1]
            
        subds = ds[ds[:, i] == 1]
        for j in range(n):
            if prob_xy[i, j, 2] == 0:
                prob_xy[i, j, 2] = (np.sum(subds[subds[:, j] == 0][:, n+k]))
                cnt = subds[subds[:, j] == 0].shape[0]
                l = prob_xy[i, j, 2]/cnt if cnt>0 and prob_xy[i, j, 2]/cnt < l else l
            if prob_xy[j, i, 1] == 0:
                prob_xy[j, i, 1] = prob_xy[i, j, 2]
            if prob_xy[i, j, 3] == 0:
                prob_xy[i, j, 3] = (np.sum(subds[subds[:, j] == 1][:, n+k]))
                cnt = subds[subds[:, j] == 1].shape[0]
                l = prob_xy[i, j, 3]/cnt if cnt>0 and prob_xy[i, j, 3]/cnt < l else l
            if prob_xy[j, i, 3] == 0:
                prob_xy[j, i, 3] = prob_xy[i, j, 3]
    return (prob_xy+l)/(m+4*l)

def mutual_info(prob_xy, n):
    I_xy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                I_xy[i, j] = prob_xy[i, j, 0]*np.log(prob_xy[i, j, 0]/(prob_xy[i, i, 0]*prob_xy[j, j, 0]))                 + prob_xy[i, j, 1]*np.log(prob_xy[i, j, 1]/(prob_xy[i, i, 0]*prob_xy[j, j, 3]))                 + prob_xy[i, j, 2]*np.log(prob_xy[i, j, 2]/(prob_xy[i, i, 3]*prob_xy[j, j, 0]))                 + prob_xy[i, j, 3]*np.log(prob_xy[i, j, 3]/(prob_xy[i, i, 3]*prob_xy[j, j, 3]))
    return I_xy

def draw_tree_old(edge_wts, prnt = False):
    edge_wts_cp = copy.deepcopy(edge_wts)
    edges = [np.unravel_index(np.argmax(edge_wts_cp), edge_wts_cp.shape)]
    visited = [[edges[-1][0],edges[-1][1]]]
    edge_wts_cp[edges[-1]] = 0
    while(len(edges) < edge_wts.shape[0]-1):
        i = j = -1
        edge = np.unravel_index(np.argmax(edge_wts_cp), edge_wts_cp.shape)
        for bag in range(len(visited)):
            if edge[0] in visited[bag]:
                i = bag
            if edge[1] in visited[bag]:
                j = bag
        if i == -1 and j != -1:
            edges.append(edge)
            visited[j].append(edge[0])
        elif i != -1 and j == -1:
            edges.append(edge)
            visited[i].append(edge[1])
        elif i == -1 and j == -1:
            edges.append(edge)
            visited.append([edge[0], edge[1]])
        elif i != -1 and j != -1 and i != j:
            edges.append(edge)
            visited[i] += visited[j]
            visited.remove(visited[j])
        elif i == j != -1:
            pass
        else:
            #pass
            print("Discarded in else", edge)
        edge_wts_cp[edge] = 0
    
    new_tree = []
    make_tree(edges, new_tree, edges[0][0])
    
    return new_tree

def draw_tree(edge_wts, prnt = False, k=0, step=0):
    edge_wts_cp = 1/copy.deepcopy(edge_wts)
    X = csr_matrix(edge_wts_cp)
    Tcsr = minimum_spanning_tree(X)
    edges1 = [(item[0],item[1]) for item in np.transpose(np.nonzero(Tcsr.toarray()))]
    new_tree1 = []
    make_tree(edges1, new_tree1, edges1[0][0])
    return new_tree1

def make_tree(ls, new_tree, parent):
    for node in [item for item in ls if parent in item]:
        if node[0] == parent:
            new_tree.append(node)
            ls.remove(node)
            make_tree(ls, new_tree, node[1])
        else:
            new_tree.append((node[1],node[0]))
            ls.remove(node)
            make_tree(ls, new_tree, node[0])

def count_matrix(ds, tree, cols):
    count_xy = np.zeros((len(tree), cols))
    for idx, node in enumerate(tree):
        i, j = node
        count_xy[idx] = [ds[(ds[:, i]==0) & (ds[:, j]==0)].shape[0], ds[(ds[:, i]==0) & (ds[:, j]==1)].shape[0], ds[(ds[:, i]==1) & (ds[:, j]==0)].shape[0], ds[(ds[:, i]==1) & (ds[:, j]==1)].shape[0]]
    return count_xy

def exist_matrix(ds, tree, cols):
    rows = ds.shape[0]
    exist_xy = np.zeros((rows, len(tree), cols))
    for idx, node in enumerate(tree):
        i, j = node
        exist_xy[:,idx,:] = np.hstack((((ds[:, i]==0) & (ds[:, j]==0)).astype(int).reshape(rows,1), ((ds[:, i]==0) & (ds[:, j]==1)).astype(int).reshape(rows,1), ((ds[:, i]==1) & (ds[:, j]==0)).astype(int).reshape(rows,1), ((ds[:, i]==1) & (ds[:, j]==1)).astype(int).reshape(rows,1)))
    return exist_xy

def M_step(ds, m, n, k, step, prnt, pk):
    trees = []
    cond_probs = []
    for ki in range(k):
        prob_xy = prob_matrix(ds, m, n, ki)
        I_xy = mutual_info(prob_xy, n)        
        tree = draw_tree(I_xy, prnt, ki, step)
        tree = [(tree[0][0], tree[0][0])] + tree
        trees.append(tree)
        cond_prob = np.zeros((len(tree), prob_xy.shape[2]))
        for idx, node in enumerate(tree):
            if node[0] == node[1]:
                cond_prob[idx] = prob_xy[node[0], node[1],:]
            else:
                cond_prob[idx] = np.hstack(((prob_xy[node[0], node[1],:2]/prob_xy[node[0], node[0], 0]),(prob_xy[node[0], node[1],2:]/prob_xy[node[0], node[0], 3])))
        cond_probs.append(cond_prob)
    return trees, cond_probs

def random_init(ds, m, n, k, step, prnt):
    trees = []
    cond_probs = []
    for ki in range(k):
        prob_xy = prob_matrix(ds[np.random.choice(m, 8, replace=False), :], m, n, ki)
        I_xy = mutual_info(prob_xy, n)        
        tree = draw_tree(I_xy, prnt, ki, step)
        tree = [(tree[0][0], tree[0][0])] + tree
        trees.append(tree)
        cond_prob = np.zeros((len(tree), prob_xy.shape[2]))
        for idx, node in enumerate(tree):
            if node[0] == node[1]:
                cond_prob[idx] = prob_xy[node[0], node[1],:]
            else:
                cond_prob[idx] = np.hstack(((prob_xy[node[0], node[1],:2]/prob_xy[node[0], node[0], 0]),(prob_xy[node[0], node[1],2:]/prob_xy[node[0], node[0], 3])))
        cond_probs.append(cond_prob)
    return trees, cond_probs

def E_step(ds, m, n, k, trees, cond_probs):
    ph = ds[:, n:].sum(axis = 0)/m
    weight_ij = np.zeros((m, k))
    minval = np.full((m,1),10**-100)
    
    for j in range(k):
        weight_ij[:, j] = np.max(np.hstack(((ph[j] * np.sum(exist_matrix(ds, trees[j], 4)*cond_probs[j], axis = 2).prod(axis=1)).reshape(m,1),minval)),axis=1)

    weight_ij = weight_ij/np.sum(weight_ij, axis = 1).reshape(m, 1)
    ds[:, n:] = weight_ij
    pk = None
    return ds, pk

if __name__ == "__main__":
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    if len(sys.argv) != 5:
        print("Usage:python MT.py <k> <training-dataset> <validation-dataset> <testing-dataset>")
    else:
        train, m, n = loadfile(sys.argv[2],sys.argv[3])
        ts, m1, n1 = loadfile(sys.argv[4])
        k = int(sys.argv[1])
        print("k="+str(k))
        LL = []
        for x in range(10):
            weight_ij = np.random.rand(m, k)
            weight_ij = weight_ij/np.sum(weight_ij, axis = 1).reshape(m, 1)
            ds = np.hstack((train, weight_ij))
            trees, cond_probs = random_init(ds, m, n, k, 0, False)
            old_ph = [0]*k

            for step in range(1, 100): 
                ds, pk = E_step(ds, m, n, k, trees, cond_probs)
                trees, cond_probs = M_step(ds, m, n, k, step, False, pk)   
                new_ph = ds[:, n:].sum(axis = 0)/m
                if (np.abs(old_ph - new_ph)==0.00000).all():
                    break
                old_ph = new_ph

            lambda_k = ds[:, n:].sum(axis = 0)/m
            L = 0
            result = np.zeros((m1,k))
            for j in range(k):
                result[:,j] = np.log(np.sum(exist_matrix(ts, trees[j], 4)*cond_probs[j], axis = 2)).sum(axis=1)
            print("LL:", (result.max(axis=1)).sum()/ts.shape[0])
            LL.append((result.max(axis=1)).sum()/ts.shape[0])

        print("Mean="+str(np.mean(LL))+", Std="+str(np.std(LL))+", Best="+str(np.max(LL)))
