#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:36:57 2019

@author: gaetandissez
"""

import os
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["VECLIB_MAXIMUM_THREADS"] = "5"

import numpy as np
from spherecluster import SphericalKMeans
from method_NMTF import NMTF

M10 = np.load('./tmp/M10.npy')

K = [500, 141, 500, 500, 300]
max_iter = 10
nb_init = 3

#We will initialize G1, G2, G3, G5 once and for all to have a more stable result and to focus only on the effects of k4.
R12_train = np.multiply(NMTF.R12, M10)
skm1 = SphericalKMeans(n_clusters=K[0])
skm1.fit(R12_train.transpose())
skm2 = SphericalKMeans(n_clusters=K[1])
skm2.fit(R12_train)
skm3 = SphericalKMeans(n_clusters=K[2])
skm3.fit(NMTF.R23)
skm5 = SphericalKMeans(n_clusters=K[4])
skm5.fit(NMTF.R25)
            
def connectivity(H):
    n, m = H.shape
    P = np.zeros((n,m))
    for i in range(n):
        j = np.argmax(H[i])
        P[i, j] = 1
    return(np.dot(P, P.transpose()))


Rho_4  = []

for j in tqdm_notebook(range(10, 1000, 20)):
    print(j)
    K[3] = j
    nmtf = NMTF('skmeans', K, M10)
    nmtf.R12_train = R12_train
    connect = []

    for i in range(nb_init):
        skm4 = SphericalKMeans(n_clusters=K[3])
        skm4.fit(NMTF.R34)

        nmtf.G1 = skm1.cluster_centers_.transpose()
        nmtf.G2 = skm2.cluster_centers_.transpose()
        nmtf.G3 = skm3.cluster_centers_.transpose()
        nmtf.G4 = skm4.cluster_centers_.transpose()
        nmtf.G5 = skm5.cluster_centers_.transpose()
        
        nmtf.S12 = np.linalg.multi_dot([nmtf.G1.transpose(), nmtf.R12_train, nmtf.G2])
        nmtf.S23 = np.linalg.multi_dot([nmtf.G2.transpose(), nmtf.R23, nmtf.G3])
        nmtf.S34 = np.linalg.multi_dot([nmtf.G3.transpose(), nmtf.R34, nmtf.G4])
        nmtf.S25 = np.linalg.multi_dot([nmtf.G2.transpose(), nmtf.R25, nmtf.G5])
        
        for p in range(max_iter):
            nmtf.iterate()
                
        R34_found = np.linalg.multi_dot([nmtf.G3, nmtf.S34, nmtf.G4.transpose()])

        connect.append(connectivity(R34_found))
    CONSENSUS = np.mean(connect, axis=0)
    
    n1 = len(CONSENSUS)
    rho = 0

    for i in range(n1):
        for j in range(n1):
            rho += (CONSENSUS[i, j] - 0.5)**2
    rho = rho*4/(n1**2)
    
    Rho_4.append(rho)
    np.save('./tmp/rho4', Rho_4)       