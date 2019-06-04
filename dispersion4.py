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

M10 = np.load('./tmp/M10.npy')

from method_NMTF import NMTF

K = [500, 141, 500, 500, 300]
max_iter = 10
nb_init = 5

def connectivity(H):
    n, m = H.shape
    P = np.zeros((n,m))
    for i in range(n):
        j = np.argmax(H[i])
        P[i, j] = 1
    return(np.dot(P, P.transpose()))


RHO4 = []

for j in range(10, 1000, 10):
    K[3] = j
    nmtf = NMTF('skmeans', K, M10)
    connect = []
    for i in range(nb_init):
        nmtf.initialize()
        for p in range(max_iter):
            nmtf.iterate()
        
        R34_found = np.linalg.multi_dot([nmtf.G3, nmtf.S34, nmtf.G4.transpose()])

        connect.append(connectivity(R34_found))
    CONSENSUS = np.mean(connect, axis=0)
    
    n = len(CONSENSUS)
    
    rho = 0
    for i in range(n):
        for j in range(n):
            rho += (CONSENSUS[i, j] - 0.5)**2
    rho = rho*4/(n**2)
       
    RHO4.append(rho)

    np.save('./tmp/rho4', RHO4)