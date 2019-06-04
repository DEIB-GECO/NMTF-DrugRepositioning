#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:36:33 2019

@author: gaetandissez

Comment: this file aims at creating the new matrices R12 including -or not- different data.
The models are written in method_NMTF_DatasetContribution.
"""


import os
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["VECLIB_MAXIMUM_THREADS"] = "5"

#The different class are loaded.
from method_NMTF_DatasetContribution import NMTF1, NMTF2, NMTF3, NMTF4, NMTF5
import numpy as np

#Create all models the load the mask matrix
M10 = np.load('./tmp/M10.npy')
max_iter = 50

K = [500, 141, 500, 500, 300]

nmtf1 = NMTF1(K, M10)
nmtf2 = NMTF2(K, M10)
nmtf3 = NMTF3(K, M10)
nmtf4 = NMTF4(K, M10)
nmtf5 = NMTF5(K, M10)

models = [nmtf1, nmtf2, nmtf3, nmtf4, nmtf5]


for i in range(len(models)):
    #for every model, initialize it
    models[i].initialize()
    print(models[i])
    
    loss_old = models[i].loss()
    
    while models[i].iter < max_iter: 
        #and iterate the model until the stop criterion stops the algorithm
        models[i].iterate()
        
        loss_new = models[i].loss()
        CRIT = abs((loss_new - loss_old) / loss_new) 
        if CRIT < 2e-2:
            break
       
        loss_old = loss_new
    #For each model, the matrix R12 is saved after the algorithm stops
    R12_found = np.linalg.multi_dot([models[i].G1, models[i].S12, models[i].G2.transpose()])
    np.save('./tmp/R12_found_model' + str(i+1) + '_dataset', R12_found)

    print(models[i].validate())