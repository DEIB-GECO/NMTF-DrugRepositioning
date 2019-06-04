#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:36:20 2019

@author: gaetandissez
"""

import os
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["VECLIB_MAXIMUM_THREADS"] = "5"

from method_NMTF import NMTF
import numpy as np


"""
MODEL 1: Random init, bad parameters, max_iter
MODEL 2: change init to skmeans
MODEL 3: but good parameters
MODEL 4: perfect
"""

#Create once and for all models the mask matrix and the associated R12_r which will be approximated
M10 = np.load('./tmp/M10.npy')
max_iter = 500

K_bad = [1000, 50, 200, 250, 30]
K_good = [500, 141, 500, 500, 300]



nmtf1 = NMTF('random', K_bad, M10)
nmtf2 = NMTF('skmeans', K_bad, M10)
nmtf34 = NMTF('skmeans', K_good, M10)

nmtf1.initialize()
nmtf2.initialize()
nmtf34.initialize()

#model 1
print(nmtf1)
while nmtf1.iter < max_iter:
    nmtf1.iterate()
R12_found_1 = np.linalg.multi_dot([nmtf1.G1, nmtf1.S12, nmtf1.G2.transpose()])
np.save('./tmp/R12_found_1', R12_found_1)
print(nmtf1.validate())

#model 2
print(nmtf2)
while nmtf2.iter < max_iter:
    nmtf2.iterate()
R12_found_2 = np.linalg.multi_dot([nmtf2.G1, nmtf2.S12, nmtf2.G2.transpose()])
np.save('./tmp/R12_found_2', R12_found_2)
print(nmtf2.validate())

#model 3 & 4
print(nmtf34)
not_done = True
loss_old = nmtf34.loss()
while nmtf34.iter < max_iter:
    nmtf34.iterate() 
    if not_done:
        loss_new = nmtf34.loss()
        CRIT = abs((loss_new - loss_old) / loss_new)  
        if CRIT < 2e-2:
            not_done = False
            R12_found_4 = np.linalg.multi_dot([nmtf34.G1, nmtf34.S12, nmtf34.G2.transpose()])
            np.save('./tmp/R12_found_4', R12_found_4)
            print(nmtf34.validate())
        loss_old = loss_new
        
print(nmtf34.validate())
R12_found_3 = np.linalg.multi_dot([nmtf34.G1, nmtf34.S12, nmtf34.G2.transpose()])
np.save('./tmp/R12_found_3', R12_found_3)

