#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:45:44 2019

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

M10 = np.load('./tmp/M10.npy')

K = [100, 50, 200, 100, 600]
max_iter = 500
nb_init = 10

INIT = ['random', 'acol', 'kmeans', 'skmeans']


for init in INIT:
    nmtf = NMTF(init, K, M10)
    loss, aps = np.zeros((nb_init, max_iter//10)), np.zeros((nb_init, max_iter//10))
    for i in range(nb_init):
        nmtf.initialize()
        for p in range(max_iter):
            nmtf.iterate()
            if p % 10 == 0:
                loss[i, p//10], aps[i, p//10] = nmtf.loss(), nmtf.validate()
        result = [loss, aps]
        np.save('./tmp/initialization_' + init, result)