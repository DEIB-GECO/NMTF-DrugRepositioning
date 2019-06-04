#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:36:31 2019

@author: gaetandissez

Important note:
    We initialize factor matrices once and for all so that each new model uses the same ones as the previous ones.
    It makes the results more stable because they depend on the initialization.
"""

import numpy as np
import sklearn.metrics as metrics
from spherecluster import SphericalKMeans
from scipy import sparse


class NMTF1:
    #First load and convert to numpy arrays the data
    R12 = sparse.load_npz('./tmp/R12.npz').toarray()
    eps = 1e-8
    n1, n2 = R12.shape

    def update(self, A, num, den):
        return A*(num / (den + NMTF1.eps))**0.5
    
    vupdate = np.vectorize(update)
    
    
    def __init__(self, parameters, mask):
        self.K = parameters
        self.M = mask
        self.iter = 0
    
    def initialize(self):
    
        self.R12_train = np.multiply(NMTF1.R12, self.M)
        
        """spherical k-means"""
        skm1 = SphericalKMeans(n_clusters=self.K[0])
        skm1.fit(self.R12_train.transpose())
        skm2 = SphericalKMeans(n_clusters=self.K[1])
        skm2.fit(self.R12_train)
        
        self.G1 = skm1.cluster_centers_.transpose()
        self.G2 = skm2.cluster_centers_.transpose()
       
        self.S12 = np.linalg.multi_dot([self.G1.transpose(), self.R12_train, self.G2])
        
        #Save the factor matrices for the mext models
        NMTF1.G1 = self.G1
        NMTF1.G2 = self.G2
       
    def iterate(self):
        Gt2G2 = np.dot(self.G2.transpose(), self.G2)
        G2Gt2 = np.dot(self.G2, self.G2.transpose())
        R12G2 = np.dot(self.R12_train, self.G2)
       
        R12G2St12 = np.dot(R12G2, self.S12.transpose())
        G1G1tR12G2St12 = np.linalg.multi_dot([self.G1, self.G1.transpose(), R12G2St12])
        Rt12G1S12 = np.linalg.multi_dot([self.R12_train.transpose(), self.G1, self.S12])
        G2Gt2Rt12G1S12 = np.dot(G2Gt2, Rt12G1S12)
        
        Gt1R12G2 = np.dot(self.G1.transpose(),R12G2)
        Gt1G1S12Gt2G2 = np.linalg.multi_dot([self.G1.transpose(), self.G1, self.S12, Gt2G2])
       
        self.G1 = NMTF1.vupdate(self, self.G1, R12G2St12, G1G1tR12G2St12)
        self.G2 = NMTF1.vupdate(self, self.G2, Rt12G1S12, G2Gt2Rt12G1S12)
        self.S12 = NMTF1.vupdate(self, self.S12, Gt1R12G2, Gt1G1S12Gt2G2) 
        self.iter += 1
        
    def validate(self, metric='aps'):
        n, m = NMTF1.R12.shape
        R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])
        R12_2 = []
        R12_found_2 = []
    
        for i in range(n):
            for j in range(m):
                if self.M[i, j] ==  0:
                    R12_2.append(NMTF1.R12[i, j])
                    R12_found_2.append(R12_found[i, j])
        if metric == 'auroc':
            fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
            return metrics.auc(fpr, tpr)
        if metric == 'aps':
            return metrics.average_precision_score(R12_2, R12_found_2)
        
    def loss(self):
        J = np.linalg.norm(self.R12_train - np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()]), ord='fro')**2
        return J
    
    def __repr__(self):
        return 'Model NMTF with (k1, k2) = ({}, {})'.format(self.K[0], self.K[1])


class NMTF2:
    #First load and convert to numpy arrays the data
    R12 = sparse.load_npz('./tmp/R12.npz').toarray()
    R23 = sparse.load_npz('./tmp/R23.npz').toarray()
    
    eps = 1e-8
    n1, n2 = R12.shape
    _, n3 = R23.shape
    
    
    def update(self, A, num, den):
        return A*(num / (den + NMTF2.eps))**0.5
    
    vupdate = np.vectorize(update)
    
    
    def __init__(self, parameters, mask):
        self.K = parameters
        self.M = mask
        self.iter = 0
    
    def initialize(self):
        
        self.R12_train = np.multiply(NMTF2.R12, self.M)
    
        """spherical k-means"""
        skm3 = SphericalKMeans(n_clusters=self.K[2])
        skm3.fit(NMTF2.R23)
        
        #Reload matrices that have already been used before
        self.G1 = NMTF1.G1
        self.G2 = NMTF1.G2
        self.G3 = skm3.cluster_centers_.transpose()
       
        self.S12 = np.linalg.multi_dot([self.G1.transpose(), self.R12_train, self.G2])
        self.S23 = np.linalg.multi_dot([self.G2.transpose(), NMTF2.R23, self.G3])
        
        #Save G3 for the next models
        NMTF2.G3 = self.G3
        
    def iterate(self):
        Gt2G2 = np.dot(self.G2.transpose(), self.G2)
        G2Gt2 = np.dot(self.G2, self.G2.transpose())
        G3Gt3 = np.dot(self.G3, self.G3.transpose())
        Gt3G3 = np.dot(self.G3.transpose(), self.G3)
        
        R12G2 = np.dot(self.R12_train, self.G2)
        R23G3 = np.dot(NMTF2.R23, self.G3)
        
        R12G2St12 = np.dot(R12G2, self.S12.transpose())
        G1G1tR12G2St12 = np.linalg.multi_dot([self.G1, self.G1.transpose(), R12G2St12])
        Rt12G1S12 = np.linalg.multi_dot([self.R12_train.transpose(), self.G1, self.S12])
        G2Gt2Rt12G1S12 = np.dot(G2Gt2, Rt12G1S12)
        R23G3St23 = np.dot(R23G3, self.S23.transpose())
        G2Gt2R23G3St23 = np.dot(G2Gt2, R23G3St23)
        Rt23G2S23 = np.linalg.multi_dot([NMTF2.R23.transpose(),self.G2, self.S23])
        G3Gt3Rt23G2S23 = np.dot(G3Gt3,Rt23G2S23)
        
        Gt1R12G2 = np.dot(self.G1.transpose(),R12G2)
        Gt2R23G3 = np.dot(self.G2.transpose(),R23G3)
       
        Gt1G1S12Gt2G2 = np.linalg.multi_dot([self.G1.transpose(), self.G1, self.S12, Gt2G2])
        Gt2G2S23Gt3G3 = np.linalg.multi_dot([Gt2G2, self.S23, Gt3G3])

        self.G1 = NMTF2.vupdate(self, self.G1, R12G2St12, G1G1tR12G2St12)
        self.G2 = NMTF2.vupdate(self, self.G2, Rt12G1S12 + R23G3St23, G2Gt2Rt12G1S12 + G2Gt2R23G3St23)
        self.G3 = NMTF2.vupdate(self, self.G3, Rt23G2S23, G3Gt3Rt23G2S23)
      
        self.S12 = NMTF2.vupdate(self, self.S12, Gt1R12G2, Gt1G1S12Gt2G2) 
        self.S23 = NMTF2.vupdate(self, self.S23, Gt2R23G3, Gt2G2S23Gt3G3)

        self.iter += 1
        
    def validate(self, metric='aps'):
        n, m = NMTF2.R12.shape
        R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])
        R12_2 = []
        R12_found_2 = []
    
        for i in range(n):
            for j in range(m):
                if self.M[i, j] ==  0:
                    R12_2.append(NMTF2.R12[i, j])
                    R12_found_2.append(R12_found[i, j])
        if metric == 'auroc':
            fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
            return metrics.auc(fpr, tpr)
        if metric == 'aps':
            return metrics.average_precision_score(R12_2, R12_found_2)
        
    def loss(self):
        J = np.linalg.norm(self.R12_train - np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF2.R23 - np.linalg.multi_dot([self.G2, self.S23, self.G3.transpose()]), ord='fro')**2
        return J
    
    def __repr__(self):
        return 'Model NMTF with (k1, k2, k3) = ({}, {}, {})'.format(self.K[0], self.K[1], self.K[2])


class NMTF3:
    #First load and convert to numpy arrays the data
    R12 = sparse.load_npz('./tmp/R12.npz').toarray()
    R23 = sparse.load_npz('./tmp/R23.npz').toarray()
    R34 = sparse.load_npz('./tmp/R34.npz').toarray()
   
    eps = 1e-8

    n1, n2 = R12.shape
    n3, n4 = R34.shape
    
    def update(self, A, num, den):
        return A*(num / (den + NMTF3.eps))**0.5

    vupdate = np.vectorize(update)
    
    def __init__(self, parameters, mask):
        self.K = parameters
        self.M = mask
        self.iter = 0
    
    def initialize(self):
        
        self.R12_train = np.multiply(NMTF3.R12, self.M)
        
        """spherical k-means"""
        skm4 = SphericalKMeans(n_clusters=self.K[3])
        skm4.fit(NMTF3.R34)

        self.G4 = skm4.cluster_centers_.transpose()
        
        #Use the same matrices as those precedently computed
        self.G1 = NMTF1.G1
        self.G2 = NMTF1.G2
        self.G3 = NMTF2.G3
        
       
        self.S12 = np.linalg.multi_dot([self.G1.transpose(), self.R12_train, self.G2])
        self.S23 = np.linalg.multi_dot([self.G2.transpose(), NMTF3.R23, self.G3])
        self.S34 = np.linalg.multi_dot([self.G3.transpose(), NMTF3.R34, self.G4])
        
        #Save G4 for next models
        NMTF3.G4 = self.G4
        
    def iterate(self):
        Gt2G2 = np.dot(self.G2.transpose(), self.G2)
        G2Gt2 = np.dot(self.G2, self.G2.transpose())
        G3Gt3 = np.dot(self.G3, self.G3.transpose())
        Gt3G3 = np.dot(self.G3.transpose(), self.G3)
        G4Gt4 = np.dot(self.G4, self.G4.transpose())
        
        R12G2 = np.dot(self.R12_train, self.G2)
        R23G3 = np.dot(NMTF3.R23, self.G3)
        R34G4 = np.dot(NMTF3.R34, self.G4)

        R12G2St12 = np.dot(R12G2, self.S12.transpose())
        G1G1tR12G2St12 = np.linalg.multi_dot([self.G1, self.G1.transpose(), R12G2St12])
        Rt12G1S12 = np.linalg.multi_dot([self.R12_train.transpose(), self.G1, self.S12])
        G2Gt2Rt12G1S12 = np.dot(G2Gt2, Rt12G1S12)
        R23G3St23 = np.dot(R23G3, self.S23.transpose())
        G2Gt2R23G3St23 = np.dot(G2Gt2, R23G3St23)
        Rt23G2S23 = np.linalg.multi_dot([NMTF3.R23.transpose(),self.G2, self.S23])
        G3Gt3Rt23G2S23 = np.dot(G3Gt3,Rt23G2S23)
        R34G4St34 = np.dot(R34G4, self.S34.transpose())
        G3Gt3R34G4St34 = np.dot(G3Gt3,R34G4St34)
        Rt34G3S34 = np.linalg.multi_dot([NMTF3.R34.transpose(),self.G3, self.S34])
        G4Gt4Rt34G3S34 = np.dot(G4Gt4,Rt34G3S34)
        
        Gt1R12G2 = np.dot(self.G1.transpose(),R12G2)
        Gt2R23G3 = np.dot(self.G2.transpose(),R23G3)
        Gt3R34G4 = np.dot(self.G3.transpose(),R34G4)
        Gt1G1S12Gt2G2 = np.linalg.multi_dot([self.G1.transpose(), self.G1, self.S12, Gt2G2])
        Gt2G2S23Gt3G3 = np.linalg.multi_dot([Gt2G2, self.S23, Gt3G3])
        Gt3G3S34Gt4G4 = np.linalg.multi_dot([Gt3G3, self.S34, self.G4.transpose(), self.G4])
       
        
        self.G1 = NMTF3.vupdate(self, self.G1, R12G2St12, G1G1tR12G2St12)
        self.G2 = NMTF3.vupdate(self, self.G2, Rt12G1S12 + R23G3St23, G2Gt2Rt12G1S12 + G2Gt2R23G3St23)
        self.G3 = NMTF3.vupdate(self, self.G3, Rt23G2S23 + R34G4St34, G3Gt3Rt23G2S23 + G3Gt3R34G4St34)
        self.G4 = NMTF3.vupdate(self, self.G4, Rt34G3S34, G4Gt4Rt34G3S34)
        
        self.S12 = NMTF3.vupdate(self, self.S12, Gt1R12G2, Gt1G1S12Gt2G2) 
        self.S23 = NMTF3.vupdate(self, self.S23, Gt2R23G3, Gt2G2S23Gt3G3)
        self.S34 = NMTF3.vupdate(self, self.S34, Gt3R34G4, Gt3G3S34Gt4G4)
        
        self.iter += 1
        
    def validate(self, metric='aps'):
        n, m = NMTF3.R12.shape
        R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])
        R12_2 = []
        R12_found_2 = []
    
        for i in range(n):
            for j in range(m):
                if self.M[i, j] ==  0:
                    R12_2.append(NMTF3.R12[i, j])
                    R12_found_2.append(R12_found[i, j])
        if metric == 'auroc':
            fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
            return metrics.auc(fpr, tpr)
        if metric == 'aps':
            return metrics.average_precision_score(R12_2, R12_found_2)
        
    def loss(self):
        J = np.linalg.norm(self.R12_train - np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF3.R23 - np.linalg.multi_dot([self.G2, self.S23, self.G3.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF3.R34 - np.linalg.multi_dot([self.G3, self.S34, self.G4.transpose()]), ord='fro')**2
        return J
    
    def __repr__(self):
        return 'Model NMTF with (k1, k2, k3, k4) = ({}, {}, {}, {})'.format(self.K[0], self.K[1], self.K[2], self.K[3])


class NMTF4:
    #First load and convert to numpy arrays the data
    R12 = sparse.load_npz('./tmp/R12.npz').toarray()
    R23 = sparse.load_npz('./tmp/R23.npz').toarray()
    R34 = sparse.load_npz('./tmp/R34.npz').toarray()
   
    W3 = sparse.load_npz('./tmp/W3.npz').toarray()
    W4 = sparse.load_npz('./tmp/W4.npz').toarray()
    L3 = sparse.load_npz('./tmp/L3.npz').toarray()
    L4 = sparse.load_npz('./tmp/L4.npz').toarray()
    D3 = L3 + W3
    D4 = L4 + W4
    
    eps = 1e-8

    n1, n2 = R12.shape
    n3, n4 = R34.shape
   
    
    def update(self, A, num, den):
        return A*(num / (den + NMTF4.eps))**0.5
    
    vupdate = np.vectorize(update)
    
    
    def __init__(self, parameters, mask):
        self.K = parameters
        self.M = mask
        self.iter = 0
    
    def initialize(self):
        
        self.R12_train = np.multiply(NMTF4.R12, self.M)
        """spherical k-means"""

        #Only use the initial factors of the former model
        self.G1 = NMTF1.G1
        self.G2 = NMTF1.G2
        self.G3 = NMTF2.G3
        self.G4 = NMTF3.G4

        self.S12 = np.linalg.multi_dot([self.G1.transpose(), self.R12_train, self.G2])
        self.S23 = np.linalg.multi_dot([self.G2.transpose(), NMTF4.R23, self.G3])
        self.S34 = np.linalg.multi_dot([self.G3.transpose(), NMTF4.R34, self.G4])
        
    def iterate(self):
        Gt2G2 = np.dot(self.G2.transpose(), self.G2)
        G2Gt2 = np.dot(self.G2, self.G2.transpose())
        G3Gt3 = np.dot(self.G3, self.G3.transpose())
        Gt3G3 = np.dot(self.G3.transpose(), self.G3)
        G4Gt4 = np.dot(self.G4, self.G4.transpose())
        
        R12G2 = np.dot(self.R12_train, self.G2)
        R23G3 = np.dot(NMTF4.R23, self.G3)
        R34G4 = np.dot(NMTF4.R34, self.G4)
 
        W3G3 = np.dot(NMTF4.W3, self.G3)
        W4G4 = np.dot(NMTF4.W4, self.G4)
        D3G3 = np.dot(NMTF4.D3, self.G3)
        D4G4 = np.dot(NMTF4.D4, self.G4)
        G3Gt3D3G3 = np.dot(G3Gt3, D3G3)
        G4Gt4D4G4 = np.dot(G4Gt4, D4G4)
        G3Gt3W3G3 = np.dot(G3Gt3, W3G3)
        G4Gt4W4G4 = np.dot(G4Gt4, W4G4)
        
        R12G2St12 = np.dot(R12G2, self.S12.transpose())
        G1G1tR12G2St12 = np.linalg.multi_dot([self.G1, self.G1.transpose(), R12G2St12])
        Rt12G1S12 = np.linalg.multi_dot([self.R12_train.transpose(), self.G1, self.S12])
        G2Gt2Rt12G1S12 = np.dot(G2Gt2, Rt12G1S12)
        R23G3St23 = np.dot(R23G3, self.S23.transpose())
        G2Gt2R23G3St23 = np.dot(G2Gt2, R23G3St23)
        Rt23G2S23 = np.linalg.multi_dot([NMTF4.R23.transpose(),self.G2, self.S23])
        G3Gt3Rt23G2S23 = np.dot(G3Gt3,Rt23G2S23)
        R34G4St34 = np.dot(R34G4, self.S34.transpose())
        G3Gt3R34G4St34 = np.dot(G3Gt3,R34G4St34)
        Rt34G3S34 = np.linalg.multi_dot([NMTF4.R34.transpose(),self.G3, self.S34])
        G4Gt4Rt34G3S34 = np.dot(G4Gt4,Rt34G3S34)
        
        Gt1R12G2 = np.dot(self.G1.transpose(),R12G2)
        Gt2R23G3 = np.dot(self.G2.transpose(),R23G3)
        Gt3R34G4 = np.dot(self.G3.transpose(),R34G4)
        Gt1G1S12Gt2G2 = np.linalg.multi_dot([self.G1.transpose(), self.G1, self.S12, Gt2G2])
        Gt2G2S23Gt3G3 = np.linalg.multi_dot([Gt2G2, self.S23, Gt3G3])
        Gt3G3S34Gt4G4 = np.linalg.multi_dot([Gt3G3, self.S34, self.G4.transpose(), self.G4])        
        
        self.G1 = NMTF4.vupdate(self, self.G1, R12G2St12, G1G1tR12G2St12)
        self.G2 = NMTF4.vupdate(self, self.G2, Rt12G1S12 + R23G3St23, G2Gt2Rt12G1S12 + G2Gt2R23G3St23)
        self.G3 = NMTF4.vupdate(self, self.G3, Rt23G2S23 + R34G4St34 + W3G3 + G3Gt3D3G3, G3Gt3Rt23G2S23 + G3Gt3R34G4St34 + G3Gt3W3G3 + D3G3)
        self.G4 = NMTF4.vupdate(self, self.G4, Rt34G3S34 + W4G4 + G4Gt4D4G4, G4Gt4Rt34G3S34 + G4Gt4W4G4 + D4G4)
        
        self.S12 = NMTF4.vupdate(self, self.S12, Gt1R12G2, Gt1G1S12Gt2G2) 
        self.S23 = NMTF4.vupdate(self, self.S23, Gt2R23G3, Gt2G2S23Gt3G3)
        self.S34 = NMTF4.vupdate(self, self.S34, Gt3R34G4, Gt3G3S34Gt4G4)
        
        self.iter += 1
        
    def validate(self, metric='aps'):
        n, m = NMTF4.R12.shape
        R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])
        R12_2 = []
        R12_found_2 = []
    
        for i in range(n):
            for j in range(m):
                if self.M[i, j] ==  0:
                    R12_2.append(NMTF4.R12[i, j])
                    R12_found_2.append(R12_found[i, j])
        if metric == 'auroc':
            fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
            return metrics.auc(fpr, tpr)
        if metric == 'aps':
            return metrics.average_precision_score(R12_2, R12_found_2)
        
    def loss(self):
        Gt3L3G3 = np.linalg.multi_dot([self.G3.transpose(), NMTF4.L3, self.G3])
        Gt4L4G4 = np.linalg.multi_dot([self.G4.transpose(), NMTF4.L4, self.G4])
        
        J = np.linalg.norm(self.R12_train - np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF4.R23 - np.linalg.multi_dot([self.G2, self.S23, self.G3.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF4.R34 - np.linalg.multi_dot([self.G3, self.S34, self.G4.transpose()]), ord='fro')**2
        J += np.trace(Gt3L3G3) + np.trace(Gt4L4G4)
        return J
    
    def __repr__(self):
        return 'Model NMTF with (k1, k2, k3, k4) = ({}, {}, {}, {})'.format(self.K[0], self.K[1], self.K[2], self.K[3])



class NMTF5:
    #First load and convert to numpy arrays the data
    R12 = sparse.load_npz('./tmp/R12.npz').toarray()
    R23 = sparse.load_npz('./tmp/R23.npz').toarray()
    R34 = sparse.load_npz('./tmp/R34.npz').toarray()
    R25 = sparse.load_npz('./tmp/R25.npz').toarray()
    W3 = sparse.load_npz('./tmp/W3.npz').toarray()
    W4 = sparse.load_npz('./tmp/W4.npz').toarray()
    L3 = sparse.load_npz('./tmp/L3.npz').toarray()
    L4 = sparse.load_npz('./tmp/L4.npz').toarray()
    D3 = L3 + W3
    D4 = L4 + W4
    
    eps = 1e-8

    n1, n2 = R12.shape
    n3, n4 = R34.shape
    n5 = R25.shape[1] 
    
    
    def update(self, A, num, den):
        return A*(num / (den + NMTF5.eps))**0.5
    
    vupdate = np.vectorize(update)
    
    
    def __init__(self, parameters, mask):
        self.K = parameters
        self.M = mask
        self.iter = 0
    
    def initialize(self):
        
        self.R12_train = np.multiply(NMTF5.R12, self.M)
        
        
        """spherical k-means"""
        skm5 = SphericalKMeans(n_clusters=self.K[4])
        skm5.fit(NMTF5.R25)
        
        self.G1 = NMTF1.G1
        self.G2 = NMTF1.G2
        self.G3 = NMTF2.G3
        self.G4 = NMTF3.G4
        self.G5 = skm5.cluster_centers_.transpose()
            
       
        self.S12 = np.linalg.multi_dot([self.G1.transpose(), self.R12_train, self.G2])
        self.S23 = np.linalg.multi_dot([self.G2.transpose(), NMTF5.R23, self.G3])
        self.S34 = np.linalg.multi_dot([self.G3.transpose(), NMTF5.R34, self.G4])
        self.S25 = np.linalg.multi_dot([self.G2.transpose(), NMTF5.R25, self.G5])
        
    def iterate(self):
        Gt2G2 = np.dot(self.G2.transpose(), self.G2)
        G2Gt2 = np.dot(self.G2, self.G2.transpose())
        G3Gt3 = np.dot(self.G3, self.G3.transpose())
        Gt3G3 = np.dot(self.G3.transpose(), self.G3)
        G4Gt4 = np.dot(self.G4, self.G4.transpose())
        
        R12G2 = np.dot(self.R12_train, self.G2)
        R23G3 = np.dot(NMTF5.R23, self.G3)
        R34G4 = np.dot(NMTF5.R34, self.G4)
        R25G5 = np.dot(NMTF5.R25, self.G5)
        
        W3G3 = np.dot(NMTF5.W3, self.G3)
        W4G4 = np.dot(NMTF5.W4, self.G4)
        D3G3 = np.dot(NMTF5.D3, self.G3)
        D4G4 = np.dot(NMTF5.D4, self.G4)
        G3Gt3D3G3 = np.dot(G3Gt3, D3G3)
        G4Gt4D4G4 = np.dot(G4Gt4, D4G4)
        G3Gt3W3G3 = np.dot(G3Gt3, W3G3)
        G4Gt4W4G4 = np.dot(G4Gt4, W4G4)
        
        R12G2St12 = np.dot(R12G2, self.S12.transpose())
        G1G1tR12G2St12 = np.linalg.multi_dot([self.G1, self.G1.transpose(), R12G2St12])
        Rt12G1S12 = np.linalg.multi_dot([self.R12_train.transpose(), self.G1, self.S12])
        G2Gt2Rt12G1S12 = np.dot(G2Gt2, Rt12G1S12)
        R23G3St23 = np.dot(R23G3, self.S23.transpose())
        G2Gt2R23G3St23 = np.dot(G2Gt2, R23G3St23)
        Rt23G2S23 = np.linalg.multi_dot([NMTF5.R23.transpose(),self.G2, self.S23])
        G3Gt3Rt23G2S23 = np.dot(G3Gt3,Rt23G2S23)
        R34G4St34 = np.dot(R34G4, self.S34.transpose())
        G3Gt3R34G4St34 = np.dot(G3Gt3,R34G4St34)
        Rt34G3S34 = np.linalg.multi_dot([NMTF5.R34.transpose(),self.G3, self.S34])
        G4Gt4Rt34G3S34 = np.dot(G4Gt4,Rt34G3S34)
        Rt25G2S25 = np.linalg.multi_dot([NMTF5.R25.transpose(), self.G2, self.S25])
        G5G5tRt25G2S25 = np.linalg.multi_dot([self.G5, self.G5.transpose(), Rt25G2S25])
        R25G5St25 = np.dot(R25G5, self.S25.transpose())
        G2Gt2R25G5St25 = np.dot(G2Gt2, R25G5St25)
        
        Gt1R12G2 = np.dot(self.G1.transpose(),R12G2)
        Gt2R23G3 = np.dot(self.G2.transpose(),R23G3)
        Gt3R34G4 = np.dot(self.G3.transpose(),R34G4)
        Gt2R25G5 = np.dot(self.G2.transpose(), R25G5)
        Gt1G1S12Gt2G2 = np.linalg.multi_dot([self.G1.transpose(), self.G1, self.S12, Gt2G2])
        Gt2G2S23Gt3G3 = np.linalg.multi_dot([Gt2G2, self.S23, Gt3G3])
        Gt3G3S34Gt4G4 = np.linalg.multi_dot([Gt3G3, self.S34, self.G4.transpose(), self.G4])
        Gt2G2S25Gt5G5 = np.linalg.multi_dot([Gt2G2, self.S25, self.G5.transpose(), self.G5])
        
        
        self.G1 = NMTF5.vupdate(self, self.G1, R12G2St12, G1G1tR12G2St12)
        self.G2 = NMTF5.vupdate(self, self.G2, Rt12G1S12 + R23G3St23 + R25G5St25, G2Gt2Rt12G1S12 + G2Gt2R23G3St23 + G2Gt2R25G5St25)
        self.G3 = NMTF5.vupdate(self, self.G3, Rt23G2S23 + R34G4St34 + W3G3 + G3Gt3D3G3, G3Gt3Rt23G2S23 + G3Gt3R34G4St34 + G3Gt3W3G3 + D3G3)
        self.G4 = NMTF5.vupdate(self, self.G4, Rt34G3S34 + W4G4 + G4Gt4D4G4, G4Gt4Rt34G3S34 + G4Gt4W4G4 + D4G4)
        self.G5 = NMTF5.vupdate(self, self.G5, Rt25G2S25, G5G5tRt25G2S25)
        
        self.S12 = NMTF5.vupdate(self, self.S12, Gt1R12G2, Gt1G1S12Gt2G2) 
        self.S23 = NMTF5.vupdate(self, self.S23, Gt2R23G3, Gt2G2S23Gt3G3)
        self.S34 = NMTF5.vupdate(self, self.S34, Gt3R34G4, Gt3G3S34Gt4G4)
        self.S25 = NMTF5.vupdate(self, self.S25, Gt2R25G5, Gt2G2S25Gt5G5)
        
        self.iter += 1
        
    def validate(self, metric='aps'):
        n, m = NMTF5.R12.shape
        R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])
        R12_2 = []
        R12_found_2 = []
    
        for i in range(n):
            for j in range(m):
                if self.M[i, j] ==  0:
                    R12_2.append(NMTF5.R12[i, j])
                    R12_found_2.append(R12_found[i, j])
        if metric == 'auroc':
            fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
            return metrics.auc(fpr, tpr)
        if metric == 'aps':
            return metrics.average_precision_score(R12_2, R12_found_2)
        
    def loss(self):
        
        Gt3L3G3 = np.linalg.multi_dot([self.G3.transpose(), NMTF5.L3, self.G3])
        Gt4L4G4 = np.linalg.multi_dot([self.G4.transpose(), NMTF5.L4, self.G4])
        
        J = np.linalg.norm(self.R12_train - np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF5.R23 - np.linalg.multi_dot([self.G2, self.S23, self.G3.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF5.R34 - np.linalg.multi_dot([self.G3, self.S34, self.G4.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF5.R25 - np.linalg.multi_dot([self.G2, self.S25, self.G5.transpose()]), ord='fro')**2
        J += np.trace(Gt3L3G3) + np.trace(Gt4L4G4)
        
        return J
    
    def __repr__(self):
        return 'Model NMTF with (k1, k2, k3, k4, k5) = ({}, {}, {}, {}, {})'.format(self.K[0], self.K[1], self.K[2], self.K[3], self.K[4])
