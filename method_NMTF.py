#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:37:37 2019

@author: gaetandissez
"""

import numpy as np
import sklearn.metrics as metrics
from spherecluster import SphericalKMeans
from sklearn.cluster import KMeans
from scipy import sparse


class NMTF:
    #First load and convert to numpy arrays the data
    R12 = sparse.load_npz('./tmp/R12.npz').toarray()
    R23 = sparse.load_npz('./tmp/R23.npz').toarray()
    R34 = sparse.load_npz('./tmp/R34.npz').toarray()
    R25 = sparse.load_npz('./tmp/R25.npz').toarray()
    W3 = sparse.load_npz('./tmp/W3.npz').toarray()
    W4 = sparse.load_npz('./tmp/W4.npz').toarray()
    L3 = sparse.load_npz('./tmp/L3.npz').toarray()
    L4 = sparse.load_npz('./tmp/L4.npz').toarray()
    
    #Those matrices are called Degree matrices
    D3 = L3 + W3
    D4 = L4 + W4
    
    #eps is a constant needed experimentally in update rules to make sure that the denominator is never null
    eps = 1e-8
    
    n1, n2 = R12.shape
    n3, n4 = R34.shape
    n5 = R25.shape[1] 
    
    
    def update(self, A, num, den):
        return A*(num / (den + NMTF.eps))**0.5
    
    vupdate = np.vectorize(update)
    
    
    def __init__(self, init_method, parameters, mask):
        self.init_method = init_method
        self.K = parameters
        self.M = mask
        self.iter = 0
    
    def initialize(self):
        
        self.R12_train = np.multiply(NMTF.R12, self.M)
        
        if self.init_method == 'random':
            """Random uniform"""
            self.G1 = np.random.rand(NMTF.n1, self.K[0])
            self.G2 = np.random.rand(NMTF.n2, self.K[1])
            self.G3 = np.random.rand(NMTF.n3, self.K[2])
            self.G4 = np.random.rand(NMTF.n4, self.K[3])
            self.G5 = np.random.rand(NMTF.n5, self.K[4])
        
        if self.init_method == 'skmeans':
            """spherical k-means"""
            
            #Sperical k-means clustering is done on the initial data
            skm1 = SphericalKMeans(n_clusters=self.K[0])
            skm1.fit(self.R12_train.transpose())
            skm2 = SphericalKMeans(n_clusters=self.K[1])
            skm2.fit(self.R12_train)
            skm3 = SphericalKMeans(n_clusters=self.K[2])
            skm3.fit(NMTF.R23)
            skm4 = SphericalKMeans(n_clusters=self.K[3])
            skm4.fit(NMTF.R34)
            skm5 = SphericalKMeans(n_clusters=self.K[4])
            skm5.fit(NMTF.R25)
            
            #Factor matrices are initialized with the center coordinates
            self.G1 = skm1.cluster_centers_.transpose()
            self.G2 = skm2.cluster_centers_.transpose()
            self.G3 = skm3.cluster_centers_.transpose()
            self.G4 = skm4.cluster_centers_.transpose()
            self.G5 = skm5.cluster_centers_.transpose()
            
        if self.init_method == 'acol':
            """random ACOL"""
            #We will "shuffle" the columns of R matrices and take the mean of k batches
            Num1 = np.random.permutation(NMTF.n2)
            Num2 = np.random.permutation(NMTF.n1)
            Num3 = np.random.permutation(NMTF.n2)
            Num4 = np.random.permutation(NMTF.n3)
            Num5 = np.random.permutation(NMTF.n2)
            
            G1 = []
            for l in np.array_split(Num1, self.K[0]):
                G1.append(np.mean(self.R12_train[:,l], axis = 1))
            self.G1 = np.array(G1).transpose()
            
            G2 = []
            for l in np.array_split(Num2, self.K[1]):
                G2.append(np.mean(self.R12_train.transpose()[:,l], axis = 1))
            self.G2 = np.array(G2).transpose()
            
            G3 = []
            for l in np.array_split(Num3, self.K[2]):
                G3.append(np.mean(NMTF.R23.transpose()[:,l], axis = 1))
            self.G3 = np.array(G3).transpose()
            
            G4 = []
            for l in np.array_split(Num4, self.K[3]):
                G4.append(np.mean(NMTF.R34.transpose()[:,l], axis = 1))
            self.G4 = np.array(G4).transpose()
            
            G5 = []
            for l in np.array_split(Num5, self.K[4]):
                G5.append(np.mean(NMTF.R25.transpose()[:,l], axis = 1))
            self.G5 = np.array(G5).transpose()
        
        if self.init_method == 'kmeans':
            """k-means with clustering on previous item"""
            #As for spherical k-means, factor matrices will be initialized with the centers of clusters.
            km1 = KMeans(n_clusters=self.K[0], n_init = 10).fit_predict(self.R12_train.transpose())
            km2 = KMeans(n_clusters=self.K[1], n_init = 10).fit_predict(self.R12_train)
            km3 = KMeans(n_clusters=self.K[2], n_init = 10).fit_predict(self.R23)
            km4 = KMeans(n_clusters=self.K[3], n_init = 10).fit_predict(self.R34)
            km5 = KMeans(n_clusters=self.K[4], n_init = 10).fit_predict(self.R25)
            
            self.G1 = np.array([np.mean([self.R12_train[:,i] for i in range(len(km1)) if km1[i] == p], axis = 0) for p in range(self.K[0])]).transpose()
            self.G2 = np.array([np.mean([self.R12_train[i] for i in range(len(km2)) if km2[i] == p], axis = 0) for p in range(self.K[1])]).transpose()
            self.G3 = np.array([np.mean([self.R23[i] for i in range(len(km3)) if km3[i] == p], axis = 0) for p in range(self.K[2])]).transpose()
            self.G4 = np.array([np.mean([self.R34[i] for i in range(len(km4)) if km4[i] == p], axis = 0) for p in range(self.K[3])]).transpose()
            self.G5 = np.array([np.mean([self.R25[i] for i in range(len(km5)) if km5[i] == p], axis = 0) for p in range(self.K[4])]).transpose()
            
        self.S12 = np.linalg.multi_dot([self.G1.transpose(), self.R12_train, self.G2])
        self.S23 = np.linalg.multi_dot([self.G2.transpose(), self.R23, self.G3])
        self.S34 = np.linalg.multi_dot([self.G3.transpose(), self.R34, self.G4])
        self.S25 = np.linalg.multi_dot([self.G2.transpose(), self.R25, self.G5])
        
    def iterate(self):
        #These following lines compute the matrices needed for our update rules
        Gt2G2 = np.dot(self.G2.transpose(), self.G2)
        G2Gt2 = np.dot(self.G2, self.G2.transpose())
        G3Gt3 = np.dot(self.G3, self.G3.transpose())
        Gt3G3 = np.dot(self.G3.transpose(), self.G3)
        G4Gt4 = np.dot(self.G4, self.G4.transpose())
        
        R12G2 = np.dot(self.R12_train, self.G2)
        R23G3 = np.dot(NMTF.R23, self.G3)
        R34G4 = np.dot(NMTF.R34, self.G4)
        R25G5 = np.dot(NMTF.R25, self.G5)
        
        W3G3 = np.dot(NMTF.W3, self.G3)
        W4G4 = np.dot(NMTF.W4, self.G4)
        D3G3 = np.dot(NMTF.D3, self.G3)
        D4G4 = np.dot(NMTF.D4, self.G4)
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
        Rt23G2S23 = np.linalg.multi_dot([NMTF.R23.transpose(),self.G2, self.S23])
        G3Gt3Rt23G2S23 = np.dot(G3Gt3,Rt23G2S23)
        R34G4St34 = np.dot(R34G4, self.S34.transpose())
        G3Gt3R34G4St34 = np.dot(G3Gt3,R34G4St34)
        Rt34G3S34 = np.linalg.multi_dot([NMTF.R34.transpose(),self.G3, self.S34])
        G4Gt4Rt34G3S34 = np.dot(G4Gt4,Rt34G3S34)
        Rt25G2S25 = np.linalg.multi_dot([NMTF.R25.transpose(), self.G2, self.S25])
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
        
        #Here factor matrices are updated.
        self.G1 = NMTF.vupdate(self, self.G1, R12G2St12, G1G1tR12G2St12)
        self.G2 = NMTF.vupdate(self, self.G2, Rt12G1S12 + R23G3St23 + R25G5St25, G2Gt2Rt12G1S12 + G2Gt2R23G3St23 + G2Gt2R25G5St25)
        self.G3 = NMTF.vupdate(self, self.G3, Rt23G2S23 + R34G4St34 + W3G3 + G3Gt3D3G3, G3Gt3Rt23G2S23 + G3Gt3R34G4St34 + G3Gt3W3G3 + D3G3)
        self.G4 = NMTF.vupdate(self, self.G4, Rt34G3S34 + W4G4 + G4Gt4D4G4, G4Gt4Rt34G3S34 + G4Gt4W4G4 + D4G4)
        self.G5 = NMTF.vupdate(self, self.G5, Rt25G2S25, G5G5tRt25G2S25)
        
        self.S12 = NMTF.vupdate(self, self.S12, Gt1R12G2, Gt1G1S12Gt2G2) 
        self.S23 = NMTF.vupdate(self, self.S23, Gt2R23G3, Gt2G2S23Gt3G3)
        self.S34 = NMTF.vupdate(self, self.S34, Gt3R34G4, Gt3G3S34Gt4G4)
        self.S25 = NMTF.vupdate(self, self.S25, Gt2R25G5, Gt2G2S25Gt5G5)
        
        self.iter += 1
        
    def validate(self, metric='aps'):
        n, m = NMTF.R12.shape
        R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])
        R12_2 = []
        R12_found_2 = []
        
        #We first isolate the validation set and the corresponding result
        for i in range(n):
            for j in range(m):
                if self.M[i, j] ==  0:
                    R12_2.append(NMTF.R12[i, j])
                    R12_found_2.append(R12_found[i, j])
        #We can asses the quality of our output with APS or AUROC score
        if metric == 'auroc':
            fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
            return metrics.auc(fpr, tpr)
        if metric == 'aps':
            return metrics.average_precision_score(R12_2, R12_found_2)
        
    def loss(self):
        
        Gt3L3G3 = np.linalg.multi_dot([self.G3.transpose(), NMTF.L3, self.G3])
        Gt4L4G4 = np.linalg.multi_dot([self.G4.transpose(), NMTF.L4, self.G4])
        
        J = np.linalg.norm(self.R12_train - np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF.R23 - np.linalg.multi_dot([self.G2, self.S23, self.G3.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF.R34 - np.linalg.multi_dot([self.G3, self.S34, self.G4.transpose()]), ord='fro')**2
        J += np.linalg.norm(NMTF.R25 - np.linalg.multi_dot([self.G2, self.S25, self.G5.transpose()]), ord='fro')**2
        J += np.trace(Gt3L3G3) + np.trace(Gt4L4G4)
        
        return J
    
    def __repr__(self):
        return 'Model NMTF with (k1, k2, k3, k4, k5)=({}, {}, {}, {}, {}) and {} initialization'.format(self.K[0], self.K[1], self.K[2], self.K[3], self.K[4], self.init_method)
