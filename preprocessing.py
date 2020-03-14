# This file create a new drug-protein association matrix based on shortest-path calculations.  

import numpy as np
import csv
import networkx as nx
import joblib
from tqdm import tqdm
import pandas as pd

with open('./data/DrugsToProteins.txt', "r") as f:
    R_DP = [element.split() for element in f.readlines()]
f.close()

PROTEINS = list(set([x[1] for x in R_DP]))
DRUGS = list(set([x[0] for x in R_DP]))
drug_set = set(DRUGS)

with open('./data/DrugsToLabels.txt', "r") as f:
    R_DL_all = [[element.split()[0], " ".join(element.split()[1:])] for element in f.readlines()]
    R_DL = [x for x in R_DL_all if x[0] in drug_set]
f.close()

LABELS = list(set([x[1] for x in R_DL]))

with open('./data/ProteinsToProteins.txt', "r") as f:
    R_PP = [element.split()[:2] for element in f.readlines()]
f.close()

revert_edge = lambda x : [x[1], x[0]]

PROTEINS.sort()
DRUGS.sort()

G = nx.DiGraph()
G.add_nodes_from(PROTEINS)
G.add_nodes_from(DRUGS)
G.add_edges_from([revert_edge(x) for x in R_DP])
G.add_edges_from(R_PP + [revert_edge(x) for x in R_PP])
        
n2, n3 = len(DRUGS), len(PROTEINS)

R23_new = np.zeros((n2, n3))

for i in tqdm(range(n2)):
    for j in (range(n3)):
        if nx.has_path(G, PROTEINS[j], DRUGS[i]):
            R23_new[i,j] = nx.shortest_path_length(G, source = PROTEINS[j], target = DRUGS[i])

for i in range(n2):
    for j in range(n3):
        if R23_new[i,j] > 3:
            R23_new[i,j] = 0
            
R23_new_1 = R23_new.astype('float32')
for i in range(n2):
    for j in range(n3):
        if int(R23_new_1[i,j]) != 0:
            R23_new_1[i,j] = 0.2 ** int(R23_new_1[i,j]-1)
            
np.save('R23_enhanced_matrix.npy', R23_new)
