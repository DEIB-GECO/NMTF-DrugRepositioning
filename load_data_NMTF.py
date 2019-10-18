 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:23:26 2019

@author: gaetandissez

This file load create a loader class to import the data from txt and csv files and create required matrices for our problem
"""
#We use networkx as a way to interpret the data and to transform it easily through adjacency matrices
import networkx as nx
import csv

class loader:
    
    #we initialize the loader by giving the paths to the files.
    def __init__(self, f_drugslabels, f_drugsproteins, f_proteinspathways, f_drugsdiseases, f_protprot, f_pathpath):
        self.drugslabels_file = './data/' + f_drugslabels
        self.drugsproteins_file = './data/' + f_drugsproteins
        self.proteinspathways_file = './data/' + f_proteinspathways
        self.drugsdiseases_file = './data/' + f_drugsdiseases
        self.intraprot_file = './data/' + f_protprot
        self.intrapath_file = './data/' + f_pathpath
    
    
    #Then we can use this method to return the needed matrices
    def association_matrices(self):

        drug_set = set()
        protein_set = set()
        with open(self.drugsproteins_file, "r") as dp:
            for line in dp:
                (drug, protein) = line.strip().split("\t")
                drug_set.add(drug)
                protein_set.add(protein)
        dp.close()
        drugs = list(drug_set)
        proteins = list(protein_set)

        pathway_set = set()
        with open(self.proteinspathways_file, "r") as pp:
            for line in pp:
                (protein, pathway) = line.strip().split("\t")
                pathway_set.add(pathway)
        pp.close()
        pathways = list(pathway_set)

        #TODO: check that the loaded proteins list are coherent (same for drugs and paths)

        with open(self.drugslabels_file, "r") as f:
            LabelToDrug = [element.strip().split('\t') for element in f.readlines()]
            labels = [i[1] for i in LabelToDrug if i[0] in drugs]
        f.close()
        labels = list(set(labels))
        labels.sort() #list of labels, sorted in the alphabetical order
        edges12 = [(link[0], link[1]) for link in LabelToDrug] #edges12 contains edges between drugs and labels
        
        with open(self.drugsproteins_file, "r") as f:
            data_graph = [element.split() for element in f.readlines()]
        f.close()
        edges23 = [(element[0],element[1]) for element in data_graph] #edges23 contains edges between drugs and proteins

        with open(self.proteinspathways_file, "r") as f:
            data_graph = [element.split() for element in f.readlines()]
        f.close()
        edges34 = [(element[0],element[1]) for element in data_graph] #edges34 contains edges between proteins and pathways


        with open(self.drugsdiseases_file, "r") as d:
            d.readline()
            reader = csv.reader(d)
            diseases = []
            edgesdd = []
            for row in reader:
                dd = row[0].split(';')
                diseases.append(dd[1].lstrip())
                edgesdd.append((dd[0], dd[1].lstrip()))
        d.close()
        diseases = list(set(diseases))
        diseases.sort()
        diseases.pop()

        #We create the two adjacency matrices (W3 and W4) of intralinks between proteins and between pathways
        W3 = nx.adjacency_matrix(nx.read_weighted_edgelist(self.intraprot_file, nodetype=str), nodelist=proteins)
        W4 = nx.adjacency_matrix(nx.read_weighted_edgelist(self.intrapath_file, nodetype=str), nodelist=pathways)
        
        #The graph G contains all the inter edges of the problem
        G = nx.Graph()
        G.add_nodes_from(labels)
        G.add_nodes_from(proteins)
        G.add_nodes_from(pathways)
        G.add_nodes_from(diseases)
        G.add_nodes_from(drugs)
        G.add_edges_from(edges12)
        G.add_edges_from(edges23)
        G.add_edges_from(edges34)
        G.add_edges_from(edgesdd)

        R = nx.adjacency_matrix(G, nodelist=labels + drugs + proteins + pathways + diseases)
        n_drugs = len(drugs)
        n_proteins = len(proteins) 
        n_labels = len(labels)
        n_pathways = len(pathways)
        R12 = R[:n_labels, n_labels:(n_drugs+n_labels)]
        R23 = R[n_labels:(n_drugs+n_labels), (n_drugs+n_labels):(n_drugs+n_labels+n_proteins)]
        R34 = R[(n_drugs+n_labels):(n_drugs+n_labels+n_proteins), (n_drugs+n_labels+n_proteins):(n_drugs+n_labels+n_proteins+n_pathways)]
        R25 = R[n_labels:(n_drugs+n_labels), (n_drugs+n_labels+n_proteins+n_pathways):]
        
        return R12, R23, R34, R25, W3, W4
