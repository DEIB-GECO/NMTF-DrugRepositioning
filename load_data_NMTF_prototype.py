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
from AssociationMatrix import AssociationMatrix
from network import Network


class loader:

    def __init__(self, graph_topology_file):
        self.init_strategy = "random"
        self.association_matrices = []
        self.datasets = {}
        with open(graph_topology_file) as f:
            for line in f:
                if line.strip().startswith("#integration.strategy"):
                    s = line.strip().split("\t")
                    if s[1] == "union":
                        self.integration_strategy = lambda x, y: x.union(y)
                    elif s[1] == "intersection":
                        self.integration_strategy = lambda x, y: x.intersection(y)
                    else:
                        print("Option '{}' not supported".format(s[1]))
                        exit(-1)

        #For each category of nodes, compute the intersection between the different matrices
        with open(graph_topology_file) as f:
            for line in f:
                if not line.strip().startswith("#"):
                    s = line.strip().split()
                    filename = s[2]
                    ds1_name = s[0].upper()
                    ds2_name = s[1].upper()

                    ds1_entities = set()
                    ds2_entities = set()
                    with open(filename) as af:
                        for edge in af:
                            s_edge = edge.strip().split("\t")
                            ds1_entities.add(s_edge[0])
                            ds2_entities.add(s_edge[1])
                    if ds1_name in self.datasets:
                        self.datasets[ds1_name] = self.integration_strategy(self.datasets[ds1_name], ds1_entities)
                    else:
                        self.datasets[ds1_name] = ds1_entities

                    if ds2_name in self.datasets:
                        self.datasets[ds2_name] = self.integration_strategy(self.datasets[ds2_name], ds2_entities)
                    else:
                        self.datasets[ds2_name] = ds2_entities

        #sort the nodes, such that each matrix receives the same ordered list of nodes
        for k in self.datasets.keys():
            self.datasets[k] = list(sorted(list(self.datasets[k])))

        print(self.datasets.keys())

        with open(graph_topology_file) as f:
            for line in f:
                if not line.strip().startswith("#"):
                    s = line.strip().split()
                    filename = s[2]
                    ds1_name = s[0].upper()
                    ds2_name = s[1].upper()
                    k1 = int(s[3])
                    k2 = int(s[4])

                    self.association_matrices.append(
                        AssociationMatrix(filename,
                                          ds1_name,
                                          ds2_name,
                                          self.datasets[ds1_name],
                                          self.datasets[ds2_name],
                                          k1,
                                          k2))

        for m1 in self.association_matrices:
            for m2 in self.association_matrices:
                if m1 != m2:
                    if m1.leftds == m2.leftds:
                        m1.dep_own_left_other_left.append(m2)
                        #m2.dep_own_left_other_left.append(m1)
                    elif m1.rightds == m2.rightds:
                        m1.dep_own_right_other_right.append(m2)
                       #m2.dep_own_right_other_right.append(m1)
                    elif m1.rightds == m2.leftds:
                        m1.dep_own_right_other_left.append(m2)
                        #m2.dep_own_left_other_right.append(m1)
                    elif m1.leftds == m2.rightds:
                        m1.dep_own_left_other_right.append(m2)
                        #m2.dep_own_right_other_left.append(m1)


        for am in self.association_matrices:
            print(am)
            print("-------------------")

        for am in self.association_matrices:
            am.initialize(self.init_strategy)

        for am in self.association_matrices:
            am.create_update_rules()





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

        with open(self.drugsdiseases_file, "r") as f:
            data_graph = [element.strip().split("\t") for element in f.readlines()]
        f.close()
        edges25 = [(element[0],element[1]) for element in data_graph]#edges25 contains edges between drugs and diseases
        diseases = list(set(element[1] for element in data_graph))
        diseases.sort()
        
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
        G.add_edges_from(edges25)

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
