import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

class AssociationMatrix():
    def __init__(self, filename, leftds, rightds, left_sorted_terms, right_sorted_terms, k1, k2):
        self.filename = filename
        self.leftds = leftds
        self.rightds = rightds
        self.dep_own_right_other_right = []
        self.dep_own_right_other_left = []
        self.dep_own_left_other_right = []
        self.dep_own_left_other_left = []
        self.left_sorted_term_list = left_sorted_terms
        self.right_sorted_term_list = right_sorted_terms
        self.k1 = k1
        self.k2 = k2

        with open(self.filename, "r") as f:
            data_graph = [element.strip().split("\t") for element in f.readlines()]
        self.edges = [(el[0],el[1])
                      for el in data_graph
                      if el[0] in set(self.left_sorted_term_list) and el[1] in set(self.right_sorted_term_list)]

        graph = nx.Graph()
        graph.add_nodes_from(list(self.left_sorted_term_list), bipartite=0)
        graph.add_nodes_from(list(self.right_sorted_term_list), bipartite=1)
        graph.add_edges_from(self.edges)

        self.association_matrix = nx.algorithms.bipartite.matrix.biadjacency_matrix(graph,
                                                                                    self.left_sorted_term_list,
                                                                                    self.right_sorted_term_list)
        print("association_matrix = {}".format(np.sum(self.association_matrix)))
        self.association_matrix = self.association_matrix.toarray()

        self.G_left = None
        self.G_left_primary = False
        self.G_right = None
        self.G_right_primary = False
        self.S = None

        print(self.leftds, self.rightds, self.association_matrix.shape)


    def initialize(self, initialize_strategy):
        if initialize_strategy == "random":
            if self.G_left is None:
                self.G_left = np.random.rand(self.association_matrix.shape[0], self.k1)
                self.G_left_primary = True
            if self.G_right is None:
                self.G_right = np.random.rand(self.association_matrix.shape[1], self.k2)
                self.G_right_primary = True
        elif initialize_strategy == "kmeans":
            if self.G_left is None:
                km = KMeans(n_clusters=self.k1).fit(self.association_matrix)
                self.G_left = np.zeros((self.association_matrix.shape[0], self.k1))
                for row in range(self.association_matrix.shape[0]):
                    for col in range(self.k1):
                        self.G_left[row,col] = np.linalg.norm(self.association_matrix[row] - km.cluster_centers_[col])
                self.G_left_primary = True
            if self.G_right is None:
                km = KMeans(n_clusters=self.k2).fit(self.association_matrix.transpose())
                self.G_right = np.zeros((self.association_matrix.shape[1], self.k2))
                for row in range(self.association_matrix.shape[1]):
                    for col in range(self.k2):
                        self.G_right[row,col] = np.linalg.norm(self.association_matrix.transpose()[row] - km.cluster_centers_[col])
                self.G_right_primary = True



        for am in self.dep_own_left_other_left:
            if am.G_left is None:
                am.G_left = self.G_left
        for am in self.dep_own_left_other_right:
            if am.G_right is None:
                am.G_right = self.G_left
        for am in self.dep_own_right_other_left:
            if am.G_left is None:
                am.G_left = self.G_right
        for am in self.dep_own_right_other_right:
            if am.G_right is None:
                am.G_right = self.G_right

        self.S = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])

    def get_error(self):
        self.rebuilt_association_matrix = np.linalg.multi_dot([self.G_left, self.S, self.G_right.transpose()])
        return np.linalg.norm(self.rebuilt_association_matrix - self.association_matrix)

    def create_update_rules(self):
        if self.G_right_primary:
            def update_G_r():
                num = np.linalg.multi_dot([self.association_matrix.transpose(), self.G_left, self.S])
                den = np.linalg.multi_dot([self.G_right, self.G_right.transpose(), self.association_matrix.transpose(), self.G_left, self.S])
                for am in self.dep_own_right_other_right:
                    num += np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
                    den += np.linalg.multi_dot([am.G_right, am.G_right.transpose(), am.association_matrix.transpose(), am.G_left, am.S])
                for am in self.dep_own_right_other_left:
                    num += np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
                    den += np.linalg.multi_dot([am.G_left, am.G_left.transpose(), am.association_matrix, am.G_right, am.S.transpose()])
                div=np.sqrt(np.divide(num, den+0.00000001))
                return np.multiply(self.G_right, div)

            self.update_G_right = update_G_r

        if self.G_left_primary:
            def update_G_l():
                num = np.linalg.multi_dot([self.association_matrix, self.G_right, self.S.transpose()])
                den = np.linalg.multi_dot([self.G_left,self.G_left.transpose(), self.association_matrix, self.G_right, self.S.transpose()])
                for am in self.dep_own_left_other_left:
                    num += np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
                    den += np.linalg.multi_dot([self.G_left, self.G_left.transpose(), self.association_matrix, self.G_right, self.S.transpose()])
                for am in self.dep_own_left_other_right:
                    num += np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
                    den += np.linalg.multi_dot([am.G_right, am.G_right.transpose(), am.association_matrix.transpose(), am.G_left, am.S])
                div = np.sqrt(np.divide(num, den+0.00000001))
                return np.multiply(self.G_left, div)

            self.update_G_left = update_G_l

        def update_S():
            num = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])
            den = np.linalg.multi_dot([self.G_left.transpose(), self.G_left, self.S, self.G_right.transpose(), self.G_right])
            div = np.sqrt(np.divide(num, den+0.00000001))
            return np.multiply(self.S, div)

        self.update_S = update_S

    def update(self):
        if self.G_right_primary:
            self.G_right = self.update_G_right()
        if self.G_left_primary:
            self.G_left = self.update_G_left()
        self.S = self.update_S()

    def __str__(self):
        own_r_other_r = ", ".join([x.filename for x in self.dep_own_right_other_right])
        own_r_other_l = ", ".join([x.filename for x in self.dep_own_right_other_left])
        own_l_other_l = ", ".join([x.filename for x in self.dep_own_left_other_left])
        own_l_other_r = ", ".join([x.filename for x in self.dep_own_left_other_right])

        return "left: {}, right: {}, filename: {}\n own_r_other_l: {}\nown_r_other_r: {}\nown_l_other_l: {}\nown_l_other_r: {}"\
            .format(self.leftds, self.rightds, self.filename, own_r_other_l, own_r_other_r, own_l_other_l, own_l_other_r)
