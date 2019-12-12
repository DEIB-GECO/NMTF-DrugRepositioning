from AssociationMatrix import AssociationMatrix

class Network():
    def __init__(self, graph_topology_file):
        self.init_strategy = "random"
        self.integration_strategy = lambda x, y: x.intersection(y)
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


    def get_error(self):
        return sum([am.get_error() for am in self.association_matrices])

    def update(self):
        for am in self.association_matrices:
            am.update()