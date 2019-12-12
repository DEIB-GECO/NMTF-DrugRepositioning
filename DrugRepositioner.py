from network import Network

network = Network("data/graph_topology.tsv")
initial_error = network.get_error()
print("initial error: {}".format(initial_error))

for i in range(200):
    network.update()
    error = network.get_error()
    print("iteration {}, error = {}".format(i, error))

