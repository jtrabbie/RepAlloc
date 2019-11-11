import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

from programs import EdgeBasedProgram, PathBasedProgram


def read_graph(file, draw=False):
    G = nx.read_gml(file)
    file_name = file[0:-4]
    if file_name == 'Surfnet':
        # We are dealing with the Netherlands dataset
        city_list = ["Vlissingen", "Groningen", "Den_Haag", "Maastricht"]
    elif file_name == 'Colt':
        # This is the European dataset
        city_list = ['Dublin', 'Bordeaux', 'Trieste', 'The_Hague', 'Prague']
    else:
        raise NotImplementedError("Dataset {} not implemented (no city list defined)".format(file_name))
    pos = {}
    color_map = []
    for node, nodedata in G.nodes.items():
        pos[node] = [nodedata['Longitude'], nodedata['Latitude']]
        if node in city_list:
            color_map.append('green')
            nodedata['type'] = 'City'
        else:
            color_map.append([30/255, 144/255, 255/255])
            nodedata['type'] = 'Node'

    if draw:
        plt.figure(1)
        nx.draw(G, with_labels=True, font_weight='bold', pos=pos, node_color=color_map, node_size=200)
        plt.show()

    return G


def create_graph(draw=False, node_pos=None):
    if node_pos is None:
        raise NotImplementedError("Random graph must contain x and y coordinates")
    city_list = []
    num_nodes = 0
    for key in node_pos:
        if type(key) is str:
            city_list.append(key)
        else:
            num_nodes += 1
    if len(city_list) == 0:
        raise ValueError("Must have at least one city")
    # Create a random graph
    graph = nx.fast_gnp_random_graph(n=num_nodes, p=0.8, seed=np.random)
    for node in graph.nodes():
        graph.nodes[node]['xcoord'] = node_pos[node][0]
        graph.nodes[node]['ycoord'] = node_pos[node][1]
        graph.nodes[node]['type'] = 'Node'
    for city in city_list:
        # Add the four city nodes and randomly connect them to 3 other nodes
        graph.add_node(city)
        graph.nodes[city]['xcoord'] = node_pos[city][0]
        graph.nodes[city]['ycoord'] = node_pos[city][1]
        graph.nodes[city]['type'] = 'City'
        connected_edges = np.random.choice(range(num_nodes), 3, replace=False)
        for edge in connected_edges:
            graph.add_edge(city, edge)
    color_map = ['blue'] * len(graph.nodes)
    # Save seed for drawing
    global numpy_seed
    numpy_seed = np.random.get_state()
    for idx, node in enumerate(graph.nodes):
        if type(node) == str:
            color_map[idx] = 'olive'
    if draw:
        plt.figure(1)
        nx.draw(graph, with_labels=True, font_weight='bold',
                node_color=color_map, pos=node_pos)
        plt.show()
    # Convert node labels to strings
    label_remapping = {key: str(key) for key in range(num_nodes)}
    graph = nx.relabel_nodes(graph, label_remapping)

    return graph


if __name__ == "__main__":
    # G = read_graph("Colt.gml", draw=True)
    np.random.seed(188)
    node_pos = {'Lei': [0, 0],
                'Haa': [10, 10],
                'Ams': [0, 10],
                'Del': [10, 0],
                0: [1, 1],
                1: [3, 2],
                2: [8, 7],
                3: [2, 4],
                4: [3, 7],
                5: [4, 5],
                6: [8, 1],
                7: [7, 4]}
    G = create_graph(draw=True, node_pos=node_pos)
    R = 2
    starttime = time.time()
    # prog = EdgeBasedProgram(G, R)
    prog = PathBasedProgram(G, R)
    endtime = time.time()
    print("Constructing graph takes: {}".format(datetime.timedelta(seconds=endtime - starttime)))
    prog.solve()


