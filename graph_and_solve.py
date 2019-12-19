import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from programs import EdgeBasedProgram, PathBasedProgram
from program_alt_obj import MinMaxEdgeBasedProgram, MinRepEdgeBasedProgram


def read_graph(file, draw=False):
    G = nx.read_gml(file)
    file_name = file[0:-4]
    if file_name == 'Surfnet':
        # We are dealing with the Netherlands dataset
        city_list = ["Vlissingen", "Groningen", "DenHaag", "Maastricht"]
    elif file_name == 'Colt':
        # This is the European dataset
        # QIA Members: IQOQI, UOI (Innsbruck), CNRS (Paris), ICFO (Barcelona), IT (Lisbon),
        #              MPQ (Garching [DE] -> Munich), NBI (Copenhagen), QuTech (Delft -> The Hague), UOB (Basel),
        #              UOG (Geneva)
        # NOTE: Graching replaced by Munic, Delft by The Hague
        city_list = ['Innsbruck', 'Paris', 'Barcelona', 'Lisbon', 'Copenhagen', 'TheHague', 'Basel', 'Geneva',
                     'Stuttgart']
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
    # G = create_graph(node_pos=node_pos, draw=False)
    filename = "Colt.gml"
    G = read_graph(filename, draw=False)
    R_max = 1
    L_max = 1
    prog = MinRepEdgeBasedProgram(graph=G, num_allowed_repeaters=R_max, L_max=L_max, alpha=0, read_from_file=True)
    prog.update_parameters(L_max_new=1000, R_max_new=6, alpha_new=1/75000)
    print(prog.solve(draw_solution=True))
    # dict_list = []
    # for alpha in [0, 1 / 75000]:
    #     for L_max in range(550, 1275, 25):
    #         R_max = 6
    #         prog.update_parameters(L_max_new=L_max, R_max_new=R_max, alpha_new=alpha)
    #         output = prog.solve()
    #         print(output)
    #         dict_list.append(output)
    # final_dict = {k: [d[k] for d in dict_list] for k in dict_list[0]}
    # now = str(datetime.now())[0:-7].replace(" ", "_").replace(":", "-")
    # pd.DataFrame(final_dict).to_csv('./data/repAllocData_{}_{}.csv'.format(filename[0:-4], now), index=False)
    # print(prog.solve())
    # prog.update_parameters(L_max_new=100, R_max_new=10, alpha_new=1/2500)
    # print(prog.solve())
    # for L_max in range(65, 105, 5):
    #     for R_max in range(10, 11):
    #         prog.update_lmax_rmax(L_max_new=L_max, R_max_new=R_max)
    #         # print("Running program for L_max = {} and R_max = {}".format(L_max, R_max))
    #         print(prog.solve())

    # prog = EdgeBasedProgram(G, R_max)
    # prog = MinMaxEdgeBasedProgram(G, R_max)
    # prog = PathBasedProgram(G, R_max)


