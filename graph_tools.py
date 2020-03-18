import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import ast


class GraphContainer:
    """Class that holds all information to solve the repeater location problem with cplex."""
    def __init__(self, graph):
        self.graph = graph
        self.end_nodes = []
        self.possible_rep_nodes = []
        for node, nodedata in graph.nodes.items():
            if nodedata["type"] == 'end_node':
                self.end_nodes.append(node)
            else:
                self.possible_rep_nodes.append(node)
        self.num_end_nodes = len(self.end_nodes)
        if self.num_end_nodes == 0:
            raise ValueError("Must have at least one city.")
        self.num_repeater_nodes = len(self.possible_rep_nodes)
        if self.num_repeater_nodes == 0:
            # Trivial graph
            return
        for city in self.end_nodes:
            if city not in self.graph.nodes():
                raise ValueError("City {} not found in list of nodes {}".format(city, self.graph.nodes()))
        self.unique_end_node_pairs = list(itertools.combinations(self.end_nodes, r=2))
        self.num_unique_pairs = len(list(self.unique_end_node_pairs))
        # Add length parameter to edges if this is not defined yet
        for i, j in graph.edges():
            if 'length' not in graph[i][j]:
                if 'Longitude' in graph.nodes[self.possible_rep_nodes[0]]:
                    self._compute_dist_lat_lon(graph)
                else:
                    self._compute_dist_cartesian(graph)
                break
        # print("Constructed graph container. Number of nodes: {}, number of edges {}, number of cities to connect: {}."
        #       .format(self.num_nodes, len(self.graph.edges()), self.num_cities))

        # total_length = 0
        # num_edges = len(graph.edges())
        # max_length = 0
        # min_length = 1e10
        # for i, j in graph.edges():
        #     total_length += graph[i][j]['length']
        #     if graph[i][j]['length'] > max_length:
        #         max_length = graph[i][j]['length']
        #     if graph[i][j]['length'] < min_length:
        #         min_length = graph[i][j]['length']
        # print("Total number of nodes:", len(graph.nodes()))
        # print("Total number of edges:", len(graph.edges()))
        # print("Average length is", total_length/num_edges)
        # print("Maximum edge length is ", max_length)
        # print("Minimum edge length is ", min_length)

    @staticmethod
    def _compute_dist_lat_lon(graph):
        """Compute the distance in km between two points based on their latitude and longitude.
        Assumes both are given in radians."""
        R = 6371  # Radius of the earth in km
        for edge in graph.edges():
            node1 = edge[0]
            node2 = edge[1]
            lon1 = np.radians(graph.nodes[node1]['Longitude'])
            lon2 = np.radians(graph.nodes[node2]['Longitude'])
            lat1 = np.radians(graph.nodes[node1]['Latitude'])
            lat2 = np.radians(graph.nodes[node2]['Latitude'])
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lon / 2) ** 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            dist = R * c
            graph.edges[node1, node2]['length'] = dist

    @staticmethod
    def _compute_dist_cartesian(graph):
        """Compute the distance in km between two points based on their Cartesian coordinates."""
        for edge in graph.edges():
            node1 = edge[0]
            node2 = edge[1]
            dx = np.abs(graph.nodes[node1]['xcoord'] - graph.nodes[node2]['xcoord'])
            dy = np.abs(graph.nodes[node1]['ycoord'] - graph.nodes[node2]['ycoord'])
            dist = np.round(np.sqrt(np.square(dx) + np.square(dy)), 5)
            graph.edges[node1, node2]['length'] = dist


def read_graph(file, draw=False):
    G = nx.read_gml(file)
    file_name = file[0:-4]
    if file_name == 'Surfnet':
        # We are dealing with the Netherlands dataset
        # end_node_list = ["Vlissingen", "Groningen", "DenHaag", "Maastricht"]
        # end_node_list = ["Amsterdam", "Leiden", "Groningen", "Maastricht", "Nijmegen", "Utrecht", "Delft",
        #                  "Rotterdam", "Tilburg"]
        end_node_list = ["Middelburg", "Groningen", "Maastricht", "Enschede", "Delft", "Amsterdam", "Utrecht", "Den_Helder"]
    elif file_name == "SurfnetFiberdata":
        end_node_list = ["Asd001b", "Mt001a", "GN001A", "DT001A"]
    elif file_name == "SurfnetCore":
        end_node_list = ["Amsterdam 1", "Delft 1", "Groningen 1", "Maastricht", "Enschede 2"]
    elif file_name == 'Colt':
        # This is the European dataset
        # QIA Members: IQOQI, UOI (Innsbruck), CNRS (Paris), ICFO (Barcelona), IT (Lisbon),
        #              MPQ (Garching [DE] -> Munich), NBI (Copenhagen), QuTech (Delft -> The Hague), UOB (Basel),
        #              UOG (Geneva)
        # NOTE: Graching replaced by Munic, Delft by The Hague
        # end_node_list = ['Innsbruck', 'Paris', 'Barcelona', 'Lisbon', 'Copenhagen', 'TheHague', 'Basel', 'Geneva',
        #             'Stuttgart']
        end_node_list = ['Innsbruck', 'Paris', 'Copenhagen', 'TheHague', 'Rome', 'Barcelona']
    else:
        raise NotImplementedError("Dataset {} not implemented (no city list defined)".format(file_name))
    pos = {}
    color_map = []

    for node, nodedata in G.nodes.items():
        if "position" in nodedata:
            pos[node] = ast.literal_eval(nodedata["position"])
        elif "Longitude" in nodedata and "Latitude" in nodedata:
            pos[node] = [nodedata['Longitude'], nodedata['Latitude']]
        else:
            raise ValueError("Cannot determine node position.")
        if node in end_node_list:
            color_map.append('green')
            nodedata['type'] = 'end_node'
        else:
            color_map.append([30 / 255, 144 / 255, 255 / 255])
            nodedata['type'] = 'repeater_node'

    nx.set_node_attributes(G, pos, name='pos')

    if draw:
        draw_graph(G)

    return G


def create_graph(draw=False):
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
        graph.nodes[node]['type'] = 'repeater_node'
    for city in city_list:
        # Add the four city nodes and randomly connect them to 3 other nodes
        graph.add_node(city)
        graph.nodes[city]['xcoord'] = node_pos[city][0]
        graph.nodes[city]['ycoord'] = node_pos[city][1]
        graph.nodes[city]['type'] = 'end_node'
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
        draw_graph(graph)
    # Convert node labels to strings
    label_remapping = {key: str(key) for key in range(num_nodes)}
    graph = nx.relabel_nodes(graph, label_remapping)

    return graph


def create_graph_on_unit_cube(n_repeaters, radius, draw, seed=2):
    """Create a geometric graph where nodes randomly get assigned a position. Two nodes are connected if their distance
    does not exceed the given radius."""
    np.random.seed = seed
    G = nx.random_geometric_graph(n=n_repeaters, radius=radius, dim=2, seed=seed)
    for node in G.nodes():
        G.nodes[node]['type'] = 'repeater_node'
    color_map = ['blue'] * len(G.nodes)
    # Create the end nodes
    G.add_node("C1", pos=[0, 0], type='end_node')
    G.add_node("C2", pos=[1, 1], type='end_node')
    G.add_node("C3", pos=[0, 1], type='end_node')
    G.add_node("C4", pos=[1, 0], type='end_node')
    G.add_edge("C1", 1)
    G.add_edge("C1", 5)
    G.add_edge("C1", 9)
    G.add_edge("C2", 0)
    G.add_edge("C2", 7)
    G.add_edge("C2", 8)
    G.add_edge("C3", 4)
    G.add_edge("C3", 6)
    G.add_edge("C3", 7)
    G.add_edge("C4", 3)
    G.add_edge("C4", 5)
    G.add_edge("C4", 8)
    color_map.extend(['green'] * 4)
    for node in G.nodes():
        G.nodes[node]['xcoord'] = G.nodes[node]['pos'][0]
        G.nodes[node]['ycoord'] = G.nodes[node]['pos'][1]
    # Convert node labels to strings
    label_remapping = {key: str(key) for key in G.nodes() if type(key) is not str}
    G = nx.relabel_nodes(G, label_remapping)
    if draw:
        draw_graph(G)
    return G


def create_graph_and_partition(num_nodes, radius, draw=False, seed=None):
    # seed = 19, 69
    """Create a geometric graph where nodes randomly get assigned a position. Two nodes are connected if their distance
        does not exceed the given radius. Finds the convex hull of this graph and assigns a random subset of this hull
        as consumer nodes, i.e. end nodes."""
    np.random.seed(seed)
    G = nx.random_geometric_graph(n=num_nodes, radius=radius, dim=2, seed=seed)
    # Check for isolated nodes (degree 0) which should not be assigned as end nodes
    isolates = list(nx.isolates(G))
    if len(isolates) > 0:
        return None
    repeater_nodes = list(G.nodes())
    for node in G.nodes():
        G.nodes[node]['type'] = 'repeater_node'
    pos = nx.get_node_attributes(G, 'pos')
    hull = ConvexHull(np.array(list(pos.values())))
    # print(hull.vertices)
    end_nodes = []
    for node in hull.vertices:
        if node not in isolates:
            G.nodes[node]['type'] = 'end_node'
            repeater_nodes.remove(node)
            end_nodes.append(node)
    for node in G.nodes():
        G.nodes[node]['xcoord'] = G.nodes[node]['pos'][0]
        G.nodes[node]['ycoord'] = G.nodes[node]['pos'][1]

    if draw:
        draw_graph(G)
    # Convert node labels to strings
    label_remapping = {key: str(key) for key in G.nodes() if type(key) is not str}
    G = nx.relabel_nodes(G, label_remapping)
    return G


def draw_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    repeater_nodes = []
    end_nodes = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'repeater_node':
            repeater_nodes.append(node)
        else:
            end_nodes.append(node)
    fig, ax = plt.subplots(figsize=(10, 7))
    end_nodes = nx.draw_networkx_nodes(G=G, pos=pos, nodelist=end_nodes, node_shape='s', node_size=500,
                                       node_color=[[1.0, 140 / 255, 0.]], label="End Node")
    end_nodes.set_edgecolor('k')
    rep_nodes = nx.draw_networkx_nodes(G=G, pos=pos, nodelist=repeater_nodes, node_size=500,
                                       node_color=[[0 / 255, 166 / 255, 214 / 255]], label="Repeater Node")
    rep_nodes.set_edgecolor('k')
    label_pos = {city: [pos[city][0], pos[city][1] - 0.07] for city in pos}
    nx.draw_networkx_labels(G=G, pos=label_pos, font_size=11, font_weight="bold")
    nx.draw_networkx_edges(G=G, pos=pos, width=0.3)
    plt.axis('off')
    margin = 0.33
    fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
    ax.axis('equal')
    fig.tight_layout()
    plt.show()
