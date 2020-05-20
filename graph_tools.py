import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import ast


class GraphContainer:
    """
    Class that holds the graph and some relevant properties to solve the repeater allocation problem.

    Parameters
    ----------
    graph : networkx.Graph
        The NetworkX undirected graph representing the network (a set of nodes and edges).

    Attributes
    ----------
    end_nodes : list
        List of the end nodes, i.e. the set C in the paper.
    num_end_nodes : int
        The total number of end nodes, so the size of set C.
    possible_rep_nodes : list
        List of all possible repeater node locations, i.e. the set R in the paper.
    num_repeater_nodes : int
        The total number of repeater nodes, so the size of the set R.
    unique_end_node_pairs : list
        List of all unique source-destination pairs, i.e. the set Q in the paper. Note that we assume that the graph
        is undirected and therefore the inverse path from destination to source is always the same.
    num_unique_pairs : int
        The total number of unique source-destination pairs, so the sie of the set Q.
    """
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

    @staticmethod
    def _compute_dist_lat_lon(graph):
        """Compute the distance in km between two points based on their latitude and longitude.
        Assumes both are given in radians."""
        R = 6371  # Radius of the earth in km
        for edge in graph.edges():
            node1, node2 = edge
            lon1 = np.radians(graph.nodes[node1]['Longitude'])
            lon2 = np.radians(graph.nodes[node2]['Longitude'])
            lat1 = np.radians(graph.nodes[node1]['Latitude'])
            lat2 = np.radians(graph.nodes[node2]['Latitude'])
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lon / 2) ** 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            dist = np.round(R * c, 5)
            graph.edges[node1, node2]['length'] = dist

    @staticmethod
    def _compute_dist_cartesian(graph):
        """Compute the distance in km between two points based on their Cartesian coordinates."""
        for edge in graph.edges():
            node1, node2 = edge
            dx = np.abs(graph.nodes[node1]['xcoord'] - graph.nodes[node2]['xcoord'])
            dy = np.abs(graph.nodes[node1]['ycoord'] - graph.nodes[node2]['ycoord'])
            dist = np.round(np.sqrt(np.square(dx) + np.square(dy)), 5)
            graph.edges[node1, node2]['length'] = dist

    def print_graph_data(self):
        total_length, min_length, max_length = 0, 1e10, 0
        num_edges = len(self.graph.edges())
        for i, j in self.graph.edges():
            total_length += self.graph[i][j]['length']
            if self.graph[i][j]['length'] > max_length:
                max_length = self.graph[i][j]['length']
            if self.graph[i][j]['length'] < min_length:
                min_length = self.graph[i][j]['length']
        print("Total number of nodes:", len(self.graph.nodes()))
        print("Total number of edges:", num_edges)
        print("Average length is", total_length/num_edges)
        print("Maximum edge length is ", max_length)
        print("Minimum edge length is ", min_length)


def read_graph_from_gml(file, draw=False):
    G = nx.read_gml(file)
    file_name = file[0:-4]
    if file_name == 'Surfnet':
        # The Dutch Topology Zoo dataset
        end_node_list = ["Middelburg", "Groningen", "Maastricht", "Enschede", "Delft", "Amsterdam", "Utrecht", "Den_Helder"]
    elif file_name == "SurfnetFiberdata":
        end_node_list = ["Asd001b", "Mt001a", "GN001A", "DT001A"]
    elif file_name == "SurfnetCore":
        end_node_list = ["Amsterdam 1", "Delft 1", "Groningen 1", "Maastricht", "Enschede 2"]
    elif file_name == 'Colt':
        # The European Topology Zoo dataset
        # Use QIA members: IQOQI, UOI (Innsbruck), CNRS (Paris), ICFO (Barcelona), IT (Lisbon),
        #              MPQ (Garching [DE] -> Munich), NBI (Copenhagen), QuTech (Delft -> The Hague), UOB (Basel),
        #              UOG (Geneva)
        # NOTE: Graching replaced by Munich, Delft by The Hague
        end_node_list = ['Innsbruck', 'Paris', 'Barcelona', 'Lisbon', 'Copenhagen', 'TheHague', 'Basel', 'Geneva',
                         'Stuttgart']
    else:
        raise NotImplementedError("Dataset {} not implemented (no city list defined)".format(file_name))
    pos = {}
    for node, nodedata in G.nodes.items():
        if "position" in nodedata:
            pos[node] = ast.literal_eval(nodedata["position"])
        elif "Longitude" in nodedata and "Latitude" in nodedata:
            pos[node] = [nodedata['Longitude'], nodedata['Latitude']]
        else:
            raise ValueError("Cannot determine node position.")
        if node in end_node_list:
            nodedata['type'] = 'end_node'
        else:
            nodedata['type'] = 'repeater_node'
    nx.set_node_attributes(G, pos, name='pos')
    if draw:
        draw_graph(G)
    return G


def create_graph_on_unit_cube(n_repeaters, radius, draw, seed=2):
    """Create a geometric graph where nodes randomly get assigned a position. Two nodes are connected if their distance
    does not exceed the given radius."""
    np.random.seed = seed
    G = nx.random_geometric_graph(n=n_repeaters, radius=radius, dim=2, seed=seed)
    for node in G.nodes():
        G.nodes[node]['type'] = 'repeater_node'
    color_map = ['blue'] * len(G.nodes)
    # Create the end nodes
    G.add_node("C", pos=[0, 0], type='end_node')
    G.add_node("B", pos=[1, 1], type='end_node')
    G.add_node("A", pos=[0, 1], type='end_node')
    G.add_node("D", pos=[1, 0], type='end_node')
    G.nodes[3]['pos'] = [0.953, 0.750]
    G.nodes[5]['pos'] = [0.25, 0.50]
    # Manually connect the end nodes to the three nearest nodes
    G.add_edge("C", 8)
    G.add_edge("C", 5)
    G.add_edge("C", 2)
    G.add_edge("B", 9)
    G.add_edge("B", 4)
    G.add_edge("B", 3)
    G.add_edge("A", 1)
    G.add_edge("A", 2)
    G.add_edge("A", 9)
    G.add_edge("D", 3)
    G.add_edge("D", 6)
    G.add_edge("D", 7)
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
    """Create a geometric graph where nodes randomly get assigned a position. Two nodes are connected if their distance
    does not exceed the given radius. Finds the convex hull of this graph and assigns a random subset of this hull
    as end nodes."""
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
    fig, ax = plt.subplots(figsize=(7, 7))
    end_nodes = nx.draw_networkx_nodes(G=G, pos=pos, nodelist=end_nodes, node_shape='s', node_size=1500,
                                       node_color=[[1.0, 120 / 255, 0.]], label="End Node", linewidths=3)
    end_nodes.set_edgecolor('k')
    rep_nodes = nx.draw_networkx_nodes(G=G, pos=pos, nodelist=repeater_nodes, node_size=1500,
                                       node_color=[[1, 1, 1]], label="Repeater Node")
    rep_nodes.set_edgecolor('k')
    end_node_labels = {}
    repeater_node_labels = {}
    for node, nodedata in G.nodes.items():
        # labels[node] = node
        if G.nodes[node]['type'] == 'end_node':  # or node in self.repeater_nodes_chosen:
            end_node_labels[node] = node
        else:
            repeater_node_labels[node] = node
    nx.draw_networkx_labels(G=G, pos=pos, labels=end_node_labels, font_size=30, font_weight="bold", font_color="w",
                            font_family='serif')
    # nx.draw_networkx_labels(G=G, pos=pos, labels=repeater_node_labels, font_size=30, font_weight="bold")
    nx.draw_networkx_edges(G=G, pos=pos, width=1)
    plt.axis('off')
    margin = 0.33
    fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
    ax.axis('equal')
    fig.tight_layout()
    plt.show()
