import cplex
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

city_list = ['Lei', 'Haa', 'Ams', 'Del']  # list of cities
city_pos = {'Lei': [0, 0],
            'Haa': [10, 10],
            'Ams': [0, 10],
            'Del': [10, 0]}
num_cities = len(city_list)  # number of customers (cities)
unique_pairs = list(itertools.combinations(city_list, r=2))  # list of unique pair tuples
num_pairs = len(list(unique_pairs))  # number of unique customer pairs
num_nodes = 8  # number of non-customer nodes
node_pos = {0: [1, 1],
            1: [3, 2],
            2: [8, 7],
            3: [2, 4],
            4: [3, 7],
            5: [4, 5],
            6: [8, 1],
            7: [7, 4]}
all_pos = {**city_pos, **node_pos}
R = 1  # maximum number of repeaters
M = num_pairs  # dummy parameter for linking constraint
# Set fixed seed for reproducibility
np.random.seed(13)
# Variable map for referencing
varmap = {}


def create_graph(draw=False):
    # Create a graph and add random weights
    graph = nx.fast_gnp_random_graph(n=num_nodes, p=0.8, seed=np.random)
    for (i, j) in graph.edges():
        # Add weight to each edge based on distance between nodes
        graph.edges[i, j]['weight'] = compute_dist(i, j)
    for city in city_list:
        # Add four city nodes and randomly connect them to 3 other nodes
        graph.add_node(city)
        connected_edges = np.random.choice(range(num_nodes), 3, replace=False)
        for edge in connected_edges:
            graph.add_edge(city, edge, weight=compute_dist(city, edge))
    color_map = ['blue'] * len(graph.nodes)
    # Save seed for drawing
    global numpy_seed
    numpy_seed = np.random.get_state()
    for idx, node in enumerate(graph.nodes):
        if type(node) == str:
            color_map[idx] = 'olive'
    if draw:
        plt.subplots()
        nx.draw(graph, with_labels=True, font_weight='bold',
                node_color=color_map, pos=all_pos)
        plt.show()

    return graph, color_map


def compute_dist(i, j):
    # Compute the Pythagorean distance from one node to another
    dx = np.abs(all_pos[i][0] - all_pos[j][0])
    dy = np.abs(all_pos[i][1] - all_pos[j][1])
    return np.round(np.sqrt(np.square(dx) + np.square(dy)))


def solve_cplex(graph):
    # Create new CPLEX model
    prob = cplex.Cplex()
    # Set objective to minimization
    prob.objective.set_sense(prob.objective.sense.minimize)
    # Add constraints for connecting each pair exactly once
    prob.linear_constraints.add(rhs=[1] * num_pairs, senses=['E'] * num_pairs)
    # Add variables column wise
    add_variables(prob=prob, graph=graph)
    # Solve for the optimal solution and print
    prob.solve()
    for i, x in enumerate(prob.solution.get_values()):
        if x == 1.0:
            path = prob.variables.get_names(i)
            print('Chosen path:', path, 'with cost', prob.objective.get_linear(path))

    print('Optimal objective value:', prob.solution.get_objective_value())
    return


def add_variables(prob, graph):
    variable_dict = {}
    for idx, pair in enumerate(unique_pairs):
        # Generate all paths with a maximum length of 4
        paths = nx.all_simple_paths(graph, source=pair[0], target=pair[1], cutoff=6)
        for path in paths:
            cost, pathname = compute_cost(graph, path)
            if cost != np.inf:  # only add variables with finite cost
                column = [cplex.SparsePair(ind=[idx], val=[1.0])]
                var = prob.variables.add(obj=[cost], lb=[0.0], ub=[1.0], types=['B'],
                                         columns=column, names=[pathname])
                variable_dict[var] = path


def compute_cost(graph, path):
    cost = 0
    pathname = 'x'
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]['weight']
        pathname += '_' + str(path[i])

    pathname += '_' + str(path[-1])
    return cost, pathname


if __name__ == "__main__":
    graph, color_map = create_graph(draw=True)
    print('Shortest Dijkstra paths:')
    tot_cost = 0
    for pair in unique_pairs:
        shortest_path = nx.shortest_path(graph, source=pair[0], target=pair[1], weight='weight')
        cost, _ = compute_cost(graph, shortest_path)
        tot_cost += cost
        print(shortest_path, 'with cost', cost)
    print('Sum of shortest path costs:', tot_cost)
    solve_cplex(graph=graph)




