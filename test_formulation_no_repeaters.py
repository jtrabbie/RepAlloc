import cplex
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

city_list = ['Lei', 'Haa', 'Ams', 'Del']
unique_pairs = list(itertools.combinations(city_list, r=2))
num_pairs = len(list(unique_pairs))


def create_graph(draw=False):
    # Set fixed seed for reproducibility
    np.random.seed(1234)
    # Create a graph and add random weights
    graph = nx.fast_gnp_random_graph(n=20, p=0.4, seed=np.random)
    # Change nodes to four Dutch cities
    mapping = {3: city_list[0], 7: city_list[1], 4: city_list[2], 19: city_list[3]}
    graph = nx.relabel_nodes(graph, mapping)
    for (u, v) in graph.edges():
        if u in city_list and v in city_list:
            # Set weight from city to city directly to inf
            graph.edges[u, v]['weight'] = np.inf
        else:
            graph.edges[u, v]['weight'] = np.random.randint(1, 10)
    if draw:
        color_map = ['blue'] * len(graph.nodes)
        for idx, node in enumerate(graph.nodes):
            if type(node) == str:
                color_map[idx] = 'olive'
        plt.subplots()
        nx.draw(graph, with_labels=True, font_weight='bold',
                node_color=color_map)
        plt.show()

    return graph


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
                var = prob.variables.add(obj=[cost], lb=[0.0], ub=[1.0], types=[prob.variables.type.binary],
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
    graph = create_graph(draw=False)
    print('Shortest Dijkstra paths:')
    tot_cost = 0
    for pair in unique_pairs:
        shortest_path = nx.shortest_path(graph, source=pair[0], target=pair[1], weight='weight')
        cost, _ = compute_cost(graph, shortest_path)
        tot_cost += cost
        print(shortest_path, 'with cost', cost)
    print('Sum of shortest path costs:', tot_cost)
    solve_cplex(graph=graph)




