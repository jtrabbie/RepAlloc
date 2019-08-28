import cplex
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

city_list = ['Lei', 'Haa', 'Ams', 'Del']  # list of cities
num_cities = len(city_list)  # number of customers (cities)
unique_pairs = list(itertools.combinations(city_list, r=2))  # list of unique pair tuples
num_pairs = len(list(unique_pairs))  # number of unique customer pairs
num_nodes = 16  # number of non-customer nodes
R = 1  # maximum number of repeaters
M = num_pairs  # dummy parameter for linking constraint


def create_graph(draw=False):
    # Set fixed seed for reproducibility
    np.random.seed(1234)
    # Create a graph and add random weights
    graph = nx.fast_gnp_random_graph(n=num_cities + num_nodes, p=0.4, seed=np.random)
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
    add_constraints(prob)
    # Add variables column wise
    add_variables(prob=prob, graph=graph)
    # Solve for the optimal solution and print
    prob.write('test_after_colgen.lp')
    prob.solve()
    for i, x in enumerate(prob.solution.get_values()):
        if x > 0.0:
            path = prob.variables.get_names(i)
            print('Chosen path ', x, 'x:', path, 'with cost', prob.objective.get_linear(path))

    print('Optimal objective value:', prob.solution.get_objective_value())


def add_constraints(prob):
    # Constraints for linking path variables to repeater variables
    link_con_names = ['LinkCon_' + str(s) for s in range(1, num_nodes + 1)]
    prob.linear_constraints.add(rhs=[0] * num_nodes,
                                senses=['L'] * num_nodes, names=link_con_names)
    # Add repeater variables with a column in the linking constraint
    var_names = ['y_' + str(s) for s in range(1, num_nodes + 1)]
    link_constr_column = []
    for i in range(num_nodes):
        link_constr_column.append([[i], [-M]])
    prob.variables.add(obj=[0] * num_nodes, lb=[0] * num_nodes, ub=[1] * num_nodes, names=var_names,
                       types=['B'] * num_nodes, columns=link_constr_column)
    # Constraint for maximum number of repeaters
    constr = [cplex.SparsePair(ind=range(num_nodes), val=[1.0] * num_nodes)]
    prob.linear_constraints.add(lin_expr=constr, rhs=[R], names=['RepCon'], senses=['L'])
    # Constraints for connecting each pair exactly once
    pair_con_names = ['PairCon_' + ''.join(pair) for pair in unique_pairs]
    prob.linear_constraints.add(rhs=[1] * num_pairs, senses=['E'] * num_pairs, names=pair_con_names)
    # Save formulation to .lp file
    prob.write('test.lp')


def add_variables(prob, graph):
    variable_dict = {}
    for idx, pair in enumerate(unique_pairs):
        # Generate all paths with a maximum length of 4
        paths = nx.all_simple_paths(graph, source=pair[0], target=pair[1], cutoff=6)
        for path in paths:
            cost, pathname = compute_cost(graph, path)
            if cost != np.inf:  # only add variables with finite cost
                column = [cplex.SparsePair(ind=[num_nodes + 1 + idx], val=[1.0])]
                var = prob.variables.add(obj=[cost], lb=[0.0], ub=[1.0], types=['C'],
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




