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
# Set fixed seed for reproducibility
np.random.seed(1234)


def create_graph(draw=False):
    # Create a graph and add random weights
    graph = nx.fast_gnp_random_graph(n=num_cities + num_nodes, p=0.4, seed=np.random)
    # Change nodes to four Dutch cities
    mapping = {3: city_list[0], 4: city_list[2], 5: 3, 6: 4, 7: city_list[1], 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: city_list[3], 20: 16}
    graph = nx.relabel_nodes(graph, mapping)
    for (u, v) in graph.edges():
        graph.edges[u, v]['weight'] = np.random.randint(1, 10)
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
                node_color=color_map)
        plt.show()

    return graph, color_map, mapping


def solve_cplex(graph, color_map, mapping):
    # Create new CPLEX model
    prob = cplex.Cplex()
    # Set objective to minimization
    prob.objective.set_sense(prob.objective.sense.minimize)
    add_constraints(prob)
    # Add variables column wise
    add_variables(prob=prob, graph=graph)
    # Solve for the optimal solution and print
    prob.write('test_after_colgen.lp')
    print('Total number of variables:', prob.variables.get_num())
    prob.solve()
    for i, x in enumerate(prob.solution.get_values()):
        if x > 0.0:
            var_name = prob.variables.get_names(i)
            if 'y' in var_name:
                rep_node_chosen = int(var_name[2::])
            else:
                print('Chosen path', x, 'x:', var_name, 'with cost', prob.objective.get_linear(var_name))

    print('Optimal objective value:', prob.solution.get_objective_value())

    # Draw graph with repeater placement
    plt.subplot(211)
    np.random.set_state(numpy_seed)
    nx.draw(graph, with_labels=True, font_weight='bold',
            node_color=color_map)
    color_map[list(mapping.keys())[list(mapping.values()).index(rep_node_chosen)]] = 'pink'
    plt.subplot(212)
    np.random.set_state(numpy_seed)
    nx.draw(graph, with_labels=True, font_weight='bold',
            node_color=color_map, seed=123)
    plt.show()

def add_constraints(prob):
    # Constraints for linking path variables to repeater variables
    link_con_names = ['LinkCon_' + str(s) for s in range(num_nodes)]
    prob.linear_constraints.add(rhs=[0] * num_nodes,
                                senses=['L'] * num_nodes, names=link_con_names)
    # Add repeater variables with a column in the linking constraint
    var_names = ['y_' + str(s) for s in range(num_nodes)]
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
    for idx, pair in enumerate(unique_pairs):
        # Generate all paths with a maximum length of 4
        paths = nx.all_simple_paths(graph, source=pair[0], target=pair[1], cutoff=5)
        for path in paths:
            if sum(isinstance(node, str) for node in path) == 2 and len(path) > 2:
                # Path does not visit any cities intermediately and has length of at least 3
                cost, pathname = compute_cost(graph, path)
                # Add path without repeaters
                column = [cplex.SparsePair(ind=['PairCon_' + ''.join(pair)], val=[1.0])]
                prob.variables.add(obj=[cost], lb=[0.0], ub=[1.0], types=['B'],
                                   columns=column, names=[pathname])
                # Now iterate and generate paths with repeater
                for node in path[1:-1]:
                    rep_pathname = pathname + '___R' + str(node)
                    column_contributions = [cplex.SparsePair(ind=['LinkCon_'+str(node), 'PairCon_' + ''.join(pair)],
                                                             val=[1.0, 1.0])]
                    prob.variables.add(obj=[cost-10], lb=[0.0], ub=[1.0], types=['B'],
                                       columns=column_contributions, names=[rep_pathname])


def compute_cost(graph, path):
    cost = 0
    pathname = 'x'
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]['weight']
        pathname += '_' + str(path[i])

    pathname += '_' + str(path[-1])
    return cost, pathname


if __name__ == "__main__":
    graph, color_map, mapping = create_graph(draw=False)
    solve_cplex(graph=graph, color_map=color_map, mapping=mapping)






