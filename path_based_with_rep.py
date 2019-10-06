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


def solve_cplex(graph, color_map, draw):
    # Create new CPLEX model
    prob = cplex.Cplex()
    #prob.set_log_stream(None)
    #prob.set_error_stream(None)
    #prob.set_warning_stream(None)
    #prob.set_results_stream(None)
    # Set objective to minimization
    prob.objective.set_sense(prob.objective.sense.minimize)
    add_constraints(prob)
    # Add variables column wise
    add_variables(prob=prob, graph=graph)
    # Solve for the optimal solution and print
    #prob.write('test_after_colgen.lp')
    print('Total number of variables:', prob.variables.get_num())
    prob.solve()
    print()
    for i, x in enumerate(prob.solution.get_values()):
        if x > 1e-5:
            var_name = prob.variables.get_names(i)
            var_ind = prob.variables.get_indices(var_name)
            path_tuple = varmap.get(var_ind)
            if 'y' in var_name:
                rep_node_chosen = int(var_name[2::])
                print('Selected repeater node:', rep_node_chosen)
            else:
                print('Chosen path', x, 'x:', path_tuple[0], 'with cost', path_tuple[1])

    print('Optimal objective value:', prob.solution.get_objective_value())

    if draw:
        # Draw graph with repeater placement
        plt.subplot(211)
        np.random.set_state(numpy_seed)
        nx.draw(graph, with_labels=True, font_weight='bold', node_color=color_map)
        #color_map[list(mapping.keys())[list(mapping.values()).index(rep_node_chosen)]] = 'pink'
        plt.subplot(212)
        np.random.set_state(numpy_seed)
        nx.draw(graph, with_labels=True, font_weight='bold', node_color=color_map, seed=123)
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
    prob.linear_constraints.add(rhs=[1] * num_pairs, senses=['G'] * num_pairs, names=pair_con_names)
    # Save formulation to .lp file
    #prob.write('test.lp')


def add_variables(prob, graph):
    for idx, pair in enumerate(unique_pairs):
        # Generate all paths with a maximum length of 4
        paths = get_all_paths(graph, source=pair[0], target=pair[1], cutoff=5)
        for path_tuple in paths:
            path = path_tuple[0]
            cost = path_tuple[1]
            # Add path without repeaters
            column = [cplex.SparsePair(ind=['PairCon_' + ''.join(pair)], val=[1.0])]
            cplex_var = prob.variables.add(obj=[cost], lb=[0.0], ub=[1.0], types=['B'],
                               columns=column)
            varmap[cplex_var[0]] = path_tuple
            # Now iterate and generate paths with repeater
            for node in path[1:-1]:
                column_contributions = [cplex.SparsePair(ind=['LinkCon_'+str(node), 'PairCon_' + ''.join(pair)],
                                                         val=[1.0, 1.0])]
                cplex_var = prob.variables.add(obj=[cost - 1], lb=[0.0], ub=[1.0], types=['B'],
                                   columns=column_contributions)
                varmap[cplex_var[0]] = (path, cost - 1)


def get_all_paths(graph, source, target, cutoff):
    '''Adaptation of networkx code for finding the shortest path.
    Additional functionality:
     - Returns the cost of a path;
     - Avoids visiting cities intermediately.
    '''
    paths = []
    if cutoff < 1:
        return
    # Set the status to visited of all cities except the target by making a shallow copy
    visited = [source]
    stack = [iter(graph[source])]
    cost = 0
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            if len(visited) > 1:
                cost -= graph[visited[-2]][visited[-1]]['weight']
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child == target and len(visited) > 1:
                paths.append((visited + [target],
                              cost + graph[visited[-1]][target]['weight']))
            elif child not in visited and type(child) != str:
                cost += graph[visited[-1]][child]['weight']
                visited.append(child)
                stack.append(iter(graph[child]))
        else:  # len(visited) == cutoff:
            if child == target or target in children:
                paths.append((visited + [target],
                              cost + graph[visited[-1]][target]['weight']))
            cost -= graph[visited[-2]][visited[-1]]['weight']
            stack.pop()
            visited.pop()

    return paths


if __name__ == "__main__":
    np.random.seed(123)
    graph, color_map = create_graph(draw=False)
    solve_cplex(graph=graph, color_map=color_map, draw=False)
