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


def solve_cplex(graph, color_map):
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
    prob.write('test_after_colgen.lp')
    print('Total number of variables:', prob.variables.get_num())
    prob.solve()
    print()
    chosen_paths = []
    for i, x in enumerate(prob.solution.get_values()):
        if x > 1e-5:
            var_name = prob.variables.get_names(i)
            var_ind = prob.variables.get_indices(var_name)
            path_tuple = varmap[var_ind]
            chosen_paths.append(path_tuple)

    for q in unique_pairs:
        str = ["Path from {} to {}:".format(q[0], q[1])]
        for idx in range(len(chosen_paths)):
            if chosen_paths[idx][0][0] == q[0]:  # Path starts with source city
                path, cost = chosen_paths.pop(idx)
                break
        for idx in range(len(chosen_paths)):
            if chosen_paths[idx][0][0] == path[-1]:
                addpath, addcost = chosen_paths.pop(idx)
                cost += addcost
                path.remove(path[-1])
                path.extend(addpath)
                break
        str.extend(path)
        print(str)

    print('Optimal objective value:', prob.solution.get_objective_value())


def add_constraints(prob):
    # Add three types of constraints per unique pair
    for q in unique_pairs:
        pairname = q[0] + "_" + q[1]
        # Each source should have exactly one outgoing arc
        prob.linear_constraints.add(rhs=[1], senses=['E'], names=['SourceCon_' + pairname])
        flow_cons_names = ['FlowCon_' + pairname + "_" + str(s) for s in range(num_nodes)]
        # Each regular node should have equal inflow and outflow
        prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['E'] * num_nodes, names=flow_cons_names)
        # Each sink should have exactly one ingoing arc
        prob.linear_constraints.add(rhs=[-1], senses=['E'], names=['SinkCon_' + pairname])
    # prob.write('test.lp')


def add_variables(prob, graph):
    # Add a set of variables per customer pair
    for q in unique_pairs:
        pairname = q[0] + "_" + q[1]
        for i in graph.nodes():
            for j in graph.nodes():
                if (type(i) != int and i not in q) or (type(j) != int and j not in q) or i == j:
                    # Skip paths from/to cities that are not considered in this pair
                    pass
                else:
                    sp = nx.shortest_path(graph, source=i, target=j, weight='weight')
                    path_cost, _ = compute_path_cost(graph, sp)
                    if i == q[0]:  # Node i is the source
                        if j == q[1]:  # Node j is the sink
                            column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                            'SinkCon_' + pairname], val=[1.0, -1.0])]
                        else:  # Node j is a regular node
                            column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                            'FlowCon_' + pairname + "_" + str(j)], val=[1.0, -1.0])]
                    elif i == q[1]:  # Node i is the sink
                        if j == q[0]:  # Node j is the source
                            column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                            'SinkCon_' + pairname], val=[-1.0, 1.0])]
                        else:  # Node j is a regular node
                            column = [cplex.SparsePair(ind=['SinkCon_' + pairname,
                                                            'FlowCon_' + pairname + "_" + str(j)], val=[1.0, -1.0])]
                    else:  # Node i is a regular node
                        if j == q[0]:  # Node j is the source
                            column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                            'FlowCon_' + pairname + "_" + str(i)], val=[-1.0, 1.0])]
                        elif j == q[1]:  # Node j is the sink
                            column = [cplex.SparsePair(ind=['SinkCon_' + pairname,
                                                            'FlowCon_' + pairname + "_" + str(i)], val=[-1.0, 1.0])]
                        else:  # Node j is also a regular node
                            column = [cplex.SparsePair(ind=['FlowCon_' + pairname + "_" + str(i),
                                                            'FlowCon_' + pairname + "_" + str(j)], val=[1.0, -1.0])]
                    cplex_var = prob.variables.add(obj=[path_cost], lb=[0.0], columns=column,
                                                   names=["x_" + pairname + "_" + str(i) + "," + str(j)])
                    varmap[cplex_var[0]] = (sp, path_cost)


def compute_path_cost(graph, path):
    cost = 0
    pathname = 'x'
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]['weight']
        pathname += '_' + str(path[i])

    pathname += '_' + str(path[-1])
    return cost, pathname


if __name__ == "__main__":
    graph, color_map = create_graph(draw=True)
    solve_cplex(graph=graph, color_map=color_map)
    tot_cost = 0
    for pair in unique_pairs:
        shortest_path = nx.shortest_path(graph, source=pair[0], target=pair[1], weight='weight')
        cost, _ = compute_path_cost(graph, shortest_path)
        tot_cost += cost
        print(shortest_path, 'with cost', cost)
    print('Sum of shortest path costs:', tot_cost)




