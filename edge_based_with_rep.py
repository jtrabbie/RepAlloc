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

    return graph


def compute_dist(i, j):
    # Compute the Pythagorean distance from one node to another
    dx = np.abs(all_pos[i][0] - all_pos[j][0])
    dy = np.abs(all_pos[i][1] - all_pos[j][1])
    return np.round(np.sqrt(np.square(dx) + np.square(dy)))


def solve_cplex(graph):
    # Create new CPLEX model
    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    # Set objective to minimization
    prob.objective.set_sense(prob.objective.sense.minimize)
    add_constraints(prob)
    #prob.write('edge_form.lp')
    # Add variables column wise
    add_variables(prob=prob, graph=graph)
    # Solve for the optimal solution and print
    #prob.write('edge_form_with_vars.lp')
    print('Total number of variables:', prob.variables.get_num())
    prob.solve()
    print()
    chosen_paths = []
    repeater_nodes = []
    for i, x in enumerate(prob.solution.get_values()):
        if x > 1e-5:
            var_name = prob.variables.get_names(i)
            var_ind = prob.variables.get_indices(var_name)
            if "y" in var_name:
                repeater_nodes.append(int(var_name[2:]))
            else:
                path_tuple = varmap[var_ind]
                chosen_paths.append(path_tuple)
    if len(repeater_nodes) > 0:
        print("Repeater(s) chosen: {}".format(repeater_nodes))

    tot_cost = 0
    for q in unique_pairs:
        string = "Path from {} to {}: ".format(q[0], q[1])
        edges_current_pair = []
        [edges_current_pair.append(tup) for tup in chosen_paths if tup[0] == q]
        path = []
        cost = 0
        num_edges = 0
        while num_edges < len(edges_current_pair):
            for edge in edges_current_pair:
                if len(path) == 0 and edge[1][0] == q[0]:  # Initial edge from source
                    path.extend(edge[1])
                    cost += edge[2]
                    num_edges += 1
                elif len(path) > 0 and edge[1][0] == path[-1]:  # Extension of current path
                    path = path[0:-1]
                    path.extend(edge[1])
                    cost += edge[2]
                    num_edges += 1
        print("Optimal path for pair {}: {}, with cost {}".format(q, path, cost))

    print('Optimal objective value:', prob.solution.get_objective_value())
    return prob.solution.get_objective_value()


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
        # Constraints for linking path variables to repeater variables
    link_con_names = ['LinkCon_' + str(s) for s in range(num_nodes)]
    # Constraints for linking the x and y variables
    prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['L'] * num_nodes, names=link_con_names)
    # Add repeater variables with a column in the linking constraint
    var_names = ['y_' + str(s) for s in range(num_nodes)]
    link_constr_column = []
    [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + str(i)], val=[-M])]) for i in range(num_nodes)]
    prob.variables.add(obj=[0] * num_nodes, names=var_names, lb=[0] * num_nodes, ub=[1] * num_nodes,
                       types=['B'] * num_nodes, columns=link_constr_column)
    # Constraint for maximum number of repeaters
    constr = [cplex.SparsePair(ind=range(num_nodes), val=[1.0] * num_nodes)]
    prob.linear_constraints.add(lin_expr=constr, rhs=[R], names=['RepCon'], senses=['L'])


def add_variables(prob, graph):
    # Add a set of variables per customer pair
    for q in unique_pairs:
        pairname = q[0] + "_" + q[1]
        for i in graph.nodes():
            for j in graph.nodes():
                if i == j or i == q[1] or j == q[0] or \
                        (type(i) != int and i not in q) or (type(j) != int and j not in q):
                    # Skip paths where source and sink are equal or paths that start (end) at the sink (source)
                    # And also skip paths that start or end at a city not in the currently considered pair
                    pass
                else:
                    sp = nx.shortest_path(graph, source=i, target=j, weight='weight')
                    path_cost, _ = compute_path_cost(graph, sp)
                    if type(j) == int:
                        path_cost -= 1
                    if i == q[0]:  # Node i is the source
                        if j == q[1]:  # Node j is the sink
                            column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                            'SinkCon_' + pairname], val=[1.0, -1.0])]
                        else:  # Node j is a regular node
                            column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                            'FlowCon_' + pairname + "_" + str(j),
                                                            'LinkCon_' + str(j)], val=[1.0, -1.0, 1.0])]
                    else:  # Node i is a regular node
                        if j == q[1]:  # Node j is the sink
                            column = [cplex.SparsePair(ind=['SinkCon_' + pairname,
                                                            'FlowCon_' + pairname + "_" + str(i)], val=[-1.0, 1.0])]
                        else:  # Node j is also a regular node
                            column = [cplex.SparsePair(ind=['FlowCon_' + pairname + "_" + str(i),
                                                            'FlowCon_' + pairname + "_" + str(j),
                                                            'LinkCon_' + str(j)], val=[1.0, -1.0, 1.0])]
                    cplex_var = prob.variables.add(obj=[path_cost], lb=[0.0], columns=column,
                                                   names=["x_" + pairname + "_" + str(i) + "," + str(j)])
                    varmap[cplex_var[0]] = (q, sp, path_cost)


def compute_path_cost(graph, path):
    cost = 0
    pathname = 'x'
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]['weight']
        pathname += '_' + str(path[i])

    pathname += '_' + str(path[-1])
    return cost, pathname


if __name__ == "__main__":
    np.random.seed(123)
    graph = create_graph(draw=False)
    obj_val = solve_cplex(graph=graph)
    if R == 0:
        tot_cost = 0
        for pair in unique_pairs:
            shortest_path = nx.shortest_path(graph, source=pair[0], target=pair[1], weight='weight')
            cost, _ = compute_path_cost(graph, shortest_path)
            tot_cost += cost
            #print(shortest_path, 'with cost', cost)
        if tot_cost != obj_val:
            raise RuntimeError("Critical Error! Shortest Dijkstra path costs do not equal CPLEX cost!")




