import cplex
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools


class GraphContainer:
    """Class that holds all information to solve the repeater location problem with cplex."""
    def __init__(self, graph):
        self.graph = graph
        self.city_list = []
        self.nodes = []
        for node, nodedata in graph.nodes.items():
            if nodedata["type"] == 'City':
                self.city_list.append(node)
            else:
                self.nodes.append(node)
        self.num_cities = len(self.city_list)
        if self.num_cities == 0:
            raise ValueError("Must have at least one city.")
        self.unique_city_pairs = list(itertools.combinations(self.city_list, r=2))
        self.num_unique_pairs = len(list(self.unique_city_pairs))
        self.num_nodes = len(self.nodes)
        # Add length parameter to edges if this is not defined yet
        for i, j in graph.edges():
            if 'length' not in graph[i][j]:
                if 'Longitude' in graph.nodes[self.nodes[0]]:
                    self._compute_dist_lat_lon(graph)
                else:
                    self._compute_dist_cartesian(graph)
                break

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
            dist = np.round(np.sqrt(np.square(dx) + np.square(dy)))
            graph.edges[node1, node2]['length'] = dist


class Program:
    """Abstract base class for a mixed integer program for solving repeater allocation."""

    def __init__(self, graph, num_allowed_repeaters):
        self.graph_container = GraphContainer(graph=graph)
        self.R = num_allowed_repeaters
        self.M = self.graph_container.num_unique_pairs
        self.varmap = {}
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def _process_solution(self):
        return [], []

    def _draw_solution(self, repeater_nodes, chosen_paths):
        pos = {}
        labels = {}
        color_map = []
        visited_cities = []
        for tup in chosen_paths:
            visited_cities.extend(tup[1])
        visited_cities = list(set(visited_cities))
        for node, nodedata in self.graph_container.graph.nodes.items():
            if 'Longitude' in nodedata:
                pos[node] = [nodedata['Longitude'], nodedata['Latitude']]
            else:
                pos[node] = [nodedata['xcoord'], nodedata['ycoord']]
            if node in self.graph_container.city_list:
                labels[node] = node
                color_map.append('green')
            elif node in repeater_nodes:
                labels[node] = node
                color_map.append('pink')
            elif node in visited_cities:
                labels[node] = node
                color_map.append([30 / 255, 144 / 255, 255 / 255])
            else:
                labels[node] = ""
                color_map.append('none')
        plt.figure(1)
        nx.draw(self.graph_container.graph, labels=labels, with_labels=True, font_weight='bold',
                pos=pos, node_color=color_map, node_size=200)
        plt.show()

    def solve(self):
        self.prob.solve()
        repeater_nodes, chosen_paths = self._process_solution()
        self._draw_solution(repeater_nodes, chosen_paths)


class EdgeBasedProgram(Program):
    def __init__(self, graph, num_allowed_repeaters):
        super().__init__(graph=graph, num_allowed_repeaters=num_allowed_repeaters)
        self._add_constraints()
        self.prob.write("edge_based_with_con.lp")
        self._add_variables()
        print('Total number of variables: {} (expected {} variables)'
              .format(self.prob.variables.get_num(), self._compute_expected_number_of_variables()))
        self.prob.write("edge_based_with_vars.lp")

    def _compute_expected_number_of_variables(self):
        num_vars = (self.graph_container.num_nodes * (self.graph_container.num_nodes + 1) + 1) \
                   * len(self.graph_container.unique_city_pairs) + self.graph_container.num_nodes
        return int(num_vars)

    def _add_constraints(self):
        prob = self.prob
        num_nodes = self.graph_container.num_nodes
        # Add three types of constraints per unique pair
        for q in self.graph_container.unique_city_pairs:
            pairname = q[0] + "_" + q[1]
            # Each source should have exactly one outgoing arc
            prob.linear_constraints.add(rhs=[1], senses=['E'], names=['SourceCon_' + pairname])
            flow_cons_names = ['FlowCon_' + pairname + "_" + s for s in self.graph_container.nodes]
            # Each regular node should have equal inflow and outflow
            prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['E'] * num_nodes, names=flow_cons_names)
            # Each sink should have exactly one ingoing arc
            prob.linear_constraints.add(rhs=[-1], senses=['E'], names=['SinkCon_' + pairname])
            # Constraints for linking path variables to repeater variables
        link_con_names = ['LinkCon_' + s for s in self.graph_container.nodes]
        # Constraints for linking the x and y variables
        prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['L'] * num_nodes, names=link_con_names)
        # Add repeater variables with a column in the linking constraint
        var_names = ['y_' + s for s in self.graph_container.nodes]
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + i],
                                                     val=[-self.M])]) for i in self.graph_container.nodes]
        # Note that these variables have a lower bound of 0 by default
        prob.variables.add(obj=[0] * num_nodes, names=var_names, ub=[1] * num_nodes, types=['B'] * num_nodes,
                           columns=link_constr_column)
        # Constraint for maximum number of repeaters
        constr = [cplex.SparsePair(ind=range(num_nodes), val=[1.0] * num_nodes)]
        prob.linear_constraints.add(lin_expr=constr, rhs=[self.R], names=['RepCon'], senses=['L'])

    def _add_variables(self):
        graph = self.graph_container.graph
        # Add a set of variables per customer pair
        for q in self.graph_container.unique_city_pairs:
            pairname = q[0] + "_" + q[1]
            all_nodes = list(graph.nodes())
            for i in all_nodes:
                for j in all_nodes:
                    if not (i == j or i == q[1] or j == q[0] or (i in self.graph_container.city_list and i not in q) or
                            (j in self.graph_container.city_list and j not in q)):
                        # Skip paths where source and sink are equal or paths that start (end) at the sink (source)
                        # And also skip paths that start or end at a city not in the currently considered pair
                        (path_cost, sp) = nx.single_source_dijkstra(G=graph, source=i, target=j, weight='length')
                        # Subtract 1 from the path cost if uses a repeater
                        if j not in self.graph_container.city_list:
                            path_cost -= 1
                            #reduc = 1
                            #path_cost -= reduc if path_cost >= reduc else 0
                            if path_cost < 0:
                                raise ValueError("Negative path cost of {} from {} to {}".format(path_cost, i, j))
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
                        # Note that these variables have a lower bound of 0 by default
                        cplex_var = self.prob.variables.add(obj=[path_cost], columns=column,
                                                            names=["x_" + pairname + "_" + str(i) + "," + str(j)])
                        self.varmap[cplex_var[0]] = (q, sp, path_cost)

    def _process_solution(self):
        chosen_paths = []
        repeater_nodes = []
        for i, x in enumerate(self.prob.solution.get_values()):
            if x > 1e-5:
                var_name = self.prob.variables.get_names(i)
                var_ind = self.prob.variables.get_indices(var_name)
                if "y" in var_name:
                    repeater_nodes.append(var_name[2:])
                else:
                    path_tuple = self.varmap[var_ind]
                    chosen_paths.append(path_tuple)
        if len(repeater_nodes) > 0:
            print("Repeater(s) chosen: {}".format(repeater_nodes))

        for q in self.graph_container.unique_city_pairs:
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

        print('Optimal objective value:', self.prob.solution.get_objective_value())
        return repeater_nodes, chosen_paths


class PathBasedProgram(Program):
    def __init__(self, graph, num_allowed_repeaters):
        super().__init__(graph=graph, num_allowed_repeaters=num_allowed_repeaters)
        self._add_constraints()
        self.prob.write("path_based_with_con.lp")
        self._add_variables()
        print('Total number of variables: {} (expected {} variables)'
              .format(self.prob.variables.get_num(), self._compute_expected_number_of_variables()))
        self.prob.write("path_based_with_vars.lp")

    def _compute_expected_number_of_variables(self):
        num_vars_per_pair = 1
        for r in range(1, self.R + 1):
            num_vars_per_pair += np.math.factorial(self.graph_container.num_nodes) / \
                                 np.math.factorial(self.graph_container.num_nodes - r)
        num_vars = self.graph_container.num_unique_pairs * num_vars_per_pair + self.graph_container.num_nodes
        return int(num_vars)

    def _add_constraints(self):
        num_nodes = self.graph_container.num_nodes
        num_pairs = self.graph_container.num_unique_pairs
        unique_pairs = self.graph_container.unique_city_pairs
        prob = self.prob
        # Constraints for linking path variables to repeater variables
        link_con_names = ['LinkCon_' + s for s in self.graph_container.nodes]
        prob.linear_constraints.add(rhs=[0] * num_nodes,
                                    senses=['L'] * num_nodes, names=link_con_names)
        # Add repeater variables with a column in the linking constraint
        var_names = ['y_' + s for s in self.graph_container.nodes]
        link_constr_column = []
        for i in range(num_nodes):
            link_constr_column.append([[i], [-self.M]])
        prob.variables.add(obj=[0] * num_nodes, lb=[0] * num_nodes, ub=[1] * num_nodes, names=var_names,
                           types=['B'] * num_nodes, columns=link_constr_column)
        # Constraint for maximum number of repeaters
        constr = [cplex.SparsePair(ind=range(num_nodes), val=[1.0] * num_nodes)]
        prob.linear_constraints.add(lin_expr=constr, rhs=[self.R], names=['RepCon'], senses=['L'])
        # Constraints for connecting each pair exactly once
        pair_con_names = ['PairCon_' + ''.join(pair) for pair in unique_pairs]
        prob.linear_constraints.add(rhs=[1] * num_pairs, senses=['G'] * num_pairs, names=pair_con_names)

    def _add_variables(self):
        graph = self.graph_container.graph
        unique_pairs = self.graph_container.unique_city_pairs
        city_list = self.graph_container.city_list
        num_nodes = self.graph_container.num_nodes
        for q in unique_pairs:
            desc = nx.descendants(graph, q[0])
            if q[1] not in desc:
                raise RuntimeError("Destination {} not in descendants of source {}".format(q[1], q[0]))
            for city in city_list:
                try:
                    desc.remove(city)
                except KeyError:
                    if city == q[0]:
                        pass
                    else:
                        raise RuntimeError("Tried to remove {} which is not in descendants {}".format(city, desc))
            if len(desc) != num_nodes:
                raise RuntimeError("Not all cities are reachable from source {} with descendants {}".format(q[0], desc))
            # Generate all variables without repeaters
            (path_cost, sp) = nx.single_source_dijkstra(G=graph, source=q[0], target=q[1], weight='length')
            # Note that these variables have a lower bound of 0 by default
            column = [cplex.SparsePair(ind=['PairCon_' + ''.join(q)], val=[1.0])]
            cplex_var = self.prob.variables.add(obj=[path_cost], lb=[0.0], ub=[1.0], types=['B'],
                                                columns=column)
            self.varmap[cplex_var[0]] = (q, sp, path_cost)
            # Now generate variables for every possible combination of repeaters
            for num_repeaters in range(1, self.R + 1):
                all_repeater_combinations = itertools.permutations(desc, num_repeaters)
                for combination in all_repeater_combinations:
                    # Create initial path from source to first repeater
                    (cost, path) = nx.single_source_dijkstra(G=graph, source=q[0], target=combination[0],
                                                             weight='length')
                    # Add cost and path for each intermediate repeater
                    for idx in range(len(combination[1:])):
                        (intm_cost, intm_path) = nx.single_source_dijkstra(G=graph, source=combination[idx],
                                                                           target=combination[idx + 1],
                                                                           weight='length')
                        cost += intm_cost
                        path.extend(intm_path[1:])
                    # End path with path from last repeater to target
                    (final_cost, final_subpath) = nx.single_source_dijkstra(G=graph, source=combination[-1],
                                                                            target=q[1], weight='length')
                    cost += final_cost
                    path.extend(final_subpath[1:])
                    # print(combination, q, path)
                    indices = ['LinkCon_' + c for c in combination] + ['PairCon_' + ''.join(q)]
                    column_contributions = [cplex.SparsePair(ind=indices, val=[1.0] * len(indices))]
                    cplex_var = self.prob.variables.add(obj=[cost - num_repeaters], lb=[0.0], ub=[1.0], types=['B'],
                                                        columns=column_contributions)
                    self.varmap[cplex_var[0]] = (q, path, cost - num_repeaters)

    def _process_solution(self):
        chosen_paths = []
        repeater_nodes = []
        for i, x in enumerate(self.prob.solution.get_values()):
            if x > 1e-5:
                var_name = self.prob.variables.get_names(i)
                var_ind = self.prob.variables.get_indices(var_name)
                if "y" in var_name:
                    repeater_nodes.append(var_name[2:])
                else:
                    path_tuple = self.varmap[var_ind]
                    chosen_paths.append(path_tuple)
        if len(repeater_nodes) > 0:
            print("Repeater(s) chosen: {}".format(repeater_nodes))

        for q in self.graph_container.unique_city_pairs:
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

        print('Optimal objective value:', self.prob.solution.get_objective_value())
        return repeater_nodes, chosen_paths
