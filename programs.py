import cplex
import networkx as nx
import numpy as np
import itertools
import time
import datetime
from solution import Solution


class Program:
    """Abstract base class for a mixed integer program for solving repeater allocation."""

    def __init__(self, graph_container, cplex_verbose=False, read_from_file=False):
        self.graph_container = graph_container
        self.M = self.graph_container.num_unique_pairs
        self.varmap = {}
        self.prob = cplex.Cplex()
        # Default value is 1e-4, but in `graph_tools._compute_dist_cartesian', the costs are rounded to 5 decimals!
        self.prob.parameters.mip.tolerances.mipgap.set(1e-6)
        self.read_from_file = read_from_file
        if not cplex_verbose:
            pass
            self.prob.set_log_stream(None)
            # self.prob.set_error_stream(None)
            # self.prob.set_warning_stream(None)
            self.prob.set_results_stream(None)
        if read_from_file:
            self.prob.read("colt_with_QIA_cities.lp")
        else:
            self.prob.objective.set_sense(self.prob.objective.sense.minimize)
            starttime = time.time()
            self._add_constraints()
            self._add_variables()
            endtime = time.time()
            # self.prob.write("test_form.lp")
            # print("Constructing program takes: {}".format(datetime.timedelta(seconds=endtime - starttime)))
            # print('Total number of variables: {} (expected at most {} variables)'
            #       .format(self.prob.variables.get_num(), self._compute_expected_number_of_variables()))

    def _add_constraints(self):
        pass

    def _add_variables(self):
        pass

    def _compute_expected_number_of_variables(self):
        pass

    def solve(self):
        starttime = self.prob.get_time()
        self.prob.solve()
        comp_time = self.prob.get_time() - starttime
        sol = Solution(self)
        return sol, comp_time


class LinkBasedProgram(Program):
    def __init__(self, graph_container, R_max=cplex.infinity, L_max=cplex.infinity, k=1.0, D=cplex.infinity, alpha=0.,
                 read_from_file=False):
        self.R_max = R_max
        if self.R_max > graph_container.num_repeater_nodes:
            self.R_max = graph_container.num_repeater_nodes
        self.L_max = L_max
        self.k = int(k)
        self.D = D
        self.alpha = alpha
        super().__init__(graph_container=graph_container, read_from_file=read_from_file)
        # self._check_for_feasibility()
        # self.prob.write("test_form.lp")

    def _check_for_feasibility(self):
        """Check whether a feasible solution can exist given the maximum length on the elementary link length."""
        for city in self.graph_container.end_nodes:
            for edge in self.graph_container.graph.edges(city):
                edge_length = self.graph_container.graph[edge[0]][edge[1]]['length']
                if edge_length <= self.L_max:
                    # There is at least one edge that can be used to 'leave' this node
                    break
            else:
                print("No feasible solution exists! There are no edges leaving {} with length smaller than "
                      "or equal to {}".format(city, self.L_max))

    def _compute_expected_number_of_variables(self):
        num_vars = (self.graph_container.num_repeater_nodes * (self.graph_container.num_repeater_nodes + 1) + 1) \
                   * len(self.graph_container.unique_end_node_pairs) + self.graph_container.num_repeater_nodes
        return int(num_vars)

    def _add_constraints(self):
        prob = self.prob
        rep_nodes = self.graph_container.possible_rep_nodes
        num_repeater_nodes = self.graph_container.num_repeater_nodes
        # Add constraints per unique pair and for every value of k
        for k in range(self.k):
            for q in self.graph_container.unique_end_node_pairs:
                pairname = "(" + q[0] + "," + q[1] + ")"
                # Each source should have exactly one outgoing arc
                prob.linear_constraints.add(rhs=[1], senses=['E'], names=['SourceCon' + pairname + '#' + str(k)])
                flow_cons_names = ['FlowCon' + pairname + "_" + s + '#' + str(k) for s in rep_nodes]
                # Each regular node should have equal inflow and outflow
                prob.linear_constraints.add(rhs=[0] * num_repeater_nodes, senses=['E'] * num_repeater_nodes,
                                            names=flow_cons_names)
                # Each sink should have exactly one ingoing arc
                prob.linear_constraints.add(rhs=[-1], senses=['E'], names=['SinkCon' + pairname + '#' + str(k)])
                # Constraint for maximum number of repeaters per (s,t) pair
                prob.linear_constraints.add(rhs=[self.R_max], senses=['L'], names=['MaxRepCon' + pairname + '#' +
                                                                                   str(k)])
                if k == 0:
                    link_x_con_names = []
                    for i in rep_nodes + [q[0]]:
                        for j in rep_nodes + [q[1]]:
                            if not i == j:
                                link_x_con_names.append('LinkXCon' + pairname + '_' + i + ',' + j)
                    num_cons = len(link_x_con_names)
                    prob.linear_constraints.add(rhs=[1] * num_cons, senses=['L'] * num_cons, names=link_x_con_names)
            # Constraints for linking y_i^k variables to y_i variables
            link_y_con_names = ['LinkYCon_' + s + '#' + str(k) for s in rep_nodes]
            prob.linear_constraints.add(rhs=[0] * num_repeater_nodes, senses=['G'] * num_repeater_nodes,
                                        names=link_y_con_names)
            # Add y^k_i variables with a column in the x-y and y-y linking constraint
            var_names = ['y_' + i + '#' + str(k) for i in rep_nodes]
            # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
            link_constr_column = []
            [link_constr_column.extend([cplex.SparsePair(ind=['LinkYCon_' + i + '#' + str(k)],
                                                         val=[-1.0])]) for i in rep_nodes]
            # Note that these variables have a lower bound of 0 by default
            prob.variables.add(obj=[0] * num_repeater_nodes, names=var_names, ub=[1] * num_repeater_nodes,
                               types=['B'] * num_repeater_nodes, columns=link_constr_column)
        # Constraints for linking the x and y variables
        link_xy_con_names = ['LinkXYCon_' + s for s in rep_nodes]
        prob.linear_constraints.add(rhs=[0] * num_repeater_nodes, senses=['L'] * num_repeater_nodes,
                                    names=link_xy_con_names)
        # Add y_i variables with a column only in the y-y linking constraints and the objective function
        var_names = ['y_' + i for i in rep_nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkYCon_' + i + '#' + str(k) for k in range(self.k)] +
                                                         ['LinkXYCon_' + i],
                                                     val=[1.0] * self.k + [-self.D])]) for i in rep_nodes]
        # Note that these variables have a lower bound of 0 by default
        prob.variables.add(obj=[1] * num_repeater_nodes, names=var_names, ub=[1] * num_repeater_nodes,
                           types=['B'] * num_repeater_nodes, columns=link_constr_column)

    def _add_variables(self):
        graph = self.graph_container.graph
        rep_nodes = self.graph_container.possible_rep_nodes
        for k in range(self.k):
            for q in self.graph_container.unique_end_node_pairs:
                pairname = "(" + q[0] + "," + q[1] + ")"
                for i in rep_nodes + [q[0]]:
                    for j in rep_nodes + [q[1]]:
                        if not i == j:
                            # Skip paths where source and sink are equal or paths that start (end) at the sink (source)
                            # And also skip paths that start or end at a city not in the currently considered pair
                            (path_cost, sp) = nx.single_source_dijkstra(G=graph, source=i, target=j, weight='length')
                            if path_cost <= self.L_max:  # Exclude elementary links that exceed L_max
                                # Select correct constraints for this elementary links variable
                                if i == q[0]:  # Node i is the source
                                    if j == q[1]:  # Node j is the sink
                                        column = [cplex.SparsePair(ind=['SourceCon' + pairname + '#' + str(k),
                                                                        'SinkCon' + pairname + '#' + str(k),
                                                                        'LinkXCon' + pairname + '_' + i + ',' + j],
                                                                   val=[1.0, -1.0, 1.0])]
                                    else:  # Node j is a possible repeater node
                                        column = [cplex.SparsePair(ind=['SourceCon' + pairname + '#' + str(k),
                                                                        'FlowCon' + pairname + "_" + str(j) + '#' + str(k),
                                                                        'MaxRepCon' + pairname + '#' + str(k),
                                                                        'LinkXCon' + pairname + '_' + i + ',' + j,
                                                                        'LinkXYCon_' + j],
                                                                   val=[1.0, -1.0, 1.0, 1.0, 1.0])]
                                else:  # Node i is a possible repeater node (note that i cannot be the sink)
                                    if j == q[1]:  # Node j is the sink
                                        column = [cplex.SparsePair(ind=['SinkCon' + pairname + '#' + str(k),
                                                                        'FlowCon' + pairname + "_" + str(i) + '#' + str(k),
                                                                        'LinkXCon' + pairname + '_' + i + ',' + j],
                                                                   val=[-1.0, 1.0, 1.0])]
                                    else:  # Node j is also a possible repeater node (note that j cannot be the source)
                                        column = [cplex.SparsePair(ind=['FlowCon' + pairname + "_" + str(i) + '#' + str(k),
                                                                        'FlowCon' + pairname + "_" + str(j) + '#' + str(k),
                                                                        'MaxRepCon' + pairname + '#' + str(k),
                                                                        'LinkXCon' + pairname + '_' + i + ',' + j,
                                                                        'LinkXYCon_' + j],
                                                                   val=[1.0, -1.0, 1.0, 1.0, 1.0])]
                                # Add x_{ij}^{q,k} variables
                                cplex_var = self.prob.variables.add(obj=[self.alpha * path_cost], ub=[1],
                                                                    columns=column, types=['B'],
                                                                    names=["x" + pairname + "_" + str(i) + "," + str(j)
                                                                           + '#' + str(k)])
                                # Add it to our variable map for future reference
                                self.varmap[cplex_var[0]] = (q, sp, path_cost)

    def update_parameters(self, R_max_new, alpha_new):
        R_max_old = self.R_max
        alpha_old = self.alpha
        if R_max_old == R_max_new and alpha_old == alpha_new:
            # Nothing has changed, no need for updating
            return
        self.R_max = R_max_new
        self.alpha = alpha_new
        rep_nodes = self.graph_container.possible_rep_nodes
        for q in self.graph_container.unique_city_pairs:
            pairname = q[0] + "_" + q[1]
            if R_max_new != R_max_old:
                # Update the RHS of the maximum number of repeaters per pair constraint
                self.prob.linear_constraints.set_rhs('MaxRepCon_' + pairname, self.R_max)
            for i in rep_nodes + [q[0]]:
                for j in rep_nodes + [q[1]]:
                    if not i == j:
                        if alpha_new != alpha_old:
                            # Get the CPLEX index of the variable for this q, i and j
                            ind = self.prob.variables.get_indices("x_" + pairname + "_" + str(i) + "," + str(j))
                            # Get path cost from the corresponding constraint
                            path_cost = self.prob.linear_constraints.get_coefficients(
                                'MaxLengthCon_' + pairname + '_' + i + '_' + j, ind)
                            self.prob.objective.set_linear(ind, self.alpha * path_cost)


class PathBasedProgram(Program):
    def __init__(self, graph_container, R_max=cplex.infinity, L_max=cplex.infinity, k=1.0, D=cplex.infinity, alpha=0.,
                 read_from_file=False):
        self.R_max = R_max
        if self.R_max > graph_container.num_repeater_nodes:
            self.R_max = graph_container.num_repeater_nodes
        self.L_max = L_max
        self.k = int(k)
        self.D = D
        self.alpha = alpha
        super().__init__(graph_container=graph_container, read_from_file=read_from_file)

    def _compute_expected_number_of_variables(self):
        num_vars_per_pair = 1
        for r in range(1, self.R_max + 1):
            num_vars_per_pair += np.math.factorial(self.graph_container.num_repeater_nodes) / \
                                 np.math.factorial(self.graph_container.num_repeater_nodes - r)
        num_vars = self.graph_container.num_unique_pairs * num_vars_per_pair + self.graph_container.num_repeater_nodes
        return int(num_vars)

    def _add_constraints(self):
        # For short reference
        num_repeater_nodes = self.graph_container.num_repeater_nodes
        # Constraints for connecting each pair exactly once
        pair_con_names = ['PairCon' + "(" + q[0] + "," + q[1] + ")" for q in self.graph_container.unique_end_node_pairs]
        self.prob.linear_constraints.add(rhs=[self.k] * self.graph_container.num_unique_pairs,
                                         senses=['E'] * self.graph_container.num_unique_pairs, names=pair_con_names)
        # Constraints for linking path variables to repeater variables
        link_con_names = ['LinkCon_' + s for s in self.graph_container.possible_rep_nodes]
        self.prob.linear_constraints.add(rhs=[0] * num_repeater_nodes,
                                         senses=['L'] * num_repeater_nodes, names=link_con_names)
        # Constraints for disjoint elementary link paths
        disjoint_con_names = []
        for q in self.graph_container.unique_end_node_pairs:
            pairname = "(" + q[0] + "," + q[1] + ")"
            for i in self.graph_container.possible_rep_nodes + [q[0]]:
                for j in self.graph_container.possible_rep_nodes + [q[1]]:
                    disjoint_con_names.append('DisjointCon' + pairname + '_' + i + ',' + j)
        self.prob.linear_constraints.add(rhs=[1] * len(disjoint_con_names), senses=['L'] * len(disjoint_con_names),
                                         names=disjoint_con_names)
        # Add repeater variables with a column in the linking constraint
        var_names = ['y_' + s for s in self.graph_container.possible_rep_nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + i],
                                                     val=[-self.D])]) for i in self.graph_container.possible_rep_nodes]
        # Note that these variables have a lower bound of 0 by default
        self.prob.variables.add(obj=[1] * num_repeater_nodes, names=var_names, ub=[1] * num_repeater_nodes,
                                types=['B'] * num_repeater_nodes, columns=link_constr_column)

    def _add_variables(self):
        # print("Possible repeater nodes:", self.graph_container.possible_rep_nodes)
        for q in self.graph_container.unique_end_node_pairs:
            pairname = "(" + q[0] + "," + q[1] + ")"
            all_paths = []
            # By construction a path starts at the source s
            self._generate_paths(path=[q[0]], sink=q[1], r_ip=[], w_p=0, b_ijpq=[], all_paths=all_paths)
            for tup in all_paths:
                # Now generate a variable for each path
                full_path = tup[0]
                # if
                r_ip = tup[1]
                full_path_cost = tup[2]
                b_ijpq = tup[3]
                # Note that these variables have a lower bound of 0 by default
                indices = ['PairCon' + pairname] + ['LinkCon_' + i for i in r_ip] + \
                          ['DisjointCon' + pairname + '_' + i + ',' + j for (i, j) in b_ijpq]
                column_contributions = [cplex.SparsePair(ind=indices, val=[1.0] * len(indices))]
                cplex_var = self.prob.variables.add(obj=[self.alpha * full_path_cost], ub=[1.0], types=['B'],
                                                    columns=column_contributions)
                # Add it to our variable map for future reference
                self.varmap[cplex_var[0]] = (q, full_path, r_ip, full_path_cost)

    def _generate_paths(self, path, sink, r_ip, w_p, b_ijpq, all_paths):
        """Function for recursively generating all (s,t) paths, together with the corresponding parameters
        w_p and r_ip."""
        # Generate a path from here to the sink t
        (path_cost, sp) = nx.single_source_dijkstra(G=self.graph_container.graph, source=path[-1],
                                                    target=sink, weight='length')
        if path_cost <= self.L_max:
            all_paths.append((path + sp[1:], r_ip, w_p + path_cost, b_ijpq + [(path[-1], sink)]))
            # print("Found path {} for q = {} with r_ip = {}".format(path + sp[1:], (path[0], sink), r_ip))
        if len(r_ip) < self.R_max:
            # print("Extending path {} with r_ip = {} , len(r_ip) = {}, R_max = {}"
            #       .format(path, r_ip, len(r_ip), self.R_max))
            for rep_node in self.graph_container.possible_rep_nodes:
                if rep_node not in r_ip and rep_node in nx.descendants(G=self.graph_container.graph, source=path[-1]):
                    (path_cost, sp) = nx.single_source_dijkstra(G=self.graph_container.graph, source=path[-1],
                                                                target=rep_node, weight='length')
                    if path_cost <= self.L_max:
                        self._generate_paths(path=path + sp[1:], sink=sink, r_ip=r_ip + [rep_node], w_p=w_p + path_cost,
                                             b_ijpq=b_ijpq + [(path[-1], rep_node)], all_paths=all_paths)
