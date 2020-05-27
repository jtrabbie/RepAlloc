import cplex
import networkx as nx
import numpy as np
import time
import datetime
from solution import Solution


class Formulation:
    """
    Base class for a integer linear program (ILP) formulation for solving repeater allocation.
    This is subclassed to construct both the link- and path-based formulations.

    Parameters
    ----------
    graph_container : GraphContainer object
        Graph container that contains the graph and some convenient pre-computed properties.
    N_max : int
        Maximum number of allowed repeaters on a path per source-destination pair. Assumed to be equal for all pairs.
        Its value is upper bounded by the total number of possible repeater node locations in the graph.
    L_max : float
        Maximum elementary link length per source-destination pair. Assumed to be equal for all pairs.
    D : int
        Capacity parameter, which denotes the number of quantum-communication sessions that one quantum repeater
        can facilitate simultaneously. Assumed to be equal for all repeater nodes. Its value is upper bounded by the
        number of unique source-destination pairs.
    K : int
        Required robustness-parameter, which denotes the number of quantum-repeater nodes and elementary links that
        must be incapacitated before network operation is compromised. Assumed to be equal for all pairs. Its value is
        upper bounded by the total number of repeaters plus one.
    alpha : float , optional
        Parameter with which the combined elementary link costs should be scaled. If this is set to 0, only the total
        number of repeaters is minimized. For any positive value of alpha, setting it too low can cause errors in the
        numerical precision of the solver, while a too high value can cause the optimal repeater placement to be
        affected (when the second term in the objective value exceeds 1).
    read_from_file : bool, optional
        Whether the formulation should be constructed from scratch or read from a file. Can be used if constructing
        the program takes a long time and one wants to generate results on this same graph (e.g. the Colt data set).
    """

    def __init__(self, graph_container, N_max: int, L_max: float, D: int, K: int, alpha=0., read_from_file=False):
        self.graph_container = graph_container
        if N_max < 1:
            raise ValueError("N_max must be a non-negative integer.")
        elif N_max > self.graph_container.num_repeater_nodes:
            print("Value of N_max exceeds the total number of repeaters {}. Manually set to {}.".format(
                self.graph_container.num_repeater_nodes, self.graph_container.num_repeater_nodes))
            N_max = self.graph_container.num_repeater_nodes
        self.N_max = N_max
        self._check_if_feasible(L_max)
        self.L_max = L_max
        if D < 0:
            raise ValueError("D must be a positive integer.")
        elif D > self.graph_container.num_unique_pairs:
            print("Value of D exceeds the total number of source-destination pairs {}. Manually set to {}".format(
                self.graph_container.num_unique_pairs, self.graph_container.num_unique_pairs))
            D = self.graph_container.num_unique_pairs
        self.D = D
        if K < 1 or K > self.graph_container.num_repeater_nodes + 1:
            raise ValueError("K must be a positive integer that cannot exceed the total number of repeaters plus one.")
        self.K = K
        if alpha < 0:
            raise ValueError("alpha must be a non-negative float")
        self.alpha = alpha
        self.read_from_file = read_from_file
        # Variable map for linking an abstract CPLEX variable to an actual path or elementary link
        self.varmap = {}
        # Create new CPLEX problem and set the mip tolerance
        self.cplex = cplex.Cplex()
        # Default value is 1e-4, but in `graph_tools._compute_dist_cartesian' the costs are rounded to 5 decimals, so
        # set tolerance to 1e-6.
        self.cplex.parameters.mip.tolerances.mipgap.set(1e-6)
        # Suppress output of CPLEX (comment to receive output statistics)
        self.cplex.set_log_stream(None)
        # self.prob.set_error_stream(None)
        # self.prob.set_warning_stream(None)
        self.cplex.set_results_stream(None)
        if read_from_file:
            # Read from file (use this when ILP takes very long to construct)
            self.cplex.read("colt_with_QIA_cities.lp")
        else:
            # Set objective sense to minimization
            self.cplex.objective.set_sense(self.cplex.objective.sense.minimize)
            # Time and add constraints and variables
            start_time = time.time()
            self._add_constraints()
            self._add_variables()
            comp_time = datetime.timedelta(seconds=time.time() - start_time)
            # print("Constructing program takes: {} s".format(comp_time))
            # print('Total number of variables: {} (expected at most {} variables)'
            #       .format(self.prob.variables.get_num(), self._compute_expected_number_of_variables()))
            # # Write linear program to text file for debugging purposes
            # self.prob.write("test_form.lp")

    def _check_if_feasible(self, L_max):
        """Check whether a feasible solution can exist with the provided value of L_max."""
        if L_max < 0:
            raise ValueError("L_max must be a positive float.")
        for end_node in self.graph_container.end_nodes:
            for edge in self.graph_container.graph.edges(end_node):
                edge_length = self.graph_container.graph[edge[0]][edge[1]]['length']
                if edge_length <= L_max:
                    # There is at least one edge that can be used to 'leave' this node
                    break
            else:
                raise ValueError("No feasible solution exists! There are no edges leaving {} with length smaller than "
                                 "or equal to {}".format(end_node, L_max))

    def _compute_expected_number_of_variables(self) -> int:
        """Compute the expected number of variables for L_max -> infty and N_max -> |R| + 1."""
        pass

    def _add_constraints(self):
        """Base attribute for adding constraints to the formulation. Should be overwritten."""
        pass

    def _add_variables(self):
        """Base attribute for adding variables to the formulations. Should be overwritten."""
        pass

    def solve(self):
        """Solve the formulation and return the Solution object as well as the computation time."""
        starttime = self.cplex.get_time()
        self.cplex.solve()
        comp_time = self.cplex.get_time() - starttime
        sol = Solution(self)
        return sol, comp_time

    def clear(self):
        """Clear the reference to the CPLEX object to free up memory when creating multiple formulations."""
        self.cplex.end()
        self.varmap = {}


class LinkBasedFormulation(Formulation):
    """Subclass for the link-based formulation, which can be found in TODO: add paper reference."""
    def __init__(self, graph_container, N_max, L_max, K, D, alpha, read_from_file=False):
        super().__init__(graph_container=graph_container, N_max=N_max, L_max=L_max, D=D, K=K, alpha=alpha,
                         read_from_file=read_from_file)

    def _compute_expected_number_of_variables(self):
        num_vars = (self.graph_container.num_repeater_nodes * (self.graph_container.num_repeater_nodes + 1) + 1) \
                   * len(self.graph_container.unique_end_node_pairs) * self.K + self.graph_container.num_repeater_nodes
        return int(num_vars)

    def _add_constraints(self):
        """Add all the constraints of the link-based formulation. Note that the constraint that uses L_max is
        incorporated in `self._add_variables`. TODO: add paper reference to constraint."""
        # Use some local references for shorter notation
        prob = self.cplex
        rep_nodes = self.graph_container.possible_rep_nodes
        num_repeater_nodes = self.graph_container.num_repeater_nodes
        # Constraints for linking the x and y variables
        link_xy_con_names = ['LinkXYCon_' + s for s in rep_nodes]
        prob.linear_constraints.add(rhs=[0] * num_repeater_nodes, senses=['L'] * num_repeater_nodes,
                                    names=link_xy_con_names)
        # Add constraints per unique pair and for every value of K
        for q in self.graph_container.unique_end_node_pairs:
            pairname = "(" + q[0] + "," + q[1] + ")"
            # Constraint that enforces that the path from s to t can be used at most once
            prob.linear_constraints.add(rhs=[1], senses=['L'], names=['STCon' + pairname])
            # Constraint for generating K node-disjoint paths (note that this has no effect for K = 1)
            disjoint_con_names = []
            for u in rep_nodes + [q[0]]:
                disjoint_con_names.append('DisLinkCon' + pairname + '_' + u)
            num_cons = len(disjoint_con_names)
            prob.linear_constraints.add(rhs=[1] * num_cons, senses=['L'] * num_cons, names=disjoint_con_names)
            for k in range(1, self.K + 1):
                # Each source should have exactly one outgoing arc
                prob.linear_constraints.add(rhs=[1], senses=['E'], names=['SourceCon' + pairname + '#' + str(k)])
                flow_cons_names = ['FlowCon' + pairname + "_" + s + '#' + str(k) for s in rep_nodes]
                # Each regular node should have equal inflow and outflow
                prob.linear_constraints.add(rhs=[0] * num_repeater_nodes, senses=['E'] * num_repeater_nodes,
                                            names=flow_cons_names)
                # Each sink should have exactly one ingoing arc
                prob.linear_constraints.add(rhs=[-1], senses=['E'], names=['SinkCon' + pairname + '#' + str(k)])
                # Constraint for maximum number of repeaters per (s,t) pair
                prob.linear_constraints.add(rhs=[self.N_max], senses=['L'], names=['MaxRepCon' + pairname + '#' +
                                                                                   str(k)])

    def _add_variables(self):
        """Generate all the variables of the link-based formulation, add them to the correct corresponding constraints
        and also to the objective function if alpha is greater than zero."""
        # Use some local references for shorter notation
        graph = self.graph_container.graph
        rep_nodes = self.graph_container.possible_rep_nodes
        # Add y_i variables with a column only in the x-y linking constraints and the objective function. Note that
        # we actually implement sum_{q in Q} sum_{v: (u, v) in E_q} sum_{K = 1}^K x_{uv}^{q,K} - D y_u <= 0 since
        # all decision variables must be on the left-hand side for CPLEX.
        var_names = ['y_' + i for i in rep_nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkXYCon_' + i], val=[-self.D])]) for i in rep_nodes]
        self.cplex.variables.add(obj=[1.0] * len(rep_nodes), names=var_names, ub=[1.0] * len(rep_nodes),
                                 types=['B'] * len(rep_nodes), columns=link_constr_column)
        # Start with finding shortest paths once and store them in a dictionary for later use
        shortest_path_dict = {}
        for i in rep_nodes:
            for j in rep_nodes:
                if i != j:
                    # Use NetworkX to generate the shortest path with Dijkstra's algorithm
                    (path_cost, sp) = nx.single_source_dijkstra(G=graph, source=i, target=j, weight='length')
                    # Store the path cost and the shortest path itself as a tuple in the dictionary
                    shortest_path_dict[(i, j)] = (path_cost, sp)
        for q in self.graph_container.unique_end_node_pairs:
            pairname = "(" + q[0] + "," + q[1] + ")"
            for i in rep_nodes + [q[0]]:
                for j in rep_nodes + [q[1]]:
                    if not i == j:
                        # Skip paths where source and sink are equal or paths that start (end) at the sink (source)
                        # And also skip paths that start or end at a city not in the currently considered pair
                        if i in rep_nodes and j in rep_nodes:
                            # We have already pre-computed this path
                            (path_cost, sp) = shortest_path_dict[(i, j)]
                        else:
                            # Find shortest (s,t), (s,j) or (i,t) path
                            (path_cost, sp) = nx.single_source_dijkstra(G=graph, source=i, target=j, weight='length')
                        # Exclude elementary links of which the length exceeds L_max, which replaces the L_max
                        # constraint of the formulation
                        if path_cost <= self.L_max:
                            for k in range(1, self.K + 1):
                                # Select correct constraints for this elementary links variable
                                if i == q[0]:  # Node i is the source
                                    if j == q[1]:  # Node j is the sink
                                        column = [cplex.SparsePair(ind=['SourceCon' + pairname + '#' + str(k),
                                                                        'SinkCon' + pairname + '#' + str(k),
                                                                        'STCon' + pairname],
                                                                   val=[1.0, -1.0, 1.0])]
                                    else:  # Node j is a possible repeater node
                                        column = [cplex.SparsePair(ind=['SourceCon' + pairname + '#' + str(k),
                                                                        'FlowCon' + pairname + "_" + str(j) + '#' + str(k),
                                                                        'MaxRepCon' + pairname + '#' + str(k),
                                                                        'DisLinkCon' + pairname + '_' + j,
                                                                        'LinkXYCon_' + j],
                                                                   val=[1.0, -1.0, 1.0, 1.0, 1.0])]
                                else:  # Node i is a possible repeater node (note that i cannot be the sink)
                                    if j == q[1]:  # Node j is the sink
                                        column = [cplex.SparsePair(ind=['SinkCon' + pairname + '#' + str(k),
                                                                        'FlowCon' + pairname + "_" + str(i) + '#' + str(k)],
                                                                   val=[-1.0, 1.0])]
                                    else:  # Node j is also a possible repeater node (note that j cannot be the source)
                                        column = [cplex.SparsePair(ind=['FlowCon' + pairname + "_" + str(i) + '#' + str(k),
                                                                        'FlowCon' + pairname + "_" + str(j) + '#' + str(k),
                                                                        'MaxRepCon' + pairname + '#' + str(k),
                                                                        'DisLinkCon' + pairname + '_' + j,
                                                                        'LinkXYCon_' + j],
                                                                   val=[1.0, -1.0, 1.0, 1.0, 1.0])]
                                # Add x_{ij}^{q,K} variables
                                cplex_var = self.cplex.variables.add(obj=[self.alpha * path_cost], ub=[1],
                                                                     columns=column, types=['B'],
                                                                     names=["x" + pairname + "_" + str(i) + "," + str(j)
                                                                           + '#' + str(k)])
                                # Add it to our variable map for future reference
                                self.varmap[cplex_var[0]] = (q, sp, path_cost)


class PathBasedFormulation(Formulation):
    """Subclass for the path-based formulation, which can be found in TODO: add paper reference."""
    def __init__(self, graph_container, L_max, N_max, K, D, alpha, read_from_file=False):
        super().__init__(graph_container=graph_container, L_max=L_max, N_max=N_max, K=K,
                         D=D, alpha=alpha, read_from_file=read_from_file, )

    def _compute_expected_number_of_variables(self):
        num_vars_per_pair = 1
        for r in range(1, self.N_max + 1):
            num_vars_per_pair += np.math.factorial(self.graph_container.num_repeater_nodes) / \
                                 np.math.factorial(self.graph_container.num_repeater_nodes - r)
        num_vars = self.graph_container.num_unique_pairs * num_vars_per_pair + self.graph_container.num_repeater_nodes
        return int(num_vars)

    def _add_constraints(self):
        """Add the constraints of the path-based formulation. Note that the constraints that use L_max and N_max are
        applied while adding the variables, since this requires less decision variables in total."""
        # For short reference
        num_repeater_nodes = self.graph_container.num_repeater_nodes
        # Constraints for connecting each pair exactly K times. Note that this differs from the formulation in the paper
        # because it is easier to implement this compared to defining all the sets P_q for all q in Q.
        pair_con_names = ['PairCon' + "(" + q[0] + "," + q[1] + ")" for q in self.graph_container.unique_end_node_pairs]
        self.cplex.linear_constraints.add(rhs=[float(self.K)] * self.graph_container.num_unique_pairs,
                                          senses=['E'] * self.graph_container.num_unique_pairs, names=pair_con_names)
        # Constraints for linking path variables to repeater variables
        link_con_names = ['LinkCon_' + s for s in self.graph_container.possible_rep_nodes]
        self.cplex.linear_constraints.add(rhs=[0] * num_repeater_nodes,
                                          senses=['L'] * num_repeater_nodes, names=link_con_names)
        # Constraints for disjoint elementary link paths
        disjoint_con_names = []
        for q in self.graph_container.unique_end_node_pairs:
            pairname = "(" + q[0] + "," + q[1] + ")"
            for u in self.graph_container.possible_rep_nodes + [q[0]]:
                disjoint_con_names.append('NodeDisjointCon' + pairname + '_' + u)
        self.cplex.linear_constraints.add(rhs=[1] * len(disjoint_con_names), senses=['L'] * len(disjoint_con_names),
                                          names=disjoint_con_names)
        # Add repeater variables with a column in the linking constraint. Note that we actually implement
        # sum_{p in P} r_up x_p - D y_u <= 0 since all decision variables must be on the left-hand side for CPLEX.
        var_names = ['y_' + s for s in self.graph_container.possible_rep_nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + i],
                                                     val=[-self.D])]) for i in self.graph_container.possible_rep_nodes]
        # Note that these variables have a lower bound of 0 by default
        self.cplex.variables.add(obj=[1] * num_repeater_nodes, names=var_names, ub=[1] * num_repeater_nodes,
                                 types=['B'] * num_repeater_nodes, columns=link_constr_column)

    def _add_variables(self):
        """Generate all possible feasible paths that adhere to the L_max and N_max constraints and link them to the
        corresponding constraints and possibly the objective function if alpha is greater than zero."""
        for q in self.graph_container.unique_end_node_pairs:
            pairname = "(" + q[0] + "," + q[1] + ")"
            all_paths = []
            # By construction a path starts at the source s
            self._generate_paths(path=[q[0]], sink=q[1], r_up=[], w_p=0, all_paths=all_paths)
            for tup in all_paths:
                # Now generate a variable for each path
                full_path = tup[0]
                r_up = tup[1]
                full_path_cost = tup[2]
                # Note that these variables have a lower bound of 0 by default
                indices = ['PairCon' + pairname] + ['LinkCon_' + i for i in r_up] + \
                          ['NodeDisjointCon' + pairname + '_' + i for i in r_up]
                column_contributions = [cplex.SparsePair(ind=indices, val=[1.0] * len(indices))]
                cplex_var = self.cplex.variables.add(obj=[self.alpha * full_path_cost], ub=[1.0], types=['B'],
                                                     columns=column_contributions)
                # Add it to our variable map for future reference
                self.varmap[cplex_var[0]] = (q, full_path, r_up, full_path_cost)

    def _generate_paths(self, path, sink, r_up, w_p, all_paths):
        """Function for recursively generating all (s, t) paths, together with the corresponding parameters r_up and
        w_p, where w_p denotes the total cost (length) of path p."""
        # Generate a path from here to the sink t
        (path_cost, sp) = nx.single_source_dijkstra(G=self.graph_container.graph, source=path[-1],
                                                    target=sink, weight='length')
        if path_cost <= self.L_max:
            all_paths.append((path + sp[1:], r_up, w_p + path_cost))
        if len(r_up) < self.N_max:
            for rep_node in self.graph_container.possible_rep_nodes:
                if rep_node not in r_up and rep_node in nx.descendants(G=self.graph_container.graph, source=path[-1]):
                    (path_cost, sp) = nx.single_source_dijkstra(G=self.graph_container.graph, source=path[-1],
                                                                target=rep_node, weight='length')
                    if path_cost <= self.L_max:
                        self._generate_paths(path=path + sp[1:], sink=sink, r_up=r_up + [rep_node], w_p=w_p + path_cost,
                                             all_paths=all_paths)
