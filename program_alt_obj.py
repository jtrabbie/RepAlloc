from programs import Program
import cplex
import networkx as nx
import numpy as np
import itertools


class MinMaxEdgeBasedProgram(Program):
    def __init__(self, graph, num_allowed_repeaters):
        super().__init__(graph=graph, num_allowed_repeaters=num_allowed_repeaters)
        self.prob.write("edge_based_alt_obj_form.lp")

    def _compute_expected_number_of_variables(self):
        num_vars = (self.graph_container.num_nodes * (self.graph_container.num_nodes + 1) + 1) \
                   * len(self.graph_container.unique_city_pairs) + self.graph_container.num_nodes \
                   + 1
        return int(num_vars)

    def _add_constraints(self):
        prob = self.prob
        num_nodes = self.graph_container.num_nodes
        # Add three types of constraints per unique pair
        row_names = []  # Save the row names for the z variables
        for q in self.graph_container.unique_city_pairs:
            pairname = q[0] + "_" + q[1]
            # Each source should have exactly one outgoing arc
            prob.linear_constraints.add(rhs=[1], senses=['E'], names=['SourceCon_' + pairname])
            flow_cons_names = ['FlowCon_' + pairname + "_" + s for s in self.graph_container.nodes]
            # Each regular node should have equal inflow and outflow
            prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['E'] * num_nodes, names=flow_cons_names)
            # Each sink should have exactly one ingoing arc
            prob.linear_constraints.add(rhs=[-1], senses=['E'], names=['SinkCon_' + pairname])
            all_nodes = list(self.graph_container.graph.nodes())
            # Constraints for min-max objective function
            for i in all_nodes:
                for j in all_nodes:
                    if not (i == j or i == q[1] or j == q[0] or (i in self.graph_container.city_list and i not in q) or
                            (j in self.graph_container.city_list and j not in q)):
                        row_name = ['MinMaxCon_' + pairname + '_' + i + '_' + j]
                        row_names.extend(row_name)
                        prob.linear_constraints.add(rhs=[0], senses=['G'],
                                                    names=row_name)
        # Note that we need a single SparsePair for the single z variable
        min_max_columns = [cplex.SparsePair(ind=row_names, val=[1] * len(row_names))]
        # Constraints for linking path variables to repeater variables
        link_con_names = ['LinkCon_' + s for s in self.graph_container.nodes]
        # Constraints for linking the x and y variables
        prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['L'] * num_nodes, names=link_con_names)
        # Add repeater variables with a column in the linking constraint
        var_names = ['y_' + s for s in self.graph_container.nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + i],
                                                      val=[-self.M])]) for i in self.graph_container.nodes]
        # Note that these variables have a lower bound of 0 by default
        prob.variables.add(obj=[0] * num_nodes, names=var_names, ub=[1] * num_nodes, types=['B'] * num_nodes,
                           columns=link_constr_column)
        # Constraint for maximum number of repeaters
        constr = [cplex.SparsePair(ind=range(num_nodes), val=[1.0] * num_nodes)]
        prob.linear_constraints.add(lin_expr=constr, rhs=[self.R_max], names=['RepCon'], senses=['L'])
        # Variable for min-max objective function
        prob.variables.add(obj=[1], names='z', columns=min_max_columns)

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
                            reduc = 0
                            path_cost -= reduc if path_cost >= reduc else 0
                            if path_cost < 0:
                                raise ValueError("Negative path cost of {} from {} to {}".format(path_cost, i, j))
                        if i == q[0]:  # Node i is the source
                            if j == q[1]:  # Node j is the sink
                                column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                                'SinkCon_' + pairname,
                                                                'MinMaxCon_' + pairname + '_' + i + '_' + j],
                                                           val=[1.0, -1.0, -path_cost])]
                            else:  # Node j is a regular node
                                column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                                'FlowCon_' + pairname + "_" + str(j),
                                                                'LinkCon_' + str(j),
                                                                'MinMaxCon_' + pairname + '_' + i + '_' + j],
                                                           val=[1.0, -1.0, 1.0, -path_cost])]
                        else:  # Node i is a regular node
                            if j == q[1]:  # Node j is the sink
                                column = [cplex.SparsePair(ind=['SinkCon_' + pairname,
                                                                'FlowCon_' + pairname + "_" + str(i),
                                                                'MinMaxCon_' + pairname + '_' + i + '_' + j],
                                                           val=[-1.0, 1.0, -path_cost])]
                            else:  # Node j is also a regular node
                                column = [cplex.SparsePair(ind=['FlowCon_' + pairname + "_" + str(i),
                                                                'FlowCon_' + pairname + "_" + str(j),
                                                                'LinkCon_' + str(j),
                                                                'MinMaxCon_' + pairname + '_' + i + '_' + j],
                                                           val=[1.0, -1.0, 1.0, -path_cost])]
                        # Note that these variables have a lower bound of 0 by default
                        # We need to define these as binary variables with the alternative objective function
                        cplex_var = self.prob.variables.add(obj=[0], columns=column, types=['B'],
                                                            names=["x_" + pairname + "_" + str(i) + "," + str(j)])
                        self.varmap[cplex_var[0]] = (q, sp, path_cost)

    def _process_solution(self):
        chosen_paths = []
        repeater_nodes = []
        for i, x in enumerate(self.prob.solution.get_values()):
            if x > 1e-5:
                var_name = self.prob.variables.get_names(i)
                var_ind = self.prob.variables.get_indices(var_name)
                if var_name[0:2] == "y_":
                    repeater_nodes.append(var_name[2:])
                elif var_name == "z":
                    pass
                else:
                    path_tuple = self.varmap[var_ind]
                    chosen_paths.append(path_tuple)
        if len(repeater_nodes) > 0:
            print("Repeater(s) chosen: {}".format(repeater_nodes))
        total_cost = 0
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
            total_cost += cost
            print("Optimal path for pair {}: {}, with cost {}".format(q, path, cost))

        print('Optimal objective value:', self.prob.solution.get_objective_value(), "total cost of paths:", total_cost)
        return repeater_nodes, chosen_paths


class MinRepEdgeBasedProgram(Program):
    def __init__(self, graph, num_allowed_repeaters, L_max, alpha=0, read_from_file=False):
        self.L_max = L_max
        self.alpha = alpha
        super().__init__(graph=graph, num_allowed_repeaters=num_allowed_repeaters, read_from_file=read_from_file)
        #self._check_for_feasibility()
        #self.prob.write("edge_based_min_rep_form.lp")

    def _check_for_feasibility(self):
        """Check whether a feasible solution can exist given the maximum length on the elementary link length."""
        for city in self.graph_container.city_list:
            for edge in self.graph_container.graph.edges(city):
                edge_length = self.graph_container.graph[edge[0]][edge[1]]['length']
                if edge_length <= self.L_max:
                    # There is at least one edge that can be used to 'leave' this node
                    break
            else:
                print("No feasible solution exists! There are no edges leaving {} with length smaller than "
                      "or equal to {}".format(city, self.L_max))

    def _compute_expected_number_of_variables(self):
        num_vars = (self.graph_container.num_nodes * (self.graph_container.num_nodes + 1) + 1) \
                   * len(self.graph_container.unique_city_pairs) + self.graph_container.num_nodes
        return int(num_vars)

    def _add_constraints(self):
        prob = self.prob
        num_nodes = self.graph_container.num_nodes
        # Add three types of constraints per unique pair
        row_names = []  # Save the row names for the z variables
        for q in self.graph_container.unique_city_pairs:
            pairname = q[0] + "_" + q[1]
            # Each source should have exactly one outgoing arc
            prob.linear_constraints.add(rhs=[1], senses=['E'], names=['SourceCon_' + pairname])
            flow_cons_names = ['FlowCon_' + pairname + "_" + s for s in self.graph_container.nodes]
            # Each regular node should have equal inflow and outflow
            prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['E'] * num_nodes, names=flow_cons_names)
            # Each sink should have exactly one ingoing arc
            prob.linear_constraints.add(rhs=[-1], senses=['E'], names=['SinkCon_' + pairname])
            all_nodes = list(self.graph_container.graph.nodes())
            # Constraints for maximum length of elementary links
            for i in all_nodes:
                for j in all_nodes:
                    if not (i == j or i == q[1] or j == q[0] or (i in self.graph_container.city_list and i not in q) or
                            (j in self.graph_container.city_list and j not in q)):
                        row_name = ['MaxLengthCon_' + pairname + '_' + i + '_' + j]
                        row_names.extend(row_name)
                        prob.linear_constraints.add(rhs=[self.L_max], senses=['L'],
                                                    names=row_name)
            # Constraint for maximum number of repeaters per (s,t) pair
            prob.linear_constraints.add(rhs=[self.R_max], senses=['L'], names=['MaxRepCon_' + pairname])
        # Constraints for linking path variables to repeater variables
        link_con_names = ['LinkCon_' + s for s in self.graph_container.nodes]
        # Constraints for linking the x and y variables
        prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['L'] * num_nodes, names=link_con_names)
        # Add repeater variables with a column in the linking constraint
        var_names = ['y_' + s for s in self.graph_container.nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + i],
                                                     val=[-self.M])]) for i in self.graph_container.nodes]
        # Note that these variables have a lower bound of 0 by default
        prob.variables.add(obj=[1] * num_nodes, names=var_names, ub=[1] * num_nodes, types=['B'] * num_nodes,
                           columns=link_constr_column)
        # Constraint for maximum number of repeaters
        # constr = [cplex.SparsePair(ind=range(num_nodes), val=[1.0] * num_nodes)]
        # prob.linear_constraints.add(lin_expr=constr, rhs=[self.R_max], names=['RepCon'], senses=['L'])

    def _add_variables(self):
        # Small nudge in objective function towards short elementary links
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
                        # if j not in self.graph_container.city_list:
                        #     reduc = 0
                        #     path_cost -= reduc if path_cost >= reduc else 0
                        #     if path_cost < 0:
                        #         raise ValueError("Negative path cost of {} from {} to {}".format(path_cost, i, j))
                        if i == q[0]:  # Node i is the source
                            if j == q[1]:  # Node j is the sink
                                column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                                'SinkCon_' + pairname,
                                                                'MaxLengthCon_' + pairname + '_' + i + '_' + j],
                                                           val=[1.0, -1.0, path_cost])]
                            else:  # Node j is a regular node
                                column = [cplex.SparsePair(ind=['SourceCon_' + pairname,
                                                                'FlowCon_' + pairname + "_" + str(j),
                                                                'LinkCon_' + str(j),
                                                                'MaxLengthCon_' + pairname + '_' + i + '_' + j,
                                                                'MaxRepCon_' + pairname],
                                                           val=[1.0, -1.0, 1.0, path_cost, 1.0])]
                        else:  # Node i is a regular node (note that i cannot be the sink due to line 216)
                            if j == q[1]:  # Node j is the sink
                                column = [cplex.SparsePair(ind=['SinkCon_' + pairname,
                                                                'FlowCon_' + pairname + "_" + str(i),
                                                                'MaxLengthCon_' + pairname + '_' + i + '_' + j],
                                                           val=[-1.0, 1.0, path_cost])]
                            else:  # Node j is also a regular node
                                column = [cplex.SparsePair(ind=['FlowCon_' + pairname + "_" + str(i),
                                                                'FlowCon_' + pairname + "_" + str(j),
                                                                'LinkCon_' + str(j),
                                                                'MaxLengthCon_' + pairname + '_' + i + '_' + j,
                                                                'MaxRepCon_' + pairname],
                                                           val=[1.0, -1.0, 1.0, path_cost, 1.0])]
                        # Note that these variables have a lower bound of 0 by default
                        # We need to define these as binary variables with the alternative objective function
                        cplex_var = self.prob.variables.add(obj=[self.alpha * path_cost], ub=[1], columns=column, types=['B'],
                                                            names=["x_" + pairname + "_" + str(i) + "," + str(j)])
                        self.varmap[cplex_var[0]] = (q, sp, path_cost)

    def _process_solution(self):
        sol_status = self.prob.solution.get_status_string()
        if 'infeasible' in sol_status:
            print("Solution is infeasible!")
            data = {"L_max": self.L_max,
                    "R_max": self.R_max,
                    "alpha": self.alpha,
                    "Opt_obj_val": cplex.infinity,
                    "Num_reps": cplex.infinity,
                    "Tot_path_cost": cplex.infinity,
                    "Avg_path_len": cplex.infinity,
                    "Avg_EL_len": cplex.infinity
                    }
            return [], [], data
        elif 'optimal' not in sol_status:
            print("Invalid solution status (?): {}".format(sol_status))
            return [], [], {}
        chosen_paths = []
        repeater_nodes = []
        for i, x in enumerate(self.prob.solution.get_values()):
            if x > 1e-5:
                var_name = self.prob.variables.get_names(i)
                var_ind = self.prob.variables.get_indices(var_name)
                if var_name[0:2] == "y_":
                    repeater_nodes.append(var_name[2:])
                elif var_name == "z":
                    pass
                else:
                    if self.read_from_file:
                        pass
                        # There is no access to varmap
                        var_list = var_name.split("_")
                        if len(var_list) != 4:
                            print("Something has gone wrong with splitting {}, result: {}".format(var_name, var_list))
                        pair_name = (var_list[1], var_list[2])
                        st = var_list[3].split(",")
                        s = st[0]
                        t = st[1]
                        (path_cost, path) = nx.single_source_dijkstra(G=self.graph_container.graph,
                                                                      source=s, target=t, weight='length')
                        path_tuple = (pair_name, path, path_cost)
                    else:
                        path_tuple = self.varmap[var_ind]
                    chosen_paths.append(path_tuple)
        # if len(repeater_nodes) > 0:
        #     print("{} Repeater(s) chosen: {}".format(len(repeater_nodes), repeater_nodes))
        total_cost = 0
        tot_num_el = 0
        for q in self.graph_container.unique_city_pairs:
            elementary_links_current_pair = []
            [elementary_links_current_pair.append(tup) for tup in chosen_paths if tup[0] == q]
            # Remove cyclic paths from solution
            for el1 in elementary_links_current_pair:
                for el2 in elementary_links_current_pair:
                    if el1[1][-1] == el2[1][0] and el1[1][0] == el2[1][-1]:
                        #print("Removing cyclic path: ", el1, el2)
                        elementary_links_current_pair.remove(el1)
                        elementary_links_current_pair.remove(el2)
            path = [None]
            cost = 0
            #print(q, elementary_links_current_pair)
            while path[-1] != q[1]:
                for edge in elementary_links_current_pair:
                    if len(path) == 1 and edge[1][0] == q[0]:  # Initial edge from source
                        path.remove(path[0])
                        path.extend(edge[1])
                        cost += edge[2]
                    elif len(path) > 1 and edge[1][0] == path[-1]:  # Extension of current path
                        path = path[0:-1]
                        path.extend(edge[1])
                        cost += edge[2]
            #print(q, cost, len(elementary_links_current_pair))
            total_cost += cost
            tot_num_el += len(elementary_links_current_pair)
            #print("Optimal path for pair {}: {}, using {} repeaters and cost {}".format(q, path,
            #                                                                             len(elementary_links_current_pair) - 1, cost))


        #print("Total number of elementary links:", tot_num_el)
        data = {"L_max": self.L_max,
                "R_max": self.R_max,
                "alpha": self.alpha,
                "Opt_obj_val": round(self.prob.solution.get_objective_value(), 3),
                "Num_reps": len(repeater_nodes),
                "Tot_path_cost": round(total_cost),
                "Avg_path_len": round(total_cost/self.graph_container.num_unique_pairs),
                "Avg_EL_len": round(total_cost/tot_num_el)
                }

        #print('Optimal obj val:', data["Opt_obj_val"], "num reps", data["Num_reps"],
        #      "tot path cost:", data["Tot_path_cost"], "avg EL length:", data["Avg_EL_len"])

        return repeater_nodes, chosen_paths, data

    # def _draw_solution(self, repeater_nodes, chosen_paths):
    #     pass

    def update_parameters(self, L_max_new, R_max_new, alpha_new):
        L_max_old = self.L_max
        R_max_old = self.R_max
        alpha_old = self.alpha
        self.L_max = L_max_new
        self.R_max = R_max_new
        self.alpha = alpha_new
        if L_max_old == L_max_new and R_max_old == R_max_new and alpha_old == alpha_new:
            # Nothing has changed, no need for updating
            return
        graph = self.graph_container.graph
        for q in self.graph_container.unique_city_pairs:
            pairname = q[0] + "_" + q[1]
            all_nodes = list(graph.nodes())
            if R_max_new != R_max_old:
                # Update the RHS of the maximum number of repeaters per pair constraint
                self.prob.linear_constraints.set_rhs('MaxRepCon_' + pairname, self.R_max)
            for i in all_nodes:
                for j in all_nodes:
                    if not (i == j or i == q[1] or j == q[0] or (i in self.graph_container.city_list and i not in q) or
                            (j in self.graph_container.city_list and j not in q)):
                        if L_max_new != L_max_old:
                            # Update the RHS of the maximum elementary link length per pair constraint
                            self.prob.linear_constraints.set_rhs('MaxLengthCon_' + pairname + '_' + i + '_' + j,
                                                                 self.L_max)
                        if alpha_new != alpha_old:
                            # Get the CPLEX index of the variable for this q, i and j
                            ind = self.prob.variables.get_indices("x_" + pairname + "_" + str(i) + "," + str(j))
                            # Get path cost from the corresponding constraint
                            path_cost = self.prob.linear_constraints.get_coefficients(
                                'MaxLengthCon_' + pairname + '_' + i + '_' + j, ind)
                            # (path_cost2, _) = nx.single_source_dijkstra(G=graph, source=i, target=j, weight='length')
                            # if round(path_cost, 5) != round(path_cost2, 5):
                            #     print("Disagreement in path cost: {}, {}, {}, {}".format(i, j, path_cost, path_cost2))
                            # Set the objective coefficient to alpha * c'_ij
                            self.prob.objective.set_linear(ind, self.alpha * path_cost)





# TODO: add alternative objective functions to path based program?
class PathBasedProgram(Program):
    def __init__(self, graph, num_allowed_repeaters):
        super().__init__(graph=graph, num_allowed_repeaters=num_allowed_repeaters)
        self.prob.write("path_based_form.lp")

    def _compute_expected_number_of_variables(self):
        num_vars_per_pair = 1
        for r in range(1, self.R_max + 1):
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
        prob.linear_constraints.add(lin_expr=constr, rhs=[self.R_max], names=['RepCon'], senses=['L'])
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
            for num_repeaters in range(1, self.R_max + 1):
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