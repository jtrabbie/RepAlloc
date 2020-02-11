from programs import Program
import cplex
import networkx as nx
import matplotlib.pyplot as plt
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
            flow_cons_names = ['FlowCon_' + pairname + "_" + s for s in self.graph_container.possible_rep_nodes]
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
        link_con_names = ['LinkCon_' + s for s in self.graph_container.possible_rep_nodes]
        # Constraints for linking the x and y variables
        prob.linear_constraints.add(rhs=[0] * num_nodes, senses=['L'] * num_nodes, names=link_con_names)
        # Add repeater variables with a column in the linking constraint
        var_names = ['y_' + s for s in self.graph_container.possible_rep_nodes]
        # Node that if we want to add 6 variables, we need to have 6 separate SparsePairs
        link_constr_column = []
        [link_constr_column.extend([cplex.SparsePair(ind=['LinkCon_' + i],
                                                     val=[-self.M])]) for i in self.graph_container.possible_rep_nodes]
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
            total_cost += cost
            tot_num_el += len(elementary_links_current_pair)
            # print("Optimal path for pair {}: {}, using {} repeaters and cost {}".format(q, path,
            #       len(elementary_links_current_pair) - 1, cost))


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

    def draw_fixed_solution(self):
        """Draw solution of the Colt data set for L_max = 900, R_max = 6 and beta = 1 / 75000."""
        # Fixed solution values
        repeater_nodes = ['Munich', 'Dusseldorf', 'Bordeaux', 'Rouen', 'Madrid']
        chosen_paths = [(('TheHague', 'Lisbon'), ['TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris', 'Rouen'], 568.1262055866217),
                        (('TheHague', 'Lisbon'), ['Bordeaux', 'Madrid'], 553.9975778357486),
                        (('TheHague', 'Lisbon'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('TheHague', 'Lisbon'), ['Madrid', 'Lisbon'], 502.312485724141),
                        (('TheHague', 'Copenhagen'), ['TheHague', 'Hoofddorp', 'Amsterdam', 'Dusseldorf', 'Essen', 'Hamburg', 'Copenhagen'], 862.110598442322),
                        (('TheHague', 'Innsbruck'), ['TheHague', 'Hoofddorp', 'Amsterdam', 'Dusseldorf', 'Cologne', 'Frankfurt', 'Mannheim', 'Karlsruhe', 'Stuttgart', 'Munich'], 800.9213497397336),
                        (('TheHague', 'Innsbruck'), ['Munich', 'Vienna', 'Linz', 'Salzburg', 'Innsbruck'], 756.1610898157807),
                        (('TheHague', 'Basel'), ['TheHague', 'Hoofddorp', 'Amsterdam', 'Dusseldorf', 'Cologne', 'Frankfurt', 'Mannheim', 'Karlsruhe', 'Strasbourg', 'Basel'], 726.8449670699096),
                        (('TheHague', 'Stuttgart'), ['TheHague', 'Hoofddorp', 'Amsterdam', 'Dusseldorf', 'Cologne', 'Frankfurt', 'Mannheim', 'Karlsruhe', 'Stuttgart'], 610.0887229223069),
                        (('TheHague', 'Geneva'), ['TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris', 'Rouen'], 568.1262055866217),
                        (('TheHague', 'Geneva'), ['Rouen', 'Paris', 'Lyon', 'Geneva'], 617.3118302774578),
                        (('TheHague', 'Barcelona'), ['TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris', 'Rouen'], 568.1262055866217),
                        (('TheHague', 'Barcelona'), ['Bordeaux', 'Madrid'], 553.9975778357486), (('TheHague', 'Barcelona'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('TheHague', 'Barcelona'), ['Madrid', 'Valencia', 'Barcelona'], 605.033093501733),
                        (('TheHague', 'Paris'), ['TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris'], 456.0624993638363),
                        (('Lisbon', 'Copenhagen'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Copenhagen'), ['Dusseldorf', 'Essen', 'Hamburg', 'Copenhagen'], 627.41016231492),
                        (('Lisbon', 'Copenhagen'), ['Rouen', 'Paris', 'Lille', 'Ghent', 'Brussels', 'Antwerp', 'TheHague', 'Hoofddorp', 'Amsterdam', 'Dusseldorf'], 802.8266417140237),
                        (('Lisbon', 'Copenhagen'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen'], 626.4617287383495),
                        (('Lisbon', 'Copenhagen'), ['Madrid', 'Bordeaux'], 553.9975778357486),
                        (('Lisbon', 'Innsbruck'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Innsbruck'), ['Munich', 'Vienna', 'Linz', 'Salzburg', 'Innsbruck'], 756.1610898157807),
                        (('Lisbon', 'Innsbruck'), ['Rouen', 'Paris', 'Strasbourg', 'Karlsruhe', 'Stuttgart', 'Munich'], 829.1630535089),
                        (('Lisbon', 'Innsbruck'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen'], 626.4617287383495),
                        (('Lisbon', 'Innsbruck'), ['Madrid', 'Bordeaux'], 553.9975778357486),
                        (('Lisbon', 'Basel'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Basel'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen'], 626.4617287383495),
                        (('Lisbon', 'Basel'), ['Madrid', 'Bordeaux'],553.9975778357486),
                        (('Lisbon', 'Basel'), ['Rouen', 'Paris', 'Strasbourg', 'Basel'], 622.9541518426047),
                        (('Lisbon', 'Stuttgart'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Stuttgart'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen'], 626.4617287383495),
                        (('Lisbon', 'Stuttgart'), ['Madrid', 'Bordeaux'], 553.9975778357486),
                        (('Lisbon', 'Stuttgart'), ['Rouen', 'Paris', 'Strasbourg', 'Karlsruhe', 'Stuttgart'], 638.3304266914732),
                        (('Lisbon', 'Geneva'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Geneva'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen'], 626.4617287383495),
                        (('Lisbon', 'Geneva'), ['Madrid', 'Bordeaux'], 553.9975778357486),
                        (('Lisbon', 'Geneva'), ['Rouen', 'Paris', 'Lyon', 'Geneva'], 617.3118302774578),
                        (('Lisbon', 'Barcelona'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Barcelona'), ['Madrid', 'Valencia', 'Barcelona'], 605.033093501733),
                        (('Lisbon', 'Paris'), ['Lisbon', 'Madrid'], 502.312485724141),
                        (('Lisbon', 'Paris'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen', 'Paris'], 738.5254349611349),
                        (('Lisbon', 'Paris'), ['Madrid', 'Bordeaux'], 553.9975778357486),
                        (('Copenhagen', 'Innsbruck'), ['Copenhagen', 'Hamburg', 'Essen', 'Dusseldorf'], 627.41016231492),
                        (('Copenhagen', 'Innsbruck'), ['Munich', 'Vienna', 'Linz', 'Salzburg', 'Innsbruck'], 756.1610898157807),
                        (('Copenhagen', 'Innsbruck'), ['Dusseldorf', 'Cologne', 'Frankfurt', 'Mannheim', 'Karlsruhe', 'Stuttgart', 'Munich'], 566.2209136123315),
                        (('Copenhagen', 'Basel'), ['Copenhagen', 'Hamburg', 'Essen', 'Dusseldorf'], 627.41016231492),
                        (('Copenhagen', 'Basel'), ['Dusseldorf', 'Cologne', 'Frankfurt', 'Mannheim', 'Karlsruhe', 'Strasbourg', 'Basel'], 492.1445309425075),
                        (('Copenhagen', 'Stuttgart'), ['Copenhagen', 'Hamburg', 'Essen', 'Dusseldorf'], 627.41016231492),
                        (('Copenhagen', 'Stuttgart'), ['Dusseldorf', 'Cologne', 'Frankfurt', 'Mannheim','Karlsruhe', 'Stuttgart'], 375.38828679490484),
                        (('Copenhagen', 'Geneva'), ['Copenhagen', 'Hamburg', 'Essen', 'Dusseldorf'], 627.41016231492),
                        (('Copenhagen', 'Geneva'), ['Dusseldorf', 'Amsterdam', 'Hoofddorp', 'TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris', 'Rouen'], 802.8266417140237),
                        (('Copenhagen', 'Geneva'), ['Rouen', 'Paris', 'Lyon', 'Geneva'], 617.3118302774578),
                        (('Copenhagen', 'Barcelona'), ['Copenhagen', 'Hamburg', 'Essen', 'Dusseldorf'], 627.41016231492),
                        (('Copenhagen', 'Barcelona'), ['Dusseldorf', 'Amsterdam', 'Hoofddorp', 'TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris', 'Rouen'], 802.8266417140237),
                        (('Copenhagen', 'Barcelona'), ['Bordeaux', 'Madrid'], 553.9975778357486),
                        (('Copenhagen', 'Barcelona'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('Copenhagen', 'Barcelona'), ['Madrid', 'Valencia', 'Barcelona'], 605.033093501733),
                        (('Copenhagen', 'Paris'), ['Copenhagen', 'Hamburg', 'Essen', 'Dusseldorf'], 627.41016231492),
                        (('Copenhagen', 'Paris'), ['Dusseldorf', 'Amsterdam', 'Hoofddorp', 'TheHague', 'Antwerp', 'Brussels', 'Ghent', 'Lille', 'Paris'], 690.7629354912383),
                        (('Innsbruck', 'Basel'), ['Innsbruck', 'Salzburg', 'Linz', 'Vienna', 'Munich'], 756.1610898157807),
                        (('Innsbruck', 'Basel'), ['Munich', 'Stuttgart', 'Karlsruhe', 'Strasbourg', 'Basel'], 433.399602015763),
                        (('Innsbruck', 'Stuttgart'), ['Innsbruck', 'Salzburg', 'Linz', 'Vienna', 'Munich'], 756.1610898157807),
                        (('Innsbruck', 'Stuttgart'), ['Munich', 'Stuttgart'], 190.83262681742673),
                        (('Innsbruck', 'Geneva'), ['Innsbruck', 'Salzburg', 'Linz', 'Vienna', 'Munich'], 756.1610898157807),
                        (('Innsbruck', 'Geneva'), ['Munich', 'Stuttgart', 'Karlsruhe', 'Strasbourg', 'Paris', 'Rouen'], 829.1630535089),
                        (('Innsbruck', 'Geneva'), ['Rouen', 'Paris', 'Lyon', 'Geneva'], 617.3118302774578),
                        (('Innsbruck', 'Barcelona'), ['Innsbruck', 'Salzburg', 'Linz', 'Vienna', 'Munich'], 756.1610898157807),
                        (('Innsbruck', 'Barcelona'), ['Munich', 'Stuttgart', 'Karlsruhe', 'Strasbourg', 'Paris', 'Rouen'], 829.1630535089),
                        (('Innsbruck', 'Barcelona'), ['Bordeaux', 'Madrid'], 553.9975778357486),
                        (('Innsbruck', 'Barcelona'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('Innsbruck', 'Barcelona'), ['Madrid', 'Valencia','Barcelona'], 605.033093501733),
                        (('Innsbruck', 'Paris'), ['Innsbruck', 'Salzburg', 'Linz', 'Vienna', 'Munich'], 756.1610898157807),
                        (('Innsbruck', 'Paris'), ['Munich', 'Stuttgart', 'Karlsruhe', 'Strasbourg', 'Paris'], 717.0993472861146),
                        (('Basel', 'Stuttgart'), ['Basel', 'Strasbourg', 'Karlsruhe', 'Stuttgart'], 242.56697519833625),
                        (('Basel', 'Geneva'), ['Basel', 'Strasbourg', 'Paris', 'Rouen'], 622.9541518426048),
                        (('Basel', 'Geneva'), ['Rouen', 'Paris', 'Lyon', 'Geneva'], 617.3118302774578),
                        (('Basel', 'Barcelona'), ['Basel', 'Strasbourg', 'Paris', 'Rouen'], 622.9541518426048),
                        (('Basel', 'Barcelona'), ['Bordeaux', 'Madrid'], 553.9975778357486),
                        (('Basel', 'Barcelona'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('Basel', 'Barcelona'), ['Madrid', 'Valencia', 'Barcelona'], 605.033093501733),
                        (('Basel', 'Paris'), ['Basel', 'Strasbourg', 'Paris'], 510.89044561981933),
                        (('Stuttgart', 'Geneva'), ['Stuttgart', 'Karlsruhe', 'Strasbourg', 'Paris', 'Rouen'], 638.3304266914732),
                        (('Stuttgart', 'Geneva'), ['Rouen', 'Paris', 'Lyon', 'Geneva'], 617.3118302774578),
                        (('Stuttgart', 'Barcelona'), ['Stuttgart', 'Karlsruhe', 'Strasbourg', 'Paris', 'Rouen'], 638.3304266914732),
                        (('Stuttgart', 'Barcelona'), ['Bordeaux', 'Madrid'], 553.9975778357486),
                        (('Stuttgart', 'Barcelona'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('Stuttgart', 'Barcelona'), ['Madrid', 'Valencia', 'Barcelona'], 605.033093501733),
                        (('Stuttgart', 'Paris'), ['Stuttgart', 'Karlsruhe', 'Strasbourg', 'Paris'], 526.2667204686878),
                        (('Geneva', 'Barcelona'), ['Geneva', 'Lyon', 'Paris', 'Rouen'], 617.311830277458),
                        (('Geneva', 'Barcelona'), ['Bordeaux', 'Madrid'], 553.9975778357486),
                        (('Geneva', 'Barcelona'), ['Rouen', 'Rennes', 'Nantes', 'Bordeaux'], 626.4617287383495),
                        (('Geneva', 'Barcelona'), ['Madrid', 'Valencia', 'Barcelona'], 605.033093501733),
                        (('Geneva', 'Paris'), ['Geneva', 'Lyon', 'Paris'], 505.2481240546725),
                        (('Barcelona', 'Paris'), ['Barcelona', 'Valencia', 'Madrid'], 605.033093501733),
                        (('Barcelona', 'Paris'), ['Bordeaux', 'Nantes', 'Rennes', 'Rouen', 'Paris'], 738.5254349611349),
                        (('Barcelona', 'Paris'), ['Madrid', 'Bordeaux'], 553.9975778357486)]
        pos = {}
        labels = {}
        color_map = []
        visited_cities = []
        rep_nodes_drawing = []
        le_nodes_drawing = []
        edge_list = []
        #print(self.graph_container.city_list)
        #print(self.graph_container.unique_city_pairs)
        for tup in chosen_paths:
            elementary_link_path = tup[1]
            visited_cities.extend(elementary_link_path)
            #print("EL_path", elementary_link_path)
            for i in range(len(elementary_link_path) - 1):
                edge_list.append((elementary_link_path[i], elementary_link_path[i+1]))
            #print("Edge_list", edge_list)
        edge_list = list(set(edge_list))
        visited_cities = list(set(visited_cities))
        for node, nodedata in self.graph_container.graph.nodes.items():
            if 'Longitude' in nodedata:
                pos[node] = [nodedata['Longitude'], nodedata['Latitude']]
            else:
                pos[node] = [nodedata['xcoord'], nodedata['ycoord']]
            if node in self.graph_container.city_list:
                if node == "Geneve":
                    labels[node] = "Geneva"
                elif node == "TheHague":
                    labels[node] = "The Hague"
                else:
                    labels[node] = node
                color_map.append('green')
            elif node in repeater_nodes:
                labels[node] = node
                rep_nodes_drawing.append(node)
                color_map.append('pink')
            elif node in visited_cities:
                #labels[node] = node
                le_nodes_drawing.append(node)
                color_map.append([30 / 255, 144 / 255, 255 / 255])
            else:
                labels[node] = ""
                color_map.append('none')
        fig, ax = plt.subplots(figsize=(10, 6))
        # First draw end nodes
        #print(repeater_nodes)
        pos["Innsbruck"][1] = pos["Innsbruck"][1] - 1.4
        end_nodes = nx.draw_networkx_nodes(G=self.graph_container.graph, pos=pos, nodelist=self.graph_container.city_list,
                                           node_shape='s', node_color=[[0.66, 0.93, 0.73]], label="End Node")
        end_nodes.set_edgecolor('k')
        if repeater_nodes:
            rep_nodes = nx.draw_networkx_nodes(G=self.graph_container.graph, pos=pos, nodelist=rep_nodes_drawing,
                                               node_color=[[30 / 255, 144 / 255, 255 / 255]], label="Repeater Node")
            rep_nodes.set_edgecolor('k')
        le_nodes = nx.draw_networkx_nodes(G=self.graph_container.graph, pos=pos, nodelist=le_nodes_drawing,
                                          node_color=[[0.83, 0.83, 0.83]], label="Link Extension", alpha=0.2)
        le_nodes.set_edgecolor('k')
        nx.draw_networkx_edges(G=self.graph_container.graph, pos=pos, edgelist=edge_list)
        # Modify pos for labels
        pos["Copenhagen"][0] = 8.7
        pos["TheHague"][1] = pos["TheHague"][1] + 1.1
        pos["Dusseldorf"][0] = 10.2
        pos["Rouen"] = [pos["Rouen"][0] - 2.1, pos["Rouen"][1] + 0.1]
        pos["Paris"] = [4.2, 49.5]
        pos["Stuttgart"] = [11.5, 49.6]
        pos["Madrid"][1] = 39.2
        pos["Lisbon"][1] = 39.6
        pos["Barcelona"][1] = 42.3
        pos["Bordeaux"][0] = - 3.5
        pos["Geneva"] = [7.1, 45.2]
        pos["Basel"][0] = 5.7
        pos["Innsbruck"] = [pos["Innsbruck"][0] + 3.2, pos["Innsbruck"][1] - 0.1]
        pos["Munich"] = [10.3, 47.2]
        nx.draw_networkx_labels(G=self.graph_container.graph, pos=pos, labels=labels, font_size=20, font_weight="bold")
        #nx.draw(self.graph_container.graph, labels=labels, with_labels=True, font_weight='bold',
        #        pos=pos, node_color=color_map, node_size=200)
        plt.axis('off')
        #margin = 0.33
        #fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        ax.axis('equal')
        plt.tight_layout()
        plt.legend(loc='upper left', fontsize=20)
        plt.show()



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