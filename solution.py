import cplex
import networkx as nx
import matplotlib.pyplot as plt


class Solution:
    """Helper class that parses the solution of a formulation."""
    def __init__(self, formulation):
        self.formulation = formulation
        self.parameters, self.overall_data = self._setup_solution()
        self.feasible = "infeasible" not in self.get_status_string()
        if not self.parameters:
            return
        self.x_variables_chosen, self.repeater_nodes_chosen = self._interpret_variables()
        self.path_data = self._process_x_variables()
        self._create_virtual_solution_graph()

    def _setup_solution(self):
        """Parse the solution of the formulation."""
        sol_status = self.get_status_string()
        parameters = {}
        overall_data = {}
        if 'infeasible' in sol_status:
            print("Solution is infeasible!")
        elif 'optimal' not in sol_status:
            print("Invalid solution status (?): {}".format(sol_status))
        else:
            parameters = {"L_max": self.formulation.L_max,
                          "N_max": self.formulation.N_max,
                          "K": self.formulation.K,
                          "D": self.formulation.D,
                          "alpha": self.formulation.alpha,
                          }
            overall_data = {"opt_obj_val": cplex.infinity,
                            "num_reps": 0,
                            "tot_path_cost": 0,
                            "avg_path_len": 0,
                            "tot_num_el": 0,
                            "repeater_node_degree": {}
                            }
        return parameters, overall_data

    def _interpret_variables(self):
        """Interpret the chosen decision variables, which either gets the paths or the elementary links, depending
        on which formulation is used"""
        x_variables_chosen = []
        repeater_nodes_chosen = []
        for idx, val in enumerate(self.formulation.cplex.solution.get_values()):
            if val > 1e-5:
                var_name = self.formulation.cplex.variables.get_names(idx)
                if var_name[0:2] == "y_":
                    # This is a repeater node (y) variable
                    repeater_nodes_chosen.append(var_name[2:])
                else:
                    if self.formulation.read_from_file:
                        pass
                        # There is no access to varmap
                        var_list = var_name.split("_")
                        if len(var_list) != 4:
                            print("Something has gone wrong with splitting {}, result: {}".format(var_name, var_list))
                        pair_name = (var_list[1], var_list[2])
                        st = var_list[3].split(",")
                        s = st[0]
                        t = st[1]
                        (path_cost, path) = nx.single_source_dijkstra(G=self.formulation.graph_container.graph,
                                                                      source=s, target=t, weight='length')
                        path_tuple = (pair_name, path, path_cost)
                    else:
                        path_tuple = self.formulation.varmap[idx]
                    x_variables_chosen.append(path_tuple)
        # if len(repeater_nodes_chosen) > 0:
        #     print("{} Repeater(s) chosen: {}".format(len(repeater_nodes_chosen), repeater_nodes_chosen))
        self.overall_data['num_reps'] = len(repeater_nodes_chosen)
        opt_obj_val = round(self.formulation.cplex.solution.get_objective_value(), 5)
        if opt_obj_val - len(repeater_nodes_chosen) > 1:
            print("WARNING: value of alpha too large, influenced objective function! Optimal objective value: {}, "
                  "number of repeaters: {}".format(opt_obj_val, len(repeater_nodes_chosen)))
        self.overall_data['opt_obj_val'] = opt_obj_val
        return x_variables_chosen, repeater_nodes_chosen

    def _process_x_variables(self):
        path_data = {(q[0], q[1]): {} for q in self.formulation.graph_container.unique_end_node_pairs}
        repeater_node_degree = {u: 0 for u in self.repeater_nodes_chosen}
        total_cost, tot_num_el = 0, 0
        for q in self.formulation.graph_container.unique_end_node_pairs:
            paths, repeater_nodes_used, num_el_used, cost_per_path = [], [], [], []
            if "Path" in str(type(self.formulation)):
                # We are processing a solution of the path-based formulation
                path_properties = [tup for tup in self.x_variables_chosen if tup[0] == q]
                for k in range(self.formulation.K):
                    r_up = path_properties[k][2]
                    for u in r_up:
                        repeater_node_degree[u] += 1
                        if u not in self.repeater_nodes_chosen:
                            print("Warning! r_up = {} but corresponding y_i is not 1 (this should never happen)")
                    paths.append(path_properties[k][1])
                    num_el = (len(r_up) + 1)
                    num_el_used.append(num_el)
                    tot_num_el += num_el
                    cost_this_path = path_properties[k][3]
                    cost_per_path.append(cost_this_path)
                    total_cost += cost_this_path
                    repeater_nodes_used.append(r_up)
                local_dict = path_data[(q[0], q[1])]
                local_dict['paths'] = paths
                local_dict['num_el_used'] = num_el_used
                local_dict['repeater_nodes_used'] = repeater_nodes_used
                local_dict['path_cost'] = cost_per_path
            else:
                # We are processing a link-based formulation
                elementary_links_current_pair = [tup for tup in self.x_variables_chosen if tup[0] == q]
                # Remove cyclic paths from solution, which can occur if alpha is set to zero
                for el1 in elementary_links_current_pair:
                    for el2 in elementary_links_current_pair:
                        if el1[1][-1] == el2[1][0] and el1[1][0] == el2[1][-1]:
                            elementary_links_current_pair.remove(el1)
                            elementary_links_current_pair.remove(el2)
                num_el = len(elementary_links_current_pair)
                tot_num_el += num_el
                for _ in range(self.formulation.K):
                    path = [None]
                    old_len_rep_nodes = len(repeater_nodes_used)
                    rep_nodes_current_path = []
                    num_el_current_path = 0
                    cost_current_path = 0
                    while path[-1] != q[1]:
                        for edge in elementary_links_current_pair:
                            if len(path) == 1 and edge[1][0] == q[0]:  # Initial edge from source
                                path.remove(path[0])
                                path.extend(edge[1])
                                cost_current_path += edge[2]
                                num_el_current_path += 1
                                elementary_links_current_pair.remove(edge)
                            elif len(path) > 1 and edge[1][0] == path[-1]:  # Extension of current path
                                rep_nodes_current_path.append(path[-1])
                                repeater_node_degree[path[-1]] += 1
                                path = path[0:-1]
                                path.extend(edge[1])
                                cost_current_path += edge[2]
                                num_el_current_path += 1
                                elementary_links_current_pair.remove(edge)
                    num_el_used.append([num_el_current_path])
                    cost_per_path.append([cost_current_path])
                    total_cost += cost_current_path
                    repeater_nodes_used.append(rep_nodes_current_path)
                    if len(repeater_nodes_used) == old_len_rep_nodes:
                        repeater_nodes_used.append([])
                    paths.append(path)
                local_dict = path_data[(q[0], q[1])]
                local_dict['paths'] = paths
                local_dict['num_el_used'] = num_el_used
                local_dict['repeater_nodes_used'] = repeater_nodes_used
                local_dict['path_cost'] = cost_per_path

        used_elementary_links, used_edges, visited_nodes = [], [], set()

        for tup in self.x_variables_chosen:
            elementary_link_path = tup[1]
            if "Path" in str(type(self.formulation)):
                # Deconstruct the path to its elementary links
                r_up = tup[2]
                if not r_up:
                    # Direct elementary link is used as a path
                    used_elementary_links.append((elementary_link_path[0], elementary_link_path[-1]))
                else:
                    starting_node = elementary_link_path[0]
                    for i in range(1, len(elementary_link_path)):
                        cur_node = elementary_link_path[i]
                        if cur_node in r_up:
                            used_elementary_links.append((starting_node, cur_node))
                            starting_node = cur_node
                    # Add one more elementary link from the last repeater node to t
                    used_elementary_links.append((starting_node, elementary_link_path[-1]))
            else:
                used_elementary_links.append((elementary_link_path[0], elementary_link_path[-1]))
            visited_nodes.update(elementary_link_path)
            for i in range(len(elementary_link_path) - 1):
                used_edges.append((elementary_link_path[i], elementary_link_path[i+1]))
        self.visited_nodes = list(visited_nodes)
        self.link_extension_nodes = list(set([i for i in visited_nodes if i not in self.repeater_nodes_chosen
                                              and i not in self.formulation.graph_container.end_nodes]))
        self.used_elementary_links = used_elementary_links
        self.used_edges = list(set(used_edges))
        self.unused_edges = list(self.formulation.graph_container.graph.edges())
        for tup in self.used_edges:
            if tup in self.unused_edges:
                self.unused_edges.remove(tup)
            elif tuple(reversed(tup)) in self.unused_edges:
                self.unused_edges.remove(tuple(reversed(tup)))

        self.overall_data['tot_path_cost'] = round(total_cost, 3)
        self.overall_data['avg_path_len'] = round(total_cost / self.formulation.graph_container.num_unique_pairs)
        self.overall_data['tot_num_el'] = tot_num_el
        self.overall_data['repeater_node_degree'] = repeater_node_degree
        return path_data

    def _create_virtual_solution_graph(self):
        """Create a virtual graph based on the solution where the edges are elementary links."""
        pos = nx.get_node_attributes(self.formulation.graph_container.graph, 'pos')
        self.virtual_solution_graph = nx.Graph()
        self.virtual_solution_graph.add_nodes_from(self.formulation.graph_container.end_nodes + self.repeater_nodes_chosen)
        nx.set_node_attributes(self.virtual_solution_graph, pos, name='pos')
        self.virtual_solution_graph.add_edges_from(self.used_elementary_links)

    def get_status_string(self):
        return self.formulation.cplex.solution.get_status_string()

    def get_parameters(self):
        return self.parameters

    def get_solution_data(self):
        if "min_node_connectivity" not in self.overall_data.keys() and 'optimal' in self.get_status_string():
            min_node_connectivity, avg_node_connectivity = self.compute_node_connectivity()
            min_edge_connectivity, avg_edge_connectivity = self.compute_edge_connectivity()
            self.overall_data.update({"min_node_connectivity": min_node_connectivity,
                                      "avg_node_connectivity": avg_node_connectivity,
                                      "min_edge_connectivity": min_edge_connectivity,
                                      "avg_edge_connectivity": avg_edge_connectivity})
        return self.overall_data

    def get_path_data(self):
        return self.path_data

    def print_path_data(self):
        for k in range(self.formulation.K):
            for q in self.path_data:
                print("K = {}, q = {}: path = {}, num_el = {}, reps = {}, path_cost = {}".format(k + 1, q,
                      self.path_data[q]['paths'][k],
                      self.path_data[q]['num_el_used'][k],
                      self.path_data[q]['repeater_nodes_used'][k],
                      self.path_data[q]['path_cost'][k]))

    def draw_virtual_solution_graph(self):
        pos = nx.get_node_attributes(self.virtual_solution_graph, 'pos')
        # Create blank figure
        fig, ax = plt.subplots(figsize=(7, 7))
        # First draw the end nodes
        end_nodes = nx.draw_networkx_nodes(G=self.virtual_solution_graph, pos=pos, node_size=1500,
                                           nodelist=self.formulation.graph_container.end_nodes,
                                           node_shape='s', node_color=[[255 / 255, 120 / 255, 0 / 255]],
                                           label="End Node",
                                           linewidths=3)
        end_nodes.set_edgecolor('K')
        # Then draw the repeater nodes
        if self.repeater_nodes_chosen:
            rep_nodes = nx.draw_networkx_nodes(G=self.virtual_solution_graph, pos=pos, node_size=1500,
                                               node_shape='h', nodelist=self.repeater_nodes_chosen,
                                               node_color=[[0 / 255, 166 / 255, 214 / 255]], label="Repeater Node",
                                               linewidths=3)
            rep_nodes.set_edgecolor('K')
        # Finally draw the elementary links
        nx.draw_networkx_edges(G=self.virtual_solution_graph, pos=pos, edgelist=self.used_elementary_links, width=8)
        # Draw all the node labels
        labels = {node: node if node in self.formulation.graph_container.end_nodes else ""
                  for node, nodedata in self.virtual_solution_graph.nodes.items()}
        nx.draw_networkx_labels(G=self.virtual_solution_graph, pos=pos, labels=labels, font_size=30,
                                font_weight="bold", font_color="w", font_family='serif')
        # Change some margins etc
        plt.axis('off')
        margin = 0.33
        fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        ax.axis('equal')
        fig.tight_layout()
        # Show the plot
        plt.show()

    def draw_physical_solution_graph(self):
        pos = nx.get_node_attributes(self.formulation.graph_container.graph, 'pos')
        labels = {}
        for node, nodedata in self.formulation.graph_container.graph.nodes.items():
            if node in self.formulation.graph_container.end_nodes:
                labels[node] = node
            else:
                labels[node] = ""
        # Empty figure
        fig, ax = plt.subplots(figsize=(7, 7))
        # First draw end nodes
        end_nodes = nx.draw_networkx_nodes(G=self.formulation.graph_container.graph, pos=pos, node_size=1500,
                                           nodelist=self.formulation.graph_container.end_nodes,
                                           node_shape='s', node_color=[[255 / 255, 120 / 255, 0 / 255]],
                                           label="End Node", linewidths=3)
        end_nodes.set_edgecolor('k')
        # Then draw the repeater nodes
        if self.repeater_nodes_chosen:
            rep_nodes = nx.draw_networkx_nodes(G=self.formulation.graph_container.graph, pos=pos, node_size=1500,
                                               node_shape='h', nodelist=self.repeater_nodes_chosen,
                                               node_color=[[0 / 255, 166 / 255, 214 / 255]], label="Repeater Node",
                                               linewidths=3)
            rep_nodes.set_edgecolor('k')
        # Then draw the link-extension nodes
        if self.link_extension_nodes:
            le_nodes = nx.draw_networkx_nodes(G=self.formulation.graph_container.graph, pos=pos, node_size=1500,
                                              nodelist=self.link_extension_nodes, node_color=[[1, 1, 1]],
                                              label="Link Extension")
            le_nodes.set_edgecolor('k')
        # Also draw all the unused nodes
        unused_nodes = []
        [unused_nodes.append(n) for n in self.formulation.graph_container.graph.nodes() if
         (n not in self.link_extension_nodes and n not in self.repeater_nodes_chosen
          and n not in self.formulation.graph_container.end_nodes)]
        if unused_nodes:
            unu_nodes = nx.draw_networkx_nodes(G=self.formulation.graph_container.graph, pos=pos, node_size=1500,
                                               nodelist=unused_nodes, node_color=[[1, 1, 1]])
            unu_nodes.set_edgecolor('k')
        # Draw the unused edges but less visible
        nx.draw_networkx_edges(G=self.formulation.graph_container.graph, pos=pos, edgelist=self.unused_edges,
                               edge_color=[[170 / 255, 170 / 255, 170 / 255]], width=1)
        # Draw the used edges
        nx.draw_networkx_edges(G=self.formulation.graph_container.graph, pos=pos, edgelist=self.used_edges, width=8,
                               edge_color=[[0 / 255, 0 / 255, 0 / 255]])
        # Draw all the node labels
        nx.draw_networkx_labels(G=self.formulation.graph_container.graph, pos=pos, labels=labels, font_size=30,
                                font_weight="bold", font_color="w", font_family='serif')
        plt.axis('off')
        margin = 0.33
        fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        ax.axis('equal')
        fig.tight_layout()
        plt.show()

    def compute_node_connectivity(self):
        connectivity_dict = nx.all_pairs_node_connectivity(self.virtual_solution_graph,
                                                           self.formulation.graph_container.end_nodes)
        min_connectivity = 1e20
        total_connectivity = 0
        for key, val in connectivity_dict.items():
            vals = list(val.values())
            total_connectivity += sum(vals)
            if min(vals) < min_connectivity:
                min_connectivity = min(vals)
        # Divide by 2 times the unique (s,t) pairs, since connectivity is computed in both directions
        avg_connectivity = total_connectivity / (2 * self.formulation.graph_container.num_unique_pairs)
        return min_connectivity, avg_connectivity

    def compute_edge_connectivity(self):
        total_edge_connectivity = 0
        minimum_edge_connectivity = 1e20
        for q in self.formulation.graph_container.unique_end_node_pairs:
            edge_connectivity_this_pair = nx.edge_connectivity(G=self.virtual_solution_graph, s=q[0], t=q[1])
            total_edge_connectivity += edge_connectivity_this_pair
            if edge_connectivity_this_pair < minimum_edge_connectivity:
                minimum_edge_connectivity = edge_connectivity_this_pair
        avg_edge_connectivity = total_edge_connectivity / self.formulation.graph_container.num_unique_pairs
        return minimum_edge_connectivity, avg_edge_connectivity
