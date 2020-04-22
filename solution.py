import cplex
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
import scipy.stats as stats


class Solution:
    def __init__(self, program):
        self.program = program
        self.parameters, self.overall_data = self._check_solution_status()
        if self.parameters == {}:
            return
        self.x_variables_chosen, self.repeater_nodes_chosen = self._interpret_variables()
        self.path_data = self._process_x_variables()
        self._create_virtual_solution_graph()

    def _check_solution_status(self):
        sol_status = self.program.prob.solution.get_status_string()
        parameters = {}
        overall_data = {}
        if 'infeasible' in sol_status:
            print("Solution is infeasible!")
        elif 'optimal' not in sol_status:
            print("Invalid solution status (?): {}".format(sol_status))
        else:
            parameters = {"L_max": self.program.L_max,
                          "R_max": self.program.R_max,
                          "k": self.program.k,
                          "D": self.program.D,
                          "alpha": self.program.alpha,
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
        x_variables_chosen = []
        repeater_nodes_chosen = []
        for i, x in enumerate(self.program.prob.solution.get_values()):
            if x > 1e-5:
                var_name = self.program.prob.variables.get_names(i)
                # print(var_name, x)
                if var_name[0:2] == "y_" and '#' not in var_name:
                    repeater_nodes_chosen.append(var_name[2:])
                elif var_name[0:1] == 'x' and (var_name[1:2] == '(' and '#' in var_name) or var_name[1:2] != '(':
                    if self.program.read_from_file:
                        pass
                        # There is no access to varmap
                        var_list = var_name.split("_")
                        if len(var_list) != 4:
                            print("Something has gone wrong with splitting {}, result: {}".format(var_name, var_list))
                        pair_name = (var_list[1], var_list[2])
                        st = var_list[3].split(",")
                        s = st[0]
                        t = st[1]
                        (path_cost, path) = nx.single_source_dijkstra(G=self.program.graph_container.graph,
                                                                      source=s, target=t, weight='length')
                        path_tuple = (pair_name, path, path_cost)
                    else:
                        path_tuple = self.program.varmap[i]
                    x_variables_chosen.append(path_tuple)
        self.overall_data['num_reps'] = len(repeater_nodes_chosen)
        opt_obj_val = round(self.program.prob.solution.get_objective_value(), 5)
        if opt_obj_val - len(repeater_nodes_chosen) > 1:
            print("WARNING: value of alpha too large, influenced objective function! Optimal objective value: {},"
                  "number of repeaters: {}".format(opt_obj_val, len(repeater_nodes_chosen)))
        self.overall_data['opt_obj_val'] = opt_obj_val
        # if len(repeater_nodes_chosen) > 0:
        #     print("{} Repeater(s) chosen: {}".format(len(repeater_nodes_chosen), repeater_nodes_chosen))
        return x_variables_chosen, repeater_nodes_chosen

    def _process_x_variables(self):
        pair_dict = {(q[0], q[1]): {} for q in self.program.graph_container.unique_end_node_pairs}
        repeater_node_degree = {k: 0 for k in self.repeater_nodes_chosen}
        total_cost = 0
        tot_num_el = 0
        for q in self.program.graph_container.unique_end_node_pairs:
            cost = 0
            paths = []
            repeater_nodes_used = []
            num_el_used = []
            cost_per_path = []
            if "PathBasedProgram" in self.program.__class__.__name__:
                path_properties = []
                [path_properties.append(tup) for tup in self.x_variables_chosen if tup[0] == q]
                for k in range(self.program.k):
                    r_ip = path_properties[k][2]
                    for i in r_ip:
                        repeater_node_degree[i] += 1
                        if i not in self.repeater_nodes_chosen:
                            print("Warning! r_ip = {} but corresponding y_i is not 1 (this should never happen)")
                    paths.append(path_properties[k][1])
                    num_el = (len(r_ip) + 1)
                    num_el_used.append(num_el)
                    tot_num_el += num_el
                    cost_this_path = path_properties[k][3]
                    cost_per_path.append(cost_this_path)
                    total_cost += cost_this_path
                    repeater_nodes_used.append(r_ip)
            else:
                elementary_links_current_pair = []
                [elementary_links_current_pair.append(tup) for tup in self.x_variables_chosen if tup[0] == q]
                # Remove cyclic paths from solution
                for el1 in elementary_links_current_pair:
                    for el2 in elementary_links_current_pair:
                        if el1[1][-1] == el2[1][0] and el1[1][0] == el2[1][-1]:
                            # print("Removing cyclic path: ", el1, el2)
                            elementary_links_current_pair.remove(el1)
                            elementary_links_current_pair.remove(el2)
                # print(q, elementary_links_current_pair)
                num_el = len(elementary_links_current_pair)
                tot_num_el += num_el
                for _ in range(self.program.k):
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
                local_dict = pair_dict[(q[0], q[1])]
                local_dict['paths'] = paths
                local_dict['num_el_used'] = num_el_used
                local_dict['repeater_nodes_used'] = repeater_nodes_used
                local_dict['path_cost'] = cost_per_path

        used_elementary_links = []
        used_edges = []
        visited_nodes = []

        for tup in self.x_variables_chosen:
            elementary_link_path = tup[1]
            used_elementary_links.append((elementary_link_path[0], elementary_link_path[-1]))
            visited_nodes.extend(elementary_link_path)
            for i in range(len(elementary_link_path) - 1):
                used_edges.append((elementary_link_path[i], elementary_link_path[i+1]))
        self.visited_nodes = list(set(visited_nodes))
        self.link_extension_nodes = list(set([i for i in visited_nodes if i not in self.repeater_nodes_chosen
                                              and i not in self.program.graph_container.end_nodes]))
        self.used_elementary_links = used_elementary_links
        self.used_edges = list(set(used_edges))
        self.unused_edges = list(self.program.graph_container.graph.edges())
        for tup in self.used_edges:
            if tup in self.unused_edges:
                self.unused_edges.remove(tup)
            elif tuple(reversed(tup)) in self.unused_edges:
                self.unused_edges.remove(tuple(reversed(tup)))

        self.overall_data['tot_path_cost'] = round(total_cost, 3)
        self.overall_data['avg_path_len'] = round(total_cost / self.program.graph_container.num_unique_pairs)
        self.overall_data['tot_num_el'] = tot_num_el
        self.overall_data['repeater_node_degree'] = repeater_node_degree

        return pair_dict

    def _create_virtual_solution_graph(self):
        """Create a virtual graph based on the solution where the edges are elementary links."""
        pos = nx.get_node_attributes(self.program.graph_container.graph, 'pos')
        self.virtual_solution_graph = nx.Graph()
        self.virtual_solution_graph.add_nodes_from(self.program.graph_container.end_nodes + self.repeater_nodes_chosen)
        nx.set_node_attributes(self.virtual_solution_graph, pos, name='pos')
        self.virtual_solution_graph.add_edges_from(self.used_elementary_links)

    def get_status_string(self):
        return self.program.prob.solution.get_status_string()

    def get_parameters(self):
        return self.parameters

    def get_solution_data(self):
        if "min_node_connectivity" not in self.overall_data.keys():
            min_node_connectivity, avg_node_connectivity = self.compute_node_connectivy()
            min_edge_connectivity, avg_edge_connectivity = self.compute_edge_connectivity()
            self.overall_data.update({"min_node_connectivity": min_node_connectivity,
                                      "avg_node_connectivity": avg_node_connectivity,
                                      "min_edge_connectivity": min_edge_connectivity,
                                      "avg_edge_connectivity": avg_edge_connectivity})
        return self.overall_data

    def get_path_data(self):
        return self.path_data

    def print_path_data(self):
        for k in range(self.program.k):
            for key in self.path_data:
                print("k = {}, q = {}: path = {}, num_el = {}, reps = {}, path_cost = {}".format(k, key,
                      self.path_data[key]['paths'][k],
                      self.path_data[key]['num_el_used'][k],
                      self.path_data[key]['repeater_nodes_used'][k],
                      self.path_data[key]['path_cost'][k]))

    def draw_virtual_solution_graph(self):
        pos = nx.get_node_attributes(self.virtual_solution_graph, 'pos')
        # Create blank figure
        fig, ax = plt.subplots(figsize=(10, 7))
        # First draw the end nodes
        end_nodes = nx.draw_networkx_nodes(G=self.virtual_solution_graph, pos=pos, node_size=700,
                                           nodelist=self.program.graph_container.end_nodes,
                                           node_shape='s', node_color=[[1.0, 140 / 255, 0.]], label="End Node")
        end_nodes.set_edgecolor('k')
        # Then draw the repeater nodes
        if self.repeater_nodes_chosen:
            rep_nodes = nx.draw_networkx_nodes(G=self.virtual_solution_graph, pos=pos, node_size=700,
                                               nodelist=self.repeater_nodes_chosen,
                                               node_color=[[0 / 255, 166 / 255, 214 / 255]], label="Repeater Node")
            rep_nodes.set_edgecolor('k')
        # Finally draw the elementary links
        nx.draw_networkx_edges(G=self.virtual_solution_graph, pos=pos, edgelist=self.used_elementary_links, width=3)
        # And draw in the labels of the nodes
        nx.draw_networkx_labels(G=self.virtual_solution_graph, pos=pos, font_weight='bold')
        # Change some margins etc
        plt.axis('off')
        margin = 0.33
        fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        ax.axis('equal')
        fig.tight_layout()
        # Show the plot
        plt.show()

    def draw_physical_solution_graph(self):
        pos = nx.get_node_attributes(self.program.graph_container.graph, 'pos')
        labels = {}
        for node, nodedata in self.program.graph_container.graph.nodes.items():
            # labels[node] = node
            if node in self.program.graph_container.end_nodes: # or node in self.repeater_nodes_chosen:
                labels[node] = node
            else:
                labels[node] = ""
        # Empty figure
        fig, ax = plt.subplots(figsize=(7, 7))
        # First draw end nodes
        end_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos, node_size=1500,
                                           nodelist=self.program.graph_container.end_nodes,
                                           node_shape='s', node_color=[[255 / 255, 120 / 255, 0 / 255]], label="End Node",
                                           linewidths=3)  # [[1.0, 140 / 255, 0.]]
        end_nodes.set_edgecolor('k')
        # Then draw the repeater nodes
        if self.repeater_nodes_chosen:
            rep_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos, node_size=1500, node_shape='h',
                                               nodelist=self.repeater_nodes_chosen, node_color=[[0 / 255, 166 / 255, 214 / 255]],
                                               label="Repeater Node", linewidths=3)
            rep_nodes.set_edgecolor('k')
        # Then draw the link-extension nodes
        if self.link_extension_nodes:
            le_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos, node_size=1500,
                                              nodelist=self.link_extension_nodes, node_color=[[1, 1, 1]],
                                              label="Link Extension")
            le_nodes.set_edgecolor('k')
        # Also draw all the unused nodes
        unused_nodes = []
        [unused_nodes.append(n) for n in self.program.graph_container.graph.nodes() if
         (n not in self.link_extension_nodes and n not in self.repeater_nodes_chosen
          and n not in self.program.graph_container.end_nodes)]
        if unused_nodes:
            unu_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos, node_size=1500,
                                               nodelist=unused_nodes, node_color=[[1, 1, 1]])
            unu_nodes.set_edgecolor('k')
        # Draw the unused edges but less visible
        nx.draw_networkx_edges(G=self.program.graph_container.graph, pos=pos, edgelist=self.unused_edges,
                               edge_color=[[170 / 255, 170 / 255, 170 / 255]], width=1)
        # Draw the used edges
        nx.draw_networkx_edges(G=self.program.graph_container.graph, pos=pos, edgelist=self.used_edges, width=8,
                               edge_color=[[0 / 255, 166 / 255, 214 / 255]])
        # Draw all the node labels
        nx.draw_networkx_labels(G=self.program.graph_container.graph, pos=pos, labels=labels, font_size=30,
                                font_weight="bold", font_color="w", font_family='serif')

        # nx.draw(self.graph_container.graph, labels=labels, with_labels=True, font_weight='bold',
        #        pos=pos, node_color=color_map, node_size=200)
        plt.axis('off')
        margin = 0.33
        fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        ax.axis('equal')
        fig.tight_layout()
        # plt.legend(loc='center left', fontsize=13, prop={'size': 20})
        # plt.savefig("/mnt/c/Users/Julian/OneDrive/Master AP/Thesis/Paper Repeater Placement/example_square_k3_new.eps")
        plt.show()

    def compute_node_connectivy(self):
        connectivity_dict = nx.all_pairs_node_connectivity(self.virtual_solution_graph,
                                                           self.program.graph_container.end_nodes)
        min_connectivity = 1e20
        total_connectivity = 0
        for key, val in connectivity_dict.items():
            vals = list(val.values())
            total_connectivity += sum(vals)
            if min(vals) < min_connectivity:
                min_connectivity = min(vals)
        # Divide by 2 times the unique (s,t) pairs, since connectivity is computed in both directions
        avg_connectivity = total_connectivity / (2 * self.program.graph_container.num_unique_pairs)
        return min_connectivity, avg_connectivity

    def compute_edge_connectivity(self):
        total_edge_connectivity = 0
        minimum_edge_connectivity = 1e20
        for q in self.program.graph_container.unique_end_node_pairs:
            edge_connectivity_this_pair = nx.edge_connectivity(G=self.virtual_solution_graph, s=q[0], t=q[1])
            total_edge_connectivity += edge_connectivity_this_pair
            if edge_connectivity_this_pair < minimum_edge_connectivity:
                minimum_edge_connectivity = edge_connectivity_this_pair
        avg_edge_connectivity = total_edge_connectivity / self.program.graph_container.num_unique_pairs
        return minimum_edge_connectivity, avg_edge_connectivity

    def plot_degree_histogram(self):
        degree_sequence = sorted([d for n, d in self.program.graph_container.graph.degree()], reverse=True)
        degree_count = collections.Counter(degree_sequence)
        deg, _ = zip(*degree_count.items())
        mean = np.mean(degree_sequence)
        sigma = np.std(degree_sequence)
        x = np.linspace(min(degree_sequence), max(degree_sequence), 100)

        fig, ax = plt.subplots()
        hist = plt.hist(degree_sequence, width=0.8, color='b')
        dx = hist[1][1] - hist[1][0]
        plt.plot(x, stats.norm.pdf(x, mean, sigma) * len(degree_sequence) * dx, '--', color='g')

        # plt.title("Degree Histogram")
        plt.ylabel("Count", fontsize=18)
        plt.xlabel("Degree", fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        ax.set_xticks([d for d in deg])
        ax.set_xticklabels(deg)
        plt.show()
