import cplex
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
import scipy.stats as stats

# from link_based_program import LinkBasedProgram
# from path_based_program import PathBasedProgram
# from programs import Program


class Solution:
    def __init__(self, program):
        self.program = program
        self.repeater_nodes = []
        self.chosen_paths = []
        self.visited_nodes = []
        self.used_edges = []
        self.unused_edges = []

    def process(self):
        sol_status = self.program.prob.solution.get_status_string()
        if 'infeasible' in sol_status:
            print("Solution is infeasible!")
            data = {"L_max": self.program.L_max,
                    "R_max": self.program.R_max,
                    "k": self.program.k,
                    "D": self.program.D,
                    "alpha": self.program.alpha,
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
        for i, x in enumerate(self.program.prob.solution.get_values()):
            if x > 1e-5:
                var_name = self.program.prob.variables.get_names(i)
                if var_name[0:2] == "y_" and '#' not in var_name:
                    repeater_nodes.append(var_name[2:])
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
                    chosen_paths.append(path_tuple)
        opt_obj_val = round(self.program.prob.solution.get_objective_value(), 5)
        if opt_obj_val - len(repeater_nodes) > 1:
            print("WARNING: value of alpha too large, influenced objective function! Optimal objective value: {},"
                  "number of repeaters: {}".format(opt_obj_val, len(repeater_nodes)))
        # if len(repeater_nodes) > 0:
        #     print("{} Repeater(s) chosen: {}".format(len(repeater_nodes), repeater_nodes))
        repeater_node_degree = {k: 0 for k in repeater_nodes}
        total_cost = 0
        tot_num_el = 0
        for q in self.program.graph_container.unique_end_node_pairs:
            cost = 0
            paths = []
            repeater_nodes_this_path = []
            if "PathBasedProgram" in self.program.__class__.__name__:
                path_properties = []
                [path_properties.append(tup) for tup in chosen_paths if tup[0] == q]
                for k in range(self.program.k):
                    r_ip = path_properties[k][2]
                    for i in r_ip:
                        repeater_node_degree[i] += 1
                        if i not in repeater_nodes:
                            print("Warning! r_ip = {} but corresponding y_i is not 1 (this should never happen)")
                    paths.append(path_properties[k][1])
                    tot_num_el += (len(r_ip) + 1)
                    cost += path_properties[k][3]
                    repeater_nodes_this_path.append(r_ip)
                    # print(q, path_properties[i][1], r_ip, path_properties[i][3])
            else:
                elementary_links_current_pair = []
                [elementary_links_current_pair.append(tup) for tup in chosen_paths if tup[0] == q]
                # Remove cyclic paths from solution
                for el1 in elementary_links_current_pair:
                    for el2 in elementary_links_current_pair:
                        if el1[1][-1] == el2[1][0] and el1[1][0] == el2[1][-1]:
                            # print("Removing cyclic path: ", el1, el2)
                            elementary_links_current_pair.remove(el1)
                            elementary_links_current_pair.remove(el2)
                # print(q, elementary_links_current_pair)
                tot_num_el += len(elementary_links_current_pair)
                for _ in range(self.program.k):
                    path = [None]
                    # print(q, elementary_links_current_pair)
                    old_len_rep_nodes = len(repeater_nodes_this_path)
                    while path[-1] != q[1]:
                        for edge in elementary_links_current_pair:
                            if len(path) == 1 and edge[1][0] == q[0]:  # Initial edge from source
                                path.remove(path[0])
                                path.extend(edge[1])
                                cost += edge[2]
                                elementary_links_current_pair.remove(edge)
                            elif len(path) > 1 and edge[1][0] == path[-1]:  # Extension of current path
                                repeater_nodes_this_path.append(path[-1])
                                repeater_node_degree[path[-1]] += 1
                                path = path[0:-1]
                                path.extend(edge[1])
                                cost += edge[2]
                                elementary_links_current_pair.remove(edge)
                    if len(repeater_nodes_this_path) == old_len_rep_nodes:
                        repeater_nodes_this_path.append([])
                    paths.append(path)
            # print(q, paths, cost, repeater_nodes_this_path)
            total_cost += cost
        # print(repeater_node_degree)
            # print("Optimal path for pair {}: {}, using {} repeaters and cost {}".format(q, path,
            #       len(elementary_links_current_pair) - 1, cost))

        # print("Total number of elementary links:", tot_num_el)

        self.repeater_nodes = repeater_nodes
        self.chosen_paths = chosen_paths
        for tup in self.chosen_paths:
            elementary_link_path = tup[1]
            self.visited_nodes.extend(elementary_link_path)
            for i in range(len(elementary_link_path) - 1):
                self.used_edges.append((elementary_link_path[i], elementary_link_path[i+1]))
        self.visited_nodes = list(set(self.visited_nodes))
        self.used_edges = list(set(self.used_edges))
        self.unused_edges = list(self.program.graph_container.graph.edges())
        for tup in self.used_edges:
            if tup in self.unused_edges:
                self.unused_edges.remove(tup)
            elif tuple(reversed(tup)) in self.unused_edges:
                self.unused_edges.remove(tuple(reversed(tup)))

        data = {"L_max": self.program.L_max,
                "R_max": self.program.R_max,
                "k": self.program.k,
                "D": self.program.D,
                "alpha": self.program.alpha,
                "Opt_obj_val": opt_obj_val,
                "Num_reps": len(repeater_nodes),
                "Tot_path_cost": round(total_cost, 3),
                "Avg_path_len": round(total_cost / self.program.graph_container.num_unique_pairs),
                "Tot_num_el": tot_num_el
                }

        return data

    def draw(self):
        labels = {}
        color_map = []
        rep_nodes_drawing = []
        le_nodes_drawing = []
        pos = nx.get_node_attributes(self.program.graph_container.graph, 'pos')
        for node, nodedata in self.program.graph_container.graph.nodes.items():
            if node in self.program.graph_container.end_nodes:
                if node == "Geneve":
                    labels[node] = "Geneva"
                else:
                    labels[node] = node
                color_map.append('green')
            elif node in self.repeater_nodes:
                labels[node] = node
                rep_nodes_drawing.append(node)
                color_map.append('pink')
            elif node in self.visited_nodes:
                #labels[node] = node
                le_nodes_drawing.append(node)
                color_map.append([30 / 255, 144 / 255, 255 / 255])
            else:
                labels[node] = ""
                color_map.append('none')
        fig, ax = plt.subplots(figsize=(8, 6))

        # First draw end nodes
        end_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos,
                                           nodelist=self.program.graph_container.end_nodes,
                                           node_shape='s', node_color=[[1.0, 140 / 255, 0.]], label="End Node")
        #[[0.66, 0.93, 0.73]] (255, 140, 0)
        end_nodes.set_edgecolor('k')
        if rep_nodes_drawing:
            rep_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos, nodelist=rep_nodes_drawing,
                                               node_color=[[0 / 255, 166 / 255, 214 / 255]], label="Repeater Node")
            #[[30 / 255, 144 / 255, 255 / 255]]
            rep_nodes.set_edgecolor('k')
        if le_nodes_drawing:
            le_nodes = nx.draw_networkx_nodes(G=self.program.graph_container.graph, pos=pos, nodelist=le_nodes_drawing,
                                              node_color=[[0.83, 0.83, 0.83]], label="Link Extension", alpha=0.4)
            le_nodes.set_edgecolor('k')
        nx.draw_networkx_labels(G=self.program.graph_container.graph, pos=pos, labels=labels, font_size=13, font_weight="bold")
        nx.draw_networkx_edges(G=self.program.graph_container.graph, pos=pos, edgelist=self.used_edges, width=3)
        nx.draw_networkx_edges(G=self.program.graph_container.graph, pos=pos, edgelist=self.unused_edges, edge_color="k",
                               width=1, alpha=0.2)

        #nx.draw(self.graph_container.graph, labels=labels, with_labels=True, font_weight='bold',
        #        pos=pos, node_color=color_map, node_size=200)
        #plt.axis('off')
        #margin = 0.33
        #fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        #ax.axis('equal')
        fig.tight_layout()
        plt.legend(loc='upper right', fontsize=13)
        plt.show()

    def compute_average_connectivy(self):
        G_sol = nx.Graph()
        G_sol.add_nodes_from(self.program.graph_container.end_nodes + self.repeater_nodes)
        G_sol.add_edges_from(self.used_edges)
        # fig, ax = plt.subplots(figsize=(8, 6))
        # nx.draw_networkx(G_sol, pos=nx.get_node_attributes(self.program.graph_container.graph, 'pos'))
        # plt.show()
        connectivity_dict = nx.all_pairs_node_connectivity(G_sol, self.program.graph_container.end_nodes)
        total_connectivity = 0
        for key, val in connectivity_dict.items():
            total_connectivity += sum(list(val.values()))
        # Divide by 2 times the unique (s,t) pairs, since connectivity is computed in both directions
        avg_connectivity = total_connectivity / (2 * self.program.graph_container.num_unique_pairs)
        return avg_connectivity

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

    @staticmethod
    def plot_connectivity_and_num_reps(res, xdata, xlabel):
        y_con = []
        std_y_con = []
        num_graphs = len(list(res.values())[0][0])
        for val in res.values():
            y_con.append(np.mean(val[0]))
            std_y_con.append(np.std(val[0]))
        # Convert to standard deviation of the mean
        std_y_con /= np.sqrt(num_graphs)
        y_num_rep = []
        std_y_num_rep = []
        for val in res.values():
            y_num_rep.append(np.mean(val[1]))
            std_y_num_rep.append(np.std(val[1]))
        std_y_num_rep /= np.sqrt(num_graphs)
        # Plotting
        overall_fs = 18
        # Connectivity on left y-axis
        plt.figure(figsize=(9, 7))
        ax = plt.gca()
        ax.set_ylim(ymin=2, ymax=8)
        plt.xticks(fontsize=overall_fs)
        plt.yticks(fontsize=overall_fs)
        h1 = ax.errorbar(xdata, y_con, marker='.', markersize=15, yerr=std_y_con, linestyle='None', linewidth=3,
                         color='g')
        fit = np.poly1d(np.polyfit(xdata, y_con, deg=1))
        xfit = np.linspace(1, max(xdata), 10)
        h2, = ax.plot(xfit, fit(xfit), '--', linewidth=3, color='k')
        ax.set_xlabel(xlabel, fontsize=overall_fs)
        ax.set_ylabel('Average connectivity over all (s,t) pairs', fontsize=overall_fs)
        # Number of repeaters on right y-axis
        ax2 = ax.twinx()
        plt.yticks(fontsize=overall_fs)
        ax2.set_ylabel('Number of repeaters', fontsize=overall_fs)
        h3 = ax2.errorbar(xdata, y_num_rep, marker='s', markersize=7, yerr=std_y_num_rep, linewidth=3,
                          color='r')
        ax2.set_ylim(ymin=-0.4, ymax=7)
        plt.legend((h1, h3), ('Average Connectivity', 'Number of Repeaters'), fontsize=overall_fs,
                   loc='best')
        plt.show()


