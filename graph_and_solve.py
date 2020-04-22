from programs import EdgeDisjointLinkBasedProgram, EdgeDisjointPathBasedProgram,\
    NodeDisjointPathBasedProgram, NodeDisjointLinkBasedProgram
from graph_tools import GraphContainer, create_graph_and_partition, read_graph, create_graph_on_unit_cube
from solution import Solution

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_connectivity_and_num_reps(res, xdata, xlabel, L_max, R_max, k, D, n):
    """Function for plotting the connectivity and number of repeaters for different values of k or D."""
    # Plotting
    overall_fs = 18
    # Connectivity on left y-axis
    plt.figure(figsize=(11, 7))
    plt.xticks(fontsize=overall_fs)
    plt.yticks(fontsize=overall_fs)
    ax = plt.gca()
    ax.set_ylabel('Number of repeaters', fontsize=overall_fs)
    ax.set_xlabel(xlabel, fontsize=overall_fs)
    ax.set_ylim(ymin=1, ymax=18)
    ax2 = ax.twinx()
    ax2.set_ylabel('Connectivity', fontsize=overall_fs)
    ax2.set_ylim(ymin=1, ymax=14)
    plt.yticks(fontsize=overall_fs)
    num_graphs = len(list(res.values())[0][0])
    colors = ['red', 'orange', 'green', 'blue', 'black']
    linestyles = ['--', '-', '-', '-', '-']
    h = []
    for i in range(5):
        y_data = []
        std_y_data = []
        for val in res.values():
            y_data.append(np.mean(val[i]))
            std_y_data.append(np.std(val[i]))
        # Convert to standard deviation of the mean
        # std_y_data /= np.sqrt(num_graphs)
        if i > 0:
            ax = ax2
        h.append(ax.errorbar(xdata, y_data, marker='.', markersize=15, yerr=std_y_data, linewidth=3,
                             linestyle=linestyles[i], color=colors[i]))
    plot_tuple = (h[0], h[1], h[2])
    plt.legend(plot_tuple, ('Number of Repeaters', 'Minimum Node Connectivity', 'Minimum Edge Connectivity'),
               fontsize=overall_fs, loc='best')
    if xlabel == 'k':
        plt.title('n = {}, L_max = {}, R_max = {}, D = {}, num_feas_graphs = {}'.format(n, L_max, R_max, D,
                                                                                        num_graphs),
                  fontsize=overall_fs)
    elif xlabel == 'D':
        plt.title('n = {}, L_max = {}, R_max = {}, k = {}, num_feas_graphs = {}'.format(n, L_max, R_max, k,
                                                                                        num_graphs),
                  fontsize=overall_fs)
    plt.show()


if __name__ == "__main__":
    """Script to run, added a bunch of examples so you can have a look how things work."""
    # with open('./comp_times_2020-04-10_10-07-59.txt', 'r') as f:
    #     ct = eval(f.read())
    # print(ct)
    # for n in ct:
    #     print(n, np.mean(ct[n]), np.std(ct[n]))
    # comp_times = {}
    # num_graphs = 100
    # for n in range(10, 111, 10):
    #     num_suc = 0
    #     comp_times[n] = []
    #     while num_suc < num_graphs:
    #         seed = np.random.randint(0, 1e5)
    #         print(n, seed)
    #         G = create_graph_and_partition(num_nodes=n, radius=0.9, seed=seed)
    #         prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=1, R_max=6, D=8, k=2, alpha=0)
    #         sol, comp_time = prog.solve()
    #         # Only add feasible solutions to computation time data
    #         if 'infeasible' not in sol.get_status_string():
    #             print(comp_time)
    #             comp_times[n].append(comp_time)
    #             num_suc += 1
    # Save computation time data in current folder
    # now = str(datetime.now())[0:-7].replace(" ", "_").replace(":", "-")
    # with open('./comp_times_{}.txt'.format(now), 'w') as f:
    #     print(comp_times, file=f)

    """Read the Colt data set (European data) and plot the solution"""
    # G = read_graph('Colt.gml', draw=False)
    # prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=900, R_max=6, D=1e20, k=1,
    #                                     alpha=1/75000)
    # sol, comp_time = prog.solve()
    # print("Computation Time:", comp_time)
    # sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()

    """Create a random graph with 4 fixed end nodes on the vertices of a unit cube and 10 repeater nodes."""
    G = create_graph_on_unit_cube(n_repeaters=10, radius=0.6, draw=True, seed=9)
    # prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=0.75, R_max=3, D=30, k=1, alpha=1/100)
    prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=0.9, R_max=3, D=6, k=3,
                                        alpha=0)
    sol, _ = prog.solve()
    # sol.draw_virtual_solution_graph()
    sol.draw_physical_solution_graph()

    """Create random graph and use the convex hull to partition the nodes"""
    # G = create_graph_and_partition(num_nodes=25, radius=0.5, seed=130, draw=True)
    # prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=0.9, R_max=3, D=30, k=1, alpha=1/200)
    # sol, _ = prog.solve()
    # sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()
    # sol.print_path_data()

    """Create random instances and compare the solutions of the LBF and PBF"""
    # for _ in range(100):
    #     D = np.random.randint(5, 15)
    #     k = np.random.randint(1, 5)
    #     L_max = round(np.random.rand() + 0.5, 5)
    #     R_max = np.random.randint(1, 4)
    #     n = np.random.randint(15, 25)
    #     seed = np.random.randint(1, 1e5)
    #     alpha = 1 / 250
    #     print(D, k, L_max, R_max, n, seed)
    #     G = create_graph_and_partition(num_nodes=n, radius=0.7, draw=False, seed=seed)
    #     prog_LBF = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=D, k=k,
    #                                             alpha=alpha)
    #     sol_LBF, _ = prog_LBF.solve()
    #     data_LBF = sol_LBF.get_solution_data()
    #     prog_PBF = NodeDisjointPathBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=D, k=k,
    #                                             alpha=alpha)
    #     sol_PBF, _ = prog_PBF.solve()
    #     data_PBF = sol_PBF.get_solution_data()
    #     if data_LBF != data_PBF:
    #         print("LBF and PBF give different solutions!")
    #         print("LBF:", data_LBF)
    #         print("PBF:", data_PBF)
    #         print("D = {}, k = {}, L_max = {}, R_max = {}, n = {}, seed = {}, alpha = {}"
    #               .format(D, k, L_max, R_max, n, seed, alpha))
    #         break
    #     else:
    #         print("Same solutions.")

    """Generate data for the connectivity by generating random geometric graphs"""
    # Open a text file with saved data if necessary
    # with open('./data/res_k_2020-02-27_04-03-59.txt', 'r') as f:
    #     res = eval(f.read())
    # Loop over values of k
    # max_k = 6
    # k_vals = list(range(1, max_k + 1))
    # num_suc_max = 50  # Number of feasible graphs we want to extract data from
    # # TODO: fix this such that the same N graphs are used and infeasible data is not thrown away
    # n = 25  # Number of nodes *in total*. Note that the number of (s,t)-pairs differs based on the convex hull
    # L_max = 0.9
    # R_max = 5
    # D = 8
    # alpha = 1 / 250
    # changing_var = 'k'
    # res = {}
    # save_data = True
    # rep_node_degree_data = []
    # for k in k_vals:
    #     min_node_con = []
    #     avg_node_con = []
    #     min_edge_con = []
    #     avg_edge_con = []
    #     num_reps = []
    #     num_suc = 0
    #     while num_suc < num_suc_max:
    #         G = create_graph_and_partition(num_nodes=n, radius=0.7, draw=False)
    #         prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), D=D, k=k, alpha=alpha, L_max=L_max,
    #                                             R_max=R_max)
    #         sol, _ = prog.solve()
    #         if 'infeasible' in sol.get_status_string():
    #             continue
    #         print("num_suc = {}, k = {}, D = {}".format(num_suc, k, D))
    #         print(sol.get_solution_data())
    #         rep_node_degree_data.append(sol.get_solution_data()['repeater_node_degree'])
    #         print(rep_node_degree_data)
    #         # sol.draw_virtual_solution_graph()
    #
    #         min_node_connectivity, avg_node_connectivity = sol.compute_node_connectivy()
    #         min_edge_connectivity, avg_edge_connectivity = sol.compute_edge_connectivity()
    #         num_reps.append(sol.get_solution_data()['num_reps'])
    #         min_node_con.append(min_node_connectivity)
    #         avg_node_con.append(avg_node_connectivity)
    #         min_edge_con.append(min_edge_connectivity)
    #         avg_edge_con.append(avg_edge_connectivity)
    #         num_suc += 1
    #     res[k] = (num_reps, min_node_con, avg_node_con, min_edge_con, avg_edge_con)
    # sol.plot_degree_histogram()
    # if save_data:
    #     now = str(datetime.now())[0:-7].replace(" ", "_").replace(":", "-")
    #     with open('./data/res_{}_{}.txt'.format(changing_var, now), 'w') as f:
    #         print(res, file=f)
    # plot_connectivity_and_num_reps(res=res, xdata=k_vals, xlabel=changing_var, L_max=L_max, R_max=R_max,
    #                                n=n, D=8, k=2)

