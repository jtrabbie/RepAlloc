from programs import LinkBasedProgram, PathBasedProgram
from graph_tools import GraphContainer, create_graph_and_partition
from solution import Solution

import numpy as np
from datetime import datetime

if __name__ == "__main__":
    for _ in range(500):
        D = np.random.randint(1, 10)
        k = np.random.randint(1, 5)
        L_max = round(np.random.rand() + 0.5, 5)
        R_max = np.random.randint(1, 4)
        n = np.random.randint(10, 20)
        seed = np.random.randint(1, 1e5)
        alpha = 1 / 250
        print(D, k, L_max, R_max, n, seed)
        G = create_graph_and_partition(num_nodes=n, radius=0.7, draw=False, seed=seed)
        prog_LBF = LinkBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=D, k=k, alpha=alpha)
        sol_LBF, _ = prog_LBF.solve()
        data_LBF = sol_LBF.process()
        # print(data_LBF)
        #sol_LBF.draw()
        prog_PBF = PathBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=D, k=k, alpha=alpha)
        sol_PBF, _ = prog_PBF.solve()
        data_PBF = sol_PBF.process()
        # print(data_PBF)
        # sol_PBF.draw()
        # prog_PBF = PathBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=D, k=k, alpha=alpha)
        # sol_PBF, _ = prog_PBF.solve()
        # data_PBF = sol_PBF.process()
        # print(data_PBF)
        if data_LBF != data_PBF:
            print("LBF and PBF give different solutions!")
            print("LBF:", data_LBF)
            print("PBF:", data_PBF)
            print("D = {}, k = {}, L_max = {}, R_max = {}, n = {}, seed = {}, alpha = {}"
                  .format(D, k, L_max, R_max, n, seed, alpha))
            break
        else:
            print("Same solutions.")
        # prog = LinkBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=D, k=k, alpha=1/1000)
        # sol, _ = prog.solve()
        # data2 = sol.process()
        # print(data2)
        # if data1 != data2:
        #     print("LBF and PBF give different solutions!")
        #     print("PBF: ", data1)
        #     print("LBF: ", data2)
        #     print(D, L_max, R_max, n, seed)
        # else:
        #     print("Same solutions.")

    # with open('res_2020-02-25_11-51-30.txt', 'r') as f:
    #     res = eval(f.read())
    # max_k = 6
    # k_vals = list(range(1, max_k + 1))
    # n = 20
    # alpha = 1 / 250
    # num_suc_max = 10
    # res = {}
    # save_data = True
    # for k in k_vals:
    #     con = []
    #     num_reps = []
    #     num_suc = 0
    #     while num_suc < num_suc_max:
    #         G = create_graph_and_partition(num_nodes=n, radius=0.7, draw=False)
    #         prog = RevisedLinkBasedProgram(graph_container=GraphContainer(G), D=k, k=1, alpha=alpha)
    #         sol, _ = prog.solve()
    #         data = sol.process()
    #         avg_connectivity = sol.compute_average_connectivy()
    #         if avg_connectivity == 0:
    #             # Solution is infeasible, do not take this into account
    #             continue
    #         else:
    #             num_reps.append(data['Num_reps'])
    #             con.append(avg_connectivity)
    #             num_suc += 1
    #     res[k] = (con, num_reps)
    # # sol.plot_degree_histogram()
    # if save_data:
    #     now = str(datetime.now())[0:-7].replace(" ", "_").replace(":", "-")
    #     with open('res_{}.txt'.format(now), 'w') as f:
    #         print(res, file=f)
    # Solution.plot_connectivity_and_num_reps(res=res, xdata=k_vals, xlabel='D')


    # with open('res_2020-02-17_14-27-59.txt', 'r') as f:
    #     res = eval(f.read())
    # max_k = 6
    # k_vals = list(range(1, max_k + 1))
    # num_suc_max = 10
    # res = {}
    # save_data = True
    # comp_times_dict = {}
    # num_node_vals = list(range(10, 50))
    # for num_nodes in num_node_vals:
    #     num_suc = 0
    #     comp_times = []
    #     while num_suc < num_suc_max:
    #         G = create_graph_and_partition(num_nodes=num_nodes, radius=0.7)
    #         if G is None:
    #             # Graph contains isolates
    #             continue
    #         prog = LinkBasedProgram(graph=G, L_max=L_max, k=2, alpha=1/1000)
    #         if prog.graph_container.num_repeater_nodes == 0:
    #             # Every node is selected as end node by the convex hull
    #             continue
    #         sol, comp_time = prog.solve()
    #         sol.process()
    #         if 'infeasible' in sol.program.prob.solution.get_status_string():
    #             # Solution is infeasible, do not take this into account
    #             continue
    #         else:
    #             comp_times.append(comp_time)
    #             num_suc += 1
    #     comp_times_dict[num_nodes] = comp_times
    # if save_data:
    #     now = str(datetime.now())[0:-7].replace(" ", "_").replace(":", "-")
    #     with open('comp_times_{}.txt'.format(now), 'w') as f:
    #         print(comp_times_dict, file=f)
    # y_comp_time = []
    # std_y_comp_time = []
    # num_graphs = num_suc_max
    # for val in comp_times_dict.values():
    #     y_comp_time.append(np.mean(val))
    #     std_y_comp_time.append(np.std(val))
    # # Convert to standard deviation of the mean
    # std_y_comp_time /= np.sqrt(num_graphs)
    # ax = plt.gca()
    # overall_fs = 18
    # plt.xticks(fontsize=overall_fs)
    # plt.yticks(fontsize=overall_fs)
    # print(y_comp_time)
    # h1 = ax.errorbar(num_node_vals, y_comp_time, marker='.', markersize=15, yerr=std_y_comp_time,
    #                  linestyle='None', linewidth=3, color='g')
    # # fit = np.poly1d(np.polyfit(xdata, y_con, deg=1))
    # # xfit = np.linspace(1, max(xdata), 10)
    # # h2, = ax.plot(xfit, fit(xfit), '--', linewidth=3)
    # ax.set_xlabel('n', fontsize=overall_fs)
    # ax.set_ylabel('Computation time (s)', fontsize=overall_fs)
    # # plt.legend((h1, h2, h3), ('Average Connectivity', 'Linear Fit', 'Number of Repeaters'), fontsize=overall_fs,
    # #           loc='center right')
    # plt.show()

    # for k in k_vals:
    #     con = []
    #     num_reps = []
    #     num_suc = 0
    #     while num_suc < num_suc_max:
    #         G = create_graph_and_partition(num_nodes=15, radius=0.7, draw=False)
    #         prog = LinkBasedProgram(graph=G, L_max=L_max, k=k, alpha=1/1000)
    #         sol, _ = prog.solve()
    #         data = sol.process()
    #         avg_connectivity = sol.compute_average_connectivy()
    #         if avg_connectivity == 0:
    #             # Solution is infeasible, do not take this into account
    #             continue
    #         else:
    #             num_reps.append(data['Num_reps'])
    #             con.append(avg_connectivity)
    #             num_suc += 1
    #     res[k] = (con, num_reps)
    # sol.plot_degree_histogram()
    # sol.plot_connectivity_and_num_reps(res=res, xdata=k_vals, xlabel='k', )

    # G = create_graph(node_pos=node_pos, draw=False)
    # filename = "Surfnet.gml"
    # G = read_graph(filename, draw=False)
    # R_max = 11
    # L_max = 10
    # prog = LinkBasedProgram(graph=G, num_allowed_repeaters=R_max, L_max=L_max, alpha=1, read_from_file=False)
    # prog = MinRepEdgeBasedProgram(graph=G, num_allowed_repeaters=R_max, L_max=L_max, alpha=0, read_from_file=True)
    # prog.draw_fixed_solution()
    # prog.update_parameters(L_max_new=1000, R_max_new=6, alpha_new=1/75000)
    # print(prog.solve(draw_solution=True))
    # dict_list = []
    # for alpha in [1 / 75000]:
    #     for L_max in [900]:
    #         for R_max in [6]:
    #             prog.update_parameters(L_max_new=L_max, R_max_new=R_max, alpha_new=alpha)
    #             prog.draw_fixed_solution()
    #             output = prog.solve()
    #             print(output)
    #             dict_list.append(output)
    #final_dict = {k: [d[k] for d in dict_list] for k in dict_list[0]}
    #now = str(datetime.now())[0:-7].replace(" ", "_").replace(":", "-")
    #pd.DataFrame(final_dict).to_csv('./data/repAllocData_{}_{}.csv'.format(filename[0:-4], now), index=False)
    # print(prog.solve())
    # prog.update_parameters(L_max_new=100, R_max_new=10, alpha_new=1/2500)
    # print(prog.solve())
    # for L_max in range(65, 105, 5):
    #     for R_max in range(10, 11):
    #         prog.update_lmax_rmax(L_max_new=L_max, R_max_new=R_max)
    #         # print("Running program for L_max = {} and R_max = {}".format(L_max, R_max))
    #         print(prog.solve())

    # prog = EdgeBasedProgram(G, R_max)
    # prog = MinMaxEdgeBasedProgram(G, R_max)
    # prog = PathBasedProgram(G, R_max)


