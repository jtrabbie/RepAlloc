from determine_Lmax_Rmax import max_length_and_rate
from programs import NodeDisjointLinkBasedProgram
from graph_tools import read_graph, GraphContainer
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Read the Surfnet data set (Dutch data) and plot the solution.
    
    """

    # G = nx.read_gml("SurfnetCoreTest.gml")
    # pos = {}
    # x_coordinate = 10
    # for node in G.nodes:
    #     pos[node] = [x_coordinate, 5]
    #     x_coordinate = x_coordinate + 1
    # nx.draw_networkx_nodes(G, pos)
    # plt.show()
    # for node, nodedata in G.nodes.items():
    #     print(node)
    #     print(nodedata)
    target_fidelity = 0.93
    target_rate = 1  # Hertz
    L_max, R_max = max_length_and_rate(target_fidelity=0.93,
                                       target_rate=1,
                                       elementary_link_fidelity=0.99,
                                       number_of_modes=1000,
                                       swap_probability=.5)
    G = read_graph('SurfnetCore.gml', draw=False)
    prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=L_max, R_max=R_max, D=4, k=2,
                                        alpha=1 / 75000)
    sol, comp_time = prog.solve()
    print("Computation Time:", comp_time)
    sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()