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
    G = read_graph('SurfnetCore.gml', draw=True)
    prog = NodeDisjointLinkBasedProgram(graph_container=GraphContainer(G), L_max=200, R_max=6, D=1e20, k=2,
                                        alpha=1 / 75000)
    sol, comp_time = prog.solve()
    print("Computation Time:", comp_time)
    sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()