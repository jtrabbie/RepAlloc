from formulations import PathBasedFormulation, LinkBasedFormulation
from graph_tools import GraphContainer, create_graph_and_partition, read_graph_from_gml, create_graph_on_unit_cube
from determine_Lmax_Rmax import max_length_and_rate
import numpy as np


def solve_from_gml(filename, L_max, N_max, D, K, alpha):
    """Read a data set from gml file and plot the solution. Only supports 'Surfnet.gml' or 'Colt.gml'."""
    G = read_graph_from_gml(filename, draw=False)
    prog = LinkBasedFormulation(graph_container=GraphContainer(G), L_max=L_max, N_max=N_max, D=D, K=K, alpha=alpha)
    sol, comp_time = prog.solve()
    print("Computation Time:", comp_time)
    print(sol.get_solution_data())
    print(sol.get_parameters())
    sol.print_path_data()
    # sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()


def solve_on_unit_cube(L_max, N_max, D, K):
    """Create a random graph with 4 fixed end nodes on the vertices of a unit cube and 10 repeater nodes."""
    G = create_graph_on_unit_cube(n_repeaters=10, radius=0.6, draw=False, seed=9)
    prog = LinkBasedFormulation(graph_container=GraphContainer(G), L_max=L_max, N_max=N_max, D=D, K=K, alpha=0)
    sol, _ = prog.solve()
    # sol.draw_virtual_solution_graph()
    sol.draw_physical_solution_graph()
    # sol.print_path_data()


def solve_with_random_graph(num_nodes, radius, L_max, N_max, D, K, alpha):
    """Create random graph and use the convex hull to partition the nodes."""
    G = create_graph_and_partition(num_nodes=num_nodes, radius=radius, draw=True)
    prog = LinkBasedFormulation(graph_container=GraphContainer(G), L_max=L_max, N_max=N_max, D=D, K=K, alpha=alpha)
    sol, _ = prog.solve()
    sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()
    # sol.print_path_data()


def compare_formulations():
    """Create random instances and compare the solutions of the link-based formulation and path-based formulation.
    Note that we must use a non-zero value of alpha since the solutions would otherwise be degenerate."""
    for _ in range(100):
        D = np.random.randint(5, 15)
        K = np.random.randint(1, 5)
        L_max = round(np.random.rand() + 0.5, 5)
        N_max = np.random.randint(1, 4)
        n = np.random.randint(15, 25)
        seed = np.random.randint(1, 1e5)
        alpha = 1 / 100
        print(D, K, L_max, N_max, n, seed)
        G = create_graph_and_partition(num_nodes=n, radius=0.7, draw=True, seed=seed)
        link_based_form = LinkBasedFormulation(graph_container=GraphContainer(G), L_max=L_max, N_max=N_max, D=D, K=K,
                                               alpha=alpha)
        sol_LBF, _ = link_based_form.solve()
        data_LBF = sol_LBF.get_solution_data()
        path_based_form = PathBasedFormulation(graph_container=GraphContainer(G), L_max=L_max, N_max=N_max, D=D, K=K,
                                               alpha=alpha)
        sol_PBF, _ = path_based_form.solve()
        data_PBF = sol_PBF.get_solution_data()
        if data_LBF != data_PBF:
            print("LBF and PBF give different solutions! (This should never happen as they must be equivalent)")
            print("LBF:", data_LBF)
            print(sol_LBF.print_path_data())
            print("PBF:", data_PBF)
            print(sol_PBF.print_path_data())
            print("D = {}, K = {}, L_max = {}, N_max = {}, n = {}, seed = {}, alpha = {}"
                  .format(D, K, L_max, N_max, n, seed, alpha))
            break
        else:
            # Note that we must explicitly clear the formulation to remove the reference to the CPLEX object in order
            # to avoid RAM issues.
            link_based_form.clear()
            path_based_form.clear()
            print("Same solutions.")


def surfnet_solve():
    """Read the Surfnet core data set (Dutch fiber infrastructure) and plot the solution."""
    L_max, N_max = max_length_and_rate(target_fidelity=0.93,
                                       target_rate=1,
                                       elementary_link_fidelity=0.99,
                                       number_of_modes=1000,
                                       swap_probability=.5)
    G = read_graph_from_gml('SurfnetCore.gml', draw=False)
    prog = LinkBasedFormulation(graph_container=GraphContainer(G), L_max=L_max, N_max=N_max, D=4, K=2,
                                alpha=1 / 75000)
    sol, comp_time = prog.solve()
    print("Computation Time:", comp_time)
    sol.draw_physical_solution_graph()
    # sol.draw_virtual_solution_graph()


if __name__ == "__main__":
    pass
    surfnet_solve()
    # solve_from_gml("Surfnet.gml", L_max=60, N_max=11, D=100, K=1, alpha=1 / 2500)
    # solve_on_unit_cube(L_max=0.9, N_max=3, D=6, K=1)
    # solve_with_random_graph()
    # compare_formulations()
