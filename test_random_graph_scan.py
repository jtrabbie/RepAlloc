from random_graph_scan import generate_feasible_graph, generate_feasible_graphs, solve_graphs
import numpy as np

setup_params = {"num_nodes": 30,
                "radius": np.sqrt(2)}
solve_params = {"alpha": 0,
                "L_max": np.sqrt(2),
                "N_max": 100,
                "D": 1000,
                "K": 1}


def test_feasible_graph():
    graph_container, solution, computation_time = generate_feasible_graph(**setup_params, **solve_params)
    assert isinstance(computation_time, float)


def test_feasible_graphs():
    graph_containers, solutions, computation_times = generate_feasible_graphs(num_graphs=5, **setup_params,
                                                                              **solve_params)
    assert len(graph_containers) == len(solutions) == len(computation_times)
    assert len(graph_containers) == 5


def test_solve_graphs():
    graph_containers, solutions, computation_times = generate_feasible_graphs(num_graphs=5, **setup_params,
                                                                              **solve_params)
    new_solutions = solve_graphs(graph_containers, **solve_params)
    assert len(new_solutions) == len(solutions)
    for sol, new_sol in zip(solutions, new_solutions):
        assert sol.get_solution_data() == new_sol.get_solution_data()
