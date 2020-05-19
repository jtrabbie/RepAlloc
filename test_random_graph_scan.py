from random_graph_scan import generate_feasible_graph, generate_feasible_graphs, solve_graphs
import numpy as np


params = {"num_nodes": 30,
          "radius": np.sqrt(2),
          "alpha": 0,
          "L_max": np.sqrt(2),
          "N_max": 100,
          "D": 1000,
          "K": 1}


def test_feasible_graph():
    graph_container, solution, computation_time = generate_feasible_graph(**params)
    assert isinstance(computation_time, float)


def test_feasible_graphs():
    graph_containers, solutions = generate_feasible_graphs(num_graphs=5, **params)
    assert len(graph_containers) == len(solutions)
    assert len(graph_containers) == 5


def test_solve_graphs():
    graph_containers, solutions = generate_feasible_graphs(num_graphs=5, **params)
    new_solutions = solve_graphs(graph_containers)
    assert len(new_solutions) == len(solutions)
    for sol, new_sol in zip(solutions, new_solutions):
        assert sol == new_sol
