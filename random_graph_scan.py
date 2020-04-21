from programs import NodeDisjointLinkBasedProgram
from graph_tools import GraphContainer, create_graph_and_partition
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import networkx as nx
from copy import deepcopy
import time


class RandomGraphScan:
    """Data holder for parameter scan over population of random graphs.

    Parameters
    ----------
    scan_param_name : str
        Name of parameter that should be scanned over. Can be "L_max", "R_max", "D" or "k".
    scan_param_min : float
        Minimal value for parameter scan.
    scan_param_max : float
        Maximal value for parameter scan.
    scan_param_step : float
        Step size of parameter scan.
    num_graphs : int
        Number of feasible graphs required.
    num_nodes : int
        Total number of nodes in each random graph.
    radius : float
        Radius of random geometric graphs (all nodes within this distance of one another are connected by an edge).
    alpha : float
        Small number used to set secondary objective.
    L_max : int  # TODO: why can't this be a float?
        Maximum elementary-link length.
    R_max : int
        Maximum number of repeaters on path.
    D : int
        Quantum-repeater capacity.
    k : int
        Robustness parameter.

    Notes
    -----
    Value of parameter which is specified with scan_param_name is discarded.

    """

    def __init__(self, scan_param_name, scan_param_min, scan_param_max, scan_param_step,
                 num_graphs, num_nodes, radius, alpha, L_max, R_max, D, k):

        self.scan_param_name = scan_param_name
        self.scan_param_min = scan_param_min
        self.scan_param_max = scan_param_max
        self.scan_param_step = scan_param_step
        num_data_points = int(np.ceil((scan_param_max - scan_param_min) / scan_param_step + 1))
        self.range = np.linspace(start=scan_param_min,
                                 stop=scan_param_max,
                                 num=num_data_points)
        # self.num_graphs = 0
        self.num_nodes = num_nodes
        self.radius = radius
        self.L_max = L_max
        self.R_max = R_max
        self.D = D
        self.k = k
        self.alpha = alpha
        self.graphs = []
        self.solutions_data = {}
        self.computation_time = 0
        self.most_restrictive_parameters = {"L_max": L_max,
                                            "R_max": R_max,
                                            "D": D,
                                            "k": k}

        # set scan parameter to most restrictive value
        if scan_param_name == "L_max":
            self.most_restrictive_parameters["L_max"] = scan_param_min
        elif scan_param_name == "R_max":
            self.most_restrictive_parameters["R_max"] = scan_param_min
        elif scan_param_name == "D":
            self.most_restrictive_parameters["D"] = scan_param_min
        elif scan_param_name == "k":
            self.most_restrictive_parameters["k"] = scan_param_max
        else:
            raise ValueError("scan_param_name must be either L_max, R_max, D or k. Instead, it is {}."
                             .format(scan_param_name))

        # generate population of graphs which are feasible for most restrictive values
        self.generate_new_graphs(num_graphs)

    def generate_new_graphs(self, num_extra_graphs):
        """Increase graph population. Requires solve to be called again.

        Parameters
        ----------
        num_extra_graphs : int
            Number of graphs to be added to the population of each data point.

        Note
        ----
        Graphs are chosen such that they are feasible to solve for the most restrictive point of the parameter scan.
        This guarantees that there is a feasible solution for every graph at every data point.

        """

        print("adding {} graphs".format(num_extra_graphs))
        # self.num_graphs += num_extra_graphs
        start_time = time.time()
        new_graphs, new_solutions = generate_feasible_graphs(num_graphs=num_extra_graphs,
                                                             num_nodes=self.num_nodes,
                                                             radius=self.radius,
                                                             alpha=self.alpha,
                                                             **self.most_restrictive_parameters)
        self.graphs += new_graphs
        new_solutions_data = [solution.get_solution_data() for solution in new_solutions]
        if self.most_restrictive_parameters[self.scan_param_name] in self.solutions_data.keys():
            self.solutions_data[self.most_restrictive_parameters[self.scan_param_name]] += new_solutions_data
        else:
            self.solutions_data.update({self.most_restrictive_parameters[self.scan_param_name]: new_solutions_data})
        calculation_time = time.time() - start_time
        print("Adding {} graphs succeeded after {} seconds.".format(num_extra_graphs, calculation_time))
        self.computation_time += calculation_time
        assert len(self.solutions_data[self.most_restrictive_parameters[self.scan_param_name]]) == self.num_graphs

    def add_graphs_from_other_random_graph_scan(self, other_random_graph_scan, overwrite=False):
        """Add graphs from another instance of RandomGraphScan, so they can be reused.

        Parameters
        ----------
        other_random_graph_scan : RandomGraphScan instance
            Object holding the graphs which you want to add to this object.
        overwrite : bool
            If true, existing graphs are removed when adding the new graphs. All existing solutions are deleted.
        """
        if not isinstance(other_random_graph_scan, RandomGraphScan):
            raise TypeError("Can only add graphs from a RandomGraphScan object.")
        if self.num_nodes != other_random_graph_scan.num_nodes:
            raise TypeError("Can only add graphs fi the other RandomGraphsScan object has the same number of nodes.")
        if self.radius != other_random_graph_scan.radius:
            raise TypeError("Can only add graphs fi the other RandomGraphsScan object has the same radius.")
        if self.most_restrictive_parameters != other_random_graph_scan.most_restrictive_parameters:
            raise TypeError("Can only add graphs if the other RandomGraphScan object has the same set of most "
                            "restrictive values.")
        print("Adding {} graphs!".format(other_random_graph_scan.num_graphs))
        if not overwrite:
            self.graphs += other_random_graph_scan.graphs
        else:
            self.graphs = other_random_graph_scan.graphs
            self.solutions_data = {}

    def solve(self, overwrite=False):
        """Find the solutions. Can be computationally heavy.

        Parameters
        ----------
        overwrite : bool
            If true, existing solutions are discarded, and all graphs are solved again for each data point.

        Note
        ----
        Does not recalculate solutions that were already found in the past.

        """

        parameters = deepcopy(self.most_restrictive_parameters)
        for value in self.range:
            if value in self.solutions_data.keys() and not overwrite:
                solutions_data_this_value = self.solutions_data[value]
                num_missing_solutions = self.num_graphs - len(solutions_data_this_value)
            else:
                solutions_data_this_value = []
                num_missing_solutions = self.num_graphs
            print("solving {} graphs for {} = {}".format(num_missing_solutions, self.scan_param_name, value))
            start_time = time.time()
            if num_missing_solutions != 0:
                # skip if all solutions are already present for this parameter value
                parameters.update({self.scan_param_name: value})
                graphs_without_solution = self.graphs[-num_missing_solutions:]
                new_solutions = solve_graphs(graph_containers=graphs_without_solution,
                                             alpha=self.alpha,
                                             **parameters)
                solutions_data_this_value += [new_solution.get_solution_data() for new_solution in new_solutions]
            calculation_time = time.time() - start_time
            print("Solving {} graphs succeeded after {} seconds.".format(num_missing_solutions, calculation_time))
            self.computation_time += calculation_time
            assert len(solutions_data_this_value) == self.num_graphs
            self.solutions_data.update({value: solutions_data_this_value})

    def save(self, save_name):
        """Save object as pickle.

        Parameters
        ----------
        save_name : str
            Name/path to which the object will be saved.

        Note
        ----
        This object can be loaded using
        random_graph_scan = pickle.load(open(save_name, "rb"))

        """

        pickle.dump(self, open(save_name, "wb"))

    @property
    def num_graphs(self):
        return len(self.graphs)


def plot_random_graph_scan(random_graph_scan, quantity, ylabel):
    """Plot a quantity, evaluated on solutions, as a parameter scan over random geometric graphs.

    Parameters
    ----------
    random_graph_scan : RandomGraphScan object
        Random graph scan data to plot.
    quantity : str
        Name of quantity which should be plotted on the y-axis.
    ylabel : str
        Label to put on y-axis.

    """

    quantity_average = []
    quantity_error = []
    for value in random_graph_scan.range:
        if value not in random_graph_scan.solutions_data.keys():
            raise ValueError("No solutions available for {}={}, cannot process."
                             .format(random_graph_scan.scan_param_name, value))
        solutions_data_this_value = random_graph_scan.solutions_data[value]
        quantity_list = []
        for solution_data in solutions_data_this_value:
            quantity_list.append(solution_data[quantity])
        quantity_average.append(np.mean(quantity_list))
        quantity_error.append(np.std(quantity_list) / np.sqrt(len(quantity_list)))
    plt.errorbar(x=random_graph_scan.range, y=quantity_average, yerr=quantity_error)
    plt.ylabel(ylabel)
    plt.xlabel(random_graph_scan.scan_param_name)
    plt.show()


def generate_feasible_graphs(num_graphs, num_nodes, radius, alpha, L_max, R_max, D, k):
    """Generate a population of geometric random graphs with feasible solutions for specified parameters.

    Parameters
    ----------
    num_graphs : int
        Number of feasible graphs required.
    num_nodes : int
        Total number of nodes in each random graph.
    radius : float
        Radius of random geometric graphs (all nodes within this distance of one another are connected by an edge).
    alpha : float
        Small number used to set secondary objective.
    L_max : int  # TODO: why can't this be a float?
        Maximum elementary-link length.
    R_max : int
        Maximum number of repeaters on path.
    D : int
        Quantum-repeater capacity.
    k : int
        Robustness parameter.

    Returns
    -------
    graph_containers : list
        Contains number_of_graphs graph container objects.
    solutions : list
        Contains number_of_graphs solutions. solutions[i] corresponds to graph_containers[i].

    """

    graph_containers, solutions = [], []
    number_of_found_graphs = 0
    number_of_tries = 0
    while number_of_found_graphs < num_graphs:
        number_of_tries += 1
        if number_of_tries == 100 and number_of_found_graphs == 0:
            raise RuntimeError("Could not find a feasible graph in 100 tries.")
        graph = create_graph_and_partition(num_nodes=num_nodes, radius=radius, draw=False)
        if graph is None:
            print("generated graph is None")  # TODO: why does this sometimes happen?
            continue
        if not nx.is_connected(graph):
            continue
        graph_container = GraphContainer(graph)
        prog = NodeDisjointLinkBasedProgram(graph_container=graph_container, D=D, k=k, alpha=alpha, L_max=L_max,
                                            R_max=R_max)
        solution, _ = prog.solve()
        if 'infeasible' in solution.get_status_string():
            continue
        graph_containers.append(graph_container)
        solutions.append(solution)
        number_of_found_graphs += 1
    return graph_containers, solutions


def solve_graphs(graph_containers, alpha, L_max, R_max, D, k):
    """Solve repeater allocation problem for a collection of graphs.

    Parameters
    ----------
    graph_containers : list
        Contains graph container objects.
    alpha : float
        Small number used to set secondary objective.
    L_max : int  # TODO: why can't this be a float?
        Maximum elementary-link length.
    R_max : int
        Maximum number of repeaters on path.
    D : int
        Quantum-repeater capacity.
    k : int
        Robustness parameter.

    Returns
    -------
    solutions : list
        Contains solutions. solutions[i] corresponds to graph_containers[i].

    """
    solutions = []
    for graph_container in graph_containers:
        prog = NodeDisjointLinkBasedProgram(graph_container=graph_container, D=D, k=k, alpha=alpha, L_max=L_max,
                                            R_max=R_max)
        solution, _ = prog.solve()
        if 'infeasible' in solution.get_status_string():
            raise ValueError("Not all graphs allow for a solution of the repeater allocation problem for the"
                             "specified parameters.")
        solutions.append(solution)
    return solutions


if __name__ == "__main__":

    results = RandomGraphScan(scan_param_name="L_max",
                              scan_param_min=0.7,
                              scan_param_max=1.2,
                              scan_param_step=0.1,
                              num_graphs=50,
                              num_nodes=25,
                              radius=0.9,
                              alpha=1 / 250,
                              L_max=0.7,
                              R_max=6,
                              D=4,
                              k=6)
    results.solve()
    filename = "effect_of_Lmax_v3.p"
    results.save(filename)
    # results = pickle.load(open(filename, "rb"))
    # other_filename = "effect_of_Lmax_v3.p"
    # other_results = pickle.load(open(other_filename, "rb"))
    # results.add_graphs_from_other_random_graph_scan(other_results)

    for _ in range(19):
        results.generate_new_graphs(50)
        results.solve()
        results.save(filename)

    # plot_random_graph_scan(random_graph_scan=results,
    #                       quantity="min_node_connectivity",
    #                       ylabel="Average Node Connectivity")
    print(results.computation_time)

