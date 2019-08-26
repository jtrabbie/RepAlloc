""""
Example of CPLEX optimization for a simple linear program, where it is constructed per column

Objective function

max   50X + 120Y
s.t. 100X + 200Y <= 10,000
      10X +  30Y <= 1,200
        X +    Y <= 110
        X        >= 0
               Y >= 0

Optimal solution: X = 60, Y = 20, obj_value = 5400
"""

import cplex
from cplex.exceptions import CplexError

# Problem Statement
my_obj = [50, 120]  # coefficients of objective function
my_lb = [0, 0]  # lower bounds for variables
my_ub = [cplex.infinity, cplex.infinity]  # upper bounds for variables
my_rhs = [10000, 1200, 110]  # right hand side of constraints
my_sense = ["L"] * len(my_rhs)  # constrains should be less than or equal


def add_columns(prob):
    prob.linear_constraints.add(rhs=my_rhs, names=["C1", "C2", "C3"], senses=my_sense)

    # Add X and Y with the given coefficients in constrains 0, 1 and 2 with
    c = [[[0, 1, 2], [100.0, 10.0, 1.0]],
         [[0, 1, 2], [200.0, 30.0, 1.0]]]

    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, columns=c, names=["X", "Y"])


if __name__ == "__main__":

    try:
        my_prob = cplex.Cplex()
        my_prob.objective.set_sense(my_prob.objective.sense.maximize)
        add_columns(my_prob)
        my_prob.write("example1.lp")
        my_prob.solve()
    except CplexError as exc:
        print(exc)
        exit()

    print("Solution status = ", my_prob.solution.get_status(), ":", end=' ')
    print(my_prob.solution.status[my_prob.solution.get_status()])
    print("Solution value  = ", my_prob.solution.get_objective_value())

    var_vals = my_prob.solution.get_values()
    for v in var_vals:
        print(v)
