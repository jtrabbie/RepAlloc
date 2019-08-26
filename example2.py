""""
Example of CPLEX optimization for a simple linear program, where it is constructed per row

Objective function

max   25X + 20Y
s.t.  20X + 12Y <= 2,000
       5X +  5Y <= 540
        X       >= 0
              Y >= 0

Optimal solution: X = 88, Y = 20, obj_val = 2600
"""

import cplex
from cplex.exceptions import CplexError

# Problem Statement
my_obj = [25, 20]  # coefficients of objective function
my_lb = [0, 0]  # lower bounds for variables
my_ub = [cplex.infinity, cplex.infinity]  # upper bounds for variables
my_rhs = [2000, 540]  # right hand side of constraints
my_sense = ["L"] * len(my_rhs)  # constrains should be less than or equal


def add_rows(prob):
    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, names=["X", "Y"])

    # Add the constraints
    C1 = cplex.SparsePair(ind=[0, 1], val=[20, 12])
    C2 = cplex.SparsePair(ind=[0, 1], val=[5, 5])
    rows = [C1, C2]

    prob.linear_constraints.add(lin_expr=rows, rhs=my_rhs, names=["C1", "C2"], senses=my_sense)


if __name__ == "__main__":

    try:
        my_prob = cplex.Cplex()
        my_prob.objective.set_sense(my_prob.objective.sense.maximize)
        add_rows(my_prob)
        my_prob.write("example2.lp")
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
