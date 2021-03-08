import cplex


class LPAlgo:

    def __init__(self, adjacency_matrix, edges):
        print("LP Algo!")
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges

    def get_min_vertex_cover(self):
        (cover, cover_count) = lp_algorithm(self.nodes, self.edges)
        print("Vertex cover of the LP algorithm consists {} nodes".format(cover_count))
        print("###########################################################################")

        return cover, cover_count


def lp_algorithm(nodes, edges):
    prob = cplex.Cplex()
    prob.set_problem_name("Minimum Vertex Cover")
    prob.set_problem_type(cplex.Cplex.problem_type.LP)
    prob.objective.set_sense(prob.objective.sense.minimize)

    nodes_names = ["A" + str(u) for u in range(nodes)]
    # print(nodes_names)

    # Objective (linear) weights
    w_obj = [1 for u in range(nodes)]
    # print(w_obj)

    # Lower bounds. Since these are all zero, we could simply not pass them in as
    # all zeroes is the default.
    low_bnd = [0 for u in range(nodes)]

    # Upper bounds. The default here would be cplex.infinity, or 1e+20.
    upr_bnd = [1 for u in range(nodes)]

    prob.variables.add(names=nodes_names, obj=w_obj, lb=low_bnd, ub=upr_bnd)

    # How to set the variable types
    # Must be AFTER adding the variables
    #
    # Option #1: Single variable name (or number) with type
    # prob.variables.set_types("0", prob.variables.type.continuous)
    # Option #2: List of tuples in the form (var_name, type)
    # prob.variables.set_types([("1", prob.variables.type.integer), \
    #                           ("2", prob.variables.type.binary), \
    #                           ("3", prob.variables.type.semi_continuous), \
    #                           ("4", prob.variables.type.semi_integer)])
    #
    # Vertex cover requires only integers
    all_int = [(var, prob.variables.type.integer) for var in nodes_names]
    prob.variables.set_types(all_int)

    constraints = []
    for e in edges:
        (u, v) = e
        constraints.append([["A" + str(u), "A" + str(v)], [1, 1]])

    # Constraint names
    constraint_names = ["".join(x[0]) for x in constraints]

    # Each edge must have at least one vertex
    rhs = [1] * len(constraints)

    # We need to enter the senses of the constraints. That is, we need to tell Cplex
    # whether each constrains should be treated as an upper-limit (≤, denoted "L"
    # for less-than), a lower limit (≥, denoted "G" for greater than) or an equality
    # (=, denoted "E" for equality)
    constraint_senses = ["G"] * len(constraints)

    # And add the constraints
    prob.linear_constraints.add(names=constraint_names,
                                lin_expr=constraints,
                                senses=constraint_senses,
                                rhs=rhs)
    # Solve the problem
    # print("Problem Type: %s" % prob.problem_type[prob.get_problem_type()])
    print("###########################################################################")
    prob.solve()
    print("###########################################################################")
    cover = prob.solution.get_values()
    cover_count = len(list(filter(lambda u: u == 1, cover)))

    return cover, cover_count
