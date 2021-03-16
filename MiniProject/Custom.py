import random
import copy
import matplotlib.pyplot as plt


def default_fitness(cover, edges, covers):
    score = 1
    for edge in edges:
        (u, v) = edge
        if cover[u] == 0 and cover[v] == 0:
            score += 0
        elif cover[u] == 1 and cover[v] == 1:
            score += 1
        else:
            score += 2
    return score


def default_crossover(cover1, cover2):
    length = len(cover1)
    mid = length // 2
    child1 = cover1[0: mid] + cover2[mid: length]
    child2 = cover2[0: mid] + cover1[mid: length]
    return child1, child2


def default_mutation(cover):
    length = len(cover)
    random_bit = random.randint(0, length - 1)

    if cover[random_bit] == 0:
        return cover[0: random_bit] + [1] + cover[random_bit + 1: length]
    else:
        return cover[0: random_bit] + [0] + cover[random_bit + 1: length]


class CustomAlgo:

    def __init__(self, adjacency_matrix, edges):
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges

    def get_min_vertex_cover(self, init_pop=1000, num_generations=1000, crossover_prob=0.995, mutation_prob=0.05,
                             fitness_func=default_fitness, crossover_func=default_crossover,
                             mutation_func=default_mutation, description="", fixed_cover=None):
        print("Custom Algo!")
        print(description)
        # init_pop - even number- check and raise exception.
        # crossover_prob - float between 0 - 1 - check and raise exception.
        # fitness_func - function that gets 3 arguments-
        #                cover (binary list), edges (list of tuples- (i, j)), covers (list of binary lists)
        #                return fitness score (R)
        # crossover_func - function that gets 2 arguments- cover1 (binary list), cover2 (binary list)
        #                  return (child1, child2) - binary lists of the same size
        # mutation_func - function that gets 1 argument- cover (binary list)
        #                  return cover - binary list of the same size

        if crossover_func is None:
            crossover_func = default_crossover
        if mutation_func is None:
            mutation_func = default_mutation
        parameters = custom_algorithm(self.nodes, self.edges, init_pop, num_generations, crossover_prob, mutation_prob,
                                      fitness_func, crossover_func, mutation_func, fixed_cover)
        (cover, fitness, cover_count) = parameters
        print("Vertex cover of the custom algorithm consists {} nodes".format(cover_count))
        if is_cover(cover, self.edges):
            print("Is cover!")
        else:
            print("Is not cover!")
        print("###########################################################################")

        return cover, cover_count


def custom_algorithm(nodes, edges, init_pop, num_generations, crossover_prob, mutation_prob,
                     fitness_func, crossover_func, mutation_func, fixed_cover):
    best_fitness = 0
    best_cover = []
    best_cover_count = 0
    cover_fit_list = []

    x_gen = []
    y_fit = []

    covers = init_covers(nodes, init_pop, fixed_cover)

    for cover in covers:  # initial calculation
        fit_score = fitness_func(cover, edges, covers)
        cover_fit_list += [(cover, fit_score)]
        best_parameters = get_best_parameters(best_cover, best_fitness, [(cover, fit_score)])

    for gen in range(num_generations):  # generations
        x_gen += [gen]
        y_fit += [best_parameters[1]]
        (cover_fit_list, covers) = selection(cover_fit_list)  # select the good covers

        cover_fit_list, best_parameters = crossover_mutation(edges, covers, cover_fit_list, crossover_prob,
                                                             best_parameters, crossover_func, mutation_prob,
                                                             mutation_func, fitness_func)

        print_results(gen, best_parameters, edges)

    # plt.plot(x_gen, y_fit)
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.grid()
    # plt.show()
    return best_parameters


def init_covers(nodes, init_pop, fixed_cover):
    covers = []
    if fixed_cover is None:
        for i in range(init_pop):  # initial covers
            covers += [generate_cover(nodes)]
        return covers

    for i in range(init_pop):
        covers += [fixed_cover]
    return covers


def generate_cover(length):
    binary_list = []
    for i in range(length):
        binary_list += [random.randint(0, 1)]
    return binary_list


def selection(cover_fit_list):
    ranges_list = []
    total_range = 0
    for cover, fit_score in cover_fit_list:
        ranges_list += [(total_range, total_range + fit_score)]
        total_range += fit_score

    new_cover_fit_list = []
    covers = []
    for cover, fit_score in cover_fit_list:
        rand = random.uniform(0, total_range)
        for start, end in ranges_list:
            if start <= rand <= end:
                new_cover_fit_list += [(cover, 0)]
                covers += [cover]
    return new_cover_fit_list, covers


def is_cover(cover, edges):
    if len(cover) == 0:
        return False

    for edge in edges:
        (u, v) = edge
        if cover[u] == 0 and cover[v] == 0:
            return False
    return True


def crossover_mutation(edges, covers, cover_fit_list, crossover_prob, parameters, crossover_func, mutation_prob,
                       mutation_func, fitness_func):
    new_cover_fit_list = []
    best_parameters = parameters
    best_cover, best_fitness, best_cover_count = best_parameters

    for i in range(len(covers) // 2):
        cover_index = i * 2
        cover1 = cover_fit_list[cover_index][0]
        cover2 = cover_fit_list[cover_index + 1][0]
        crossover_rand = random.random()
        if crossover_rand < crossover_prob:
            (cover1, cover2) = crossover_func(cover1, cover2)
        mutation_rand = random.random()
        if mutation_rand < mutation_prob:
            cover1 = mutation_func(cover1)
            cover2 = mutation_func(cover2)

        fit_score1 = fitness_func(cover1, edges, covers)
        fit_score2 = fitness_func(cover2, edges, covers)
        new_cover_fit_list += [(cover1, fit_score1), (cover2, fit_score2)]

        best_parameters = get_best_parameters(best_cover, best_fitness, [(cover1, fit_score1), (cover2, fit_score2)])

    return new_cover_fit_list, best_parameters


def get_best_parameters(cover, fitness, cover_fit_list):
    best_fitness = fitness
    best_cover = cover
    for cover, fit_score in cover_fit_list:
        if fit_score > best_fitness:
            best_fitness = fit_score
            best_cover = cover

    return best_cover, best_fitness, len(list(filter(lambda bit: bit == 1, best_cover)))


def print_results(gen, parameters, edges):
    (cover, fitness, cover_count) = parameters
    if (gen % 200) == 0:
        print("Generation = {}, Best Cover Score= {}, Number of Nodes = {}".format(gen, fitness,
                                                                                   cover_count))
        if is_cover(cover, edges):
            print("Is cover!")
        else:
            print("Is not cover!")
