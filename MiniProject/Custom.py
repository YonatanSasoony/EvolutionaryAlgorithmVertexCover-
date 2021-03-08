import random
import copy
import matplotlib.pyplot as plt
import numpy as np

def default_fitness(cover, edges, covers):
    score = 1
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            score += 0
        elif int(cover[u]) == 1 and int(cover[v]) == 1:
            score += 1
        else:
            score += 2
    return score


def fitness2(cover, edges, covers):
    #  assume the optimal cover has less than 100 vertices.. otherwise he will get bad fitness  for it
    #  and we will miss it
    length = len(cover)
    x = 100
    for i in range(0, length):
        if cover[i] == '1':
            x = x - 1
    result = 1/(1+np.exp((-1)*x))
    return result


def default_crossover(cover1, cover2):
    length = len(cover1)
    mid = length // 2
    child1 = cover1[0: mid] + cover2[mid: length]
    child2 = cover2[0: mid] + cover1[mid: length]
    return child1, child2


def default_mutation(cover):
    length = len(cover)
    random_bit = random.randint(0, length - 1)

    if cover[random_bit] == "0":
        return cover[0: random_bit] + "1" + cover[random_bit + 1: length]
    else:
        return cover[0: random_bit] + "0" + cover[random_bit + 1: length]


class CustomAlgo:

    def __init__(self, adjacency_matrix, edges):
        print("Custom Algo!")
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges

    def get_min_vertex_cover(self, init_pop=1000, num_generations=1000, crossover_prob=0.7,
                             fitness_func=default_fitness, crossover_func=default_crossover,
                             mutation_func=default_mutation):

        # init_pop - even number- check and raise exception.
        # crossover_prob - float between 0 - 1 - check and raise exception.
        # fitness_func - function that gets 3 arguments-
        #                cover (binary string), edges (list of tuples- (i, j)), covers (list of binary strings)
        #                return fitness score (R)
        # crossover_func - function that gets 2 arguments- cover1 (binary string), cover2 (binary string)
        #                  return (child1, child2) - binary strings
        # mutation_func - function that gets 1 argument- cover (binary string)
        #                  return cover - binary string

        (cover, cover_count) = custom_algorithm(self.nodes, self.edges, init_pop, num_generations, crossover_prob,
                                                fitness_func, crossover_func, mutation_func)
        print("Vertex cover of the custom algorithm consists {} nodes".format(cover_count))
        print("###########################################################################")

        return cover, cover_count


def custom_algorithm(nodes, edges, init_pop, num_generations, crossover_prob, fitness_func, crossover_func,
                     mutation_func):
    best_fitness = 0
    best_cover = ""
    best_cover_count = 0
    cover_fit_list = []
    temp = []
    x_gen = []
    y_fit = []
    covers = []
    for i in range(init_pop):  # initial covers
        covers += [generate_cover(nodes)]

    for cover in covers:  # initial covers
        fit_score = fitness_func(cover, edges, covers)
        cover_fit_list += [(cover, fit_score)]
        if fit_score > best_fitness:
            best_fitness = fit_score
            best_cover = cover
            best_cover_count = len(list(filter(lambda bit: int(bit) == 1, best_cover)))

    for gen in range(num_generations):  # generations
        x_gen += [gen]
        y_fit += [best_fitness]
        (cover_fit_list, covers) = selection(cover_fit_list)  # select the good covers
        if (gen % 50) == 0:
            print("Generation = {}, Best Cover Score= {}, Number of Nodes = {}".format(gen, best_fitness,
                                                                                       best_cover_count))
            if is_cover(best_cover, edges):
                print("Is cover!")
            else:
                print("Is not cover!")
        for cover_index in range(init_pop // 2):
            cover1 = cover_fit_list[cover_index][0]
            cover2 = cover_fit_list[cover_index + 1][0]
            crossover_rand = random.random()
            if crossover_rand <= crossover_prob:  # probability to do crossover
                (cover1, cover2) = crossover_func(cover1, cover2)
            mutation_rand = random.random()
            if mutation_rand < 0.001:  # 1/1000% do mutation
                cover1 = mutation_func(cover1)
                cover2 = mutation_func(cover2)
            fit_score1 = fitness_func(cover1, edges, covers)
            if fit_score1 > best_fitness:
                best_fitness = fit_score1
                best_cover = cover1
                best_cover_count = len(list(filter(lambda bit: int(bit) == 1, best_cover)))
            temp += [(cover1, fit_score1)]
            fit_score2 = fitness_func(cover2, edges, covers)
            if fit_score2 > best_fitness:
                best_fitness = fit_score2
                best_cover = cover2
            temp += [(cover2, fit_score2)]
        cover_fit_list = copy.deepcopy(temp)
        temp.clear()
    print("the best cover is: " + str(best_cover))
    # plt.plot(x_gen, y_fit)
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.grid()
    # plt.show()
    return best_cover, best_cover_count


def generate_cover(length):
    binary_str = ""
    for i in range(length):
        next_bit = str(random.randint(0, 1))
        binary_str += next_bit
    return binary_str


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
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            return False
    return True
