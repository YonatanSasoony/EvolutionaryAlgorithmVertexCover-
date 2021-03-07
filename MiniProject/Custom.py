import random
import copy
import matplotlib.pyplot as plt


def fitness(cover, edges):
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


class CustomAlgo:

    def __init__(self, adjacency_matrix, edges):
        print("Custom Algo!")
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges

    def get_min_vertex_cover(self, init_pop=1000, num_generations=1000, crossover_prob=0.7,
                             fitness_func=fitness):

        # init_pop - even number- check and raise exception.
        # crossover_prob - float between 0 - 1 - check and raise exception.
        algo_result = custom_algorithm(self.nodes, self.edges, init_pop, num_generations, crossover_prob, fitness_func)
        print("Vertex cover of the custom algorithm consists {} nodes".format(algo_result))

        return algo_result


def custom_algorithm(nodes, edges, init_pop, num_generations, crossover_prob, fitness_func):
    # graph = [[0, 1, 1, 0, 0, 0],
    #          [1, 0, 1, 1, 1, 1],
    #          [1, 1, 0, 0, 0, 0],
    #          [0, 1, 0, 0, 0, 0],
    #          [0, 1, 0, 0, 0, 0],
    #          [0, 1, 0, 0, 0, 0]]

    # graph = generate_graph(nodes=50)

    # graph = np.zeros((20, 20))
    # for i in range(20):
    #     graph[i, 0] = 1
    #     graph[0, i] = 1
    # graph_length = len(graph)

    best_fitness = 0
    best_cover = ""
    best_cover_count = 0
    cover_fit_list = []
    temp = []
    x_gen = []
    y_fit = []

    for i in range(init_pop):  # initial covers
        cover = generate_cover(nodes)
        fit_score = fitness_func(cover, edges)
        cover_fit_list += [(cover, fit_score)]
        if fit_score > best_fitness:
            best_fitness = fit_score
            best_cover = cover
            best_cover_count = len(list(filter(lambda bit: int(bit) == 1, best_cover)))

    for gen in range(num_generations):  # generations
        x_gen += [gen]
        y_fit += [best_fitness]
        cover_fit_list = selection(cover_fit_list)  # select the good covers
        if (gen % 50) == 0:
            print("Generation = {}, Best Cover = {}".format(gen, best_cover_count))
            if is_cover(best_cover, edges):
                print("Is cover!")
            else:
                print("Is not cover!")
        for covers_couple in range(int(init_pop / 2)):
            cover1 = cover_fit_list.pop()[0]
            cover2 = cover_fit_list.pop()[0]
            crossover_rand = random.random()
            if crossover_rand <= crossover_prob:  # probability to do crossover
                (cover1, cover2) = crossover(cover1, cover2)
            mutation_rand = random.random()
            if mutation_rand < 0.001:  # 1/1000% do mutation
                cover1 = mutation(cover1)
                cover2 = mutation(cover2)
            fit_score1 = fitness_func(cover1, edges)
            if fit_score1 > best_fitness:
                best_fitness = fit_score1
                best_cover = cover1
                best_cover_count = len(list(filter(lambda bit: int(bit) == 1, best_cover)))
            temp += [(cover1, fit_score1)]
            fit_score2 = fitness_func(cover2, edges)
            if fit_score2 > best_fitness:
                best_fitness = fit_score2
                best_cover = cover2
            temp += [(cover2, fit_score2)]
        cover_fit_list = copy.deepcopy(temp)
        temp.clear()
    print("the best cover is: " + str(best_cover))
    plt.plot(x_gen, y_fit)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()
    return best_cover_count


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
    for cover, fit_score in cover_fit_list:
        rand = random.uniform(0, total_range)
        for start, end in ranges_list:
            if start <= rand <= end:
                new_cover_fit_list += [(cover, 0)]
    return new_cover_fit_list


def crossover(cover1, cover2):
    length = len(cover1)
    mid = length / 2
    child1 = cover1[0: int(mid)] + cover2[int(mid): int(length)]
    child2 = cover2[0: int(mid)] + cover1[int(mid): int(length)]
    return child1, child2


def mutation(cover):
    length = len(cover)
    bit = random.randint(0, length - 1)
    if cover[bit] == "0":
        return cover[0: bit] + "1" + cover[bit + 1: length]
    else:
        return cover[0: bit] + "0" + cover[bit + 1: length]


def is_cover(cover, edges):
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            return False
    return True
