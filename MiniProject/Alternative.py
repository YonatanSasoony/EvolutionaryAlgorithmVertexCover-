import random
import copy
import matplotlib.pyplot as plt
import Greedy


def default_crossover(cover1, cover2):  # uniform crossover
    length = min(len(cover1), len(cover2))
    child1 = []
    child2 = []
    flag = 0
    for i in range((length // 2)):
        bit = i * 2
        flag = 1 - flag
        if flag == 1:
            child1 += [cover1[bit]] + [cover1[bit + 1]]
            child2 += [cover2[bit]] + [cover2[bit + 1]]
        else:
            child1 += [cover2[bit]] + [cover2[bit + 1]]
            child2 += [cover1[bit]] + [cover1[bit + 1]]

    return list(dict.fromkeys(child1)), list(dict.fromkeys(child2))


# Random Resetting
# Random Resetting is an extension of the bit flip for the integer representation.
# In this, a random value from the set of permissible values is assigned to a randomly chosen gene.
def default_mutation(cover, nodes):
    mutation = copy.deepcopy(cover)
    rand_index = random.randint(0, len(cover)-1)
    rand_node = random.randint(0, nodes-1)
    mutation[rand_index] = rand_node
    return list(dict.fromkeys(mutation))


class AlternativeAlgo:

    def __init__(self, adjacency_matrix, edges):
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges
        self.degree_dict = create_degree_dict(self.nodes, self.edges)
        greedy_algo = Greedy.GreedyAlgo(adjacency_matrix, edges)
        (binary_cover_greedy, cover_count_greedy) = greedy_algo.get_min_vertex_cover()
        self.greedy_cover = []
        for node in range(len(binary_cover_greedy)):
            if binary_cover_greedy[node] == 1:
                self.greedy_cover += [node]

    def get_min_vertex_cover(self, init_pop=1000, num_generations=1000, crossover_prob=0.995, mutation_prob=0.05,
                             crossover_func=default_crossover,
                             mutation_func=default_mutation):
        print("Alternative Algo!")
        # init_pop - even number- check and raise exception.
        # crossover_prob - float between 0 - 1 - check and raise exception.
        # crossover_func - function that gets 2 arguments- cover1 (nodes list), cover2 (nodes list)
        #                  return (child1, child2) - nodes lists
        # mutation_func - function that gets 1 argument- cover (nodes list)
        #                  return cover - nodes list

        (cover, cover_count) = custom_algorithm(self.nodes, self.edges, init_pop, num_generations, crossover_prob,
                                                self.degree_dict, mutation_prob, self.greedy_cover, crossover_func,
                                                mutation_func)
        print("Vertex cover of the custom algorithm consists {} nodes".format(cover_count))
        if is_cover(cover, self.edges):
            print("Is cover!")
        else:
            print("Is not cover!")
        print("###########################################################################")

        return cover, cover_count


def custom_algorithm(nodes, edges, init_pop, num_generations, crossover_prob,
                     degree_dict, mutation_prob, greedy_cover, crossover_func,
                     mutation_func):
    best_fitness = 0
    best_cover = []
    best_cover_count = 0
    cover_fit_list = []
    temp = []
    x_gen = []
    y_fit = []
    covers = []

    # for i in range(init_pop):  # initial covers
    #     covers += [generate_cover(nodes)]
    for i in range(init_pop):  # initial covers
        covers += [greedy_cover]

    for cover in covers:  # initial covers
        fit_score = fitness(cover, edges, covers, degree_dict)
        cover_fit_list += [(cover, fit_score)]
        if fit_score > best_fitness:
            best_fitness = fit_score
            best_cover = cover
            best_cover_count = len(list(filter(lambda bit: bit == 1, best_cover)))

    for gen in range(num_generations):  # generations
        x_gen += [gen]
        y_fit += [best_fitness]
        (cover_fit_list, covers) = selection(cover_fit_list)  # select the good covers
        if (gen % 50) == 0:
            print("Generation = {}, Best Cover Score= {}, Number of Nodes = {}".format(gen, best_fitness,
                                                                                       best_cover_count))
            mutation_prob -= 0.01
            if is_cover(best_cover, edges):
                print("Is cover!")
            else:
                print("Is not cover!")
        for cover_index in range(init_pop // 2):
            cover1 = cover_fit_list[cover_index][0]
            cover2 = cover_fit_list[cover_index + 1][0]
            crossover_rand = random.random()
            if crossover_rand <= crossover_prob:
                (cover1, cover2) = crossover_func(cover1, cover2)
            mutation_rand = random.random()
            if mutation_rand <= crossover_prob:
                cover1 = mutation_func(cover1, nodes)
                cover2 = mutation_func(cover2, nodes)
            fit_score1 = fitness(cover1, edges, covers, degree_dict)
            if fit_score1 > best_fitness:
                best_fitness = fit_score1
                best_cover = cover1
                best_cover_count = len(list(filter(lambda bit: bit == 1, best_cover)))
            temp += [(cover1, fit_score1)]
            fit_score2 = fitness(cover2, edges, covers, degree_dict)
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


def generate_cover(nodes):
    cover = []
    for node in range(nodes):
        rand = random.random()
        if rand < 0.5:
            cover += [node]
    return cover


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
        if u not in cover and v not in cover:
            return False
    return True


def fitness(cover, edges, covers, degree_dict):
    total = 0
    for cover in covers:
        total += degree(cover, degree_dict)
    score = 0
    for cover in covers:
        score += degree(cover, degree_dict)

    score /= total
    if is_cover(cover, edges):
        score += 200
    return score


def degree(cover, degree_dict):
    degree_sum = 0
    for node in cover:
        degree_sum += degree_dict[node]
    return degree_sum


def node_degree_graph(node, edges):
    degree_sum = 0
    for edge in edges:
        (u, v) = edge
        if node == u or node == v:
            degree_sum += 1
    return degree_sum


def create_degree_dict(nodes, edges):
    degree_dict = {}
    for node in range(nodes):
        degree_sum = 0
        for edge in edges:
            (u, v) = edge
            if node == u or node == v:
                degree_sum += 1
        degree_dict[node] = degree_sum
    return degree_dict
