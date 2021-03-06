import sys
import datetime
import numpy as np
import random
import matplotlib.pylab as plt
import networkx as nx
from scipy.io import mmread
import Article
import Greedy
import LP
import Custom


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout

    def write(self, message):
        with open("logfile.txt", "a", encoding='utf-8') as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def generate_graph(nodes, edge_probability):
    adjacency_matrix = np.zeros((nodes, nodes))
    edges = []

    for i in range(nodes):
        for j in range(i):
            edge_rand = random.random()
            if edge_rand < edge_probability:
                adjacency_matrix[i, j] = 1
                edges.append((i, j))

    return adjacency_matrix, edges


def get_known_graph(graph_name):
    mtx = mmread('benchmark graphs/' + graph_name + '.mtx')
    graph = mtx.todense()
    edges = []

    for i in range(len(graph)):
        for j in range(i):
            edge_rand = random.random()
            if graph[i, j] == 1:
                edges.append((i, j))

    return graph, edges


def draw_graph(edges, nodes):
    print("Number of edges {}".format(len(edges)))
    G = nx.Graph()
    G.add_nodes_from(list(range(0, nodes)))
    G.add_edges_from(edges)
    plt.figure(figsize=(12, 6))
    nx.draw(G, node_color='r', node_size=18, alpha=0.8)
    plt.show()


# Fitness #

def fitness2(cover, edges, covers):
    #  assume the optimal cover has less than 100 vertices.. otherwise he will get bad fitness  for it
    #  and we will miss it
    # maybe init x as greedy cover count?
    length = len(cover)
    x = 100
    for i in range(length):
        if cover[i] == 1:
            x -= 1
    result = 1 / (1 + np.exp(-x))
    return result


def fitness_revers_article(cover, edges, covers):
    obstacles = 1
    for e in edges:
        (u, v) = e
        if (cover[u] == 0) & (cover[v] == 0):
            obstacles += 1
    score = 1 / obstacles * 100
    if is_cover(cover, edges):
        score += len(edges)
    return score


def fitness_sum(cover, edges, covers):
    score = 0

    n = len(cover)
    for i in range(n):
        sub_score = 0
        for j in range(n):
            if (i, j) in edges:
                sub_score += 1 - cover[j]
        score += cover[i] + n * (1 - cover[i]) * sub_score

    score = 1 / score * 100
    if is_cover(cover, edges):
        score += len(edges)
    return score


def fitness005(cover, edges, covers):
    score = 1
    for edge in edges:
        (u, v) = edge
        if cover[u] == 0 and cover[v] == 0:
            score += 0
        elif cover[u] == 1 and cover[v] == 1:
            score += 0
        else:
            score += 5
    return score


def fitness_k(cover, edges, covers, k):
    score = 1
    for edge in edges:
        (u, v) = edge
        if cover[u] == 0 and cover[v] == 0:
            score += 0
        elif cover[u] == 1 and cover[v] == 1:
            score += 1
        else:
            score += k * len(edges)
    return score


def fitness_cover_bonus(cover, edges, covers):
    score = 1
    for edge in edges:
        (u, v) = edge
        if cover[u] == 0 and cover[v] == 0:
            score += 0
        elif cover[u] == 1 and cover[v] == 1:
            score += 1
        else:
            score += 5
    if is_cover(cover, edges):
        score += len(edges)
    return score


def is_cover(cover, edges):
    for edge in edges:
        (u, v) = edge
        if cover[u] == 0 and cover[v] == 0:
            return False
    return True


def fitness_cover_count_ratio(cover, edges, covers):
    score = 0
    for edge in edges:
        (u, v) = edge
        if cover[u] == 1 or cover[v] == 1:
            score += 1

    score -= (2 * len(list(filter(lambda bit: bit == 1, cover))))

    if is_cover(cover, edges):
        score += 3 * len(edges)
    return max(score, 1)


def fitness_new_article(cover, edges, covers, degree_dict):
    total = total_new_article(covers, degree_dict)
    score = 0
    # for cover in covers:
    score += degree_new_article(cover, degree_dict)
    return (score * 300 / total) - len(list(filter(lambda bit: bit == 1, cover)))


def total_new_article(covers, degree_dict):
    total = 0
    for cover in covers:
        total += degree_new_article(cover, degree_dict)
    return total


def degree_new_article(cover, degree_dict):
    degree_sum = 0
    for bit in range(len(cover)):
        if cover[bit] == 1:
            degree_sum += degree_dict[bit]
    return degree_sum


def node_degree_graph(node, edges):
    degree = 0
    for edge in edges:
        (u, v) = edge
        if node == u or node == v:
            degree += 1
    return degree


def create_degree_dict(nodes, edges):
    degree_dict = {}
    for node in range(nodes):
        degree = 0
        for edge in edges:
            (u, v) = edge
            if node == u or node == v:
                degree += 1
        degree_dict[node] = degree
    return degree_dict


def uncovered_edges(cover, edges):
    uncovered = 0
    for e in edges:
        (u, v) = e
        if cover[u] == 0 and cover[v] == 0:
            uncovered += 1
    return uncovered


def fitness_penalty(cover, edges, covers):
    uncovered = uncovered_edges(cover, edges)
    cover_size = len(list(filter(lambda u: u == 1, cover)))
    m = 2 * len(cover)
    penalty = (m * uncovered + cover_size)
    score = 10000 / penalty

    if is_cover(cover, edges):
        score += 10
    return score


# cover1 - 01101|00010
# cover2 - 10100|01000

# child1 - 01101|01000
# child2 - 10100|00010

# Crossovers #

# random point crossover: select bit- do cross. example:
# cover1 - 011|0100010
# cover2 - 101|0001000

# child1 - 011|0001000
# child2 - 101|0100010
#

def random_point_crossover(cover1, cover2):
    length = len(cover1)
    bit1 = random.randint(0, length - 1)

    child1 = cover1[0: bit1] + cover2[bit1: length]
    child2 = cover2[0: bit1] + cover1[bit1: length]
    return child1, child2


# double point crossover: select 2 bits- do cross. example:
# cover1 - 01|101|00010
# cover2 - 10|100|01000

# child1 - 01|100|00010
# child2 - 10|101|01000
#
def double_point_crossover(cover1, cover2):
    length = len(cover1)
    bit1 = random.randint(0, length - 1)
    bit2 = random.randint(0, length - 1)
    while bit1 == bit2:
        bit2 = random.randint(0, length - 1)

    min1 = min(bit1, bit2)
    bit2 = max(bit1, bit2)
    bit1 = min1

    child1 = cover1[0: bit1] + cover2[bit1: bit2] + cover1[bit2: length]
    child2 = cover2[0: bit1] + cover1[bit1: bit2] + cover2[bit2: length]
    return child1, child2


# uniform crossover: each 2 bits- do cross. example:
# cover1 - 01|10|10|00|10
# cover2 - 10|10|00|10|00

# child1 - 01|10|10|10|10
# child2 - 10|10|00|00|00
#
def uniform_crossover(cover1, cover2):
    length = len(cover1)
    child1 = []
    child2 = []
    flag = 0
    for i in range(length // 2):
        bit = i * 2
        flag = 1 - flag
        if flag == 1:
            child1 += [cover1[bit]] + [cover1[bit + 1]]
            child2 += [cover2[bit]] + [cover2[bit + 1]]
        else:
            child1 += [cover2[bit]] + [cover2[bit + 1]]
            child2 += [cover1[bit]] + [cover1[bit + 1]]

    return child1, child2


# uniform crossover: each bit- flip a coin to decide which get which. example:
# cover1 - 0110100010
# cover2 - 1010001000

# child1 - 0010001010
# child2 - 1110100000
#
def random_uniform_crossover(cover1, cover2):
    length = len(cover1)
    child1 = []
    child2 = []
    flag = 0
    for bit in range(length):
        rand_cover = random.random()
        if rand_cover < 0.5:
            child1 += [cover1[bit]]
            child2 += [cover2[bit]]
        else:
            child1 += [cover2[bit]]
            child2 += [cover1[bit]]

    return child1, child2


def random_choice_crossover(cover1, cover2):
    crossover = random.randint(0, 6)
    if crossover == 3:
        return random_uniform_crossover(cover1, cover2)
    elif crossover == 2:
        return uniform_crossover(cover1, cover2)
    elif crossover == 1:
        return double_point_crossover(cover1, cover2)
    else:
        return random_point_crossover(cover1, cover2)


# Mutations #

# more read here - https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm

def bit_flip_mutation(cover):
    length = len(cover)
    bit = random.randint(0, length - 1)
    cover[bit] = 1 - cover[bit]
    return cover


def each_bit_mutation(cover, bit_prob):
    for bit in cover:
        rand = random.random()
        if rand < bit_prob:
            cover[bit] = 1 - cover[bit]
    return cover


def swap_mutation(cover):
    length = len(cover)
    bit1 = random.randint(0, length - 1)
    bit2 = random.randint(0, length - 1)
    while bit1 == bit2:
        bit2 = random.randint(0, length - 1)

    temp = cover[bit1]
    cover[bit1] = cover[bit2]
    cover[bit2] = temp

    return cover


def inversion_mutation(cover):
    length = len(cover)
    bit1 = random.randint(0, length - 1)
    bit2 = random.randint(0, length - 1)
    while bit1 == bit2:
        bit2 = random.randint(0, length - 1)

    temp = cover[min(bit1, bit2):max(bit1, bit2)]
    reverse = reversed(temp)
    cover = cover[0:min(bit1, bit2)] + reverse + cover[max(bit1, bit2) + 1: length]

    return cover


def scramble_mutation(cover):
    length = len(cover)
    bit1 = random.randint(0, length - 1)
    bit2 = random.randint(0, length - 1)
    while bit1 == bit2:
        bit2 = random.randint(0, length - 1)

    shuffled = cover[min(bit1, bit2):max(bit1, bit2)]
    random.shuffle(shuffled)
    cover = cover[min(bit1, bit2)] + shuffled + cover[max(bit1, bit2) + 1: length]

    return cover


def random_choice_mutation(cover, bit_prob):
    mutation = random.randint(0, 6)
    if mutation == 3:
        return each_bit_mutation(cover, bit_prob)
    elif mutation == 2:
        return swap_mutation(cover)
    elif mutation == 1:
        return inversion_mutation(cover)
    else:
        return bit_flip_mutation(cover)


def main():
    sys.stdout = Logger()
    print("")
    print("##############################################")
    print("##############################################")
    print("##########" + str(datetime.datetime.now()) + "##########")
    print("##############################################")
    print("##############################################")
    print("")
    random_nodes = 100
    edge_probability = .02
    random_graph, random_edges = generate_graph(random_nodes, edge_probability)

    # graphs- from here http://networkrepository.com/dimacs.php
    # benchmark_graph, benchmark_edges = get_known_graph('hamming6-4')  # V-64 E-704
    # benchmark_graph, benchmark_edges = get_known_graph('johnson8-2-4')  # V-28 E-420
    benchmark_graph, benchmark_edges = get_known_graph('MANN-a9')  # V-45 E-918

    benchmark_nodes = len(benchmark_graph)

    # draw_graph(random_edges, random_nodes)
    draw_graph(benchmark_edges, benchmark_nodes)

    greedy_algo = Greedy.GreedyAlgo(benchmark_graph, benchmark_edges)
    lp_algo = LP.LPAlgo(benchmark_graph, benchmark_edges)
    custom_algo = Custom.CustomAlgo(benchmark_graph, benchmark_edges)

    greedy_algo_start_time = datetime.datetime.now()
    (cover_greedy, cover_count_greedy) = greedy_algo.get_min_vertex_cover()
    greedy_algo_run_time = datetime.datetime.now() - greedy_algo_start_time
    print("Greedy Algo runtime: " + str(greedy_algo_run_time))

    article_algo = Article.ArticleAlgo(benchmark_graph, benchmark_edges, cover_count_greedy, 100)
    lp_algo_start_time = datetime.datetime.now()
    (cover_lp, cover_count_lp) = lp_algo.get_min_vertex_cover()
    lp_algo_run_time = datetime.datetime.now() - lp_algo_start_time
    print("LP Algo runtime: " + str(lp_algo_run_time))

    description = "GENERATIONS= 500, FITNESS COVER BONUS, POP=500, RANDOM COVERS, MUTATION PROB=0.05, RANDOM POINT " \
                  "CROSSOVER "
    custom_algo_start_time = datetime.datetime.now()
    (cover_custom, cover_count_custom) = custom_algo.get_min_vertex_cover(description=description,
                                                                          fitness_func=fitness_cover_bonus,
                                                                          init_pop=500,
                                                                          fixed_cover=None,
                                                                          mutation_func=swap_mutation,
                                                                          mutation_prob=0.05,
                                                                          crossover_func=random_point_crossover,
                                                                          num_generations=1000)
    custom_algo_run_time = datetime.datetime.now() - custom_algo_start_time
    print("Custom Algo runtime: " + str(custom_algo_run_time))

    article_algo_start_time = datetime.datetime.now()
    (cover_article, cover_count_article) = article_algo.get_min_vertex_cover()
    article_algo_run_time = datetime.datetime.now() - article_algo_start_time
    print("Article Algo runtime: " + str(article_algo_run_time))


if __name__ == "__main__":
    main()
