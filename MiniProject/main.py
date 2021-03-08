import numpy as np
import random
import matplotlib.pylab as plt
import networkx as nx
import Article
import Greedy
import LP
import Custom


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


def draw_graph(edges, nodes):
    print("Number of edges {}".format(len(edges)))
    G = nx.Graph()
    G.add_nodes_from(list(range(0, nodes)))
    G.add_edges_from(edges)
    plt.figure(figsize=(12, 6))
    nx.draw(G, node_color='r', node_size=18, alpha=0.8)
    plt.show()


# Fitness #


def fitness005(cover, edges, covers):
    score = 1
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            score += 0
        elif int(cover[u]) == 1 and int(cover[v]) == 1:
            score += 0
        else:
            score += 5
    return score


def fitness_k(cover, edges, covers, k):
    score = 1
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            score += 0
        elif int(cover[u]) == 1 and int(cover[v]) == 1:
            score += 1
        else:
            score += k * len(edges)
    return score


def fitness_cover_bonus(cover, edges, covers):
    score = 1
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            score += 0
        elif int(cover[u]) == 1 and int(cover[v]) == 1:
            score += 0
        else:
            score += 5
    if is_cover(cover, edges):
        score += len(edges)
    return score


def is_cover(cover, edges):
    for edge in edges:
        (u, v) = edge
        if int(cover[u]) == 0 and int(cover[v]) == 0:
            return False
    return True


def fitness_new_article(cover, edges, covers, degree_dict):
    total = total_new_article(covers, degree_dict)
    score = 0
    for cover in covers:
        score += degree_new_article(cover, degree_dict)
    return score / total


def total_new_article(covers, degree_dict):
    total = 0
    for cover in covers:
        total += degree_new_article(cover, degree_dict)
    return total


def degree_new_article(cover, degree_dict):
    degree_sum = 0
    for bit in range(len(cover)):
        if int(cover[bit]) == 1:
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
            child1 += cover1[bit] + cover1[bit+1]
            child2 += cover2[bit] + cover2[bit+1]
        else:
            child1 += cover2[bit] + cover2[bit+1]
            child2 += cover1[bit] + cover1[bit+1]

    return child1, child2


def main():

    nodes = 150
    edge_probability = .02
    adjacency_matrix, edges = generate_graph(nodes, edge_probability)
    # draw_graph(edges, nodes)
    degree_dict = create_degree_dict(nodes, edges)
    greedy_algo = Greedy.GreedyAlgo(adjacency_matrix, edges)
    lp_algo = LP.LPAlgo(adjacency_matrix, edges)
    custom_algo = Custom.CustomAlgo(adjacency_matrix, edges)

    (cover_greedy, cover_count_greedy) = greedy_algo.get_min_vertex_cover()
    (cover_lp, cover_count_lp) = lp_algo.get_min_vertex_cover()
    (cover_custom, cover_count_custom) = custom_algo.get_min_vertex_cover()

    article_algo = Article.ArticleAlgo(adjacency_matrix, edges, cover_count_greedy, 1000)
    (cover_article, cover_count_article) = article_algo.get_min_vertex_cover()

    for i in range(5):
        k = (i + 1) / 10
        # print("FITNESS K " + str(k))
        # (cover_custom, cover_count_custom) =
        # custom_algo.get_min_vertex_cover(fitness_func=lambda cover, edges : fitness_k(cover, edges, k))

    # print("FITNESS COVER BONUS ")
    # res_custom = custom_algo.get_min_vertex_cover(fitness_func=fitness_cover_bonus)
    # print("FITNESS NEW ARTICLE ")
    # (cover_custom, cover_count_custom) = custom_algo.get_min_vertex_cover(fitness_func=lambda cover, edges, covers:
    # fitness_new_article(cover, edges, covers, degree_dict))


if __name__ == "__main__":
    main()
