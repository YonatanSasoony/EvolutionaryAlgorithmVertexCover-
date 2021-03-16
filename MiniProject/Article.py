import numpy as np
import random
import math
import gc


class ArticleAlgo:

    def __init__(self, adjacency_matrix, edges, approximation_algo_result, generations_num):
        print("Article Algo!")
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges
        self.approximation_algo_result = approximation_algo_result
        self.generations_num = generations_num

    def get_min_vertex_cover(self):
        (cover, cover_count) = article_algorithm(self.nodes, self.edges, self.approximation_algo_result,
                                                 self.generations_num)
        print("Vertex cover of the Article algorithm consists {} nodes".format(cover_count))
        print("###########################################################################")

        return cover, cover_count


def article_algorithm(nodes, edges, approximation_algo_result, generations_num):
    pop_total = int(50 * max(1, round(nodes / 5.0)))  # max population allowed in the environment
    pop_init = int(20 * max(1, round(nodes / 5.0)))
    # generations_num = int(7 * max(1, round(self.nodes / 5.0)))
    print("N = {}\nPopulation Total = {}\nPopulation Initial = {}\nMax Generations = {}\n".format(nodes,
                                                                                                  pop_total,
                                                                                                  pop_init,
                                                                                                  generations_num))
    free_memory()
    result_dict = mfind(nodes, 0.05, pop_init, pop_total, generations_num, edges,
                        int(approximation_algo_result / 2),
                        nodes)
    print(result_dict.keys())
    cover_count = nodes
    cover = ""
    for k in result_dict.keys():
        cover_count = min(cover_count, k)
        if cover_count == k:
            cover = result_dict[k]
    return cover, cover_count


def chromosomes_gen(n, k, pop_init):
    lst = []
    for i in range(pop_init):
        chromosome = np.zeros(n)
        samples = random.sample(range(0, n), k=k)
        for j in range(k):
            chromosome[samples[j]] = 1
        #         print(chromosome)
        lst.append(chromosome)
    return lst


def cost(cmbn, n, edges):
    obstacles = 0
    for e in edges:
        (u, v) = e
        if (cmbn[u] == 0) & (cmbn[v] == 0):
            obstacles += 1
    return obstacles


def selection(lst, pop_total, n, edges):
    score = []
    output_lst = []
    len_lst = len(lst)
    for i in range(len_lst):
        score.append(cost(lst[i], n, edges))
    sorted_index = np.argsort(score)

    for i in range(len_lst):
        output_lst.append(lst[sorted_index[i]])
        if (i + 1) == pop_total:
            break
    lst = output_lst
    return lst, score[sorted_index[0]]


def helper_print(lst, n):
    res = []
    for i in range(n):
        if lst[i] == 1:
            res.append(i)
    print(res)


def cross_over_mutate_extended(lst, n, k, mutat_prob, pop_total, edges):
    new_lst = lst.copy()
    len_lst = len(lst)
    cross_over_prob = 0.50
    mutat_prob = 0.05
    variations = 1

    # Crossover
    for i in range(len_lst):
        for v in range(variations):
            tmp = lst[i].copy()

            mate_with = lst[int(random.uniform(0, len_lst))]

            tmp_unique = []
            mate_with_unique = []

            for j in range(n):
                if tmp[j] == 1:
                    tmp_unique.append(j)
                if mate_with[j] == 1:
                    mate_with_unique.append(j)

            tmp_unique = np.setdiff1d(tmp, mate_with)
            random.shuffle(tmp_unique)
            mate_with_unique = np.setdiff1d(mate_with, tmp)
            random.shuffle(mate_with_unique)

            swap = math.ceil(cross_over_prob * min(len(tmp_unique), len(mate_with_unique)))

            for j in range(swap):
                tmp[mate_with_unique[j]] = 1
                tmp[tmp_unique[j]] = 0

            # Mutation
            zeroes = []
            ones = []
            for j in range(n):
                if tmp[j] == 1:
                    ones.append(j)
                else:
                    zeroes.append(j)

            random.shuffle(ones)
            random.shuffle(zeroes)

            coin_toss = random.random()
            if coin_toss <= 0.5:
                swaps = min(len(ones), len(zeroes))

                for j in range(swaps):
                    coin_toss2 = random.random()
                    if coin_toss2 < mutat_prob:
                        tmp[ones[j]] = 0
                        tmp[zeroes[j]] = 1
                        # Swapping logic
                        dummy = ones[j]
                        ones[j] = zeroes[j]
                        zeroes[j] = dummy
            else:

                mutate_lst = []
                for e in edges:
                    (u, v) = e
                    if (tmp[u] == 0) & (tmp[v] == 0):
                        coin_toss2 = random.random()
                        if coin_toss2 < mutat_prob:
                            coin_toss3 = random.random()
                            if coin_toss3 <= 0.5:
                                if u not in mutate_lst:
                                    mutate_lst.append(u)
                            else:
                                if v not in mutate_lst:
                                    mutate_lst.append(v)

                random.shuffle(mutate_lst)
                sz = min(len(ones), len(mutate_lst))

                for j in range(sz):
                    tmp[ones[j]] = 0
                    tmp[mutate_lst[j]] = 1
                    # Swapping logic
                    dummy = ones[j]
                    ones[j] = mutate_lst[j]
                    mutate_lst[j] = dummy

            new_lst.append(tmp)

    return new_lst


def environment(n, k, mutation_prob, pop_init, pop_total, max_iterate, edges):
    lst = chromosomes_gen(n, k, pop_init)
    for it in range(max_iterate):
        lst = cross_over_mutate_extended(lst, n, k, mutation_prob, pop_total, edges)
        #         return
        lst, cost_value = selection(lst, pop_total, n, edges)
        if (it % 10) == 9:
            print("k = {}, Generation = {}, Cost = {}".format(k, it + 1, cost_value))
        if cost_value == 0:
            break
    result = []
    soln = lst[0]
    for j in range(len(soln)):
        if soln[j] == 1:
            result.append(j)
    print("k = {}, Generation = {}, Cost = {}\nSoln = {}".format(k, it, cost_value, result))
    return cost_value, result


def free_memory():
    gc.collect()


def mfind(n, mutat_prob, pop_init, pop_total, max_iterate, edges, start, end):
    result_dict = {}
    l = start
    h = end
    ans = 0
    while l <= h:
        m = int((l + h) / 2.0)
        cost_value, result = environment(n, m, mutat_prob, pop_init, pop_total, max_iterate, edges)
        #         print("Cost is {} result is {}".format(cost_value,result))
        if cost_value == 0:
            result_dict[m] = result
            h = m - 1
        else:
            l = m + 1
    return result_dict
