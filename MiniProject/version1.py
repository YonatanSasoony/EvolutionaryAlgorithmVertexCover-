import matplotlib.pyplot as plt
import random
import copy
import networkx as nx
import numpy as np

def generate_string(length):
    binary_str = ""
    for i in range(0, length):
        next_bit = str(random.randint(0, 1))
        binary_str += next_bit
    return binary_str


# def fitness(s, graph):
#     n = len(graph)
#     for i in range(0, n):
#         for j in range(0, n):
#             if graph[i][j] == 1 and int(s[i]) == 0 and int(s[j]) == 0:
#                 return 1
#     k = 0
#     for i in range(0, n):
#         k += int(s[i])
#     return n - k + 10


def fitness(s, graph):
    n = len(graph)
    sum = 1
    for i in range(0, n):
        for j in range(i, n):
            if graph[i][j] == 1:
                if int(s[i]) == 0 and int(s[j]) == 0:
                    sum += 0
                elif int(s[i]) == 1 and int(s[j]) == 1:
                    sum += 1
                else:
                    sum += 2

    return sum


def selection(str_fit_list):
    length = len(str_fit_list)
    range_list = []
    sum = 0
    for c, fit in str_fit_list:
        range_list += [(c, sum, sum + fit)]
        sum += fit

    str_fit_list.clear()
    for i in range(0, length):
        rand = random.uniform(0, sum)
        for s, start, end in range_list:
            if start <= rand <= end:
                str_fit_list += [(s, 0)]
    return str_fit_list


def crossover(str1, str2):
    length = len(str1)
    mid = length / 2
    child1 = str1[0: int(mid)] + str2[int(mid): int(length)]
    child2 = str2[0: int(mid)] + str1[int(mid): int(length)]
    return child1, child2


def mutation(s):
    length = len(s)
    bit = random.randint(0, length - 1)
    if s[bit] == "0":
        return s[0: bit] + "1" + s[bit + 1: length]
    else:
        return s[0: bit] + "0" + s[bit + 1: length]


def generate_graph(nodes=100):
    edge_probability = 0.0585
    adjacency_matrix = np.zeros((nodes, nodes))
    edges = []
    edges_cnt = 0
    for i in range(nodes):
        for j in range(i):
            prob = random.random()
            if prob < edge_probability:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                edges.append((i, j))
                edges_cnt += 1
    G = nx.Graph()
    G.add_nodes_from(list(range(0,nodes)))
    G.add_edges_from(edges)
    nx.draw(G,node_color='r', node_size=18, alpha=0.8)
    plt.show()
    return adjacency_matrix


def isCover(s, graph):
    n = len(graph)
    for i in range(0, n):
        for j in range(0, n):
            if graph[i][j] == 1 and int(s[i]) == 0 and int(s[j]) == 0:
                return False
    return True


# def main():
graph = [[0, 1, 1, 0, 0, 0],
         [1, 0, 1, 1, 1, 1],
         [1, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0]]

graph = generate_graph(nodes=50)

# graph = np.zeros((20, 20))
# for i in range(20):
#     graph[i, 0] = 1
#     graph[0, i] = 1
graph_length = len(graph)
num_of_covers = 1000  # even number
num_generations = 1000

max_fitness = 0
max_cover = ""
covers = []
temp = []
x_gen = []
y_fit = []
for i in range(0, num_of_covers):  # covers
    c = generate_string(graph_length)
    fit = fitness(c, graph)
    covers += [(c, fit)]
    if fit > max_fitness:
        max_fitness = fit
        max_cover = c
for i in range(0, num_generations):  # generations
    x_gen += [i]
    y_fit += [max_fitness]
    covers = selection(covers)  # select the good covers
    for j in range(0, int(num_of_covers/2)):  # for each pair of covers
        t1 = covers.pop()[0]
        t2 = covers.pop()[0]
        rand1 = random.randint(1, 100)
        if rand1 <= 70:  # 70% do crossover
            childrens = crossover(t1, t2)
            t1 = childrens[0]
            t2 = childrens[1]
        rand2 = random.randint(1, 1000)
        if rand2 == 500:  # 1/1000% do mutation
            t1 = mutation(t1)
            t2 = mutation(t2)
        fit = fitness(t1, graph)
        if fit > max_fitness:
            max_fitness = fit
            max_cover = t1
        temp += [(t1, fit)]
        fit = fitness(t2, graph)
        if fit > max_fitness:
            max_fitness = fit
            max_cover = t2
        temp += [(t2, fit)]
    covers = copy.deepcopy(temp)
    temp.clear()
print("the best cover is: " + str(max_cover))
plt.plot(x_gen, y_fit)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid()
plt.show()
print(max_cover)
#return max_cover


# if __name__ == "__main__":
#     main()

