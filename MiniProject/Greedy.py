import numpy as np


class GreedyAlgo:

    def __init__(self, adjacency_matrix, edges):
        print("Greedy Algo!")
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges

    def get_min_vertex_cover(self):
        algo_result = greedy_algorithm(self.nodes, self.edges)
        print("Vertex cover of the greedy algorithm consists {} nodes".format(algo_result))

        return algo_result


def greedy_algorithm(nodes, edges):
    visited = np.zeros(nodes)
    cover_count = 0
    for e in edges:
        (u, v) = e
        if (visited[u] == 0) & (visited[v] == 0):
            visited[u] = 1
            visited[v] = 1
            cover_count += 2
    return cover_count
