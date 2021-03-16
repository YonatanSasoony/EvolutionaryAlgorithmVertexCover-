import numpy as np


class GreedyAlgo:

    def __init__(self, adjacency_matrix, edges):
        self.adjacency_matrix = adjacency_matrix
        self.nodes = len(adjacency_matrix)
        self.edges = edges

    def get_min_vertex_cover(self):
        print("Greedy Algo!")
        (cover, cover_count) = greedy_algorithm(self.nodes, self.edges)
        print("Vertex cover of the greedy algorithm consists {} nodes".format(cover_count))
        print("###########################################################################")

        return cover, cover_count


def greedy_algorithm(nodes, edges):
    cover = [0 for node in range(nodes)]
    cover_count = 0
    for e in edges:
        (u, v) = e
        if (cover[u] == 0) & (cover[v] == 0):
            cover[u] = 1
            cover[v] = 1
            cover_count += 2
    return cover, cover_count
