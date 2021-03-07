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

    def fitness005(cover, edges):
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

def main():

    nodes = 150
    edge_probability = .02
    adjacency_matrix, edges = generate_graph(nodes, edge_probability)
    # draw_graph(edges, nodes)


    # greedy_algo = Greedy.GreedyAlgo(adjacency_matrix, edges)
    # lp_algo = LP.LPAlgo(adjacency_matrix, edges)
    custom_algo = Custom.CustomAlgo(adjacency_matrix, edges)

    # res_greedy = greedy_algo.get_min_vertex_cover()
    # res_lp = lp_algo.get_min_vertex_cover()
    res_custom = custom_algo.get_min_vertex_cover()

    # article_algo = Article.ArticleAlgo(adjacency_matrix, edges, res_greedy, 1000)
    # res_article = article_algo.get_min_vertex_cover()


if __name__ == "__main__":
    main()
