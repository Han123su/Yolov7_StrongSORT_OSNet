import numpy as np
import networkx as nx

from math import dist

def build_tree_nodes(sk):
    pts, ll, endpoints, new_sk_idx = [], [], [], []
    for i in range(sk.n_paths):
        pts.append(tuple(sk.path_coordinates(i)[0]))
        pts.append(tuple(sk.path_coordinates(i)[-1]))
    for i in range(0, len(pts), 2):
        ll.append((pts[i], pts[i + 1], {"w": dist(pts[i], pts[i + 1])}))

    for i in range(len(pts)):
        cnt = 0
        for j in range(len(pts)):
            if pts[i] != pts[j]:
                cnt += 1
            if cnt == len(pts) - 1:
                endpoints.append(pts[i])
                break

    return nx.Graph(ll), endpoints

def dfs_tree_with_weight(G, source=None, depth_limit=None):
    T = nx.DiGraph()
    if source is None:
        T.add_nodes_from(G)
    else:
        T.add_node(source)
    T.add_edges_from([
        (u, v, G.get_edge_data(u, v))
        for u, v in nx.dfs_edges(G, source, depth_limit)
    ])
    return T

def dfs_longest_path(sk, path_coor, max_path):
    new_sk_idx = []
    for i in range(sk.n_paths):
        for j in range(len(max_path) - 1):
            if (path_coor[i][0] == max_path[j]).all() and (
                    path_coor[i][-1] == max_path[j + 1]).all():
                if new_sk_idx == []:
                    new_sk_idx.extend(path_coor[i])
                else:
                    new_sk_idx.extend(path_coor[i][1:])
    new_sk_idx = np.array(new_sk_idx)
    return new_sk_idx