import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import dist
from scipy.interpolate import CubicSpline

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

def curve_fitting(curve):
    y = curve[:, 0]
    x = curve[:, 1]
    z = np.polyfit(x, y, 4)

    p = np.poly1d(z)
    new_y = p(x)

    ##cubic

    a = np.array([0])
    xx = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    xx = np.cumsum(xx)
    xx = (np.c_[np.array([0]), [xx]])[0]
    cs = CubicSpline(xx, curve)
    new_x = np.linspace(0, xx[-1], 20)

    # plt.figure(1)
    # plt.imshow(img)
    # plt.plot(x, y, 'b--')
    # # plt.plot(x, new_y, 'r--')
    # plt.plot(cs(new_x)[:, 1], cs(new_x)[:, 0], 'g--')
    # plt.show()
    # plt.close()
    return cs(new_x)