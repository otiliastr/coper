from __future__ import absolute_import, division, print_function

import six

from collections import defaultdict, Iterable

__all__ = ['Graph', 'load_from_edge_list']


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return list(self.keys())

    def make_consistent(self):
        for k in six.iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        return self

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        return len(self)

    def number_of_edges(self):
        return sum([self.degree(x) for x in list(self.keys())]) / 2

    def number_of_nodes(self):
        return self.order()

    def gToDict(self):
        d = {}
        for k, v in self.items():
            d[k] = v
        return d


def load_from_edge_list(file, undirected=True):
    G = Graph()
    with open(file) as f:
        for l in f:
            if len(l.strip().split()[:2]) > 1:
                x, y = l.strip().split()[:2]
                x = int(x)
                y = int(y)
                G[x].append(y)
                if undirected:
                    G[y].append(x)
            else:
                x = l.strip().split()[:2]
                x = int(x[0])
                G[x] = []
    G.make_consistent()
    return G
