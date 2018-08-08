from __future__ import absolute_import, division, print_function

import logging
import math
import os
import random
import six

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

import numpy as np

from fastdtw import fastdtw

from . import graph
from .utils import *

LOGGER = logging.getLogger(__name__)


def exec_struc2vec(args):
    if args['OPT3']:
        until_layer = args['until_layer']
    else:
        until_layer = None
    LOGGER.info("reading graph")
    G = graph.load_from_edge_list(args['input'], undirected=True)
    LOGGER.info("constructing struc2vec")
    G = Graph(G, args['directed'], args['workers'], until_layer=until_layer)
    LOGGER.info("done constructing struc2vec")
    LOGGER.info("preprocessing...")
    if args['OPT1']:
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()
    LOGGER.info("preprocessed")
    LOGGER.info("calculating distances...")
    if args['OPT2']:
        G.create_vectors()
        G.calc_distances(compact_degree=args['OPT1'])
    else:
        G.calc_distances_all_vertices(compact_degree=args['OPT1'])
    LOGGER.info("calculated distances")
    LOGGER.info("creating distances of network...")
    G.create_distances_network()
    LOGGER.info("distances created, preprocessing parameters..")
    G.preprocess_parameters_random_walk()
    LOGGER.info("preprocessed parameters, sumulating walks...")
    G.simulate_walks(args['num_walks'], args['walk_length'])
    LOGGER.info("finished simulating walks, returning...")

    return G


class Graph(object):
    def __init__(self, g, is_directed, workers, until_layer=None):
        self.G = g.gToDict()
        self.num_vertices = g.number_of_nodes()
        self.num_edges = g.number_of_edges()
        self.is_directed = is_directed
        self.workers = workers
        self.calc_until_layer = until_layer

        LOGGER.info('Graph - Number of vertices: {}'.format(self.num_vertices))
        LOGGER.info('Graph - Number of edges: {}'.format(self.num_edges))

    def preprocess_neighbors_with_bfs(self):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(_exec_bfs, self.G, self.workers, self.calc_until_layer)
            job.result()

    def preprocess_neighbors_with_bfs_compact(self):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(_exec_bfs_compact, self.G, self.workers, self.calc_until_layer)
            job.result()

    def preprocess_degree_lists(self):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(_preprocess_degree_lists)
            job.result()

    def create_vectors(self):
        LOGGER.info("Creating degree vectors...")
        degrees = {}
        degrees_sorted = set()
        for v in list(self.G.keys()):
            degree = len(self.G[v])
            degrees_sorted.add(degree)
            if degree not in degrees:
                degrees[degree] = {}
                degrees[degree]['vertices'] = deque()
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        degrees_sorted = np.sort(degrees_sorted)

        degrees_sorted_len = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if index > 0:
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if index < (degrees_sorted_len - 1):
                degrees[degree]['after'] = degrees_sorted[index + 1]
        LOGGER.info("Degree vectors created.")
        LOGGER.info("Saving degree vectors...")
        save_variable_on_disk(degrees, 'degrees_vector')

    def calc_distances_all_vertices(self, compact_degree=False):
        LOGGER.info("Using compact_degree: {}".format(compact_degree))
        if self.calc_until_layer:
            LOGGER.info("Calculations until layer: {}".format(self.calc_until_layer))

        futures = {}
        vertices = list(reversed(sorted(list(self.G.keys()))))
        if compact_degree:
            LOGGER.info("Recovering compact_degree_list from disk...")
            degree_list = restore_variable_from_disk('compact_degree_list')
        else:
            LOGGER.info("Recovering degree_list from disk...")
            degree_list = restore_variable_from_disk('degree_list')

        t0 = time()
        parts = self.workers
        chunks = partition(vertices, parts)
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                LOGGER.info("Executing part {}...".format(part))
                list_v = []
                for v in c:
                    list_v.append(
                        [vd for vd in list(degree_list.keys()) if vd > v])
                job = executor.submit(
                    _calc_distances_all, c, list_v, degree_list, part,
                    compact_degree=compact_degree)
                futures[job] = part
                part += 1

            LOGGER.info("Receiving results...")
            for job in as_completed(futures):
                job.result()
                r = futures[job]
                LOGGER.info("Part {} Completed.".format(r))

        t1 = time()
        LOGGER.info('Distances calculated.')
        LOGGER.info('Time : {}m'.format((t1 - t0) / 60))

    def calc_distances(self, compact_degree=False):
        LOGGER.info("Using compact_degree: {}".format(compact_degree))
        if self.calc_until_layer:
            LOGGER.info("Calculations until layer: {}".format(self.calc_until_layer))

        futures = {}
        vertices = list(self.G.keys())
        chunks = partition(vertices, self.workers)

        with ProcessPoolExecutor(max_workers=1) as executor:
            LOGGER.info("Split degree List...")
            part = 1
            for c in chunks:
                job = executor.submit(_split_degree_list, part, c, self.G, compact_degree)
                job.result()
                LOGGER.info("degreeList {} completed.".format(part))
                part += 1

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for _ in chunks:
                LOGGER.info("Executing part {}...".format(part))
                job = executor.submit(_calc_distances, part, compact_degree=compact_degree)
                futures[job] = part
                part += 1

            LOGGER.info("Receiving results...")
            for job in as_completed(futures):
                job.result()
                r = futures[job]
                LOGGER.info("Part {} completed.".format(r))

    def consolidate_distances(self):
        distances = {}
        for part in range(1, self.workers + 1):
            d = restore_variable_from_disk('distances-' + str(part))
            _preprocess_consolidated_distances(distances)
            distances.update(d)
        _preprocess_consolidated_distances(distances)
        save_variable_on_disk(distances, 'distances')

    def create_distances_network(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(_generate_distances_network, self.workers)
            job.result()

    def preprocess_parameters_random_walk(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(_generate_parameters_random_walk)
            job.result()

    def simulate_walks(self, num_walks, walk_length):
        LOGGER.info('Simulating walks.')

        # For large graphs, we use serial execution, due to memory limitations.
        if len(self.G) > 500000:
            _generate_random_walks_large_graphs(
                num_walks, walk_length, list(self.G.keys()))
        else:
            _generate_random_walks(
                num_walks, walk_length, self.workers, list(self.G.keys()))


def _exec_bfs(G, workers, calc_until_layer):
    futures = {}
    degree_list = {}
    t0 = time()
    vertices = list(G.keys())
    parts = workers
    chunks = partition(vertices, parts)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(
                _get_degree_lists_vertices, G, c, calc_until_layer)
            futures[job] = part
            part += 1
        for job in as_completed(futures):
            dl = job.result()
            degree_list.update(dl)
    LOGGER.info('Saving degree_list on disk...')
    save_variable_on_disk(degree_list, 'degree_list')
    t1 = time()
    LOGGER.info('Execution time - BFS: {}m'.format((t1 - t0) / 60))


def _exec_bfs_compact(G, workers, calc_until_layer):
    futures = {}
    degree_list = {}
    t0 = time()
    vertices = list(G.keys())
    parts = workers
    chunks = partition(vertices, parts)
    LOGGER.info('Capturing larger degree...')
    max_degree = 0
    for v in vertices:
        if len(G[v]) > max_degree:
            max_degree = len(G[v])
    LOGGER.info('Larger degree captured')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(
                _get_compact_degree_lists_vertices, G, c, calc_until_layer)
            futures[job] = part
            part += 1
        for job in as_completed(futures):
            dl = job.result()
            degree_list.update(dl)
    LOGGER.info('Saving degree_list on disk...')
    save_variable_on_disk(degree_list, 'compactDegreeList')
    t1 = time()
    LOGGER.info('Execution time - BFS: {}m'.format((t1 - t0) / 60))


def _get_degree_lists_vertices(g, vertices, calc_until_layer):
    degree_list = {}
    for v in vertices:
        degree_list[v] = _get_degree_lists(g, v, calc_until_layer)
    return degree_list


def _get_compact_degree_lists_vertices(g, vertices, calc_until_layer):
    degree_list = {}
    for v in vertices:
        degree_list[v] = _get_compact_degree_lists(g, v, calc_until_layer)
    return degree_list


def _get_compact_degree_lists(g, root, calc_until_layer):
    t0 = time()
    lists = {}
    vetor_marcacao = [0] * (max(g) + 1)
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}
    depth = 0
    pending_depth_increase = 0
    time_to_depth_increase = 1
    while queue:
        vertex = queue.popleft()
        time_to_depth_increase -= 1
        d = len(g[vertex])
        if d not in l:
            l[d] = 0
        l[d] += 1
        for v in g[vertex]:
            if vetor_marcacao[v] == 0:
                vetor_marcacao[v] = 1
                queue.append(v)
                pending_depth_increase += 1
        if time_to_depth_increase == 0:
            list_d = []
            for degree, freq in six.iteritems(l):
                list_d.append((degree, freq))
            list_d.sort(key=lambda x: x[0])
            lists[depth] = np.array(list_d, dtype=np.int32)
            l = {}
            if calc_until_layer == depth:
                break
            depth += 1
            time_to_depth_increase = pending_depth_increase
            pending_depth_increase = 0
    t1 = time()
    LOGGER.info('BFS vertex {}. Time: {}s'.format(root, t1 - t0))
    return lists


def _get_degree_lists(g, root, calc_until_layer):
    t0 = time()
    lists = {}
    vetor_marcacao = [0] * (max(g) + 1)
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = deque()
    depth = 0
    pending_depth_increase = 0
    time_to_depth_increase = 1
    while queue:
        vertex = queue.popleft()
        time_to_depth_increase -= 1
        l.append(len(g[vertex]))
        for v in g[vertex]:
            if vetor_marcacao[v] == 0:
                vetor_marcacao[v] = 1
                queue.append(v)
                pending_depth_increase += 1
        if time_to_depth_increase == 0:
            lp = np.array(l, dtype='float')
            lp = np.sort(lp)
            lists[depth] = lp
            l = deque()
            if calc_until_layer == depth:
                break
            depth += 1
            time_to_depth_increase = pending_depth_increase
            pending_depth_increase = 0
    t1 = time()
    LOGGER.info('BFS vertex {}. Time: {}s'.format(root, t1 - t0))
    return lists


def _exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length, amount_neighbours):
    original_v = v
    t0 = time()
    initial_layer = 0
    layer = initial_layer
    path = deque()
    path.append(v)
    while len(path) < walk_length:
        r = random.random()
        if r < 0.3:
            # Draw sample from a non-uniform discrete distribution using alias sampling.
            idx = int(np.floor(np.random.rand() * len(alias_method_j[layer][v])))
            if np.random.rand() >= alias_method_q[layer][v][idx]:
                idx = alias_method_j[layer][v][idx]
            path.append(graphs[layer][v][idx])
        else:
            r = random.random()
            x = math.log(amount_neighbours[layer][v] + math.e)
            prob_move_up = x / (x + 1)
            if r > prob_move_up:
                if layer > initial_layer:
                    layer = layer - 1
            else:
                if (layer + 1) in graphs and v in graphs[layer + 1]:
                    layer = layer + 1
    t1 = time()
    LOGGER.info('RW - vertex {}. Time : {}s'.format(original_v,(t1-t0)))
    return path


def _exec_random_walks_for_chunk(
        vertices, graphs, alias_method_j, alias_method_q,
        walk_length, amount_neighbours):
    walks = deque()
    for v in vertices:
        walks.append(_exec_random_walk(
            graphs, alias_method_j, alias_method_q, v,
            walk_length, amount_neighbours))
    return walks


def _generate_random_walks_large_graphs(num_walks, walk_length, vertices):
    LOGGER.info('Loading distances_nets from disk...')
    graphs = restore_variable_from_disk('distances_nets_graphs')
    alias_method_j = restore_variable_from_disk('nets_weights_alias_method_j')
    alias_method_q = restore_variable_from_disk('nets_weights_alias_method_q')
    amount_neighbours = restore_variable_from_disk('amount_neighbours')
    LOGGER.info('Creating RWs...')
    t0 = time()
    walks = deque()
    for walk_iter in range(num_walks):
        random.shuffle(vertices)
        LOGGER.info("Execution iteration {} ...".format(walk_iter))
        walk = _exec_random_walks_for_chunk(
            vertices, graphs, alias_method_j, alias_method_q,
            walk_length, amount_neighbours)
        walks.extend(walk)
        LOGGER.info("Iteration {} executed.".format(walk_iter))

    t1 = time()
    LOGGER.info('RWs created. Time : {}m'.format((t1-t0)/60))
    LOGGER.info("Saving Random Walks on disk...")
    _save_random_walks(walks)


def _generate_random_walks(num_walks, walk_length, workers, vertices):
    LOGGER.info('Loading distances_nets on disk...')
    graphs = restore_variable_from_disk('distances_nets_graphs')
    alias_method_j = restore_variable_from_disk('nets_weights_alias_method_j')
    alias_method_q = restore_variable_from_disk('nets_weights_alias_method_q')
    amount_neighbours = restore_variable_from_disk('amount_neighbours')
    LOGGER.info('Creating RWs...')
    t0 = time()
    walks = deque()
    if workers > num_walks:
        workers = num_walks
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            job = executor.submit(
                _exec_random_walks_for_chunk, vertices, graphs, alias_method_j,
                alias_method_q, walk_length, amount_neighbours)
            # ValueError:job
            # BUGGGg
            futures[job] = walk_iter
        LOGGER.info("Receiving results...")
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            LOGGER.info("Iteration {} executed.".format(r))
            walks.extend(walk)
            del futures[job]
    t1 = time()
    LOGGER.info('RWs created. Time: {}m'.format((t1-t0)/60))
    LOGGER.info("Saving Random Walks on disk...")
    _save_random_walks(walks)


def _save_random_walks(walks):
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v)+' '
            line += '\n'
            file.write(line)


def _split_degree_list(part, c, G, compact_degree):
    if compact_degree:
        LOGGER.info("Recovering compact_degree_list from disk...")
        degree_list = restore_variable_from_disk('compact_degree_list')
    else:
        LOGGER.info("Recovering degree_list from disk...")
        degree_list = restore_variable_from_disk('degree_list')
    LOGGER.info("Recovering degree vector from disk...")
    degrees = restore_variable_from_disk('degrees_vector')
    degree_lists_selected = {}
    vertices = {}
    a_vertices = len(G)
    for v in c:
        nbs = _get_vertices(v, len(G[v]), degrees, a_vertices)
        vertices[v] = nbs
        degree_lists_selected[v] = degree_list[v]
        for n in nbs:
            degree_lists_selected[n] = degree_list[n]
    save_variable_on_disk(vertices, 'split-vertices-' + str(part))
    save_variable_on_disk(degree_lists_selected, 'split-degree_list-' + str(part))


def _verify_degrees(degree_v_root, degree_a, degree_b):
    if degree_b == -1:
        degree_now = degree_a
    elif degree_a == -1:
        degree_now = degree_b
    elif abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root):
        degree_now = degree_b
    else:
        degree_now = degree_a
    return degree_now


def _get_vertices(v, degree_v, degrees, a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices, 2)
    vertices = deque()
    try:
        c_v = 0
        for v2 in degrees[degree_v]['vertices']:
            if v != v2:
                vertices.append(v2)
                c_v += 1
                if c_v > a_vertices_selected:
                    raise StopIteration
        if 'before' not in degrees[degree_v]:
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if 'after' not in degrees[degree_v]:
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if degree_b == -1 and degree_a == -1:
            raise StopIteration
        degree_now = _verify_degrees(degree_v, degree_a, degree_b)
        while True:
            for v2 in degrees[degree_now]['vertices']:
                if v != v2:
                    vertices.append(v2)
                    c_v += 1
                    if c_v > a_vertices_selected:
                        raise StopIteration
            if degree_now == degree_b:
                if 'before' not in degrees[degree_b]:
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if 'after' not in degrees[degree_a]:
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            if degree_b == -1 and degree_a == -1:
                raise StopIteration
            degree_now = _verify_degrees(degree_v, degree_a, degree_b)
    except StopIteration:
        return list(vertices)


def _preprocess_degree_lists():
    LOGGER.info("Recovering degree_list from disk...")
    degree_list = restore_variable_from_disk('degree_list')
    LOGGER.info("Creating compact_degree_list...")
    dList = {}
    dFrequency = {}
    for v, layers in six.iteritems(degree_list):
        dFrequency[v] = {}
        for layer, degreeListLayer in six.iteritems(layers):
            dFrequency[v][layer] = {}
            for degree in degreeListLayer:
                if degree not in dFrequency[v][layer]:
                    dFrequency[v][layer][degree] = 0
                dFrequency[v][layer][degree] += 1
    for v, layers in six.iteritems(dFrequency):
        dList[v] = {}
        for layer, frequencyList in six.iteritems(layers):
            list_d = []
            for degree, freq in six.iteritems(frequencyList):
                list_d.append((degree, freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d, dtype='float')
    LOGGER.info('compact_degree_list created!')
    save_variable_on_disk(dList, 'compact_degree_list')


def _calc_distances_all(
        vertices, list_vertices, degree_list, part, compact_degree=False):
    distances = {}
    cont = 0
    if compact_degree:
        dist_func = _cost_max
    else:
        dist_func = _cost
    for v1 in vertices:
        lists_v1 = degree_list[v1]
        for v2 in list_vertices[cont]:
            lists_v2 = degree_list[v2]
            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1, v2] = {}
            for layer in range(0, max_layer):
                dist, path = fastdtw(
                    lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                distances[v1, v2][layer] = dist
        cont += 1
    _preprocess_consolidated_distances(distances)
    save_variable_on_disk(distances, 'distances-' + str(part))


def _calc_distances(part, compact_degree=False):
    vertices = restore_variable_from_disk('split-vertices-' + str(part))
    degree_list = restore_variable_from_disk('split-degree_list-' + str(part))
    distances = {}
    if compact_degree:
        dist_func = _cost_max
    else:
        dist_func = _cost
    for v1, nbs in six.iteritems(vertices):
        lists_v1 = degree_list[v1]
        for v2 in nbs:
            t00 = time()
            lists_v2 = degree_list[v2]
            max_layer = min(len(lists_v1), len(lists_v2))
            distances[v1, v2] = {}
            for layer in range(0, max_layer):
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                distances[v1, v2][layer] = dist
            t11 = time()
            LOGGER.info(
                'fastDTW between vertices ({}, {}). Time: {}s'.format(
                    v1, v2, (t11 - t00)))
    _preprocess_consolidated_distances(distances)
    save_variable_on_disk(distances, 'distances-' + str(part))


def _cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return (m / mi) - 1


def _cost_min(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * min(a[1], b[1])


def _cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


def _preprocess_consolidated_distances(distances, start_layer=1):
    LOGGER.info('Consolidating distances...')
    for vertices, layers in six.iteritems(distances):
        keys_layers = sorted(list(layers.keys()))
        start_layer = min(len(keys_layers), start_layer)
        for layer in range(0, start_layer):
            keys_layers.pop(0)
        for layer in keys_layers:
            layers[layer] += layers[layer - 1]
    LOGGER.info('Distances consolidated.')


def _generate_parameters_random_walk():
    LOGGER.info('Loading distances_nets from disk...')

    sum_weights = {}
    amount_edges = {}
    layer = 0
    while isPickle('distances_nets_weights-layer-' + str(layer)):
        LOGGER.info('Executing layer {}...'.format(layer))
        weights = restore_variable_from_disk('distances_nets_weights-layer-' + str(layer))
        for k, list_weights in six.iteritems(weights):
            if layer not in sum_weights:
                sum_weights[layer] = 0
            if layer not in amount_edges:
                amount_edges[layer] = 0
            for w in list_weights:
                sum_weights[layer] += w
                amount_edges[layer] += 1
        LOGGER.info('Layer {} executed.'.format(layer))
        layer += 1
    average_weight = {}
    for layer in list(sum_weights.keys()):
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]
    LOGGER.info('Saving average_weights on disk...')
    save_variable_on_disk(average_weight, 'average_weight')

    amount_neighbours = {}
    layer = 0
    while isPickle('distances_nets_weights-layer-' + str(layer)):
        LOGGER.info('Executing layer {}...'.format(layer))
        weights = restore_variable_from_disk('distances_nets_weights-layer-' + str(layer))
        amount_neighbours[layer] = {}
        for k, list_weights in six.iteritems(weights):
            cont_neighbours = 0
            for w in list_weights:
                if w > average_weight[layer]:
                    cont_neighbours += 1
            amount_neighbours[layer][k] = cont_neighbours
        LOGGER.info('Layer {} executed.'.format(layer))
        layer += 1
    LOGGER.info('Saving amount_neighbours on disk...')
    save_variable_on_disk(amount_neighbours, 'amount_neighbours')


def _generate_distances_network_part1(workers):
    parts = workers
    weights_distances = {}
    for part in range(1,parts + 1):

        LOGGER.info('Executing part {}...'.format(part))
        distances = restore_variable_from_disk('distances-' + str(part))

        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance

        LOGGER.info('Part {} executed.'.format(part))

    for layer,values in weights_distances.items():
        save_variable_on_disk(values, 'weights_distances-layer-' + str(layer))
    return


def _generate_distances_network_part2(workers):
    parts = workers
    graphs = {}
    for part in range(1,parts + 1):

        LOGGER.info('Executing part {}...'.format(part))
        distances = restore_variable_from_disk('distances-' + str(part))

        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs):
                    graphs[layer] = {}
                if(vx not in graphs[layer]):
                    graphs[layer][vx] = []
                if(vy not in graphs[layer]):
                    graphs[layer][vy] = []
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
        LOGGER.info('Part {} executed.'.format(part))

    for layer,values in graphs.items():
        save_variable_on_disk(values, 'graphs-layer-' + str(layer))

    return

def _generate_distances_network_part3():

    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        graphs = restore_variable_from_disk('graphs-layer-' + str(layer))
        weights_distances = restore_variable_from_disk('weights_distances-layer-' + str(layer))

        LOGGER.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}

        for v,neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                if (v,n) in weights_distances:
                    wd = weights_distances[v,n]
                else:
                    wd = weights_distances[n,v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = _alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        save_variable_on_disk(weights, 'distances_nets_weights-layer-' + str(layer))
        save_variable_on_disk(alias_method_j, 'alias_method_j-layer-' + str(layer))
        save_variable_on_disk(alias_method_q, 'alias_method_q-layer-' + str(layer))
        LOGGER.info('Layer {} executed.'.format(layer))
        layer += 1

    LOGGER.info('Weights created.')

    return


def _generate_distances_network_part4():
    LOGGER.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        LOGGER.info('Executing layer {}...'.format(layer))
        graphs = restore_variable_from_disk('graphs-layer-' + str(layer))
        graphs_c[layer] = graphs
        LOGGER.info('Layer {} executed.'.format(layer))
        layer += 1


    LOGGER.info("Saving distancesNets on disk...")
    save_variable_on_disk(graphs_c, 'distances_nets_graphs')
    LOGGER.info('Graphs consolidated.')
    return


def _generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while(isPickle('alias_method_j-layer-'+str(layer))):
        LOGGER.info('Executing layer {}...'.format(layer))
        alias_method_j = restore_variable_from_disk('alias_method_j-layer-' + str(layer))
        alias_method_j_c[layer] = alias_method_j
        LOGGER.info('Layer {} executed.'.format(layer))
        layer += 1

    LOGGER.info("Saving nets_weights_alias_method_j on disk...")
    save_variable_on_disk(alias_method_j_c, 'nets_weights_alias_method_j')

    return


def _generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while(isPickle('alias_method_q-layer-'+str(layer))):
        LOGGER.info('Executing layer {}...'.format(layer))
        alias_method_q = restore_variable_from_disk('alias_method_q-layer-' + str(layer))
        alias_method_q_c[layer] = alias_method_q
        LOGGER.info('Layer {} executed.'.format(layer))
        layer += 1

    LOGGER.info("Saving nets_weights_alias_method_q on disk...")
    save_variable_on_disk(alias_method_q_c, 'nets_weights_alias_method_q')

    return


def _generate_distances_network(workers):
    t0 = time()
    LOGGER.info('Creating distance network...')

    os.system("rm "+returnPathStruc2vec()+"/../pickles/weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(_generate_distances_network_part1, workers)
        job.result()
    t1 = time()
    t = t1-t0
    LOGGER.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(_generate_distances_network_part2, workers)
        job.result()
    t1 = time()
    t = t1-t0
    LOGGER.info('- Time - part 2: {}s'.format(t))
    LOGGER.info('distance network created.')

    LOGGER.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(_generate_distances_network_part3)
        job.result()
    t1 = time()
    t = t1-t0
    LOGGER.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(_generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0
    LOGGER.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(_generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0
    LOGGER.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(_generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1-t0
    LOGGER.info('- Time - part 6: {}s'.format(t))


def _alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

