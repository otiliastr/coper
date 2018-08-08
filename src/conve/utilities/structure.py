from __future__ import absolute_import, division, print_function

import os

import numpy as np

__all__ = [
    'load_adjacency_matrix', 'prune_adjacency_matrix',
    'generate_structure_train_file']


def load_adjacency_matrix(path, num_ent):
    print(
        ('Loading the adjacency matrix for %d entities from %s'
         % (num_ent, path)))
    adj_matrix = np.zeros((num_ent, num_ent))
    with open(path, 'r') as walks:
        for walk in walks:
            sequence = walk.strip().split(' ')
            for node_idx in range(len(sequence) - 1):
                src_ent = int(sequence[node_idx])
                tgt_ent = int(sequence[node_idx + 1])
                adj_matrix[src_ent, tgt_ent] += 1
    return adj_matrix


def prune_adjacency_matrix(adj_matrix):
    print('Pruning the adjacency matrix.')
    avg_num_edges = np.mean(adj_matrix, axis=-1)
    avg_num_edges[avg_num_edges <= 0] = 0
    adj_matrix[adj_matrix < avg_num_edges[:, None]] = 0
    adj_matrix[adj_matrix >= avg_num_edges[:, None]] = 1
    return adj_matrix


def generate_structure_train_file(adj_matrix, directory, output_filename):
    print('Generating the structure train file.')
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, output_filename)
    with open(output_file, 'w+') as file:
        for source_entity in range(len(adj_matrix)):
            comparison_row = adj_matrix[source_entity]
            similar_entities = np.where(comparison_row == 1)[0]
            for entity in similar_entities:
                train_line = '{0} {1}\n'.format(source_entity, entity)
                file.write(train_line)
    return output_file
