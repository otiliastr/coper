import numpy as np

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, 
    AddToVocab, StreamToHDF5, CustomTokenizer, 
    ConvertTokenToIdx, ToLower, DictKey2ListMapper
from spodernet.utils.logger import Logger

__all__ = [
    'preprocess_dataset', 'load_adjacency_matrix', 'prune_adjacency_matrix']


def preprocess_dataset(dataset_name, input_keys, delete_data=False):
    """Preprocesses the knowledge graph using Spodernet."""
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1'        # Subject entities
    keys2keys['rel'] = 'rel'      # Relations
    keys2keys['rel_eval'] = 'rel' # Evaluation relations
    keys2keys['e2'] = 'e1'        # Object entities
    keys2keys['e2_multi1'] = 'e1' # ???
    keys2keys['e2_multi2'] = 'e1' # ???

    # Process the full vocabulary and save it to disk.
    streamer = DatasetStreamer(input_keys)
    streamer.add_stream_processor(JsonLoaderProcessors())
    streamer.add_stream_processor(DictKey2ListMapper(input_keys))
    streamer.set_path(full_path)

    pipeline = Pipeline(
        dataset_name, delete_data, keys=input_keys, skip_transformation=True)
    pipeline.add_sent_processor(ToLower())
    pipeline.add_sent_processor(
        CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    pipeline.add_token_processor(AddToVocab())
    pipeline.execute(streamer)
    pipeline.save_vocabs()

    # Process train, dev, and test sets and save them to HDF5 format.
    pipeline.skip_transformation = False
    paths = zip([train_path, dev_path, test_path], ['train', 'dev', 'test'])
    for path, name in paths:
        streamer.set_path(path)
        pipeline.clear_processors()
        pipeline.add_sent_processor(ToLower())
        pipeline.add_sent_processor(
            CustomTokenizer(lambda x: x.split(' ')),
            keys=['e2_multi1', 'e2_multi2'])
        pipeline.add_post_processor(
            ConvertTokenToIdx(keys2keys=keys2keys), keys=input_keys)
        pipeline.add_post_processor(
            StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        pipeline.execute(streamer)


def load_adjacency_matrix(path, num_ent):
    Logger.info(
        'Loading the adjacency matrix for %d entities from %s' 
        % (num_ent, path))
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
    Logger.info('Pruning the adjacency matrix.')
    for row_idx in range(adj_matrix.shape[0]):
        # TODO: Make sure streambatcher bugs don't mess this up.
        if adj_matrix[row_idx].mean() > 0:
            avg_num_edges = adj_matrix[row_idx].mean()
        else:
            avg_num_edges = 1
        similar_idxs = adj_matrix[row_idx] >= avg_num_edges
        disimilar_idxs = adj_matrix[row_idx] < avg_num_edges
        adj_matrix[row_idx, disimilar_idxs] = 0
        adj_matrix[row_idx, similar_idxs] = 1
    return adj_matrix


# def get_embeddings(path):
#     emb_map = {}
#     first_line = True
#     num_rows, num_cols = None, None
#     with open(path, 'r') as file:
#         for line in file:
#             if first_line:
#                 first_line = False
#                 dims = line.split(" ")
#                 num_rows = int(dims[0])
#                 num_cols = int(dims[1])
#                 # embedding = np.zeros((int(dims[0]), int(dims[1])))
#             else:
#                 mapping = line.split(" ")  # entity, embedding1, embedding2, ..., embeddingn
#                 entity = int(mapping[0])
#                 embedding = list(map(lambda elem: float(elem), mapping[1:]))
#                 emb_map[entity] = embedding

#     # sometimes the min index isn't 0 because the Spdernet code is broken.
#     # So we instead get the minimum id and pad our embedding array to
#     # the min_id index.
#     min_idx = min(emb_map)
#     tot_rows = num_rows + min_idx
#     embeddings = np.zeros((tot_rows, num_cols))
#     for e_id in emb_map:
#         embeddings[e_id, :] = emb_map[e_id]
#     return embeddings
