"""Plots relation and entity embeddings learnt during training."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import yaml

from qa_cpg import data
from qa_cpg.utils.dict_with_attributes import AttributeDict


def prefix(name, suffix='_reverse'):
    return name.rsplit(suffix, 1)[0]


def plot_similarity(embed, embed_type, names, k=10, ignore_reverse=False):
    from scipy.spatial.distance import pdist, squareform
    similarity = squareform(pdist(embed, metric='cosine'))

    if plot_heatmap:
        from .utils.plotting import heatmap
        fig, ax = plt.subplots()
        im, cbar = heatmap(similarity, relations, relations, ax=ax, cmap="YlGn", cbarlabel="cosine dist", annotate=False)
        fig.tight_layout()
        plt.show()

    indices = np.argsort(similarity, axis=None)
    indices = np.unravel_index(indices, similarity.shape)
    print('-' * 100)
    print('Top %d most similar %s:' % (k, embed_type))
    count = 0
    idx = 0
    while count < k and idx < len(indices[0]):
        l = indices[0][idx]
        c = indices[1][idx]
        if l >= c or (ignore_reverse and prefix(names[l]) == prefix(names[c])):
            idx += 1
            continue
        count += 1
        print('%30s %30s %.2f' % (names[l], names[c], similarity[l, c]))
        idx += 1
    print('-' * 100)
    print('Top %d least similar %s:' % (k, embed_type))
    count = 0
    idx = len(indices[0]) - 1
    while count < k and idx >= 0:
        l = indices[0][idx]
        c = indices[1][idx]
        if l >= c:
            idx -= 1
            continue
        count += 1
        print('%30s %30s %.2f' % (names[l], names[c], similarity[l, c]))
        idx -= 1
    print('-' * 100)

plot_heatmap = False

# Provide an embeddings filename, otherwise compute filename automatically from model name inferred from config file.
embeddings_filename = None
relations_filename = None
entities_filename = None
if embeddings_filename is None:
    logging.info('Computing filename automatically from model name inferred from config file...')
    use_cpg = False
    data_loader = data.NationsLoader()

    # Load configuration parameters.
    model_descr = 'cpg' if use_cpg else 'plain'
    config_path = 'qa_cpg/configs/config_%s_%s.yaml' % (data_loader.dataset_name, model_descr)
    with open(config_path, 'r') as file:
        cfg = yaml.load(file)
    print(cfg)
    cfg = AttributeDict(cfg)

    # Compose model name based on config params.
    model_name = '{}-{}-ent_emb_{}-rel_emb_{}-batch_{}'.format(
        model_descr,
        data_loader.dataset_name,
        cfg.model.entity_embedding_size,
        cfg.model.relation_embedding_size,
        cfg.training.batch_size)
    # Add more CPG-specific params to the model name.
    suffix = '-context_batchnorm_{}'.format(cfg.context.context_rel_use_batch_norm) if use_cpg else ''
    model_name += suffix

    # Put together paths.
    working_dir = os.path.join(os.getcwd(), 'temp', data_loader.dataset_name)
    data_dir = os.path.join(working_dir, 'data')
    eval_path = os.path.join(working_dir, 'evaluation', model_name)
    embeddings_filename = os.path.join(eval_path, 'best_embeddings.ckpt')

    # Compute relations and embeddings filename.
    relations_filename = os.path.join(data_dir, data_loader.dataset_name, data_loader.dataset_name, 'relations.txt') # TODO: avoid using the dataset name twice.
    entities_filename = os.path.join(data_dir, data_loader.dataset_name, data_loader.dataset_name, 'entities.txt')

# Load embeddings.
logging.info('Loading embeddings from file...')
rel_embed, ent_embed = pickle.load(open(embeddings_filename, 'rb'))

# Load relation and entity names.
entities = []
relations = []
logging.info('Loading entities from file: %s' % entities_filename)
with open(entities_filename, 'r') as handle:
    for line in handle:
        line = line.strip()
        entities.append(line)
logging.info('Loading relations from file: %s' % relations_filename)
with open(relations_filename, 'r') as handle:
    for line in handle:
        line = line.strip()
        relations.append(line)

# Compute relation embedding similarity matrix.
plot_similarity(rel_embed, 'relations', relations, ignore_reverse=True)
plot_similarity(ent_embed, 'entities', entities)

logging.info('Done.')