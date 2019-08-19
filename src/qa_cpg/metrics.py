from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import tensorflow as tf
from collections import defaultdict

__all__ = ['ranking_and_hits']

logger = logging.getLogger(__name__)


def _write_data_to_file(file_path, data):
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w+'
    with open(file_path, append_write) as handle:
        handle.write(str(data) + "\n")


def ranking_and_hits(model, results_dir, data_iterator_handle, name, session=None, hits_to_compute=(1, 3, 5, 10, 20),
                     enable_write_to_file=False, get_relation_metrics=False, id_rel_map=None):
    os.makedirs(results_dir, exist_ok=True)
    logger.info('')
    logger.info('-' * 50)
    logger.info(name)
    logger.info('-' * 50)
    logger.info('')

    hits = {hits_level: [] for hits_level in hits_to_compute}
    # Computing relationwise metrics
    if get_relation_metrics:
        relation_hits = defaultdict(lambda: hits.copy())

    ranks = []

    stopped = False
    count = 0
    while not stopped:
        try:
            e1, e2, rel, e2_multi, pred = session.run(
                (model.e1, model.e2, model.rel, model.e2_multi, model.predictions_all),
                feed_dict={model.input_iterator_handle: data_iterator_handle})

            target_values = pred[np.arange(0, len(pred)), e2]
            pred[e2_multi == 1] = -np.inf
            pred[np.arange(0, len(pred)), e2] = target_values
            count += e1.shape[0]
            for i in range(len(e1)):
                pred1_args = np.argsort(-pred[i])
                rank = int(np.where(pred1_args == e2[i])[0]) + 1
                ranks.append(rank)

                if 'ranks' not in relation_hits[rel]:
                    relation_hits[rel]['ranks'] = []
                relation_hits[rel]['ranks'].append(rank)

                for hits_level in hits_to_compute:
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                        if get_relation_metrics:
                            relation_hits[rel][hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        if get_relation_metrics:
                            relation_hits[rel][hits_level].append(0.0)

        except tf.errors.OutOfRangeError:
            stopped = True

    logger.info('Evaluated %d samples.' % count)

    # Save results.
    for hits_level in hits_to_compute:
        hits_value = np.mean(hits[hits_level])
        logger.info('Hits @%d: %10.6f', hits_level, hits_value)
        hits[hits_level] = hits_value
        # Write hits to respective files.
        if enable_write_to_file:
            hits_at_path = os.path.join(results_dir, 'hits_at_{}.txt'.format(hits_level))
            _write_data_to_file(hits_at_path, hits_value)

    if get_relation_metrics and enable_write_to_file:
        hit2str = {1: 'hits_at_1', 3: 'hits_at_3', 5: 'hits_at_5', 10: 'hits_at_10', 20: 'hits_at_20'}
        for rel in relation_hits.keys():
            for hits_level in hits_to_compute:

                relation_save_path = os.path.join(results_dir, '{}_relation_{}.txt'.format(name, hit2str[hits_level]))

                hits_value = np.mean(relation_hits[rel][hits_level])
                relation_hits[rel][hits_level] = hits_value

                _write_data_to_file(relation_save_path, '{}\t{}'.format(id_rel_map[rel], str(hits_value)))

            ranks = relation_hits[rel]['ranks']
            mr = np.mean(ranks)
            mrr = np.mean(1. / np.array(ranks))

            relation_hits[rel]['mr'] = mr
            relation_hits[rel]['mrr'] = mrr

            relation_mr_path = os.path.join(results_dir, '{}_relation_{}.txt'.format(name, 'mr'))
            relation_mrr_path = os.path.join(results_dir, '{}_relation_{}.txt'.format(name, 'mrr'))

            _write_data_to_file(relation_mr_path, '{}\t{}'.format(id_rel_map[rel], str(mr)))
            _write_data_to_file(relation_mrr_path, '{}\t{}'.format(id_rel_map[rel], str(mrr)))

    # Write MRR to respective files.
    mr = np.mean(ranks)
    mrr = np.mean(1. / np.array(ranks))
    logging.info('Mean rank: %10.6f', mr)
    logging.info('Mean reciprocal rank: %10.6f', mrr)
    if enable_write_to_file:
        path_mr = os.path.join(results_dir, 'mean_rank.txt')
        path_mrr = os.path.join(results_dir, 'mrr.txt')
        _write_data_to_file(path_mr, mr)
        _write_data_to_file(path_mrr, mrr)
    logging.info('-' * 50)

    if not get_relation_metrics:
        return mr, mrr, hits, None
    else:
        return mr, mrr, hits, relation_hits
