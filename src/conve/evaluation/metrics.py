from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import tensorflow as tf

__all__ = ['ranking_and_hits']

LOGGER = logging.getLogger(__name__)


def _write_data_to_file(file_path, data):
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w+'
    with open(file_path, append_write) as handle:
        handle.write(str(data) + "\n")


def ranking_and_hits(model, results_dir, data_iterator_handle, name, session=None):
    os.makedirs(results_dir, exist_ok=True)
    LOGGER.info('')
    LOGGER.info('-' * 50)
    LOGGER.info(name)
    LOGGER.info('-' * 50)
    LOGGER.info('')

    hits = []
    ranks = []
    for i in range(10):
        hits.append([])

    stopped = False
    while not stopped:
        try:
            e1, e2, rel, e2_multi1, pred1 = \
                session.run(
                    fetches=(
                        model.next_input_sample['e1'],
                        model.next_input_sample['e2'],
                        model.next_input_sample['rel'],
                        model.next_input_sample['e2_multi1'],
                        model.predictions),
                    feed_dict={
                        model.input_iterator_handle: data_iterator_handle,
                        model.input_dropout: 0.0,
                        model.hidden_dropout: 0.0,
                        model.output_dropout: 0.0})

            target_values = pred1[np.arange(0, len(pred1)), e2[:, 0]]
            pred1[e2_multi1 == 1] = -np.inf
            pred1[np.arange(0, len(pred1)), e2[:, 0]] = target_values

            sorted_pred_ids = np.argsort(-pred1, 1)

            for i in range(len(e1)):
                rank = int(np.where(sorted_pred_ids[i] == e2[i, 0])[0])
                ranks.append(rank + 1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        except tf.errors.OutOfRangeError:
            stopped = True

    for i in range(10):
        # write hits to respective files
        hits_at_path = os.path.join(results_dir, 'hits_at_{}.txt'.format(i + 1))
        _write_data_to_file(hits_at_path, np.mean(hits[i]))

        LOGGER.info('Hits @%d: %10.6f', i + 1, np.mean(hits[i]))

    # write mrrs to respective files
    mean_rank = os.path.join(results_dir, 'mean_rank.txt')
    mrr = os.path.join(results_dir, 'mrr.txt')
    _write_data_to_file(mean_rank, np.mean(ranks))
    _write_data_to_file(mrr, np.mean(1. / np.array(ranks)))

    LOGGER.info('Mean rank: %10.6f', np.mean(ranks))
    LOGGER.info('Mean reciprocal rank: %10.6f', np.mean(1. / np.array(ranks)))
