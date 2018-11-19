from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import tensorflow as tf

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
                     enable_write_to_file=False):
    os.makedirs(results_dir, exist_ok=True)
    logger.info('')
    logger.info('-' * 50)
    logger.info(name)
    logger.info('-' * 50)
    logger.info('')

    hits = {hits_level: [] for hits_level in hits_to_compute}
    ranks = []

    stopped = False
    count = 0
    while not stopped:
        try:
            e1, e2, rel, e2_multi1, pred1, pred_vec = \
                session.run(
                    fetches=(
                        model.next_eval_sample['e1'],
                        model.next_eval_sample['e2'],
                        model.next_eval_sample['rel'],
                        model.next_eval_sample['e2_multi1'],
                        model.eval_predictions,
                        model.eval_prediction_vector),
                    feed_dict={
                        model.eval_iterator_handle: data_iterator_handle})

            target_values = pred1[np.arange(0, len(pred1)), e2]
            pred1[e2_multi1 == 1] = -np.inf
            pred1[np.arange(0, len(pred1)), e2] = target_values
            count += e1.shape[0]
            for i in range(len(e1)):
                pred1_args = np.argsort(-pred1[i])
                rank = int(np.where(pred1_args == e2[i])[0]) + 1
                ranks.append(rank + 1)
                for hits_level in hits_to_compute:
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

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

    return mr, mrr, hits
