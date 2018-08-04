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


def ranking_and_hits(model, results_dir, data_iterator_handles, name, session=None):
    os.makedirs(results_dir, exist_ok=True)
    LOGGER.info('')
    LOGGER.info('-' * 50)
    LOGGER.info(name)
    LOGGER.info('-' * 50)
    LOGGER.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    stopped = False
    while not stopped:
        try:
            e1, e2, rel, rel_reverse, e2_multi1, e2_multi2, pred1 = \
            session.run(
                fetches=(
                    model.next_sample['e1'],
                    model.next_sample['e2'],
                    model.next_sample['rel'],
                    model.next_sample['rel_eval'],
                    model.next_sample['e2_multi1'],
                    model.next_sample['e2_multi2'],
                    model.predictions),
                feed_dict={
                    model.input_iterator_handle: data_iterator_handles[0],
                    model.input_dropout: 0.0,
                    model.hidden_dropout: 0.0,
                    model.output_dropout: 0.0})

            pred2 = session.run(
                fetches=model.predictions, 
                feed_dict={
                    model.input_iterator_handle: data_iterator_handles[1],
                    model.input_dropout: 0.0,
                    model.hidden_dropout: 0.0,
                    model.output_dropout: 0.0})

            for i in range(len(e1)):
                # these filters contain ALL labels
                filter1 = np.int32(e2_multi1[i])
                filter2 = np.int32(e2_multi2[i])

                # save the prediction that is relevant
                target_value1 = pred1[i, e2[i, 0]]
                target_value2 = pred2[i, e1[i, 0]]
                # zero all known cases (this are not interesting)
                # this corresponds to the filtered setting
                pred1[i][filter1] = -np.inf
                pred2[i][filter2] = -np.inf
                # write base the saved values
                pred1[i][e2[i]] = target_value1
                pred2[i][e1[i]] = target_value2

            # sort and rank
            argsort1 = np.argsort(pred1, 1)[:, ::-1]
            argsort2 = np.argsort(pred2, 1)[:, ::-1]

            for i in range(len(e1)):
                # find the rank of the target entities
                rank1 = np.where(argsort1[i] == e2[i, 0])[0]
                rank2 = np.where(argsort2[i] == e1[i, 0])[0]
                # rank+1, since the lowest rank is rank 1 not rank 0
                ranks.append(rank1 + 1)
                ranks_left.append(rank1 + 1)
                ranks.append(rank2 + 1)
                ranks_right.append(rank2 + 1)

                # this could be done more elegantly, but here you go
                for hits_level in range(10):
                    if rank1 <= hits_level:
                        hits[hits_level].append(1.0)
                        hits_left[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        hits_left[hits_level].append(0.0)

                    if rank2 <= hits_level:
                        hits[hits_level].append(1.0)
                        hits_right[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        hits_right[hits_level].append(0.0)
        except tf.errors.OutOfRangeError:
            stopped = True

    for i in range(10):
        # write hits to respective files
        hits_right_path = os.path.join(results_dir, 'hits_right_at_{}.txt'.format(i + 1))
        hits_left_path = os.path.join(results_dir, 'hits_left_at_{}.txt'.format(i + 1))
        hits_at_path = os.path.join(results_dir, 'hits_at_{}.txt'.format(i + 1))
        _write_data_to_file(hits_right_path, np.mean(hits_left[i]))
        _write_data_to_file(hits_left_path, np.mean(hits_right[i]))
        _write_data_to_file(hits_at_path, np.mean(hits[i]))

        LOGGER.info('Hits left @%d: %10.6f', i + 1, np.mean(hits_left[i]))
        LOGGER.info('Hits right @%d: %10.6f', i + 1, np.mean(hits_right[i]))
        LOGGER.info('Hits @%d: %10.6f', i + 1, np.mean(hits[i]))

    # write mrrs to respective files
    mean_rank_left = os.path.join(results_dir, 'mean_rank_left.txt')
    mean_rank_right = os.path.join(results_dir, 'mean_rank_right.txt')
    mean_rank = os.path.join(results_dir, 'mean_rank.txt')
    mrr_right = os.path.join(results_dir, 'mrr_right.txt')
    mrr_left = os.path.join(results_dir, 'mrr_left.txt')
    mrr = os.path.join(results_dir, 'mrr.txt')
    _write_data_to_file(mean_rank_left, np.mean(ranks_left))
    _write_data_to_file(mean_rank_right, np.mean(ranks_right))
    _write_data_to_file(mean_rank, np.mean(ranks))
    _write_data_to_file(mrr_right, np.mean(1. / np.array(ranks_right)))
    _write_data_to_file(mrr_left, np.mean(1. / np.array(ranks_left)))
    _write_data_to_file(mrr, np.mean(1. / np.array(ranks)))

    LOGGER.info('Mean rank left: %10.6f', np.mean(ranks_left))
    LOGGER.info('Mean rank right: %10.6f', np.mean(ranks_right))
    LOGGER.info('Mean rank: %10.6f', np.mean(ranks))
    LOGGER.info('Mean reciprocal rank left: %10.6f', np.mean(1. / np.array(ranks_left)))
    LOGGER.info('Mean reciprocal rank right: %10.6f', np.mean(1. / np.array(ranks_right)))
    LOGGER.info('Mean reciprocal rank: %10.6f', np.mean(1. / np.array(ranks)))
