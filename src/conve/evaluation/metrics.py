from __future__ import absolute_import, division, print_function

import datetime
import errno
import os

import numpy as np
import tensorflow as tf

from sklearn import metrics

__all__ = ['ranking_and_hits']


def _write_data_to_file(file_path, data):
    if os.path.exists(file_path):
        append_write = 'a'
    else:
        append_write = 'w+'
    with open(file_path, append_write) as handle:
        handle.write(str(data) + "\n")


def ranking_and_hits(model, model_name, dev_rank_batcher, vocab, name, sess=None):
    log = Logger('evaluation_{0}_{1}.py.txt'.format(model_name, datetime.datetime.now()))
    file_prefix = '/usr0/home/ostretcu/code/george/models/prelim_tests/ConvE/logs/{0}/{1}'.format(model_name, name)
    os.makedirs(file_prefix, exist_ok=True)

    log.info('')
    log.info('-' * 50)
    log.info(name)
    log.info('-' * 50)
    log.info('')
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

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        rel_reverse = str2var['rel_eval']
        e2_multi1 = str2var['e2_multi1']
        e2_multi2 = str2var['e2_multi2']

        pred1 = sess.run(model.predictions, feed_dict={model.e1: e1,
                                                    model.rel: rel,
                                                    model.input_dropout: 0,
                                                    model.hidden_dropout: 0,
                                                    model.output_dropout: 0})

        pred2 = sess.run(model.predictions, feed_dict={model.e1: e2,
                                                    model.rel: rel_reverse,
                                                    model.input_dropout: 0,
                                                    model.hidden_dropout: 0,
                                                    model.output_dropout: 0})
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = np.int32(e2_multi1[i])
            filter2 = np.int32(e2_multi2[i])

            num = e1[i, 0]
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
        # pred1_tensor = torch.from_numpy(pred1)
        # pred2_tensor = torch.from_numpy(pred2)
        argsort1 = np.argsort(pred1, 1)[:, ::-1]
        argsort2 = np.argsort(pred2, 1)[:, ::-1]

        # argsort1 = argsort1.cpu().numpyi()
        # argsort2 = argsort2.cpu().numpy()
        for i in range(Config.batch_size):
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

        dev_rank_batcher.state.loss = [0]

    for i in range(10):
        # write hits to respective files
        hits_right_path = file_prefix + 'hits_right_at_{}.txt'.format(i + 1)
        hits_left_path = file_prefix + 'hits_left_at_{}.txt'.format(i + 1)
        hits_at_path = file_prefix + 'hits_at_{}.txt'.format(i + 1)
        _write_data_to_file(hits_right_path, np.mean(hits_left[i]))
        _write_data_to_file(hits_left_path, np.mean(hits_right[i]))
        _write_data_to_file(hits_at_path, np.mean(hits[i]))

        log.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
        log.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
        log.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))

    # write mrrs to respective files
    mean_rank_left = file_prefix + 'mean_rank_left.txt'
    mean_rank_right = file_prefix + 'mean_rank_right.txt'
    mean_rank = file_prefix + 'mean_rank.txt'
    mrr_right = file_prefix + 'mrr_right.txt'
    mrr_left = file_prefix + 'mrr_left.txt'
    mrr = file_prefix + 'mrr.txt'
    _write_data_to_file(mean_rank_left, np.mean(ranks_left))
    _write_data_to_file(mean_rank_right, np.mean(ranks_right))
    _write_data_to_file(mean_rank, np.mean(ranks))
    _write_data_to_file(mrr_right, np.mean(1. / np.array(ranks_right)))
    _write_data_to_file(mrr_left, np.mean(1. / np.array(ranks_left)))
    _write_data_to_file(mrr, np.mean(1. / np.array(ranks)))

    log.info('Mean rank left: {0}', np.mean(ranks_left))
    log.info('Mean rank right: {0}', np.mean(ranks_right))
    log.info('Mean rank: {0}', np.mean(ranks))
    log.info('Mean reciprocal rank left: {0}', np.mean(1. / np.array(ranks_left)))
    log.info('Mean reciprocal rank right: {0}', np.mean(1. / np.array(ranks_right)))
    log.info('Mean reciprocal rank: {0}', np.mean(1. / np.array(ranks)))
