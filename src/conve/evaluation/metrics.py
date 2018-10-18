from __future__ import absolute_import, division, print_function

import logging
import os
import pickle 

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

def save_obj(obj, fpath):
    #directory = os.getcwd()
    #fpath = os.path.join(directory, 'obj', name)
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fpath):
    #directory = os.getcwd()
    #fpath = os.path.join(directory, 'obj', name)
    with open(fpath, 'rb') as f:
        return pickle.load(f)

# FB15k-237
dataset = 'FB15k-237'
sequence_similarity_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/sequence_similarities.pkl'.format(dataset, dataset)
query_e2s_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/query_e2s.pkl'.format(dataset, dataset)

if os.path.exists(sequence_similarity_path):
    sequence_similarities = load_obj(sequence_similarity_path)
    query_e2s = load_obj(query_e2s_path)
else:
    sequence_similarities = dict()
    query_e2s = dict()


def ranking_and_hits(model, results_dir, data_iterator_handle, name, session=None):
    os.makedirs(results_dir, exist_ok=True)
    LOGGER.info('')
    LOGGER.info('-' * 50)
    LOGGER.info(name)
    LOGGER.info('-' * 50)
    LOGGER.info('')
    if 'dev' in name:
        scores_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/scores_dev.pkl'.format(dataset, dataset)
    else:
        scores_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/scores_test.pkl'.format(dataset, dataset)
    if os.path.exists(scores_path):
        scores = load_obj(scores_path)
        scores_computed = True
    else:
        scores = dict()
        scores_computed = False
    #scores_path = os.path.join(results_dir, 'scores.pkl')
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])

    stopped = False
    tot_seen = 0
    tot_predicted = 0
    while not stopped:
        try:
            e1, e2, rel, e2_multi1, pred1 = \
                session.run(
                    fetches=(
                        model.next_eval_sample['e1'],
                        model.next_eval_sample['e2'],
                        model.next_eval_sample['rel'],
                        model.next_eval_sample['e2_multi1'],
                        model.eval_predictions),
                    feed_dict={
                        model.eval_iterator_handle: data_iterator_handle})

            target_values = pred1[np.arange(0, len(pred1)), e2]
            pred1[e2_multi1 == 1] = -np.inf
            pred1[np.arange(0, len(pred1)), e2] = target_values

            sorted_pred_ids = np.argsort(-pred1, 1)

            for i in range(len(e1)):
                tot_seen += 1
                e1_i, rel_i, e2_i = e1[i], rel[i], e2[i]
                if scores_computed:
                    query = (e1_i, rel_i, e2_i)
                    LOGGER.info("Query: {}".format(query))
                    pred_ids = scores[query]
                else:
                    query = (e1_i, rel_i, e2_i)
                    scores[query] = np.zeros(e2_multi1.shape[1]) #if query not in scores else scores[query]
                    rel_query = (rel_i, )
                    sequences_and_weights = sequence_similarities.get(rel_query, {})
                    sequences_and_weights[rel_query] = 1
                    for sequence, weight in sequences_and_weights.items():
                        pair = (e1_i, sequence)
                        predicted_e2s = query_e2s.get(pair, set())
                        for pred_id in predicted_e2s:
                            scores[query][pred_id] += weight

                    target_score = scores[query][e2[i]]
                    
                    
                    scores[query][e2_multi1[i] == 1] = -np.inf
                    scores[query][e2[i]] = target_score
                    scores[query] = np.argsort(-scores[query])
                    pred_ids = scores[query]

                if query == (7599, 66, 6484):
                    LOGGER.info("HERE: {}".format(pred_ids))
                # cannot predict from simple model
                if max(pred_ids) < 0.01:
                    
                    LOGGER.info("MAX PREDICTED ELEMENT: {}".format(pred_ids))
                    sorted_pred_ids = sorted_pred_ids[i]
                # using predicitons from simpe model
                else:
                    tot_predicted += 1
                    sorted_pred_ids = pred_ids
                
                rank = int(np.where(sorted_pred_ids == e2[i])[0])
                ranks.append(rank + 1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        except tf.errors.OutOfRangeError:
            stopped = True
    LOGGER.info("PROPORTION PREDICTED: {}".format(tot_predicted/tot_seen))
    if not os.path.exists(scores_path):
        save_obj(scores, scores_path)

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
