from __future__ import absolute_import, division, print_function

import logging
import os
import pickle 
import scipy as sp
import copy

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
dataset = 'kinship'
threshold = .05
sequence_similarity_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/sequence_similarities.pkl'.format(dataset, dataset)
query_e2s_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/query_e2s.pkl'.format(dataset, dataset)

if os.path.exists(sequence_similarity_path):
    sequence_similarities = load_obj(sequence_similarity_path)
    query_e2s = load_obj(query_e2s_path)
else:
    sequence_similarities = dict()
    query_e2s = dict()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #e_x = np.exp(x - np.max(x))
    e_x = np.exp(x - x.max(axis=0))
    return e_x / e_x.sum(axis=0) # only difference
"""
def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)
"""
def sigmoid(x):
    z = np.exp(-(x))# - x.max(axis=0)))
    return 1 / (1 + z)

def minmax(x):
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min)/(x_max - x_min)


def ranking_and_hits(model, results_dir, data_iterator_handle, name, session=None):
    #global rank1_vectors, rest_vectors, rank2_10_vectors, rank1_e2_multi, rank2_10_e2_multi, rest_e2_multi, rank1_e2, rank2_10_e2, rest_e2, rank1_e1, rank2_10_e1, rest_e1, rank1_samples, rank2_10_samples, rest_samples

    rank1_vectors = list()
    rest_vectors = list()
    rank2_10_vectors = list()

    rank1_e2_multi = list()
    rank2_10_e2_multi = list()
    rest_e2_multi = list()

    rank1_e2 = list()
    rank2_10_e2 = list()
    rest_e2 = list()

    rank1_e1 = list()
    rank2_10_e1 = list()
    rest_e1 = list()

    rank1_samples = list()
    rank2_10_samples =list()
    rest_samples = list()
    
    pred_vecs = list()

    os.makedirs(results_dir, exist_ok=True)
    LOGGER.info('')
    LOGGER.info('-' * 50)
    LOGGER.info(name)
    LOGGER.info('-' * 50)
    LOGGER.info('')
    if 'dev' in name:
        scores_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/scores_dev.pkl'.format(dataset, dataset, str(threshold))
    else:
        scores_path = '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/scores_test_{}.pkl'.format(dataset, dataset, str(threshold))
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
    tot_samples = 0
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
            #for i in range(pred1.shape[0]):
             #   pred1[i][e2_multi1[i]==1] = -np.inf#pred1[i].min(axis=0)
            pred_copy = copy.deepcopy(pred1)
            pred1[e2_multi1 == 1] = -np.inf#pred1.min(axis=1) #-np.inf
            pred1[np.arange(0, len(pred1)), e2] = target_values
            #sorted_pred_ids = np.argsort(-pred1, 1)
            tot_samples += e1.shape[0]
            for i in range(len(e1)):
                #print("Here")
                """
                tot_seen += 1
                e1_i, rel_i, e2_i = e1[i], rel[i], e2[i]
                if scores_computed:
                    query = (e1_i, rel_i, e2_i)
                    #LOGGER.info("Query: {}".format(query))
                    pred_ids = scores.get(query, np.ones(e2_multi1.shape[1]))
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
                    
                    
                    scores[query][e2_multi1[i] == 1] = np.min(scores[query])#-np.inf
                    scores[query][e2[i]] = target_score
                    #scores[query] = np.argsort(-scores[query])
                    pred_ids = scores[query]
                

                pred_ids_args = np.argsort(-pred_ids)
                rank_summer = int(np.where(pred_ids_args == e2[i])[0])
                """
                #print("Pred1 shape is: {}".format(pred1.shape))
                pred1_args = np.argsort(-pred1[i])
                rank = int(np.where(pred1_args == e2[i])[0])
                if rank == 0:
                    rank1_vectors.append(pred_copy[i])
                    rank1_e2_multi.append(e2_multi1[i])
                    rank1_e2.append(e2[i])
                    rank1_e1.append(e1[i])
                    rank1_samples.append((e1[i], rel[i], e2[i]))
                elif rank < 10:
                    rank2_10_vectors.append(pred_copy[i])
                    rank2_10_e2_multi.append(e2_multi1[i])
                    rank2_10_e2.append(e2[i])
                    rank2_10_e1.append(e1[i])
                    rank2_10_samples.append((e1[i], rel[i], e2[i]))
                else:
                    rest_vectors.append(pred_copy[i])
                    rest_e2_multi.append(e2_multi1[i])
                    rest_e2.append(e2[i])
                    rest_e1.append(e1[i])
                    rest_samples.append((e1[i], rel[i], e2[i]))
                pred_vecs.append(pred_vec[i])
                """
                #if query == (7599, 66, 6484):
                 #   LOGGER.info("HERE: {}".format(pred_ids))
                # cannot predict from simple model
                if max(pred_ids) == 0.0:
                    
                    #LOGGER.info("MAX PREDICTED ELEMENT: {}".format(pred_ids))
                    #sorted_pred_ids = sorted_pred_ids[i]
                    rank = int(np.where(sorted_pred_ids[i] == e2[i])[0])
                # using predicitons from simpe model
                else:
                    tot_predicted += 1
                    #sorted_pred_ids = pred_ids
                    rank = int(np.where(pred_ids == e2[i])[0])
                #sorted_pred_ids = sorted_pred_ids[i]
                #rank = int(np.where(sorted_pred_ids == e2[i])[0])
                """
                ranks.append(rank + 1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        except tf.errors.OutOfRangeError:
            stopped = True
    #LOGGER.info("PROPORTION PREDICTED: {}, TOT SEEN: {}, TOT SAMPLES: {}".format(tot_predicted/tot_seen, tot_seen, tot_samples))
    if not os.path.exists(scores_path):
        save_obj(scores, scores_path)

    rank1_vectors = np.array(rank1_vectors)
    rank1_vectors = np.reshape(rank1_vectors, (rank1_vectors.shape[0], e2_multi1.shape[1]))
    rank2_10_vectors = np.array(rank2_10_vectors)
    rank2_10_vectors = np.reshape(rank2_10_vectors, (rank2_10_vectors.shape[0], e2_multi1.shape[1]))
    rest_vectors = np.array(rest_vectors)
    rest_vectors = np.reshape(rest_vectors, (rest_vectors.shape[0], e2_multi1.shape[1]))
    
    rank1_e2_multi = np.reshape(rank1_e2_multi, (len(rank1_e2_multi), e2_multi1.shape[1]))
    rank2_10_e2_multi = np.reshape(rank2_10_e2_multi, (len(rank2_10_e2_multi), e2_multi1.shape[1]))
    rest_e2_multi = np.reshape(rest_e2_multi, (len(rest_e2_multi), e2_multi1.shape[1]))
    
    rank1_e2 = np.array(rank1_e2)
    rank2_10_e2 = np.array(rank2_10_e2)
    rest_e2 = np.array(rest_e2)

    rank1_e1 = np.array(rank1_e1)
    rank2_10_e1 = np.array(rank2_10_e1)
    rest_e1 = np.array(rank1_e1)

    pred_vecs = np.reshape(pred_vec, (len(pred_vec), 200))

    save_obj(rank1_vectors, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank1_vectors.pkl'.format(dataset, dataset))
    save_obj(rest_vectors, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rest_vectors.pkl'.format(dataset, dataset))
    save_obj(rank2_10_vectors, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank2_10_vectors.pkl'.format(dataset, dataset))

    save_obj(rank1_e2_multi, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank1_e2_multi.pkl'.format(dataset, dataset))
    save_obj(rank2_10_e2_multi, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank2_10_e2_multi.pkl'.format(dataset, dataset))
    save_obj(rest_e2_multi, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rest_e2_multi.pkl'.format(dataset, dataset))

    save_obj(rank1_e2, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank1_e2.pkl'.format(dataset, dataset))
    save_obj(rank2_10_e2, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank2_10_e2.pkl'.format(dataset, dataset))
    save_obj(rest_e2, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rest_e2.pkl'.format(dataset, dataset))

    save_obj(rank1_e1, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank1_e1.pkl'.format(dataset, dataset))
    save_obj(rank2_10_e1, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank2_10_e1.pkl'.format(dataset, dataset))
    save_obj(rest_e1, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rest_e1.pkl'.format(dataset, dataset))

    save_obj(rank1_samples, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank1_samples.pkl'.format(dataset, dataset))
    save_obj(rank2_10_samples, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rank2_10_samples.pkl'.format(dataset, dataset))
    save_obj(rest_samples, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/rest_samples.pkl'.format(dataset, dataset))
    
    save_obj(pred_vecs, '/usr0/home/ostretcu/code/qa_types/src/temp/data/{}/{}/pred_vecs.pkl'.format(dataset, dataset))
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
