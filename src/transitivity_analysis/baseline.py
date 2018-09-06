import copy
import itertools
import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt

########################################################################################################################
################################################# Compute & Assign Ids #################################################
########################################################################################################################

def compute_ids(fpath):
    ent_ids = {}
    rel_ids = {}
    ent_id = 0
    rel_id = 0
    with open(fpath, 'r') as handle:
        for line in handle:
            e1, rel, e2 = line.strip().split("\t")
            if e1 not in ent_ids.keys():
                ent_ids[e1] = ent_ids.get(e1, ent_id)
                ent_id += 1
            if e2 not in ent_ids.keys():
                ent_ids[e2] = ent_ids.get(e2, ent_id)
                ent_id += 1
            if rel not in rel_ids.keys():
                rel_ids[rel] = rel_ids.get(rel, rel_id)
                rel_id += 1

    return ent_ids, rel_ids

def assign_ids(input_path, output_path, ent_ids, rel_ids):
    with open(output_path, 'w+') as output_handle:
        with open(input_path, 'r') as input_handle:
            for line in input_handle:
                e1, rel, e2 = line.strip().split("\t")
                # cannot predict on this instance, set is broken...
                if (e1 not in ent_ids) or (e2 not in ent_ids) or (rel not in rel_ids):
                    continue
                enc_line = "\t".join([str(ent_ids[e1]), str(rel_ids[rel]), str(ent_ids[e2])])
                output_handle.write(enc_line + "\n")

########################################################################################################################
################################################### Create Full Graph ##################################################
########################################################################################################################

# given a file path, returns dictionary look up for (e1, rel) --> e2 for all queries in file.
def get_path_dict(fpath_list):
    path_dict = {}
    for fpath in fpath_list:
        with open(fpath, 'r') as handle:
            for query in handle:
                e1, rel, e2 = list(map(lambda elem: int(elem), query.strip().split("\t")))
                key = (e1, rel)
                if key not in path_dict:
                    path_dict[key] = []
                path_dict[key].append(e2)
    return path_dict

########################################################################################################################
######################################### Extract Information from Training Set ########################################
########################################################################################################################


# stream through train file to get neighbors dictionary and hashmap
# neighbors dict: Dictionary with source entities (keys) and their neighbors or target entities (values)
# hashmap: mapper given a source and target relation, the corresponding relation.
def get_neighbors_and_hashmap(fpath):
    neighbors_dict = {}
    hashmap = {}
    with open(fpath, 'r') as handle:
        for line in handle:
            e1, rel, e2 = list(map(lambda elem: int(elem), line.strip().split("\t")))
            if e1 not in neighbors_dict.keys():
                neighbors_dict[e1] = set()
            neighbors_dict[e1].add(e2)
            # possible that multiple relations from same entity map to same target entity
            if (e1, e2) not in hashmap.keys():
                hashmap[(e1, e2)] = set()
            hashmap[(e1, e2)].add(rel)
    return neighbors_dict, hashmap


# Using neighbors and relations, compute all possible sequences of length up to 3 from each e1. Store this information
# into three dictionaries. The first comprises of all e1 that each sequences originates from. The second is all
# e2 that each (e1, sequence) pair leads to. The third is all sequences tied to each entity:
#   1. {sequence_1: set(e1_1, e1_2, ...), ...}
#   2. {(e1, sequence_1): set(e2_1, e2_2, ...), ...}
#   3. {e1: set(sequence_1, sequence_2, ...), ...}
def compute_sequence_eval_sets(neighbors_dict, hashmap):
    seq_e1_sets = {}
    seq_e1_pair_sets = {}
    e1_seq_set = {}
    total_entities = len(neighbors_dict)
    total_entities_done = 0
    usable_entities = set(neighbors_dict.keys())
    for e1 in usable_entities:
        e1_neighbors = neighbors_dict[e1]

        for e2 in e1_neighbors:
            one_sequence_endpoints = (e1, e2)
            first_connection_relations = hashmap[one_sequence_endpoints]
            for rel1 in first_connection_relations:
                sequence = (rel1,)

                e1_seq_pair = (e1, sequence)
                if sequence not in seq_e1_sets:
                    seq_e1_sets[sequence] = set()
                if e1_seq_pair not in seq_e1_pair_sets:
                    seq_e1_pair_sets[e1_seq_pair] = set()
                if e1 not in e1_seq_set:
                    e1_seq_set[e1] = set()

                seq_e1_sets[sequence].add(e1)
                seq_e1_pair_sets[e1_seq_pair].add(e2)
                e1_seq_set[e1].add(sequence)

                if e2 in usable_entities:
                    e2_neighbors = neighbors_dict[e2]

                    for e3 in e2_neighbors:

                        second_connection = (e2, e3)
                        second_connection_relations = hashmap[second_connection]

                        for rel2 in second_connection_relations:
                            sequence = (rel1, rel2)
                            pair = (e1, sequence)
                            if sequence not in seq_e1_sets:
                                seq_e1_sets[sequence] = set()
                            if pair not in seq_e1_pair_sets:
                                seq_e1_pair_sets[pair] = set()

                            seq_e1_sets[sequence].add(e1)
                            seq_e1_pair_sets[pair].add(e3)
                            e1_seq_set[e1].add(sequence)

                            if e3 in usable_entities:
                                e3_neighbors = neighbors_dict[e3]

                                for e4 in e3_neighbors:

                                    third_connection = (e3, e4)
                                    third_relations = hashmap[third_connection]

                                    for rel3 in third_relations:
                                        sequence = (rel1, rel2, rel3)
                                        pair = (e1, sequence)

                                        if sequence not in seq_e1_sets:
                                            seq_e1_sets[sequence] = set()
                                        if pair not in seq_e1_pair_sets:
                                            seq_e1_pair_sets[pair] = set()

                                        seq_e1_sets[sequence].add(e1)
                                        seq_e1_pair_sets[pair].add(e4)
                                        e1_seq_set[e1].add(sequence)

        total_entities_done += 1.
        percent_done = total_entities_done / float(total_entities) * 100
        print("Percent done: {}".format(percent_done))

    return seq_e1_sets, seq_e1_pair_sets, e1_seq_set


# Compute how similar sequences of length up to 3 are to each relation. It is possible to further hard threshold
# the computed similarity using the seq_threshold variable. Furthermore, if you wish to constrain the sequences you
# look at, you can state which lengths of sequences you'd like to consider for similarity. Given a relation, once you
# compute similarities of all sequences, you can filter ones to keep using seq_threshold and store the remainder
# in a dictionary of sequence_similarities.
def get_rel_seq_sims(e1_seq_set, e1_seq_e2_set, seq_threshold, seq_lengths):
    print("Getting Relation Sequence Similarities....")
    number_relations_passed = 0
    sequence_similarities = {}
    len_1_seqs = []
    valid_sequences = {}
    for sequence in e1_seq_set.keys():
        if len(sequence) == 1:
            len_1_seqs += [sequence]
        if len(sequence) in seq_lengths:
            valid_sequences[sequence] = e1_seq_set[sequence]

    total_relations = len(len_1_seqs)

    for seq1 in len_1_seqs:
        seq1_e1 = e1_seq_set[seq1]

        if seq1 not in sequence_similarities:
            sequence_similarities[seq1] = {}

        gen = ((seq2, seq2_e1) for seq2, seq2_e1 in valid_sequences.items() if (seq1 != seq2))
        for seq2, seq2_e1 in gen:
            intersection_e1 = seq1_e1.intersection(seq2_e1)
            aggregate_similarity = 0

            for e1 in intersection_e1:
                seq1_pair = (e1, seq1)
                seq2_pair = (e1, seq2)

                seq1_e2 = e1_seq_e2_set[seq1_pair]
                seq2_e2 = e1_seq_e2_set[seq2_pair]

                e2_in_common = seq1_e2.intersection(seq2_e2)
                similarity = len(e2_in_common)/len(seq2_e2)
                aggregate_similarity += similarity

            total_subgraphs = len(intersection_e1)
            if total_subgraphs > 0:
                subgraph_similarity = aggregate_similarity/total_subgraphs
            else:
                subgraph_similarity = 0

            if subgraph_similarity >= seq_threshold:
                sequence_similarities[seq1][seq2] = subgraph_similarity

        number_relations_passed += 1.
        percent_done = number_relations_passed/total_relations * 100
        print("Percent done is: {}".format(percent_done))

    return sequence_similarities


########################################################################################################################
##################################################### Prediction #######################################################
########################################################################################################################

# Using the similary sequences computed, predict the target e2s from each test triple. To do this, we simply search all
# sequences that are similar to the test rel, aggregate the results, rank them according to similarity weights, and
# predict HITS@k for K=1, 3, 10. Print accuracies below.
def predict_from_similar_sequences(test_enc_file, sequence_similarities, e1_seq_e2_set, path_dict):
    prediction_weights = {}
    total_correct = 0
    predictable_entities = 0
    total_queries = 0
    top10_correct = 0
    top3_correct = 0
    top1_correct = 0
    with open(test_enc_file, 'r') as handle:
        for test_triple in handle:
            e1, rel, e2_gt = list(map(lambda elem: int(elem), test_triple.strip().split("\t")))
            search_query = (e1, rel)
            prediction_weights[search_query] = {}
            rel_sequence = (rel, )
            sequences_and_weights = sequence_similarities.get(rel_sequence, {})
            sequences_and_weights[rel_sequence] = 1.
            for sequence, weight in sequences_and_weights.items():
                pair = (e1, sequence)
                predicted_e2 = e1_seq_e2_set.get(pair, set())

                for e2_pred in predicted_e2:
                    prediction_weights[search_query][e2_pred] = prediction_weights[search_query].get(e2_pred, 0) + \
                                                                weight
            predicted_e2s = prediction_weights[search_query]
            all_correct_e2s = path_dict[search_query]
            for correct_e2 in all_correct_e2s:
                if correct_e2 != e2_gt:
                    predicted_e2s[correct_e2] = -np.inf

            ranked_predictions = list(map(lambda x: x[0],
                                          sorted(predicted_e2s.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)))

            top10_predictions = ranked_predictions[:10]
            top3_predictions = ranked_predictions[:3]
            top1_predictions = ranked_predictions[:1]
            if e2_gt in top10_predictions:
                top10_correct += 1
            if e2_gt in top3_predictions:
                top3_correct += 1
            if e2_gt in top1_predictions:
                top1_correct += 1

            if len(ranked_predictions) > 1:
                predictable_entities += 1
            if e2_gt in ranked_predictions:
                total_correct += 1

            total_queries += 1

        print("Accuracy is {}, predictable accuracy {}".format(total_correct / total_queries,
                                                               total_correct / predictable_entities))
        print("HITS@10: {0}, HITS@3: {1}, HITS@1: {2}".format(top10_correct / total_queries * 100,
                                                               top3_correct / total_queries * 100,
                                                               top1_correct / total_queries * 100))



########################################################################################################################
###################################################### Explainers ######################################################
########################################################################################################################

# Compute average degrees of all relations
def compute_avg_discrete_rels(neighbors_dict, hashmap):
    num_distinct_rels = 0
    for e1 in neighbors_dict.keys():
        rel_set = set()
        neighbors = neighbors_dict[e1]
        for e2 in neighbors:
            rels = hashmap[(e1, e2)]
            for rel in rels:
                rel_set.add(rel)
        num_distinct_rels += len(rel_set)

    print("Avg discrete rels: {}".format(float(num_distinct_rels)/float(len(neighbors_dict.keys()))))

# print dictionary in key: value per line
def print_dict(d):
    for key in sorted(d.keys()):
        values = d[key]
        # values = dict(sorted(values.items(), key = lambda x: x[1], reverse=True))
        print("{0}: {1}".format(key, values))


# get average degrees from each entity
def get_average_degree(neighbors_dict):
    total_degrees = 0
    for node in neighbors_dict.keys():
        total_degrees += len(neighbors_dict[node])
    print("Avg degree: {}".format(float(total_degrees)/float(len(neighbors_dict.keys()))))


# print number of similar sequences per relations (get understanding of relation spread through graph)
def similarity_lens(similarity_sequences):
    for seq, similar_seqs in similarity_sequences.items():
        print("{}: {}".format(seq, len(similar_seqs)))


########################################################################################################################
#################################################### Saving/Loading ####################################################
########################################################################################################################


def save_obj(obj, name):
    directory = os.getcwd()
    fpath = os.path.join(directory, 'obj', name)
    with open(fpath + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    directory = os.getcwd()
    fpath = os.path.join(directory, 'obj', name)
    with open(fpath + '.pkl', 'rb') as f:
        return pickle.load(f)



dataset = 'FB15K-237'
seq_threshold = 0
ent_threshold = 1
seq_lengths = [2, 3]

train_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/{}/train.txt'.format(dataset)
train_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/{}/train_enc.txt'.format(dataset)
test_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/{}/test.txt'.format(dataset)
test_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/{}/test_enc.txt'.format(dataset)
valid_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/{}/valid.txt'.format(dataset)
valid_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/{}/valid_enc.txt'.format(dataset)

print("{} Thresholding {} , Sequence Lengths are {} is....".format(dataset, seq_threshold, seq_lengths))
print("Computing ids...")
ent_ids, rel_ids = compute_ids(train_file)
print("Computed Ids, assigning ids...")
assign_ids(train_file, train_enc_file, ent_ids, rel_ids)
assign_ids(test_file, test_enc_file, ent_ids, rel_ids)
assign_ids(valid_file, valid_enc_file, ent_ids, rel_ids)

print("Assigned ids, computing neighbor_dict and hashmap...")
neighbors_dict, hashmap = get_neighbors_and_hashmap(train_enc_file)

compute_avg_discrete_rels(neighbors_dict, hashmap)
get_average_degree(neighbors_dict)
print("There are {0} entities and {1} relations".format(len(ent_ids), len(rel_ids)))
print("Obtaining graph path dict...")
path_dict = get_path_dict([train_enc_file, valid_enc_file, test_enc_file])

print("Computing {} Seq Evals....".format(dataset))


seq_e1_sets, seq_e1_pair_sets, e1_seqs_set = compute_sequence_eval_sets(neighbors_dict, hashmap)
save_obj(seq_e1_sets, '{}_seq_e1_set'.format(dataset))
save_obj(seq_e1_pair_sets, '{}_e1_seq_e2_set'.format(dataset))
save_obj(e1_seqs_set, '{}_e1_seq_set'.format(dataset))

seq_e1_sets = load_obj('{}_seq_e1_set'.format(dataset))
seq_e1_pair_sets = load_obj('{}_e1_seq_e2_set'.format(dataset))


print("Computing {} Sequence Similarities...".format(dataset))
sequence_similarities = get_rel_seq_sims(seq_e1_sets,
                                         seq_e1_pair_sets,
                                         seq_threshold=seq_threshold,
                                         seq_lengths=seq_lengths)


# save_obj(sequence_similarities, '{}_threshold_{}_sequence_similarities'.format(dataset, seq_threshold))
# sequence_similarities = load_obj('{}_threshold_{}_sequence_similarities'.format(dataset, seq_threshold))

similarity_lens(sequence_similarities)

print("Predicting {}....".format(dataset))
predict_from_similar_sequences(test_enc_file, sequence_similarities, seq_e1_pair_sets, path_dict)
