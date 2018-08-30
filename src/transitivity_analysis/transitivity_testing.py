import copy
import itertools
import numpy as np
import os
import pickle


# fpath = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/train.txt'
# output_path = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/train_ids.txt'
def compute_ids(fpath):
    ent_ids = {}
    rel_ids = {}
    ent_id = 0
    rel_id = 0
    with open(fpath, 'r') as handle:
        for line in handle:
            # print(line.strip().split(" "))
            # TODO Change " " to "\t" when put into production
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
                # TODO Change " " to "\t" when put into production
                e1, rel, e2 = line.strip().split("\t")
                # cannot predict on this instance, set is broken...
                if (e1 not in ent_ids) or (e2 not in ent_ids) or (rel not in rel_ids):
                    continue
                enc_line = "\t".join([str(ent_ids[e1]), str(rel_ids[rel]), str(ent_ids[e2])])
                output_handle.write(enc_line + "\n")

def print_dict(d):
    for key in sorted(d.keys()):
        values = d[key]
        # values = dict(sorted(values.items(), key = lambda x: x[1], reverse=True))
        print("{0}: {1}".format(key, values))


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

# Compute sparse probability matrix store for transitive relations of path equal to 2
# neighbors dict: Dictionary with source entities (keys) and their neighbors or target entities (values)
# hashmap: mapper given a source and target relation, the corresponding relation.
# returns dict of dict. First level contains directed sequence length 2 of relations. Second level contains
#   single relation that maps to sequence and number of times it maps there.
def compute_count_matrix(neighbors_dict, hashmap):
    count_matrix = {}
    total_keys = float(len(neighbors_dict.keys()))
    progress = 0.
    for first_entity in neighbors_dict.keys():
        first_entity_neighbors = neighbors_dict[first_entity]
        for second_entity in first_entity_neighbors:
            # Path from target entity
            if second_entity in neighbors_dict.keys():
                second_entity_neighbors = neighbors_dict[second_entity]
                for third_entity in second_entity_neighbors:
                    endpoints = (first_entity, third_entity)
                    # relationship exists between endpoints
                    if hashmap.get(endpoints, None):
                        first_path = (first_entity, second_entity)
                        second_path = (second_entity, third_entity)
                        first_relations = hashmap[first_path]
                        second_relations = hashmap[second_path]
                        match_relations = hashmap[endpoints]
                        for first_relation in first_relations:
                            for second_relation in second_relations:
                                for match_relation in match_relations:
                                    sequence = (first_relation, second_relation)
                                    # no data for sequence
                                    if not count_matrix.get(sequence, None):
                                        count_matrix[sequence] = {}
                                    count_matrix[sequence][match_relation] = count_matrix[sequence].get(match_relation, 0) + 1
        progress += 1
        percent_done = progress/total_keys
        print("The percent done is {}".format(percent_done))
    return count_matrix

def compute_match_count_matrix(neighbors_dict, hashmap, sample_size = 30):
    count_matrix = {}
    total_keys = len(neighbors_dict)
    total_keys_done = 0
    neighbors_keys_set = set(neighbors_dict.keys())
    for e1 in neighbors_dict.keys():
        # print("looking up e1...")
        e1_neighbors = neighbors_dict[e1]

        e1_useful_neighbors = list(e1_neighbors.intersection(neighbors_keys_set))
        # if len(e1_useful_neighbors) > 0:
        #     e1_neighbors_sample = np.random.choice(e1_useful_neighbors,
        #                                            min(sample_size, len(e1_useful_neighbors)),
        #                                            replace = False)
        # else:
        e1_neighbors_sample = e1_useful_neighbors

        print("---------------------------------------------------------------")
        print("The number of neighbors of e1 are: {}".format(len(e1_neighbors_sample)))
        print("---------------------------------------------------------------")
        e1_neighbor_progress = 0
        for e2 in e1_neighbors_sample:

            one_sequence_endpoints = (e1, e2)
            relations = list(hashmap[one_sequence_endpoints])
            if len(relations) > 1:
                # print("computing 1 sequence relations...")
                for rel_idx in range(len(relations)):
                    rel = relations[rel_idx]
                    sequence = (rel,)
                    matching_rels = relations[:rel_idx] + relations[rel_idx + 1:]

                    if sequence not in count_matrix:
                        count_matrix[sequence] = {}

                    for matching_rel in matching_rels:
                        curr_count = count_matrix[sequence].get(matching_rel, 0)
                        count_matrix[sequence][matching_rel] = curr_count + 1

            # compute at least two sequence transitivity
            if e2 in neighbors_dict.keys():
                e2_neighbors = neighbors_dict[e2]

                e2_useful_neighbors = list(e2_neighbors.intersection(neighbors_keys_set))
                # if len(e2_useful_neighbors) > 0:
                #     e2_neighbors_sample = np.random.choice(e2_useful_neighbors,
                #                                            min(sample_size, len(e2_useful_neighbors)),
                #                                            replace = False)
                # else:
                e2_neighbors_sample = e2_useful_neighbors

                print("    The number of neighbors of e2 are: {}".format(len(e2_neighbors_sample)))
                print("-------------------------------------------------------------------")
                # print("computing 2 sequence relations. There are {} neighbors to parse...".format(len(e2_neighbors)))
                for e3 in e2_neighbors_sample:
                    # continue
                    # compute two sequence transitivity
                    two_sequence_endpoints = (e1, e3)
                    # all_distinct = (e1 != e2) and (e1 != e3) and (e2 != e3)
                    all_distinct = True
                    if (two_sequence_endpoints in hashmap) and (all_distinct):

                        first_connection = (e1, e2)
                        second_connection = (e2, e3)
                        first_relations = hashmap[first_connection]
                        second_relations = hashmap[second_connection]
                        sequence_matches = hashmap[two_sequence_endpoints]
                        for first_relation in first_relations:
                            for second_relation in second_relations:
                                for match_relation in sequence_matches:
                                    # if first_relation != second_relation:
                                    sequence = (first_relation, second_relation)
                                    # no data for sequence
                                    if sequence not in count_matrix:
                                        count_matrix[sequence] = {}

                                    curr_count = count_matrix[sequence].get(match_relation, 0)
                                    count_matrix[sequence][match_relation] = curr_count + 1

                    # compute three sequence transitivity
                    if e3 in neighbors_dict.keys():
                        e3_neighbors = neighbors_dict[e3]

                        e3_useful_neighbors = list(e3_neighbors.intersection(neighbors_keys_set))
                        # if len(e3_useful_neighbors) > 0:
                        #     e3_neighbors_sample = np.random.choice(e3_useful_neighbors,
                        #                                            min(sample_size, len(e3_useful_neighbors)),
                        #                                            replace = False)
                        # else:
                        e3_neighbors_sample = e3_useful_neighbors

                        print("        The number of neighbors of e3 are: {}".format(len(e3_neighbors_sample)))
                        print("-----------------------------------------------------------------------")
                        # print("computing 3 sequence relations. There are {} neighbors to parse...".format(
                        #     len(e3_neighbors)))
                        for e4 in e3_neighbors_sample:
                            # continue
                            three_sequence_endpoints = (e1, e4)
                            all_ent_distinct = True
                            # all_ent_distinct = ((e1 != e2) and
                            #                 (e1 != e3) and
                            #                 (e1 != e4) and
                            #                 (e2 != e3) and
                            #                 (e2 != e4) and
                            #                 (e3 != e4))

                            if (three_sequence_endpoints in hashmap) and (all_ent_distinct):

                                first_connection = (e1, e2)
                                second_connection = (e2, e3)
                                third_connection = (e3, e4)
                                first_relations = hashmap[first_connection]
                                second_relations = hashmap[second_connection]
                                third_relations = hashmap[third_connection]
                                sequence_matches = hashmap[three_sequence_endpoints]
                                print("        Size of first_rels: {0}, second_rels: {1}, third_rels: {2}, match_rels: {3}".
                                      format(len(first_relations),
                                             len(second_relations),
                                             len(third_relations),
                                             len(sequence_matches)))

                                for first_relation in first_relations:
                                    for second_relation in second_relations:
                                        for third_relation in third_relations:
                                            for match_relation in sequence_matches:
                                                # if ((first_relation != second_relation) and
                                                #         (first_relation != third_relation) and
                                                #         (second_relation != third_relation)):
                                                sequence = (first_relation, second_relation, third_relation)
                                                # no data for sequence
                                                if sequence not in count_matrix:
                                                    count_matrix[sequence] = {}

                                                curr_count = count_matrix[sequence].get(match_relation, 0)
                                                count_matrix[sequence][match_relation] = curr_count + 1
            e1_neighbor_progress += 1
            print("The current e1 neighbors progress is {}".format(float(e1_neighbor_progress)/len(e1_neighbors)))
        total_keys_done +=1.
        percent_done = total_keys_done/float(total_keys)
        print("###################################################################")
        print("Percent done: {}".format(percent_done))
        print("###################################################################")
    return count_matrix

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

def get_threshold_match_rels(count_matrix, threshold, num_rel):
    match_synonyms = {}
    for sequence in count_matrix:
        matches = count_matrix[sequence]
        if threshold == 'one':
            threshold = 1
        elif threshold == 'avg':
            threshold = float(sum(matches.values()))/float(num_rel)
        # get all synonyms
        synomyms = []
        for match_rel in matches.keys():
            match_freq = matches[match_rel]
            if match_freq >= threshold:
                synomyms.append(match_rel)
        # add synonyms to each other
        for rel_idx in range(len(synomyms)):
            # extract current relation
            start_rel = synomyms[rel_idx]
            # get synonyms of relation
            rel_synonyms = synomyms[:rel_idx] + synomyms[(rel_idx+1):]
            # current relation not in match dictionary
            if not match_synonyms.get(start_rel, False):
                match_synonyms[start_rel] = {}
            # add synonyms conditioned on sequence
            match_synonyms[start_rel][sequence] = rel_synonyms

    return match_synonyms

def get_match_sequences_from_counts(count_matrix):
    match_sequences = {}
    for sequence in count_matrix:
        matches = count_matrix[sequence].keys()
        for match in matches:
            match_sequences[match] = match_sequences.get(match, []) + [sequence]
    return match_sequences

def get_match_sequences_from_synonyms(match_synonyms):
    match_sequences = {}
    for rel in match_synonyms:
        sequences = match_synonyms[rel].keys()
        match_sequences[rel] = sequences
    return match_sequences

# def get_set_descripancies(queries, train_data, train_dict, match_sequences):
#     _, train_data_set = predict_using_synonyms(queries, train_data, match_sequences)
#     _, train_dict_set = predict_using_synonyms_train_dict(queries, train_dict, match_sequences)
#     print("Differences: {}".format(train_data_set - train_dict_set))


# given a file path, returns dictionary look up for (e1, rel) --> e2 for all queries in file.
def get_train_dict(fpath):
    train_dict = {}
    with open(fpath, 'r') as handle:
        for query in handle:
            e1, rel, e2 = list(map(lambda elem: int(elem), query.strip().split("\t")))
            key = (e1, rel)
            if key not in train_dict:
                train_dict[key] = []
            train_dict[key].append(e2)
    return train_dict


def train_dict_comp_file(train_dict, file):
    train_list = []
    count = 0
    train_set = set()
    file_set = set()
    for e1_rel in train_dict:
        e1 = e1_rel[0]
        rel = e1_rel[1]
        e2s = train_dict[e1_rel]
        for e2 in e2s:
            train_list.append("{0}\t{1}\t{2}\n".format(e1, rel, e2))
    print("The length of the train_list is {}".format(len(train_list)))
    print("The length of the file is {}".format(len(file)))
    for idx in range(len(file)):
        file_line = file[idx]
        train_line = train_list[idx]
        if file_line != train_file:
            # print("file line: {}, train line: {}".format(file_line, train_line))
            train_set.add(train_line)
            file_set.add(file_line)
            count += 1
    print("total inconsistencies: {}".format(count))
    print("Missing from train_dict: {}".format(file_set - train_set))
    print("Missing from file: {}".format(train_set - file_set))

########################################################################################################################
########################################### Relation Similarity Computations ###########################################
########################################################################################################################

# Convert the count_matrix to relation sequence matrix. So instead of first level being
# sequence then relation, have first level be relation then sequence.
def count_matrix_to_rel_seq_matrix(count_matrix):
    rel_seq_matrix = {}
    # {seq: {rel_1:freq_1, rel_2: freq_2, ...}, ...}  --> {seq: rel_freqs, ...}
    for seq, rel_freqs in count_matrix.items():
        # rel_freqs --> {rel_1: freq_1, rel_2:freq_2, ...}
        for rel, freq in rel_freqs.items():
            if rel not in rel_seq_matrix:
                rel_seq_matrix[rel] = {}
            rel_seq_matrix[rel][seq] = freq
    return rel_seq_matrix

# Compute min-max normalization scaling between 0-1. Add 1 to both numerator and
# denominator to avoid possible division by 0. Return normalized x
def min_max_norm(x_list):
    normed_x = []
    min_x, max_x = min(x_list), max(x_list)
    for elem in x_list:
        normed_elem = (elem - min_x + 1)/(max_x-min_x + 1)
        normed_x += [normed_elem]
    return normed_x

# returns dictionary same format as count matrix except instead of match frequency,
# we have normalized weight for specific match relation.
def get_sequence_norms(rel_seq_matrix):
    rel_seq_norms  = {}
    # {seq: {rel_1:freq_1, rel_2: freq_2, ...}, ...}  --> {seq: rel_freqs, ...}
    for rel, seq_freqs in rel_seq_matrix.items():
        seqs = list(seq_freqs.keys())
        freqs = list(seq_freqs.values())
        # print("Relation is: {}, seq_freqs: {}".format(rel, seq_freqs))
        normed_freqs = min_max_norm(freqs)
        # {rel: {seq_1: norm_1, seq_2: norm_2, ...}, ...}
        rel_seq_norms[rel] = {seq: normed_freq for (seq, normed_freq) in zip(seqs, normed_freqs)}
    return rel_seq_norms

# get relation sequence similarities from count_matrix. These are conditional
# similarities between two relations conditioned on the sequence under which they
# match. Returns relation conditional similarities
def get_rel_seq_similarities(count_matrix):
    rel_seq_sim_mat = {}
    # compute sequence relation norms
    rel_seq_norm = get_sequence_norms(count_matrix_to_rel_seq_matrix(count_matrix))
    # {seq: {rel_1:freq_1, rel_2: freq_2, ...}, ...}  --> {seq: rel_freqs, ...}
    for seq, rel_freqs in count_matrix.items():
        # rel_freqs --> {rel_1: freq_1, rel_2:freq_2, ...}
        for rel_1, freq_1 in rel_freqs.items():
            # print("Rel: {}, Seq: {}".format(rel_1, seq))
            # create tuple of sequence and the frequency of its matches with rel_1
            seq_weight = (seq, rel_seq_norm[rel_1][seq])
            # initialize elements if they don't exist
            if rel_1 not in rel_seq_sim_mat:
                rel_seq_sim_mat[rel_1] = {}
            if seq_weight not in rel_seq_sim_mat[rel_1]:
                rel_seq_sim_mat[rel_1][seq_weight] = {}
            # create generator for relation comparisons
            gen = ((rel_2, freq_2) for rel_2, freq_2 in rel_freqs.items() if rel_2 != rel_1)
            for rel_2, freq_2 in gen:
                # compute similarity
                similarity = min(freq_1, freq_2)/max(freq_1, freq_2)
                rel_seq_sim_mat[rel_1][seq_weight][rel_2] = similarity
    return rel_seq_sim_mat

# Given the relation conditional similarities, compute the total weight for
# similar relations to original one.
def get_rel_similarities(rel_seq_sim_mat):
    rel_sim_mat = {}
    # {rel: {(seq_1, norm_1):{rel_2: rel_2_sim_rel_1, ...}, ...}, ...}
    for rel_1, seq_rels in rel_seq_sim_mat.items():
        rel_sim_mat[rel_1] = {}
        for seq_weight, rels_sim in seq_rels.items():
            weight = seq_weight[1]
            # {rel_2: rel_2_sim_rel_1, ...}
            for rel_2, sim in rels_sim.items():
                # multiple normed sequence weight by relation similarity weight
                weighted_sim = weight * sim
                if rel_2 not in rel_sim_mat[rel_1]:
                    rel_sim_mat[rel_1][rel_2] = 0
                # aggregate all weights
                rel_sim_mat[rel_1][rel_2] += weighted_sim
    return rel_sim_mat


# given two relation sets from separate nodes, compute how similar the two sets
# are according to the threshold. This is hard assignment here.
def get_set_similarity(set_one, set_two, threshold):
    set_intersection = set_one.intersection(set_two)
    intersection_len = float(len(set_intersection))
    set_one_len = float(len(set_one))
    if (intersection_len/set_one_len) >= threshold:
        are_similar = 1
    else:
        are_similar = 0

    return are_similar

# given the relation connections dict and a threshold, create dictionary of all ents and their relation similar others.
#   - {e1: [rel_sim_e2, rel_sim_e3, ...], ...}
def get_sim_rel_ents(rel_connections, threshold):
    sim_rel_ents = {}
    for comp_ent_1 in rel_connections.keys():
        sim_rel_ents[comp_ent_1] = []
        for comp_ent_2 in rel_connections.keys():
            if comp_ent_1 != comp_ent_2:
                are_similar = get_set_similarity(rel_connections[comp_ent_1], rel_connections[comp_ent_2], threshold)
                if are_similar:
                    sim_rel_ents[comp_ent_1].append(comp_ent_2)
        sim_rel_ents[comp_ent_1] = sorted(sim_rel_ents[comp_ent_1])
    return sim_rel_ents

########################################################################################################################
############################################## Compute Entity Similarity ###############################################
########################################################################################################################


# Compute first degree relation connections. This is basically a neighbors dictionary
# where instead of a dictionary of neighboring entities, we have a dictionary of
# outgoing relations from each node
def compute_first_degree_rel_connections(train_enc_file):
    rel_connections = {}
    with open(train_enc_file, 'r') as handle:
        for line in handle:
            e1, rel, e2 = list(map(lambda elem: int(elem), line.strip().split("\t")))
            if e1 not in rel_connections:
                rel_connections[e1] = set()
            rel_connections[e1].add(rel)
    return rel_connections

def get_ent_sim(set_1, set_2):
    intersection_len = len(set_1.intersection(set_2))
    set_1_len = len(set_1)
    similarity = float(intersection_len)/float(set_1_len)
    return similarity

def compute_ent_sim_matrix(rel_connections):
    ent_similarities = {}
    gen_1 = ((ent, rel_set) for ent, rel_set in rel_connections.items())
    for ent_1, rel_set_1 in gen_1:
        gen_2 = ((ent, rel_set) for ent, rel_set in rel_connections.items() if ent != ent_1)
        for ent_2, rel_set_2 in gen_2:
            similarity = get_ent_sim(rel_set_1, rel_set_2)
            if ent_1 not in ent_similarities.keys():
                ent_similarities[ent_1] = {}
            ent_similarities[ent_1][ent_2] = similarity
    return ent_similarities

########################################################################################################################
########################################################################################################################

def prediction(test_enc_file, train_dict, ent_similarities, rel_similarities):
    prediction_ranks = {}
    print("Predictions: ")
    total_correct = 0
    total_queries = 0
    with open(test_enc_file, 'r') as handle:
        for test_query in handle:
            total_queries += 1
            e1, rel, e2 = list(map(lambda elem: int(elem), test_query.strip().split("\t")))
            sim_rels = rel_similarities[rel]
            sim_ents = ent_similarities[e1]
            # set relation and entity weights of test query
            sim_rels[rel] = 1
            sim_ents[e1] = 1
            # initialize test query rank dict
            if (e1, rel) not in prediction_ranks.keys():
                prediction_ranks[(e1, rel)] = {}
            proposed_ent_set = set()
            # print("Query is {}".format((e1, rel)))
            # print("There are {} ents, {} rels".format(len(sim_ents), len(sim_rels)))
            for sim_rel in sim_rels.keys():
                for sim_ent in sim_ents.keys():
                    query = (sim_ent, sim_rel)
                    # print("The query is {}".format(query))
                    pair_weight = sim_rels[sim_rel] * sim_ents[sim_ent]
                    # print("the rel weight is {} and the ent weight is {}, pair weight is: {}".format(sim_rels[sim_rel], sim_ents[sim_ent]))
                    proposed_ents = set(train_dict.get(query, []))
                    # print("The proposed ents are: {}".format(proposed_ents))
                    proposed_ent_set = proposed_ent_set.union(proposed_ents)
                    # print("There are {} proposed ents".format(len(proposed_ent_set)))
                    for proposed_ent in proposed_ents:
                        if proposed_ent not in prediction_ranks[(e1, rel)].keys():
                            prediction_ranks[(e1, rel)][proposed_ent] = 0
                        # if proposed_ent == 87:
                            # print("The rank for ent 87 is: {}".format(prediction_ranks[(e1, rel)][proposed_ent]))
                        prediction_ranks[(e1, rel)][proposed_ent] += pair_weight
                        # print("Prediction ranks for (e1, rel) are {}".format(prediction_ranks[(e1, rel)]))

            # print("The query is {}".format(query))
            # Break


            proposed_ent_ranks = prediction_ranks[(e1, rel)]
            sorted_ranks = sorted(proposed_ent_ranks.items(), key = lambda x: x[1], reverse = True)
            sorted_ranks_trimmed = dict(sorted_ranks[:10])
            print("Top 10 Ranks for Query {}: {}".format((e1, rel), sorted_ranks_trimmed))
            if e2 in sorted_ranks_trimmed:
                total_correct += 1
    print("Accuracy is {}".format(total_correct/total_queries))


def predict_using_synonyms(queries, train_dict, match_sequences, rel_ent_sim):
    flatten = itertools.chain.from_iterable
    tot_correct = 0
    tot_neg = 0
    tot_lookups = 0
    correct_set = set()
    total_queries = len(queries)
    queries_computed = 0.
    for query in queries:
        predicted_e2 = []
        e1, rel, e2_gt = list(map(lambda elem: int(elem), query.strip().split("\t")))
        if rel in match_sequences:
            synonyms = list(flatten(match_sequences[rel].values())) + [rel]
            sim_ents = rel_ent_sim.get(e1, [])
            for synonym in synonyms:
                for ent in sim_ents:
                    incomplete_query = (ent, synonym)
                    predicted_e2 += train_dict.get(incomplete_query, [])


            if len(predicted_e2) == 0:
                tot_neg += 1.
            tot_lookups += 1.
        # if len(predicted_e2) != 0:
        #     # e2_pred = np.argmax(np.bincount(predicted_e2))
        #     e2_pred = predicted_e2
        if (e1, rel) == (99, 20):
            print("The prediction are: {}".format(set(predicted_e2)))
            return
        is_correct = e2_gt in predicted_e2
        if is_correct:
            query = (e1, rel, e2_gt)
            correct_set.add(query)
        tot_correct += is_correct
        queries_computed += 1.
        print("The progress is {}".format(queries_computed/total_queries))
    print("total -1: {}, total lookups: {}".format(tot_neg, tot_lookups))
    print("tot correct: {}".format(tot_correct))
    return float(tot_correct)/float(total_queries)

def predict_using_sequences(test_enc_file, train_dict, sequence_norms):
    prediction_ranks = {}
    total_correct = 0
    predictable_entities = 0
    total_queries = 0
    top10_correct = 0
    top3_correct = 0
    top1_correct = 0
    with open(test_enc_file, 'r') as handle:
        for test_query in handle:
            e1, rel, e2_gt = list(map(lambda elem: int(elem), test_query.strip().split("\t")))
            partial_query = (e1, rel)
            prediction_ranks[partial_query] = {}
            sequences = sequence_norms.get(rel, {})
            sorted_sequences_trimmed = sorted(sequences.items(), key=lambda x:x[1], reverse=True)
            for sequence, norm in sorted_sequences_trimmed:
                source_ents = [e1]
                actual_targets = set()
                # actual_targets = {}
                for seq_rel_idx in range(len(sequence)):
                    seq_rel = sequence[seq_rel_idx]
                    for source_ent in source_ents:
                        query = (source_ent, seq_rel)
                        targets = train_dict.get(query, [])
                        source_ents = targets
                        if seq_rel_idx == len(sequence) - 1:
                            # for target in targets:
                            #     actual_targets[target] = actual_targets.get(target, 0) + 1
                            actual_targets = actual_targets.union(set(targets))
                for e2_pred in actual_targets:
                    if e2_pred not in prediction_ranks[partial_query].keys():
                        prediction_ranks[partial_query][e2_pred] = 0
                    # prediction_ranks[partial_query][e2_pred] += norm * actual_targets[e2_pred]
                    prediction_ranks[partial_query][e2_pred] += norm
            trimmed_ranks = list(map(lambda x: x[0],
                                     sorted(prediction_ranks[partial_query].items(),
                                            key=lambda x:x[1],
                                            reverse=True)))
            # print("The Ranked Predictions of {} are: {}".format(partial_query, trimmed_ranks))
            if len(trimmed_ranks) != 0:
                predictable_entities += 1
            if e2_gt in trimmed_ranks:
                total_correct += 1
            if e2_gt in trimmed_ranks[:1]:
                top1_correct += 1
            if e2_gt in trimmed_ranks[:3]:
                top3_correct += 1
            if e2_gt in trimmed_ranks[:10]:
                top10_correct += 1
            if e2_gt not in trimmed_ranks[:10]:
                print("Unmatched test triple is: {}".format((e1, rel, e2_gt)))

            total_queries += 1
        print("Accuracy is {}, predictable accuracy {}".format(total_correct/total_queries,
                                                               total_correct/predictable_entities))
        print("HITS@10: {0}, HITS@3: {1}, HITS@10: {2}".format(top10_correct/total_queries*100,
                                                               top3_correct/total_queries*100,
                                                               top1_correct/total_queries*100))


def get_count_matrix_distribution(count_matrix):
    one_sequence_count = 0
    two_sequence_count = 0
    three_sequence_count = 0
    for sequence in count_matrix:
        if len(sequence) == 1:
            one_sequence_count += 1
        elif len(sequence) == 2:
            two_sequence_count += 1
        elif len(sequence) == 3:
            three_sequence_count += 1
        else:
            print("Faulty sequence: {}".format(sequence))
    print("of the {0} sequences in the dataset, {1} are 1 relation, {2} are 2 relations, and {3} are three relations".format(
        len(count_matrix.keys()), one_sequence_count, two_sequence_count, three_sequence_count
    ))

def write_dict(d, file):
    with open(file, 'w+') as handle:
        for key in sorted(d.keys()):
            value = d[key]
            line = "{0}: {1}\n".format(key, value)
            handle.write(line)

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

def get_average_degree(neighbors_dict):
    total_degrees = 0
    for node in neighbors_dict.keys():
        total_degrees += len(neighbors_dict[node])
    print("Avg degree: {}".format(float(total_degrees)/float(len(neighbors_dict.keys()))))

# train_enc_path = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/train_ids.txt'
# transitivity_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/transitivity_dict.txt'
# train_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/train.txt'
# train_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/train_enc.txt'
# test_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/test.txt'
# test_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/kinship/test_enc.txt'
# FB15K-237
train_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/Kinship/train.txt'
train_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/Kinship/train_enc.txt'
test_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/Kinship/test.txt'
test_enc_file = '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/Kinship/test_enc.txt'

print("Computing ids...")
ent_ids, rel_ids = compute_ids(train_file)
# print("number of entity: {}".format(len(ent_ids)))
# print_dict(ent_ids)
# print("Relation ids: {}".format(len(rel_ids)))
# print_dict(rel_ids)
print("Computed Ids, assigning ids...")
# assign_ids(train_file, train_enc_file, ent_ids, rel_ids)
# assign_ids(test_file, test_enc_file, ent_ids, rel_ids)
#
print("Assigned ids, computing neighbor_dict and hashmap...")
neighbors_dict, hashmap = get_neighbors_and_hashmap(train_enc_file)
compute_avg_discrete_rels(neighbors_dict, hashmap)
get_average_degree(neighbors_dict)
print("There are {0} entities and {1} relations".format(len(ent_ids), len(rel_ids)))
# Break
# print_dict(neighbors_dict)
count_matrix = compute_count_matrix(neighbors_dict, hashmap)

# obtaining relation similarities
# rel_seq_norm = get_sequence_norms(count_matrix)
# print("Printing rel_seq_norm...")
# print_dict(rel_seq_norm)
# print("Done printing")
rel_seq_matrix = count_matrix_to_rel_seq_matrix(count_matrix)
rel_seq_norm = get_sequence_norms(rel_seq_matrix)
print("Relation sequence norms...")
print_dict(rel_seq_norm)

# rel_seq_sim_mat = get_rel_seq_similarities(count_matrix)
# print_dict(rel_seq_sim_mat)
# rel_similarities = get_rel_similarities(rel_seq_sim_mat)
# print_dict(rel_similarities)
# obtain entity similarities
# train_rel_connections = compute_first_degree_rel_connections(train_enc_file)
# ent_similarities = compute_ent_sim_matrix(train_rel_connections)
print("###################################################")
print("entity similarities:")
# print_dict(ent_similarities)
# print(sorted(ent_similarities[82].items(), key = lambda x: x[1], reverse=True))
# obtaining training dictionary
train_dict = get_train_dict(train_enc_file)
print("###################################################")
print("Train Dict:")
# print_dict(train_dict)
# prediction(test_enc_file, train_dict, ent_similarities, rel_similarities)
predict_using_sequences(test_enc_file, train_dict, rel_seq_norm)
# Break
# match_synonyms = get_threshold_match_rels(count_matrix, 'avg', len(rel_ids))
# print("Computed dicts, computing count_matrix...")
# count_matrix = compute_match_count_matrix(neighbors_dict, hashmap)
# end_time = time.time()
# print("Computed count_matrixs, saving it...")
# save_obj(count_matrix, 'count_matrix_FB15k-237')
# print("Saved, writing count_matrix to txt file...")
# write_dict(count_matrix, '/Users/georgestoica/Desktop/Research/struc_hard_enforcment/FB15k-237/count_matrix_.txt')
# count_matrix = load_obj('count_matrix_FB15K-237')
# print("Saved, computing distribution....")
# get_count_matrix_distribution(count_matrix)
# print("Getting synonyms...")

# print("Getting sequences....")
# train_dict = get_train_dict(train_enc_file)
# match_sequences = get_match_sequences_from_counts(count_matrix)


# match_sequences = get_match_sequences_from_counts(count_matrix)
#
# with open(train_enc_file, 'r') as train_handle:
#     train_queries = train_handle.readlines()
#
# print("Loading test into memory...")

# match_synonyms = get_threshold_match_rels(count_matrix, 'avg', len(rel_ids))
# with open(test_enc_file, 'r') as test_handle:
#     test_queries = test_handle.readlines()
# train_rel_connections = compute_first_degree_rel_connections(train_enc_file)
# sim_rel_ents = get_sim_rel_ents(train_rel_connections, .9)
# acc = predict_using_synonyms(test_queries, train_dict, match_synonyms, sim_rel_ents)

# print("Getting train_dict...")
# train_dict = get_train_dict(train_enc_file)
# # print_dict(train_dict)
# # print_dict(match_synonyms)
# print("Computing train entity relation dict...")

# # print_dict(train_rel_connections)
# # print("-----------------------------------------")
# print("Computing ent_rel similarity...")

# # print_dict(sim_rel_ents)
# print("Predicting...")

# print("The accuracy is {}".format(acc))
# # get_set_descripancies(test_queries, train_queries, train_dict, match_synonyms)
# # train_dict_comp_file(train_dict, train_queries)


# print("The entity ids: ")
# print_dict(ent_ids)
# print("The relation ids: ")
# print_dict(rel_ids)
# print("The neighbors dict:")
# print_dict(neighbors_dict)
# print("The hashmap:")
# print_dict(hashmap)
# print_dict(count_matrix)
# print("The length of count_matrix is {0}".format(len(count_matrix)))
# print_dict(match_synonyms)
# print("--------------------------------")
# print_dict(match_sequences)


# print_dict(neighbors_dict)
# print_dict(hashmap)
