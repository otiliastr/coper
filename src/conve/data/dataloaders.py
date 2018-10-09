from __future__ import absolute_import, division, print_function

import abc
import json
import logging
import os
import tarfile
import six

import numpy as np
import requests
import tensorflow as tf

from tqdm import tqdm

#from ..utilities.structure import *

__all__ = [
    'Loader', 'NationsLoader', 'UMLSLoader', 'KinshipLoader',
    'WN18RRLoader', 'YAGO310Loader', 'FB15k237Loader']

logger = logging.getLogger(__name__)


class Loader(six.with_metaclass(abc.ABCMeta, object)):
    def __init__(self, url, filenames):
        self.url = url
        self.filenames = filenames

    @abc.abstractmethod
    def load_and_preprocess(self, directory, buffer_size=1024 * 1024):
        pass

    def maybe_extract(self, directory, buffer_size=1024 * 1024):
        """Downloads the required dataset files, if needed, and extracts all
        '.tar.gz' files that may have been downloaded, if needed.

        Arguments:
            directory (str): Directory in which to download the files.
            buffer_size (int, optional): Buffer size to use while downloading.
        """
        self.maybe_download(directory, buffer_size)
        for filename in self.filenames:
            if filename.endswith('.tar.gz'):
                path = os.path.join(directory, filename)
                extracted_path = path[:-7]
                if not os.path.exists(extracted_path):
                    logger.info('Extracting file: %s', path)
                    with tarfile.open(path, 'r:*') as handle:
                        handle.extractall(path=extracted_path)

    def maybe_download(self, directory, buffer_size=1024 * 1024):
        """Downloads the required dataset files, if needed.

        Arguments:
            directory (str): Directory in which to download the files.
            buffer_size (int, optional): Buffer size to use while downloading.
        """
        os.makedirs(directory, exist_ok=True)
        for filename in self.filenames:
            path = os.path.join(directory, filename)
            url = self.url + '/' + filename
            if not os.path.exists(path):
                logger.info('Downloading \'%s\' to \'%s\'.', url, directory)
                response = requests.get(url, stream=True)
                with open(os.path.join(directory, filename), 'wb') as handle:
                    for data in tqdm(response.iter_content(buffer_size)):
                        handle.write(data)


class _DataLoader(Loader):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        url = 'https://github.com/TimDettmers/ConvE/raw/master'
        super(_DataLoader, self).__init__(url, [dataset_name + '.tar.gz'])

        # TODO: This is a bad way of "leaking" this information because it may be incomplete when querried.
        self.num_ent = None
        self.num_rel = None

    def train_dataset(self,
                      directory,
                      batch_size,
                      relreg_args,
                      include_inv_relations=True,
                      num_parallel_readers=32,
                      num_parallel_batches=32,
                      buffer_size=1024 * 1024,
                      prefetch_buffer_size=128):
        conve_parser, relreg_parser, filenames = self.create_tf_record_files(
            directory, relreg_args, buffer_size=buffer_size)

        conve_files = tf.data.Dataset.from_tensor_slices(filenames['train'])
        relreg_files = tf.data.Dataset.from_tensor_slices(filenames['relreg'])

        def map_fn(sample):
            sample = conve_parser(sample)
            e2_multi1 = tf.to_float(tf.sparse_to_indicator(
                sample['e2_multi1'], self.num_ent))
            return {
                'e1': sample['e1'],
                'e2': sample['e2'],
                'rel': sample['rel'],
                'e2_multi1': e2_multi1,
                'is_inverse': tf.cast(sample['is_inverse'], tf.bool)}

        def filter_inv_relations(sample):
            return tf.logical_not(sample['is_inverse'])

        def remove_is_inverse(sample):
            return {
                'e1': sample['e1'],
                'e2': sample['e2'],
                'rel': sample['rel'],
                'e2_multi1': sample['e2_multi1']}

        def relreg_map_fn(sample):
            sample = relreg_parser(sample)
            seq_0 = sample['seq'][0]
            seq_1 = sample['seq'][1]
            seq_2 = sample['seq'][2]
            #seq = tf.to_int64(sample['seq'])
            #seq_mask_0 = tf.to_int64([sample['seq_mask'][0]])
            #seq_mask_1 = tf.to_int64([sample['seq_mask'][1]])
            #seq_mask_2 = tf.to_int64([sample['seq_mask'][2]])
            seq_mask = tf.to_int64(sample['seq_mask'])
            #print("Seq mask is {}".format(seq_mask))
            return {'s_ent': sample['s_ent'],
                    'rel': sample['rel'],
                    'seq_0': seq_0,
                    'seq_1': seq_1,
                    'seq_2': seq_2,
                    #'seq': seq, # sample['seq'],
                    'sim': sample['sim'],
                    #'agg_sim':sample['agg_sim'],
                    #'seq_mask_0': seq_mask_0,
                    #'seq_mask_1': seq_mask_1,
                    #'seq_mask_2': seq_mask_2,
                    'seq_mask': seq_mask}


        conve_data = conve_files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_parallel_readers,
            block_length=batch_size, sloppy=True)) \
            .map(map_fn, num_parallel_calls=num_parallel_batches)

        if not include_inv_relations:
            conve_data = conve_data.filter(filter_inv_relations)

        conve_data = conve_data \
            .map(remove_is_inverse) \
            .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000)) \
            .batch(batch_size) \
            .prefetch(prefetch_buffer_size)

        relreg_data = relreg_files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_parallel_readers,
            block_length=batch_size, sloppy=True)) \
            .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000)) \
            .apply(tf.contrib.data.map_and_batch(
            map_func=relreg_map_fn,
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches)) \
            .prefetch(prefetch_buffer_size)

        return conve_data, relreg_data

    def dev_dataset(self,
                    directory,
                    batch_size,
                    include_inv_relations=True,
                    buffer_size=1024 * 1024,
                    prefetch_buffer_size=128):
        return self._eval_dataset(
            directory, 'dev', batch_size, include_inv_relations,
            buffer_size, prefetch_buffer_size)

    def test_dataset(self,
                     directory,
                     batch_size,
                     include_inv_relations=True,
                     buffer_size=1024 * 1024,
                     prefetch_buffer_size=128):
        return self._eval_dataset(
            directory, 'test', batch_size, include_inv_relations,
            buffer_size, prefetch_buffer_size)

    def _eval_dataset(self,
                      directory,
                      dataset_type,
                      batch_size,
                      include_inv_relations=True,
                      buffer_size=1024 * 1024,
                      prefetch_buffer_size=128):
        parser, _, filenames = self.create_tf_record_files(
            directory, relreg_args=None, buffer_size=buffer_size)

        def map_fn(sample):
            sample = parser(sample)
            e2_multi1 = tf.to_float(tf.sparse_to_indicator(sample['e2_multi1'], self.num_ent))
            return {
                'e1': sample['e1'],
                'e2': sample['e2'],
                'rel': sample['rel'],
                'e2_multi1': e2_multi1,
                'is_inverse': tf.cast(sample['is_inverse'], tf.bool)}

        def filter_inv_relations(sample):
            return tf.logical_not(sample['is_inverse'])

        def remove_is_inverse(sample):
            return {
                'e1': sample['e1'],
                'e2': sample['e2'],
                'rel': sample['rel'],
                'e2_multi1': sample['e2_multi1']}

        data = tf.data.TFRecordDataset(filenames[dataset_type]) \
            .map(lambda s: map_fn(s))

        if not include_inv_relations:
            data = data.filter(filter_inv_relations)

        return data \
            .map(remove_is_inverse) \
            .batch(batch_size) \
            .prefetch(prefetch_buffer_size)

    # generate json files and ids
    def generate_json_files_and_ids(self, directory, buffer_size=1024 * 1024):
        json_files = self.load_and_preprocess(directory, buffer_size)
        entity_ids, relation_ids = self._assign_ids(json_files)
        entity_ids['None'] = -1
        relation_ids['None'] = -1
        self.num_ent = len(entity_ids) - 1
        self.num_rel = len(relation_ids) - 1
        return json_files, entity_ids, relation_ids


    @staticmethod
    def get_neighbors_and_hashmap(json_file, entity_ids, relation_ids):
        neighbors_dict = {}
        hashmap = {}
        def print_dict(d):
            sorted_d = sorted(d.keys())
            for key in sorted_d:
                values = d[key]
                print("{}: {}".format(key, values))
        # print_dict(entity_ids)
        with open(json_file, 'r') as handle:
            for line in handle:
                sample = json.loads(line)
                e1 = entity_ids[sample['e1'].strip()]
                rel = relation_ids[sample['rel'].strip()]
                e2_multi = set(map(lambda e2: entity_ids[e2], sample['e2_multi1'].strip().split(" ")))
                if '_reverse' not in sample['rel'].strip():
                    if e1 not in neighbors_dict.keys():
                        neighbors_dict[e1] = set()
                    neighbors_dict[e1] = neighbors_dict[e1].union(e2_multi)
                    # possible that multiple relations from same entity map to same target entity
                    for e2 in e2_multi:
                        if (e1, e2) not in hashmap.keys():
                            hashmap[(e1, e2)] = set()
                        hashmap[(e1, e2)].add(rel)
        return neighbors_dict, hashmap

    
    def get_full_graph(self, json_file, entity_ids, relation_ids):
        e2_multi_dict = {}
        with open(json_file, 'r') as handle:
            for line in handle:
                sample = json.loads(line)
                e1 = entity_ids[sample['e1'].strip()]
                rel = relation_ids[sample['rel'].strip()]
                e2_multi = set(map(lambda e2: entity_ids[e2], sample['e2_multi1'].strip().split(" ")))
                if '_reverse' not in sample['rel'].strip():
                    e2_multi_dict[(e1, rel)] = set(e2_multi)
        return e2_multi_dict

    @staticmethod
    def print_dict(d):
        sorted_d = sorted(d.keys())
        for key in sorted_d:
            values = d[key]
            print("{}: {}".format(key, values))

    # Using neighbors and relations, compute all possible sequences of length up to 3 from each e1. Store this information
    # into three dictionaries. The first comprises of all e1 that each sequences originates from. The second is all
    # e2 that each (e1, sequence) pair leads to. The third is all sequences tied to each entity:
    #   1. {sequence_1: set(e1_1, e1_2, ...), ...}
    #   2. {(e1, sequence_1): set(e2_1, e2_2, ...), ...}
    #   3. {e1: set(sequence_1, sequence_2, ...), ...}
    @staticmethod
    def compute_sequence_eval_sets(neighbors_dict, hashmap):
        # print(neighbors_dict)
        # print(hashmap)
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
    @staticmethod
    def get_rel_seq_sims(e1_seq_set, e1_seq_e2_set, seq_threshold, seq_lengths):
        print("Getting Relation Sequence Similarities....")
        number_relations_passed = 0
        sequence_similarities = {}
        rel_agg_similarities = {}
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
            if seq1 not in rel_agg_similarities:
                rel_agg_similarities[seq1] = 0.0

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
                    similarity = len(e2_in_common) / len(seq2_e2)
                    aggregate_similarity += similarity

                total_subgraphs = len(intersection_e1)
                if total_subgraphs > 0:
                    subgraph_similarity = aggregate_similarity / total_subgraphs
                else:
                    subgraph_similarity = 0

                if subgraph_similarity > 0.0:
                    #if subgraph_similarity >= seq_thresholdi
                    sequence_similarities[seq1][seq2] = (subgraph_similarity)
                    #rel_agg_similarities[seq1] += subgraph_similarity

            number_relations_passed += 1.
            percent_done = number_relations_passed / total_relations * 100
            print("Percent done is: {}".format(percent_done))

        return sequence_similarities#, rel_agg_similarities

    @staticmethod
    def get_indv_rel_seq_sims(e1_seq_set, e1_seq_e2_set, seq_threshold, seq_lengths):
        number_relations_passed = 0
        rel_seq_sim_and_s_ents = {}
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

            gen = ((seq2, seq2_e1) for seq2, seq2_e1 in valid_sequences.items() if (seq1 != seq2))
            for seq2, seq2_e1 in gen:
                intersection_e1 = seq1_e1.intersection(seq2_e1)
                if len(intersection_e1) > 0:
                    sampled_e1 = np.random.choice(list(intersection_e1), 5, replace = True)
                else:
                    sampled_e1 = []

                for e1 in sampled_e1:
                    seq1_pair = (e1, seq1)
                    seq2_pair = (e1, seq2)

                    seq1_e2 = e1_seq_e2_set[seq1_pair]
                    seq2_e2 = e1_seq_e2_set[seq2_pair]

                    e2_in_common = seq1_e2.intersection(seq2_e2)
                    similarity = len(e2_in_common) / len(seq2_e2)
                    
                    if similarity > 0.0:
                        #print("Similarity larger than .9: {}".format((e1, seq1[0], seq2, similarity)))
                        double = (e1, seq1)
                        if double not in rel_seq_sim_and_s_ents:
                            rel_seq_sim_and_s_ents[double] = {}
                        rel_seq_sim_and_s_ents[double][seq2] = similarity

            number_relations_passed += 1.
            percent_done = number_relations_passed / total_relations * 100
            print("Percent done is: {}".format(percent_done))

        return rel_seq_sim_and_s_ents


    def parse_rel_seq_sim(self, sequence_similarities):#, rel_agg_similarities):
        rel_siq_sim = {}
        # TODO: Remove this hardcoded max path length
        seq_max_len = 3
        for rel, sequences in sequence_similarities.items():
            for sequence, similarity in sequences.items():
                padded_seq = tuple(list(sequence) + [0] * (seq_max_len-len(sequence)))
                seq_idx = len(sequence)-1
                seq_mask = [0] * seq_max_len
                seq_mask[seq_idx] = 1
                #agg_sim = rel_agg_similarities[rel]
                triple = (rel, padded_seq, tuple(seq_mask))

                rel_siq_sim[triple] = (similarity)#, agg_sim)
        return rel_siq_sim

    def extract_rel_seq_info(self, rel_seq_info):
        rel_seq_data = {}
        seq_max_len = 3
        for (s_ent, rel), sequences in rel_seq_info.items():
            for sequence, similarity in sequences.items():
                padded_seq = tuple(list(sequence) + [0] * (seq_max_len-len(sequence)))
                seq_idx = len(sequence)-1
                seq_mask = [0] * seq_max_len
                seq_mask[seq_idx] = 1
                #quadruple = (s_ent, rel, padded_seq, tuple(seq_mask))
                double = (s_ent, rel)
                quadruple = (s_ent, rel, padded_seq, tuple(seq_mask))
                #rel_seq_data[quadruple] = (sim)
                rel_seq_data[quadruple] = similarity
        return rel_seq_data

    def extend_train_data(self, train_file, rel_seq_sim, write_fp, entity_ids, relation_ids):
        print("Extending training data... ")
        amount_done = 0
        with open(write_fp, 'w') as write_handle:
            with open(train_file, 'r') as handle:
                #print("The size of training file is {}".format(len(handle.readlines())))
                for line in handle:
                    sample = json.loads(line)
                    e1 = entity_ids[sample['e1'].strip()]
                    rel = relation_ids[sample['rel'].strip()]
                    e2_multi = list(map(lambda e2: entity_ids[e2], sample['e2_multi1'].strip().split(" ")))
                    if '_reverse' not in sample['rel'].strip():
                        lookup_pair = (e1, rel)
                        # if no sequence exists, place dummy sequence
                        sequences = rel_seq_sim.get(lookup_pair, [((0, 0, 0), (1, 0, 0), 0.0)])
                        sequence_idx = np.random.choice(len(sequences), 1)[0]
                        sequence = sequences[sequence_idx]
                    else:
                        sequence = ((0, 0, 0), (1, 0, 0), 0.0)
                
                    write_sample = {
                        'e1': e1,
                        #'e2': 'None', 
                        'rel': rel,
                        'seq': ' '.join(list(map(lambda elem: str(elem), sequence[0]))),
                        'seq_mask': ' '.join(list(map(lambda elem: str(elem), sequence[1]))),
                        'sim': sequence[2],
                        'e2_multi': ' '.join(list(map(lambda elem: str(elem), e2_multi)))
                    }
                    amount_done = 1
                    #print("Done with {} samples".format(amount_done))
                    write_handle.write(json.dumps(write_sample) + '\n')
        return write_fp

    def create_tf_record_files(self,
                               directory,
                               relreg_args,
                               max_records_per_file=10000,
                               buffer_size=1024 * 1024):
        logger.info(
            'Creating TF record files for the \'%s\' dataset.',
            self.dataset_name)

        # We first load the entity and relation ID maps and handle missing
        # entries using -1 as their index.
        json_files, entity_ids, relation_ids = self.generate_json_files_and_ids(directory, buffer_size)
        directory = os.path.dirname(json_files['full'])


        # Check whether or not to include structure.
        save_directory = directory
        save_path = os.path.join(save_directory, 'rel_seq_path.json')
        print("The path is: {}".format(save_path))
        print("The directory is {}".format(directory))
        if relreg_args is not None and not os.path.exists(save_path):
            # Create edgelist
            # edgelist_filename = self.create_struc_edgelist(directory, json_files['full'], entity_ids)
            e1_neighbors, e1e2_rel_map = self.get_neighbors_and_hashmap(json_files['train'], entity_ids, relation_ids)
            #e2_multi_dict = self.get_full_graph(json_files['train'], entity_ids, relation_ids)
            seq_e1_sets, seq_e1_pair_sets, _ = self.compute_sequence_eval_sets(e1_neighbors, e1e2_rel_map)
            sequence_similarities = self.get_indv_rel_seq_sims(seq_e1_sets,
                                                          seq_e1_pair_sets,
                                                          relreg_args['seq_threshold'],
                                                          relreg_args['seq_lengths'])
            rel_seq_sim = self.extract_rel_seq_info(sequence_similarities)
            #rel = self.extend_train_data(json_files['train'], rel_seq_sim)
            rel_seq_path = os.path.join(directory, 'rel_seq_path.json')
            self._write_graph(rel_seq_path, rel_seq_sim, 'relreg')
            # Generate random walks for structure regularization
            # self.run_struc2vec(directory, edgelist_filename, struc2vec_args)
            json_files['relreg'] = rel_seq_path
         
        if relreg_args is not None:
            filetypes = ['relreg', 'train', 'dev', 'test']
            rel_seq_path = os.path.join(directory, 'rel_seq_path.json')
            json_files['relreg'] = rel_seq_path
        else:
            filetypes = ['train', 'dev', 'test']

        tf_record_filenames = {}
        print("The filetypes are: {}".format(filetypes))
        for filetype in filetypes:
            print("Filetype is: {}".format(filetype))
            count = 0
            file_index = 0
            filename = os.path.join(
                directory, '{0}-{1}.tfrecords'.format(filetype, file_index))
            tf_record_filenames[filetype] = [filename]
            print("filename: {}".format(filename))
            if not os.path.exists(filename):
                tf_records_writer = tf.python_io.TFRecordWriter(filename)
                with open(json_files[filetype], 'r') as handle:
                    for line in handle:
                        #print("the current filetype is: {}".format(filetype))
                        if filetype != 'relreg':
                            sample = json.loads(line)
                            record = self._encode_sample_as_tf_record(
                                sample, entity_ids, relation_ids)
                        else:
                            #print("Here")
                            sample = json.loads(line)
                            record = self._encode_relreg_sample_as_tf_record(sample, relation_ids)
                            #print("The record is {}".format(record))
                            #BUG
                            # source, target, weight = line.strip().split(' ')
                            # sample = {'source': int(source), 'target': int(target), 'weight': float(weight)}
                            # record = self._encode_struc_sample_as_tf_record(sample)
                        tf_records_writer.write(record.SerializeToString())
                        count += 1
                        if count >= max_records_per_file:
                            tf_records_writer.close()
                            count = 0
                            file_index += 1
                            filename = os.path.join(
                                directory,
                                '{0}-{1}.tfrecords'.format(filetype, file_index))
                            tf_record_filenames[filetype].append(filename)
                            tf_records_writer = tf.python_io.TFRecordWriter(filename)
                tf_records_writer.close()
        

        def conve_tf_record_parser(r):
            features = {
                'e1': tf.FixedLenFeature([], tf.int64),
                'e2': tf.FixedLenFeature([], tf.int64),
                'rel': tf.FixedLenFeature([], tf.int64),
                'e2_multi1': tf.VarLenFeature(tf.int64),
                'is_inverse': tf.FixedLenFeature([], tf.int64)}
            return tf.parse_single_example(r, features=features)

        def relreg_tf_record_parser(r):
            features = {
                's_ent': tf.FixedLenFeature([], tf.int64),
                'rel': tf.FixedLenFeature([], tf.int64),
                'seq': tf.FixedLenFeature([3], tf.int64),
                'sim': tf.FixedLenFeature([], tf.float32),
                #'agg_sim': tf.FixedLenFeature([], tf.float32),
                'seq_mask':tf.FixedLenFeature([3], tf.int64),
                #'e2_multi': tf.VarLenFeature(tf.int64)
            }
            return tf.parse_single_example(r, features=features)

        return conve_tf_record_parser, relreg_tf_record_parser, tf_record_filenames

    def load_and_preprocess(self, directory, buffer_size=1024 * 1024):
        logger.info(
            'Loading and preprocessing the \'%s\' dataset.', self.dataset_name)

        # Download and potentially extract all needed files.
        directory = os.path.join(directory, self.dataset_name)
        self.maybe_extract(directory, buffer_size)

        # One more directory is created due to the archive extraction.
        directory = os.path.join(directory, self.dataset_name)
        # Load and preprocess the data.
        full_graph = {}  # Maps from (e1, rel) to set of e2 values.
        graphs = {}  # Maps from filename to dictionaries like labels.
        files = ['train.txt', 'valid.txt', 'test.txt']
        for f in files:
            graphs[f] = {}
            with open(os.path.join(directory, f), 'r') as handle:
                for line in handle:
                    e1, rel, e2 = line.split('\t')
                    e1 = e1.strip()
                    e2 = e2.strip()
                    rel = rel.strip()
                    rel_reverse = rel + '_reverse'

                    # Add potentially missing keys to dictionaries.
                    if (e1, rel) not in full_graph:
                        full_graph[(e1, rel)] = set()
                    if (e2, rel_reverse) not in full_graph:
                        full_graph[(e2, rel_reverse)] = set()
                    if (e1, rel) not in graphs[f]:
                        graphs[f][(e1, rel)] = set()
                    if (e2, rel_reverse) not in graphs[f]:
                        graphs[f][(e2, rel_reverse)] = set()

                    # Add observations.
                    full_graph[(e1, rel)].add(e2)
                    full_graph[(e2, rel_reverse)].add(e1)
                    graphs[f][(e1, rel)].add(e2)
                    graphs[f][(e2, rel_reverse)].add(e1)

        # Write preprocessed files in a standardized JSON format.
        e1rel_to_e2_train = os.path.join(directory, 'e1rel_to_e2_train.json')
        e1rel_to_e2_dev = os.path.join(directory, 'e1rel_to_e2_dev.json')
        e1rel_to_e2_test = os.path.join(directory, 'e1rel_to_e2_test.json')
        e1rel_to_e2_full = os.path.join(directory, 'e1rel_to_e2_full.json')
        self._write_graph(e1rel_to_e2_train, graphs['train.txt'])
        self._write_graph(e1rel_to_e2_dev, graphs['valid.txt'], full_graph)
        self._write_graph(e1rel_to_e2_test, graphs['test.txt'], full_graph)
        self._write_graph(e1rel_to_e2_full, full_graph, full_graph)

        return {
            'train': e1rel_to_e2_train,
            'dev': e1rel_to_e2_dev,
            'test': e1rel_to_e2_test,
            'full': e1rel_to_e2_full}


    @staticmethod
    def _write_graph(filename, graph, labels=None):
        with open(filename, 'w') as handle:
            for key, value in six.iteritems(graph):
                if labels is None:
                    sample = {
                        'e1': key[0],
                        'e2': 'None',
                        'rel': key[1],
                        'e2_multi1': ' '.join(list(value))}
                    handle.write(json.dumps(sample) + '\n')
                elif labels == 'relreg':
                    #s_ent = int(key[0])
                    s_ent = key[0]
                    rel = key[1]
                    seq = key[2]
                    seq_mask = key[3]
                    sim = value #[0]
                    #e2_multi = value[1]
                    #agg_sim = value[1]

                    sample = {
                        's_ent': str(s_ent),
                        'rel': rel,
                        'seq': ' '.join(list(map(lambda elem: str(elem), seq))),
                        'sim': sim,
                        #'agg_sim': agg_sim,
                        'seq_mask': ' '.join(list(map(lambda elem: str(elem), seq_mask))),
                        #'e2_multi': ' '.join(list(map(lambda elem: str(elem), e2_multi)))
                    }
                    # print("Sample is {}".format(sample))
                    # print("The type of s_ent is {}".format(type(s_ent)))
                    # print('The type of rel is {}'.format(type(rel)))
                    # print("The type of seq is {}".format(type(seq)))
                    # print("The type of seq_mask is {}".format(type(seq_mask)))
                    # print("The type of sim is {}".format(sim))
                    handle.write(json.dumps(sample) + '\n')
                else:
                    e1, rel = key
                    e2_multi1 = ' '.join(list(labels[key]))
                    for e2 in value:
                        sample = {
                            'e1': e1,
                            'e2': e2,
                            'rel': rel,
                            'e2_multi1': e2_multi1}
                        handle.write(json.dumps(sample) + '\n')

    @staticmethod
    def _assign_ids(json_files):
        directory = os.path.dirname(json_files['full'])
        entities_file = os.path.join(directory, 'entities.txt')
        relations_file = os.path.join(directory, 'relations.txt')

        entities_exist = os.path.exists(entities_file)
        relations_exist = os.path.exists(relations_file)

        entity_names = {}
        entity_ids = {}
        relation_names = {}
        relation_ids = {}

        # Check if any of the index files already exist.
        if entities_exist:
            with open(entities_file, 'r') as handle:
                for i, line in enumerate(handle):
                    line = line.strip()
                    entity_names[i] = line
                    entity_ids[line] = i
        if relations_exist:
            with open(relations_file, 'r') as handle:
                for i, line in enumerate(handle):
                    line = line.strip()
                    relation_names[i] = line
                    relation_ids[line] = i

        # Create any of the entity and relation index maps, if needed.
        if not entities_exist or not relations_exist:
            with open(json_files['full'], 'r') as handle:
                full_data = []
                for line in handle:
                    full_data.append(json.loads(line))
            num_ent = 0
            num_rel = 0
            for sample in full_data:
                if not entities_exist:
                    entities = set()
                    entities.add(sample['e1'])
                    entities.add(sample['e2'])
                    entities.update(sample['e2_multi1'].split(' '))
                    for entity in entities:
                        if entity != 'None' and entity not in entity_ids:
                            entity_names[num_ent] = entity
                            entity_ids[entity] = num_ent
                            num_ent += 1
                if not relations_exist:
                    relation = sample['rel']
                    if relation != 'None' and \
                            relation not in relation_ids:
                        relation_names[num_rel] = relation
                        relation_ids[relation] = num_rel
                        num_rel += 1

        # Store the index maps in text files, if needed.
        # TODO: Can be done more efficiently using ordered dictionaries.
        if not entities_exist:
            with open(entities_file, 'w') as handle:
                for entity_id in range(num_ent):
                    handle.write(entity_names[entity_id] + '\n')
        if not relations_exist:
            with open(relations_file, 'w') as handle:
                for relation_id in range(num_rel):
                    handle.write(relation_names[relation_id] + '\n')

        return entity_ids, relation_ids

    @staticmethod
    def _encode_sample_as_tf_record(sample, entity_ids, relation_ids):
        e1 = entity_ids[sample['e1']]
        e2 = entity_ids[sample['e2']]
        rel = relation_ids[sample['rel']]
        e2_multi1 = [entity_ids[e]
                     for e in sample['e2_multi1'].split(' ')
                     if e != 'None']

        def _int64(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=values))

        features = tf.train.Features(feature={
            'e1': _int64([e1]),
            'e2': _int64([e2]),
            'rel': _int64([rel]),
            'e2_multi1': _int64(e2_multi1),
            'is_inverse': _int64([sample['rel'].endswith('_reverse')])})

        return tf.train.Example(features=features)

    @staticmethod
    def _encode_relreg_sample_as_tf_record(sample, relation_ids):
        #print('seq is {}'.format(sample['seq']))
        #print("sim is {}".format(sample['sim']))
        #print('rel is {}'.format(sample['rel']))
        #print('seq_mask is {}'.format(sample['seq_mask']))
        #print("agg_sim is {}".format(sample['agg_sim']))
        # dtypes:
        # rel: [rel_id], seq: 'rel_1_id, rel_2_id, rel_3_id', sim: .7777, seq_mask: 'bool, bool, bool'
        s_ent = int(sample['s_ent'])
        rel = sample['rel'][0] #list(map(lambda elem: int(elem), sample['rel']))
        seq = [int(rel) for rel in sample['seq'].split(' ') if rel != None]
        sim = sample['sim']
        #agg_sim = sample['agg_sim']
        seq_mask = [int(bool_val) for bool_val in sample['seq_mask'].split(' ')]
        #e2_multi = [int(e2) for e2 in sample['e2_multi'].split(' ')]

        def _int64(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=values))

        def _float(values):
            return tf.train.Feature(
                    float_list=tf.train.FloatList(value=values))

        features = tf.train.Features(feature={
            's_ent': _int64([s_ent]),
            'rel': _int64([rel]),
            'seq': _int64(seq),
            'sim': _float([sim]),
            #'agg_sim': _float([agg_sim]),
            'seq_mask': _int64(seq_mask),
            #'e2_multi': _int64(e2_multi)
        })

        return tf.train.Example(features=features)

class NationsLoader(_DataLoader):
    def __init__(self):
        dataset_name = 'nations'
        super(NationsLoader, self).__init__(dataset_name)


class UMLSLoader(_DataLoader):
    def __init__(self):
        dataset_name = 'umls'
        super(UMLSLoader, self).__init__(dataset_name)


class KinshipLoader(_DataLoader):
    def __init__(self):
        dataset_name = 'kinship'
        super(KinshipLoader, self).__init__(dataset_name)


class WN18RRLoader(_DataLoader):
    def __init__(self):
        dataset_name = 'WN18RR'
        super(WN18RRLoader, self).__init__(dataset_name)


class YAGO310Loader(_DataLoader):
    def __init__(self):
        dataset_name = 'YAGO3-10'
        super(YAGO310Loader, self).__init__(dataset_name)


class FB15k237Loader(_DataLoader):
    def __init__(self):
        dataset_name = 'FB15k-237'
        super(FB15k237Loader, self).__init__(dataset_name)

