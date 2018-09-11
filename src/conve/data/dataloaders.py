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

from ..utilities.structure import *

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

        # def struc_map_fn(sample):
        #     sample = struc_parser(sample)
        #     return {'source': sample['source'],
        #             'target': sample['target'],
        #             'weight': sample['weight']}

        def relreg_map_fn(sample):
            sample = relreg_parser(sample)
            return {'seq': sample['seq'],
                    'seq_multi': sample['seq_multi'],
                    'seq_len': sample['seq_len'],
                    'seq_mask': sample['seq_mask']
                    }

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

    # create the structure edgelist
    # @staticmethod
    # def create_struc_edgelist(directory, full_graph_file, entity_ids):
    #     struc_filename = os.path.join(directory, 'edgelist.txt')
    #     with open(struc_filename, 'a+') as edgelist_file:
    #         with open(full_graph_file, 'r') as input_file:
    #             for line in input_file:
    #                 sample = json.loads(line)
    #                 if not sample['rel'].endswith('_reverse'):
    #                     e2_multi1 = sample['e2_multi1'].strip().split(' ')
    #                     e1 = entity_ids[sample['e1']]
    #                     for e2_name in e2_multi1:
    #                         e2 = entity_ids[e2_name]
    #                         edgelist_addition = '{0} {1}\n'.format(e1, e2)
    #                         edgelist_file.write(edgelist_addition)
    #     return struc_filename

    # @staticmethod
    # def run_struc2vec(directory, edgelist_filename, struc2vec_args):
    #     struc2vec_args['input'] = edgelist_filename
    #     # TODO: Hack to change the working directory because the struc2vec code
    #     # is pretty bad and requires us to hardcode paths.
    #     old_cwd = os.getcwd()
    #     os.chdir(directory)
    #     exec_struc2vec(struc2vec_args)
    #     os.chdir(old_cwd)
    #
    # def decode_random_walks(self, directory):
    #     random_walks_path = os.path.join(directory, 'random_walks.txt')
    #     adj_matrix = load_adjacency_matrix(random_walks_path, self.num_rel)
    #     adj_matrix = prune_adjacency_matrix(adj_matrix)
    #     return generate_structure_train_file(adj_matrix,
    #                                          directory=directory,
    #                                          output_filename='struc_train.txt')

    @staticmethod
    def get_mappings(json_file, entity_ids, relation_ids):
        ent_rel_map = {}
        rel_ent_map = {}
        with open(json_file, 'r') as handle:
            for line in handle:
                sample = json.loads(line)
                if '_reverse' not in sample['rel']:
                    e1 = entity_ids[sample['e1'].strip()]
                    rel = relation_ids[sample['rel'].strip()]
                    e2_multi = list(map(lambda e2: entity_ids[e2], sample['e2_multi1'].strip().split(" ")))
                    if e1 not in ent_rel_map.keys():
                        ent_rel_map[e1] = set()
                    if rel not in rel_ent_map.keys():
                        rel_ent_map[rel] = set()

                    ent_rel_map[e1].add(rel)
                    rel_ent_map[rel] = rel_ent_map.get(rel, []) + e2_multi
        return ent_rel_map, rel_ent_map

    @staticmethod
    def generate_relation_train_file_from_mappings(directory, ent_rel_map, rel_ent_map):
        struc_filename = os.path.join(directory, 'edgelist.txt')
        rel_rel_set = set()
        for rel in rel_ent_map.keys():
            entities = rel_ent_map[rel]
            for ent in entities:
                if ent not in ent_rel_map.keys():
                    continue
                target_relations = ent_rel_map[ent]
                for target_relation in target_relations:
                    if target_relation != rel:
                        rel_rel = (rel, target_relation)
                        rel_rel_set.add(rel_rel)

        with open(struc_filename, 'w+') as file:
            for rel_rel in rel_rel_set:
                file.write("{0} {1}\n".format(rel_rel[0], rel_rel[1]))

        return struc_filename

    @staticmethod
    def get_neighbors_and_hashmap(json_file, entity_ids, relation_ids):
        neighbors_dict = {}
        hashmap = {}
        with open(json_file, 'r') as handle:
            for line in handle:
                sample = json.loads(line)
                e1 = entity_ids[sample['e1'].strip()]
                rel = relation_ids[sample['e1'].strip()]
                e2_multi = set(map(lambda e2: entity_ids[e2], sample['e2_multi'].strip().split(" ")))

                if e1 not in neighbors_dict.keys():
                    neighbors_dict[e1] = set()
                neighbors_dict[e1].union(e2_multi)
                # possible that multiple relations from same entity map to same target entity
                for e2 in e2_multi:
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
    @staticmethod
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
    @staticmethod
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
                    similarity = len(e2_in_common) / len(seq2_e2)
                    aggregate_similarity += similarity

                total_subgraphs = len(intersection_e1)
                if total_subgraphs > 0:
                    subgraph_similarity = aggregate_similarity / total_subgraphs
                else:
                    subgraph_similarity = 0

                if subgraph_similarity >= seq_threshold:
                    sequence_similarities[seq1][seq2] = subgraph_similarity

            number_relations_passed += 1.
            percent_done = number_relations_passed / total_relations * 100
            print("Percent done is: {}".format(percent_done))

        return sequence_similarities

    @staticmethod
    def gen_lookup_tables(seq_similarity):
        seqrel_weights = dict()
        for rel, sequences in seq_similarity.items():
            for sequence, weight in sequences.items():
                if sequence not in seqrel_weights.keys():
                    seqrel_weights[sequence] = dict()
                seqrel_weights[sequence][rel] = weight

        seqrelweight_map = dict()
        list_len = len(seq_similarity.keys())
        for sequence in seqrel_weights.keys():
            rels_and_weights = seqrel_weights[sequence]
            # print_dict(rels_and_weights)
            seqrelweight_map[sequence] = [0] * list_len
            for rel, weight in rels_and_weights.items():
                rel_id = rel[0]
                seqrelweight_map[sequence][rel_id] = weight
        return seqrelweight_map

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
        if relreg_args is not None:
            # Create edgelist
            # edgelist_filename = self.create_struc_edgelist(directory, json_files['full'], entity_ids)
            e1_neighbors, e1e2_rel_map = self.get_mappings(json_files['full'], entity_ids, relation_ids)
            seq_e1_sets, seq_e1_pair_sets, _ = self.compute_sequence_eval_sets(e1_neighbors, e1e2_rel_map)
            sequence_similarities = self.get_rel_seq_sims(seq_e1_sets,
                                                          seq_e1_pair_sets,
                                                          relreg_args['seq_threshold'],
                                                          relreg_args['seq_lengths'])

            seq_multi = self.gen_lookup_tables(sequence_similarities)
            seq_multi_path = os.path.join(directory, 'seq_multi.json')
            self._write_graph(seq_multi_path, seq_multi, 'relreg')
            # Generate random walks for structure regularization
            # self.run_struc2vec(directory, edgelist_filename, struc2vec_args)
            json_files['relreg'] = seq_multi_path
            filetypes = ['train', 'dev', 'test', 'relreg']
        else:
            filetypes = ['train', 'dev', 'test']

        tf_record_filenames = {}

        for filetype in filetypes:
            count = 0
            file_index = 0
            filename = os.path.join(
                directory, '{0}-{1}.tfrecords'.format(filetype, file_index))
            tf_record_filenames[filetype] = [filename]
            if not os.path.exists(filename):
                tf_records_writer = tf.python_io.TFRecordWriter(filename)
                with open(json_files[filetype], 'r') as handle:
                    for line in handle:
                        if filetype != 'relreg':
                            sample = json.loads(line)
                            record = self._encode_sample_as_tf_record(
                                sample, entity_ids, relation_ids)
                        else:
                            sample = json.loads(line)
                            record = self._encode_relreg_sample_as_tf_record(sample, relation_ids)
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
                'seq': tf.FixedLenFeature([], tf.int64),
                'seq_multi': tf.FixedLenFeature([], tf.float32),
                'seq_len': tf.FixedLenFeature([], tf.int32),
                'seq_mask': tf.FixedLenFeature([], tf.float32)
            }
            return tf.parse_single_example(r, features=features)

        # def struc_tf_record_parser(r):
        #     features = {
        #         'source': tf.FixedLenFeature([], tf.int64),
        #         'target': tf.FixedLenFeature([], tf.int64),
        #         'weight': tf.FixedLenFeature([], tf.float32)}
        #     return tf.parse_single_example(r, features=features)

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
        self._write_graph(e1rel_to_e2_dev, graphs['valid.txt'], 'full_graph')
        self._write_graph(e1rel_to_e2_test, graphs['test.txt'], 'full_graph')
        self._write_graph(e1rel_to_e2_full, full_graph, 'full_graph')

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
                    def pad_seq(seq):
                        seq_list = list(seq)
                        # TODO: Remove hardcoded 3 from here
                        seq_list += [0] * (3 - len(seq_list))
                        return seq_list

                    seq_len = len(key)
                    seq_mask = np.zeros((3,))
                    seq_mask[seq_len-1] = 1.
                    padded_seq = pad_seq(key)

                    sample = {
                        'seq': ' '.join(padded_seq),
                        'seq_multi': ' '.join(list(value)),
                        'seq_len': seq_len,
                        'seq_mask': seq_mask
                    }
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
        seq = [relation_ids[rel] for rel in sample['seq'].split(' ') if rel != None]
        seq_multi = [weight for weight in sample['seq_multi'].split(' ')]
        seq_len = sample['seq_len']
        seq_mask = sample['seq_mask']

        def _int64(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=values))

        def _float(values):
            return tf.train.Feature(
                    float_list=tf.train.FloatList(value=values))

        features = tf.train.Feature(feature={
            'seq': _int64(seq),
            'seq_multi': _float(seq_multi),
            'seq_len': _int64(seq_len),
            'seq_mask': _float(seq_mask)
        })

        return tf.train.Example(features=features)

    # @staticmethod
    # def _encode_struc_sample_as_tf_record(sample):
    #     source = sample['source']
    #     target = sample['target']
    #     weight = sample['weight']
    #
    #     def _int64(values):
    #         return tf.train.Feature(
    #             int64_list=tf.train.Int64List(value=values))
    #
    #     def _float(values):
    #         return tf.train.Feature(
    #             float_list=tf.train.FloatList(value=values))
    #
    #     features = tf.train.Features(feature={
    #         'source': _int64([source]),
    #         'target': _int64([target]),
    #         'weight': _float([weight])})
    #
    #     return tf.train.Example(features=features)


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
