from __future__ import absolute_import, division, print_function

import abc
import json
import logging
import os
import tarfile
import six

import requests
import tensorflow as tf

from struc2vec.struc2vec import exec_struc2vec
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


class _ConvELoader(Loader):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        url = 'https://github.com/TimDettmers/ConvE/raw/master'
        super(_ConvELoader, self).__init__(url, [dataset_name + '.tar.gz'])

        # TODO: This is a bad way of "leaking" this information because it may be incomplete when querried.
        self.num_ent = None
        self.num_rel = None

    def train_dataset(self,
                      directory,
                      batch_size,
                      struc2vec_args,
                      include_inv_relations=True,
                      num_parallel_readers=32,
                      num_parallel_batches=32,
                      buffer_size=1024 * 1024, 
                      prefetch_buffer_size=128):
        conve_parser, struc_parser, filenames = self.create_tf_record_files(
            directory, struc2vec_args, buffer_size=buffer_size)

        conve_files = tf.data.Dataset.from_tensor_slices(filenames['train'])
        struc_files = tf.data.Dataset.from_tensor_slices(filenames['struc'])

        def map_fn(sample):
            sample = conve_parser(sample)
            e2_multi1 = tf.to_float(tf.sparse_to_indicator(
                sample['e2_multi1'], self.num_ent))
            return {
                # [None] creates a new dimension at axis 0. So sommething originally (1, 3) will be (1, 1, 3) after [None]
                'e1': sample['e1'][None],
                'e2': sample['e2'][None],
                'rel': sample['rel'][None],
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

        def struc_map_fn(sample):
            sample = struc_parser(sample)
            return {'e1': sample['e1'][None],
                    'e2': sample['e2'][None]}

        conve_data = conve_files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_parallel_readers,
            block_length=batch_size, sloppy=True))\
            .map(map_fn, num_parallel_calls=num_parallel_batches)

        if not include_inv_relations:
            conve_data = conve_data.filter(filter_inv_relations)

        conve_data = conve_data\
            .map(remove_is_inverse)\
            .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))\
            .batch(batch_size)\
            .prefetch(prefetch_buffer_size)

        struc_data = struc_files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_parallel_readers,
            block_length=batch_size, sloppy=True))\
            .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))\
            .apply(tf.contrib.data.map_and_batch(
            map_func=struc_map_fn,
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches))\
            .prefetch(prefetch_buffer_size)

        return conve_data, struc_data

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
            directory, struc2vec_args=None, buffer_size=buffer_size)

        def map_fn(sample):
            sample = parser(sample)
            e2_multi1 = tf.to_float(tf.sparse_to_indicator(sample['e2_multi1'], self.num_ent))
            return {
                'e1': sample['e1'][None],
                'e2': sample['e2'][None],
                'rel': sample['rel'][None],
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

        data = tf.data.TFRecordDataset(filenames[dataset_type])\
            .map(lambda s: map_fn(s))

        if not include_inv_relations:
            data = data.filter(filter_inv_relations)

        return data\
            .map(remove_is_inverse)\
            .batch(batch_size)\
            .prefetch(prefetch_buffer_size)

    # generate json files and ids
    def generate_json_files_and_ids(self, directory, buffer_size=1024*1024):
        json_files = self.load_and_preprocess(directory, buffer_size)
        entity_ids, relation_ids = self._assign_ids(json_files)
        entity_ids['None'] = -1
        relation_ids['None'] = -1
        self.num_ent = len(entity_ids) - 1
        self.num_rel = len(relation_ids) - 1
        return json_files, entity_ids, relation_ids

    # create the structure edgelist
    def create_struc_edgelist(self, directory, full_graph_file, entity_ids):
        # create and write edgelist
        struc_filename = os.path.join(directory, 'edgelist.txt')
        with open(struc_filename, 'a+') as edgelist_file:
            with open(full_graph_file, 'r') as input_file:
                for line in input_file:
                    sample = json.loads(line)
                    # make sure no reverse relationships exist
                    if '_reverse' not in sample['rel']:
                        e2_multi1 = sample['e2_multi1'].strip().split(" ")
                        e1 = entity_ids[sample['e1']]
                        for e2_name in e2_multi1:
                            e2 = entity_ids[e2_name]
                            edgelist_addition = '{0} {1}\n'.format(e1, e2)
                            edgelist_file.write(edgelist_addition)
        return struc_filename

    def run_struc2vec(self, directory, edgelist_filename, struc2vec_args):
        struc2vec_args['input'] = edgelist_filename
        # TODO: Hack to change the working directory because the struc2vec code
        # is pretty bad and requires us to hardcode paths.
        old_cwd = os.getcwd()
        os.chdir(directory)
        exec_struc2vec(struc2vec_args)
        os.chdir(old_cwd)

    def decode_random_walks(self, directory):
        random_walks_path = os.path.join(directory, 'random_walks.txt')
        adj_matrix = load_adjacency_matrix(random_walks_path, self.num_ent)
        adj_matrix = prune_adjacency_matrix(adj_matrix)
        return generate_structure_train_file(adj_matrix,
                                             directory=directory,
                                             output_filename='struc_train.txt')

    def create_tf_record_files(self,
                               directory,
                               struc2vec_args,
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
        if struc2vec_args is not None:
            # Create edgelist
            edgelist_filename = self.create_struc_edgelist(directory, json_files['full'], entity_ids)
            # Generate random walks for structure regularization
            self.run_struc2vec(directory, edgelist_filename, struc2vec_args)
            json_files['struc'] = self.decode_random_walks(directory)
            filetypes = ['train', 'dev', 'test', 'struc']
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
                        if filetype != 'struc':
                            sample = json.loads(line)
                            record = self._encode_sample_as_tf_record(
                                sample, entity_ids, relation_ids)
                        else:
                            e1, e2 = line.strip().split(" ")
                            sample = {'e1': int(e1), 'e2': int(e2)}
                            record = self._encode_struc_sample_as_tf_record(sample)
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

        def struc_tf_record_parser(r):
            features = {
                'e1': tf.FixedLenFeature([], tf.int64),
                'e2': tf.FixedLenFeature([], tf.int64)}
            return tf.parse_single_example(r, features=features)

        return conve_tf_record_parser, struc_tf_record_parser, tf_record_filenames

    def load_and_preprocess(self, directory, buffer_size=1024 * 1024):
        logger.info(
            'Loading and preprocessing the \'%s\' dataset.', self.dataset_name)

        # Download and potentially extract all needed files.
        directory = os.path.join(directory, self.dataset_name)
        self.maybe_extract(directory, buffer_size)

        # One more directory is created due to the archive extraction.
        directory = os.path.join(directory, self.dataset_name)

        # Load and preprocess the data.
        full_graph = {} # Maps from (e1, rel) to set of e2 values.
        graphs = {}     # Maps from filename to dictionaries like labels.
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

    def _write_graph(self, filename, graph, labels=None):
        with open(filename, 'w') as handle:
            for key, value in six.iteritems(graph):
                if labels is None:
                    sample = {
                        'e1': key[0],
                        'e2': 'None',
                        'rel': key[1],
                        'e2_multi1': ' '.join(list(value))}
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

    def _assign_ids(self, json_files):
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

    def _encode_sample_as_tf_record(self, sample, entity_ids, relation_ids):
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

    def _encode_struc_sample_as_tf_record(self, sample):
        e1 = sample['e1']
        e2 = sample['e2']

        def _int64(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=values))

        features = tf.train.Features(feature={
            'e1': _int64([e1]),
            'e2': _int64([e2])})

        return tf.train.Example(features=features)


class NationsLoader(_ConvELoader):
    def __init__(self):
        dataset_name = 'nations'
        super(NationsLoader, self).__init__(dataset_name)


class UMLSLoader(_ConvELoader):
    def __init__(self):
        dataset_name = 'umls'
        super(UMLSLoader, self).__init__(dataset_name)


class KinshipLoader(_ConvELoader):
    def __init__(self):
        dataset_name = 'kinship'
        super(KinshipLoader, self).__init__(dataset_name)


class WN18RRLoader(_ConvELoader):
    def __init__(self):
        dataset_name = 'WN18RR'
        super(WN18RRLoader, self).__init__(dataset_name)


class YAGO310Loader(_ConvELoader):
    def __init__(self):
        dataset_name = 'YAGO3-10'
        super(YAGO310Loader, self).__init__(dataset_name)


class FB15k237Loader(_ConvELoader):
    def __init__(self):
        dataset_name = 'FB15k-237'
        super(FB15k237Loader, self).__init__(dataset_name)
