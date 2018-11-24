from __future__ import absolute_import, division, print_function

import abc
import glob
import json
import logging
import os
import tarfile
import six

import requests
import tensorflow as tf

from tqdm import tqdm

__all__ = [
    'Loader', 'NationsLoader', 'UMLSLoader', 'KinshipLoader', 'WN18RRLoader', 'YAGO310Loader', 'FB15k237Loader',
    'CountriesS1Loader', 'CountriesS2Loader', 'CountriesS3Loader', 'NELL995Loader']

logger = logging.getLogger(__name__)


class Loader(six.with_metaclass(abc.ABCMeta, object)):
    def __init__(self, url, filenames, dataset_name):
        self.url = url
        self.filenames = filenames
        self.dataset_name = dataset_name

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
        extracted = False
        for filename in self.filenames:
            if filename.endswith('.tar.gz'):
                path = os.path.join(directory, filename)
                extracted_path = path[:-7]
                if not os.path.exists(extracted_path):
                    logger.info('Extracting file: %s', path)
                    with tarfile.open(path, 'r:*') as handle:
                        handle.extractall(path=extracted_path)
                extracted = True
        return extracted

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
    def __init__(self, url, filenames, dataset_name, filetypes=['train', 'dev', 'test'],  needs_test_set_cleaning=False,
                 add_reverse_per_filetype=None):
        self.filetypes = filetypes
        if add_reverse_per_filetype is None:
            add_reverse_per_filetype = [True for _ in range(len(filetypes))]
        self.add_reverse_per_filetype = add_reverse_per_filetype
        self.needs_test_set_cleaning = needs_test_set_cleaning
        super(_DataLoader, self).__init__(url, filenames, dataset_name)

        # TODO: This is a bad way of "leaking" this information because it may be incomplete when querried.
        self.num_ent = None
        self.num_rel = None

    def train_dataset(self,
                      directory,
                      batch_size,
                      include_inv_relations=True,
                      num_parallel_readers=32,
                      num_parallel_batches=32,
                      buffer_size=1024 * 1024,
                      prefetch_buffer_size=128,
                      prop_negatives=10.0,
                      num_labels=100):
        conve_parser, filenames = self.create_tf_record_files(
            directory, buffer_size=buffer_size)

        conve_files = filenames['train']

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

        conve_data = tf.data.Dataset.from_tensor_slices(conve_files) \
            .interleave(tf.data.TFRecordDataset,
                        cycle_length=num_parallel_readers,
                        block_length=batch_size) \
            .map(lambda s: map_fn(s), num_parallel_calls=num_parallel_batches)

        if not include_inv_relations:
            conve_data = conve_data.filter(filter_inv_relations)
   
        conve_data = conve_data.map(remove_is_inverse)
             
        do_negative_sample = True
        if do_negative_sample:
            assert num_labels > prop_negatives, 'Parameter `num_labels` needs to be larger than `prop_negatives`.'
            conve_data = conve_data.map(
                lambda sample: self._sample_negatives(
                    sample=sample,
                    prop_negatives=prop_negatives,
                    num_labels=num_labels))

        conve_data = conve_data \
            .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000)) \
            .batch(batch_size) \
            .prefetch(prefetch_buffer_size)

        return conve_data

    def eval_dataset(self,
                     directory,
                     dataset_type,
                     batch_size,
                     include_inv_relations=True,
                     buffer_size=1024 * 1024,
                     prefetch_buffer_size=128):
        parser, filenames = self.create_tf_record_files(
            directory, buffer_size=buffer_size)
        filenames = filenames[dataset_type]

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

        data = tf.data.Dataset.from_tensor_slices(filenames)\
            .flat_map(tf.data.TFRecordDataset)\
            .map(lambda s: map_fn(s))

        if not include_inv_relations:
            data = data.filter(filter_inv_relations)

        return data \
            .map(remove_is_inverse) \
            .batch(batch_size) \
            .prefetch(prefetch_buffer_size)

    @staticmethod
    def _sample_negatives(sample, prop_negatives, num_labels):
        e1 = sample['e1']
        e2 = sample['e2']
        rel = sample['rel']
        e2_multi= sample['e2_multi1']
        
        zero = tf.constant(0, dtype=tf.float32)
        one = tf.constant(1, dtype=tf.float32)
        correct_e2s = tf.where(tf.equal(e2_multi, one))[:, 0]
        wrong_e2s = tf.where(tf.equal(e2_multi, zero))[:, 0]
        correct_e2s = tf.random_shuffle(correct_e2s)
        wrong_e2s = tf.random_shuffle(wrong_e2s)

        num_positives = tf.size(correct_e2s)
        num_negatives = tf.size(wrong_e2s)

        num_positives_needed = int(1.0 / (1.0 + prop_negatives) * num_labels)
        print('num_positives_needed: ', num_positives_needed)

        def _less_positives():
            num_neg = num_labels - num_positives
            return tf.concat([
                correct_e2s[:num_positives],
                wrong_e2s[:num_neg]], axis=0)

        def _more_positives():
            num_negatives_needed = num_labels - num_positives_needed
            num_neg = tf.minimum(num_negatives, num_negatives_needed)
            num_positives = num_labels - num_neg
            return tf.concat([
                correct_e2s[:num_positives],
                wrong_e2s[:num_neg]], axis=0)

        indexes = tf.cond(
            tf.less_equal(num_positives, num_positives_needed),
            _less_positives,
            _more_positives)
        lookup_values = tf.cast(indexes, tf.int32)

        return {
                'e1': e1,
                'e2': e2,
                'rel': rel,
                'e2_multi': e2_multi,
                'lookup_values': lookup_values}

    def generate_json_files_and_ids(self, directory, buffer_size=1024 * 1024):
        json_files = self.load_and_preprocess(directory, buffer_size)
        entity_ids, relation_ids = self._assign_ids(json_files)
        entity_ids['None'] = -1
        relation_ids['None'] = -1
        self.num_ent = len(entity_ids) - 1
        self.num_rel = len(relation_ids) - 1
        return json_files, entity_ids, relation_ids

    def create_tf_record_files(self,
                               directory,
                               max_records_per_file=10000,
                               buffer_size=1024 * 1024):
        logger.info(
            'Creating TF record files for the \'%s\' dataset.',
            self.dataset_name)

        # We first load the entity and relation ID maps and handle missing
        # entries using -1 as their index.
        json_files, entity_ids, relation_ids = self.generate_json_files_and_ids(directory, buffer_size)
        directory = os.path.dirname(json_files['full'])
        logger.info('The directory is {}'.format(directory))

        filetypes = ['train', 'dev', 'test']
        tf_record_filenames = {}
        for filetype in filetypes:
            count = 0
            total = 0
            file_index = 0
            filenames = glob.glob(os.path.join(
                directory, '{0}-{1}.tfrecords'.format(filetype, '*')))
            tf_record_filenames[filetype] = filenames
            if len(filenames) == 0:
                filename = os.path.join(
                    directory, '{0}-{1}.tfrecords'.format(filetype, file_index))
                logger.info('Writing to file: {}'.format(filename))
                tf_record_filenames[filetype] = [filename]
                tf_records_writer = tf.python_io.TFRecordWriter(filename)
                with open(json_files[filetype], 'r') as handle:
                    for line in handle:
                        sample = json.loads(line)
                        record = self._encode_sample_as_tf_record(
                            sample, entity_ids, relation_ids)
                        tf_records_writer.write(record.SerializeToString())
                        count += 1
                        total += 1
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
            print('Total records in %s: %d' % (filetype, total))

        def conve_tf_record_parser(r):
            features = {
                'e1': tf.FixedLenFeature([], tf.int64),
                'e2': tf.FixedLenFeature([], tf.int64),
                'rel': tf.FixedLenFeature([], tf.int64),
                'e2_multi1': tf.VarLenFeature(tf.int64),
                'is_inverse': tf.FixedLenFeature([], tf.int64)}
            return tf.parse_single_example(r, features=features)

        return conve_tf_record_parser, tf_record_filenames

    def load_and_preprocess(self, directory, buffer_size=1024 * 1024):
        logger.info(
            'Loading and preprocessing the \'%s\' dataset.', self.dataset_name)

        # Download and potentially extract all needed files.
        directory = os.path.join(directory, self.dataset_name)
        if self.maybe_extract(directory, buffer_size):
            # One more directory is created due to the archive extraction.
            directory = os.path.join(directory, self.dataset_name)

        # Load and preprocess the data.
        full_graph = {}  # Maps from (e1, rel) to set of e2 values.
        graphs = {}  # Maps from filename to dictionaries like labels.
        files = ['%s.txt' % f for f in self.filetypes]
        for i, f in enumerate(files):
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
                    if self.add_reverse_per_filetype[i]:
                        graphs[f][(e2, rel_reverse)].add(e1)

        # Write preprocessed files in a standardized JSON format.
        e1rel_to_e2_train = os.path.join(directory, 'e1rel_to_e2_train.json')
        e1rel_to_e2_dev = os.path.join(directory, 'e1rel_to_e2_dev.json')
        e1rel_to_e2_test = os.path.join(directory, 'e1rel_to_e2_test.json')
        e1rel_to_e2_full = os.path.join(directory, 'e1rel_to_e2_full.json')

        # Potentially remove from the test set the entities that do not appear in train.
        if self.needs_test_set_cleaning:
            assert 'train.txt' in graphs
            allowed_entities = set()
            allowed_relations = set()
            for key, value in six.iteritems(graphs['train.txt']):
                e1, rel = key
                e2_multi = list(value)
                allowed_entities.add(e1)
                allowed_entities.update(e2_multi)
                allowed_relations.add(rel)
        else:
            allowed_entities = None
            allowed_relations = None

        self._write_graph(e1rel_to_e2_train, graphs[files[0]])
        self._write_graph(e1rel_to_e2_dev, graphs[files[1]], full_graph,
                          allowed_entities=allowed_entities, allowed_relations=allowed_relations)
        self._write_graph(e1rel_to_e2_test, graphs[files[2]], full_graph,
                          allowed_entities=allowed_entities, allowed_relations=allowed_relations)
        self._write_graph(e1rel_to_e2_full, full_graph, full_graph)

        return {
            'train': e1rel_to_e2_train,
            'dev': e1rel_to_e2_dev,
            'test': e1rel_to_e2_test,
            'full': e1rel_to_e2_full}

    @staticmethod
    def _write_graph(filename, graph, labels=None, allowed_entities=None, allowed_relations=None):
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
                    if allowed_entities is not None and e1 not in allowed_entities:
                        print('Skipping e1 %50s.' % e1)
                        continue
                    if allowed_relations is not None and rel not in allowed_relations:
                        print('Skipping rel %50s.' % rel)
                        continue
                    e2_multi1 = ' '.join(list(labels[key]))
                    for e2 in value:
                        if allowed_entities is not None and e2 not in allowed_entities:
                            print('Skipping e2 %50s.' % e2)
                            continue
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


class _ConvEDataLoader(_DataLoader):
    def __init__(self, dataset_name):
        url = 'https://github.com/TimDettmers/ConvE/raw/master'
        filetypes = ['train', 'valid', 'test']
        add_reverse_per_filetype = [True, False, False]
        super(_ConvEDataLoader, self).__init__(url, [dataset_name + '.tar.gz'], dataset_name, filetypes,
                                               add_reverse_per_filetype=add_reverse_per_filetype)


class _MinervaDataLoader(_DataLoader):
    def __init__(self, dataset_name, needs_test_set_cleaning=False):
        url = 'https://raw.githubusercontent.com/shehzaadzd/MINERVA/master/datasets/data_preprocessed/%s' % dataset_name
        filenames = ['train.txt', 'dev.txt', 'test.txt']
        filetypes = ['train', 'dev', 'test']
        add_reverse_per_filetype = [True, False, False]
        super(_MinervaDataLoader, self).__init__(url, filenames, dataset_name, filetypes, needs_test_set_cleaning,
                                                 add_reverse_per_filetype=add_reverse_per_filetype)


class NationsLoader(_ConvEDataLoader):
    def __init__(self):
        dataset_name = 'nations'
        super(NationsLoader, self).__init__(dataset_name)


class UMLSLoader(_ConvEDataLoader):
    def __init__(self):
        dataset_name = 'umls'
        super(UMLSLoader, self).__init__(dataset_name)


class KinshipLoader(_ConvEDataLoader):
    def __init__(self):
        dataset_name = 'kinship'
        super(KinshipLoader, self).__init__(dataset_name)


class WN18RRLoader(_ConvEDataLoader):
    def __init__(self):
        dataset_name = 'WN18RR'
        super(WN18RRLoader, self).__init__(dataset_name)


class YAGO310Loader(_ConvEDataLoader):
    def __init__(self):
        dataset_name = 'YAGO3-10'
        super(YAGO310Loader, self).__init__(dataset_name)


class FB15k237Loader(_ConvEDataLoader):
    def __init__(self):
        dataset_name = 'FB15k-237'
        super(FB15k237Loader, self).__init__(dataset_name)


class CountriesS1Loader(_MinervaDataLoader):
    def __init__(self):
        dataset_name = 'countries_S1'
        super(CountriesS1Loader, self).__init__(dataset_name)


class CountriesS2Loader(_MinervaDataLoader):
    def __init__(self):
        dataset_name = 'countries_S2'
        super(CountriesS2Loader, self).__init__(dataset_name)


class CountriesS3Loader(_MinervaDataLoader):
    def __init__(self):
        dataset_name = 'countries_S3'
        super(CountriesS3Loader, self).__init__(dataset_name)


class NELL995Loader(_MinervaDataLoader):
    def __init__(self):
        dataset_name = 'nell-995'
        # NELL contains some test entities that do not appear during training. We remove those.
        super(NELL995Loader, self).__init__(dataset_name, needs_test_set_cleaning=False)
