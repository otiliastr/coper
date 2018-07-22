from __future__ import absolute_import, division, print_function

import abc
import json
import logging
import os
import tarfile
import six

import requests

from tqdm import tqdm

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

    def load_and_preprocess(self, directory, buffer_size=1024 * 1024):
        logger.info('Loading ConvE dataset: %s', self.dataset_name)

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
        self._write_graph(e1rel_to_e2_full, full_graph)

    def _write_graph(self, filename, graph, labels=None):
        with open(filename, 'w') as handle:
            for key, value in six.iteritems(graph):
                if labels is None:
                    sample = {
                        'e1': key[0],
                        'e2': 'None',
                        'rel': key[1],
                        'rel_eval': 'None',
                        'e2_multi1': ' '.join(list(value)),
                        'e2_multi2': 'None'}
                    handle.write(json.dumps(sample)  + '\n')
                elif not key[1].endswith('_reverse'):
                    e1, rel = key
                    rel_reverse = rel + '_reverse'
                    e2_multi1 = ' '.join(list(graph[key]))
                    for e2 in value:
                        key_reverse = (e2, rel_reverse)
                        e2_multi2 = ' '.join(list(graph[key_reverse]))
                        sample = {
                            'e1': e1,
                            'e2': e2,
                            'rel': rel,
                            'rel_eval': rel_reverse,
                            'e2_multi1': e2_multi1,
                            'e2_multi2': e2_multi2}
                        handle.write(json.dumps(sample)  + '\n')


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
