from __future__ import absolute_import, division, print_function

from time import time
import logging,inspect
import pickle
from itertools import islice
import os.path

__all__ = [
    'returnPathStruc2vec', 'isPickle', 'chunks', 'partition',
    'restore_variable_from_disk', 'save_variable_on_disk']

LOGGER = logging.getLogger(__name__)

dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
folder_pickles = dir_f + "/../pickles/"
os.makedirs(folder_pickles, exist_ok=True)


def returnPathStruc2vec():
    return dir_f


def isPickle(fname):
    return os.path.isfile(dir_f+'/../pickles/'+fname+'.pickle')


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}


def partition(lst, n):
    lst = list(lst)
    division = len(lst) / float(n)
    output = [lst[int(round(division * i)): int(round(division * (i + 1)))]
              for i in range(n)]
    return output


def restore_variable_from_disk(name):
    logging.info('Recovering variable...')
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()
    logging.info('Variable recovered. Time: {}m'.format((t1-t0)/60))

    return val


def save_variable_on_disk(f, name):
    logging.info('Saving variable on disk...')
    t0 = time()
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variable saved. Time: {}m'.format((t1-t0)/60))
