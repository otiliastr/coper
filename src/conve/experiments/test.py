from __future__ import absolute_import, division, print_function

import os

from ..data.loaders import KinshipLoader

working_dir = os.path.join(os.getcwd(), 'temp')
data_dir = os.path.join(working_dir, 'data')

loader = KinshipLoader()
loader.load_and_preprocess(data_dir)
