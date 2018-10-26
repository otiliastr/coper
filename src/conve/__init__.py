from __future__ import absolute_import, division, print_function

import logging.config
import os
import yaml

from . import cpg
from . import data
from . import evaluation
from . import experiments
from . import models
from . import utilities

from .cpg import *
from .data import *
from .evaluation import *
from .experiments import *
from .models import *
from .utilities import *

__all__ = ['cpg', 'data', 'evaluation', 'experiments', 'models', 'utilities']
__all__.extend(cpg.__all__)
__all__.extend(data.__all__)
__all__.extend(evaluation.__all__)
__all__.extend(experiments.__all__)
__all__.extend(utilities.__all__)

__logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
if os.path.exists(__logging_config_path):
    with open(__logging_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.getLogger('').addHandler(logging.NullHandler())
