import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from . import cache


def load_results(result_cls, keys):
    ffilter = ",".join("%s:(.*)" % key for key in keys)
    ffilter += ",seed:.*"
    ffilter = re.compile(ffilter)
    cdir = (cache.cache_dir if result_cls.subdir is None
            else os.path.join(cache.cache_dir, result_cls.subdir))
    data = defaultdict(list)

    for fname in os.listdir(cdir):
        match = ffilter.match(fname)
        if match is not None:
            res = result_cls.load(fname[:-4])
            data[match.groups()].append(res)
    return data


def results2dataframe(results, keys):
    # First, get all the result dictionaries
    rdicts = []
    for group in results:
        pass




# #####################################
# Model 1: Neural cepstral coefficients
# #####################################



# ############################
# Model 2: Syllable production
# ############################


# #############################
# Model 3: Syllable recognition
# #############################
