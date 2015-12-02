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
    data = []

    for fname in os.listdir(cdir):
        match = ffilter.match(fname)
        if match is not None:
            res = result_cls.load(fname[:-4]).__dict__
            for i, key in enumerate(keys):
                res[key] = match.group(i+1)
            data.append(res)
    if len(data) == 0:
        raise ValueError("No files matched those keys. Typo?")
    df = pd.DataFrame(data)
    for key in result_cls.to_float:
        df[key] = df[key].apply(float)
    return df





# #####################################
# Model 1: Neural cepstral coefficients
# #####################################



# ############################
# Model 2: Syllable production
# ############################


# #############################
# Model 3: Syllable recognition
# #############################
