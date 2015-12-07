import copy
import os
import re
import string
from itertools import groupby

import numpy as np
import pandas as pd

from . import cache, vtl
from .utils import rescale


def load_results(result_cls, keys):
    ffilter = ",".join("%s:(.*)" % key for key in keys)
    ffilter += ",seed:(.*)\.npz"
    ffilter = re.compile(ffilter)
    cdir = (cache.cache_dir if result_cls.subdir is None
            else os.path.join(cache.cache_dir, result_cls.subdir))
    data = []
    keys.append('seed')

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
    df['seed'] = df['seed'].apply(int)
    return df


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

# The only analysis here is classification accuracy,
# so `np.mean(predited == actual)`.
# Most of the effort is in running simualtions and aggregating results.

# ############################
# Model 2: Syllable production
# ############################

def gs_combine(scores):
    """Combine a set of gesture scores into a single gesture score."""
    default_tau = 0.015

    # Start by deepcopying the first gesture score
    gs = copy.deepcopy(scores[0])
    seqs = {seq.type: seq for seq in gs.sequences}

    for score in scores[1:]:
        # To ensure time alignment, we first extend the current score
        # to the length of the longest sequence
        t_end = gs.t_end
        for seq in gs.sequences:
            t_diff = t_end - seq.t_end
            if t_diff > 0:
                # Add a neutral gesture to fill time
                seq.gestures.append(
                    vtl.Gesture("", 0., t_diff, default_tau, True))

        # Add all the gestures from the new score
        for seq in score.sequences:
            seqs[seq.type].gestures.extend(copy.deepcopy(seq.gestures))

    return gs


def gs2strings(gs, neutral_th=0.05):
    """Convert a gesture score to a list of strings.

    Each string represents the gesture sequence for one
    of the articulator sets, with no timing information.
    """
    strings = {}
    for seq in gs.sequences:
        chars = []
        for gest in seq.gestures:
            if gest.neutral:
                if gest.duration_s > neutral_th:
                    chars.append("0")
            elif gest.value in gs.labels:
                # Use ascii_letters; there are 52, so sufficient for gests
                chars.append(string.ascii_letters[gs.labels.index(gest.value)])
            elif seq.numerical:
                label = seq.type[:-len("-gestures")]
                old_min, old_max = vtl.VTL.numerical_range[label]
                val = rescale(gest.value, old_min, old_max, 0, 1)
                val = np.clip(val, 0, 0.999)
                chars.append(str(int(val * 10)))
            else:
                raise ValueError("gest.value '%s' not recognized" % gest.value)
        # Use groupby to remove repeated instances of the same char
        st = "".join(c for c, _ in groupby(chars))
        st = st[:-1] if st.endswith("0") else st
        strings[seq.type] = st
    return "".join([strings[k] for k in sorted(list(strings))])


def gs_accuracy(gs, targets):
    """Compare a gesture score to a collection of gesture score targets."""
    target = gs_combine(targets)

    # We'll compare sequence by sequence
    # for seq in


def gs_accuracy_baseline():
    """Determine a baseline for gesture score accuracy.

    In the experiments, we generate a syllable sequence and compare the
    reconstructed gesture score to the actual gesture scores.
    This function compares all existing syllable gestures to all others
    in order to determine the baseline accuracy measure for comparing
    different sequences to one another, the idea being that if we do a
    good job reconstructing, our measure will be significantly larger
    than this value.
    """

def gs_timing(gs, targets):
    """Compare the timing of a gesture score to a collection of targets.

    The difference in timing is the difference in durations between gestures
    that are identified as being the same (using a similar algorithm as
    gs_accuracy). We return the mean accuracy, which should indicate the
    overall speed bias, and the variance, which is how reliably the
    reconstruction
    """
    pass


def gs_cooccur(gs, targets):
    """Compare the timing of cooccurring gestures to those in targets.

    Gesture that start or end at the same time are critical to well-formed
    speech. Here, we ensure that those cooccurrences are captured in a
    given gesture compared to the targets.
    """
    pass



# #############################
# Model 3: Syllable recognition
# #############################
