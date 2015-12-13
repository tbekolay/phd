import copy
import os
import random
import re
import string

import numpy as np
import nwalign as nw
import pandas as pd
from nengo.utils.compat import range

from . import cache, ges_path, vtl
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
    for key in result_cls.to_int + ['seed']:
        df[key] = df[key].apply(int)
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

def get_syllables(n_syllables, minfreq, maxfreq, rng=np.random):
    allpaths = []
    for gdir in ['ges-de-ccv', 'ges-de-cv', 'ges-de-cvc', 'ges-de-v']:
        allpaths.extend([ges_path(gdir, gfile)
                         for gfile in os.listdir(ges_path(gdir))])
    indices = rng.permutation(len(allpaths))
    return ([allpaths[indices[i]] for i in range(n_syllables)],
            [rng.uniform(minfreq, maxfreq) for i in range(n_syllables)])


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
    """Convert a gesture score to a string for alignment.

    The string represents the gesture sequence, grouped by articulator set,
    with no timing information. Timing information is instead included in
    the other return values: duration, start time, and end time of all
    of the
    """
    strings = {}
    durations = {}
    t_start = {}
    t_end = {}
    for seq in gs.sequences:
        chars = []
        durs = []
        for gest in seq.gestures:
            if gest.neutral:
                if gest.duration_s > neutral_th:
                    chars.append("0")
                else:
                    continue
            elif gest.value in gs.labels:
                # Use ascii_letters; there are 52, so sufficient for gests
                chars.append(string.ascii_letters[gs.labels.index(gest.value)])
            elif seq.numerical:
                label = seq.type[:-len("-gestures")]
                old_min, old_max = vtl.VTL.numerical_range[label]
                val = rescale(gest.value, old_min, old_max, 0, 1)
                chars.append("9" if val > 0.3 else "0")
            else:
                raise ValueError("gest.value '%s' not recognized" % gest.value)
            durs.append(gest.duration_s)

        assert len(chars) == len(durs)

        # Need to remove repeated instances of the same char,
        # and sum their durations at the same time.
        i = 0
        while i+1 < len(chars):
            if chars[i] == chars[i+1]:
                del chars[i+1]
                durs[i] += durs[i+1]
                del durs[i+1]
            else:
                i += 1
        if len(chars) > 0 and chars[-1] == '0':
            # Remove ending neutral gesture
            del chars[-1]
            del durs[-1]

        # Determine the starting and ending times for each gesture
        t_s = []
        t_e = []
        for i in range(len(durs)):
            if i == 0:
                t_s.append(0.)
                t_e.append(durs[i])
            else:
                t_s.append(t_s[-1] + durs[i - 1])
                t_e.append(t_e[-1] + durs[i])

        strings[seq.type] = "".join(chars)
        durations[seq.type] = durs
        t_start[seq.type] = t_s
        t_end[seq.type] = t_e

    return ("".join([strings[k] for k in sorted(list(strings))]),
            [d for k in sorted(list(strings)) for d in durations[k]],
            [ts for k in sorted(list(strings)) for ts in t_start[k]],
            [te for k in sorted(list(strings)) for te in t_end[k]])


def gs_align(left, right):
    lstring, lduration, lstart, lend = left
    rstring, rduration, rstart, rend = right

    lalign, ralign = nw.global_align(lstring, rstring)

    def insert_placeholders(align, duration, start, end):
        for i, ch in enumerate(align):
            if ch == '-':
                duration.insert(i, None)
                start.insert(i, None)
                end.insert(i, None)
    insert_placeholders(lalign, lduration, lstart, lend)
    insert_placeholders(ralign, rduration, rstart, rend)
    align_ix = [i for i in range(len(lalign)) if lalign[i] == ralign[i]]
    return lalign, ralign, align_ix


def gs_accuracy(gs, targets):
    """Compare a gesture score to a collection of gesture score targets."""
    target = gs_combine(targets)
    gsinfo, tginfo = gs2strings(gs), gs2strings(target)
    lalign, ralign, _ = gs_align(gsinfo, tginfo)

    n_gestures = len(tginfo[0])
    n_sub = n_del = n_ins = 0
    for lchar, rchar in zip(lalign, ralign):
        if lchar == '-':
            n_ins += 1
        elif rchar == '-':
            n_del += 1
        elif lchar != rchar:
            n_sub += 1
    return (float(n_gestures - n_sub - n_del - n_ins) / n_gestures,
            n_sub, n_del, n_ins)


def _gs_accuracy_baseline():
    """Determine a baseline for gesture score accuracy.

    In the experiments, we generate a syllable sequence and compare the
    reconstructed gesture score to the actual gesture scores.
    This function compares all existing syllable gestures to all others
    in order to determine the baseline accuracy measure for comparing
    different sequences to one another, the idea being that if we do a
    good job reconstructing, our measure will be significantly larger
    than this value.
    """
    acc = []

    # There are 417 syllables
    paths, _ = get_syllables(n_syllables=417, minfreq=1, maxfreq=1)
    gests = [vtl.parse_ges(path) for path in paths]
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            acc.append(gs_accuracy(gests[i], [gests[j]])[0])

    return np.mean(acc)

# Obtained by running the above function (it's expensive though)
gs_accuracy_baseline = 0.51642353200356783


def gs_timing(gs, targets):
    """Compare the timing of a gesture score to a collection of targets.

    The difference in timing is the difference in durations between gestures
    that are identified as being the same (using a similar algorithm as
    gs_accuracy). We return the mean accuracy, which should indicate the
    overall speed bias, and the variance, which is how reliably the
    reconstruction
    """
    target = gs_combine(targets)
    gsinfo, tginfo = gs2strings(gs), gs2strings(target)
    lalign, ralign, align_ix = gs_align(gsinfo, tginfo)

    gsdurs = np.array(gsinfo[1])[align_ix]
    tgdurs = np.array(tginfo[1])[align_ix]
    durdiff = gsdurs - tgdurs
    return np.mean(durdiff), np.var(durdiff)


def gs_cooccur(gs, targets, th=0.005):
    """Compare the timing of cooccurring gestures to those in targets.

    Gesture that start or end at the same time are critical to well-formed
    speech. Here, we ensure that those cooccurrences are captured in a
    given gesture compared to the targets.
    """
    target = gs_combine(targets)
    gsinfo, tginfo = gs2strings(gs), gs2strings(target)
    lalign, ralign, align_ix = gs_align(gsinfo, tginfo)

    def filtaligned(l):
        return np.array(l)[align_ix].tolist()
    gs_start, gs_end = filtaligned(gsinfo[2]), filtaligned(gsinfo[3])
    tg_start, tg_end = filtaligned(tginfo[2]), filtaligned(tginfo[3])

    # Shuffled versions for determining chance levels
    shuff_starts = random.sample(gs_start, len(gs_start))
    shuff_ends = random.sample(gs_end, len(gs_end))

    # Get the cooccurring indices from the target score
    def cooccurr_ix(l, th=0.001):
        ix = []
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                if abs(l[i] - l[j]) < th:
                    ix.append((i, j))
        return ix
    costart_ix = cooccurr_ix(tg_start)
    coend_ix = cooccurr_ix(tg_end)

    def accuracy(starts, ends):
        good = bad = 0

        for lix, rix in costart_ix:
            if abs(starts[lix] - starts[rix]) < th:
                good += 1
            else:
                bad += 1
        for lix, rix in coend_ix:
            if abs(ends[lix] - ends[rix]) < th:
                good += 1
            else:
                bad += 1
        if good + bad == 0:
            # Ignore situations with no co-occurrence
            return np.nan
        return good / float(good + bad)

    # Accuracy of gs; chance accuracy
    return accuracy(gs_start, gs_end), accuracy(shuff_starts, shuff_ends)


# #############################
# Model 3: Syllable recognition
# #############################

def classinfo(classdata, dmpdata):
    dclass = np.diff((classdata > 0.001).astype(float), axis=0)
    time_ix, _ = np.where(dclass > 0)

    # Find the iDMP most active when the classification happened
    class_ix = np.argmax(dmpdata[time_ix], axis=1)
    return time_ix, class_ix


def cl2string(recorded, target):
    all_labels = list(set([l for _, l in recorded] + [l for _, l in target]))
    l2char = {all_labels[i]: string.ascii_letters[i]
              for i in range(len(all_labels))}
    rec_s = "".join(l2char[l] for _, l in recorded)
    tgt_s = "".join(l2char[l] for _, l in target)

    recalign, tgtalign = nw.global_align(rec_s, tgt_s)
    rectimes = [t for t, _ in recorded]
    tgttimes = [t for t, _ in target]

    for i, (recchar, tgtchar) in enumerate(zip(recalign, tgtalign)):
        if tgtchar == '-':
            tgttimes.insert(i, None)
        if recchar == '-':
            rectimes.insert(i, None)
    return recalign, tgtalign, rectimes, tgttimes


def cl_accuracy(recorded, target):
    recalign, tgtalign, _, _ = cl2string(recorded, target)

    n_phones = len(target)
    n_sub = n_del = n_ins = 0
    for recchar, tgtchar in zip(recalign, tgtalign):
        if tgtchar == '-':
            n_ins += 1
        elif recchar == '-':
            n_del += 1
        elif tgtchar != recchar:
            n_sub += 1
    acc = float(n_phones - n_sub - n_del - n_ins) / n_phones

    return acc, n_sub, n_del, n_ins


def cl_timing(recorded, target):
    recalign, tgtalign, rectimes, tgttimes = cl2string(recorded, target)

    correct_ix = [i for i in range(len(recalign))
                  if recalign[i] == tgtalign[i]]
    t_diff = [rectimes[i] - tgttimes[i] for i in correct_ix]

    return np.mean(t_diff), np.var(t_diff)
