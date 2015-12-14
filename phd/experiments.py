from __future__ import print_function

import os
import sys
import warnings

import nengo
import nengo.utils.numpy as npext
import numpy as np
import scipy
from nengo import spa
from nengo.cache import NoDecoderCache
from nengo.utils.compat import range
from nengo.utils.stdlib import Timer
from sklearn.svm import LinearSVC
from scipy.interpolate import interp1d
from scipy import stats

from . import analysis, cache, config, parallel, vtl
from .timit import TIMIT
from .utils import derivative, rescale


def log(msg):
    if config.log_experiments:
        print(msg)
        sys.stdout.flush()


class ExperimentResult(object):

    saved = []
    to_float = []
    to_int = []
    subdir = None

    def __init__(self, **kwargs):
        for key in self.saved:
            setattr(self, key, kwargs.pop(key, None))
        # kwargs should be empty now
        for key in kwargs:
            raise TypeError("got an unexpected keyword argument '%s'" % key)

    def save_dict(self):
        return {k: getattr(self, k) for k in self.saved}

    @classmethod
    def load(cls, key):
        data = cache.load_obj(key, ext='npz', subdir=cls.subdir)
        return cls(**data)

    def save(self, key):
        cache.cache_obj(self.save_dict(),
                        key=key, ext='npz', subdir=self.subdir)


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

def mfcc(model, sample, zscore):
    model.audio = sample
    feat = model.mfcc()
    if zscore and feat.shape[0] > 2:
        # TRY: take the zscore for each block separately
        #      where block is normal, deriv1, deriv2, etc
        feat = stats.zscore(feat, axis=0)    # If variance is 0, can get nans.
        feat[np.isnan(feat)] = 0.
    return feat


def mfccs(model, audio, zscore):
    out = {label: [] for label in audio}
    for label in audio:
        for sample in audio[label]:
            out[label].append(mfcc(model, sample, zscore))
    model.audio = np.zeros(1)
    return out


def ncc(model, sample, zscore, seed, upsample=False):
    if upsample:
        fs_scale = 50000. / TIMIT.fs
        resample_len = int(sample.shape[0] * fs_scale)
        sample = lengthen(sample, resample_len)
        model.fs = 50000

    model.audio = sample
    net = model.build(nengo.Network(seed=seed))
    with net:
        pr = nengo.Probe(net.output, synapse=0.01)
    # Disable decoder cache for this model
    _model = nengo.builder.Model(dt=0.001, decoder_cache=NoDecoderCache())
    sim = nengo.Simulator(net, model=_model)
    sim.run(model.t_audio, progress_bar=False)
    model.audio = np.zeros(1)
    feat = sim.data[pr]
    if zscore:
        feat = stats.zscore(feat, axis=0)
        feat[np.isnan(feat)] = 0.  # If variance is 0, can get nans.
    return feat


def nccs(model, audio, zscore, seed, upsample):
    out = {label: [] for label in audio}

    if parallel.get_pool() is not None:  # Asynchronous / parallel version
        jobs = {label: [] for label in audio}

        for label in audio:
            for sample in audio[label]:
                jobs[label].append(parallel.get_pool().apply_async(
                    ncc, [model, sample, zscore, seed, upsample]))
        for label in jobs:
            for result in jobs[label]:
                out[label].append(result.get())
    else:  # Synchronous / serial version
        for label in audio:
            for sample in audio[label]:
                out[label].append(ncc(model, sample, zscore, seed, upsample))

    return out


def normalize(features, n_frames):
    out = {label: [] for label in features}
    for label in features:
        for sample in features[label]:
            if sample.shape[0] < n_frames:
                sample = lengthen(sample, n_frames)
            elif sample.shape[0] > n_frames:
                sample = shorten(sample, n_frames)
            # Flatten the vector so that SVM can handle it
            out[label].append(sample.reshape(-1))
    return out


def shorten(feature, n_frames):
    """Compute neighbourhood mean to shorten the feature vector."""
    assert feature.shape[0] >= n_frames
    scale = int(feature.shape[0] / n_frames)
    pad_size = int(np.ceil(float(feature.shape[0]) / scale) * scale
                   - feature.shape[0])
    feature = np.vstack([feature,
                         np.zeros((pad_size, feature.shape[1])) * np.nan])
    return scipy.nanmean(
        feature.reshape(-1, scale, feature.shape[1]), axis=1)[:n_frames]


def lengthen(feature, n_frames):
    """Use linear interpolation to lengthen the feature vector."""
    assert feature.shape[0] <= n_frames
    if feature.shape[0] == 1:
        feature = np.tile(feature, (2, 1))
    interp_x = np.linspace(0, n_frames, feature.shape[0])
    f = interp1d(interp_x, feature, axis=0, assume_sorted=True)
    return f(np.arange(n_frames))


def test_svm(svm, x, y, data="Testing"):
    pred_y = svm.predict(x)
    acc = np.mean(pred_y == y)
    log("%s accuracy: %.4f" % (data, acc))
    return pred_y, y, acc


class AuditoryFeaturesExperiment(object):
    def __init__(self, model, phones=None, words=None,
                 zscore=None, seed=None, upsample=False):
        self.model = model
        assert phones is None or words is None, "Can only set one, not both"
        self.phones = phones
        self.words = words
        self.seed = np.random.randint(npext.maxint) if seed is None else seed
        self.zscore = zscore
        self.upsample = upsample
        self.timit = TIMIT(config.timit_root)

    def _get_audio(self, corpus):
        if self.phones is not None:
            return self.timit.phn_samples(self.phones, corpus=corpus)
        elif self.words is not None:
            return self.timit.word_samples(self.words, corpus=corpus)

    def _get_feature(self, feature, audio, result=None, n_frames=None):
        labels = sorted(list(audio))
        with Timer() as t:
            if feature == 'mfcc':
                # Default to zscoring for MFCCs
                zscore = True if self.zscore is None else self.zscore
                x = mfccs(self.model, audio, zscore)
            elif feature == 'ncc':
                # Default to not zscoring for NCCs
                zscore = False if self.zscore is None else self.zscore
                x = nccs(self.model, audio, zscore, self.seed, self.upsample)
            else:
                raise ValueError("Possible features: 'mfcc', 'ncc'")
        log("%ss generated in %.3f seconds" % (feature.upper(), t.duration))
        if result is not None:
            setattr(result, "%s_time" % feature, t.duration)

        if n_frames is None:
            n_frames = max(max(xx.shape[0] for xx in x[l]) for l in audio)
        x = normalize(x, n_frames)
        return np.vstack([np.vstack(x[l]) for l in labels])

    @staticmethod
    def _get_labels(audio):
        labels = sorted(list(audio))
        return np.array([l for l in labels for _ in range(len(audio[l]))])

    def train(self, result, n_frames=None):

        def fit_svm(x, y, feature):
            svm = LinearSVC(random_state=self.seed)
            with Timer() as t:
                svm.fit(x, y)
            log("SVM fitting for %ss done in %.3f seconds"
                % (feature.upper(), t.duration))
            setattr(result, "%s_fit_time" % feature, t.duration)
            _, _, acc = test_svm(svm, x, y, "Training")
            setattr(result, "%s_train_acc" % feature, acc)
            return svm

        audio = self._get_audio(corpus="train")

        # NB! Do MFCC first to get n_frames for NCC.
        x_mfcc = self._get_feature('mfcc', audio, result, n_frames=n_frames)
        n_frames = int(x_mfcc.shape[1] // self.model.dimensions)
        y = self._get_labels(audio)
        mfcc_svm = fit_svm(x_mfcc, y, 'mfcc')

        x_ncc = self._get_feature('ncc', audio, result, n_frames=n_frames)
        ncc_svm = fit_svm(x_ncc, y, 'ncc')

        # Store all in result
        result.y = y

        return mfcc_svm, ncc_svm

    def test(self, result, svm, feature):
        audio = self._get_audio(corpus="test")
        n_frames = int(svm.coef_.shape[1] // self.model.dimensions)
        x = self._get_feature(feature, audio, n_frames=n_frames)
        y = self._get_labels(audio)
        pred, y, acc = test_svm(svm, x, y, "Testing")
        setattr(result, "%s_test_acc" % feature, acc)
        setattr(result, "%s_pred" % feature, pred)

    def cache_file(self, key=None):
        key = cache.generic_key(self) if key is None else key
        return cache.cache_file(key, ext='npz', subdir='ncc')

    def run(self, key=None, n_frames=None):
        key = cache.generic_key(self) if key is None else key
        if cache.cache_file_exists(key, ext='npz', subdir='ncc'):
            log("%s.npz in the cache. Loading." % key)
            result = AuditoryFeaturesResult.load(key)
        else:
            result = AuditoryFeaturesResult()
            log("%s.npz not in the cache. Running." % key)
            log("==== Training ====")
            mfcc_svm, ncc_svm = self.train(result, n_frames=n_frames)
            log("==== Testing ====")
            self.test(result, mfcc_svm, 'mfcc')
            self.test(result, ncc_svm, 'ncc')
            result.save(key)
            log("Experiment run saved to the cache.")
        return key


class AuditoryFeaturesResult(ExperimentResult):

    saved = ['mfcc_time',
             'mfcc_fit_time',
             'mfcc_train_acc',
             'mfcc_test_acc',
             'mfcc_pred',
             'ncc_time',
             'ncc_fit_time',
             'ncc_train_acc',
             'ncc_test_acc',
             'ncc_pred',
             'y']
    to_float = ['mfcc_time',
                'mfcc_fit_time',
                'mfcc_train_acc',
                'mfcc_test_acc',
                'ncc_time',
                'ncc_fit_time',
                'ncc_train_acc',
                'ncc_test_acc']
    subdir = "ncc"


# ############################
# Model 2: Syllable production
# ############################

def ix2seqlabel(ix, labels):
    if ix < labels.index('ll-labial-nas'):
        return 'vowel-gestures'
    elif ix < labels.index('tt-alveolar-nas'):
        return 'lip-gestures'
    elif ix < labels.index('tb-palatal-fric'):
        return 'tongue-tip-gestures'
    elif ix < labels.index('breathy'):
        return 'tongue-body-gestures'
    elif ix < labels.index('velic'):
        return 'glottal-shape-gestures'
    elif ix < labels.index('lung-pressure'):
        return 'velic-gestures'
    else:
        return 'lung-pressure-gestures'


def gesture_score(traj, dt, dspread=18, dthresh=0.011):
    """Construct a gesture score given a trajectory."""
    # --- Take derivative and find times with high derivative
    trajd = np.abs(derivative(traj, dspread))
    slices = trajd > dthresh

    # --- Find x and y indices for slice starts and ends
    diff_in = np.vstack([np.zeros(trajd.shape[1]),
                         slices,
                         np.zeros(trajd.shape[1])])
    x_ind, y_ind = np.where(np.abs(np.diff(diff_in, axis=0)))
    if x_ind.size % 2 != 0:
        raise ValueError("Odd number of slice start/ends, not good.")

    # --- Sort by seqs
    synth = vtl.VTL()
    labels = synth.gesture_labels()
    labels.remove("f0")
    seqs = np.array([ix2seqlabel(yi, labels) for yi in y_ind])
    sort_ix = np.argsort(seqs)
    seqs = seqs[sort_ix]
    x_ind = x_ind[sort_ix]
    y_ind = y_ind[sort_ix]

    # --- Sort by x within seqs
    for seq in np.unique(seqs):
        subset = seqs == seq
        x_order = np.argsort(x_ind[subset])
        x_ind[subset] = x_ind[subset][x_order]
        y_ind[subset] = y_ind[subset][x_order]

    # Convert to lists so that we can process and delete them
    seqs = seqs.tolist()
    x_ind = x_ind.tolist()
    y_ind = y_ind.tolist()

    # --- Construct the gesture score
    gs = vtl.GestureScore(labels)
    # Make an initial sequence
    if len(seqs) == 0:
        return gs
    seq = vtl.GestureSequence(seqs[0])
    gs.sequences.append(seq)

    while len(x_ind) > 0:
        this_start, this_y, this_seq = x_ind.pop(0), y_ind.pop(0), seqs.pop(0)

        # Get only the x values for the same y
        filt_x = [(i, x_ind[i]) for i in range(len(y_ind))
                  if y_ind[i] == this_y]
        ix, this_end = filt_x[0]
        assert x_ind[ix] == this_end
        del x_ind[ix]
        assert y_ind[ix] == this_y
        del y_ind[ix]
        assert seqs[ix] == this_seq
        del seqs[ix]

        if len(filt_x) == 1:
            # At the end of this gesture
            next_start = next_end = trajd.shape[0]
        else:
            next_start, next_end = filt_x[1][1], filt_x[2][1]

        if this_seq != seq.type:
            # Moved to the next sequence!
            seq = vtl.GestureSequence(this_seq)
            gs.sequences.append(seq)

        # Use the midpoint of the high derivative slice
        st = int((this_end + this_start) // 2)
        ed = int((next_end + next_start) // 2)
        tr_slice = traj[st:ed, this_y]

        # Median gives a somewhat more robust measure
        # of if we're on or off
        if np.median(tr_slice) < 0.1:
            # We ignore neutral gesture that occur, but later add our own
            continue

        if seq.numerical:
            value = tr_slice.max()  # Max gives a better measure than median
            value = rescale(value, 0, 1,
                            *synth.numerical_range[labels[this_y]])
        else:
            value = labels[this_y]

        tau = (this_end - this_start) * dt * 0.5  # scale to match examples
        duration = (ed - st) * dt

        # Add in a neutral gesture to make sure this gesture is time-aligned
        t_diff = round(st * dt, 3) - round(seq.t_end, 3)
        if t_diff > 0:
            seq.gestures.append(vtl.Gesture("", 0., t_diff, tau, True))
        elif t_diff < 0:
            warnings.warn("Gesture start time is before seq.t_end; difference "
                          "is %s. Trying to compensate..." % t_diff)
            duration += t_diff

        if duration < 0:
            warnings.warn("Duration is negative. Messed up! Skipping gesture.")
        else:
            # Finally, add the gesture for this time slice
            seq.gestures.append(vtl.Gesture(value, 0., duration, tau, False))
    return gs


def ideal_traj(model, sequence):
    traj = []
    for syll in sequence:
        syllable = model.syllable_dict[syll.upper()]
        t_frames = int((1. / syllable.freq) / model.trial.dt)
        if t_frames >= syllable.trajectory.shape[0]:
            traj.append(lengthen(syllable.trajectory, t_frames))
        else:
            traj.append(shorten(syllable.trajectory, t_frames))
    return np.vstack(traj)


def path2label(ges_path):
    return os.path.basename(ges_path)[:-4].upper()


class ProductionExperiment(object):
    def __init__(self, model, n_syllables, sequence_len,
                 minfreq=0.8, maxfreq=3.0, seed=None):
        self.model = model
        self.n_syllables = n_syllables
        self.sequence_len = sequence_len
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

    def run_model(self):
        res = ProductionResult()
        rng = np.random.RandomState(self.seed)
        dt = self.model.trial.dt

        paths, freqs = analysis.get_syllables(
            self.n_syllables, self.minfreq, self.maxfreq, rng)

        t = 0.8  # 0.2 for initial stuff, 0.6 fudge factor
        gs_targets = []
        for path, freq in zip(paths, freqs):
            gs = vtl.parse_ges(path)
            gs_targets.append(gs)
            self.model.add_syllable(label=path2label(path),
                                    freq=freq,
                                    trajectory=gs.trajectory(dt))
            t += 1. / freq

        # Determine and save syllable sequence
        if self.model.trial.repeat:
            seq_ix = rng.randint(len(paths), size=self.sequence_len)
        else:
            seq_ix = rng.permutation(len(paths))[:self.sequence_len]
        seq = [path2label(paths[i]) for i in seq_ix]
        seq_str = " + ".join(["%s*POS%d" % (label, i+1)
                              for i, label in enumerate(seq)])
        self.model.trial.sequence = seq_str
        res.seq = np.array(seq)

        # Use a minimum number of syllables
        self.model.sequence.n_positions = self.n_syllables

        # Save frequencies for that sequence
        res.freqs = np.array([self.model.syllables[i].freq for i in seq_ix])

        net = self.model.build(nengo.Network(seed=self.seed))
        with net:
            p_out = nengo.Probe(net.production_info.output, synapse=0.01)

        sim = nengo.Simulator(net)
        sim.run(t)

        # Get ideal trajectory; compare RMSE
        delay_frames = int(self.model.trial.t_release / dt)
        traj = ideal_traj(self.model, seq)
        res.traj = traj

        simtraj = sim.data[p_out][delay_frames:]
        simtraj = simtraj[:traj.shape[0]]
        traj = traj[:simtraj.shape[0]]
        res.simtraj = simtraj
        res.simrmse = npext.rmse(traj, simtraj)

        # Reconstruct gesture score; compare to originals
        gs = gesture_score(simtraj, dt)
        res.accuracy, res.n_sub, res.n_del, res.n_ins = (
            analysis.gs_accuracy(gs, gs_targets))
        res.timing_mean, res.timing_var = analysis.gs_timing(gs, gs_targets)
        res.cooccur, res.co_chance = analysis.gs_cooccur(gs, gs_targets)
        log("Accuracy: %.3f" % res.accuracy)

        # Get the reconstructed trajectory and audio
        reconstructed = gs.trajectory(dt=dt)
        res.reconstructed = reconstructed
        minsize = min(reconstructed.shape[0], traj.shape[0])
        res.reconstructedrmse = npext.rmse(traj[:minsize],
                                           reconstructed[:minsize])
        audio, fs = gs.synthesize()
        res.audio = audio
        res.fs = fs

        tgt_gs = analysis.gs_combine([vtl.parse_ges(paths[i]) for i in seq_ix])
        audio, _ = tgt_gs.synthesize()
        res.clean_audio = audio

        return res

    def cache_file(self, key=None):
        key = cache.generic_key(self) if key is None else key
        return cache.cache_file(key, ext='npz', subdir='prod')

    def run(self, key=None):
        key = cache.generic_key(self) if key is None else key
        if cache.cache_file_exists(key, ext='npz', subdir='prod'):
            log("%s.npz in the cache. Loading." % key)
            result = ProductionResult.load(key)
        else:
            log("%s.npz not in the cache. Running." % key)
            result = self.run_model()
            result.save(key)
            log("Experiment run saved to the cache.")
        return key


class ProductionResult(ExperimentResult):

    saved = ['seq',
             'freqs',
             'traj',
             'simtraj',
             'simrmse',
             'reconstructed',
             'reconstructedrmse',
             'accuracy',
             'n_sub',
             'n_del',
             'n_ins',
             'timing_mean',
             'timing_var',
             'cooccur',
             'co_chance',
             'audio',
             'fs',
             'clean_audio']
    to_int = ['n_sub',
              'n_del',
              'n_ins']
    to_float = ['simrmse',
                'reconstructedrmse',
                'accuracy',
                'timing_mean',
                'timing_var',
                'cooccur',
                'co_chance']
    subdir = "prod"


# #############################
# Model 3: Syllable recognition
# #############################

class RecognitionExperiment(object):
    def __init__(self, model, n_syllables, sequence_len,
                 minfreq=1.3, maxfreq=1.5, seed=None):
        self.model = model
        self.n_syllables = n_syllables
        self.sequence_len = sequence_len
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

    def run_model(self):
        res = RecognitionResult()
        rng = np.random.RandomState(self.seed)

        paths, freqs = analysis.get_syllables(
            self.n_syllables, self.minfreq, self.maxfreq, rng)

        for path, freq in zip(paths, freqs):
            traj = vtl.parse_ges(path).trajectory(self.model.trial.dt)
            self.model.add_syllable(label=path2label(path),
                                    freq=freq,
                                    trajectory=traj)

        # Determine and save syllable sequence
        if self.model.trial.repeat:
            seq_ix = rng.randint(len(paths), size=self.sequence_len)
        else:
            seq_ix = rng.permutation(len(paths))[:self.sequence_len]
        seq = [path2label(paths[i]) for i in seq_ix]
        res.seq = np.array(seq)

        # Determine how long to run
        simt = 0.0
        tgt_time = []
        for label in res.seq:
            syllable = self.model.syllable_dict[label]
            simt += 1. / syllable.freq
            tgt_time.append(simt)

        # Set that sequence in the model
        traj = ideal_traj(self.model, seq)
        self.model.trial.trajectory = traj

        # Save frequencies for that sequence
        res.freqs = np.array([self.model.syllables[i].freq for i in seq_ix])

        # -- Run the model
        net = self.model.build(nengo.Network(seed=self.seed))
        with net:
            p_dmps = [nengo.Probe(dmp.state[0], synapse=0.01)
                      for dmp in net.syllables]
            p_class = nengo.Probe(net.classifier, synapse=0.01)
            p_mem = nengo.Probe(net.memory.output, synapse=0.01)

        sim = nengo.Simulator(net)
        sim.run(simt)

        # Save iDMP system states
        res.dmps = np.hstack([sim.data[p_d] for p_d in p_dmps])
        res.dmp_labels = np.array([s.label for s in self.model.syllables])

        # Save working memory similarities
        res.memory = spa.similarity(sim.data[p_mem], net.vocab, True)

        # Determine classification times and labels
        t_ix, class_ix = analysis.classinfo(sim.data[p_class], res.dmps)
        res.class_time = sim.trange()[t_ix]
        res.class_labels = np.array([path2label(paths[ix]) for ix in class_ix])

        # Calculate accuracy / timing metrics
        recinfo = [(t, l) for t, l in zip(res.class_time, res.class_labels)]
        tgtinfo = [(t, l) for t, l in zip(tgt_time, res.seq)]
        res.acc, res.n_sub, res.n_del, res.n_ins = (
            analysis.cl_accuracy(recinfo, tgtinfo))
        res.tdiff_mean, res.tdiff_var = analysis.cl_timing(recinfo, tgtinfo)
        log("Accuracy: %.3f" % res.acc)

        # Determine if memory representation is correct
        tgt_time = np.asarray(tgt_time)
        mem_times = (tgt_time[1:] + tgt_time[:-1]) * 0.5
        mem_ix = (mem_times / self.model.trial.dt).astype(int)
        mem_class = np.argmax(res.memory[mem_ix], axis=1)
        slabels = [s.label for s in self.model.syllables]
        actual = np.array([slabels.index(lbl) for lbl in res.seq[:-1]])
        res.memory_acc = np.mean(mem_class == actual)

        return res

    def cache_file(self, key=None):
        key = cache.generic_key(self) if key is None else key
        return cache.cache_file(key, ext='npz', subdir='recog')

    def run(self, key=None):
        key = cache.generic_key(self) if key is None else key
        if cache.cache_file_exists(key, ext='npz', subdir='recog'):
            log("%s.npz in the cache. Loading." % key)
            result = RecognitionResult.load(key)
        else:
            log("%s.npz not in the cache. Running." % key)
            result = self.run_model()
            result.save(key)
            log("Experiment run saved to the cache.")
        return key


class RecognitionResult(ExperimentResult):

    saved = ['seq',
             'freqs',
             'dmps',
             'dmp_labels',
             'memory',
             'class_time',
             'class_labels',
             'acc',
             'n_sub',
             'n_del',
             'n_ins',
             'tdiff_mean',
             'tdiff_var',
             'memory_acc']
    to_int = ['n_sub',
              'n_del',
              'n_ins']
    to_float = ['acc',
                'tdiff_mean',
                'tdiff_var',
                'memory_acc']
    subdir = "recog"
