from __future__ import print_function

import subprocess
import sys
import warnings

import nengo
import nengo.utils.numpy as npext
import numpy as np
import scipy
from multiprocess import Pool  # Uses dill instead of pickle
from nengo.cache import NoDecoderCache
from nengo.utils.compat import iteritems, range
from nengo.utils.stdlib import Timer
from sklearn.svm import LinearSVC
from scipy.interpolate import interp1d
from scipy import stats

from . import cache, config, filters, sermo
from .timit import TIMIT


def log(msg):
    if config.log_experiments:
        print(msg)
        sys.stdout.flush()


class ExperimentResult(object):

    saved = []
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
        d = cache.load_obj(key, ext='npz', subdir=cls.subdir)
        return cls(**data)

    def save(self, key):
        cache.cache_obj(self.save_dict(),
                        key=key, ext='npz', subdir=self.subdir)


class ExperimentTask(object):

    params = []

    def __init__(self, n_iters=5, **kwargs):
        self.n_iters = n_iters

        for key in self.params:
            setattr(self, key, kwargs.pop(key, None))
        # kwargs should be empty now
        for key in kwargs:
            raise TypeError("got an unexpected keyword argument '%s'" % key)


    def fullname(self, experiment, seed):
        ename = self.name(experiment)
        sname = "seed:%d" % seed
        return sname if ename is None else "%s,%s" % (ename, sname)

    def __call__(self):
        """Generate a set of `n_iters` tasks for the given model."""
        for seed in range(self.n_iters):
            for experiment in self:
                experiment.seed = seed
                yield {'name': self.fullname(experiment, seed),
                       'actions': [experiment.run],
                       'targets': [experiment.cache_file]}

    def __iter__(self):
        """Should yield experiment instances."""
        raise NotImplementedError()

    def name(self, experiment):
        """Should return a name based on the experiment."""
        return None


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

def mfcc(model, sample, zscore):
    model.audio = sample
    feat = model.mfcc()
    if zscore and feat.shape[0] > 2:
        # TRY: take the zscore for each block separately
        #      where block is normal, deriv1, deriv2, etc
        feat = stats.zscore(feat, axis=0)
    return feat


def mfccs(model, audio, zscore):
    out = {label: [] for label in audio}
    for label in audio:
        for sample in audio[label]:
            out[label].append(mfcc(model, sample, zscore))
    model.audio = np.zeros(1)
    return out


def ncc(model, sample, zscore, seed):
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
    return feat


def nccs(model, audio,zscore, seed, parallel=True):
    out = {label: [] for label in audio}
    if parallel:
        jobs = {label: [] for label in audio}
        pool = Pool(processes=config.n_processes)

    for label in audio:
        for sample in audio[label]:
            if parallel:
                jobs[label].append(
                    pool.apply_async(ncc, [model, sample, seed, zscore]))
            else:
                out[label].append(ncc(model, sample, seed, zscore))

    if parallel:
        for label in jobs:
            for result in jobs[label]:
                out[label].append(result.get())
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
    scale = int(feature.shape[0] / n_frames)
    pad_size = int(np.ceil(float(feature.shape[0]) / scale) * scale
                   - feature.shape[0])
    feature = np.vstack([feature,
                         np.zeros((pad_size, feature.shape[1])) * np.nan])
    return scipy.nanmean(
        feature.reshape(-1, scale, feature.shape[1]), axis=1)[:n_frames]


def lengthen(feature, n_frames):
    """Use linear interpolation to lengthen the feature vector."""
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
                 zscore=False, seed=None):
        self.model = model
        assert phones is None or words is None, "Can only set one, not both"
        self.phones = phones
        self.words = words
        self.seed = np.random.randint(npext.maxint) if seed is None else seed
        self.zscore = zscore
        self.timit = TIMIT(config.timit_root)

    def _get_audio(self, corpus):
        if self.phones is not None:
            return self.timit.phn_samples(self.phones, corpus=corpus)
        elif self.words is not None:
            return self.timit.word_samples(self.words, corpus=corpus)

    def _get_feature(self, feature, audio, n_frames=None):
        labels = sorted(list(audio))
        with Timer() as t:
            if feature == 'mfcc':
                x = mfccs(self.model, audio, self.zscore)
            elif feature == 'ncc':
                x = nccs(self.model, audio, self.zscore, self.seed)
            else:
                raise ValueError("Possible features: 'mfcc', 'ncc'")
        log("%ss generated in %.3f seconds" % (feature.upper(), t.duration))

        if n_frames is None:
            n_frames = max(max(xx.shape[0] for xx in x[l]) for l in audio)
        x = normalize(x, n_frames)
        return np.vstack([np.vstack(x[l]) for l in labels])

    @staticmethod
    def _get_labels(audio):
        labels = sorted(list(audio))
        return np.array([l for l in labels for _ in range(len(audio[l]))])

    def train(self, result):

        def fit_svm(x, y, feature):
            svm = LinearSVC(random_state=self.seed)
            with Timer() as t:
                svm.fit(x, y)
            log("SVM fitting for %ss done in %.3f seconds"
                % (feature.upper(), t.duration))
            setattr(result, "%s_time" % feature, t.duration)
            _, _, acc = test_svm(svm, x, y, "Training")
            setattr(result, "%s_train_acc", acc)
            return svm

        audio = self._get_audio(corpus="train")

        # NB! Do MFCC first to get n_frames for NCC.
        x_mfcc = self._get_feature('mfcc', audio)
        n_frames = int(x_mfcc.shape[1] // self.model.dimensions)
        y = self._get_labels(audio)
        mfcc_svm = fit_svm(x_mfcc, y, 'mfcc')

        x_ncc = self._get_feature('ncc', audio, n_frames=n_frames)
        ncc_svm = fit_svm(x_ncc, y, 'ncc')
        return mfcc_svm, ncc_svm

    def test(self, result, svm, feature):
        audio = self._get_audio(corpus="test")
        n_frames = int(svm.coef_.shape[1] // self.model.dimensions)
        x = self._get_feature(feature, audio, n_frames=n_frames)
        y = self._get_labels(audio)
        pred, y, acc = test_svm(svm, x, y, "Testing")
        setattr(result, "%s_pred" % feature, pred)

    @property
    def cache_key(self):
        return cache.generic_key(self)

    @property
    def cache_file(self):
        return cache.cache_file(self.cache_key, ext='npz')

    def run(self):
        if cache.cache_file_exists(self.cache_key, ext='npz'):
            log("%s.npz in the cache. Loading." % key)
            result = AuditoryFeaturesResult.load(key)
        else:
            result = AuditoryFeaturesResult()
            log("==== Training ====")
            mfcc_svm, ncc_svm = self.train(result)
            log("==== Testing ====")
            self.test(result, mfcc_svm, 'mfcc')
            self.test(result, ncc_svm, 'ncc')
            result.save(self.cache_key)
            log("Experiment run saved to the cache.")
        return result


class AuditoryFeaturesResult(ExperimentResult):

    saved = ['fit_time',
             'mfcc_time',
             'ncc_time',
             'mfcc_train_acc',
             'ncc_train_acc',
             'mfcc_pred',
             'ncc_pred',
             'y']
    subdir = "ncc"

    @property
    def mfcc_acc(self):
        return np.mean(self.mfcc_pred == self.y)

    @property
    def ncc_acc(self):
        return np.mean(self.ncc_pred == self.y)


class AFNNeuronsTask(ExperimentTask):

    params = ['n_neurons', 'phones']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.AuditoryFeatures()
            model.periphery.neurons_per_freq = n_neurons
            expt = AuditoryFeaturesExperiment(model, phones=self.phones)
            expt.timit.filefilt.region = 8
            yield expt

    def name(self, experiment):
        return "periphery:%d" % experiment.model.periphery.neurons_per_freq

task_af_n_neurons = lambda: AFNNeuronsTask(
    n_neurons=[5, 10, 20, 40], phones=TIMIT.phones)()


class AFPhonesTask(ExperimentTask):

    params = ['phones']

    def __iter__(self):
        for phones in self.phones:
            model = sermo.AuditoryFeatures()
            expt = AuditoryFeaturesExperiment(model, phones=phones)
            expt.timit.filefilt.region = 8
            yield expt

    def name(self, experiment):
        if experiment.phones is TIMIT.consonants:
            phones = "consonants"
        elif experiment.phones is TIMIT.vowels:
            phones = "vowels"
        elif experiment.phones is TIMIT.phones:
            phones = "all"
        else:
            phones = repr(experiment.phones)
        return "phones:%s" % phones

task_af_phones = lambda: AFPhonesTask(
    phones=[TIMIT.consonants, TIMIT.vowels, TIMIT.phones])()


class AFPostprocessingTask(ExperimentTask):

    params = ['n_derivatives', 'zscores', 'phones']

    def __iter__(self):
        for n_derivatives in self.n_derivatives:
            for zscore in self.zscores:
                model = sermo.AuditoryFeatures()
                for _ in range(n_derivatives):
                    model.add_derivative()
                expt = AuditoryFeaturesExperiment(
                    model, phones=self.phones, zscore=zscore)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        return "derivatives:%d,zscore:%s" % (
            experiment.model.mfcc.n_derivatives, experiment.zscore)


task_af_postprocessing = lambda: AFPostprocessingTask(
    n_derivatives=[1, 2], zscores=[False, True], phones=TIMIT.phones)()


class AFPeripheryTask(ExperimentTask):

    params = ['auditory_filters', 'adaptive_neurons', 'phones']

    def __iter__(self):
        for auditory_filter in self.auditory_filters:
            for adaptive_neurons in self.adaptive_neurons:
                model = sermo.AuditoryFeatures()
                model.periphery.auditory_filter = auditory_filter
                model.periphery.adaptive_neurons = adaptive_neurons
                expt = AuditoryFeaturesExperiment(model, phones=self.phones)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        return "%s,adaptive:%s" % (experiment.model.periphery.auditory_filter,
                                   experiment.model.periphery.adaptive_neurons)

task_af_periphery = lambda: AFPeripheryTask(
    auditory_filters=[# 'gammatone',  # alredy done
                      'approximate_gammatone',
                      'log_gammachirp',
                      'linear_gammachirp',
                      'tan_carney',
                      'dual_resonance',
                      'compressive_gammachirp'],
    adaptive_neurons=[False, True],
    phones=TIMIT.phones)()


class AFTimeWindowTask(ExperimentTask):

    params = ['dts', 'phones']

    def __iter__(self):
        for dt in self.dts:
            model = sermo.AuditoryFeatures()
            model.mfcc.dt = dt
            expt = AuditoryFeaturesExperiment(model, phones=self.phones)
            expt.timit.filefilt.region = 8
            yield expt

    def name(self, experiment):
        return "dt=%f" % experiment.model.mfcc.dt

task_af_timewindow = lambda: AFTimeWindowTask(
    dts=[0.001, 0.005, 0.02], phones=TIMIT.phones)()


# ############################
# Model 2: Syllable production
# ############################

class ProductionExperiment(object):
    def __init__(self):
        pass


# #############################
# Model 3: Syllable recognition
# #############################

class RecognitionExpermient(object):
    def __init__(self):
        pass
