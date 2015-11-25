from __future__ import print_function

import sys
import warnings

import ipyparallel as ipp
import nengo
import nengo.utils.numpy as npext
import numpy as np
import scipy
from nengo.utils.compat import range
from nengo.utils.stdlib import Timer
from sklearn.svm import LinearSVC
from scipy.interpolate import interp1d
from scipy import stats

from . import cache, config
from .timit import TIMIT


def log(msg):
    if config.log_experiments:
        print(msg)
        sys.stdout.flush()


class AuditoryFeaturesExperiment(object):
    def __init__(self, model, phonemes=None, words=None,
                 zscore=False, seed=None):
        self.model = model
        assert phonemes is None or words is None, "Can only set one, not both"
        self.phonemes = phonemes
        self.words = words
        self.seed = np.random.randint(npext.maxint) if seed is None else seed
        self.zscore = zscore
        self.timit = TIMIT(config.timit_root)

    def mfccs(self, audio):
        out = {label: [] for label in audio}
        for label in audio:
            for sample in audio[label]:
                self.model.audio = sample
                feat = self.model.mfcc()
                if self.zscore and feat.shape[0] > 2:
                    # TRY: take the zscore for each block separately
                    #      where block is normal, deriv1, deriv2, etc
                    feat = stats.zscore(feat, axis=0)
                out[label].append(feat)
        self.model.audio = np.zeros(1)
        return out

    def nccs(self, audio):
        out = {label: [] for label in audio}
        for label in audio:
            for sample in audio[label]:
                self.model.audio = sample
                net = self.model.build(nengo.Network(seed=self.seed))
                with net:
                    pr = nengo.Probe(net.output, synapse=0.01)
                sim = nengo.Simulator(net, dt=0.001)
                sim.run(self.model.t_audio, progress_bar=False)
                if self.zscore:
                    # TRY: take the zscore for each block separately
                    #      where block is normal, deriv1, deriv2, etc
                    out[label].append(stats.zscore(sim.data[pr], axis=0))
                else:
                    out[label].append(sim.data[pr])
        self.model.audio = np.zeros(1)
        return out

    def parallel_nccs(self, audio):
        jobs = {label: [] for label in audio}
        out = {label: [] for label in audio}
        ipclient = ipp.Client()
        ipclient[:].use_dill()
        lview = ipclient.load_balanced_view()

        # --- Start all jobs
        for label in audio:
            for sample in audio[label]:
                jobs[label].append(lview.apply_async(self._ncc,
                                                     self.model,
                                                     sample,
                                                     self.seed))
        # --- Get the results as they come
        for label in jobs:
            for result in jobs[label]:
                out[label].append(result.get())

        return out

    @staticmethod
    def _ncc(model, sample, seed):
        model.audio = sample
        net = model.build(nengo.Network(seed=seed))
        with net:
            pr = nengo.Probe(net.output, synapse=0.01)
        sim = nengo.Simulator(net, dt=0.001)
        sim.run(model.t_audio, progress_bar=False)
        model.audio = np.zeros(1)
        return sim.data[pr]


    @classmethod
    def normalize(cls, features, n_frames):
        out = {label: [] for label in features}
        for label in features:
            for sample in features[label]:
                if sample.shape[0] < n_frames:
                    sample = cls.lengthen(sample, n_frames)
                elif sample.shape[0] > n_frames:
                    sample = cls.shorten(sample, n_frames)
                # Flatten the vector so that SVM can handle it
                out[label].append(sample.reshape(-1))
        return out

    @staticmethod
    def shorten(feature, n_frames):
        """Compute neighbourhood mean to shorten the feature vector."""
        scale = int(feature.shape[0] / n_frames)
        pad_size = int(np.ceil(float(feature.shape[0]) / scale) * scale
                       - feature.shape[0])
        feature = np.vstack([feature,
                             np.zeros((pad_size, feature.shape[1])) * np.nan])
        return scipy.nanmean(
            feature.reshape(-1, scale, feature.shape[1]), axis=1)[:n_frames]

    @staticmethod
    def lengthen(feature, n_frames):
        """Use linear interpolation to lengthen the feature vector."""
        if feature.shape[0] == 1:
            feature = np.tile(feature, (2, 1))
        interp_x = np.linspace(0, n_frames, feature.shape[0])
        f = interp1d(interp_x, feature, axis=0, assume_sorted=True)
        return f(np.arange(n_frames))

    def _get_audio(self, corpus):
        if self.phonemes is not None:
            return self.timit.phn_samples(self.phonemes, corpus=corpus)
        elif self.words is not None:
            return self.timit.word_samples(self.words, corpus=corpus)

    def _get_feature(self, feature, audio, n_frames=None):
        labels = sorted(list(audio))
        with Timer() as t:
            if feature == 'mfcc':
                x = self.mfccs(audio)
            elif feature == 'ncc':
                try:
                    x = self.parallel_nccs(audio)
                except IOError:
                    warnings.warn("IPython cluster not running; running in "
                                  "serial. This may be very slow! Stop and "
                                  "start an IPython cluster to rectify.")
                    x = self.nccs(audio)
                except:
                    raise
            else:
                raise ValueError("Possible features: 'mfcc', 'ncc'")
        log("%ss generated in %.3f seconds" % (feature.upper(), t.duration))

        if n_frames is None:
            n_frames = max(max(xx.shape[0] for xx in x[l]) for l in audio)
        x = self.normalize(x, n_frames)
        return np.vstack([np.vstack(x[l]) for l in labels])

    @staticmethod
    def _get_labels(audio):
        labels = sorted(list(audio))
        return np.array([l for l in labels for _ in range(len(audio[l]))])

    def _test(self, svm, x, y, data="Testing"):
        pred_y = svm.predict(x)
        acc = np.mean(pred_y == y)
        log("%s accuracy: %.4f" % (data, acc))
        return pred_y, y, acc

    def train(self):

        def fit_svm(x, y, feature):
            svm = LinearSVC()
            with Timer() as t:
                svm.fit(x, y)
            log("SVM fitting for %ss done in %.3f seconds"
                % (feature.upper(), t.duration))
            self._test(svm, x, y, "Training")
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

    def test(self, svm, feature):
        audio = self._get_audio(corpus="test")
        n_frames = int(svm.coef_.shape[1] // self.model.dimensions)
        x = self._get_feature(feature, audio, n_frames=n_frames)
        y = self._get_labels(audio)
        return self._test(svm, x, y, "Testing")

    def run(self):
        key = cache.generic_key(self)
        if cache.cache_file_exists(key, ext='npz'):
            log("'%s.npz' in the cache. Loading." % key)
            result = AuditoryFeaturesResult.load(key)
        else:
            log("==== Training ====")
            mfcc_svm, ncc_svm = self.train()
            log("==== Testing ====")
            mfcc_pred, y, mfcc_acc = self.test(mfcc_svm, 'mfcc')
            ncc_pred, y, ncc_acc = self.test(ncc_svm, 'ncc')
            result = AuditoryFeaturesResult(mfcc_pred, ncc_pred, y)
            result.save(key)
            log("Experiment run saved to the cache.")
        return result


class AuditoryFeaturesResult(object):
    def __init__(self, mfcc_pred, ncc_pred, y):
        self.mfcc_pred = mfcc_pred
        self.ncc_pred = ncc_pred
        self.y = y

    @property
    def mfcc_acc(self):
        return np.mean(self.mfcc_pred == self.y)

    @property
    def ncc_acc(self):
        return np.mean(self.ncc_pred == self.y)

    @classmethod
    def load(cls, key):
        path = cache.cache_file(key, ext='npz')
        with np.load(path) as data:
            out = cls(data['mfcc_pred'], data['ncc_pred'], data['y'])
        return out

    def save(self, key):
        path = cache.cache_file(key, ext='npz')
        np.savez(path,
                 mfcc_pred=self.mfcc_pred,
                 ncc_pred=self.ncc_pred,
                 y=self.y)
