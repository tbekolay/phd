from __future__ import print_function

import sys

import ipyparallel as ipp
import nengo
import nengo.utils.numpy as npext
import numpy as np
import scipy
from nengo.utils.compat import range
from nengo.utils.stdlib import Timer
from sklearn.svm import LinearSVC
from scipy.interpolate import interp1d

from . import config
from .timit import TIMIT


class AuditoryFeaturesExperiment(object):
    def __init__(self, model, phonemes=None, words=None,
                 seed=None, quiet=False):
        self.model = model
        assert phonemes is None or words is None, "Can only set one, not both"
        self.phonemes = phonemes
        self.words = words
        self.seed = np.random.randint(npext.maxint) if seed is None else seed
        self.quiet = quiet
        self.timit = TIMIT(config.timit_root)

    def log(self, msg, **kwargs):
        if not self.quiet:
            print(msg, **kwargs)
            sys.stdout.flush()

    def mfccs(self, audio):
        out = {label: [] for label in audio}
        for label in audio:
            for sample in audio[label]:
                self.model.audio = sample
                out[label].append(self.model.mfcc())
        return out

    def nccs(self, audio):
        out = {label: [] for label in audio}
        for label in audio:
            for sample in audio[label]:
                self.model.audio = sample
                net = self.model.build(nengo.Network(seed=self.seed))
                with net:
                    pr = nengo.Probe(net.cepstra.output, synapse=0.01)
                sim = nengo.Simulator(net, dt=0.001)
                sim.run(self.model.t_audio, progress_bar=False)
                out[label].append(sim.data[pr])
        return out

    def parallel_nccs(self, audio):
        jobs = {label: [] for label in audio}
        out = {label: [] for label in audio}
        lview = ipp.Client().load_balanced_view()

        # --- Start all jobs
        for label in audio:
            for sample in audio[label]:
                jobs[label].append(
                    lview.apply_async(self._ncc, self.model, sample))

        # --- Get the results as they come
        for label in jobs:
            for result in jobs[label]:
                out[label].append(result.get())

        return out

    @staticmethod
    def _ncc(model, sample):
        model.audio = sample
        net = model.build(nengo.Network(seed=self.seed))
        with net:
            pr = nengo.Probe(net.cepstra.output, synapse=0.01)
        sim = nengo.Simulator(net, dt=0.001)
        sim.run(model.t_audio, progress_bar=False)
        return sim.data[pr]


    @classmethod
    def normalize(cls, features, n_frames):
        out = {label: [] for label in features}
        for label in features:
            for sample in features[label]:
                if sample.shape[0] <= 1:
                    # Too short -- ignore it
                    continue
                elif sample.shape[0] < n_frames:
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
        self.log("Generating %ss... " % feature.upper(), end="")
        with Timer() as t:
            if feature == 'mfcc':
                x = self.mfccs(audio)
            elif feature == 'ncc':
                x = self.parallel_nccs(audio)
            else:
                raise ValueError("Possible features: 'mfcc', 'ncc'")
        self.log("done in %s seconds" % t.duration)

        if n_frames is None:
            n_frames = max(max(xx.shape[0] for xx in x[l]) for l in audio)
        x = self.normalize(x, n_frames)
        return np.vstack([np.vstack(x[l]) for l in labels])

    @staticmethod
    def _get_labels(audio):
        labels = sorted(list(audio))
        return np.array([l for l in labels for _ in range(len(audio[l]))])

    def train(self):
        audio = self._get_audio(corpus="train")

        # NB! Do MFCC first to get n_frames for NCC.
        x_mfcc = self._get_feature('mfcc', audio)
        n_frames = int(x_mfcc.shape[1] // self.model.n_cepstra)
        x_ncc = self._get_feature('ncc', audio, n_frames=n_frames)
        y = self._get_labels(audio)

        def fit_svm(x, y, feature):
            svm = LinearSVC()
            self.log("Fitting SVM for %ss... " % feature.upper(), end="")
            with Timer() as t:
                svm.fit(x, y)
            self.log("done in %s seconds" % t.duration)
            self.log("%s training " % feature.upper(), end="")
            self._test(svm, x, y)

        mfcc_svm = fit_svm(x_mfcc, y, 'mfcc')
        ncc_svm = fit_svm(x_ncc, y, 'ncc')
        return mfcc_svm, ncc_svm

    def _test(self, svm, x, y):
        pred_y = svm.predict(x)
        acc = np.mean(pred_y == y)
        self.log("accuracy: %.4f" % acc)
        return pred_y, y, acc

    def test(self, svm, feature):
        audio = self._get_audio(corpus="test")
        n_frames = svm.coef_.shape[1]  # I think?
        x = self._get_feature(audio, n_frames=n_frames)
        y = self._get_labels(audio)
        self.log("%s testing " % feature.upper(), end="")
        return self._test(svm, x, y)

    def run(self):
        mfcc_svm, ncc_svm = self.train()
        accuracy = []
        for svm, feat in [(mfcc_svm, 'mfcc'), (ncc_svm, 'ncc')]:
            accuracy.append(self.test(svm, feat)[2])
        return accuracy
