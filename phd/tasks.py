import numpy as np

from . import sermo
from .experiments import (
    AuditoryFeaturesExperiment, ProductionExperiment, RecognitionExperiment)
from .timit import TIMIT


class ExperimentTask(object):

    params = []

    def __init__(self, n_iters, **kwargs):
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
                name = self.fullname(experiment, seed)
                yield {'name': name,
                       'actions': [(experiment.run, [name])],
                       'file_dep': [__file__, sermo.__file__],
                       'targets': [experiment.cache_file(name)]}

    def __iter__(self):
        """Should yield experiment instances."""
        raise NotImplementedError()

    def name(self, experiment):
        """Should return a name based on the experiment."""
        return None


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

af_iters = 10
af_phones = [TIMIT.consonants]


def phone_str(experiment):
    if experiment.phones is TIMIT.consonants:
        s = "consonants"
    elif experiment.phones is TIMIT.vowels:
        s = "vowels"
    elif experiment.phones is TIMIT.phones:
        s = "all"
    else:
        s = repr(experiment.phones)
    return "phones:%s" % s


class AFZscoreTask(ExperimentTask):

    params = ['zscore', 'phones']

    def __iter__(self):
        for zscore in self.zscore:
            for phones in self.phones:
                model = sermo.AuditoryFeatures()
                model.add_derivative()
                expt = AuditoryFeaturesExperiment(
                    model, phones=phones, zscore=zscore)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        return "zscore:%s,%s" % (
            experiment.zscore, phone_str(experiment))

task_af_zscore = lambda: AFZscoreTask(
    zscore=[False, True], phones=af_phones, n_iters=af_iters)()


class AFDerivativesTask(ExperimentTask):

    params = ['n_derivatives', 'phones']

    def __iter__(self):
        for n_derivatives in self.n_derivatives:
            for phones in self.phones:
                model = sermo.AuditoryFeatures()
                model.add_derivative()
                for _ in range(n_derivatives):
                    model.add_derivative()
                expt = AuditoryFeaturesExperiment(model, phones=phones)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        return "derivatives:%d,%s" % (
            experiment.model.mfcc.n_derivatives, phone_str(experiment))

task_af_derivatives = lambda: AFDerivativesTask(
    n_derivatives=[0, 1, 2], phones=af_phones, n_iters=af_iters)()


class AFDerivTypeTask(ExperimentTask):

    params = ['deriv_type', 'phones']

    def __iter__(self):
        for deriv_type in self.deriv_type:
            for phones in self.phones:
                model = sermo.AuditoryFeatures()
                model.add_derivative(klass=deriv_type)
                expt = AuditoryFeaturesExperiment(model, phones=phones)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        derivtype = experiment.model.derivatives[0].__class__.__name__[:-6]
        return "derivtype:%s,%s" % (derivtype, phone_str(experiment))

task_af_derivtype = lambda: AFDerivTypeTask(
    deriv_type=['FeedforwardDeriv', 'IntermediateDeriv'],
    phones=af_phones, n_iters=af_iters)()


class AFPeripheryNeuronsTask(ExperimentTask):

    params = ['n_neurons', 'phones']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            for phones in self.phones:
                model = sermo.AuditoryFeatures()
                # Set features to 20 neurons to isolate periphery effect
                model.add_derivative(n_neurons=20)
                model.cepstra.n_neurons = 20
                model.periphery.neurons_per_freq = n_neurons
                expt = AuditoryFeaturesExperiment(model, phones=phones)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        return "periphery:%d,%s" % (
            experiment.model.periphery.neurons_per_freq, phone_str(experiment))

task_af_periphery_neurons = lambda: AFPeripheryNeuronsTask(
    n_neurons=[1, 2, 4, 8, 16, 32], phones=af_phones, n_iters=af_iters)()


class AFFeatureNeuronsTask(ExperimentTask):

    params = ['n_neurons', 'phones']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            for phones in self.phones:
                model = sermo.AuditoryFeatures()
                # Set periphery to 8 neurons to isolate feature effect
                model.periphery.neurons_per_freq = 8
                model.add_derivative(n_neurons=n_neurons)
                model.cepstra.n_neurons = n_neurons
                expt = AuditoryFeaturesExperiment(model, phones=phones)
                expt.timit.filefilt.region = 8
                yield expt

    def name(self, experiment):
        return "feature:%d,%s" % (
            experiment.model.cepstra.n_neurons, phone_str(experiment))

task_af_feature_neurons = lambda: AFFeatureNeuronsTask(
    n_neurons=[1, 2, 4, 8, 12, 16, 32, 64],
    phones=af_phones,
    n_iters=af_iters)()


class AFPhonesTask(ExperimentTask):

    params = ['phones']

    def __iter__(self):
        for phones in self.phones:
            model = sermo.AuditoryFeatures()
            model.add_derivative()
            expt = AuditoryFeaturesExperiment(model, phones=phones)
            expt.timit.filefilt.region = 8
            yield expt

    def name(self, experiment):
        return phone_str(experiment)

task_af_phones = lambda: AFPhonesTask(
    phones=[TIMIT.consonants, TIMIT.vowels, TIMIT.phones], n_iters=af_iters)()


class AFTimeWindowTask(ExperimentTask):

    params = ['dts', 'phones']

    def __iter__(self):
        for dt in self.dts:
            for phones in self.phones:
                model = sermo.AuditoryFeatures()
                model.add_derivative()
                model.mfcc.dt = dt
                expt = AuditoryFeaturesExperiment(model, phones=phones)
                expt.timit.filefilt.region = 8
                yield expt

    def __call__(self):
        """Generate a set of `n_iters` tasks for the given model."""
        for task in super(AFTimeWindowTask, self).__call__():
            # Overwrite the action to include n_frames=35
            #  consonants: 22 frames; vowels: 35 frames
            task['actions'][0][1].append(35)
            yield task

    def name(self, experiment):
        return "dt:%f,%s" % (experiment.model.mfcc.dt, phone_str(experiment))

task_af_timewindow = lambda: AFTimeWindowTask(
    dts=[0.001, 0.005, 0.01], phones=af_phones, n_iters=af_iters)()


class AFPeripheryTask(ExperimentTask):

    params = ['auditory_filters', 'adaptive_neurons', 'phones']

    def __iter__(self):
        for auditory_filter in self.auditory_filters:
            for adaptive_neurons in self.adaptive_neurons:
                for phones in self.phones:
                    model = sermo.AuditoryFeatures()
                    model.add_derivative()
                    model.periphery.auditory_filter = auditory_filter
                    model.periphery.adaptive_neurons = adaptive_neurons
                    expt = AuditoryFeaturesExperiment(
                        model, phones=phones, upsample=True)
                    expt.timit.filefilt.region = 8
                    yield expt

    def name(self, experiment):
        return "periphmodel:%s,adaptive:%s,%s" % (
            experiment.model.periphery.auditory_filter,
            experiment.model.periphery.adaptive_neurons,
            phone_str(experiment))

task_af_periphery = lambda: AFPeripheryTask(
    auditory_filters=['gammatone',
                      'log_gammachirp',
                      'dual_resonance',
                      'compressive_gammachirp',
                      'tan_carney'],
    adaptive_neurons=[False, True],
    phones=af_phones,
    n_iters=af_iters)()


# ############################
# Model 2: Syllable production
# ############################

prod_n_iters = 15


class ProdSyllableNeuronsTask(ExperimentTask):

    params = ['n_neurons']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.Production()
            model.syllable.n_per_d = n_neurons
            expt = ProductionExperiment(model, n_syllables=2, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "syllneurons:%d" % (experiment.model.syllable.n_per_d)

task_prod_syllneurons = lambda: ProdSyllableNeuronsTask(
    n_neurons=[30, 40, 50, 70, 90, 120, 150, 180, 250, 400, 600],
    n_iters=prod_n_iters)()


class ProdSyllableTauTask(ExperimentTask):

    params = ['tau']

    def __iter__(self):
        for tau in self.tau:
            model = sermo.Production()
            model.syllable.tau = tau
            expt = ProductionExperiment(model, n_syllables=2, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "tau:%.3f" % (experiment.model.syllable.tau)

task_prod_sylltau = lambda: ProdSyllableTauTask(
    tau=[0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    n_iters=prod_n_iters)()


class ProdSequencerNeuronsTask(ExperimentTask):

    params = ['n_neurons']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.Production()
            model.sequencer.n_per_d = n_neurons
            expt = ProductionExperiment(model, n_syllables=2, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "seqneurons:%d" % (experiment.model.sequencer.n_per_d)

task_prod_seqneurons = lambda: ProdSequencerNeuronsTask(
    n_neurons=[10, 20, 30, 50, 90, 180, 250, 400], n_iters=prod_n_iters)()


class ProdFreqTask(ExperimentTask):

    params = ['freqs']

    def __iter__(self):
        for freq in self.freqs:
            model = sermo.Production()
            expt = ProductionExperiment(model, minfreq=freq, maxfreq=freq,
                                        n_syllables=2, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "freq:%.2f" % (experiment.minfreq)

task_prod_freqs = lambda: ProdFreqTask(
    freqs=np.arange(1.6, 4.1, 0.4), n_iters=prod_n_iters)()


class ProdNSyllablesTask(ExperimentTask):

    params = ['n_syllables']

    def __iter__(self):
        for n_syllables in self.n_syllables:
            model = sermo.Production()
            # Also up syllable_d
            expt = ProductionExperiment(
                model, n_syllables=n_syllables, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "n_syllables:%d" % (experiment.n_syllables)

task_prod_n_syllables = lambda: ProdNSyllablesTask(
    n_syllables=list(range(9)), n_iters=prod_n_iters)()


class ProdSequenceLenTask(ExperimentTask):

    params = ['sequence_len']

    def __iter__(self):
        for sequence_len in self.sequence_len:
            model = sermo.Production()
            expt = ProductionExperiment(
                model, n_syllables=2, sequence_len=sequence_len)
            yield expt

    def name(self, experiment):
        return "sequence_len:%d" % (experiment.sequence_len)

task_prod_sequence_len = lambda: ProdSequenceLenTask(
    sequence_len=np.arange(3, 10), n_iters=prod_n_iters)()


class ProdRepeatTask(ExperimentTask):

    params = ['repeat']

    def __iter__(self):
        for repeat in self.repeat:
            model = sermo.Production()
            model.trial.repeat = repeat
            expt = ProductionExperiment(model, n_syllables=3, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "repeat:%s" % (experiment.model.trial.repeat)

task_prod_repeat = lambda: ProdRepeatTask(
    repeat=[False, True], n_iters=prod_n_iters)()


# #############################
# Model 3: Syllable recognition
# #############################

recog_n_iters = 20


class RecogSimilarityTask(ExperimentTask):

    params = ['similarity_th']

    def __iter__(self):
        for similarity_th in self.similarity_th:
            model = sermo.Recognition()
            model.syllable.similarity_th = similarity_th
            expt = RecognitionExperiment(model,
                                         n_syllables=3,
                                         sequence_len=3)
            yield expt

    def name(self, experiment):
        return "similarity:%.3f" % (experiment.model.syllable.similarity_th)

task_recog_similarity = lambda: RecogSimilarityTask(
    similarity_th=np.arange(0.65, 0.91, 0.01), n_iters=recog_n_iters)()


class RecogScaleTask(ExperimentTask):

    params = ['scale']

    def __iter__(self):
        for scale in self.scale:
            model = sermo.Recognition()
            model.syllable.scale = scale
            expt = RecognitionExperiment(model,
                                         n_syllables=3,
                                         sequence_len=3)
            yield expt

    def name(self, experiment):
        return "scale:%.3f" % (experiment.model.syllable.scale)

task_recog_scale = lambda: RecogScaleTask(
    scale=[0.5, 0.6, 0.64, 0.66, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73,
           0.74, 0.75, 0.76, 0.78, 0.8, 0.84, 0.9, 1.0],
    n_iters=recog_n_iters)()


class RecogSyllableNeuronsTask(ExperimentTask):

    params = ['n_neurons']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.Recognition()
            model.syllable.n_per_d = n_neurons
            expt = RecognitionExperiment(model,
                                         n_syllables=3,
                                         sequence_len=3)
            yield expt

    def name(self, experiment):
        return "syllneurons:%d" % (experiment.model.syllable.n_per_d)

task_recog_syllneurons = lambda: RecogSyllableNeuronsTask(
    n_neurons=[200, 250, 300, 350, 400, 450, 500,
               550, 600, 700, 800, 900, 1000],
    n_iters=recog_n_iters)()


class RecogFreqTask(ExperimentTask):

    params = ['freqs']

    def __iter__(self):
        for freq in self.freqs:
            model = sermo.Recognition()
            expt = RecognitionExperiment(model, minfreq=freq, maxfreq=freq,
                                         n_syllables=3, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "freq:%.2f" % (experiment.minfreq)

task_recog_freqs = lambda: RecogFreqTask(
    freqs=np.arange(0.6, 5.1, 0.4), n_iters=recog_n_iters)()


class RecogNSyllablesTask(ExperimentTask):

    params = ['n_syllables']

    def __iter__(self):
        for n_syllables in self.n_syllables:
            model = sermo.Recognition()
            expt = RecognitionExperiment(
                model, n_syllables=n_syllables, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "n_syllables:%d" % (experiment.n_syllables)

task_recog_n_syllables = lambda: RecogNSyllablesTask(
    n_syllables=np.arange(5, 25, 5), n_iters=recog_n_iters)()


class RecogSequenceLenTask(ExperimentTask):

    params = ['sequence_len']

    def __iter__(self):
        for sequence_len in self.sequence_len:
            model = sermo.Recognition()
            expt = RecognitionExperiment(
                model, n_syllables=3, sequence_len=sequence_len)
            yield expt

    def name(self, experiment):
        return "sequence_len:%d" % (experiment.sequence_len)

task_recog_sequence_len = lambda: RecogSequenceLenTask(
    sequence_len=np.arange(3, 10), n_iters=recog_n_iters)()


class RecogRepeatTask(ExperimentTask):

    params = ['repeat']

    def __iter__(self):
        for repeat in self.repeat:
            model = sermo.Recognition()
            model.trial.repeat = repeat
            expt = RecognitionExperiment(model, n_syllables=3, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "repeat:%s" % (experiment.model.trial.repeat)

task_recog_repeat = lambda: RecogRepeatTask(
    repeat=[False, True], n_iters=recog_n_iters)()
