from . import sermo
from .experiments import AuditoryFeaturesExperiment, ProductionExperiment
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
                    expt = AuditoryFeaturesExperiment(model, phones=phones)
                    expt.timit.filefilt.region = 8
                    yield expt

    def name(self, experiment):
        return "periphmodel:%s,adaptive:%s,%s" % (
            experiment.model.periphery.auditory_filter,
            experiment.model.periphery.adaptive_neurons,
            phone_str(experiment))

task_af_periphery = lambda: AFPeripheryTask(
    auditory_filters=['gammatone',
                      'approximate_gammatone',
                      'log_gammachirp',
                      'linear_gammachirp',
                      'tan_carney',
                      'dual_resonance',
                      'compressive_gammachirp'],
    adaptive_neurons=[False, True],
    phones=af_phones,
    n_iters=af_iters)()


# ############################
# Model 2: Syllable production
# ############################

prod_n_iters = 20

class ProdSyllableNeuronsTask(ExperimentTask):

    params = ['n_neurons']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.Production()
            model.syllable.n_per_d = n_neurons
            expt = ProductionExperiment(model, n_syllables=3, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "syllneurons:%d" % (experiment.model.syllable.n_per_d)

task_prod_syllneurons = lambda: ProdSyllableNeuronsTask(
    n_neurons=[60, 120, 180, 240], n_iters=prod_n_iters)()


class ProdSequencerNeuronsTask(ExperimentTask):

    params = ['n_neurons']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.Production()
            model.sequencer.n_per_d = n_neurons
            expt = ProductionExperiment(model, n_syllables=3, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "seqneurons:%d" % (experiment.model.sequencer.n_per_d)

task_prod_seqneurons = lambda: ProdSequencerNeuronsTask(
    n_neurons=[60, 120, 180, 240], n_iters=prod_n_iters)()


class ProdOutputNeuronsTask(ExperimentTask):

    params = ['n_neurons']

    def __iter__(self):
        for n_neurons in self.n_neurons:
            model = sermo.Production()
            model.production_info.n_per_d = n_neurons
            expt = ProductionExperiment(model, n_syllables=3, sequence_len=3)
            yield expt

    def name(self, experiment):
        return "outneurons:%d" % (experiment.model.production_info.n_per_d)

task_prod_outneurons = lambda: ProdOutputNeuronsTask(
    n_neurons=[30, 60, 90, 120], n_iters=prod_n_iters)()


# #############################
# Model 3: Syllable recognition
# #############################
