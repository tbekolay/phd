import nengo
import numpy as np
import soundfile as sf
from nengo import spa
from nengo.dists import Choice, Exponential
from nengo.networks import EnsembleArray
from nengo.utils.compat import is_array, is_string, iteritems

from . import params
from .filters import melspace
from .mfcc import mfcc
from .networks import (  # noqa: F401
    AuditoryPeriphery,
    Cepstra,
    DeadzoneDMP,
    DMP,
    FeedforwardDeriv,
    IntermediateDeriv,
    InverseDMP,
    RhythmicDMP,
    Sequencer,
    SyllableSequence,
)
from .networks.dmp import traj2func
from .processes import ArrayProcess
from .timit import TIMIT


class ParamsObject(object):
    @property
    def net(self):
        name = self.__class__.__name__
        assert name.endswith("Params"), "Class improperly named"
        return globals()[name[:-len("Params")]]

    def kwargs(self):
        args = {}
        klass = self.__class__
        for attr in dir(klass):
            if params.is_param(getattr(klass, attr)):
                args[attr] = getattr(self, attr)
        return args

    def __setattr__(self, key, value):
        """Make sure our names are correct."""
        assert hasattr(self.__class__, key), "'%s' attr does not exist" % key
        super(ParamsObject, self).__setattr__(key, value)

    def __getstate__(self):
        """Needed for pickling."""
        # Use ParamsObject specifically in case subclass shadows it
        return ParamsObject.kwargs(self)

    def __setstate__(self, state):
        """Needed for pickling."""
        for attr, val in iteritems(state):
            setattr(self, attr, val)


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

class MFCCParams(ParamsObject):
    audio = params.NdarrayParam('audio', default=None, shape=('*', 1))
    fs = params.NumberParam('fs', default=TIMIT.fs)
    dt = params.NumberParam('dt', default=0.01)
    window_dt = params.NumberParam('window_dt', default=0.025)
    n_cepstra = params.IntParam('n_cepstra', default=13)
    n_filters = params.IntParam('n_filters', default=32)
    n_fft = params.IntParam('n_fft', default=512)
    minfreq = params.NumberParam('minfreq', default=200)
    maxfreq = params.NumberParam('maxfreq', default=8000)
    preemph = params.NumberParam('preemph', default=0)
    lift = params.NumberParam('lift', default=0)
    energy = params.BoolParam('energy', default=False)
    n_derivatives = params.IntParam('n_derivatives', default=0)
    deriv_spread = params.IntParam('deriv_spread', default=2)

    def __call__(self):
        return mfcc(**self.kwargs())


class PeripheryParams(ParamsObject):
    freqs = params.NdarrayParam(
        'freqs', default=melspace(200, 8000, 32), shape=('*',))
    sound_process = params.ProcessParam('sound_process', default=None)
    auditory_filter = params.StringParam(
        'auditory_filter', default='gammatone')
    neurons_per_freq = params.IntParam('neurons_per_freq', default=8)
    fs = params.NumberParam('fs', default=TIMIT.fs)
    adaptive_neurons = params.BoolParam('adaptive_neurons', default=False)


class CepstraParams(ParamsObject):
    n_neurons = params.IntParam('n_neurons', default=20)
    n_cepstra = params.IntParam('n_cepstra', default=13)


class FeedforwardDerivParams(ParamsObject):
    n_neurons = params.IntParam('n_neurons', default=20)
    tau_fast = params.NumberParam('tau_fast', default=0.05)
    tau_slow = params.NumberParam('tau_slow', default=0.1)


class IntermediateDerivParams(ParamsObject):
    n_neurons = params.IntParam('n_neurons', default=20)
    tau = params.NumberParam('tau', default=0.1)


class AuditoryFeatures(object):
    def __init__(self):
        self.config = nengo.Config(
            nengo.Ensemble, nengo.Connection, nengo.Probe)
        self.mfcc = MFCCParams()
        self.periphery = PeripheryParams()
        self.cepstra = CepstraParams()
        self.derivatives = []
        # Set dummy audio so that pickling doesn't fail
        self.audio = np.zeros(1)

    @property
    def audio(self):
        assert self.mfcc.audio is self.periphery.sound_process.array
        return self.mfcc.audio

    @audio.setter
    def audio(self, audio_):
        if is_string(audio_):
            # Assuming this is a wav file
            audio_, fs = sf.read(audio_)
            self.fs = fs
        assert is_array(audio_)
        if audio_.ndim == 1:
            audio_ = audio_[:, np.newaxis]
        self.mfcc.audio = audio_
        self.periphery.sound_process = ArrayProcess(audio_)

    @property
    def dimensions(self):
        return self.n_cepstra * (1 + len(self.derivatives))

    @property
    def fs(self):
        assert self.mfcc.fs == self.periphery.fs
        return self.mfcc.fs

    @fs.setter
    def fs(self, fs_):
        self.mfcc.fs = fs_
        self.periphery.fs = fs_

    @property
    def freqs(self):
        assert np.allclose(self.mfcc.minfreq, self.periphery.freqs[0])
        assert np.allclose(self.mfcc.maxfreq, self.periphery.freqs[-1])
        assert self.mfcc.n_filters == self.periphery.freqs.size
        return self.periphery.freqs

    @freqs.setter
    def freqs(self, freqs_):
        self.periphery.freqs = freqs_
        self.mfcc.minfreq = freqs_[0]
        self.mfcc.maxfreq = freqs_[-1]
        self.mfcc.n_filters = freqs_.size

    @property
    def n_cepstra(self):
        assert self.mfcc.n_cepstra == self.cepstra.n_cepstra
        return self.mfcc.n_cepstra

    @n_cepstra.setter
    def n_cepstra(self, n_cepstra_):
        self.mfcc.n_cepstra = n_cepstra_
        self.cepstra.n_cepstra = n_cepstra_

    @property
    def t_audio(self):
        return self.audio.size / float(self.fs)

    def add_derivative(self, klass="IntermediateDeriv", **kwargs):
        deriv = globals()["%sParams" % klass]()
        for k, v in iteritems(kwargs):
            setattr(deriv, k, v)
        self.derivatives.append(deriv)
        self.mfcc.n_derivatives += 1
        return deriv

    def build(self, net=None):
        if net is None:
            net = nengo.Network("Sermo feature extraction")
        with net, self.config:
            self.build_periphery(net)
            self.build_cepstra(net)
            if len(self.derivatives) > 0:
                self.build_derivatives(net)
        return net

    def build_periphery(self, net):
        net.periphery = AuditoryPeriphery(**self.periphery.kwargs())

    def build_cepstra(self, net):
        net.cepstra = Cepstra(n_freqs=net.periphery.freqs.size,
                              **self.cepstra.kwargs())
        nengo.Connection(net.periphery.an.output, net.cepstra.input)
        # If no derivatives, our output is the cepstral coefficients
        net.output = net.cepstra.output

    def build_derivatives(self, net):
        net.output = nengo.Node(size_in=self.dimensions)
        nengo.Connection(net.cepstra.output, net.output[:self.n_cepstra],
                         synapse=None)

        net.derivatives = []
        target = net.cepstra  # First, do the derivative of the cepstra
        for i, deriv in enumerate(self.derivatives):
            derivnet = deriv.net(dimensions=self.n_cepstra, **deriv.kwargs())
            nengo.Connection(target.output, derivnet.input, synapse=None)
            nengo.Connection(
                derivnet.output,
                net.output[(i+1)*self.n_cepstra:(i+2)*self.n_cepstra],
                synapse=None)
            target = derivnet  # Then do derivative of derivatives
            net.derivatives.append(derivnet)


# ############################
# Model 2: Syllable production
# ############################

class SyllableSequenceParams(ParamsObject):
    n_per_d = params.IntParam('n_per_d', default=30)
    syllable_d = params.IntParam('syllable_d', default=48)
    difference_gain = params.NumberParam('difference_gain', default=15)
    n_positions = params.NumberParam('n_positions', default=7)
    threshold_memories = params.BoolParam('threshold_memories', default=True)
    add_default_output = params.BoolParam('add_default_output', default=True)


class SequencerParams(ParamsObject):
    n_per_d = params.IntParam('n_per_d', default=400)
    timer_tau = params.NumberParam('timer_tau', default=0.05)
    timer_freq = params.NumberParam('timer_freq', default=1.)
    reset_time = params.NumberParam('reset_time', default=0.85)
    reset_threshold = params.NumberParam('reset_threshold', default=0.5)
    reset_to_gate = params.NumberParam('reset_to_gate', default=-0.65)
    gate_threshold = params.NumberParam('gate_threshold', default=0.4)


class ProdSyllableParams(ParamsObject):
    n_per_d = params.IntParam('n_per_d', default=600)
    tau = params.NumberParam('tau', default=0.02)


class ProdSyllable(ParamsObject):
    label = params.StringParam('label', default=None)
    freq = params.NumberParam('freq', default=1.)
    trajectory = params.NdarrayParam('trajectory', shape=('*', 48))


class ProductionInfoParams(ParamsObject):
    n_per_d = params.IntParam('n_per_d', default=15)
    threshold = params.NumberParam('threshold', default=0.3)


class ProductionTrialParams(ParamsObject):
    dt = params.NumberParam('dt', default=0.001)
    sequence = params.StringParam('sequence', default=None)
    t_release = params.NumberParam('t_release', default=0.14)
    repeat = params.BoolParam('repeat', default=True)


class Production(object):
    def __init__(self):
        self.config = nengo.Config(
            nengo.Ensemble, nengo.Connection, nengo.Probe)
        self.sequence = SyllableSequenceParams()
        self.sequencer = SequencerParams()
        self.syllable = ProdSyllableParams()
        self.syllables = []
        self.syllable_dict = {}
        self.production_info = ProductionInfoParams()
        self.trial = ProductionTrialParams()

    def add_syllable(self, **kwargs):
        syll = ProdSyllable()
        for k, v in iteritems(kwargs):
            setattr(syll, k, v)
        self.syllables.append(syll)
        self.syllable_dict[syll.label] = syll

    def build(self, net=None):
        if net is None:
            net = nengo.Network("Sermo syllable production")
        with net, self.config:
            self.build_sequence(net)
            self.build_sequencer(net)
            self.build_syllables(net)
            self.build_connections(net)
            self.build_trial(net)
        return net

    def build_syllables(self, net):
        assert len(self.syllables) > 0, "No syllables added"

        # Make a readout for the production info coming from the DMPs
        intercepts = Exponential(0.15, self.production_info.threshold, 1)
        net.production_info = EnsembleArray(self.production_info.n_per_d,
                                            n_ensembles=48,
                                            encoders=Choice([[1]]),
                                            intercepts=intercepts,
                                            radius=1.1)

        net.syllables = []
        dt = self.trial.dt
        for syllable in self.syllables:
            forcing_f, gesture_ix = traj2func(syllable.trajectory, dt=dt)
            forcing_f.__name__ = syllable.label
            dmp = RhythmicDMP(forcing_f=forcing_f, freq=syllable.freq,
                              **self.syllable.kwargs())
            nengo.Connection(dmp.output, net.production_info.input[gesture_ix])
            net.syllables.append(dmp)

    def build_sequence(self, net):
        syllables = [s.label for s in self.syllables]
        net.sequence = SyllableSequence(syllables=syllables,
                                        **self.sequence.kwargs())

    def build_sequencer(self, net):
        net.sequencer = Sequencer(**self.sequencer.kwargs())

    def build_connections(self, net):
        # Sequencer gate control sequence working memory gates
        nengo.Connection(net.sequencer.gate, net.sequence.gate)

        for i, dmp in enumerate(net.syllables):
            # Provide sequencer with timing info
            nengo.Connection(dmp.osc, net.sequencer.timer)
            # Reset DMPs when appropriate
            nengo.Connection(net.sequencer.reset, dmp.reset)
            # Disinhibit DMPs when syllable is current or next
            # FIXME: won't work for non-tresholded AMs
            curr_ens = net.sequence.syllable.am.thresh_ens.ea_ensembles[i]
            next_ens = net.sequence.syllable_next.am.thresh_ens.ea_ensembles[i]
            nengo.Connection(curr_ens, dmp.disinhibit)
            nengo.Connection(next_ens, dmp.disinhibit)

    def build_trial(self, net):
        # At the start of of the experiment...
        vocab = net.sequence.vocab

        # initial position is POS1,
        net.init_idx = nengo.Node(lambda t: vocab.parse('POS1').v
                                  if t < self.trial.t_release + 0.1
                                  else vocab.parse('0').v)
        nengo.Connection(net.init_idx, net.sequence.pos_curr.input)

        # the sequence is as given,
        net.seq_input = nengo.Node(vocab.parse(self.trial.sequence).v)
        nengo.Connection(net.seq_input, net.sequence.sequence.input)

        # the dmps are reset,
        net.init_reset = nengo.Node(
            lambda t: 1.0 if t < self.trial.t_release else 0.0)
        nengo.Connection(net.init_reset, net.sequencer.reset)

        # the output is suppressed,
        # nengo.Connection(net.init_reset,
        #                  net.production_info.input,
        #                  transform=-np.ones((48, 1)))

        # the timer is started.
        nengo.Connection(net.init_reset, net.sequencer.timer,
                         transform=[[-1], [0]])


# #############################
# Model 3: Syllable recognition
# #############################

class RecogSyllableParams(ParamsObject):
    n_per_d = params.IntParam('n_per_d', default=1000)
    similarity_th = params.NumberParam('similarity_th', default=0.82)
    scale = params.NumberParam('scale', default=0.71)
    reset_scale = params.NumberParam('reset_scale', default=2.5)
    tau = params.NumberParam('tau', default=0.05)


class RecogSyllable(ParamsObject):
    label = params.StringParam('label', default=None)
    trajectory = params.NdarrayParam('trajectory', shape=('*', 48))


class CleanupParams(ParamsObject):
    dimensions = params.IntParam('dimensions', default=64)
    threshold = params.NumberParam('threshold', default=0.9)
    wta_inhibit_scale = params.NumberParam('wta_inhibit_scale', default=3.0)

    def kwargs(self):
        args = super(CleanupParams, self).kwargs()
        del args['dimensions']
        return args


class MemoryParams(ParamsObject):
    neurons_per_dimension = params.IntParam(
        'neurons_per_dimension', default=60)
    feedback = params.NumberParam('feedback', default=0.82)


class ClassifierParams(ParamsObject):
    reset_th = params.NumberParam('reset_th', default=0.9)
    inhib_scale = params.NumberParam('inhib_scale', default=1)


class RecognitionTrialParams(ParamsObject):
    dt = params.NumberParam('dt', default=0.001)
    trajectory = params.NdarrayParam(
        'trajectory', default=None, shape=('*', 48))
    repeat = params.BoolParam('repeat', default=True)


class Recognition(object):
    def __init__(self):
        self.config = nengo.Config(
            nengo.Ensemble, nengo.Connection, nengo.Probe)
        self.syllable = RecogSyllableParams()
        self.syllables = []
        self.syllable_dict = {}
        self.cleanup = CleanupParams()
        self.memory = MemoryParams()
        self.classifier = ClassifierParams()
        self.trial = RecognitionTrialParams()

    def add_syllable(self, **kwargs):
        syll = RecogSyllable()
        for k, v in iteritems(kwargs):
            setattr(syll, k, v)
        self.syllables.append(syll)
        self.syllable_dict[syll.label] = syll

    def build(self, net=None):
        if net is None:
            net = nengo.Network("Sermo syllable recognition")
        with net, self.config:
            self.build_input(net)
            self.build_cleanup(net)
            self.build_syllables(net)
            self.build_classifier(net)
            self.build_connections(net)
        return net

    def build_input(self, net):
        assert self.trial.trajectory is not None, "Must define trajectory"
        net.trajectory = EnsembleArray(80, n_ensembles=48)
        # Sneaky: override the net.trajectory.input node output
        net.trajectory.input.output = ArrayProcess(self.trial.trajectory)
        net.trajectory.input.size_in = 0

    def build_syllables(self, net):
        assert len(self.syllables) > 0, "No syllables added"

        net.syllables = []
        dt = self.trial.dt
        for syllable in self.syllables:
            forcing_f, gesture_ix = traj2func(syllable.trajectory, dt=dt)
            forcing_f.__name__ = syllable.label
            dmp = InverseDMP(forcing_f=forcing_f, **self.syllable.kwargs())
            # Sensitive gestures: pass on to state
            nengo.Connection(net.trajectory.output[gesture_ix], dmp.input)
            # Non-sensitive gestures: reset the state
            n_gestures = net.trajectory.output.size_in
            not_gesture_ix = np.delete(np.arange(n_gestures, dtype=int),
                                       gesture_ix)
            nengo.Connection(net.trajectory.output[not_gesture_ix], dmp.reset,
                             transform=np.ones((1, not_gesture_ix.size)))
            net.syllables.append(dmp)

    def build_cleanup(self, net):
        net.vocab = spa.Vocabulary(dimensions=self.cleanup.dimensions)
        net.vocab.parse(" + ".join(s.label for s in self.syllables))
        net.cleanup = spa.AssociativeMemory(net.vocab,
                                            wta_output=True,
                                            threshold_output=True,
                                            **self.cleanup.kwargs())
        net.memory = spa.State(dimensions=self.cleanup.dimensions,
                               vocab=net.vocab, **self.memory.kwargs())
        nengo.Connection(net.cleanup.output, net.memory.input)

    def build_classifier(self, net):
        intercepts = Exponential(0.15, self.classifier.reset_th, 1.0)
        net.classifier = nengo.Ensemble(20, dimensions=1,
                                        encoders=Choice([[1]]),
                                        neuron_type=nengo.AdaptiveLIF(),
                                        intercepts=intercepts)

        for dmp in net.syllables:
            transform = -self.classifier.inhib_scale * np.ones(
                (dmp.state.n_neurons, net.classifier.n_neurons))
            nengo.Connection(net.classifier.neurons, dmp.state.neurons,
                             transform=transform, synapse=0.01)

    def build_connections(self, net):
        for i, dmp in enumerate(net.syllables):
            # Connect syllables to associative memory
            nengo.Connection(dmp.state[0], net.cleanup.am.am_ensembles[i])
            # Classifier is driven by the associative memory
            nengo.Connection(net.cleanup.am.thresh_ens.output[i],
                             net.classifier, synapse=0.01)
