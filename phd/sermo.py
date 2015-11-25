import nengo
import numpy as np
import soundfile as sf
from nengo.dists import Choice, ClippedExpDist
from nengo.networks import EnsembleArray
from nengo.utils.compat import is_array, is_string, iteritems

from . import params
from .mfcc import mfcc
from .networks import (  # noqa: F401
    AuditoryPeriphery,
    Cepstra,
    FeedforwardDeriv,
    IntermediateDeriv,
    RhythmicDMP,
    Sequencer,
    SyllableSequence,
)
from .networks.dmp import traj2func
from .processes import ArrayProcess


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
    audio = params.NdarrayParam(default=None, shape=('*', 1))
    fs = params.NumberParam(default=16000)
    dt = params.NumberParam(default=0.01)
    window_dt = params.NumberParam(default=0.025)
    n_cepstra = params.IntParam(default=13)
    n_filters = params.IntParam(default=26)
    n_fft = params.IntParam(default=512)
    minfreq = params.NumberParam(default=0)
    maxfreq = params.NumberParam(default=6000)
    preemph = params.NumberParam(default=0)
    lift = params.NumberParam(default=0)
    energy = params.BoolParam(default=False)
    n_derivatives = params.IntParam(default=0)
    deriv_spread = params.IntParam(default=2)

    def __call__(self):
        return mfcc(**self.kwargs())

class PeripheryParams(ParamsObject):
    freqs = params.NdarrayParam(default=None, shape=('*',))
    sound_process = params.ProcessParam(default=None)
    auditory_filter = params.BrianFilterParam(default=None)
    neurons_per_freq = params.IntParam(default=12)
    fs = params.NumberParam(default=20000)
    middle_ear = params.BoolParam(default=True)


class CepstraParams(ParamsObject):
    n_neurons = params.IntParam(default=30)
    n_cepstra = params.IntParam(default=13)


class FeedforwardDerivParams(ParamsObject):
    n_neurons = params.IntParam(default=30)
    tau_fast = params.NumberParam(default=0.005)
    tau_slow = params.NumberParam(default=0.1)


class IntermediateDerivParams(ParamsObject):
    n_neurons = params.IntParam(default=30)
    tau = params.NumberParam(default=0.1)


class AudioFeatures(object):
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
        assert self.mfcc.minfreq == self.periphery.freqs[0]
        assert self.mfcc.maxfreq == self.periphery.freqs[-1]
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
    n_per_d = params.IntParam(default=50)
    syllable_d = params.IntParam(default=96)
    difference_gain = params.NumberParam(default=15)
    n_positions = params.NumberParam(default=7)
    threshold_memories = params.BoolParam(default=True)
    add_default_output = params.BoolParam(default=True)


class SequencerParams(ParamsObject):
    n_per_d = params.IntParam(default=120)
    timer_tau = params.NumberParam(default=0.05)
    timer_freq = params.NumberParam(default=2.)
    reset_time = params.NumberParam(default=0.7)
    reset_threshold = params.NumberParam(default=0.5)
    reset_to_gate = params.NumberParam(default=-0.7)
    gate_threshold = params.NumberParam(default=0.4)


class SyllableParams(ParamsObject):
    n_per_d = params.IntParam(default=120)
    label = params.StringParam(default=None)
    freq = params.NumberParam(default=3.)
    trajectory = params.NdarrayParam(shape=('*', 48))
    tau = params.NumberParam(default=0.025)

    def kwargs(self):
        return {'n_per_d': self.n_per_d,
                'freq': self.freq,
                'tau': self.tau}


class ProductionInfoParams(ParamsObject):
    n_per_d = params.IntParam(default=60)
    dimensions = params.IntParam(default=None)
    threshold = params.NumberParam(default=0.3)


class ExperimentParams(ParamsObject):
    dt = params.NumberParam(default=0.001)
    sequence = params.StringParam(default=None)
    t_release = params.NumberParam(default=0.2)


class Production(object):
    def __init__(self):
        self.config = nengo.Config(
            nengo.Ensemble, nengo.Connection, nengo.Probe)
        self.sequence = SyllableSequenceParams()
        self.sequencer = SequencerParams()
        self.syllables = []
        self.production_info = ProductionInfoParams()
        self.experiment = ExperimentParams()

    def add_syllable(self, **kwargs):
        syll = SyllableParams()
        for k, v in iteritems(kwargs):
            setattr(syll, k, v)
        self.syllables.append(syll)

    def build(self, net=None):
        if net is None:
            net = nengo.Network("Sermo syllable production")
        with net, self.config:
            self.build_sequence(net)
            self.build_sequencer(net)
            self.build_syllables(net)
            self.build_connections(net)
            self.build_experiment(net)
        return net

    def build_syllables(self, net):
        assert len(self.syllables) > 0, "No syllables added"

        # Make a readout for the production info coming from the DMPs
        intercepts = ClippedExpDist(0.15, self.production_info.threshold, 1)
        net.production_info = EnsembleArray(self.production_info.n_per_d,
                                            self.production_info.dimensions,
                                            encoders=Choice([[1]]),
                                            intercepts=intercepts)

        net.syllables = []
        dt = self.experiment.dt
        for syllable in self.syllables:
            forcing_f, gesture_ix = traj2func(syllable.trajectory, dt=dt)
            dmp = RhythmicDMP(forcing_f=forcing_f, **syllable.kwargs())
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

    def build_experiment(self, net):
        # At the start of of the experiment...
        vocab = net.sequence.vocab

        # initial position is POS1,
        net.init_idx = nengo.Node(lambda t: vocab.parse('POS1').v
                                  if t < self.experiment.t_release + 0.1
                                  else vocab.parse('0').v)
        nengo.Connection(net.init_idx, net.sequence.pos.input)

        # the sequence is as given,
        net.seq_input = nengo.Node(vocab.parse(self.experiment.sequence).v)
        nengo.Connection(net.seq_input, net.sequence.sequence.input)

        # the dmps are reset,
        net.init_reset = nengo.Node(
            lambda t: 1.0 if t < self.experiment.t_release else 0.0)
        nengo.Connection(net.init_reset, net.sequencer.reset)

        # the timer is started.
        nengo.Connection(net.init_reset, net.sequencer.timer,
                         transform=[[-1], [0]])

        # Shouldn't be needed
        # init_gate = nengo.Node(output=lambda t: -1.0 if t < 0.2 else 0.0)
        # nengo.Connection(init_gate, gate)
