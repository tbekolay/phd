import nengo
from nengo.networks import EnsembleArray
from nengo.utils.compat import iteritems, itervalues

from .networks import (  # noqa: F401
    AuditoryPeriphery,
    PhonemeDetector,
    ProdTilePhonemeDetector,
    SumPoolPhonemeDetector,
    TrippFF,
    TrippInt,
    Voelker,
)
from . import params, timit


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
        assert hasattr(self, key), "Can only set params that exist"
        super(ParamsObject, self).__setattr__(key, value)

    def __getstate__(self):
        """Needed for pickling."""
        # Use ParamsObject specifically in case subclass shadows it
        return ParamsObject.kwargs(self)

    def __setstate__(self, state):
        """Needed for pickling."""
        for attr, val in iteritems(state):
            setattr(self, attr, val)


class PeripheryParams(ParamsObject):
    freqs = params.NdarrayParam(default=None, shape=('*',))
    sound_process = params.ProcessParam(default=None)
    auditory_filter = params.BrianFilterParam(default=None)
    neurons_per_freq = params.IntParam(default=12)
    fs = params.NumberParam(default=20000)
    middle_ear = params.BoolParam(default=True)
    zhang_synapse = params.BoolParam(default=False)


class DerivativeParams(ParamsObject):
    delay = params.NumberParam(default=None)
    n_neurons = params.IntParam(default=30)


class TrippFFParams(DerivativeParams):
    tau = params.NumberParam(default=0.005)


class TripIntParams(DerivativeParams):
    pass


class VoelkerParams(DerivativeParams):
    tau = params.NumberParam(default=0.005)
    tau_highpass = params.NumberParam(default=0.05)


class IntegratorParams(ParamsObject):
    tau = params.NumberParam(default=None)
    n_neurons = params.IntParam(default=30)


class PhonemeDetectorParams(ParamsObject):
    name = params.StringParam(default="detector")
    derivatives = params.ListParam(default=[])
    phonemes = params.ListParam(default=[])
    rms = params.NumberParam(default=0.5)
    max_simtime = params.NumberParam(default=1.0)
    sample_every = params.NumberParam(default=0.001)
    neurons_per_d = params.IntParam(default=30)

    @property
    def t_before(self):
        """Amount of time to preplay the audio when generating traning data."""
        start_transient = 0.005
        return max(self.derivatives) + start_transient

    def kwargs(self):
        """Have to return a subset of args; some are just used for training."""
        return {'neurons_per_d': self.neurons_per_d}


class SumPoolPhonemeDetectorParams(PhonemeDetectorParams):
    pooling = params.IntParam(default=3)

    def kwargs(self):
        """Have to return a subset of args; some are just used for training."""
        return {'neurons_per_d': self.neurons_per_d,
                'pooling': self.pooling}


class ProdTilePhonemeDetectorParams(PhonemeDetectorParams):
    spread = params.IntParam(default=1)
    center = params.IntParam(default=0)
    scale = params.NumberParam(default=2.0)

    def kwargs(self):
        """Have to return a subset of args; some are just used for training."""
        return {'neurons_per_d': self.neurons_per_d,
                'spread': self.spread,
                'scale': self.scale}


class RecognitionParams(object):
    def __init__(self):
        self.periphery = PeripheryParams()
        self.derivatives = {}
        self.integrators = {}
        self.detectors = {}

    @property
    def dimensions(self):
        return self.periphery.freqs.size

    def add_derivative(self, klass, **kwargs):
        assert 'delay' in kwargs, "Must define delay"
        deriv = globals()["%sParams" % klass]()
        for attr, val in iteritems(kwargs):
            setattr(deriv, attr, val)
        self.derivatives[deriv.delay] = deriv
        return deriv

    def add_integrator(self, **kwargs):
        assert 'tau' in kwargs, "Must define tau"
        integ = IntegratorParams()
        for attr, val in iteritems(kwargs):
            setattr(integ, attr, val)
        self.integrators[integ.tau] = integ
        return integ

    def add_phoneme_detector(self, hierarchical="", **kwargs):
        detector = globals()["%sPhonemeDetectorParams" % hierarchical]()
        for attr, val in iteritems(kwargs):
            setattr(detector, attr, val)
        assert detector.name not in self.detectors, "Name already used"
        self.detectors[detector.name] = detector
        return detector


class ExecutionParams(object):
    pass


class IntegrationParams(object):
    pass


class Sermo(object):
    def __init__(self, recognition=True, execution=True):
        self.recognition = RecognitionParams() if recognition else None
        self.execution = ExecutionParams() if execution else None
        self.integration = (IntegrationParams() if execution and recognition
                            else None)

    def build(self, training=False):
        net = nengo.Network()
        net.training = training
        if self.recognition is not None:
            self.build_recognition(net)
        if self.execution is not None:
            self.build_execution(net)
        if self.integration is not None:
            self.build_integration(net)
        return net

    def build_recognition(self, net):
        with net:
            # Periphery
            self.build_periphery(net)
            # Preprocessing
            self.build_derivatives(net)
            self.build_integrators(net)
            # Features
            self.build_detectors(net)

    def build_periphery(self, net):
        net.periphery = AuditoryPeriphery(
            **self.recognition.periphery.kwargs())

    def build_derivatives(self, net):
        net.derivatives = {}
        dims = self.recognition.dimensions

        for param in itervalues(self.recognition.derivatives):
            deriv = param.net(dimensions=dims, **param.kwargs())
            nengo.Connection(net.periphery.an.output, deriv.input)
            net.derivatives[param.delay] = deriv

    def build_integrators(self, net):
        """Not really integrators, just a long time constant on input."""
        net.integrators = {}
        dims = self.recognition.dimensions

        for param in itervalues(self.recognition.integrators):
            integrator = EnsembleArray(param.n_neurons, n_ensembles=dims)
            net.integrators[param.tau] = integrator
            nengo.Connection(net.periphery.an.output, integrator.input,
                             synapse=nengo.Lowpass(param.tau))

    def build_detectors(self, net):
        net.detectors = {}
        dims = self.recognition.dimensions

        for param in itervalues(self.recognition.detectors):
            total_dims = dims * (len(param.derivatives) + 1)
            training = timit.TrainingData(self, param)
            if not net.training:
                assert training.generated, "Generate training data first"
                eval_points, targets = training.get()
            else:
                eval_points, targets = dims, len(param.phonemes)
            kwargs = param.kwargs()
            kwargs['eval_points'] = eval_points
            kwargs['targets'] = targets
            kwargs['size_in'] = total_dims
            detector = param.net(**kwargs)
            net.detectors[param.name] = detector

            nengo.Connection(net.periphery.an.output,
                             detector.input[:dims])
            for i, delay in enumerate(param.derivatives):
                nengo.Connection(net.derivatives[delay].output,
                                 detector.input[(i+1)*dims:(i+2)*dims])

    def build_execution(self, net):
        pass

    def build_integtration(self, net):
        pass
