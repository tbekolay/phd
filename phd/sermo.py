import nengo
from nengo.utils.compat import iteritems, itervalues

from .networks import (
    AuditoryPeriphery,
    Derivative,
    PhonemeDetector,
)
from . import params, timit


class PeripheryParams(object):
    freqs = params.NdarrayParam(default=None, shape=('*',))
    sound_process = params.ProcessParam(default=None)
    auditory_filter = params.BrianFilterParam(default=None)
    neurons_per_freq = params.IntParam(default=12)
    fs = params.NumberParam(default=20000)
    middle_ear = params.BoolParam(default=True)
    zhang_synapse = params.BoolParam(default=False)


class DerivativeParams(object):
    delay = params.NumberParam(default=None)
    n_neurons = params.IntParam(default=30)
    klass = params.StringParam(default='Voelker')
    args = params.DictParam(default=None)


class IntegratorParams(object):
    tau = params.NumberParam(default=None)
    n_neurons = params.IntParam(default=30)


class PhonemeDetectorParams(object):
    name = params.StringParam(default="detector")
    derivatives = params.ListParam(default=[])
    phonemes = params.ListParam(default=[])
    rms = params.NumberParam(default=0.5)
    max_simtime = params.NumberParam(default=5.0)
    sample_every = params.NumberParam(default=0.001)
    neurons_per_d = params.IntParam(default=30)
    pooling = params.IntParam(default=None)


class RecognitionParams(object):
    def __init__(self):
        self.periphery = PeripheryParams()
        self.derivatives = {}
        self.integrators = {}
        self.detectors = {}

    def add_derivative(self, **kwargs):
        assert 'delay' in kwargs, "Must define delay"
        deriv = DerivativeParams()
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

    def add_phoneme_detector(self, **kwargs):
        detector = PhonemeDetectorParams()
        for attr, val in iteritems(kwargs):
            setattr(detector, attr, val)
        assert detector.name not in self.detectors, "Name already used"
        self.detectors[detector.name] = detector
        return detector


def kwargs(param_obj):
    args = {}
    klass = param_obj.__class__
    for attr in dir(klass):
        if params.is_param(getattr(klass, attr)):
            args[attr] = getattr(param_obj, attr)
    return args


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
        net.periphery = AuditoryPeriphery(**kwargs(self.recognition.periphery))

    def build_derivatives(self, net):
        net.derivatives = {}
        for param in itervalues(self.recognition.derivatives):
            deriv = Derivative(dimensions=net.periphery.freqs.size,
                               **kwargs(param))
            nengo.Connection(net.periphery.an.output, deriv.input)
            net.derivatives[param.delay] = deriv

    def build_integrators(self, net):
        """Not really integrators, just a long time constant on input."""
        net.integrators = {}

        for param in itervalues(self.recognition.integrators):
            integrator = nengo.networks.EnsembleArray(
                param.n_neurons, net.periphery.freqs.size)
            net.integrators[param.tau] = integrator.output
            nengo.Connection(net.periphery.an.output, integrator.input,
                             synapse=nengo.Lowpass(param.tau))

    def build_detectors(self, net):
        net.detectors = {}

        dims = net.periphery.freqs.size

        for param in itervalues(self.recognition.detectors):
            total_dims = dims * (len(param.derivatives) + 1)
            training = timit.TrainingData(self, param)
            if not net.training:
                assert training.generated, "Generate training data first"
                eval_points, targets = training.get()
            else:
                eval_points, targets = total_dims, dims
            detector = PhonemeDetector(
                param.neurons_per_d, eval_points, targets)
            net.detectors[param.name] = detector

            nengo.Connection(net.periphery.an.output, detector.input[:dims])
            for i, delay in enumerate(param.derivatives):
                nengo.Connection(net.derivatives[delay].output,
                                 detector.input[(i+1)*dims:(i+2)*dims])

    def build_execution(self, net):
        pass

    def build_integtration(self, net):
        pass
