import nengo
from nengo.utils.compat import iteritems

from .networks import Derivative, AuditoryPeriphery
from . import params


class RecognitionParams(object):
    class PeripheryParams(object):
        freqs = params.NdarrayParam(default=None, shape=('*',))
        sound = params.ProcessParam(default=None)
        auditory_filter = params.BrianFilterParam(default=None)
        neuron_per_freq = params.IntParam(default=12)
        fs = params.NumberParam(default=20000)
        middle_ear = params.BoolParam(default=True)
        zhang_synapse = params.BoolParam(default=False)

    class DerivativeParams(object):
        n_neurons = params.IntParam(default=30)
        klass = params.StringParam(default='Voelker')
        args = params.DictParam(default=None)

    class IntegratorParams(object):
        n_neurons = params.IntParam(default=30)

    def __init__(self):
        self.periphery = RecognitionParams.PeripheryParams()
        self.derivatives = {}
        self.integrators = {}

    def add_derivative(self, delay):
        self.derivatives[delay] = RecognitionParams.DerivativeParams()

    def add_integrator(self, tau):
        self.integrators[tau] = RecognitionParams.IntegratorParams()


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

    def build(self):
        net = nengo.Network()
        if self.recognition is not None:
            self.build_recognition(net)
        if self.execution is not None:
            self.build_execution(net)
        if self.integration is not None:
            self.build_integration(net)

    def build_recognition(self, net):
        with net:
            self.build_periphery(net)
            self.build_derivatives(net)
            self.build_integrators(net)

    def build_periphery(self, net):
        net.periphery = AuditoryPeriphery(**kwargs(self.recognition.periphery))

    def build_derivatives(self, net):
        net.derivatives = {}
        for delay, param in iteritems(self.recognition.derivatives):
            net.derivatives[delay] = Derivative(delay=delay, **kwargs(param))
            # TODO: connect periphery to derivs

    def build_integrators(self, net):
        """Not really integrators, just a long time constant on input."""
        net.integrators = {}

        for tau, param in iteritems(self.recognition.integrators):
            integrator = nengo.networks.EnsembleArray(
                param.n_neurons, self.recognition.freqs.size)
            net.integrators[tau] = integrator.output
            nengo.Connection(net.an.output, integrator.input,
                             synapse=nengo.Lowpass(tau))

    def build_execution(self, net):
        pass

    def build_integtration(self, net):
        pass
