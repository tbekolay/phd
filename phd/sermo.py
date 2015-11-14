import nengo
from nengo.utils.compat import iteritems

from .networks import (  # noqa: F401
    AuditoryPeriphery,
    Cepstra,
    FeedforwardDeriv,
    IntermediateDeriv,
)
from . import params


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


class CepstraParams(ParamsObject):
    n_neurons = params.IntParam(default=30)
    n_cepstra = params.IntParam(default=13)


class ExecutionParams(object):
    pass


class IntegrationParams(object):
    pass


class Features(object):
    def __init__(self):
        self.config = nengo.Config(
            nengo.Ensemble, nengo.Connection, nengo.Probe)
        self.periphery = PeripheryParams()
        self.cepstra = CepstraParams()

    def add_derivative(self):
        pass

    def build(self, net=None):
        if net is None:
            net = nengo.Network("Sermo feature extraction")
        with net, self.config:
            self.build_periphery(net)
            self.build_cepstra(net)
        return net

    def build_periphery(self, net):
        net.periphery = AuditoryPeriphery(**self.periphery.kwargs())

    def build_cepstra(self, net):
        net.cepstra = Cepstra(n_freqs=net.periphery.freqs.size,
                              **self.cepstra.kwargs())
        nengo.Connection(net.periphery.an.output, net.cepstra.input)
