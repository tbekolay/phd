import nengo
from nengo.utils.compat import iteritems

from .networks import (  # noqa: F401
    AuditoryPeriphery,
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
    zhang_synapse = params.BoolParam(default=False)


class RecognitionParams(object):
    def __init__(self):
        self.periphery = PeripheryParams()

    @property
    def dimensions(self):
        return self.periphery.freqs.size


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

    def build_periphery(self, net):
        net.periphery = AuditoryPeriphery(
            **self.recognition.periphery.kwargs())

    def build_execution(self, net):
        pass

    def build_integtration(self, net):
        pass
