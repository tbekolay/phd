import nengo
import numpy as np
from nengo.dists import Choice, Uniform
from scipy import stats


def radial_f(func):
    def _radial_f(x):
        return func(np.arctan2(x[1], x[0]) / (2 * np.pi) + 0.5)
    _radial_f.__name__ = func.__name__
    return _radial_f


def RhythmicDMP(n_per_d, freq, forcing_f, tau=0.025, net=None):
    if net is None:
        net = nengo.Network(label="Rhythmic DMP")

    out_dims = forcing_f(0.).size
    omega = freq * tau * 2 * np.pi
    with net:
        # Decode forcing_f from oscillator
        net.osc = nengo.Ensemble(n_per_d * 2, dimensions=2,
                                 intercepts=Uniform(0.3, 0.6),
                                 label=forcing_f.__name__)
        nengo.Connection(net.osc, net.osc,
                         synapse=tau,
                         transform=[[1, -omega], [omega, 1]])
        net.output = nengo.Node(size_in=out_dims)
        nengo.Connection(net.osc, net.output,
                         function=radial_f(forcing_f), synapse=None)

        # Kick start the oscillator
        net.kick = nengo.Node(size_in=1)
        nengo.Connection(
            net.kick, net.osc, transform=[[-1], [0.1]], synapse=None)
        # Inhibit the oscillator by default
        net.inhibit = nengo.Ensemble(20, dimensions=1,
                                     intercepts=Uniform(-0.5, 0.1),
                                     encoders=Choice([[1]]))
        nengo.Connection(net.inhibit.neurons, net.osc.neurons,
                         transform=-np.ones((n_per_d * 2, 20)))
        # Disinhibit when appropriate
        net.disinhibit = nengo.Node(size_in=1)
        nengo.Connection(net.disinhibit, net.inhibit, transform=-1)
    return net


def InverseDMP():
    pass
