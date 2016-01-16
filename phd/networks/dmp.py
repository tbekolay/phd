import nengo
import numpy as np
from nengo.dists import Choice, Exponential

from ..utils import rescale


def radial_f(func):
    def _radial_f(x):
        return func(np.arctan2(x[1], x[0]) / (2 * np.pi) + 0.5)
    _radial_f.__name__ = func.__name__
    return _radial_f


def traj2func(traj, dt=0.001):
    traj_ix = np.unique(np.nonzero(traj)[1])
    traj = traj[:, traj_ix]
    t_end = traj.shape[0] * dt

    def _trajf(x):
        # x goes from 0 to 1; normalize by t_end; leave some space at start
        if x < 0.06:
            return np.zeros(traj.shape[1])
        ix = min(int(rescale(x, 0.06, 1.0, 0., t_end) / dt), traj.shape[0]-1)
        return traj[ix]
    _trajf.traj = traj
    _trajf.ix = traj_ix
    return _trajf, traj_ix


def DMP(n_per_d, c, forcing_f, tau=0.05, net=None):
    if net is None:
        net = nengo.Network(label="DMP")

    out_dims = forcing_f(0.).size
    with net:
        # --- Decode forcing_f from oscillator
        net.state = nengo.Ensemble(n_per_d, dimensions=1,
                                   label=forcing_f.__name__)
        nengo.Connection(net.state, net.state, synapse=tau)
        net.output = nengo.Node(size_in=out_dims)
        nengo.Connection(net.state, net.output,
                         function=forcing_f, synapse=None)

        # --- Input is a small biased oscillator to ramp up smoothly
        net.osc = nengo.Ensemble(n_per_d * 2, dimensions=2, radius=0.01)
        nengo.Connection(net.osc, net.osc,
                         transform=np.array([[2, -1], [1, 2]]))  # actually?
        nengo.Connection(net.osc, net.state, transform=c,
                         function=lambda x: x[0] + 0.5)
        # Kick start the oscillator
        kick = nengo.Node(lambda t: 1 if t < 0.1 else 0)
        nengo.Connection(kick, net.osc, transform=np.ones((2, 1)))

        # --- Inhibit the state by default
        i_intercepts = Exponential(0.15, -0.5, 0.1)
        net.inhibit = nengo.Ensemble(20, dimensions=1,
                                     intercepts=i_intercepts,
                                     encoders=Choice([[1]]))
        nengo.Connection(net.inhibit.neurons, net.state.neurons,
                         transform=-np.ones((n_per_d, 20)))

        # --- Disinhibit when appropriate
        net.disinhibit = nengo.Node(size_in=1)
        nengo.Connection(net.disinhibit, net.inhibit, transform=-1)
    return net


def RhythmicDMP(n_per_d, freq, forcing_f, tau=0.025, net=None):
    if net is None:
        net = nengo.Network(label="Rhythmic DMP")

    out_dims = forcing_f(0.).size
    omega = freq * tau * 2 * np.pi
    with net:
        # --- Decode forcing_f from oscillator
        net.osc = nengo.Ensemble(n_per_d * 2, dimensions=2,
                                 intercepts=Exponential(0.15, 0.3, 0.6),
                                 label=forcing_f.__name__)
        nengo.Connection(net.osc, net.osc,
                         synapse=tau,
                         transform=[[1, -omega], [omega, 1]])
        net.output = nengo.Node(size_in=out_dims)
        nengo.Connection(net.osc, net.output,
                         function=radial_f(forcing_f), synapse=None)

        # --- Drive the oscillator to a starting position
        net.reset = nengo.Node(size_in=1)
        d_intercepts = Exponential(0.2, -0.5, 0.1)
        net.diff_inhib = nengo.Ensemble(20, dimensions=1,
                                        intercepts=d_intercepts,
                                        encoders=Choice([[1]]))
        net.diff = nengo.Ensemble(n_per_d, dimensions=2)
        nengo.Connection(net.reset, net.diff_inhib, transform=-1, synapse=None)
        nengo.Connection(net.diff_inhib.neurons, net.diff.neurons,
                         transform=-np.ones((n_per_d, 20)))
        nengo.Connection(net.osc, net.diff)
        reset_goal = np.array([-1, omega*0])
        nengo.Connection(net.diff, net.osc, function=lambda x: reset_goal - x)

        # --- Inhibit the oscillator by default
        i_intercepts = Exponential(0.15, -0.5, 0.1)
        net.inhibit = nengo.Ensemble(20, dimensions=1,
                                     intercepts=i_intercepts,
                                     encoders=Choice([[1]]))
        nengo.Connection(net.inhibit.neurons, net.osc.neurons,
                         transform=-np.ones((n_per_d * 2, 20)))

        # --- Disinhibit when appropriate
        net.disinhibit = nengo.Node(size_in=1)
        nengo.Connection(net.disinhibit, net.inhibit, transform=-1)
    return net
