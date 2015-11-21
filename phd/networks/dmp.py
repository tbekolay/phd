import nengo
import numpy as np
from nengo.dists import Choice, ClippedExpDist
from scipy import stats

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
        # x goes from 0 to 1; normalize by t_end
        # But give a bit of 0 at the start...
        if x < 0.1:
            return np.zeros(traj.shape[1])
        ix = min(int(rescale(x, 0.1, 1., 0., t_end-0.1) / dt), traj.shape[0]-1)
        return traj[ix]
    _trajf.traj = traj
    _trajf.ix = traj_ix
    return _trajf, traj_ix


def RhythmicDMP(n_per_d, freq, forcing_f, tau=0.025, net=None):
    if net is None:
        net = nengo.Network(label="Rhythmic DMP")

    out_dims = forcing_f(0.).size
    omega = freq * tau * 2 * np.pi
    with net:
        # --- Decode forcing_f from oscillator
        net.osc = nengo.Ensemble(n_per_d * 2, dimensions=2,
                                 intercepts=ClippedExpDist(0.15, 0.3, 0.6),
                                 label=forcing_f.__name__)
        nengo.Connection(net.osc, net.osc,
                         synapse=tau,
                         transform=[[1, -omega], [omega, 1]])
        net.output = nengo.Node(size_in=out_dims)
        nengo.Connection(net.osc, net.output,
                         function=radial_f(forcing_f), synapse=None)

        # --- Drive the oscillator to a starting position
        net.reset = nengo.Node(size_in=1)
        net.diff_inhib = nengo.Ensemble(20, dimensions=1,
                                        intercepts=ClippedExpDist(0.15, -0.5, 0.1),
                                        encoders=Choice([[1]]))
        net.diff = nengo.Ensemble(n_per_d, dimensions=2)
        nengo.Connection(net.reset, net.diff_inhib, transform=-1, synapse=None)
        nengo.Connection(net.diff_inhib.neurons, net.diff.neurons,
                         transform=-np.ones((n_per_d, 20)))
        nengo.Connection(net.osc, net.diff)
        reset_goal = np.array([-1, omega*0.9])
        nengo.Connection(net.diff, net.osc, function=lambda x: reset_goal - x)

        # --- Inhibit the oscillator by default
        net.inhibit = nengo.Ensemble(20, dimensions=1,
                                     intercepts=ClippedExpDist(0.15, -0.5, 0.1),
                                     encoders=Choice([[1]]))
        nengo.Connection(net.inhibit.neurons, net.osc.neurons,
                         transform=-np.ones((n_per_d * 2, 20)))

        # --- Disinhibit when appropriate
        net.disinhibit = nengo.Node(size_in=1)
        nengo.Connection(net.disinhibit, net.inhibit, transform=-1)
    return net


def InverseDMP():
    pass
