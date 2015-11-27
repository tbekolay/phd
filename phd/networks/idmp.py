import nengo
import nengo.utils.numpy as npext
import numpy as np
from nengo.dists import Choice, ClippedExpDist
from nengo.networks import Product

from .dmp import radial_f
from ..utils import rescale


def ff_inv(func, thresh=0.6, scale=0.7):
    def forced_func_inv(x):
        state, obs = x[0], x[1:]
        pred = func(state)
        dot = similarity(pred, obs) * scale
        # clip dot so only close matches count
        dot = 0. if dot < thresh else dot
        # return x[0] + dot * scale  # alternative!
        return x[0] + (dot >= thresh) * scale  # alternative!

    return forced_func_inv


def similarity(v1, v2):
    # v1 and v2 are vectors
    eps = np.nextafter(0, 1)  # smallest float above zero
    dot = np.dot(v1, v2)
    dot /= max(npext.norm(v1), eps)
    dot /= max(npext.norm(v2), eps)
    return dot


def update_osc(threshold, omega):
    def _update(x):
        x1, x2, sim = x
        # Should this be all or nothing?
        return omega * sim * x2, -omega * sim * x1
    return _update


def InverseDMP(n_per_d, freq, forcing_f,
               freq_scale=2.0, similarity_th=0.6, tau=0.05, net=None):
    if net is None:
        net = nengo.Network(label="Inverse Rhythmic DMP")

    obs_dims = forcing_f(0.).size
    state_dims = 1
    dims = state_dims + obs_dims
    # omega = freq * tau * 2 * np.pi
    with net:
        net.input = nengo.Node(size_in=obs_dims)

        net.state = nengo.Ensemble(n_per_d * dims, dimensions=dims,
                                   n_eval_points=10000, radius=1.4)
        nengo.Connection(net.state, net.state[0],
                         function=ff_inv(forcing_f), synapse=tau)
        nengo.Connection(net.input, net.state[1:], synapse=None)

        # TODO: try to get working...

        # # --- System state
        # net.osc = nengo.Ensemble(n_per_d * state_dims, dimensions=state_dims,
        #                          label=forcing_f.__name__)
        # # Integrates
        # nengo.Connection(net.osc, net.osc,
        #                  transform=[[1, 0], [0, 1]], synapse=tau)

        # # tmp?
        # nengo.Connection(nengo.Node(lambda t: [-1, 0] if t < 0.08 else 0.),
        #                  net.osc)

        # # --- Drive the oscillator to a starting position
        # net.reset = nengo.Node(size_in=1)
        # d_intercepts = ClippedExpDist(0.15, -0.5, 0.1)
        # net.diff_inhib = nengo.Ensemble(20, dimensions=1,
        #                                 intercepts=d_intercepts,
        #                                 encoders=Choice([[1]]))
        # net.diff = nengo.Ensemble(n_per_d, dimensions=2)
        # nengo.Connection(net.reset, net.diff_inhib, transform=-1, synapse=None)
        # nengo.Connection(net.diff_inhib.neurons, net.diff.neurons,
        #                  transform=-np.ones((n_per_d, 20)))
        # nengo.Connection(net.osc, net.diff)
        # reset_goal = np.array([-1, 0])
        # nengo.Connection(net.diff, net.osc, function=lambda x: reset_goal - x)

        # # --- Similarity computation
        # net.similarity = Product(n_per_d, dimensions=obs_dims)
        # # State emits prediction to A
        # nengo.Connection(net.osc, net.similarity.A,
        #                  function=radial_f(forcing_f))
        # # Observation is B
        # nengo.Connection(net.input, net.similarity.B, synapse=None)

        # # --- Normalize the dot product
        # # radius is a guess; likely no more than 3 gestures at once...
        # net.simnorm = nengo.Ensemble(n_per_d * 3, dimensions=3, radius=2.5)
        # nengo.Connection(net.similarity.output, net.simnorm[0],
        #                  transform=np.ones((1, obs_dims)))
        # nengo.Connection(net.osc, net.simnorm[1],
        #                  function=radial_f(forcing_f),
        #                  transform=np.ones((1, obs_dims)))
        # nengo.Connection(net.input, net.simnorm[2],
        #                  transform=np.ones((1, obs_dims)))

        # # --- Recurrence depends on state and normalized similarity
        # net.update = nengo.Ensemble(n_per_d * (state_dims + 1),
        #                             dimensions=state_dims + 1)
        # nengo.Connection(net.osc, net.update[:state_dims])
        # nengo.Connection(net.simnorm, net.update[state_dims],
        #                  function=lambda x: x[0] / (x[1] * x[2]))

        # # --- Amount to move the state depends on scaled similarity
        # nengo.Connection(net.update, net.osc, synapse=tau,
        #                  function=update_osc(similarity_th, omega))

    return net
