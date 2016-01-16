import nengo
import nengo.utils.numpy as npext
import numpy as np
from nengo.dists import Choice, Exponential


def ff_inv(func, thresh=0.6, scale=0.7):
    def forced_func_inv(x):
        state, obs = x[0], x[1:]
        pred = func(state)
        return x[0] + (similarity(pred, obs) >= thresh) * scale
    return forced_func_inv


def similarity(v1, v2):
    # v1 and v2 are vectors
    eps = np.nextafter(0, 1)  # smallest float above zero
    dot = np.dot(v1, v2)
    dot /= max(npext.norm(v1), eps)
    dot /= max(npext.norm(v2), eps)
    return dot


def InverseDMP(n_per_d, forcing_f, scale=0.7, reset_scale=2.5,
               similarity_th=0.6, tau=0.05, net=None):
    if net is None:
        net = nengo.Network(label="Inverse Rhythmic DMP")

    obs_dims = forcing_f(0.).size
    state_dims = 1
    dims = state_dims + obs_dims

    with net:
        net.input = nengo.Node(size_in=obs_dims)

        # --- iDMP state contains system state and observations
        net.state = nengo.Ensemble(n_per_d * dims, dimensions=dims,
                                   n_eval_points=10000, radius=1.4)
        nengo.Connection(net.input, net.state[1:], synapse=None)

        # --- Update state based on the current observation
        f = ff_inv(forcing_f, thresh=similarity_th, scale=scale)
        nengo.Connection(net.state, net.state[0], function=f, synapse=tau)

        # --- Reset system state
        net.reset = nengo.Node(size_in=1)
        d_intercepts = Exponential(0.15, -0.5, 0.1)
        net.diff_inhib = nengo.Ensemble(20, dimensions=1,
                                        intercepts=d_intercepts,
                                        encoders=Choice([[1]]))
        net.diff = nengo.Ensemble(n_per_d, dimensions=1)
        nengo.Connection(net.reset, net.diff_inhib, transform=-1, synapse=None)
        nengo.Connection(net.diff_inhib.neurons, net.diff.neurons,
                         transform=-np.ones((n_per_d, 20)))
        nengo.Connection(net.state[0], net.diff)
        nengo.Connection(net.diff, net.state[0], transform=-reset_scale)

    return net
