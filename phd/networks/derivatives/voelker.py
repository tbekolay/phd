import nengo
import numpy as np
from nengo.networks import EnsembleArray
from nengo.utils.filter_design import tf2ss

from .lti import ab_norm, exp_delay, LTI, lti2sim, multidim_lti, scale_lti


def lti(n_neurons, dimensions, ltisys, synapse=nengo.Lowpass(0.005),
        controlled=False, dt=0.001, radii=None, radius=1.0, net=None):
    if net is None:
        net = nengo.Network("Derivative (Voelker)")

    if radii is None:
        radii = ab_norm(ltisys)
    radii *= radius
    ltisys = scale_lti(ltisys, radii)
    ltisys = lti2sim(ltisys, synapse, dt)
    ltisys = multidim_lti(ltisys, dimensions)

    size_in = ltisys.b.shape[1]
    size_state = ltisys.a.shape[0]
    size_out = ltisys.c.shape[0]

    with net:
        in_ea = EnsembleArray(n_neurons, n_ensembles=size_in)
        out_ea = EnsembleArray(n_neurons, n_ensembles=size_out)
        net.state = EnsembleArray(n_neurons, n_ensembles=size_state)

        nengo.Connection(net.state.output, net.state.input,
                         transform=ltisys.a, synapse=synapse)
        nengo.Connection(in_ea.output, net.state.input,
                         transform=ltisys.b, synapse=synapse)
        nengo.Connection(net.state.output, out_ea.input,
                         transform=ltisys.c, synapse=synapse)
        nengo.Connection(in_ea.output, out_ea.input,
                         transform=ltisys.d, synapse=synapse)
        net.input = in_ea.input
        net.output = out_ea.output
    return net


def deconvolution(n_neurons, dimensions, tf, delay, degree=4, **lti_kwargs):
    """Approximate the inverse of a given transfer function using a delay."""
    num, den = [np.poly1d(tf[0]), np.poly1d(tf[1])]
    order = len(den) - len(num)
    if order >= degree:
        raise ValueError("order (%d) must be < degree (%d)" % (order, degree))
    edp, edq = exp_delay(degree - order, degree, delay)
    p, q = np.polymul(edp, den), np.polymul(edq, num)
    return lti(n_neurons, dimensions, LTI(*tf2ss(p, q)), **lti_kwargs)


def derivative(n_neurons, dimensions, tau, **deconv_kwargs):
    """Output a signal that is a derivative of the input."""
    return deconvolution(n_neurons, dimensions,
                         ([1], [tau, 0]), **deconv_kwargs)


def Voelker(delay, dimensions,
            n_neurons=100, tau=0.005, tau_highpass=0.05, net=None):
    net = derivative(n_neurons, dimensions=dimensions, delay=delay,
                     tau=tau, radius=0.1, degree=2)

    with net:
        actual_output = nengo.Node(size_in=dimensions)
        # Scale by 1 / delay, to match Tripp networks
        nengo.Connection(net.output, actual_output,
                         synapse=tau_highpass, transform=1. / delay)
        net.output = actual_output
    return net
