import nengo
import numpy as np
from nengo.networks import EnsembleArray

from .lti import LTI, multidim_lti


def FeedbackDerivative(dimensions, lti,
                       n_neurons=100, tau=0.005, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (TrippFB)")

    a = np.identity(lti.a.shape[0]) + lti.a * tau
    b = lti.b * tau
    lti = multidim_lti(LTI(a, b, lti.c, lti.d), dimensions)

    with net:
        in_ea = EnsembleArray(n_neurons, n_ensembles=dimensions)
        out_ea = EnsembleArray(n_neurons, n_ensembles=dimensions)
        net.diff = EnsembleArray(n_neurons, n_ensembles=dimensions * 2)
        nengo.Connection(in_ea.output, net.diff.input,
                         synapse=tau, transform=lti.b)
        nengo.Connection(net.diff.output, net.diff.input,
                         synapse=tau, transform=lti.a)
        nengo.Connection(net.diff.output, out_ea.input, transform=lti.c)
        net.input = in_ea.input
        net.output = out_ea.output
    return net


def TrippFBInt(delay, dimensions, n_neurons=100, net=None):
    a = np.array([[-5, -7.5], [3.3333, -15]])
    b = np.array([[10], [20]])
    c = np.array([[10, 0]])
    lti = LTI(a, b, c, None)
    # Using delay as tau; probably wrong?
    return FeedbackDerivative(dimensions, lti, n_neurons, delay, net)


def TrippButterworth(delay, dimensions, n_neurons=100, net=None):
    a = np.array([[-8.8858, 19.9931], [-3.9492, -8.8858]])
    b = np.array([[27.4892], [-12.2174]])
    c = np.array([[5.7446, 0]])
    lti = LTI(a, b, c, None)
    # Using delay as tau; probably wrong?
    return FeedbackDerivative(dimensions, lti, n_neurons, delay, net)
