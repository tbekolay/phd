import nengo
import numpy as np

from .lti import LTI, multidim_lti


def FeedbackDerivative(dimensions, lti,
                       n_neurons=100, tau=0.005, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (TrippFB)")

    a = np.identity(lti.a.shape[0]) + lti.a * tau
    b = lti.b * tau
    lti = multidim_lti(LTI(a, b, lti.c, lti.d), dimensions)

    with net:
        net.input = nengo.Ensemble(n_neurons * dimensions, dimensions)
        net.output = nengo.Ensemble(n_neurons * dimensions, dimensions)
        net.diff = nengo.Ensemble(n_neurons * dimensions * 2, dimensions * 2)
        nengo.Connection(net.input, net.diff, synapse=tau, transform=lti.b)
        nengo.Connection(net.diff, net.diff, synapse=tau, transform=lti.a)
        nengo.Connection(net.diff, net.output, transform=lti.c)
    return net


def TrippFBInt(dimensions, n_neurons=100, tau=0.01, net=None):
    a = np.array([[-5, -7.5], [3.3333, -15]])
    b = np.array([[10], [20]])
    c = np.array([[10, 0]])
    lti = LTI(a, b, c, None)
    return FeedbackDerivative(dimensions, lti, n_neurons, tau, net)


def TrippButterworth(dimensions, n_neurons=100, tau=0.01, net=None):
    a = np.array([[-8.8858, 19.9931], [-3.9492, -8.8858]])
    b = np.array([[27.4892], [-12.2174]])
    c = np.array([[5.7446, 0]])
    lti = LTI(a, b, c, None)
    return FeedbackDerivative(dimensions, lti, n_neurons, tau, net)
