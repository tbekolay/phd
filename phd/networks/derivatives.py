import nengo
from nengo.networks import EnsembleArray


def FeedforwardDeriv(n_neurons, dimensions,
                     tau_fast=0.005, tau_slow=0.1, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (two feedforward connections)")

    assert tau_slow > tau_fast, (
        "tau_fast (%s) must be > tau_slow (%s)" % (tau_fast, tau_slow))

    with net:
        net.ea_in = EnsembleArray(n_neurons, n_ensembles=dimensions)
        net.ea_out = EnsembleArray(n_neurons, n_ensembles=dimensions)
        nengo.Connection(net.ea_in.output, net.ea_out.input,
                         synapse=tau_fast, transform=1. / tau_slow)
        nengo.Connection(net.ea_in.output, net.ea_out.input,
                         synapse=tau_slow, transform=-1. / tau_slow)
        net.input = net.ea_in.input
        net.output = net.ea_out.output
    return net


def IntermediateDeriv(n_neurons, dimensions, tau=0.1, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (intermediate ensemble)")

    with net:
        net.ea_in = EnsembleArray(n_neurons, n_ensembles=dimensions)
        net.ea_interm = EnsembleArray(n_neurons, n_ensembles=dimensions)
        net.ea_out = EnsembleArray(n_neurons, n_ensembles=dimensions)
        nengo.Connection(net.ea_in.output, net.ea_interm.input, synapse=tau)
        nengo.Connection(net.ea_in.output, net.ea_out.input,
                         synapse=tau, transform=1. / tau)
        nengo.Connection(net.ea_interm.output, net.ea_out.input,
                         synapse=tau, transform=-1. / tau)
        net.input = net.ea_in.input
        net.output = net.ea_out.output
    return net
