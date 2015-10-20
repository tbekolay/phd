import nengo


def TrippFF(dimensions, n_neurons=50, tau_fast=0.005, tau_slow=0.1, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (TrippInt)")

    with net:
        tau_diff = tau_slow - tau_fast
        net.input = nengo.Ensemble(n_neurons * dimensions, dimensions)
        net.output = nengo.Ensemble(n_neurons * dimensions, dimensions)
        nengo.Connection(net.input, net.output,
                         synapse=tau_fast, transform=1. / tau_diff)
        nengo.Connection(net.input, net.output,
                         synapse=tau_slow, transform=-1. / tau_diff)
    return net


def TrippInt(dimensions, n_neurons=50, tau=0.01, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (TrippFF)")

    with net:
        net.input = nengo.Ensemble(n_neurons * dimensions, dimensions)
        net.intermediate = nengo.Ensemble(n_neurons * dimensions, dimensions)
        net.output = nengo.Ensemble(n_neurons * dimensions, dimensions)
        nengo.Connection(net.input, net.intermediate, synapse=tau)
        nengo.Connection(net.input, net.output,
                         synapse=tau, transform=1. / tau)
        nengo.Connection(net.intermediate, net.output,
                         synapse=tau, transform=-1. / tau)
    return net
