import nengo
from nengo.networks import EnsembleArray


def TrippFF(delay, dimensions, n_neurons=50, tau=0.005, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (TrippInt)")

    assert delay > tau, "Delay (%s) must be > tau (%s)" % (delay, tau)

    with net:
        tau_slow = tau + delay
        in_ea = EnsembleArray(n_neurons, n_ensembles=dimensions)
        out_ea = EnsembleArray(n_neurons, n_ensembles=dimensions)
        nengo.Connection(in_ea.output, out_ea.input,
                         synapse=tau, transform=1. / delay)
        nengo.Connection(in_ea.output, out_ea.input,
                         synapse=tau_slow, transform=-1. / delay)
        net.input = in_ea.input
        net.output = out_ea.output
    return net


def TrippInt(delay, dimensions, n_neurons=50, net=None):
    if net is None:
        net = nengo.Network(label="Derivative (TrippFF)")

    # Two connections, so tau = delay / 2
    tau = delay * 0.5

    with net:
        in_ea = EnsembleArray(n_neurons, n_ensembles=dimensions)
        net.intermediate = EnsembleArray(n_neurons, n_ensembles=dimensions)
        out_ea = EnsembleArray(n_neurons, n_ensembles=dimensions)
        nengo.Connection(in_ea.output, net.intermediate.input, synapse=tau)
        nengo.Connection(in_ea.output, out_ea.input,
                         synapse=tau, transform=1. / tau)
        nengo.Connection(net.intermediate.output, out_ea.input,
                         synapse=tau, transform=-1. / tau)
        net.input = in_ea.input
        net.output = out_ea.output
    return net
