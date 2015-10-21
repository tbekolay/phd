from .derivatives import (
    TrippFF, TrippInt, TrippFBInt, TrippButterworth, Voelker)


def Derivative(klass, n_neurons, dimensions, delay, args):
    try:
        klass = globals()[klass]
    except KeyError:
        raise ValueError("Derivative class '%s' not recognized." % klass_str)
    return klass(n_neurons=n_neurons, delay=delay, dimensions=dimensions,
                 **args)
