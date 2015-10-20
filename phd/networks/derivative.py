from .derivatives import (
    TrippFF, TrippInt, TrippFBInt, TrippButterworth, Voelker)


def Derivative(klass, **kwargs):
    try:
        klass = globals()[klass]
    except KeyError:
        raise ValueError("Derivative class '%s' not recognized." % klass_str)
    return klass(**kwargs)
