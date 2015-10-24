import nengo
import numpy as np
from nengo.networks import EnsembleArray
from nengo.utils.compat import is_array
from nengo.utils.connection import target_function


def PhonemeDetector(neurons_per_d, eval_points, targets, net=None):
    """A naive implementation of phoneme detection.

    We compress all our frequency and temporal information into
    a single ensemble and decode.
    """
    if net is None:
        net = nengo.Network("Phoneme detector")

    if is_array(eval_points) and is_array(targets):
        training = False
        in_dims = eval_points.shape[1]
        out_dims = targets.shape[1]
    else:
        training = True
        in_dims = eval_points
        out_dims = targets

    with net:
        net.input = nengo.Ensemble(neurons_per_d * in_dims, in_dims)
        net.phoneme_in = net.input
        if not training:
            net.output = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
            nengo.Connection(net.phoneme_in,
                             net.output,
                             solver=nengo.solvers.LstsqMultNoise(),
                             **target_function(eval_points, targets))

    return net


def SumPoolPhonemeDetector(neurons_per_d, eval_points, targets,
                           pooling=3, net=None):
    """A phoneme detector that sums data from nearby freqs."""
    if net is None:
        net = nengo.Network("SumPool Phoneme detector")

    if is_array(eval_points) and is_array(targets):
        training = False
        in_dims = eval_points.shape[1]
        total_in = in_dims * pooling
        out_dims = targets.shape[1]
    else:
        training = True
        total_in = eval_points
        assert total_in % pooling == 0, (
            "Pooling must divide in_dims_freqs evenly")
        in_dims = int(total_in // pooling)
        out_dims = targets

    with net:
        # Pool AN / deriv inputs
        net.input = nengo.Node(size_in=total_in)
        net.phoneme_in = nengo.Ensemble(neurons_per_d * in_dims,
                                        dimensions=in_dims,
                                        radius=pooling)

        for i in range(pooling):
            # print np.arange(total_in)[i::pooling] to see what's happening
            nengo.Connection(net.input[i::pooling], net.phoneme_in)

        if not training:
            net.output = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
            nengo.Connection(net.phoneme_in,
                             net.output,
                             solver=nengo.solvers.LstsqMultNoise(),
                             **target_function(eval_points, targets))

    return net


def ProdPoolPhonemeDetector(neurons_per_d, eval_points, targets,
                            pooling=3, scale=1.5, net=None):
    """A phoneme detector that multiplies data from nearby freqs."""
    if net is None:
        net = nengo.Network("ProdPool Phoneme detector")

    if is_array(eval_points) and is_array(targets):
        training = False
        in_dims = eval_points.shape[1]
        out_dims = targets.shape[1]
    else:
        training = True
        in_dims = eval_points
        out_dims = targets

    with net:
        # Pool AN / deriv inputs

        # TODO: Confirm radius works here...
        ea_in = EnsembleArray(neurons_per_d * pooling,
                              n_ensembles=in_dims,
                              ens_dimensions=pooling,
                              radius=1. / np.sqrt(3))
        prod = ea_in.add_output('prod', function=lambda x: x[0] * x[1] * x[2])
        net.input = ea_in.input

        net.phoneme_in = nengo.Ensemble(neurons_per_d * in_dims,
                                        dimensions=in_dims)
        nengo.Connection(prod, net.phoneme_in)

        if not training:
            net.output = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
            nengo.Connection(net.phoneme_in,
                             net.output,
                             solver=nengo.solvers.LstsqMultNoise(),
                             **target_function(eval_points, targets))

    return net


# TODO: SumTilePhonemeDetector
#       Sort of like SumPool but more of moving window over all freqs,
#       so phoneme_in is pretty close to the size on input.
# TODO: ProdTilePhonemeDetector
#       Like ProdPool but tiles like SumTile.
