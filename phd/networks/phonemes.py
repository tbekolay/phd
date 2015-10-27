import nengo
import numpy as np
from nengo.networks import EnsembleArray
from nengo.utils.compat import is_array, range
from nengo.utils.connection import target_function


def PhonemeDetector(neurons_per_d, eval_points, targets, size_in, net=None):
    """A naive implementation of phoneme detection.

    We compress all our frequency and temporal information into
    a single ensemble and decode.
    """
    if net is None:
        net = nengo.Network("Phoneme detector")

    if is_array(eval_points) and is_array(targets):
        training = False
        out_dims = targets.shape[1]
    else:
        training = True
        out_dims = targets

    with net:
        net.input = nengo.Ensemble(neurons_per_d * size_in, size_in)
        net.phoneme_in = net.input
        if not training:
            net.output = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
            nengo.Connection(net.phoneme_in,
                             net.output,
                             **target_function(eval_points, targets))

    return net


def SumPoolPhonemeDetector(neurons_per_d, eval_points, targets, size_in,
                           pooling=3, net=None):
    """A phoneme detector that sums data from nearby freqs."""
    if net is None:
        net = nengo.Network("SumPool Phoneme detector")

    if is_array(eval_points) and is_array(targets):
        training = False
        in_dims = eval_points.shape[1]
        out_dims = targets.shape[1]
    else:
        training = True
        assert size_in % pooling == 0, (
            "Pooling must divide in_dims_freqs evenly")
        in_dims = eval_points
        out_dims = targets

    with net:
        # Pool AN / deriv inputs
        net.input = nengo.Node(size_in=size_in)
        net.phoneme_in = nengo.Ensemble(neurons_per_d * in_dims,
                                        dimensions=in_dims,
                                        radius=pooling)

        for i in range(pooling):
            # print np.arange(size_in)[i::pooling] to see what's happening
            nengo.Connection(net.input[i::pooling], net.phoneme_in)

        if not training:
            net.output = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
            nengo.Connection(net.phoneme_in,
                             net.output,
                             **target_function(eval_points, targets))

    return net


def ProdTilePhonemeDetector(neurons_per_d, eval_points, targets, size_in,
                            spread=1, center=0, scale=2.0, net=None):
    """A phoneme detector that multiplies data from nearby freqs.

    Unlike SumPool, this combines information across timescales.

    `center` is which group becomes the center; 0 is the auditory nerve output,
    1 is the first derivative in the list, etc.
    """
    if net is None:
        net = nengo.Network("ProdTile Phoneme detector")

    if is_array(eval_points) and is_array(targets):
        training = False
        in_dims = eval_points.shape[1]
        out_dims = targets.shape[1]
    else:
        training = True
        in_dims = eval_points
        out_dims = targets

    n_groups = size_in // in_dims
    dims_per = 1 + (n_groups - 1) * spread * 2

    with net:
        # Pool AN / deriv inputs
        ea_in = EnsembleArray(neurons_per_d * dims_per,
                              n_ensembles=out_dims,
                              ens_dimensions=dims_per)
        # TODO: scale logarithmically instead of linearly?
        #       like, instead of prod(x) do something else
        prod = ea_in.add_output('prod', function=lambda x:
                                np.prod(x) * scale * np.log(dims_per))

        net.input = nengo.Node(size_in=size_in)

        # Center connection
        nengo.Connection(net.input[center*out_dims:(center+1)*out_dims],
                         ea_in.input[::dims_per])

        # Group connections
        # NB! Hard to explain. TODO: figure?
        groups = list(range(n_groups))
        groups.remove(center)
        offset = 0
        for group in groups:
            for dist in range(1, spread+1):
                for sign in 1, -1:
                    offset += 1
                    if sign > 0:
                        pre_view = net.input[group*out_dims+dist:
                                             (group+1)*out_dims]
                    else:
                        pre_vew = net.input[group*out_dims:
                                            (group+1)*out_dims-dist]
                    if sign > 0:
                        post_view = ea_in.input[dims_per+offset::dims_per]
                    else:
                        post_view = ea_in.input[offset:-dims_per:dims_per]
                    nengo.Connection(pre_view, post_view)

        net.phoneme_in = nengo.Ensemble(neurons_per_d * out_dims,
                                        dimensions=out_dims)
        nengo.Connection(prod, net.phoneme_in)

        if not training:
            net.output = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
            nengo.Connection(net.phoneme_in,
                             net.output,
                             **target_function(eval_points, targets))

    return net


# TODO: SumTilePhonemeDetector
#       Sort of like SumPool but more of moving window over all freqs,
#       so phoneme_in is pretty close to the size on input.
# TODO: ProdTilePhonemeDetector
#       Like ProdPool but tiles like SumTile.
# TODO: the tile ones should combine non-derivs and derivs...
#       maybe exclusively?
