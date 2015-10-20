import nengo


def PhonemeDetector(neurons_per_d, eval_points, targets, delays=None):
    """A naive implementation of phoneme detection.

    We compress all our frequency and temporal information into
    a single ensemble (one for vowels, with a long time constant,
    and one for consonants, with a short time constant)
    and decode.
    """
    assert self.has_periphery

    if delays is None:
        delays = sorted(list(self.derivatives))
    for delay in delays:
        assert delay in self.derivatives, delay

    dims = self.freqs.size
    total_dims = dims * (len(delays) + 1)
    phoneme_in = nengo.Ensemble(neurons_per_d * total_dims,
                                dimensions=total_dims)
    nengo.Connection(self.an.output, phoneme_in[:dims])
    for i, delay in enumerate(delays):
        nengo.Connection(self.derivatives[delay],
                         phoneme_in[(i+1)*dims:(i+2)*dims])
    out_dims = targets.shape[1]
    phoneme_out = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
    nengo.Connection(phoneme_in, phoneme_out,
                     **nengo.utils.connection.target_function(
                         eval_points, targets))
    return phoneme_in, phoneme_out


def HierarchicalPhonemeDetector(neurons_per_d, eval_points, targets,
                                delays=None, pool=4):
    """A hierarchical implementation of phoneme detection.

    We first make some intermediate layers that compress data from
    a few nearby frequencies; then, we use those compressed
    representations for phoneme detection.
    """
    assert self.has_periphery

    if delays is None:
        delays = sorted(list(self.derivatives))
    for delay in delays:
        assert delay in self.derivatives, delay

    assert self.freqs.size % pool == 0

    pooled_dims = self.freqs.size // pool
    total_dims = pooled_dims * (len(delays) + 1)

    # Pool raw AN responses
    an_pooled = nengo.Ensemble(neurons_per_d * pooled_dims,
                               dimensions=pooled_dims,
                               radius=pool)
    for i in range(pool):
        # print np.arange(self.freqs.size)[i::pool] to see what's happening
        nengo.Connection(self.an.output[i::pool], an_pooled)

    # Pool deriv responses
    d_pools = []
    for i, delay in enumerate(delays):
        deriv_pool = nengo.Ensemble(neurons_per_d * pooled_dims,
                                    dimensions=pooled_dims,
                                    radius=pool)
        for i in range(pool):
            nengo.Connection(self.derivatives[delay][i::pool], deriv_pool)
        d_pools.append(deriv_pool)

    phoneme_in = nengo.Ensemble(neurons_per_d * total_dims,
                                dimensions=total_dims,
                                radius=pool)
    nengo.Connection(an_pooled, phoneme_in[:pooled_dims])
    for i, deriv_pool in enumerate(d_pools):
        nengo.Connection(deriv_pool,
                         phoneme_in[(i+1)*pooled_dims:(i+2)*pooled_dims])
    out_dims = targets.shape[1]
    phoneme_out = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
    nengo.Connection(phoneme_in, phoneme_out,
                     **nengo.utils.connection.target_function(
                         eval_points, targets))
    return phoneme_in, phoneme_out
