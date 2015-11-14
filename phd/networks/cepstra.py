"""A cepstrum is the inverse Fourier transform of the compressed spectrum.

The auditory periphery gives us the compressed power spectrum of the incoming
audio signal already, so all that is left to do the cepstrum is to take
the inverse cosine transform. Easy!
"""

import numpy as np
import nengo


def idct(n, size_out):
    """Transform weights that compute the inverse discrete cosine transform.

    There are multiple types of DCTs, with and without normalization.
    This uses the typical DCT, i.e., type II, whose inverse is type III
    with some scaling and normalization factors.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Discrete_cosine_transform
    """
    k = np.arange(n)
    s = np.ones(n)
    s[0] = np.sqrt(0.5)
    idct_matrix = (np.sqrt(2. / n) * s
                   * np.cos(np.pi * np.outer(k + 0.5, k) / n))
    return idct_matrix[:size_out]



def Cepstra(size_in, size_out, net=None):
    if net is None:
        net = nengo.Network("Cepstra")

    with net:
        net.input = nengo.Node(size_in=size_in, label="input")
        net.output = nengo.Node(size_in=size_out, label="output")
        nengo.Connection(net.input, net.output,
                         synapse=None, transform=idct(size_in, size_out))
    return net
