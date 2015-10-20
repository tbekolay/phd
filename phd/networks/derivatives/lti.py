from collections import namedtuple

import nengo
import numpy as np
from nengo.utils.compat import range
from nengo.utils.filter_design import cont2discrete
from scipy.linalg import solve_lyapunov
from scipy.misc import factorial, pade

LTI = namedtuple('LTI', ['a', 'b', 'c', 'd'])


def scale_lti(lti, radii=1.0):
    """Scales the system to give an effective radius of r to x."""
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim > 1:
        raise ValueError("radii (%s) must be a 1-D array or scalar" % radii)
    elif r.ndim == 0:
        r = np.ones(lti.a.shape[0]) * r
    a = lti.a / r[:, None] * r
    b = lti.b / r[:, None]
    c = lti.c * r
    return LTI(a, b, c, lti.d)


def ab_norm(lti):
    """Returns H2-norm of each component of x in the state-space.

    Equivalently, this is the H2-norm of each component of (A, B, I, 0).
    This gives the power of each component of x in response to white-noise
    input with uniform power.

    Useful for setting the radius of an ensemble array with continuous
    dynamics (A, B).
    """
    p = solve_lyapunov(lti.a, -np.dot(lti.b, lti.b.T))  # AP + PA^H = Q
    assert np.allclose(np.dot(lti.a, p)
                       + np.dot(p, lti.a.T)
                       + np.dot(lti.b, lti.b.T), 0)
    c = np.eye(len(lti.a))
    h2norm = np.dot(c, np.dot(p, c.T))
    # The H2 norm of (A, B, C) is sqrt(tr(CXC^T)), so if we want the norm
    # of each component in the state-space representation, we evaluate
    # this for each elementary vector C separately, which is equivalent
    # to picking out the diagonals
    return np.sqrt(h2norm[np.diag_indices(len(h2norm))])


def lti2sim(lti, synapse, dt=0):
    """Maps a state-space LTI to the synaptic dynamics on A and B."""
    if not isinstance(synapse, nengo.Lowpass):
        raise TypeError("synapse (%s) must be Lowpass" % (synapse,))
    if dt <= 1e-6:
        a = np.identity(lti.a.shape[0]) + synapse.tau * lti.a
        b = lti.b * synapse.tau
    else:
        a, b, c, d, _ = cont2discrete(lti, dt=dt)
        aa = np.exp(-dt / synapse.tau)
        a = 1. / (1 - aa) * (a - aa * np.identity(a.shape[0]))
        b = 1. / (1 - aa) * b
    return LTI(a, b, c, d)


def exp_delay(p, q, c=1.0):
    """Returns F = p/q such that F(s) = e^(-sc)."""
    i = np.arange(p + q) + 1
    taylor = np.append([1.0], (-c)**i / factorial(i))
    return pade(taylor, q)


def multidim_lti(lti, dimensions):
    degree = lti.a.shape[0]
    a = np.zeros((degree * dimensions, degree * dimensions))
    b = np.zeros((degree * dimensions, dimensions))
    c = np.zeros((dimensions, degree * dimensions))
    d = None if lti.d is None else np.zeros(dimensions)

    # Replicate the existing system in the right blocks
    for dim in range(dimensions):
        blk = slice(dim * degree, (dim+1) * degree)
        a[blk, blk] = lti.a
        b[blk, dim] = lti.b[:, 0]
        c[dim, blk] = lti.c
        if lti.d is not None:
            d[dim] = lti.d
    return LTI(a, b, c, d)
