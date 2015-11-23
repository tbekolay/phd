"""Implement dynamic time-warping (DTW).

DTW is used to modify feature vectors so that they are the same length,
making it possible to do SVM on those vectors.

The code is adapted from Pierre Rouanet's implementation of DTW available at
https://github.com/pierre-rouanet/dtw,
and used under the terms of the GPL license.
"""

from collections import deque

import numpy as np


def dtw(x, y, dist_f=lambda x, y: np.linalg.norm(x - y, ord=1)):
    """Performs dynamic time warping on two sequences.

    Parameters
    ----------
    x : array_like (N1, M)
        First input sequence.
    y : array_like (N1, M)
        Second input sequence.
    dist_f : function, optional
        Distance used as cost measure. Default: L1 norm

    Returns
    -------
    mindist : float
        The minimum distance between the two seqeuences.
    dist : np.array
        The accumulated cost matrix.
    path : np.array
        The warp path.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    r, c = x.shape[0], y.shape[0]

    dist = np.zeros((r + 1, c + 1))
    dist[0, 1:] = np.inf
    dist[1:, 0] = np.inf

    for i in range(r):
        for j in range(c):
            dist[i+1, j+1] = dist_f(x[i], y[j])

    for i in range(r):
        for j in range(c):
            dist[i+1, j+1] += min(dist[i, j], dist[i, j+1], dist[i+1, j])

    dist = dist[1:, 1:]

    mindist = dist[-1, -1] / sum(dist.shape)

    return mindist, dist, warppath(dist)


def warppath(dist):
    """Traces back through the distance array to get the warp path."""
    i, j = np.array(dist.shape) - 1
    p, q = deque([i]), deque([j])

    while i > 0 and j > 0:
        tb = np.argmin((dist[i-1, j-1], dist[i-1, j], dist[i, j-1]))

        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        elif tb == 2:
            j -= 1

        p.appendleft(i)
        q.appendleft(j)

    p.appendleft(0)
    q.appendleft(0)
    return np.array(p), np.array(q)
