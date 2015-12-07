"""Tools for doing work in parallel."""

from multiprocess import Pool


def initializer():
    from phd.experiments import ncc
    assert ncc
    # Add any other things needed here


def get_pool():
    global pool
    if pool is None:
        pool = Pool(initializer=initializer)
    return pool


pool = None
