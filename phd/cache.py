import hashlib
import os
import warnings
from datetime import datetime

import dill
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_dir = os.path.join(root_dir, "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


def cache_file(key=None, ext='pkl', subdir=None):
    # Default key is current timestamp
    dir_ = cache_dir if subdir is None else os.path.join(cache_dir, subdir)
    key = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') if key is None else key
    return os.path.join(dir_, "%s.%s" % (key, ext))


def cache_file_exists(key=None, ext='pkl', subdir=None):
    path = cache_file(key, ext, subdir)
    return os.path.exists(path)


def generic_key(obj):
    h = hashlib.sha1()
    h.update(dill.dumps(obj))
    return h.hexdigest()


def load_obj(key, ext='pkl', subdir=None):
    path = cache_file(key, ext, subdir)
    if not os.path.exists(path):
        warnings.warn("'%s' not found." % path)
        return None
    if ext == 'npz':
        return np.load(path)
    with open(path, 'rb') as fp:
        obj = dill.load(fp)
    return obj


def cache_obj(obj, key=None, ext='pkl', subdir=None):
    # Default key is a hash of the object (better than timestamp!)
    key = generic_key(obj) if key is None else key
    path = cache_file(key, ext, subdir)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if ext == 'npz' and isinstance(obj, dict):
        # Special case for saving npz files
        np.savez(path, **obj)
    else:
        with open(path, 'wb') as fp:
            dill.dump(obj, fp)
