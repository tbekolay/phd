import errno
import hashlib
import os
from datetime import datetime

import dill

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_dir = os.path.join(root_dir, "cache")
try:
    os.makedirs(cache_dir)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
        pass
    else:
        raise


def cache_file(key=None, ext='pkl'):
    # Default key is current timestamp
    key = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') if key is None else key
    return os.path.join(cache_dir, "%s.%s" % (key, ext))


def cache_file_exists(key=None, ext='pkl'):
    path = cache_file(key, ext)
    return os.path.exists(path)


def generic_key(obj):
    h = hashlib.sha1()
    h.update(dill.dumps(obj))
    return h.hexdigest()


def load_obj(key):
    path = cache_file(key)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as fp:
        gs = dill.load(fp)
    return gs


def cache_obj(obj, key=None):
    # Default key is a hash of the object (better than timestamp!)
    key = generic_key(obj) if key is None else key
    path = cache_file(key)
    with open(path, 'wb') as fp:
        print "Caching %s" % key
        dill.dump(obj, fp)
