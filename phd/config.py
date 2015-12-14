"""Global configuration stuff.

In a Serious Package, this would be done with RC settings and whatnot,
but let's not get head of ourselves. This will do.
"""
import os

cache_dir = os.path.expanduser("~/Code/phd/cache")
log_experiments = True
timit_root = os.path.expanduser("~/phd_data/timit")
