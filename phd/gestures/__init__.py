import os

root = os.path.abspath(os.path.dirname(__file__))


def ges_path(*components):
    components = (root,) + components
    return os.path.join(*components)
