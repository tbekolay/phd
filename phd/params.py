from nengo.params import (  # noqa: F401
    BoolParam,
    DictParam,
    IntParam,
    is_param,
    NdarrayParam,
    NumberParam,
    Parameter,
    StringParam,
)
from nengo.base import ProcessParam  # noqa: F401


class ListParam(Parameter):
    def validate(self, instance, lst):
        if lst is not None and not isinstance(lst, list):
            raise ValueError("Must be a list; got '%s'" % str(lst))
        super(ListParam, self).validate(instance, lst)
