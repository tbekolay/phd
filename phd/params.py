import brian.hears as bh
from nengo.params import (
    BoolParam,
    DictParam,
    IntParam,
    is_param,
    NdarrayParam,
    NumberParam,
    Parameter,
    StringParam,
)
from nengo.processes import ProcessParam


class ListParam(Parameter):
    def validate(self, instance, lst):
        if lst is not None and not isinstance(lst, list):
            raise ValueError("Must be a list; got '%s'" % str(lst))
        super(ListParam, self).validate(instance, lst)


class BrianFilterParam(Parameter):
    def validate(self, instance, filt):
        if filt is not None and not isinstance(filt, bh.Filterbank):
            raise ValueError(
                "Must be a Brian Filterbank; got '%s'" % str(filt))
        super(BrianFilterParam, self).validate(instance, filt)
