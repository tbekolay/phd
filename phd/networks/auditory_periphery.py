import nengo
from nengo.dists import Choice, Uniform

from ..processes import AuditoryFilterBank


def AuditoryPeriphery(freqs, sound_process, auditory_filter,
                      neurons_per_freq=12, fs=50000.,
                      middle_ear=False, net=None):
    if net is None:
        net = nengo.Network(label="Auditory Periphery")

    net.freqs = freqs
    net.sound_process = sound_process
    net.auditory_filter = auditory_filter
    net.fs = fs

    with net:
        # Inner hair cell activity
        net.fb = AuditoryFilterBank(
            freqs, sound_process, auditory_filter, samplerate=fs)
        net.ihc = nengo.Node(output=net.fb, size_out=freqs.size)

        # Cochlear neurons projecting down auditory nerve
        net.an = nengo.networks.EnsembleArray(neurons_per_freq, freqs.size,
                                              intercepts=Uniform(-0.1, 0.5),
                                              encoders=Choice([[1]]))
        # TODO different filters may give different magnitude output?
        nengo.Connection(net.ihc, net.an.input)
    return net
