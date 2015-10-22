import nengo
from nengo.dists import Choice, Uniform

from ..processes import AuditoryFilterBank


def AuditoryPeriphery(freqs, sound_process, auditory_filter,
                      neurons_per_freq=12, fs=50000.,
                      middle_ear=False, zhang_synapse=False, net=None):
    if net is None:
        net = nengo.Network(label="Auditory Periphery")

    net.freqs = freqs
    net.sound_process = sound_process
    net.auditory_filter = auditory_filter
    net.fs = fs

    with net:
        # Inner hair cell activity
        net.fb = AuditoryFilterBank(freqs, sound_process, auditory_filter,
                                    samplerate=fs, zhang_synapse=zhang_synapse)
        net.ihc = nengo.Node(output=net.fb, size_out=freqs.size)

        # Cochlear neurons projecting down auditory nerve
        net.an = nengo.networks.EnsembleArray(neurons_per_freq, freqs.size,
                                              intercepts=Uniform(0.2, 0.8),
                                              encoders=Choice([[1]]))
        if zhang_synapse:
            nengo.Connection(net.ihc, net.an.input,
                             transform=0.1, synapse=None)
        else:
            # TODO different filters may give different magnitude output?
            nengo.Connection(net.ihc, net.an.input)
    return net
