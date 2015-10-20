import nengo
from nengo.dists import Choice, Uniform

from ..processes import AuditoryFilterBank


def AuditoryPeriphery(freqs, sound, auditory_filter,
                      neurons_per_freq=12, fs=50000.,
                      middle_ear=False, zhang_synapse=False, net=None):
    if net is None:
        net = nengo.Network(label="Auditory Periphery")

    net.freqs = freqs
    net.sound = sound
    net.auditory_filter = auditory_filter
    net.fs = fs

    with net:
        # Inner hair cell activity
        fb = AuditoryFilterBank(freqs, sound, filterbank,
                                samplerate=fs, zhang_synapse=zhang_synapse)
        self.ihc = nengo.Node(output=fb, size_out=freqs.size)

        # Cochlear neurons projecting down auditory nerve
        self.an = nengo.networks.EnsembleArray(neurons_per_freq, freqs.size,
                                               intercepts=Uniform(0.2, 0.8),
                                               encoders=Choice([[1]]))
        if zhang_synapse:
            nengo.Connection(self.ihc, self.an.input,
                             transform=0.1, synapse=None)
        else:
            # TODO different filters may give different magnitude output?
            nengo.Connection(self.ihc, self.an.input)
    return net
