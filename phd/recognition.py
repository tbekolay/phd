import nengo
import numpy as np
from nengo.utils.network import with_self

from .networks import derivative, periphery


class SpeechRecognition(nengo.Network):
    def __init__(self, *args, **kwargs):
        super(SpeechRecognition, self).__init__(*args, **kwargs)

        # Periphery stuff
        self.freqs = None
        self.sound = None
        self.auditory_filter = None
        self.ihc = None
        self.an = None

        # Derivative stuff
        self.derivatives = {}

    @property
    def fs(self):
        assert self.ihc is not None, "Periphery not added yet"
        return self.ihc.output.samplerate

    @property
    def neurons_per_freq(self):
        assert self.an is not None, "Periphery not added yet"
        return self.an.n_neurons

    @with_self
    def add_periphery(self, freqs, sound, auditory_filter,
                      neurons_per_freq=12, fs=50000.,
                      middle_ear=False, zhang_synapse=False):
        assert self.freqs is None, "Periphery already exists."
        self.freqs = freqs
        self.sound = sound
        self.auditory_filter = auditory_filter
        self.middle_ear = middle_ear
        self.zhang_synapse = zhang_synapse

        self.ihc, self.an = periphery(freqs,
                                      sound,
                                      auditory_filter,
                                      neurons_per_freq,
                                      fs,
                                      middle_ear,
                                      zhang_synapse)

    @with_self
    def probe_periphery(self):
        assert self.ihc and self.an, "Periphery not added yet"
        ihc_p = nengo.Probe(self.ihc, synapse=None)
        an_in_p = nengo.Probe(self.an.input, synapse=None)
        self.an.add_neuron_output()
        an_p = nengo.Probe(self.an.neuron_output, synapse=None)
        return ihc_p, an_in_p, an_p

    @with_self
    def add_derivative(self, n_neurons, delay, tau_highpass=0.05):
        assert self.freqs is not None, "Periphery must be added first."
        assert delay not in self.derivatives, "That derivative already exists."

        deriv = nengo.Node(size_in=self.freqs.size)
        for i, freq in enumerate(self.freqs):
            diff_in, diff_out, _ = derivative(
                n_neurons, delay=delay, tau=tau_highpass, radius=0.1)
            nengo.Connection(self.an.output[i], diff_in)
            nengo.Connection(diff_out, deriv[i], synapse=tau_highpass)
        self.derivatives[delay] = deriv

    @with_self
    def probe_derivative(self, delay, synapse=0.01):
        return nengo.Probe(self.derivatives[delay], synapse=synapse)
