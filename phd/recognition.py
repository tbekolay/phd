import nengo
import numpy as np
from nengo.utils.compat import is_iterable
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

        # Integrator stuff
        self.integrators = {}

    @property
    def fs(self):
        assert self.has_periphery
        return self.ihc.output.samplerate

    @property
    def neurons_per_freq(self):
        assert self.has_periphery
        return self.an.n_neurons

    @property
    def has_periphery(self):
        return self.freqs is not None

    @with_self
    def add_periphery(self, freqs, sound, auditory_filter,
                      neurons_per_freq=12, fs=50000.,
                      middle_ear=False, zhang_synapse=False):
        assert not self.has_periphery
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
    def add_derivative(self, n_neurons, delay, tau_highpass=0.05):
        assert self.has_periphery
        assert delay not in self.derivatives, "That derivative already exists."

        deriv = nengo.Node(size_in=self.freqs.size)
        for i, freq in enumerate(self.freqs):
            diff_in, diff_out, _ = derivative(
                n_neurons, delay=delay, tau=tau_highpass, radius=0.1)
            nengo.Connection(self.an.output[i], diff_in)
            nengo.Connection(diff_out, deriv[i], synapse=tau_highpass)
        self.derivatives[delay] = deriv

    @with_self
    def add_integrator(self, n_neurons, tau):
        """Not really an integrator, just a long time constant on input."""
        assert self.has_periphery

        integrator = nengo.networks.EnsembleArray(n_neurons, self.freqs.size)
        self.integrators[tau] = integrator.output
        nengo.Connection(self.an.output, integrator.input,
                         synapse=nengo.Lowpass(tau))

    @with_self
    def add_phoneme_detector(self, neurons_per_d, eval_points, targets,
                             delays=None):
        """A naive implementation for phoneme detection.

        We compress all our frequency and temporal information into
        a single ensemble (one for vowels, with a long time constant,
        and one for consonants, with a short time constant)
        and decode.
        """
        assert self.has_periphery

        if delays is None:
            delays = sorted(list(self.derivatives))
        for delay in delays:
            assert delay in self.derivatives, delay

        dims = self.freqs.size
        total_dims =  dims * (len(delays) + 1)
        phoneme_in = nengo.Ensemble(neurons_per_d * total_dims,
                                    dimensions=total_dims)
        nengo.Connection(self.an.output, phoneme_in[:dims])
        for i, delay in enumerate(delays):
            nengo.Connection(self.derivatives[delay],
                             phoneme_in[(i+1)*dims:(i+2)*dims])
        out_dims = targets.shape[1]
        phoneme_out = nengo.Ensemble(neurons_per_d * out_dims, out_dims)
        nengo.Connection(phoneme_in, phoneme_out,
                         **nengo.utils.connection.target_function(
                             eval_points, targets))
        return phoneme_in, phoneme_out
