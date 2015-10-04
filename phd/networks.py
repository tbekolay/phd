"""Nengo networks"""
import brian as br
import brian.hears as bh
import nengo
import numpy as np
from brian.hears.filtering.tan_carney import ZhangSynapseRate
from nengo.utils.compat import is_number, range
from scipy.signal import resample

from .filters import exp_delay, LTI


class NengoSound(bh.BaseSound):
    def __init__(self, step_f, nchannels, samplerate):
        self.step_f = step_f
        self.nchannels = nchannels
        self.samplerate = samplerate
        self.t = 0.0
        self.dt = 1. / self.samplerate

    def buffer_init(self):
        pass

    def buffer_fetch(self, start, end):
        return self.buffer_fetch_next(end - start)

    def buffer_fetch_next(self, samples):
        out = np.empty((samples, self.nchannels))
        for i in range(samples):
            self.t += self.dt
            out[i] = self.step_f(self.t)
        return out


class AuditoryFilterBank(nengo.processes.Process):
    def __init__(self, freqs, sound_process, filterbank, samplerate,
                 middle_ear=False, zhang_synapse=False):
        self.freqs = freqs
        self.sound_process = sound_process
        self.filterbank = filterbank
        self.samplerate = samplerate
        self.middle_ear = middle_ear
        self.zhang_synapse = zhang_synapse

    @staticmethod
    def bm2ihc(x):
        """Half wave rectify and compress it with a 1/3 power law."""
        return 3 * np.clip(x, 0, np.inf) ** (1. / 3.)

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        assert size_out == self.freqs.size

        # If samplerate isn't specified, we'll assume dt
        samplerate = 1. / dt if self.samplerate is None else self.samplerate
        sound_dt = 1. / samplerate

        # Set up the sound
        step_f = self.sound_process.make_step(0, 1, sound_dt, rng)
        ns = NengoSound(step_f, 1, samplerate)

        if self.middle_ear:
            ns = bh.MiddleEar(ns, gain=1)

        self.filterbank.source = ns

        duration = int(dt / sound_dt)
        self.filterbank.buffersize = duration
        ihc = bh.FunctionFilterbank(self.filterbank, self.bm2ihc)
        # Fails if we don't do this...
        ihc.cached_buffer_end = 0

        if self.zhang_synapse:
            syn = ZhangSynapseRate(ihc, self.freqs)
            s_mon = br.RecentStateMonitor(
                syn, 's', record=True, clock=syn.clock, duration=dt * br.second)
            net = br.Network(syn, s_mon)

            def step_synapse(t):
                net.run(dt * br.second)
                return s_mon.values[-1]
            return step_synapse
        else:
            def step_filterbank(t, startend=np.array([0, duration], dtype=int)):
                result = ihc.buffer_fetch(startend[0], startend[1])
                startend += duration
                return result[-1]
            return step_filterbank


def periphery(freqs, sound, filterbank, neurons_per_freq=12, fs=50000.,
              middle_ear=True, zhang_synapse=False):
    # Inner hair cell activity
    fb = AuditoryFilterBank(freqs, sound, filterbank,
                            samplerate=fs, zhang_synapse=zhang_synapse)
    ihc = nengo.Node(output=fb, size_out=freqs.size)

    # Cochlear neurons projecting down auditory nerve
    an = nengo.networks.EnsembleArray(neurons_per_freq, freqs.size,
                                      intercepts=nengo.dists.Uniform(0.4, 0.8),
                                      encoders=nengo.dists.Choice([[1]]))
    if zhang_synapse:
        nengo.Connection(ihc, an.input, transform=0.1, synapse=None)
    else:
        nengo.Connection(ihc, an.input)
    return ihc, an


# def product(n_neurons, dimensions, input_magnitude=1):
#     encoders = nengo.dists.Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]])

#     product = EnsembleArray(n_neurons, n_ensembles=dimensions,
#                             ens_dimensions=2,
#                             encoders=encoders,
#                             radius=input_magnitude * np.sqrt(2))
#     with product:
#         product.A = nengo.Node(size_in=dimensions, label="A")
#         product.B = nengo.Node(size_in=dimensions, label="B")
#         nengo.Connection(A, net.product.input[::2], synapse=None)
#         nengo.Connection(B, net.product.input[1::2], synapse=None)

#         # Remove default output
#         output = product.output
#         for conn in list(product.connections):
#             if conn.post is product.output:
#                 product.connections.remove(conn)
#         product.nodes.remove(product.output)

#         # Add product output
#         product.output = product.add_output(
#             'product', lambda x: x[0] * x[1], synapse=None)

#     return product

def lti(n_neurons, lti_system, synapse=nengo.Lowpass(0.05),
        controlled=False, dt=0.001, radii=None, radius=1.0):
    if radii is None:
        radii = lti_system.ab_norm()
    radii *= radius
    lti_system.scale_to(radii)
    lti_system.to_sim(synapse, dt)

    size_in = lti_system.b.shape[1]
    size_state = lti_system.a.shape[0]
    size_out = lti_system.c.shape[0]

    inp = nengo.Node(size_in=size_in, label="input")
    out = nengo.Node(size_in=size_out, label="output")
    # if controlled:
    #     x = product(n_neurons, size_state)
    #     x_in = x.A
    # else:
    x = nengo.networks.EnsembleArray(n_neurons, size_state)
    x_in = x.input
    # end else
    x_out = x.output

    nengo.Connection(x_out, x_in, transform=lti_system.a, synapse=synapse)
    nengo.Connection(inp, x_in, transform=lti_system.b, synapse=synapse)
    nengo.Connection(x_out, out, transform=lti_system.c, synapse=None)
    nengo.Connection(inp, out, transform=lti_system.d, synapse=None)

    return inp, out


def deconvolution(n_neurons, tf, delay, degree=4, **lti_kwargs):
    """Approximate the inverse of a given transfer function using a delay."""
    num, den = [np.poly1d(tf[0]), np.poly1d(tf[1])]
    order = len(den) - len(num)
    if order >= degree:
        raise ValueError("order (%d) must be < degree (%d)"
                         % (order, degree))
    edp, edq = exp_delay(degree - order, degree, delay)
    p, q = np.polymul(edp, den), np.polymul(edq, num)
    inp, out = lti(n_neurons, LTI.from_tf(p, q), **lti_kwargs)
    return inp, out, degree



def derivative(n_neurons, tau, delay, **deconv_kwargs):
    """Output a signal that is a derivative of the input."""
    return deconvolution(n_neurons, ([1], [tau, 0]), delay, **deconv_kwargs)
