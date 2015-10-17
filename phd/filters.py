import brian as br
import brian.hears as bh
import nengo
import numpy as np
from nengo.utils.filter_design import zpk2ss, tf2ss, cont2discrete
from scipy.linalg import solve_lyapunov
from scipy.misc import factorial, pade

dummy_sound = bh.Sound(np.zeros(1))


def erbspace(low, high, n_freq):
    """Sample ERB distribution; low and high in Hz."""
    return bh.erbspace(low * br.Hz, high * br.Hz, n_freq)


def gammatone(freqs, b=1.019):
    return bh.Gammatone(dummy_sound, freqs, b=b)


def approximate_gammatone(freqs, bw=None, order=3):
    if bw is None:
        bw = 10 ** (0.037 + 0.785 * np.log10(freqs))
    return bh.ApproximateGammatone(dummy_sound, freqs, bw, order)


def log_gammachirp(freqs, glide_slope=-2.96, time_const=1.81):
    return bh.LogGammachirp(dummy_sound, freqs, c=glide_slope, b=time_const)


def linear_gammachirp(freqs, glide_slope=0.0, time_const=None):
    if time_const is None:
        time_const = np.linspace(3, 0.3, freqs.size) * br.ms
    return bh.LinearGammachirp(
        dummy_sound, freqs, time_constant=time_const, c=glide_slope)


def tan_carney(freqs, update_interval=1):
    return bh.TanCarney(dummy_sound, freqs, update_interval=update_interval)


def dual_resonance(freqs):
    return bh.DRNL(dummy_sound, freqs, type='human')


def compressive_gammachirp(freqs, update_interval=1):
    return bh.DCGC(dummy_sound, freqs, update_interval=update_interval)


class LTI(object):
    """Methods for dealing with LTI filters in Nengo.

    Adapted from Aaron Voelker's delay notebook at
    summerschool2015/tutorials/temprep/delay.ipynb
    """
    def __init__(self, a, b, c, d):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.d = np.array(d)

    @property
    def abcd(self):
        return (self.a, self.b, self.c, self.d)

    @classmethod
    def from_synapse(cls, synapse):
        """Instantiate class from a Nengo synapse."""
        if not hasattr(synapse, 'num') or not hasattr(synapse, 'den'):
            raise ValueError("Must be a linear filter with 'num' and 'den'")
        return cls(*tf2ss(synapse.num, synapse.den))

    @classmethod
    def from_tf(cls, num, den):
        """Instantiate class from a transfer function."""
        return cls(*tf2ss(num, den))

    @classmethod
    def from_zpk(cls, z, p, k):
        """Instantiate class from a zero-pole-gain representation."""
        return cls(*zpk2ss(z, p, k))

    def copy(self):
        return LTI(*self.abcd)

    def scale_to(self, radii=1.0):
        """Scales the system to give an effective radius of r to x."""
        r = np.asarray(radii, dtype=np.float64)
        if r.ndim > 1:
            raise ValueError(
                "radii (%s) must be a 1-D array or scalar" % radii)
        elif r.ndim == 0:
            r = np.ones(len(self.a)) * r
        self.a = self.a / r[:, None] * r
        self.b /= r[:, None]
        self.c *= r

    def ab_norm(self):
        """Returns H2-norm of each component of x in the state-space.

        Equivalently, this is the H2-norm of each component of (A, B, I, 0).
        This gives the power of each component of x in response to white-noise
        input with uniform power.

        Useful for setting the radius of an ensemble array with continuous
        dynamics (A, B).
        """
        p = solve_lyapunov(self.a, -np.dot(self.b, self.b.T))  # AP + PA^H = Q
        assert np.allclose(np.dot(self.a, p)
                           + np.dot(p, self.a.T)
                           + np.dot(self.b, self.b.T), 0)
        c = np.eye(len(self.a))
        h2norm = np.dot(c, np.dot(p, c.T))
        # The H2 norm of (A, B, C) is sqrt(tr(CXC^T)), so if we want the norm
        # of each component in the state-space representation, we evaluate
        # this for each elementary vector C separately, which is equivalent
        # to picking out the diagonals
        return np.sqrt(h2norm[np.diag_indices(len(h2norm))])

    def to_sim(self, synapse, dt=0):
        """Maps a state-space LTI to the synaptic dynamics on A and B."""
        if not isinstance(synapse, nengo.Lowpass):
            raise TypeError("synapse (%s) must be Lowpass" % (synapse,))
        if dt == 0:
            a = synapse.tau * self.a + np.eye(len(self.a))
            b = synapse.tau * self.b
        else:
            a, b, c, d, _ = cont2discrete(self.abcd, dt=dt)
            aa = np.exp(-dt / synapse.tau)
            a = 1. / (1 - aa) * (a - aa * np.eye(len(a)))
            b = 1. / (1 - aa) * b
        self.a, self.b, self.c, self.d = a, b, c, d


def exp_delay(p, q, c=1.0):
    """Returns F = p/q such that F(s) = e^(-sc)."""
    i = np.arange(p+q) + 1
    taylor = np.append([1.0], (-c)**i / factorial(i))
    return pade(taylor, q)
