import nengo
import numpy as np
from nengo.dists import Exponential, Uniform


class UniformWithDeadzone(Uniform):
    def __init__(self, deadzone, low=-np.pi, high=np.pi):
        self.deadzone = deadzone
        super(UniformWithDeadzone, self).__init__(low, high)

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        sample = rng.uniform(low=self.low, high=self.high, size=shape)
        ix = sample > self.deadzone
        while np.sum(ix) > 0:
            n_resample = int(np.sum(ix))
            sample[ix] = rng.uniform(low=self.low,
                                     high=self.high,
                                     size=n_resample)
            ix = sample > self.deadzone
        return sample


def zone(deadzone):
    def _zone(x):
        theta = np.arctan2(x[1], x[0])
        if theta > deadzone:
            return [0, 0]
        return x
    return _zone


def radial_f(func, deadzone):
    # Compensate for the deadzone by giving theta * deadcomp to the function
    deadcomp = 1. + deadzone / (2 * np.pi)

    def _radial_f(x):
        theta = np.arctan2(x[1], x[0]) * deadcomp
        return func(theta / (2 * np.pi) + 0.5)
    _radial_f.__name__ = func.__name__
    return _radial_f


def DeadzoneDMP(n_per_d, freq, forcing_f, tau=0.025,
                deadzone=0.6 * np.pi, net=None):
    if net is None:
        net = nengo.Network(label="Rhythmic DMP")

    out_dims = forcing_f(0.).size
    omega = freq * tau * 2 * np.pi
    theta = UniformWithDeadzone(deadzone=deadzone).sample(n_per_d * 2, 1)
    encoders = np.hstack([np.cos(theta), np.sin(theta)])
    with net:
        # --- Decode forcing_f from oscillator
        net.osc = nengo.Ensemble(n_per_d * 2, dimensions=2,
                                 intercepts=Exponential(0.15, 0.3, 0.6),
                                 encoders=encoders,
                                 label=forcing_f.__name__)
        nengo.Connection(net.osc, net.osc,
                         synapse=tau,
                         function=zone(deadzone),
                         transform=[[1, -omega], [omega, 1]])
        net.output = nengo.Node(size_in=out_dims)
        nengo.Connection(net.osc, net.output,
                         function=radial_f(forcing_f, deadzone), synapse=None)

        # --- Kick start the oscillator
        net.kick = nengo.Node(size_in=1)
        nengo.Connection(
            net.kick, net.osc, transform=[[-1], [0.05]], synapse=None)
    return net
