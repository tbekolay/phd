import nengo


def DMP(rampspeed=0.25, net=None):
    if net is None:
        net = nengo.Network(label="DMP")

    with net:
        # Use a small oscillator as input to integrate precisely
        oscillator = nengo.Ensemble(500, dimensions=2, radius=0.01)
        tr = np.identity(2) + np.array([[1, -1], [1, 1]])
        nengo.Connection(oscillator, oscillator, transform=tr)

        # Reset the system and start the oscillator
        reset = nengo.Node(size_in=1)
        nengo.Connection(reset, oscillator, transform=np.ones((2, 1)))

        # Slowly ramp up (using a 1D ramp function rather than oscillator)
        ramp = nengo.Ensemble(1000, dimensions=1)
        nengo.Connection(ramp, ramp, synapse=0.1)
        # Integrate the oscillator input
        nengo.Connection(oscillator, ramp, transform=rampspeed,
                         function=lambda x: x[0] + 0.5)
        # Reset when requested
        nengo.Connection(reset, ramp.neurons,  # synapse=0.1 ?
                         transform=-np.ones((ramp.n_neurons, 1)))


def InverseDMP():
    pass
