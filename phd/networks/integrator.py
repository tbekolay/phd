def Integrator(input_ensemble, delay,
               n_neurons=30, tau_highpass=0.05, net=None):
    if net is None:
        net = nengo.Network(label="Derivative")

    with net:
        net.deriv = nengo.Node(size_in=self.freqs.size)
        for i, freq in enumerate(self.freqs):
            diff_in, diff_out, _ = derivative(
                n_neurons, delay=delay, tau=tau_highpass, radius=0.1)
            nengo.Connection(self.an.output[i], diff_in)
            nengo.Connection(diff_out, deriv[i], synapse=tau_highpass)
        self.derivatives[delay] = deriv
