import nengo
import numpy as np
from nengo import spa
from nengo.dists import Choice, Exponential
from nengo.networks import CircularConvolution, InputGatedMemory
from nengo.utils.compat import range

from .dmp import radial_f


def SyllableSequence(n_per_d, syllable_d, syllables,
                     difference_gain=15, n_positions=7,
                     threshold_memories=True, add_default_output=True,
                     rng=np.random, net=None):
    if net is None:
        net = spa.SPA("Syllable sequence")
    assert isinstance(net, spa.SPA), "Network must be a SPA instance."

    vocab = spa.Vocabulary(dimensions=syllable_d,
                           rng=rng,
                           unitary=['INC', 'POS1'])
    vocab.parse('INC + POS1')
    for i in range(2, n_positions+1):
        vocab.add('POS%d' % i, vocab.parse('POS%d * INC' % (i-1)))
    vocab.parse(" + ".join(syllables))
    syll_vocab = vocab.create_subset(syllables)
    net.vocab = vocab

    with net:
        # --- Input sequence
        net.sequence = spa.State(dimensions=syllable_d,
                                 neurons_per_dimension=n_per_d)

        # --- Input gated memories iterate through position vectors
        net.pos_curr = InputGatedMemory(
            n_per_d, dimensions=syllable_d, difference_gain=difference_gain)
        net.pos_next = InputGatedMemory(
            n_per_d, dimensions=syllable_d, difference_gain=difference_gain)
        # Switch from current to next with reciprocal connections
        nengo.Connection(net.pos_curr.output, net.pos_next.input)
        nengo.Connection(net.pos_next.output, net.pos_curr.input,
                         transform=vocab['INC'].get_convolution_matrix())
        # Switching depends on gate; get gate input elsewhere
        net.gate = nengo.Node(size_in=1)
        # pos gets gate
        nengo.Connection(net.gate, net.pos_curr.gate, synapse=None)
        # pos_next gets 1 - gate
        net.gate_bias = nengo.Node(output=1)
        nengo.Connection(net.gate_bias, net.pos_next.gate, synapse=None)
        nengo.Connection(net.gate, net.pos_next.gate,
                         transform=-1, synapse=None)

        # --- Get syllables with unbinding
        net.bind = CircularConvolution(
            n_per_d, syllable_d, invert_a=False, invert_b=True)
        net.bind_next = CircularConvolution(
            n_per_d, syllable_d, invert_a=False, invert_b=True)
        nengo.Connection(net.sequence.output, net.bind.A)
        nengo.Connection(net.sequence.output, net.bind_next.A)
        nengo.Connection(net.pos_curr.output, net.bind.B)
        nengo.Connection(net.pos_next.output, net.bind_next.B,
                         transform=vocab['INC'].get_convolution_matrix())

        # --- Clean up noisy syllables with associative memories
        net.syllable = spa.AssociativeMemory(
            syll_vocab, wta_output=True, threshold_output=threshold_memories)
        net.syllable_next = spa.AssociativeMemory(
            syll_vocab, wta_output=True, threshold_output=threshold_memories)
        nengo.Connection(net.bind.output, net.syllable.input)
        nengo.Connection(net.bind_next.output, net.syllable_next.input)
        if add_default_output:
            default = np.zeros(syllable_d)
            net.syllable.am.add_default_output_vector(default)
            net.syllable_next.am.add_default_output_vector(default)

    return net


def Sequencer(n_per_d, timer_tau=0.05, timer_freq=2.,
              reset_time=0.7, reset_threshold=0.5, reset_to_gate=-0.7,
              gate_threshold=0.4, net=None):
    if net is None:
        net = nengo.Network(label="Syllable sequencer")

    omega = timer_freq * timer_tau * 2 * np.pi

    with net:
        # --- Aggregate all the syllable DMPs to keep track of time
        net.timer = nengo.Ensemble(n_per_d * 2, dimensions=2)

        # --- Reset all DMPs at the right time
        r_intercepts = Exponential(0.15, reset_threshold, 1)
        net.reset = nengo.Ensemble(60, dimensions=1,
                                   encoders=Choice([[1]]),
                                   intercepts=r_intercepts)

        def reset_f(x):
            if x > reset_time:
                return 1.
            elif x < 0.05:
                return 0.8
            else:
                return 0
        nengo.Connection(net.timer, net.reset, function=radial_f(reset_f))

        # There's a catch 22 here. The reset forces the DMPs to a specific
        # point in state space, but that space is also where the timer starts
        # resetting DMPs. If we do nothing, then the DMPs will just stop.
        # To keep things moving, when the reset is active, we allow the
        # timer to oscillate on its own, rather than just using DMP input.

        # --- Make `timer` oscillate when reset is active
        net.timer_recur = nengo.Ensemble(n_per_d * 2, dimensions=2)
        nengo.Connection(net.timer, net.timer_recur)
        nengo.Connection(net.timer_recur, net.timer,
                         synapse=timer_tau,
                         transform=[[1, -omega], [omega, 1]])
        # Usually, the recurrent ensemble is inhibited
        inh_intercepts = Exponential(0.15, -0.5, 0.1)
        net.tr_inhibit = nengo.Ensemble(20, dimensions=1,
                                        encoders=Choice([[1]]),
                                        intercepts=inh_intercepts)
        nengo.Connection(net.tr_inhibit.neurons, net.timer_recur.neurons,
                         transform=-np.ones((n_per_d * 2, 20)))
        # But the reset disinhibits the recurrence
        nengo.Connection(net.reset, net.tr_inhibit, transform=-1)

        # --- `gate` switches the working memory representations
        g_intercepts = Exponential(0.15, gate_threshold, 1)
        net.gate = nengo.Ensemble(60, dimensions=1,
                                  encoders=Choice([[1]]),
                                  intercepts=g_intercepts)
        # Gate is normally always active
        net.gate_bias = nengo.Node(output=1)
        nengo.Connection(net.gate_bias, net.gate, synapse=None)
        # Reset inhibits gate, causing switch to next syllable
        nengo.Connection(net.reset, net.gate, transform=reset_to_gate)

    return net
