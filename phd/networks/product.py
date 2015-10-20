"""An alternative network to Nengo's product."""

import nengo
import numpy as np
from nengo.networks import EnsembleArray


def Product(n_neurons, dimensions, input_magnitude=1):
    encoders = nengo.dists.Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    product = EnsembleArray(n_neurons, n_ensembles=dimensions,
                            ens_dimensions=2,
                            encoders=encoders,
                            radius=input_magnitude * np.sqrt(2))
    with product:
        product.A = nengo.Node(size_in=dimensions, label="A")
        product.B = nengo.Node(size_in=dimensions, label="B")
        nengo.Connection(product.A, product.input[::2], synapse=None)
        nengo.Connection(product.B, product.input[1::2], synapse=None)

        # Remove default output
        for conn in list(product.connections):
            if conn.post is product.output:
                product.connections.remove(conn)
        product.nodes.remove(product.output)

        # Add product output
        product.output = product.add_output(
            'product', lambda x: x[0] * x[1], synapse=None)

    return product
