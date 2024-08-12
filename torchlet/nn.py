import random

from torchlet.engine import Element


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Returns a list of all parameters.

        Returns:
            list: A list of parameters (Element objects).
        """
        return []


class Neuron(Module):
    """Represents a single neuron in the network.

    Args:
        nin (int): Number of input connections.
        nonlin (bool): Whether to apply a non-linear activation function.
    """

    def __init__(self, nin, nonlin=True):
        self.w = [Element(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Element(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Computes the output of the neuron for a given input.

        Args:
            x (list of Element): Input to the neuron.

        Returns:
            Element: The output of the neuron.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """Returns the parameters of the neuron.

        Returns:
            list: A list of parameters (Element objects).
        """
        return self.w + [self.b]

    def __repr__(self):
        """Returns a string representation of the Neuron."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """Represents a layer of neurons in the network.

    Args:
        nin (int): Number of input connections to each neuron.
        nout (int): Number of neurons in the layer.
    """

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Computes the output of the layer for a given input.

        Args:
            x (list of Element): Input to the layer.

        Returns:
            list or Element: The output of the layer.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Returns the parameters of the layer.

        Returns:
            list: A list of parameters (Element objects).
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """Returns a string representation of the Layer."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Represents a multi-layer perceptron (MLP) network.

    Args:
        nin (int): Number of input features.
        nouts (list of int): List of output sizes for each layer.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """Computes the output of the MLP for a given input.

        Args:
            x (list of Element): Input to the MLP.

        Returns:
            list or Element: The output of the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Returns the parameters of the MLP.

        Returns:
            list: A list of parameters (Element objects).
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """Returns a string representation of the MLP."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
