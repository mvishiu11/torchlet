import numpy as np


class Element:
    """Stores a scalar or vector and its gradient.

    Attributes:
        data (np.ndarray): The value of the element.
        grad (np.ndarray or None): The gradient of the element, initialized to None.
        _backward (function): The function to compute the backward pass.
        _prev (set): Set of parent elements in the computation graph.
        _op (str): The operation that produced this element.
    """

    def __init__(self, data, _children=(), _op="") -> None:
        """Initializes an Element with data and optional gradient.

        Args:
            data (float or np.ndarray): The scalar or vector data.
            _children (tuple, optional): Parent elements in the computation graph.
            _op (str, optional): The operation that produced this element.
        """
        self.data = (
            np.array(data, dtype=np.float64)
            if not isinstance(data, np.ndarray)
            else data.astype(np.float64)
        )
        self.grad = None  # Lazy initialization of gradients
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _ensure_grad_initialized(self) -> None:
        """Ensures that the gradient is initialized."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    def __add__(self, other) -> "Element":
        """Performs addition with another Element or scalar.

        Args:
            other (Element or float): The element or scalar to add.

        Returns:
            Element: A new Element representing the result.
        """
        other = other if isinstance(other, Element) else Element(other)
        out = Element(self.data + other.data, (self, other), "+")

        def _backward():
            self._ensure_grad_initialized()
            other._ensure_grad_initialized()
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other) -> "Element":
        """Performs multiplication with another Element or scalar.

        Args:
            other (Element or float): The element or scalar to multiply.

        Returns:
            Element: A new Element representing the result.
        """
        other = other if isinstance(other, Element) else Element(other)
        out = Element(self.data * other.data, (self, other), "*")

        def _backward():
            self._ensure_grad_initialized()
            other._ensure_grad_initialized()
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other) -> "Element":
        """Performs exponentiation with int or scalar. Does not support Element for now.

        Args:
            other (int or float): The element or scalar to raise to the power of.

        Returns:
            Element: A new Element representing the result.
        """
        assert isinstance(
            other, (int, float)
        ), "Torchlet only supports int/float powers for now"
        out = Element(self.data**other, (self,), f"**{other}")

        def _backward():
            self._ensure_grad_initialized()
            other._ensure_grad_initialized()
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Element":
        """Applies the ReLU activation function to the Element.

        Returns:
            Element: A new Element representing the result.
        """
        out = Element(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self._ensure_grad_initialized()
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        """Computes the gradient of the Element via backpropagation (via reverse-mode autodiff on dynamic DAG)."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        """Returns a string representation of the Element."""
        return f"Element(data={self.data}, grad={self.grad})"

    def __getitem__(self, idx) -> "Element":
        """Allows subscripting of the Element, returns an Element for the item at idx.

        Args:
            idx (int): Index to access the data.

        Returns:
            Element: A new Element for the indexed data.
        """
        return Element(self.data[idx], _children=(self,), _op="getitem")

    def __setitem__(self, idx, value) -> None:
        """Allows setting item at idx for the Element's data.

        Args:
            idx (int): Index to access the data.
            value (Element or float): The value to set.
        """
        self.data[idx] = value.data if isinstance(value, Element) else value

    def __neg__(self) -> "Element":
        """Negates the Element.

        Returns:
            Element: A new Element representing the negation.
        """
        return self * -1

    def __radd__(self, other) -> "Element":
        """Performs addition with another Element or scalar. This is the reverse fallback for addition.

        Args:
            other (Element or float): The element or scalar to add.

        Returns:
            Element: A new Element representing the result.
        """
        return self + other

    def __sub__(self, other) -> "Element":
        """Performs subtraction with another Element or scalar.

        Args:
            other (Element or float): The element or scalar to subtract.

        Returns:
            Element: A new Element representing the result.
        """
        return self + (-other)

    def __rsub__(self, other) -> "Element":
        """Performs subtraction with another Element or scalar. This is the reverse fallback for subtraction.

        Args:
            other (Element or float): The element or scalar to subtract.

        Returns:
            Element: A new Element representing the result.
        """
        return other + (-self)

    def __rmul__(self, other) -> "Element":
        """Performs multiplication with another Element or scalar. This is the reverse fallback for multiplication.

        Args:
            other (Element or float): The element or scalar to multiply.

        Returns:
            Element: A new Element representing the result.
        """
        return self * other

    def __truediv__(self, other) -> "Element":
        """Performs division with another Element or scalar.

        Args:
            other (Element or float): The element or scalar to divide.

        Returns:
            Element: A new Element representing the result.
        """
        return self * other**-1

    def __rtruediv__(self, other) -> "Element":
        """Performs division with another Element or scalar. This is the reverse fallback for division.

        Args:
            other (Element or float): The element or scalar to divide.

        Returns:
            Element: A new Element representing the result.
        """
        return other * self**-1
