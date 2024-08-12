import shutil
import subprocess

import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph


def check_graphviz_installed():
    """Checks if Graphviz is installed on the system."""
    graphviz_path = shutil.which("dot")
    if graphviz_path is None:
        return False
    try:
        subprocess.run(["dot", "-V"], capture_output=True, text=True)
        return True
    except Exception:
        return False


def trace(root):
    """Builds a set of all nodes and edges in a graph."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def format_value(value):
    """Formats the value for display, handling None values."""
    return f"{value:.4f}" if value is not None else "None"


def draw_matplotlib(root):
    """Draws the computational graph of an Element using Matplotlib.

    Args:
        root (Element): The root Element of the computational graph.
    """
    nodes, edges = trace(root)
    graph = nx.DiGraph()

    # Add nodes and edges to the graph
    for n in nodes:
        value = format_value(n.data)
        gradient = format_value(n.grad)
        label = n.label if n.label is not None else ""
        node_label = f"{label}\nValue: {value}\nGrad: {gradient}"
        graph.add_node(n, label=node_label)
        for child in n._prev:
            graph.add_edge(child, n)

    pos = nx.spring_layout(graph)

    # Draw nodes
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=2000,
        node_color="lightblue",
        node_shape="o",
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={n: n._op for n in nodes if n._op},
        font_size=10,
        font_color="red",
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={
            n: f"{n.label}\nValue: {format_value(n.data)}\nGrad: {format_value(n.grad)}"
            for n in nodes
        },
        font_size=8,
        font_color="black",
    )

    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="->", arrowsize=10)

    plt.title("Computational Graph")
    plt.show()


def draw_graphviz(root):
    """Draws the computational graph of an Element using Graphviz.

    Args:
        root (Element): The root Element of the computational graph.
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        value = f"{n.data:.4f}" if n.data is not None else "None"
        gradient = f"{n.grad:.4f}" if n.grad is not None else "None"
        # Create a rectangular node for each element in the graph
        dot.node(
            name=uid,
            label=f"{{ {n._op} | Value: {value} | Grad: {gradient} }}",
            shape="record",
        )
        if n._op:
            # If this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
