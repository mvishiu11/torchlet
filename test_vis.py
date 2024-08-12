from torchlet.engine import Element

# Example of using the draw_matplotlib function
if __name__ == "__main__":
    a = Element(2.0, label="a")
    b = Element(3.0, label="b")
    c = a * b
    c.label = "c"
    d = c + a
    d.label = "d"

    d.visualize(
        method="graphviz"
    )  # method='matplotlib' for Matplotlib visualization from Micrograd
    d.backward()
    d.visualize()  # method='graphviz' for Graphviz visualization from Micrograd
