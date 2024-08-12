from torchlet.engine import Element


def test_basic_operations():
    a = Element(2.0)
    b = Element(3.0)
    c = a + b
    assert c.data == 5.0, "Addition test failed"
    d = a * b
    assert d.data == 6.0, "Multiplication test failed"


def test_backward():
    a = Element(2.0)
    b = Element(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0, "Backward test failed for a"
    assert b.grad == 1.0, "Backward test failed for b"


if __name__ == "__main__":
    test_basic_operations()
    test_backward()
    print("All tests passed!")
