
import torch

import pytest
from hypothesis import given, strategies as st, settings

from engine import Value


# Compare results with pytorch.


@settings(max_examples=20)
@given(
    st.floats(min_value=-1e4, max_value=1e4), st.floats(min_value=-1e4, max_value=1e4)
)
def test_add(x, y):
    x_eng = Value(x)
    y_eng = Value(y)
    z_eng = x_eng + y_eng
    z_eng.grad = 1.0
    z_eng._backward()

    x_torch = torch.tensor(x, requires_grad=True)
    x_torch.grad = None
    y_torch = torch.tensor(y, requires_grad=True)
    y_torch.grad = None
    z_torch = x_torch + y_torch
    z_torch.backward()

    assert pytest.approx(z_eng.data) == z_torch.item()
    assert pytest.approx(x_eng.grad) == x_torch.grad.item()
    assert pytest.approx(y_eng.grad) == y_torch.grad.item()


@settings(max_examples=20)
@given(
    st.floats(min_value=-1e4, max_value=1e4), st.floats(min_value=-1e4, max_value=1e4)
)
def test_mul(x, y):
    x_eng = Value(x)
    y_eng = Value(y)
    z_eng = x_eng * y_eng
    z_eng.grad = 1.0
    z_eng._backward()

    x_torch = torch.tensor(x, requires_grad=True)
    x_torch.grad = None
    y_torch = torch.tensor(y, requires_grad=True)
    y_torch.grad = None
    z_torch = x_torch * y_torch
    z_torch.backward()

    assert pytest.approx(z_eng.data) == z_torch.item()
    assert pytest.approx(x_eng.grad) == x_torch.grad.item()
    assert pytest.approx(y_eng.grad) == y_torch.grad.item()


@settings(max_examples=20)
@given(st.floats(min_value=-1e4, max_value=1e4))
def test_tanh(x):
    x_eng = Value(x)
    y_eng = x_eng.tanh()
    y_eng.grad = 1.0
    y_eng._backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = x_torch.tanh()
    x_torch.grad = None
    y_torch.backward()

    assert pytest.approx(y_eng.data) == y_torch.item()
    assert pytest.approx(x_eng.grad) == x_torch.grad.item()


# Bound the input values, again to reduce overflow and underflow problems.
@settings(max_examples=20)
@given(
    a=st.floats(min_value=-1e2, max_value=1),
    b=st.floats(min_value=-1e2, max_value=1),
    c=st.floats(min_value=-1e2, max_value=1),
)
def test_concatenation(a, b, c):
    """
    Test a fairly abstruse concatenation of values
    """
    # 1. Engine implementation
    a_eng = Value(a)
    b_eng = Value(b)
    c_eng = Value(c)
    x_eng = (( (a_eng + 1) * b_eng ).tanh() * (a_eng + c_eng).tanh() + b_eng).tanh()

    # 2. Pytorch implementation
    a_torch = torch.tensor(a, requires_grad=True)
    a_torch.grad = None
    b_torch = torch.tensor(b, requires_grad=True)
    b_torch.grad = None
    c_torch = torch.tensor(c, requires_grad=True)
    c_torch.grad = None
    x_torch = (( (a_torch + 1) * b_torch ).tanh() * (a_torch + c_torch).tanh() + b_torch).tanh()

    x_eng.backward()
    x_torch.backward()

    assert pytest.approx(x_eng.data, rel=1e-4, abs=1e-7) == x_torch.item()
    assert pytest.approx(a_eng.grad, rel=1e-4, abs=1e-7) == a_torch.grad.item()
    assert pytest.approx(b_eng.grad, rel=1e-4, abs=1e-7) == b_torch.grad.item()
    assert pytest.approx(c_eng.grad, rel=1e-4, abs=1e-7) == c_torch.grad.item()
