import math

import torch

import pytest
from hypothesis import given, strategies as st, settings

from engine import Value


@settings(max_examples=20)
@given(
    st.floats(min_value=-1e4, max_value=1e4), st.floats(min_value=-1e4, max_value=1e4)
)
def test_add(x, y):
    x = Value(x)
    y = Value(y)
    z = x + y
    z.grad = 1.0
    z._backward()

    assert pytest.approx(z.data) == x.data + y.data
    assert pytest.approx(x.grad) == 1.0
    assert pytest.approx(y.grad) == 1.0


@settings(max_examples=20)
@given(
    st.floats(min_value=-1e4, max_value=1e4), st.floats(min_value=-1e4, max_value=1e4)
)
def test_mul(x, y):
    x = Value(x)
    y = Value(y)
    z = x * y
    z.grad = 1.0
    z._backward()

    assert pytest.approx(z.data) == x.data * y.data
    assert pytest.approx(x.grad) == y.data
    assert pytest.approx(y.grad) == x.data


@settings(max_examples=20)
@given(st.floats(min_value=-1e4, max_value=1e4))
def test_tanh(x):
    x = Value(x)
    y = x.tanh()
    y.grad = 1.0
    y._backward()

    assert pytest.approx(y.data) == math.tanh(x.data)
    assert pytest.approx(x.grad) == (1 - y.data)


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
    x_eng = (
        ((a_eng + 1).tanh() * b_eng).tanh() * (a_eng * c_eng).tanh() + c_eng
    ).tanh()

    # 2. Pytorch implementation
    a_torch = torch.tensor(a, requires_grad=True)
    a_torch.grad = None
    b_torch = torch.tensor(b, requires_grad=True)
    b_torch.grad = None
    c_torch = torch.tensor(c, requires_grad=True)
    c_torch.grad = None
    x_torch = (
        ((a_torch + 1).tanh() * b_torch).tanh() * (a_torch * c_torch).tanh() + c_torch
    ).tanh()

    x_eng.backward()
    x_torch.backward()

    assert pytest.approx(x_eng.data) == x_torch.item()
    assert pytest.approx(a_eng.grad) == a_torch.grad.item()
    assert pytest.approx(b_eng.grad) == b_torch.grad.item()
    assert pytest.approx(c_eng.grad) == c_torch.grad.item()
