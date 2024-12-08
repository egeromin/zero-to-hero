import torch
import torch.nn as nn
from engine import Value
from nn import MLP
import pytest
from hypothesis import given, strategies as st, settings


# Define the MLP
class MLPTorch(nn.Module):
    def __init__(self):
        super().__init__()

        # Define layers
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

        # Initialize weights and biases
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.constant_(layer.weight, 1.0)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# Bound the inputs and outputs for our test case, since in our custom MLP library,
# we do not handle underflow and overflow.
@settings(max_examples=100)
@given(
    st.floats(min_value=-1e4, max_value=1e4), st.floats(min_value=-1e4, max_value=1e4)
)
def test_mlp_torch(x, y):
    print(f"Testing with {x=}, {y=}")
    mlp = MLP(2, [3, 3], 1)
    inputs = [Value(x, name="x_0"), Value(y, name="x_1")]
    output = mlp(inputs)[0].data

    # Create the model and test it
    model = MLPTorch()

    # Sample input
    input_tensor = torch.tensor([[input.data for input in inputs]])
    expected_output = model(input_tensor)
    expected_output = expected_output[0].item()

    assert pytest.approx(expected_output) == output
