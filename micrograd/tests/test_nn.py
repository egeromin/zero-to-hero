import torch
import torch.nn as nn
from engine import Value
from nn import MLP
import pytest


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


@pytest.mark.parametrize(
    "inputs", [(2, 3), (5, 10), (7, 8), (0.1, 0.9), (100, 200), (80, 20), (7, 2)]
)
def test_mlp_torch(inputs):
    mlp = MLP(2, [3, 3], 1)
    inputs = [Value(inputs[0], name="x_0"), Value(inputs[1], name="x_1")]
    output = mlp(inputs)[0].data

    # Create the model and test it
    model = MLPTorch()

    # Sample input
    input_tensor = torch.tensor([[input.data for input in inputs]])
    expected_output = model(input_tensor)
    expected_output = expected_output[0].item()

    assert pytest.approx(expected_output) == output
