from typing import Iterable

from engine import Value
from nn import MLP


def train_with_sgd(
    mlp: MLP,
    features: Iterable[Iterable[float | int]],
    labels: Iterable[float | int],
) -> tuple[MLP, float, list[float]]:
    # Collect the features and labels
    features = [list(f) for f in features]
    labels = list(labels)

    def loss_fn(outputs: list[Value]) -> Value:
        # Loss function, svm "max-margin" loss
        assert len(outputs) == len(labels)
        losses = [(1 + Value(-label) * out).relu() for label, out in zip(labels, outputs)]
        return sum(losses) * (1 / len(outputs))

    def accuracy_fn(predictions: list[int]) -> float:
        num_correct = sum(pred == label for pred, label in zip(predictions, labels))
        return num_correct / len(predictions)

    inputs = [[Value(x) for x in feat] for feat in features]
    for step in range(100):
        loss = loss_fn(outputs := [mlp(inp)[0] for inp in inputs])
        predictions = [1 if output.data > 0 else -1 for output in outputs]
        accuracy = accuracy_fn(predictions)
        print(f"Step {step}: loss = {loss.data}, accuracy = {accuracy}")

        # Gradually decrease the learning rate
        learning_rate = 1.0 - step*0.9/100
        # Backprop and update parameters.
        loss.backward()
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
            param.grad = 0.0

    final_loss = loss.data
    final_outputs = [out.data for out in outputs]
    return mlp, final_loss, final_outputs
