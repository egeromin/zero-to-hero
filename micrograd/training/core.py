import itertools
from typing import Iterable

from engine import Value
from nn import MLP


def train_with_sgd(
    mlp: MLP,
    features: Iterable[Iterable[float | int]],
    labels: Iterable[float | int],
    loss_threshold: float = 1e-3,
    batch_size: int = 10,
    max_steps: int | None = None,
) -> tuple[MLP, float, list[float]]:
    # Collect the features and labels
    features = [list(f) for f in features]
    labels = list(labels)

    def batch_maker() -> Iterable[tuple[list[list[Value]], list[float]]]:
        """Returns an iterator over mini-batches of size `batch_size`."""
        mini_batch_features = []
        mini_batch_labels = []
        for feat, label in itertools.cycle(zip(features, labels)):
            mini_batch_features.append([Value(f) for f in feat])
            mini_batch_labels.append(label)
            if len(mini_batch_features) == batch_size:
                yield mini_batch_features, mini_batch_labels
                mini_batch_features = []
                mini_batch_labels = []

    def mse(outputs: list[Value], labels: list[float]) -> Value:
        assert len(outputs) == len(labels) == batch_size
        sum_of_squares: Value = sum(
            (out + Value(-label)) ** 2 for out, label in zip(outputs, labels)
        )
        return sum_of_squares * (1 / len(outputs))

    step: int = 0
    loss = Value(1000.0)
    predictions = []
    print(f"Training with a batch size of {batch_size}")
    for batch_features, batch_labels in batch_maker():
        if (
            loss := mse(
                predictions := [mlp(inp)[0] for inp in batch_features], batch_labels
            )
        ).data <= loss_threshold:
            print(f"Loss = {loss.data} <= threshold {loss_threshold}, ending")
            break

        step += 1
        if max_steps and step > max_steps:
            print(f"Ending at {max_steps=}")
            break

        print(f"loss at step {step}: {loss.data}")
        learning_rate = 0.5 if loss.data > 0.4 else 0.1 if loss.data > 0.09 else 1e-2
        # Backprop and update parameters.
        loss.backward()
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
            param.grad = 0.0

    final_loss = loss.data
    final_predictions = [pred.data for pred in predictions]
    return mlp, final_loss, final_predictions
