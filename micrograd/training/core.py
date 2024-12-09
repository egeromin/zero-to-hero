from typing import Iterable

from engine import Value
from nn import MLP


def train_with_sgd(
    mlp: MLP,
    features: Iterable[Iterable[float | int]],
    labels: Iterable[float | int],
    loss_threshold: float = 1e-3,
    max_steps: int | None = None,
) -> tuple[MLP, float, list[float]]:
    # Collect the features and labels
    features = [list(f) for f in features]
    labels = list(labels)

    def mse(outputs: list[Value]) -> Value:
        assert len(outputs) == len(labels)
        sum_of_squares = sum(
            (out + Value(-label)) ** 2 for out, label in zip(outputs, labels)
        )
        return sum_of_squares * (1 / len(outputs))

    inputs = [[Value(x) for x in feat] for feat in features]
    step: int = 0
    while (
        loss := mse(predictions := [mlp(inp)[0] for inp in inputs])
    ).data > loss_threshold:
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
