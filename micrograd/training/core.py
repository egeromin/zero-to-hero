from engine import Value
from nn import MLP


def train_with_sgd(
    mlp: MLP,
    features: list[list[float | int]],
    labels: list[float | int],
    loss_threshold: float = 1e-3,
) -> tuple[MLP, float, list[float]]:
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
        if step % 100 == 0:
            print(f"loss at step {step}: {loss.data}")
        learning_rate = 1e-1 if loss.data > 1e-2 else 1e-2
        # Backprop and update parameters.
        loss.backward()
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
            param.grad = 0.0

    final_loss = loss.data
    final_predictions = [pred.data for pred in predictions]
    return mlp, final_loss, final_predictions
