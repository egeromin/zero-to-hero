from engine import Value
from nn import MLP


def main():
    # First, generate some training data.
    features = [
        [-1, 1, 1, -1],
        [1, -1, -1, 1],
        [-1, 1, -1, 1],
        [-1, -1, -1, 1],
    ]
    labels = [-1, 1, -1, 1]

    mlp = MLP(4, [2, 3, 2], 1)

    def mse(outputs: list[Value]) -> Value:
        assert len(outputs) == len(labels)

        # TODO: implement squaring! And backprop for it.
        return sum((out + Value(-label)) for out, label in zip(outputs, labels)) * (1/len(outputs))

    inputs = [[Value(x) for x in feat] for feat in features]
    while (loss := mse([mlp(inp)[0] for inp in inputs])).data > 1e-8:
        print(f"loss={loss.data}")
        learning_rate = 1e-2
        # Backprop and update parameters.
        loss.backward()
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
            param.grad = 0.0

    print(f"loss={loss.data}")


if __name__ == '__main__':
    main()
