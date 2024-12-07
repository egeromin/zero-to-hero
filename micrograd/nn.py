from engine import Value, draw_graph


class Neuron:
    """A neurons with many dendrites and an axon."""

    def __init__(self, n_inputs: int):
        # TODO: improve initialisation.
        self.n_inputs = n_inputs
        self.w = [Value(data=1.0, name=f"w_{i}") for i in range(n_inputs)]
        self.b = Value(data=0.0, name="b")

    def __repr__(self):
        return f"Neuron({self.n_inputs})"

    def __call__(self, inputs: list[Value]):
        if len(inputs) != self.n_inputs:
            raise RuntimeError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")
        linear_combination = sum(inp * w for inp, w in zip(inputs, self.w)) + self.b
        return linear_combination.tanh()


class MLP:
    def __init__(self, n_inputs: int, sz_layers: list[int], n_outputs: int):
        self.n_inputs = n_inputs  # number of inputs
        self.sz_layers = sz_layers  # sizes of intermediate layers
        self.n_outputs = n_outputs  # size of output layer

        all_sz_layers = self.sz_layers + [
            self.n_outputs
        ]  # sizes of all layers, intermediate and output

        # number of inputs to each layer in all_layers, intermediate and output
        all_layers_n_inputs = [self.n_inputs] + self.sz_layers
        self.neurons = [
            [Neuron(n_inputs) for _ in range(sz_layer)]
            for n_inputs, sz_layer in zip(all_layers_n_inputs, all_sz_layers)
        ]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        """
        Returns a list of output activations, given a list of input Values,
        by running the forward pass through the fully-connected MLP.
        """
        if len(inputs) != self.n_inputs:
            raise RuntimeError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")

        intermediate_values: list[list[Value]] = [inputs]
        for layer in self.neurons:
            layer_output = [neuron(intermediate_values[-1]) for neuron in layer]
            intermediate_values.append(layer_output)

        if len(intermediate_values[-1]) != self.n_outputs:
            raise RuntimeError(
                f"Something went wrong: got {len(intermediate_values[-1])} output values, expected {self.n_outputs}. This is a bug."
            )

        return intermediate_values[-1]


def main():
    mlp = MLP(3, [2, 3], 1)
    inputs = [Value(2, name="x_0"), Value(3, name="x_1"), Value(1, name="x_3")]
    output = mlp(inputs)[0]
    draw_graph(output)


if __name__ == "__main__":
    main()
