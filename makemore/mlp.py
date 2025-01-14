"""
Implement a model with variable context length using an MLP,
following the approach described in:
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""

import itertools
from pathlib import Path

import torch


def load_dataset(context_size: int):
    """
    Returns a tensor of X's, Y's, already prepared for training,
    given the context size.

    Return:
        X: training contexts
        Y: output labels
        stoi: mapping of str to int
    """
    all_names = Path("names.txt").read_text().split()
    stoi = {
        c: i
        for i, c in enumerate(
            sorted({"."} | set(itertools.chain.from_iterable(all_names)))
        )
    }
    num_training_samples = sum(len(name) + 1 for name in all_names)
    X = torch.zeros((num_training_samples, context_size), dtype=torch.long)
    Y = torch.zeros(num_training_samples, dtype=torch.long)
    pos = 0
    for i, name in enumerate(all_names):
        name_padded = "." * context_size + name + "."
        for k in range(len(name_padded) - context_size):
            input_ctx = name_padded[k : k + context_size]
            output_c = name_padded[k + context_size]
            if i < 2:
                print(f"{input_ctx} -> {output_c}")  # For debug
            X[pos, :] = torch.tensor([stoi[c] for c in input_ctx], dtype=torch.long)
            Y[pos] = stoi[output_c]
            pos += 1

    return X, Y, stoi


class MLP:
    def __init__(
        self, vocab_size: int, context_size: int, embedding_size: int, hidden_size: int
    ):
        self.embedding: torch.Tensor = torch.randn(
            size=(vocab_size, embedding_size), requires_grad=True
        )
        self.hidden_w: torch.Tensor = torch.randn(
            size=(embedding_size * context_size, hidden_size), requires_grad=True
        )
        self.hidden_b: torch.Tensor = torch.zeros(
            size=(hidden_size,), requires_grad=True, dtype=torch.float
        )
        self.output_w: torch.Tensor = torch.randn(
            size=(hidden_size, vocab_size), requires_grad=True
        )
        self.output_b: torch.Tensor = torch.zeros(
            size=(vocab_size,), requires_grad=True, dtype=torch.float
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass. Do not compute the final softmax,
        # as this will be calculated inside the loss function,
        # for numerical stability.
        print(x.shape)  # 20 * 3
        embeddings = self.embedding[x]
        print(embeddings.shape)  # 20 * 3 * 11
        embeddings_reshaped = embeddings.view(embeddings.shape[0], -1)
        print(embeddings_reshaped.shape)  # 20 * 33
        hidden = embeddings_reshaped @ self.hidden_w + self.hidden_b
        print(hidden.shape)  # 20 * 13
        hidden_nonlin = hidden.tanh()
        print(hidden.shape)  # 20 * 13
        output = hidden_nonlin @ self.output_w + self.output_b
        print(output.shape)  # 20 * 27
        return output


def train_model(mlp: MLP, X: torch.Tensor, Y: torch.Tensor) -> MLP:
    pass


def sample_from_model(mlp: MLP):
    pass


def main():
    context_size = 3
    X, Y, stoi = load_dataset(context_size=context_size)
    mlp = MLP(
        vocab_size=len(stoi),
        context_size=context_size,
        embedding_size=11,
        hidden_size=13,
    )
    output = mlp.forward(X[:20, :])
    print(output.shape)


if __name__ == "__main__":
    main()
