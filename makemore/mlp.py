"""
Implement a model with variable context length using an MLP,
following the approach described in:
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""

import itertools
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# Steps:
# -1. Fix validation and test losses and accuracy to use no_grad.
# 0. Print a graph of validation losses over epochs.
# 1. Iterate on parameters to achieve a low loss.
# 2. Plot the activation, gradient and update statistics at each layer.
# 3. Fix the initialisation using manual scaling factors:
#     a. First, the initial loss
#     b. Then, the saturated tanh
# 4. Use kaiming initialisation and compare
# 5. Implement the variants of batch norm by hand.
# 6. Pytorchify the code.


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
        self,
        vocab_size: int,
        context_size: int,
        embedding_size: int,
        hidden_size: int,
        g: torch.Generator,
    ):
        self.embedding: torch.Tensor = torch.randn(
            size=(vocab_size, embedding_size), requires_grad=True, generator=g
        )
        self.hidden_w: torch.Tensor = torch.randn(
            size=(embedding_size * context_size, hidden_size),
            requires_grad=True,
            generator=g,
        )
        self.hidden_b: torch.Tensor = torch.zeros(
            size=(hidden_size,), requires_grad=True, dtype=torch.float
        )
        self.output_w: torch.Tensor = torch.randn(
            size=(hidden_size, vocab_size), requires_grad=True, generator=g
        )
        self.output_b: torch.Tensor = torch.zeros(
            size=(vocab_size,), requires_grad=True, dtype=torch.float
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass. Do not compute the final softmax,
        # as this will be calculated inside the loss function,
        # for numerical stability.
        embeddings = self.embedding[x]
        embeddings_reshaped = embeddings.view(embeddings.shape[0], -1)
        hidden = embeddings_reshaped @ self.hidden_w + self.hidden_b
        hidden_nonlin = hidden.tanh()
        output = hidden_nonlin @ self.output_w + self.output_b
        return output

    def zero_grad(self):
        self.embedding.grad = None
        self.hidden_w.grad = None
        self.hidden_b.grad = None
        self.output_w.grad = None
        self.output_b.grad = None

    def update_parameters(self, learning_rate: float):
        self.embedding.data -= learning_rate * self.embedding.grad
        self.hidden_w.data -= learning_rate * self.hidden_w.grad
        self.hidden_b.data -= learning_rate * self.hidden_b.grad
        self.output_w.data -= learning_rate * self.output_w.grad
        self.output_b.data -= learning_rate * self.output_b.grad


@torch.no_grad()
def calculate_loss_and_accuracy(
    mlp: MLP, X: torch.Tensor, Y: torch.Tensor
) -> tuple[float, float]:
    """Returns a tuple (loss, accuracy)"""
    logits = mlp.forward(X)
    predictions = logits.argmax(dim=-1)
    assert predictions.shape == Y.shape
    accuracy = sum(predictions == Y) / Y.shape[0]
    loss = calculate_loss(mlp, logits, Y)
    return loss.item(), accuracy


def calculate_loss(mlp: MLP, logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    reg_alpha = 0.001
    model_loss = F.cross_entropy(logits, Y)
    reg_loss = reg_alpha * ((mlp.hidden_w**2).sum() + (mlp.output_w**2).sum())
    loss = model_loss + reg_loss
    return loss


def train_model(mlp: MLP, X: torch.Tensor, Y: torch.Tensor, g: torch.Generator) -> MLP:
    assert X.shape[0] == Y.shape[0]
    # split into train, test and validation sets
    # First shuffle the dataset
    perm_idx = torch.randperm(len(X), generator=g)
    X = X[perm_idx]
    Y = Y[perm_idx]
    train_split = 0.8
    val_split = 0.1
    # test_split = 0.1
    train_cutoff = int(train_split * X.shape[0])
    val_cutoff = train_cutoff + int(val_split * X.shape[0])
    X_train = X[:train_cutoff, :]
    Y_train = Y[:train_cutoff]
    X_val = X[train_cutoff:val_cutoff, :]
    Y_val = Y[train_cutoff:val_cutoff]
    X_test = X[val_cutoff:, :]
    Y_test = Y[val_cutoff:]

    num_training_iterations = 2000
    val_accuracy = 0.0
    val_losses: list[tuple[int, float]] = []
    # Heuristic learning rate, based on what we observe during training.
    learning_rate = (
        0.1 if val_accuracy < 0.25 else 0.01 if val_accuracy < 0.28 else 0.001
    )
    batch_losses: list[float] = []
    for i in range(num_training_iterations):
        # Grab a minibatch.
        batch_size = 200
        batch_idx = torch.randperm(len(X_train), generator=g)[:batch_size]
        X_batch = X_train[batch_idx]
        Y_batch = Y_train[batch_idx]
        assert X_batch.shape == (batch_size, X_train.shape[1])
        assert Y_batch.shape == (batch_size,)

        mlp.zero_grad()
        logits_batch = mlp.forward(X_batch)
        loss = calculate_loss(mlp, logits_batch, Y_batch)
        batch_losses.append(loss.item())
        loss.backward()
        mlp.update_parameters(learning_rate)

        # Calculate the validation accuracy
        if i % 100 == 0:
            val_loss, val_accuracy = calculate_loss_and_accuracy(mlp, X_val, Y_val)
            val_losses.append((i, val_loss))
            print(
                f"Step {i}: validation loss = {val_loss:.4f}, "
                f"validation accuracy = {val_accuracy * 100:.2f}%"
            )

    # Calculate the final test accuracy.
    test_loss, test_accuracy = calculate_loss_and_accuracy(mlp, X_test, Y_test)
    print(
        f"Final test loss = {test_loss:.4f}, test accuracy = {test_accuracy * 100:.2f}%"
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].plot(range(len(batch_losses)), batch_losses)
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Train loss")
    axes[0].legend()

    x, y = zip(*val_losses)
    axes[1].plot(x, y)
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Validation loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return mlp


def sample_from_model(
    mlp: MLP,
    g: torch.Generator,
    num_samples: int,
    context_size: int,
    stoi: Mapping[str, int],
):
    itos = {i: c for c, i in stoi.items()}
    for _ in range(num_samples):
        preds = []
        current_ctx = "." * context_size
        pred = None
        while pred != ".":
            x = torch.tensor([stoi[c] for c in current_ctx], dtype=torch.long).view(
                1, -1
            )
            logits = mlp.forward(x)
            probs = F.softmax(logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1, generator=g)
            pred = itos[sampled_idx.item()]
            preds.append(pred)
            current_ctx = current_ctx[1:] + pred
        generated_sample = "".join(preds)
        print(f"{generated_sample=}")


def main():
    context_size = 3
    X, Y, stoi = load_dataset(context_size=context_size)

    g = torch.Generator().manual_seed(2147483647)
    mlp = MLP(
        vocab_size=len(stoi),
        context_size=context_size,
        embedding_size=11,
        hidden_size=13,
        g=g,
    )
    output = mlp.forward(X[:20, :])
    print(output.shape)

    mlp = train_model(mlp, X, Y, g)
    sample_from_model(mlp, g=g, num_samples=20, context_size=context_size, stoi=stoi)


if __name__ == "__main__":
    main()
