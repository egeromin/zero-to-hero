"""
Implement the bigram model again, this time using pytorch layers,
as a refresher.
"""

import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD


torch.manual_seed(1337)


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


class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.forward(x)


def sample_from_model():
    # TODO
    pass


def main():
    X, Y, stoi = load_dataset(context_size=1)
    X = X.squeeze(dim=1)

    # Shuffle input data and get train/val split
    perm = torch.randperm(len(X))
    X = X[perm]
    Y = Y[perm]
    split = int(len(X) * 0.8)
    X_train = X[:split]
    Y_train = Y[:split]
    X_val = X[split:]
    Y_val = Y[split:]

    vocab_size = len(stoi)
    model = BigramModel(vocab_size=vocab_size)
    max_training_iterations = 1001
    batch_size = 32
    opt = SGD(model.parameters(), lr=0.01)

    for i in range(max_training_iterations):
        perm = torch.randperm(len(X_train))[:batch_size]
        X_batch = X_train[perm]
        Y_batch = Y_train[perm]
        opt.zero_grad()
        logits = model.forward(X_batch)
        loss = F.cross_entropy(logits, Y_batch)
        loss.backward()
        opt.step()

        if i % 100 == 0:
            logits_val = model.forward(X_val)
            preds_val = logits_val.argmax(dim=1)
            val_loss = F.cross_entropy(logits_val, Y_val)
            val_accuracy = (preds_val == Y_val).float().mean().item()
            print(
                f"{i}: train loss = {loss.item():4f}, val loss = {val_loss.item():4f}, val accuracy = {val_accuracy * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
