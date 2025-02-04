"""
Single-module implementation of GPT on TinyShakespeare data.

To-do list:

1. Dataset loader with train + val split.
2. Initial MLP model definition, and loop.
3. Define a single self attention head & measure performance locally.
4. Define multiple self attention heads & measure performance.
5. Add residual connections.
6. Add LayerNorm -> N.B, should come before the multi head attention, unlike in the paper.
7. Add dropout.
"""

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD


torch.manual_seed(1337)


def load_dataset(
    context_size: int,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, int]]:
    """
    Returns a tensor of X's, Y's, already prepared for training,
    given the context size.

    Return:
        X: training contexts
        Y: output labels
        stoi: mapping of str to int
    """
    corpus = Path("tinyshakespeare.txt").read_text()
    stoi = {c: i for i, c in enumerate(sorted(set(corpus)))}
    num_training_samples = len(corpus) - context_size
    X = torch.zeros((num_training_samples, context_size), dtype=torch.long)
    Y = torch.zeros(num_training_samples, dtype=torch.long)
    for i in range(num_training_samples):
        input_ctx = corpus[i : i + context_size]
        output_c = corpus[i + context_size]
        X[i, :] = torch.tensor([stoi[c] for c in input_ctx], dtype=torch.long)
        Y[i] = stoi[output_c]

    # Shuffle input data and get train/val split
    perm = torch.randperm(len(X))
    X = X[perm]
    Y = Y[perm]
    split = int(len(X) * 0.8)
    X_train = X[:split]
    Y_train = Y[:split]
    X_val = X[split:]
    Y_val = Y[split:]

    return {"train": X_train, "val": X_val}, {"train": Y_train, "val": Y_val}, stoi


class MiniGPT(torch.nn.Module):
    def __init__(
        self, vocab_size: int, context_size: int, embedding_size: int, hidden_size: int
    ):
        super().__init__()
        # First version: replicate MLP results
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(embedding_size * context_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding.forward(x)
        flat = self.flatten(emb)
        hidden = self.linear_1(flat)
        tanh = self.tanh(self.norm(hidden))
        return self.linear_2(tanh)


def sample_from_model(
    model: MiniGPT, input_str: str, stoi: Mapping[str, int], num_chars: int
) -> str:
    itos = {i: s for s, i in stoi.items()}
    next_ctx = None
    generated_chars = []
    for _sample in range(num_chars):
        current_ctx = next_ctx or input_str
        input = torch.tensor([[stoi[c] for c in current_ctx]], dtype=torch.long)
        logits = model.forward(input)
        probs = F.softmax(logits, dim=1)
        sample = torch.multinomial(probs, num_samples=1)
        next_char = itos[sample.item()]
        generated_chars.append(next_char)
        next_ctx = current_ctx[1:] + next_char

    return "".join(generated_chars)


def main():
    print("Loading dataset...")
    context_size = 8
    input_str = "Consider"
    assert len(input_str) == context_size
    X, Y, stoi = load_dataset(context_size=context_size)
    print(
        f"Done loading dataset. Train size = {len(X['train'])}, val size = {len(X['val'])}"
    )

    vocab_size = len(stoi)
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=64,
        context_size=context_size,
        hidden_size=32,
    )
    max_training_iterations = 10001
    batch_size = 32
    opt = SGD(model.parameters(), lr=0.01)

    for i in range(max_training_iterations):
        perm = torch.randperm(len(X["train"]))[:batch_size]
        X_batch = X["train"][perm]
        Y_batch = Y["train"][perm]
        opt.zero_grad()
        logits = model.forward(X_batch)
        loss = F.cross_entropy(logits, Y_batch)
        loss.backward()
        opt.step()

        if i % 500 == 0:
            logits_val = model.forward(X["val"])
            preds_val = logits_val.argmax(dim=1)
            val_loss = F.cross_entropy(logits_val, Y["val"])
            val_accuracy = (preds_val == Y["val"]).float().mean().item()
            print(
                f"{i}: train loss = {loss.item():4f}, val loss = {val_loss.item():4f}, val accuracy = {val_accuracy * 100:.2f}%"
            )

    print(sample_from_model(model, stoi=stoi, input_str=input_str, num_chars=100))


if __name__ == "__main__":
    main()
