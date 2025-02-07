"""
Single-module implementation of GPT on TinyShakespeare data.

To-do list:

1. Dataset loader with train + val split.  ✅
2. Initial MLP model definition, and loop.  ✅
3. Define a single self attention head, untested  ✅
4. Add self attention to model, with bag of words embedding  ✅
5. Define multi-head attention and a self attention block that uses it. ✅
5.5. Remove everything that's not a transformer from the model. ✅
6. Add positional encodings ✅
7. Add residual connections ✅
8. Add LayerNorm -> N.B, should come before the multi head attention, unlike in the paper. ✅
9. Add dropout ✅
9.5 Fix Feedforward to include non-linearity!  ✅
10. Add plots of training losses, validation losses and activations at specific points in the model.
    In particular, add estimates of train/val loss by calculating the mean cross many mini batches.
11. Scale up - multiple self attention blocks, increase parameters to what is used in lectures.
    Run locally for a few iterations and see how long it takes. Estimate how long it would take
    to run N iterations.
12. Train N iterations on GPU
13. Refactor multi head attention to use 4D tensors
14. Refactor to use flash attention, if available
"""

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW


torch.manual_seed(1337)
DROPOUT = 0.2


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


@torch.no_grad()
def estimate_loss_and_accuracy(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, num_runs: int = 20, batch_size: int = 64) -> tuple[float, float]:
    losses = []
    accuracies = []
    for _ in range(num_runs):
        perm = torch.randperm(len(X))[:batch_size]
        X_batch = X[perm]
        Y_batch = Y[perm]
        logits = model.forward(X_batch)
        preds = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, Y_batch)
        accuracy = (preds == Y_batch).float().mean().item()
        losses.append(loss.item())
        accuracies.append(accuracy)
    mean_loss = sum(losses) / len(losses)
    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_loss, mean_accuracy


class SelfAttention(nn.Module):
    # What I recall from self-attention: each element in the context
    # talks to every other element in the context
    # batch * context_size * embedding_size
    # B * C * E
    # Also, need to ensure that we can interact only with what comes
    # before us, via the masking softmax trick.
    # Now, for each element in the context, have a self attention
    # matrix, which is a parameter of the model.
    # keys, queries, values
    # We learn the keys, queries and values for any embedding vector

    def __init__(self, embedding_size: int, query_size: int, context_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.keys = nn.Linear(embedding_size, query_size)
        self.queries = nn.Linear(embedding_size, query_size)
        self.values = nn.Linear(embedding_size, embedding_size)
        # self-attention mask
        self.register_buffer(
            "mask",
            ~torch.tril(torch.ones(context_size, context_size, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, E = x.shape
        assert E == self.embedding_size
        keys = self.keys.forward(x)
        queries = self.queries.forward(x)
        assert tuple(keys.shape) == tuple(queries.shape) == (B, C, self.query_size)

        # sa == "self attention"
        sa = queries @ keys.transpose(1, 2)
        assert tuple(sa.shape) == (B, C, C)

        # Mask the *upper half* since each query should not interact
        # with keys that come after it in the context.
        masked_sa = torch.where(self.mask, -torch.inf, sa)
        assert tuple(masked_sa.shape) == (B, C, C)
        norm_masked_sa = F.softmax(
            masked_sa / torch.sqrt(torch.tensor(self.query_size).float()), dim=2
        )
        assert tuple(norm_masked_sa.shape) == (B, C, C)

        values = self.values.forward(x)
        assert tuple(values.shape) == (B, C, E)
        outputs = norm_masked_sa @ values
        return outputs


class MultiHeadAttention(nn.Module):
    # TODO: refactor and use 4-D tensors internally, for efficiency

    def __init__(
        self, embedding_size: int, query_size: int, context_size: int, num_heads: int
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.context_size = context_size
        self.num_heads = num_heads
        self.heads = [
            SelfAttention(
                embedding_size=embedding_size,
                query_size=query_size,
                context_size=context_size,
            )
            for _ in range(num_heads)
        ]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.linear = nn.Linear(embedding_size * num_heads, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, E = x.shape
        assert E == self.embedding_size
        head_outputs = [head.forward(x) for head in self.heads]
        stacked = torch.stack(head_outputs, dim=2)
        assert tuple(stacked.shape) == (B, C, self.num_heads, E)
        flat = self.flatten(stacked)
        assert tuple(flat.shape) == (B, C, self.num_heads * E)
        output = self.linear(flat)
        assert tuple(output.shape) == (B, C, E)
        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.linear_1(x)
        rl = self.relu(l1)
        return self.linear_2(rl)


class AttentionBlock(nn.Module):
    def __init__(
        self, embedding_size: int, query_size: int, context_size: int, num_heads: int
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.context_size = context_size
        self.num_heads = num_heads
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.multi_head_attention = MultiHeadAttention(
            embedding_size=embedding_size,
            query_size=query_size,
            context_size=context_size,
            num_heads=num_heads,
        )
        self.drop_1 = nn.Dropout(p=DROPOUT)
        self.norm_2 = nn.LayerNorm(embedding_size)
        self.feed_forward = FeedForward(embedding_size)
        self.drop_2 = nn.Dropout(p=DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = self.drop_1(self.multi_head_attention(self.norm_1(x))) + x
        output = self.drop_2(self.feed_forward(self.norm_2(multi))) + multi
        assert output.shape == x.shape
        return output


class MiniGPT(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_size: int,
        query_size: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Learned positional encoding.
        self.positional_encoding = nn.Embedding(context_size, embedding_size)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.attention_block = AttentionBlock(
            embedding_size=embedding_size,
            query_size=query_size,
            context_size=context_size,
            num_heads=4,
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.linear = nn.Linear(embedding_size * context_size, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.tensor(range(self.context_size), dtype=torch.long)
        positions_broadcast = torch.ones_like(x) * positions
        emb = self.embedding.forward(x)
        pos = self.positional_encoding.forward(positions_broadcast)
        drop = self.dropout(emb + pos)
        sa = self.attention_block(drop)
        flat = self.flatten(sa)
        return self.linear(flat)


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
        query_size=16,
    )
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    max_training_iterations = 10_001
    batch_size = 32
    opt = AdamW(model.parameters(), lr=0.01)

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
            model.eval()
            train_loss_estimate, _ = estimate_loss_and_accuracy(model, X["train"], Y["train"])
            val_loss_estimate, val_accuracy_estimate = estimate_loss_and_accuracy(model, X["val"], Y["val"])
            print(
                f"{i}: train loss = {train_loss_estimate:4f}, val loss = {val_loss_estimate:4f}, val accuracy = {val_accuracy_estimate * 100:.2f}%"
            )
            model.train()

    print(f"Number of parameters: {total_params}")
    model.eval()
    print(sample_from_model(model, stoi=stoi, input_str=input_str, num_chars=100))


if __name__ == "__main__":
    main()
