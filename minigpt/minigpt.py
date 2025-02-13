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
    In particular, add estimates of train/val loss by calculating the mean cross many mini batches. ✅
11. Refactor multi head attention to use 4D tensors ✅
    11.5 Use flash attention, if it's available. ✅
12. Add multiple attention blocks ✅
13. Scale up - multiple self attention blocks, increase parameters to what is used in lectures.
    Run locally for a few iterations and see how long it takes. Estimate how long it would take
    to run N iterations. ✅
14. Train N iterations on GPU
"""

import json
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
import tqdm
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW

torch.manual_seed(1337)
DROPOUT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"Using {device}, matplotlib in Agg mode")
    matplotlib.use("Agg")


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
    if not Path("train_x.pt").exists():
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

        print("Saving processed dataset to cache")
        torch.save(X, "train_x.pt")
        torch.save(Y, "train_y.pt")
        Path("stoi.json").write_text(json.dumps(stoi, indent=2))
    else:
        print("Loading cached dataset.")
        stoi = Path("stoi.json").read_text()
        X = torch.load("train_x.pt")
        Y = torch.load("train_y.pt")

    # Shuffle input data and get train/val split
    perm = torch.randperm(len(X))
    X = X[perm]
    Y = Y[perm]

    # Move the inputs and labels to GPU.
    # For tensors, .to(device) returns a copy on the desired device.
    X, Y = X.to(device), Y.to(device)

    split = int(len(X) * 0.8)
    X_train = X[:split]
    Y_train = Y[:split]
    X_val = X[split:]
    Y_val = Y[split:]

    return {"train": X_train, "val": X_val}, {"train": Y_train, "val": Y_val}, stoi


@torch.no_grad()
def estimate_loss_and_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    num_runs: int = 20,
    batch_size: int = 64,
) -> tuple[float, float]:
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


class MultiHeadSelfAttention(nn.Module):
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

    def __init__(
        self,
        embedding_size: int,
        query_size: int,
        context_size: int,
        num_heads: int,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.context_size = context_size
        self.num_heads = num_heads
        self.keys = nn.Linear(embedding_size, query_size * num_heads)
        self.queries = nn.Linear(embedding_size, query_size * num_heads)
        self.values = nn.Linear(embedding_size, embedding_size * num_heads)
        self.use_flash_attention = use_flash_attention

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.linear = nn.Linear(embedding_size * num_heads, embedding_size)

        # self-attention mask
        self.register_buffer(
            "mask",
            ~torch.tril(torch.ones(context_size, context_size, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, E = x.shape
        assert E == self.embedding_size
        assert C == self.context_size
        H = self.num_heads
        keys = self.keys.forward(x).view(B, C, H, self.query_size).transpose(2, 1)
        queries = self.queries.forward(x).view(B, C, H, self.query_size).transpose(2, 1)
        assert tuple(keys.shape) == tuple(queries.shape) == (B, H, C, self.query_size)

        values = self.values.forward(x).reshape(B, C, H, E).transpose(2, 1)
        assert tuple(values.shape) == (B, H, C, E)

        if self.use_flash_attention:
            after_attention = F.scaled_dot_product_attention(
                key=keys, query=queries, value=values, is_causal=True
            )
        else:
            # sa == "self attention"
            sa = queries @ keys.transpose(2, 3)
            assert tuple(sa.shape) == (B, H, C, C)

            # Mask the *upper half* since each query should not interact
            # with keys that come after it in the context.
            masked_sa = torch.where(self.mask, -torch.inf, sa)
            assert tuple(masked_sa.shape) == (B, H, C, C)
            scale_factor = 1 / self.query_size**0.5
            norm_masked_sa = F.softmax(masked_sa * scale_factor, dim=3)
            assert tuple(norm_masked_sa.shape) == (B, H, C, C)

            after_attention = norm_masked_sa @ values

        assert tuple(after_attention.shape) == (B, H, C, E)

        stacked = after_attention.transpose(2, 1).contiguous()
        assert tuple(stacked.shape) == (B, C, H, E)
        flat = self.flatten(stacked)
        assert tuple(flat.shape) == (B, C, H * E)
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
        self,
        embedding_size: int,
        query_size: int,
        context_size: int,
        num_heads: int,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.query_size = query_size
        self.context_size = context_size
        self.num_heads = num_heads
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.multi_head_attention = MultiHeadSelfAttention(
            embedding_size=embedding_size,
            query_size=query_size,
            context_size=context_size,
            num_heads=num_heads,
            use_flash_attention=use_flash_attention,
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
        num_heads: int,
        num_blocks: int,
        use_flash_attention: bool = False,
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
        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_size=embedding_size,
                    query_size=query_size,
                    context_size=context_size,
                    num_heads=num_heads,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_blocks)
            ]
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.linear = nn.Linear(embedding_size * context_size, vocab_size)
        self.register_buffer(
            "positions", torch.tensor(range(self.context_size), dtype=torch.long)
        )
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
        emb = self.embedding.forward(x)
        pos = self.positional_encoding.forward(self.positions)
        drop = self.dropout(emb + pos)
        sa = self.attention_blocks(drop)
        flat = self.flatten(sa)
        return self.linear(flat)


def sample_from_model(
    model: MiniGPT, context_size: int, stoi: Mapping[str, int], num_chars: int
) -> str:
    itos = {i: s for s, i in stoi.items()}
    current_ctx = "".join([itos[0] for _ in range(context_size)])
    generated_chars = []
    for _sample in range(num_chars):
        input = torch.tensor([[stoi[c] for c in current_ctx]], dtype=torch.long).to(
            device
        )
        logits = model.forward(input)
        probs = F.softmax(logits, dim=1)
        sample = torch.multinomial(probs, num_samples=1)
        next_char = itos[sample.item()]
        generated_chars.append(next_char)
        current_ctx = current_ctx[1:] + next_char

    return "".join(generated_chars)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "require-cuda":
        if device != "cuda":
            print("Cuda not available on current system. Aborting")
            sys.exit(1)
    print(f"Device: {device}")
    print("Loading dataset...")
    context_size = 256
    X, Y, stoi = load_dataset(context_size=context_size)
    print(
        f"Done loading dataset. Train size = {len(X['train'])}, val size = {len(X['val'])}"
    )

    vocab_size = len(stoi)
    print(f"Vocab size = {vocab_size}")
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=384,
        context_size=context_size,
        query_size=384 // 6,
        num_heads=6,
        num_blocks=6,
        use_flash_attention=True,
    )
    # Move the model to GPU. For nn.Module, .to(device) modifies in-place
    model.to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params // 1e6}M parameters")
    max_training_iterations = 8_001
    batch_size = 64
    opt = AdamW(model.parameters(), lr=3e-4)

    train_losses = []
    validation_losses = []
    measure_every = 500

    for i in tqdm.tqdm(range(max_training_iterations)):
        perm = torch.randperm(len(X["train"]))[:batch_size]
        X_batch = X["train"][perm]
        Y_batch = Y["train"][perm]
        opt.zero_grad()
        logits = model.forward(X_batch)
        loss = F.cross_entropy(logits, Y_batch)
        loss.backward()
        opt.step()

        if i % measure_every == 0:
            model.eval()
            train_loss_estimate, _ = estimate_loss_and_accuracy(
                model, X["train"], Y["train"]
            )
            val_loss_estimate, val_accuracy_estimate = estimate_loss_and_accuracy(
                model, X["val"], Y["val"]
            )
            train_losses.append(train_loss_estimate)
            validation_losses.append(val_loss_estimate)
            print(
                f"\n{i}: train loss = {train_loss_estimate:4f}, "
                f"val loss = {val_loss_estimate:4f}, "
                f"val accuracy = {val_accuracy_estimate * 100:.2f}%"
            )
            model.train()

    print(f"Number of parameters: {total_params // 1e6}M parameters")
    model.eval()
    sample = sample_from_model(
        model, stoi=stoi, context_size=context_size, num_chars=2000
    )
    print(sample)
    Path("generated-sample.txt").write_text(sample)

    # Plot training and validation losses
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))
    train_losses_log10 = [math.log10(e) for e in train_losses]
    val_losses_log10 = [math.log10(e) for e in validation_losses]
    train_iteration = [i * measure_every for i in range(len(train_losses))]
    axes[0].plot(train_iteration, train_losses_log10)
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Log10 Train loss")
    axes[0].set_title("Log10 training losses during the training")
    axes[1].plot(train_iteration, val_losses_log10)
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Log10 Validation loss")
    axes[1].set_title("Log10 validation losses during the training")
    plt.tight_layout()

    if device != "cuda":
        plt.show()
    else:
        plt.savefig("training-plots.png", dpi=300)


if __name__ == "__main__":
    main()
