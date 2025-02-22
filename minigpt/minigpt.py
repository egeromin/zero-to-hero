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
14. Train N iterations on GPU  ✅
"""

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

from tokenizer import Tokenizer

torch.manual_seed(1337)
DROPOUT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"Using {device}, matplotlib in Agg mode")
    matplotlib.use("Agg")


def load_dataset(
    context_size: int,
    tokenizer: Tokenizer,
    path_corpus: Path,
    cache_path: Path | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """
    Returns a tensor of X's, Y's, already prepared for training,
    given the context size.

    Return:
        X: training contexts
        Y: output labels
        stoi: mapping of str to int
    """
    if cache_path and (cache_path / "train_x.pt").exists():
        print("Loading cached dataset.")
        X = torch.load(cache_path / "train_x.pt")
        Y = torch.load(cache_path / "train_y.pt")
    else:
        print("Reading corpus...")
        corpus = path_corpus.read_text()
        print("Encoding corpus...")
        tokens = tokenizer.encode(corpus, verbose=True)
        print("Done encoding corpus.")
        num_training_samples = len(tokens) - context_size
        X = torch.zeros((num_training_samples, context_size), dtype=torch.long)
        Y = torch.zeros((num_training_samples, context_size), dtype=torch.long)
        for i in range(num_training_samples):
            input_ctx = tokens[i : i + context_size]
            output_ctx = tokens[i + 1 : i + context_size + 1]
            X[i, :] = torch.tensor(input_ctx, dtype=torch.long)
            Y[i, :] = torch.tensor(output_ctx, dtype=torch.long)

        print("Saving processed dataset to cache")
        if cache_path:
            cache_path.mkdir(exist_ok=True, parents=True)
            torch.save(X, cache_path / "train_x.pt")
            torch.save(Y, cache_path / "train_y.pt")

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

    return {"train": X_train, "val": X_val}, {"train": Y_train, "val": Y_val}


@torch.no_grad()
def estimate_loss_and_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    context_size: int,
    vocab_size: int,
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
        preds = logits.argmax(dim=2)
        loss = F.cross_entropy(
            logits.view(batch_size * context_size, vocab_size),
            Y_batch.view(batch_size * context_size),
        )
        assert preds.shape == Y_batch.shape
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
        head_size: int,
        context_size: int,
        num_heads: int,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.context_size = context_size
        self.num_heads = num_heads
        self.keys = nn.Linear(embedding_size, head_size * num_heads, bias=False)
        self.queries = nn.Linear(embedding_size, head_size * num_heads, bias=False)
        self.values = nn.Linear(embedding_size, head_size * num_heads, bias=False)
        self.use_flash_attention = use_flash_attention

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.linear = nn.Linear(head_size * num_heads, embedding_size, bias=False)

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
        keys = self.keys.forward(x).view(B, C, H, self.head_size).transpose(2, 1)
        queries = self.queries.forward(x).view(B, C, H, self.head_size).transpose(2, 1)
        assert tuple(keys.shape) == tuple(queries.shape) == (B, H, C, self.head_size)

        values = self.values.forward(x).reshape(B, C, H, self.head_size).transpose(2, 1)
        assert tuple(values.shape) == (B, H, C, self.head_size)

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
            scale_factor = 1 / self.head_size**0.5
            norm_masked_sa = F.softmax(masked_sa * scale_factor, dim=3)
            assert tuple(norm_masked_sa.shape) == (B, H, C, C)

            after_attention = norm_masked_sa @ values

        assert tuple(after_attention.shape) == (B, H, C, self.head_size)

        stacked = after_attention.transpose(2, 1).contiguous()
        assert tuple(stacked.shape) == (B, C, H, self.head_size)
        flat = self.flatten(stacked)
        assert tuple(flat.shape) == (B, C, H * self.head_size)
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
        head_size: int,
        context_size: int,
        num_heads: int,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.context_size = context_size
        self.num_heads = num_heads
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.multi_head_attention = MultiHeadSelfAttention(
            embedding_size=embedding_size,
            head_size=head_size,
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
        head_size: int,
        num_heads: int,
        num_blocks: int,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Learned positional encoding.
        self.positional_encoding = nn.Embedding(context_size, embedding_size)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_size=embedding_size,
                    head_size=head_size,
                    context_size=context_size,
                    num_heads=num_heads,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear = nn.Linear(embedding_size, vocab_size)
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
        B, C = x.shape
        emb = self.embedding.forward(x)
        pos = self.positional_encoding.forward(self.positions)
        drop = self.dropout(emb + pos)
        sa = self.attention_blocks(drop)
        out = self.linear(sa)
        assert tuple(out.shape) == (B, C, self.vocab_size)
        return out


def sample_from_model(
    model: MiniGPT,
    context_size: int,
    num_chars: int,
    vocab_size: int,
) -> list[int]:
    generated_tokens: list[int] = []
    current_ctx = [0] * context_size
    for _sample in range(num_chars):
        input = torch.tensor([current_ctx], dtype=torch.long).to(device)
        logits = model.forward(input)
        assert tuple(logits.shape) == (1, context_size, vocab_size)
        logits_last_token = logits[:, -1, :]
        assert tuple(logits_last_token.shape) == (1, vocab_size)
        probs = F.softmax(logits_last_token, dim=1)
        sample = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(sample.item())
        current_ctx = current_ctx[1:] + [sample.item()]

    return generated_tokens


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "require-cuda":
        if device != "cuda":
            print("Cuda not available on current system. Aborting")
            sys.exit(1)
    print(f"Device: {device}")
    print("Loading dataset...")
    context_size = 256
    tokenizer = Tokenizer.load(Path("tokenizer"))
    X, Y = load_dataset(
        context_size=context_size,
        tokenizer=tokenizer,
        path_corpus=Path("tinyshakespeare.txt"),
        cache_path=Path("dataset_caches/training"),
    )
    print(
        f"Done loading dataset. Train size = {len(X['train'])}, val size = {len(X['val'])}"
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size = {vocab_size}")
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=384,
        context_size=context_size,
        head_size=384 // 6,
        num_heads=6,
        num_blocks=6,
        use_flash_attention=True,
    )

    opt = AdamW(model.parameters(), lr=3e-4)
    max_training_iterations = 12_001
    model = train(model, X, Y, opt, max_training_iterations)

    model.eval()
    sampled_tokens = sample_from_model(
        model,
        context_size=model.context_size,
        num_chars=10000,
        vocab_size=model.vocab_size,
    )
    sample = tokenizer.decode(sampled_tokens)
    print(sample[:1000])
    Path("generated-sample.txt").write_text(sample)
    torch.save(model.state_dict(), "model-minigpt.pth")


def train(
    model: MiniGPT,
    X: Mapping[str, torch.Tensor],
    Y: Mapping[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    max_training_iterations: int,
):
    # Move the model to GPU. For nn.Module, .to(device) modifies in-place
    model.to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params // 1e6}M parameters")
    batch_size = 64

    train_losses = []
    validation_losses = []
    measure_every = 500

    for i in tqdm.tqdm(range(max_training_iterations)):
        perm = torch.randperm(len(X["train"]))[:batch_size]
        X_batch = X["train"][perm]
        Y_batch = Y["train"][perm]
        assert tuple(Y_batch.shape) == (batch_size, model.context_size)
        opt.zero_grad()
        logits = model.forward(X_batch)
        # Calculate the loss for each of the tokens in the input
        loss = F.cross_entropy(
            logits.view(batch_size * model.context_size, model.vocab_size),
            Y_batch.view(batch_size * model.context_size),
        )
        loss.backward()
        opt.step()

        if i % measure_every == 0:
            model.eval()
            train_loss_estimate, _ = estimate_loss_and_accuracy(
                model,
                X["train"],
                Y["train"],
                context_size=model.context_size,
                vocab_size=model.vocab_size,
            )
            val_loss_estimate, val_accuracy_estimate = estimate_loss_and_accuracy(
                model,
                X["val"],
                Y["val"],
                context_size=model.context_size,
                vocab_size=model.vocab_size,
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
    return model


if __name__ == "__main__":
    main()
