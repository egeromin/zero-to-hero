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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Iterable

import torch
import torch.nn.functional as F
import tqdm
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW

from dataloader import dataloaders_from_corpus, TrainValDataloaders, DataLoader
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
    loader: DataLoader,
    num_runs: int = 20,
) -> tuple[float, float]:
    losses = []
    accuracies = []
    for _, (X_batch, Y_batch) in zip(range(num_runs), loader):
        logits = model.forward(X_batch)
        preds = logits.argmax(dim=2)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            Y_batch.view(-1),
        )
        assert preds.shape == Y_batch.shape
        accuracy = (preds == Y_batch).float().mean().item()
        losses.append(loss.item())
        accuracies.append(accuracy)
    mean_loss = sum(losses) / len(losses)
    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_loss, mean_accuracy


@dataclass(eq=True, frozen=True)
class MiniGPTConfig:
    vocab_size: int
    max_context_length: int
    embedding_size: int
    head_size: int
    num_heads: int
    num_blocks: int
    use_flash_attention: bool = False
    attention_bias: bool = False
    final_layer_bias: bool = True
    final_layer_norm: bool = False
    ffw_use_gelu: bool = False


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

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.attn_c = nn.Linear(
            config.embedding_size,
            config.head_size * config.num_heads * 3,
            bias=config.attention_bias,
        )
        self.use_flash_attention = config.use_flash_attention

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.linear = nn.Linear(
            config.head_size * config.num_heads,
            config.embedding_size,
            bias=config.attention_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, E = x.shape
        assert E == self.config.embedding_size
        H = self.config.num_heads
        queries, keys, values = self.attn_c.forward(x).split(
            self.config.head_size * H, dim=2
        )
        queries = queries.view(B, C, H, self.config.head_size).transpose(2, 1)
        keys = keys.view(B, C, H, self.config.head_size).transpose(2, 1)
        values = values.view(B, C, H, self.config.head_size).transpose(2, 1)

        assert (
            tuple(keys.shape)
            == tuple(queries.shape)
            == tuple(values.shape)
            == (B, H, C, self.config.head_size)
        )
        assert tuple(values.shape) == (B, H, C, self.config.head_size)

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
            mask = ~torch.tril(torch.ones(C, C, dtype=torch.bool))
            masked_sa = torch.where(mask, -torch.inf, sa)
            assert tuple(masked_sa.shape) == (B, H, C, C)
            scale_factor = 1 / self.config.head_size**0.5
            norm_masked_sa = F.softmax(masked_sa * scale_factor, dim=3)
            assert tuple(norm_masked_sa.shape) == (B, H, C, C)

            after_attention = norm_masked_sa @ values

        assert tuple(after_attention.shape) == (B, H, C, self.config.head_size)

        stacked = after_attention.transpose(2, 1).contiguous()
        assert tuple(stacked.shape) == (B, C, H, self.config.head_size)
        flat = self.flatten(stacked)
        assert tuple(flat.shape) == (B, C, H * self.config.head_size)
        output = self.linear(flat)
        assert tuple(output.shape) == (B, C, E)
        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int, use_gelu: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.act = nn.GELU(approximate="tanh") if use_gelu else nn.ReLU()
        self.linear_2 = nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.linear_1(x)
        rl = self.act(l1)
        return self.linear_2(rl)


class AttentionBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.norm_1 = nn.LayerNorm(config.embedding_size)
        self.multi_head_attention = MultiHeadSelfAttention(config)
        self.drop_1 = nn.Dropout(p=DROPOUT)
        self.norm_2 = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(
            config.embedding_size, use_gelu=config.ffw_use_gelu
        )
        self.drop_2 = nn.Dropout(p=DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = self.drop_1(self.multi_head_attention(self.norm_1(x))) + x
        output = self.drop_2(self.feed_forward(self.norm_2(multi))) + multi
        assert output.shape == x.shape
        return output


class MiniGPT(torch.nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.embedding_size
        )
        # Learned positional encoding.
        self.positional_encoding = nn.Embedding(
            self.config.max_context_length, self.config.embedding_size
        )
        self.dropout = nn.Dropout(p=DROPOUT)
        self.attention_blocks = nn.Sequential(
            *[AttentionBlock(self.config) for _ in range(self.config.num_blocks)]
        )
        if self.config.final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(self.config.embedding_size)
        else:
            self.final_layer_norm = nn.Identity()
        self.linear = nn.Linear(
            self.config.embedding_size,
            self.config.vocab_size,
            bias=self.config.final_layer_bias,
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
        positions = torch.arange(C)
        pos = self.positional_encoding.forward(positions)
        hidden_state = emb + pos
        drop = self.dropout(hidden_state)
        sa = self.attention_blocks(drop)
        out = self.linear(self.final_layer_norm(sa))
        assert tuple(out.shape) == (B, C, self.config.vocab_size)
        return out


def sample_from_model(
    model: MiniGPT,
    num_chars: int,
    start_ctx: list[int] | None = None,
) -> Iterable[int]:
    current_ctx = start_ctx or [0]
    for _sample in range(num_chars):
        current_ctx = current_ctx[-model.config.max_context_length :]
        input = torch.tensor([current_ctx], dtype=torch.long).to(device)
        logits = model.forward(input)
        assert tuple(logits.shape) == (1, len(current_ctx), model.config.vocab_size)
        logits_last_token = logits[:, -1, :]
        assert tuple(logits_last_token.shape) == (1, model.config.vocab_size)
        probs = F.softmax(logits_last_token, dim=1)
        sample = torch.multinomial(probs, num_samples=1).item()
        yield sample
        current_ctx.append(sample)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "require-cuda":
        if device != "cuda":
            print("Cuda not available on current system. Aborting")
            sys.exit(1)
    print(f"Device: {device}")
    print("Loading dataset...")
    context_size = 256
    batch_size = 64
    tokenizer = Tokenizer.load(Path("tokenizer"))
    loaders = dataloaders_from_corpus(
        context_size=context_size,
        tokenizer=tokenizer,
        path_corpus=Path("tinyshakespeare.txt"),
        batch_size=batch_size,
        val_split=0.2,
    )
    print(
        f"Done loading dataset. "
        f"Train size = {len(loaders['train'].labels)}, "
        f"val size = {len(loaders['val'].labels)}"
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size = {vocab_size}")
    config = MiniGPTConfig(
        vocab_size=vocab_size,
        embedding_size=384,
        max_context_length=context_size,
        head_size=384 // 6,
        num_heads=6,
        num_blocks=6,
        use_flash_attention=True,
    )
    model = MiniGPT(config)
    opt = AdamW(model.parameters(), lr=3e-4)
    max_training_iterations = 8_001
    model = train(model, loaders, opt, max_training_iterations)

    model.eval()

    start_ctx = tokenizer.encode("I'm a language model,")
    tokens = []
    tokens.extend(start_ctx)

    written_text = ""
    for token in sample_from_model(model, start_ctx=start_ctx, num_chars=10000):
        tokens.append(token)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        sys.stdout.write(decoded[len(written_text) :])
        written_text = decoded

    sample = tokenizer.decode(tokens)
    Path("generated-sample.txt").write_text(sample)
    torch.save(model.state_dict(), "model-minigpt.pth")


def train(
    model: MiniGPT,
    loaders: TrainValDataloaders,
    opt: torch.optim.Optimizer,
    max_training_iterations: int,
):
    # Move the model to GPU. For nn.Module, .to(device) modifies in-place
    model.to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params // 1e6}M parameters")

    train_losses = []
    validation_losses = []
    measure_every = 500

    for i, (X_batch, Y_batch) in tqdm.tqdm(
        zip(range(max_training_iterations), loaders["train"])
    ):
        assert tuple(Y_batch.shape)[1] == loaders["train"].context_size
        opt.zero_grad()
        logits = model.forward(X_batch)
        # Calculate the loss for each of the tokens in the input
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            Y_batch.view(-1),
        )
        train_losses.append(loss)
        loss.backward()
        opt.step()

        if i % measure_every == 0:
            model.eval()
            val_loss_estimate, val_accuracy_estimate = estimate_loss_and_accuracy(
                model,
                loaders["val"],
            )
            validation_losses.append(val_loss_estimate)
            train_loss_estimate = sum(train_losses[-20:]) / 20
            print(
                f"\n{i}: train loss = {train_loss_estimate:4f}, "
                f"val loss = {val_loss_estimate:4f}, "
                f"val accuracy = {val_accuracy_estimate * 100:.2f}%"
            )
            model.train()

    print(f"Number of parameters: {total_params // 1e6}M parameters")
    # Plot training and validation losses
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))
    average_every = 20
    remainder = len(train_losses) % average_every
    average_train_losses = (
        torch.tensor(train_losses[:-remainder]).view(-1, average_every).mean(dim=1)
    )
    train_losses_log10 = [math.log10(e) for e in average_train_losses]
    axes[0].plot(
        [i * average_every for i in range(len(average_train_losses))],
        train_losses_log10,
    )
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Log10 Train loss")
    axes[0].set_title("Log10 training losses during the training")

    val_losses_log10 = [math.log10(e) for e in validation_losses]
    val_iteration = [i * measure_every for i in range(len(validation_losses))]
    axes[1].plot(val_iteration, val_losses_log10)
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Log10 Validation loss")
    axes[1].set_title("Log10 validation losses during the training")
    plt.tight_layout()

    if device != "cuda":
        plt.show()
    else:
        plt.savefig("training-plots.png", dpi=300)
    return model


def test_works_after_refactor():
    """Small sanity check that I can do .forward() even after making
    some small changes to the code.
    """
    print(f"Device: {device}")
    tokenizer = Tokenizer.load(Path("tokenizer"))
    max_context_length = 1024
    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_size=384,
        max_context_length=max_context_length,
        head_size=384 // 6,
        num_heads=6,
        num_blocks=6,
        use_flash_attention=False,
    )
    model = MiniGPT(config)
    input_context = torch.zeros((1, max_context_length // 2), dtype=torch.long)
    model.forward(input_context)
    print("OK, forward successful.")
    prompt = "Hello!"
    sampled_tokens = list(
        sample_from_model(
            model,
            start_ctx=tokenizer.encode(prompt),
            num_chars=5,
        )
    )
    sample = tokenizer.decode(sampled_tokens)
    print(prompt + sample)
    print("OK, sampling successful.")


if __name__ == "__main__":
    main()
