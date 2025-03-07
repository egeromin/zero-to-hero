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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tiktoken
import torch
import torch.nn.functional as F
import tqdm
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW

from dataloader import dataloaders_from_corpus, TrainValDataloaders, DataLoader

torch.manual_seed(1337)
DROPOUT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed(1337)
    print(f"Using {device}, matplotlib in Agg mode")
    matplotlib.use("Agg")
    torch.set_float32_matmul_precision("high")


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
        self.linear.SCALE_BY_NUM_BLOCKS = True

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
            mask = mask.to(device)
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
        self.linear_2.SCALE_BY_NUM_BLOCKS = True

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
        self.linear.weight = self.embedding.weight  # parameter sharing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            scale_factor = 1
            if hasattr(module, "SCALE_BY_NUM_BLOCKS"):
                # Downscale the weights for projections that contribute
                # towards the residual, depending on the total number of blocks,
                # and therefore of residual connections.
                scale_factor = (2 * self.config.num_blocks) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale_factor)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        emb = self.embedding.forward(x)
        positions = torch.arange(C).to(device)
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


def gpt2_learning_rate_schedule(
    step: int, warmup_steps: int, max_training_iterations: int
):
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * (step + 1 / warmup_steps)
    if step > max_training_iterations:
        return min_lr

    cosine_decay_factor = 0.5 * (
        1.0
        + math.cos(
            math.pi * (step - warmup_steps) / (max_training_iterations - warmup_steps)
        )
    )
    return min_lr + (max_lr - min_lr) * cosine_decay_factor


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "require-cuda":
        if device != "cuda":
            print("Cuda not available on current system. Aborting")
            sys.exit(1)
    print(f"Device: {device}")
    print("Loading dataset...")
    total_batch_size = 524288
    context_size = 1024
    batch_size = 16
    assert total_batch_size % (context_size * batch_size) == 0
    # tokenizer = Tokenizer.load(Path("tokenizer"))  # my own tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # use tiktoken
    loaders = dataloaders_from_corpus(
        context_size=context_size,
        tokenizer=tokenizer,
        path_corpus=Path("tinyshakespeare.txt"),
        batch_size=batch_size,
        val_split=0.2,
    )
    print(
        f"Done loading dataset. "
        f"Train size = {loaders['train'].num_batches} batches, {loaders['train'].num_tokens} tokens "
        f"Val size = {loaders['val'].num_batches} batches, {loaders['val'].num_tokens} tokens "
    )

    vocab_size = 50304
    assert vocab_size >= tokenizer.n_vocab
    assert vocab_size % 128 == 0
    print(f"Vocab size = {vocab_size}")
    config = MiniGPTConfig(
        vocab_size=vocab_size,
        embedding_size=768,
        max_context_length=context_size,
        head_size=768 // 12,
        num_heads=12,
        num_blocks=12,
        use_flash_attention=True,
        attention_bias=True,
        final_layer_bias=False,
        final_layer_norm=True,
        ffw_use_gelu=True,
    )
    model = MiniGPT(config)
    opt = initialize_optimizer(model)
    # max_training_iterations = 8_001
    max_training_iterations = 50
    model = train(model, loaders, opt, max_training_iterations, total_batch_size)

    # model.eval()
    #
    # start_ctx = tokenizer.encode("I'm a language model,")
    # tokens = []
    # tokens.extend(start_ctx)
    #
    # written_text = ""
    # for token in sample_from_model(model, start_ctx=start_ctx, num_chars=10000):
    #     tokens.append(token)
    #     decoded = tokenizer.decode(tokens)
    #     sys.stdout.write(decoded[len(written_text) :])
    #     written_text = decoded
    #
    # sample = tokenizer.decode(tokens)
    # Path("generated-sample.txt").write_text(sample)
    # torch.save(model.state_dict(), "model-minigpt.pth")


def initialize_optimizer(model: MiniGPT) -> torch.optim.Optimizer:
    decay_params = [
        p for _pn, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    nodecay_params = [
        p for _pn, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    weight_decay = 0.1
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    opt = AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    return opt


def train(
    model: MiniGPT,
    loaders: TrainValDataloaders,
    opt: torch.optim.Optimizer,
    max_training_iterations: int,
    total_batch_size: int | None = None,
):
    # Move the model to GPU. For nn.Module, .to(device) modifies in-place
    model.to(device)
    model = torch.compile(model)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params // 1e6}M parameters")

    train_losses = []
    validation_losses = []
    measure_every = 500
    warmup_steps = 10
    batch_size = loaders["train"].batch_size

    if total_batch_size is None:
        total_batch_size = batch_size * model.config.max_context_length

    grad_accum_steps = total_batch_size // (
        batch_size * model.config.max_context_length
    )
    print(f"Training with {grad_accum_steps} gradient accumulation steps")

    train_loader_iter = iter(loaders["train"])

    for i in tqdm.tqdm(range(max_training_iterations)):
        start = time.time()
        opt.zero_grad()
        loss_accum = 0.0
        num_accum_steps = 0
        while num_accum_steps < grad_accum_steps:
            X_batch, Y_batch = next(train_loader_iter)
            assert X_batch.shape == Y_batch.shape
            if (shape := tuple(X_batch.shape)) != (
                batch_size,
                model.config.max_context_length,
            ):
                # TODO: bleah, should move this directly to the data loader.
                print(f"Skipping batch with {shape=}, would slow down torch.compile")
                continue
            num_accum_steps += 1
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model.forward(X_batch)
                # Calculate the loss for each of the tokens in the input)
                loss = F.cross_entropy(
                    logits.view(-1, model.config.vocab_size),
                    Y_batch.view(-1),
                )
                loss = loss / grad_accum_steps
                loss.backward()
                loss_accum += loss.detach()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_losses.append(loss_accum)
        learning_rate = gpt2_learning_rate_schedule(
            i, warmup_steps, max_training_iterations
        )
        for param_group in opt.param_groups:
            param_group["lr"] = learning_rate
        opt.step()
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        n_tok = X_batch.shape[0] * X_batch.shape[1] * grad_accum_steps
        tok_ps = n_tok / elapsed

        train_loss_estimate = sum(train_losses[-20:]) / len(train_losses[-20:])
        print(
            f"\n{i}: train loss = {train_loss_estimate:4f}, "
            f"time = {elapsed * 1000:.0f}ms, "
            f"tok/s = {tok_ps:.0f}, "
            f"norm = {norm:.4f}, "
            f"lr = {learning_rate:.6f}, "
        )

        # if i % measure_every == 0:
        #     model.eval()
        #     val_loss_estimate, val_accuracy_estimate = estimate_loss_and_accuracy(
        #         model,
        #         loaders["val"],
        #     )
        #     validation_losses.append(val_loss_estimate)
        #     print(
        #         f"val loss = {val_loss_estimate:4f}, "
        #         f"val accuracy = {val_accuracy_estimate * 100:.2f}%"
        #     )
        #     model.train()

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


# def test_works_after_refactor():
#     """Small sanity check that I can do .forward() even after making
#     some small changes to the code.
#     """
#     print(f"Device: {device}")
#     tokenizer = Tokenizer.load(Path("tokenizer"))
#     max_context_length = 1024
#     vocab_size = 50304
#     assert vocab_size >= tokenizer.vocab_size
#     assert vocab_size % 128 == 0
#     config = MiniGPTConfig(
#         vocab_size=vocab_size,
#         embedding_size=384,
#         max_context_length=max_context_length,
#         head_size=384 // 6,
#         num_heads=6,
#         num_blocks=6,
#         use_flash_attention=False,
#     )
#     model = MiniGPT(config)
#     input_context = torch.zeros((1, max_context_length // 2), dtype=torch.long)
#     model.forward(input_context)
#     print("OK, forward successful.")
#     prompt = "Hello!"
#     sampled_tokens = list(
#         sample_from_model(
#             model,
#             start_ctx=tokenizer.encode(prompt),
#             num_chars=5,
#         )
#     )
#     sample = tokenizer.decode(sampled_tokens)
#     print(prompt + sample)
#     print("OK, sampling successful.")


if __name__ == "__main__":
    main()
