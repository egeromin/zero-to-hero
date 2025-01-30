"""
Implement the wavenet from lecture 5
"""

import math
from collections import defaultdict
from typing import Mapping, Iterable, Protocol

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mlp import load_dataset


# Steps:
# 1. Simplify code, remove duplicates, remove manual backprop, remove test functions  ✅
# 2. Implement Embedding layer and FlattenConsecutive layer  ✅
# 3. Fix other layers to use 3-D tensors as inputs  ✅
# 4. Refactor code to use Embedding and FlattenConsecutive - make MLP alike one of the check containers in pytorch. ✅
# 5. Test that results are the same for 32K
# 6. Implement wavenet using a context of size 8 and compare performance after 32K
# 7. Compare to a similarly sized 'MLP' model.


class LayerProtocol(Protocol):
    def __call__(self, X: torch.Tensor, training: bool = True) -> torch.Tensor: ...

    def parameters(self) -> Iterable[torch.Tensor]: ...


# Define the layers we'll use: Linear, Tanh and Softmax
class Linear:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        generator: torch.Generator,
        bias: bool = True,
        gain: float = 1.0,
        use_kaiming: bool = True,
    ):
        if use_kaiming:
            kaiming_factor = input_size**-0.5
        else:
            kaiming_factor = 1.0
        self.weights: torch.Tensor = (
            torch.randn(
                size=(input_size, output_size),
                generator=generator,
            )
            * gain
            * kaiming_factor
        )
        self.weights.requires_grad = True
        self.bias: torch.Tensor | None = None
        self.X = None  # For manual grad
        if bias:
            self.bias: torch.Tensor = torch.randn(
                size=(output_size,),
                requires_grad=True,
                generator=generator,
            )

    def __call__(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        self.out = X @ self.weights
        if self.bias is not None:
            self.out += self.bias
        if training:
            # The following are not "leaf tensors" and therefore must be explicitly instructed
            # to retain "tensor.grad".
            self.out.retain_grad()
            self.X = X
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        yield self.weights
        if self.bias is not None:
            yield self.bias


class Tanh:
    def __init__(self):
        self.out = None

    def __call__(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        self.out = X.tanh()
        if training:
            self.out.retain_grad()
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        # Simple way to define an empty iterator
        return
        yield


class BatchNormID:
    """
    We want the input of the tanh to be approximately Gaussian
    So, we can normalise each input in the batch so that the batch
    is approximately Gaussian.
    Each *neuron* should be approximately Gaussian, so we normalise
    each neuron separately, by calculating its std and mean across
    the mini-batch.
    """

    def __init__(
        self, input_size: int, generator: torch.Generator, momentum: float = 0.999
    ):
        self.input_size = input_size
        self.eps = 1e-8
        self.momentum = momentum
        self.scale = (
            torch.ones((1, input_size), dtype=torch.float)
            + torch.randn((1, input_size), generator=generator) * 0.01
        )
        self.scale.requires_grad = True
        self.shift = torch.randn((1, input_size), generator=generator) * 0.01
        self.shift.requires_grad = True
        self.means_running = torch.zeros((1, input_size), dtype=torch.float)
        self.std_running = torch.ones((1, input_size), dtype=torch.float)
        self.out = None
        self.X = None
        self.means = None
        self.std = None
        self.normalised = None
        self.u = None
        self.v = None

    def __call__(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            self.X = X
            self.means = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True)
            self.v = self.std + self.eps
            self.u = X - self.means
            self.normalised = self.u / self.v
            self.out = self.normalised * self.scale + self.shift
            self.out.retain_grad()

            with torch.no_grad():
                self.means_running = (
                    self.momentum * self.means_running
                    + (1.0 - self.momentum) * self.means
                )
                self.std_running = (
                    self.momentum * self.std_running + (1.0 - self.momentum) * self.std
                )
        else:
            self.u = X - self.means_running
            self.v = self.std_running + self.eps
            self.normalised = self.u / self.v
            self.out = self.normalised * self.scale + self.shift
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        yield self.scale
        yield self.shift


class Embedding:
    def __init__(
        self, vocab_size: int, embedding_size: int, generator: torch.Generator
    ):
        self.out = None
        self.embedding: torch.Tensor = torch.rand(
            size=(vocab_size, embedding_size), requires_grad=True, generator=generator
        )

    def __call__(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        self.out = self.embedding[X]
        if training:
            self.out.retain_grad()
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        yield self.embedding


class FlattenConsecutive:
    def __init__(self, stride: int):
        self.out = None
        self.stride = stride

    def __call__(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        self.out = X.view(
            X.shape[0], X.shape[1] // self.stride, X.shape[2] * self.stride
        )
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
        if training:
            self.out.retain_grad()
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        # Simple way to define an empty iterator
        return
        yield


class Sequential:
    # Not a completely clean API, but OK for now.

    def __init__(self, layers: list[LayerProtocol]):
        self.layers = layers

    def forward(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        out = X
        for layer in self.layers:
            out = layer(out, training=training)
        return out

    def parameters(self) -> Iterable[torch.Tensor]:
        for layer in self.layers:
            yield from layer.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def update_parameters(self, learning_rate: float):
        for parameter in self.parameters():
            parameter.data -= learning_rate * parameter.grad


@torch.no_grad()
def calculate_loss_and_accuracy(
    model: Sequential, X: torch.Tensor, Y: torch.Tensor
) -> tuple[float, float]:
    """Returns a tuple (loss, accuracy)"""
    logits = model.forward(X, training=False)
    predictions = logits.argmax(dim=-1)
    assert predictions.shape == Y.shape
    accuracy = sum(predictions == Y) / Y.shape[0]
    loss = calculate_loss(model, logits, Y)
    return loss.item(), accuracy


def calculate_loss(
    model: Sequential, logits: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    # reg_alpha = 0.001
    model_loss = F.cross_entropy(logits, Y)
    # params = iter(mlp.parameters())
    # next(params)  # exclude embedding from regularization
    # reg_loss = reg_alpha * sum((param**2).sum() for param in params if param.dim() == 2)
    # loss = model_loss + reg_loss
    loss = model_loss
    return loss


def train_model(
    model: Sequential,
    X: torch.Tensor,
    Y: torch.Tensor,
    g: torch.Generator,
) -> Sequential:
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

    # Prepare axes for the mega debug plots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 16))

    # num_training_iterations = 200_001
    # While experimenting, revert to a lower number of iterations.
    num_training_iterations = 32_001
    val_losses: list[tuple[int, float]] = []
    batch_losses: list[float] = []
    update_ratios: Mapping[int, list[float]] = defaultdict(list)
    for i in range(num_training_iterations):
        # Grab a minibatch.
        batch_size = 32
        batch_idx = torch.randperm(len(X_train), generator=g)[:batch_size]
        X_batch = X_train[batch_idx]
        Y_batch = Y_train[batch_idx]
        assert X_batch.shape == (batch_size, X_train.shape[1])
        assert Y_batch.shape == (batch_size,)

        model.zero_grad()

        logits_batch = model.forward(X_batch)
        loss = calculate_loss(model, logits_batch, Y_batch)
        batch_losses.append(loss.log10().item())

        loss.backward()

        # Heuristic learning rate
        learning_rate = 0.04 if i < 100_000 else 0.004

        # Calculate statistics for plotting
        with torch.no_grad():
            if i in (0, num_training_iterations - 1):
                plot_num = 0 if i == 0 else 1

                # Plot gradient and activation statistics
                axes[plot_num][0].set_title(
                    f"Activations of hidden layer at iteration {i}"
                )
                axes[plot_num][0].set_xlabel("Activation value")
                axes[plot_num][0].set_ylabel("Activation density")
                axes[plot_num][1].set_title(
                    f"Gradients of hidden layer at iteration {i}"
                )
                axes[plot_num][1].set_xlabel("Gradient value")
                axes[plot_num][1].set_ylabel("Gradient density")
                legends = []
                for k, layer in enumerate(model.layers):
                    if isinstance(layer, Tanh):
                        legends.append(f"Layer {k}")
                        activations = layer.out.view(-1)
                        hy, hx = torch.histogram(activations, density=True)
                        axes[plot_num][0].plot(hx[:-1].detach(), hy.detach())

                        gradients = layer.out.grad.view(-1)
                        hy, hx = torch.histogram(gradients, density=True)
                        axes[plot_num][1].plot(hx[:-1].detach(), hy.detach())

                        # Print mean, std, and saturation statistics
                        print(
                            f"Layer {k} mean = {activations.mean().item()}, std = {activations.std().item()}, "
                            f"saturation %  = {(activations > 0.97).float().mean().item() * 100:.2f}"
                        )
                        print(
                            f"Layer {k} gradient mean = {gradients.mean().item()}, std = {gradients.std().item()}, "
                            f"saturation %  = {(gradients > 0.97).float().mean().item() * 100:.2f}"
                        )
                axes[plot_num][0].legend(legends)
                axes[plot_num][1].legend(legends)

            # Keep track of update / weight ratio statistics for each iteration
            # Calculate the ratio of the std-dev of the gradients,
            # to the std-dev of the weights. We want to get a sense of whether the
            # updates we're doing are the right size.
            # Why the std-dev? Because, we get a measure of the range of values
            # that are taken, since weights and their grads will be both positive
            # and negative, symmetrically, since we're using tanh.
            # Alternatively, could also use the ratio of mean absolute values
            # to achieve a similar plot
            for k, layer in enumerate(model.layers):
                if isinstance(layer, Linear):
                    weights = next(p for p in layer.parameters() if p.dim() == 2)
                    weights_grad = next(
                        p.grad for p in layer.parameters() if p.dim() == 2
                    )
                    ratio = learning_rate * weights_grad.std() / weights.std()
                    # variant, also works:
                    # ratio = learning_rate * weights.grad.abs().mean() / weights.abs().mean()
                    # The following does not work so well, since it's too much influenced by outliers:
                    # ratios = learning_rate * weights.grad.view(-1) / weights.view(-1)
                    # ratio = ratios.mean()
                    update_ratios[k].append(ratio.log10().item())

        model.update_parameters(learning_rate)

        # Calculate the validation accuracy
        if i % 4000 == 0:
            val_loss, val_accuracy = calculate_loss_and_accuracy(model, X_val, Y_val)
            val_losses.append((i, math.log10(val_loss)))
            print(
                f"Step {i}: validation loss = {val_loss:.4f}, "
                f"validation accuracy = {val_accuracy * 100:.2f}%"
            )

    # Calculate the final test accuracy.
    test_loss, test_accuracy = calculate_loss_and_accuracy(model, X_test, Y_test)
    print(
        f"Final test loss = {test_loss:.4f}, test accuracy = {test_accuracy * 100:.2f}%"
    )

    axes[2][0].plot(range(len(batch_losses)), batch_losses)
    axes[2][0].set_xlabel("Training iteration")
    axes[2][0].set_ylabel("Log10 Train loss")
    axes[2][0].set_title("Log10 training losses during the training")

    x, y = zip(*val_losses)
    axes[2][1].plot(x, y)
    axes[2][1].set_xlabel("Training iteration")
    axes[2][1].set_ylabel("Log10 Validation loss")
    axes[2][1].set_title("Log10 validation losses during the training")

    legends = []
    axes[3][0].set_xlabel("Training iteration")
    axes[3][0].set_ylabel("Average update ratio at hidden layer")
    axes[3][0].set_title("Average update ratios")
    for k, layer_update_ratios in update_ratios.items():
        legends.append(f"Layer {k}")
        axes[3][0].plot(range(len(layer_update_ratios)), layer_update_ratios)
    axes[3][0].legend(legends)

    plt.tight_layout()
    plt.show()

    return model


@torch.no_grad()
def sample_from_model(
    model: Sequential,
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
            logits = model.forward(x, training=False)
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
    vocab_size = len(stoi)
    context_size = context_size
    embedding_size = 11
    hidden_size = 50
    model = Sequential(
        [
            Embedding(vocab_size, embedding_size, generator=g),
            FlattenConsecutive(stride=context_size),
            Linear(embedding_size * context_size, hidden_size, generator=g, gain=5 / 3),
            BatchNormID(hidden_size, generator=g),
            Tanh(),
            Linear(hidden_size, hidden_size, generator=g, gain=5 / 3),
            BatchNormID(hidden_size, generator=g),
            Tanh(),
            Linear(hidden_size, hidden_size, generator=g, gain=5 / 3),
            BatchNormID(hidden_size, generator=g),
            Tanh(),
            Linear(hidden_size, vocab_size, generator=g, gain=1.0),
        ]
    )
    model = train_model(model, X, Y, g)
    sample_from_model(model, g=g, num_samples=20, context_size=context_size, stoi=stoi)


if __name__ == "__main__":
    main()
