"""
Implement a model with variable context length using an MLP,
following the approach described in:
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Iterable

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# Steps:
# -1. Fix validation and test losses and accuracy to use no_grad.  ✅
# 0. Print a graph of training and validation losses over epochs.  ✅
# 1. Iterate on parameters to achieve a low loss.  ✅
# 2. Plot the activation, gradient and update statistics at each layer.
#     Start with the activations after the first tanh layer, bucketed.  ✅
#     Change to use density, like in the lecture.  ✅
#     Add gradient and update stats.  ✅
# 3. Pytorchify the code.  ✅
#     Add an additional tanh layer to test deeper networks, add activation and saturation stats at each stage.  ✅
#     Re-train the deeper NN end to end.  ✅
# 4. Fix the initialisation using manual scaling factors:  ✅
#     a. First, the initial loss ✅
#     b. Then, the saturated tanh ✅
# 5. Use kaiming initialisation and compare  ✅
# 6. Implement the variants of batch norm by hand and then show results.  ✅


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
        self.weights_grad = None  # Manual grad.
        self.bias: torch.Tensor | None = None
        self.bias_grad = None  # Manual grad.
        self.X = None  # For manual grad
        self.out_grad = None
        if bias:
            self.bias: torch.Tensor = torch.randn(
                size=(output_size,),
                requires_grad=True,
                generator=generator,
            )

    def __call__(
        self, X: torch.Tensor, training: bool = True, use_manual_grad: bool = False
    ) -> torch.Tensor:
        self.out = X @ self.weights
        if self.bias is not None:
            self.out += self.bias
        if training:
            self.X = X
            if not use_manual_grad:
                # The following are not "leaf tensors" and therefore must be explicitly instructed
                # to retain "tensor.grad".
                self.out.retain_grad()
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        yield self.weights
        if self.bias is not None:
            yield self.bias

    def param_grad(self) -> Iterable[torch.Tensor]:
        yield self.weights_grad
        if self.bias is not None:
            yield self.bias_grad

    # Assuming that we just have a stack of linear layers
    @torch.no_grad()
    def manual_backprop(self, output_grad: torch.Tensor) -> torch.Tensor:
        """Given the output grads with respect to the loss,
        returns the input grads with respect to the loss."""
        self.out_grad = output_grad
        self.weights_grad = self.X.T @ output_grad
        assert self.weights_grad.shape == self.weights.shape
        if self.bias is not None:
            self.bias_grad = output_grad.sum(dim=0)
            assert self.bias_grad.shape == self.bias.shape
        X_grad = output_grad @ self.weights.T
        return X_grad

    def manual_zero_grad(self):
        self.out_grad = None
        self.weights_grad = None
        self.bias_grad = None


class Tanh:
    def __init__(self):
        self.out = None
        self.out_grad = None

    def __call__(
        self, X: torch.Tensor, training: bool = True, use_manual_grad: bool = False
    ) -> torch.Tensor:
        self.out = X.tanh()
        if training and not use_manual_grad:
            self.out.retain_grad()
        return self.out

    def parameters(self) -> Iterable[torch.Tensor]:
        # Simple way to define an empty iterator
        return
        yield

    def param_grad(self) -> Iterable[torch.Tensor]:
        return
        yield

    @torch.no_grad()
    def manual_backprop(self, output_grad: torch.Tensor) -> torch.Tensor:
        self.out_grad = output_grad
        return output_grad * (1 - self.out**2)

    def manual_zero_grad(self):
        self.out_grad = None


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
        self.scale_grad = None
        self.shift = torch.randn((1, input_size), generator=generator) * 0.01
        self.shift.requires_grad = True
        self.shift_grad = None
        self.means_running = torch.zeros((1, input_size), dtype=torch.float)
        self.std_running = torch.ones((1, input_size), dtype=torch.float)
        self.out = None
        self.out_grad = None
        self.X = None
        self.X_grad = None
        self.means = None
        self.std = None
        self.std_grad = None
        self.normalised = None
        self.normalised_grad = None
        self.u = None
        self.v = None
        self.u_grad = None
        self.v_grad = None
        self.means_grad = None
        self.var_grad = None

    def __call__(
        self, X: torch.Tensor, training: bool = True, use_manual_grad: bool = False
    ) -> torch.Tensor:
        if training:
            self.X = X
            self.means = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True)
            self.v = self.std + self.eps
            self.u = X - self.means
            self.normalised = self.u / self.v
            self.out = self.normalised * self.scale + self.shift
            if not use_manual_grad:
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

    def param_grad(self) -> Iterable[torch.Tensor]:
        yield self.scale_grad
        yield self.shift_grad

    @torch.no_grad()
    def manual_backprop(self, output_grad: torch.Tensor) -> torch.Tensor:
        self.out_grad = output_grad
        self.scale_grad = (output_grad * self.normalised).sum(dim=0, keepdim=True)
        self.shift_grad = output_grad.sum(dim=0, keepdim=True)

        B = output_grad.shape[0]
        self.normalised_grad = output_grad * self.scale
        self.u_grad = self.normalised_grad / self.v
        self.v_grad = (
            -(self.normalised_grad * self.normalised).sum(dim=0, keepdim=True) / self.v
        )
        self.std_grad = self.v_grad
        self.var_grad = self.std_grad / (2 * self.std)
        self.means_grad = -self.u_grad.sum(dim=0, keepdim=True)
        self.X_grad = (
            self.u_grad + self.means_grad / B + self.var_grad * self.u * 2 / (B - 1)
        )
        return self.X_grad

    def manual_zero_grad(self):
        self.out_grad = None
        self.scale_grad = None
        self.shift_grad = None
        self.normalised_grad = None
        self.u_grad = None
        self.v_grad = None
        self.std_grad = None
        self.var_grad = None
        self.means_grad = None
        self.X_grad = None


class MLP:
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_size: int,
        hidden_size: int,
        g: torch.Generator,
    ):
        self.embedding: torch.Tensor = torch.rand(
            size=(vocab_size, embedding_size), requires_grad=True, generator=g
        )
        self.embedding_grad = None
        self.x = None
        self.selected_embeddings = None
        self.selected_embeddings_grad = None
        self.layers = [
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

    def forward(
        self, x: torch.Tensor, training: bool = True, use_manual_grad: bool = False
    ) -> torch.Tensor:
        # Forward pass. Do not compute the final softmax,
        # as this will be calculated inside the loss function,
        # for numerical stability.
        self.x = x
        self.selected_embeddings = self.embedding[x]
        selected_embeddings_reshaped = self.selected_embeddings.view(
            self.selected_embeddings.shape[0], -1
        )
        output = selected_embeddings_reshaped
        for layer in self.layers:
            output = layer(output, training=training, use_manual_grad=use_manual_grad)
        return output

    @torch.no_grad()
    def manual_backprop(self, logits: torch.Tensor, Y: torch.Tensor):
        # Backward pass with respect to the logits,
        batch_size = Y.shape[0]

        logits_grad = F.softmax(logits, dim=-1)
        for i, yi in enumerate(Y):
            logits_grad[i, yi] -= 1.0
        logits_grad /= batch_size

        # Uncomment the following for debug
        # assert torch.allclose(logits_grad, logits.grad)

        # Backward pass through all remaining layers
        output_grad = logits_grad
        for layer in reversed(self.layers):
            output_grad = layer.manual_backprop(output_grad)

        # Backward pass through the embedding
        selected_embeddings_reshaped_grad = output_grad
        self.selected_embeddings_grad = selected_embeddings_reshaped_grad.view(
            *self.selected_embeddings.shape
        )

        # Accumulate the gradients for each character
        self.embedding_grad = torch.zeros(size=self.embedding.shape, dtype=torch.float)
        for i, idx_chars in enumerate(self.x):  # 0 <= i < batch_size
            for j, idx_char in enumerate(idx_chars):  # 0 <= j < context_size
                self.embedding_grad[idx_char] += self.selected_embeddings_grad[i, j]

    def parameters(self) -> Iterable[torch.Tensor]:
        yield self.embedding
        for layer in self.layers:
            yield from layer.parameters()

    def param_grad(self) -> Iterable[torch.Tensor]:
        yield self.embedding_grad
        for layer in self.layers:
            yield from layer.param_grad()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def manual_zero_grad(self):
        for layer in self.layers:
            layer.manual_zero_grad()
        self.embedding_grad = None

    def update_parameters(self, learning_rate: float):
        for parameter in self.parameters():
            parameter.data -= learning_rate * parameter.grad

    def manual_update_parameters(self, learning_rate: float):
        for p, p_grad in zip(self.parameters(), self.param_grad()):
            p.data -= learning_rate * p_grad


@torch.no_grad()
def calculate_loss_and_accuracy(
    mlp: MLP, X: torch.Tensor, Y: torch.Tensor
) -> tuple[float, float]:
    """Returns a tuple (loss, accuracy)"""
    logits = mlp.forward(X, training=False)
    predictions = logits.argmax(dim=-1)
    assert predictions.shape == Y.shape
    accuracy = sum(predictions == Y) / Y.shape[0]
    loss = calculate_loss(mlp, logits, Y)
    return loss.item(), accuracy


def calculate_loss(mlp: MLP, logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # reg_alpha = 0.001
    model_loss = F.cross_entropy(logits, Y)
    # params = iter(mlp.parameters())
    # next(params)  # exclude embedding from regularization
    # reg_loss = reg_alpha * sum((param**2).sum() for param in params if param.dim() == 2)
    # loss = model_loss + reg_loss
    loss = model_loss
    return loss


def train_model(
    mlp: MLP,
    X: torch.Tensor,
    Y: torch.Tensor,
    g: torch.Generator,
    use_manual_grad: bool = False,
) -> MLP:
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
    num_training_iterations = 4_001
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

        mlp.zero_grad()

        if use_manual_grad:
            mlp.manual_zero_grad()

        logits_batch = mlp.forward(
            X_batch, use_manual_grad=use_manual_grad, training=True
        )
        loss = calculate_loss(mlp, logits_batch, Y_batch)
        batch_losses.append(loss.log10().item())

        # Backward pass. Use either autograd,
        # or manually implemented backward pass,
        # depending on input arguments.
        if use_manual_grad:
            mlp.manual_backprop(logits_batch, Y_batch)
        else:
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
                for k, layer in enumerate(mlp.layers):
                    if isinstance(layer, Tanh):
                        legends.append(f"Layer {k}")
                        activations = layer.out.view(-1)
                        hy, hx = torch.histogram(activations, density=True)
                        axes[plot_num][0].plot(hx[:-1].detach(), hy.detach())

                        if use_manual_grad:
                            gradients = layer.out_grad.view(-1)
                        else:
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
            for k, layer in enumerate(mlp.layers):
                if isinstance(layer, Linear):
                    weights = next(p for p in layer.parameters() if p.dim() == 2)
                    if use_manual_grad:
                        weights_grad = next(
                            p_grad for p_grad in layer.param_grad() if p_grad.dim() == 2
                        )
                    else:
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

        if use_manual_grad:
            mlp.manual_update_parameters(learning_rate)
        else:
            mlp.update_parameters(learning_rate)

        # Calculate the validation accuracy
        if i % 4000 == 0:
            val_loss, val_accuracy = calculate_loss_and_accuracy(mlp, X_val, Y_val)
            val_losses.append((i, math.log10(val_loss)))
            print(
                f"Step {i}: validation loss = {val_loss:.4f}, "
                f"validation accuracy = {val_accuracy * 100:.2f}%"
            )

    # Calculate the final test accuracy.
    test_loss, test_accuracy = calculate_loss_and_accuracy(mlp, X_test, Y_test)
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

    return mlp


@torch.no_grad()
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
            logits = mlp.forward(x, training=False)
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
        hidden_size=50,
        g=g,
    )
    use_manual_grad = True

    if use_manual_grad:
        with torch.no_grad():
            mlp = train_model(mlp, X, Y, g, use_manual_grad=True)
    else:
        mlp = train_model(mlp, X, Y, g, use_manual_grad=False)
    sample_from_model(mlp, g=g, num_samples=20, context_size=context_size, stoi=stoi)


def test_manual_backprop():
    g = torch.Generator().manual_seed(2147483647)
    input_size = 11
    output_size = 19
    batch_size = 32
    layers = [
        Linear(input_size, 13, generator=g),
        BatchNormID(13, generator=g),
        Tanh(),
        Linear(13, 17, generator=g),
        BatchNormID(17, generator=g),
        Tanh(),
        Linear(17, output_size, generator=g),
    ]
    x = torch.rand(batch_size, input_size, generator=g, requires_grad=True)
    y = torch.tensor(list(range(output_size)) + list(range(batch_size - output_size)))
    output = x
    for layer in layers:
        output = layer(output)
    logits = output
    logits.retain_grad()
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # Backward pass with respect to the logits,
    # based on derivative of the cross entropy formula
    logits_grad = F.softmax(logits, dim=-1)
    for i, yi in enumerate(y):
        logits_grad[i, yi] -= 1.0
    logits_grad /= batch_size

    assert logits_grad.shape == logits.grad.shape
    assert torch.allclose(logits.grad, logits_grad)

    # Backward pass through each of the layers.
    output_grad = logits_grad
    for layer in reversed(layers):
        assert output_grad.shape == layer.out.shape == layer.out.grad.shape
        try:
            assert torch.allclose(output_grad, layer.out.grad)
        except AssertionError:
            print(layer.out.grad[0, :4])
            print(output_grad[0, :4])
            raise
        output_grad = layer.manual_backprop(output_grad)
        if isinstance(layer, Linear):
            assert layer.bias_grad.shape == layer.bias.shape == layer.bias.grad.shape
            assert (
                layer.weights_grad.shape
                == layer.weights.shape
                == layer.weights.grad.shape
            )
            assert torch.allclose(
                layer.bias_grad, layer.bias.grad, rtol=1e-5, atol=1e-5
            )
            assert torch.allclose(layer.weights_grad, layer.weights.grad)
        if isinstance(layer, BatchNormID):
            assert layer.scale_grad.shape == layer.scale.shape == layer.scale.grad.shape
            assert layer.shift_grad.shape == layer.shift.shape == layer.shift.grad.shape
            assert torch.allclose(layer.scale_grad, layer.scale.grad)
            assert torch.allclose(layer.shift_grad, layer.shift.grad)

    # Final check
    assert output_grad.shape == x.grad.shape
    assert torch.allclose(output_grad, x.grad)


def test_manual_backprop_mlp():
    context_size = 3
    X, Y, stoi = load_dataset(context_size=context_size)

    g = torch.Generator().manual_seed(2147483647)
    mlp = MLP(
        vocab_size=len(stoi),
        context_size=context_size,
        embedding_size=11,
        hidden_size=50,
        g=g,
    )
    mlp.zero_grad()
    mlp.manual_zero_grad()

    # Test on a batch of size 32
    perm_idx = torch.randperm(len(X), generator=g)[:32]
    X = X[perm_idx]
    Y = Y[perm_idx]

    logits = mlp.forward(X, training=True)
    loss = calculate_loss(mlp, logits, Y)

    # Backward in two different ways
    loss.backward()
    # store the autograd gradients before running backward manually
    autograd_gradients = [p.grad.clone() for p in mlp.parameters()]

    mlp.manual_backprop(logits, Y)

    # Test that the two methods give the same result, for all params
    assert len(autograd_gradients) == len(list(mlp.param_grad()))
    for p_auto_grad, p_grad in zip(autograd_gradients, mlp.param_grad()):
        print("-------------------------------------------------------")
        print(p_auto_grad.shape, p_grad.shape)
        print(p_auto_grad.view(-1)[:6])
        print(p_grad.view(-1)[:6])
        print(p_auto_grad.view(-1)[:6] / p_grad.view(-1)[:6])
        assert p_auto_grad.shape == p_grad.shape
        assert torch.allclose(p_auto_grad, p_grad, rtol=1e-5, atol=1e-5)
        print("ok")


if __name__ == "__main__":
    main()
