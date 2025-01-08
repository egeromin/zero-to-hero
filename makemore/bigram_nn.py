"""
Implement a bi-gram model by learning a probability distribution
using stochastic gradient descent, rather than counting frequencies.
"""

import sys

import torch
import torch.nn.functional as F

from bigram_counts import load_bigram_counts, sample_from_model


def learn_prob_matrix(
    bigrams: list[tuple[int, int]], num_chars: int, num_steps: int = 500
) -> torch.Tensor:
    # How do we learn? Through 1-hot encodings
    # one_hot(input) * N = one_hot(output)
    # What is the loss function?
    # Why can we interpret these as log counts?
    # Reasoning was: real values, rather than positive integers - and can take both positive and negative values.
    # And what is the loss? Negative log-likelihood = cross entropy.

    print(f"Number of bigrams: {len(bigrams)}")

    # Initialise the negative log counts.
    g = torch.Generator().manual_seed(2147483647)
    log_counts = torch.randn(
        size=(num_chars, num_chars), generator=g, requires_grad=True
    )
    batch_size = 1000

    inputs = F.one_hot(
        torch.tensor([b[0] for b in bigrams], dtype=torch.long), num_classes=num_chars
    ).float()
    labels = F.one_hot(
        torch.tensor([b[1] for b in bigrams], dtype=torch.long), num_classes=num_chars
    ).float()

    assert len(inputs) == len(bigrams) == len(labels)

    # Learn over mini-batches
    learning_rate = 50
    for i in range(num_steps):
        log_counts.grad = None

        # Grab a minibatch
        batch_idx = torch.randperm(len(inputs), generator=g)[:batch_size]
        inputs_batch = inputs[batch_idx]
        labels_batch = labels[batch_idx]
        assert labels_batch.shape == (batch_size, num_chars)

        logits_batch = inputs_batch @ log_counts
        log_probs_batch = F.log_softmax(logits_batch, dim=-1)
        assert log_probs_batch.shape == (batch_size, num_chars)  # 100 x 27

        loss = -(log_probs_batch * labels_batch).sum() / batch_size
        print(f"Loss at step {i}: {loss.item()}")
        loss.backward()
        log_counts.data -= learning_rate * log_counts.grad

    probs = F.softmax(log_counts, dim=-1)
    return probs


def main():
    if len(sys.argv) < 2:
        print("Usage: bigram_counts.py <num_samples_to_generate>")
        num_samples = 5
    else:
        num_samples = int(sys.argv[1])

    _counts, stoi, bigrams = load_bigram_counts()
    itos = {i: c for c, i in stoi.items()}

    # Compute the probability matrix. Each row should be a probability distribution
    probs = learn_prob_matrix(bigrams, num_chars=len(itos))

    # Sample from probability matrix.
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(num_samples):
        print(sample_from_model(probs, itos, g))


if __name__ == "__main__":
    main()
