"""
Implement a bi-gram model by keeping track of frequencies in a table.
"""

import itertools
import sys
from pathlib import Path
from typing import Mapping

import torch
from matplotlib import pyplot as plt


def load_bigram_counts() -> tuple[torch.Tensor, Mapping[str, int]]:
    """
    Parses the `names.txt` file to return the bi-gram counts,
    as well as the mapping from indices to characters.
    """
    all_names = Path("names.txt").read_text()
    stoi = {
        c: i
        for i, c in enumerate(
            itertools.chain.from_iterable((["."], sorted(set(all_names) - {"\n"})))
        )
    }

    counts = torch.zeros((len(stoi), len(stoi)), dtype=torch.int)
    for name in all_names.split():
        for c1, c2 in zip("." + name, name + "."):
            counts[stoi[c1], stoi[c2]] += 1

    return counts, stoi


def sample_from_model(probs: torch.Tensor):
    pass


def main():
    if len(sys.argv) < 2:
        print("Usage: bigram_counts.py <num_samples_to_generate>")
        num_samples = 5
    else:
        num_samples = int(sys.argv[1])

    counts, stoi = load_bigram_counts()
    itos = {i: c for c, i in stoi.items()}

    # Display the counts using matplotlib.
    # Copied straight from the lecture notebook.
    plt.figure(figsize=(16, 16))
    plt.imshow(counts, cmap="Blues")
    for i in sorted(stoi.values()):
        for j in sorted(stoi.values()):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, counts[i, j].item(), ha="center", va="top", color="gray")
    plt.show()

    # Compute the probability matrix.
    # P = ...

    # Sample from probability matrix.
    for _ in range(num_samples):
        pass


if __name__ == "__main__":
    main()
