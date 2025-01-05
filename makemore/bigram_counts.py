"""
Implement a bi-gram model by keeping track of frequencies in a table.
"""

import itertools
import sys
from pathlib import Path
from typing import Mapping

import torch


def load_bigram_counts() -> tuple[torch.Tensor, Mapping[str, int]]:
    """
    Parses the `names.txt` file to return the bi-gram counts,
    as well as the mapping from indices to characters.
    """
    all_names = Path("names.txt").read_text()
    stoi = {
        c: i
        for i, c in enumerate(
            itertools.chain.from_iterable((["."], sorted(set(all_names))))
        )
    }

    counts = torch.zeros((len(stoi), len(stoi)), dtype=torch.int)
    for name in all_names.split("\n"):
        for c1, c2 in zip("." + name, name):
            print(c1, c2)
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

    # Compute the probability matrix.
    # P = ...

    # Sample from probability matrix.
    for _ in range(num_samples):
        pass


if __name__ == "__main__":
    main()
