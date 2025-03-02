from pathlib import Path
from typing import Protocol, Iterator

import torch


class TokenizerInterface(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class DataLoader:
    """
    Which properties do we want from the DataLoader?

    It should give a new batch of training samples + labels, when requested,
    using a specific tokenizer.
    It should loop through the training set.
    """

    def __init__(
        self,
        path_corpus: Path,
        tokenizer: TokenizerInterface,
        batch_size: int,
        context_size: int,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.context_size = context_size
        self.corpus = path_corpus.read_text()
        print("Encoding corpus...")
        self.tokens = tokenizer.encode(self.corpus)
        print("Done encoding corpus.")

    def __iter__(self) -> Iterator[tuple[torch.tensor, torch.tensor]]:
        remainder = (len(self.tokens) - 1) % self.context_size
        final_batch = None
        final_labels = None
        if remainder:
            print(f"There is a final batch of {remainder} tokens.")
            final_batch = torch.tensor([self.tokens[-remainder:-1]], dtype=torch.long)
            final_labels = torch.tensor(
                [self.tokens[-remainder + 1 :]], dtype=torch.long
            )

        inputs = torch.tensor(self.tokens[: -remainder - 1], dtype=torch.long).view(
            -1, self.context_size
        )
        labels = torch.tensor(self.tokens[1:-remainder], dtype=torch.long).view(
            -1, self.context_size
        )
        assert inputs.shape == labels.shape
        batch_start_idx: int = 0
        while batch_start_idx < len(inputs):
            yield (
                inputs[batch_start_idx : batch_start_idx + self.batch_size],
                labels[batch_start_idx : batch_start_idx + self.batch_size],
            )
            batch_start_idx += self.batch_size

        if final_batch is not None:
            yield final_batch, final_labels


# Testing
def main():
    from tokenizer import Tokenizer

    tokenizer = Tokenizer.load(Path("tokenizer"))
    loader = DataLoader(
        tokenizer=tokenizer,
        path_corpus=Path("tinyshakespeare.txt"),
        context_size=256,
        batch_size=64,
    )
    x, y = None, None
    for x, y in loader:
        print(tuple(x.shape), tuple(y.shape))

    if x is not None:
        print(x)
        print(y)


if __name__ == "__main__":
    main()
