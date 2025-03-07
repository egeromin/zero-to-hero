from pathlib import Path
from typing import Protocol, Iterator, TypedDict

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
        tokens: list[int],
        batch_size: int,
        context_size: int,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ):
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_size = context_size
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

    @property
    def num_tokens(self) -> int:
        remainder = len(self.tokens) % (self.batch_size * self.context_size)
        return len(self.tokens) - remainder

    @property
    def num_batches(self) -> int:
        return len(self.tokens) // (self.batch_size * self.context_size)

    def __iter__(self) -> Iterator[tuple[torch.tensor, torch.tensor]]:
        start_pos = self.ddp_rank * self.batch_size * self.context_size
        current_pos = start_pos
        stride = self.batch_size * self.context_size * self.ddp_world_size

        while True:
            inputs = self.tokens[current_pos:current_pos + stride]
            labels = self.tokens[current_pos+1:current_pos + stride+1]
            inputs_tensor = torch.tensor(inputs, dtype=torch.long).view(self.batch_size, self.context_size)
            labels_tensor = torch.tensor(labels, dtype=torch.long).view(self.batch_size, self.context_size)
            yield inputs_tensor, labels_tensor
            current_pos += stride
            if current_pos >= len(self.tokens):
                current_pos = start_pos


class TrainValDataloaders(TypedDict):
    train: DataLoader
    val: DataLoader | None


def dataloaders_from_corpus(
    path_corpus: Path,
    tokenizer: TokenizerInterface,
    batch_size: int,
    context_size: int,
    val_split: float | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> TrainValDataloaders:
    corpus = path_corpus.read_text()
    print("Encoding corpus...")
    tokens = tokenizer.encode(corpus)
    print("Done encoding corpus.")
    if val_split is not None:
        split = int(len(tokens) * (1.0 - val_split))
        train_tokens = tokens[:split]
        val_tokens = tokens[split:]
        train_dataloader = DataLoader(
            train_tokens, batch_size=batch_size, context_size=context_size, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
        )
        val_dataloader = DataLoader(
            val_tokens, batch_size=batch_size, context_size=context_size, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
        )
    else:
        train_dataloader = DataLoader(
            tokens, batch_size=batch_size, context_size=context_size, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
        )
        val_dataloader = None
    return {"train": train_dataloader, "val": val_dataloader}


# Testing
def main():
    from tokenizer import Tokenizer

    tokenizer = Tokenizer.load(Path("tokenizer"))
    loader = dataloaders_from_corpus(
        tokenizer=tokenizer,
        path_corpus=Path("tinyshakespeare.txt"),
        context_size=256,
        batch_size=64,
        val_split=0.1,
    )["train"]
    x, y = None, None
    for _, (x, y) in zip(range(20), loader):
        print(tuple(x.shape), tuple(y.shape))

    if x is not None:
        print(x)
        print(y)


if __name__ == "__main__":
    main()
