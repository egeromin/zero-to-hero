from pathlib import Path
from typing import Protocol, Iterator, TypedDict

import numpy as np
import torch


class TokenizerInterface(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class DataLoader:

    def __init__(
        self,
        batch_size: int,
        context_size: int,
        tokens: list[int] | None = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        files: list[Path] = None,
    ):
        if tokens is None and files is None:
            raise ValueError("One of `tokens` or `files` must be provided.")
        if files is not None and tokens is not None:
            raise ValueError("Please provide at most one of `files` or `tokens`.")
        self.tokens = tokens or []
        self.batch_size = batch_size
        self.context_size = context_size
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.files = files
        self.current_file_idx = 0
        self.current_file = None
        self.max_tokens_to_load = 1e6
        self.current_global_pos = 0

    @property
    def n_tokens_in_batch(self) -> int:
        return self.batch_size * self.context_size

    @property
    def local_offset(self) -> int:
        return self.ddp_rank * self.n_tokens_in_batch

    @property
    def current_pos(self) -> int:
        return self.current_global_pos + self.local_offset

    @property
    def stride(self) -> int:
        return self.n_tokens_in_batch * self.ddp_world_size

    @property
    def num_tokens(self) -> int:
        remainder = len(self.tokens) % (self.batch_size * self.context_size)
        return len(self.tokens) - remainder

    @property
    def num_batches(self) -> int:
        return len(self.tokens) // (self.batch_size * self.context_size)

    def refresh_tokens(self):
        tokens = self.tokens[self.current_global_pos:]
        if self.files is None:
            tokens += self.tokens[:self.current_global_pos]
        else:
            while len(tokens) < self.max_tokens_to_load:
                if self.current_file is None:
                    self.current_file = self.files[self.current_file_idx].open("rb")
                new = np.fromfile(self.current_file, dtype=np.int16, count=self.max_tokens_to_load)
                tokens += new
                if len(new) == 0:
                    self.current_file.close()
                    self.current_file = None
                    self.current_file_idx = (self.current_file_idx + 1) % len(self.files)

        self.tokens = tokens
        self.current_global_pos = 0

    def __iter__(self) -> Iterator[tuple[torch.tensor, torch.tensor]]:

        while True:
            inputs = self.tokens[self.current_pos : self.current_pos + self.n_tokens_in_batch]
            labels = self.tokens[self.current_pos + 1 : self.current_pos + self.n_tokens_in_batch + 1]
            inputs_tensor = torch.tensor(inputs, dtype=torch.long).view(
                self.batch_size, self.context_size
            )
            labels_tensor = torch.tensor(labels, dtype=torch.long).view(
                self.batch_size, self.context_size
            )
            yield inputs_tensor, labels_tensor
            self.current_global_pos += self.stride
            if self.current_global_pos + self.stride + 1 > len(self.tokens):
                self.refresh_tokens()


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
            tokens=train_tokens,
            batch_size=batch_size,
            context_size=context_size,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        val_dataloader = DataLoader(
            tokens=val_tokens,
            batch_size=batch_size,
            context_size=context_size,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
    else:
        train_dataloader = DataLoader(
            tokens=tokens,
            batch_size=batch_size,
            context_size=context_size,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
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

        
def test_from_files():
    pass


if __name__ == "__main__":
    main()
