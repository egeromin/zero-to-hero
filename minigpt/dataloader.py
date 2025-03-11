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
        files: list[Path] | None = None,
        max_tokens_to_load: int | None = None,
        files_dtype: np.dtype = np.uint16,
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
        self.max_tokens_to_load = max_tokens_to_load or int(1e6)
        self.current_global_pos = 0
        self.files_dtype = files_dtype

        assert self.max_tokens_to_load >= self.n_tokens_in_batch, (
            "Need to load more tokens than number of tokens in batch."
        )

        self.refresh_tokens()

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
        tokens = self.tokens[self.current_global_pos :]
        if self.files is None:
            tokens += self.tokens[: self.current_global_pos]
        else:
            while len(tokens) < self.max_tokens_to_load:
                if self.current_file is None:
                    self.current_file = self.files[self.current_file_idx].open("rb")
                new = list(
                    np.fromfile(
                        self.current_file,
                        dtype=self.files_dtype,
                        count=self.max_tokens_to_load,
                    )
                )
                tokens += new
                if len(new) == 0:
                    self.current_file.close()
                    self.current_file = None
                    self.current_file_idx = (self.current_file_idx + 1) % len(
                        self.files
                    )

        self.tokens = tokens
        self.current_global_pos = 0

    def __iter__(self) -> Iterator[tuple[torch.tensor, torch.tensor]]:
        while True:
            inputs = self.tokens[
                self.current_pos : self.current_pos + self.n_tokens_in_batch
            ]
            labels = self.tokens[
                self.current_pos + 1 : self.current_pos + self.n_tokens_in_batch + 1
            ]
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
    dataloader = DataLoader(
        files=[Path(f"test-dataloader/batch_{i}.npy") for i in range(10)],
        batch_size=16,
        context_size=1024,
        max_tokens_to_load=459812,
        files_dtype=np.uint32,
    )
    train = []
    labels = []
    max_to_fetch = 1000000
    dataloader_it = iter(dataloader)
    while len(train) < max_to_fetch:
        train_t, labels_t = next(dataloader_it)
        train += train_t.view(-1).tolist()
        labels += labels_t.view(-1).tolist()
    train = train[:max_to_fetch]
    labels = labels[:max_to_fetch]

    # Print the first diff
    for i in range(len(train)):
        if train[i] != i:
            print(f"{i}th position of 'train' is {train[i]}")
            break

    assert train == list(range(max_to_fetch))

    # Print the first diff
    for i in range(len(labels)):
        if labels[i] != i + 1:
            print(f"{i}th position of 'labels' is {labels[i]}")
            break

    assert labels == list(range(1, max_to_fetch)) + [0]

    # TODO: test with different values of world_size and ddp_rank


if __name__ == "__main__":
    test_from_files()
