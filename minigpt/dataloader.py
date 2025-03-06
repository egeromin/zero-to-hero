from pathlib import Path
from typing import Protocol, Iterator, TypedDict

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        shuffle: bool = True,
        use_final_batch: bool = True,
    ):
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_size = context_size
        self.shuffle = shuffle

        remainder = (len(self.tokens) - 1) % self.context_size
        self.final_batch = None
        self.final_labels = None
        if remainder and use_final_batch:
            print(f"There is a final batch of {remainder} tokens.")
            final_batch = torch.tensor([self.tokens[-remainder:-1]], dtype=torch.long)
            final_labels = torch.tensor(
                [self.tokens[-remainder + 1 :]], dtype=torch.long
            )
            self.final_batch, self.final_labels = (
                final_batch.to(device),
                final_labels.to(device),
            )

        inputs = torch.tensor(self.tokens[: -remainder - 1], dtype=torch.long).view(
            -1, self.context_size
        )
        labels = torch.tensor(self.tokens[1:-remainder], dtype=torch.long).view(
            -1, self.context_size
        )
        assert inputs.shape == labels.shape
        if self.shuffle:
            perm = torch.randperm(len(inputs))
            inputs = inputs[perm]
            labels = labels[perm]
        self.inputs, self.labels = inputs.to(device), labels.to(device)

    @property
    def num_tokens(self) -> int:
        num = self.labels.shape[0] * self.labels.shape[1]
        if self.final_batch is not None:
            num += self.final_batch.shape[1]
        return int(num)

    @property
    def num_batches(self) -> int:
        num = len(self.labels)
        if self.final_batch is not None:
            num += 1
        return num

    def __iter__(self) -> Iterator[tuple[torch.tensor, torch.tensor]]:
        batch_start_idx: int = 0
        while True:
            yield (
                self.inputs[batch_start_idx : batch_start_idx + self.batch_size],
                self.labels[batch_start_idx : batch_start_idx + self.batch_size],
            )
            batch_start_idx += self.batch_size
            if batch_start_idx >= len(self.inputs):
                if self.final_batch is not None:
                    yield self.final_batch, self.final_labels
                batch_start_idx = 0


class TrainValDataloaders(TypedDict):
    train: DataLoader
    val: DataLoader | None


def dataloaders_from_corpus(
    path_corpus: Path,
    tokenizer: TokenizerInterface,
    batch_size: int,
    context_size: int,
    val_split: float | None = None,
    use_final_batch: bool = True,
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
            train_tokens,
            batch_size=batch_size,
            context_size=context_size,
            use_final_batch=use_final_batch,
        )
        val_dataloader = DataLoader(
            val_tokens,
            batch_size=batch_size,
            context_size=context_size,
            use_final_batch=use_final_batch,
        )
    else:
        train_dataloader = DataLoader(
            tokens,
            batch_size=batch_size,
            context_size=context_size,
            use_final_batch=use_final_batch,
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
