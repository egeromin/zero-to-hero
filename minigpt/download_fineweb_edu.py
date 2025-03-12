"""
Download the fineweb-edu dataset.
"""

import io
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer


def encode_doc(doc: dict) -> list[int]:
    tokenizer = get_tokenizer()
    eot = tokenizer._special_tokens["<|endoftext|>"]
    tokens = [eot]
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    return tokens


def save_tokens(tokens: list[int], file: io.FileIO):
    tokens_np = np.array(tokens, dtype=np.uint16)
    tokens_np.tofile(file)


def download_dataset():
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    dataset_dir = Path("fineweb-edu")
    dataset_dir.mkdir(exist_ok=True)

    # 100M tokens per shard, 100 shards in total
    max_tokens_per_shard = int(1e8)
    current_shard = 0
    num_tokens_in_shard = 0
    progress_bar = tqdm(
        total=max_tokens_per_shard, unit="tokens", desc=f"Shard {current_shard}"
    )
    remaining_tokens = []
    file_shard = (dataset_dir / "val-0.npy").open("wb")

    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        for tokens in pool.imap(encode_doc, fw, chunksize=32):
            tokens = remaining_tokens + tokens
            remaining_capacity = max_tokens_per_shard - num_tokens_in_shard
            remaining_tokens = tokens[remaining_capacity:]
            to_write = tokens[:remaining_capacity]
            save_tokens(to_write, file_shard)
            num_tokens_in_shard += len(to_write)
            progress_bar.update(len(to_write))
            if num_tokens_in_shard >= max_tokens_per_shard:
                file_shard.flush()
                file_shard.close()
                current_shard = current_shard + 1
                file_shard = (dataset_dir / f"train-{current_shard}.npy").open("wb")
                num_tokens_in_shard = 0
                progress_bar = tqdm(
                    total=max_tokens_per_shard,
                    unit="tokens",
                    desc=f"Shard {current_shard}",
                )


if __name__ == "__main__":
    download_dataset()
