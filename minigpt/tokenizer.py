from collections import defaultdict
from pathlib import Path
import os
from typing import Mapping

import regex
import sentencepiece as spm
import tiktoken


# TODOs:
# Simplify the implementation to use `merges` and `vocab` dictionary.  ✅
# Fix decoding invalid tokens using the "replace" mechanism  ✅
# Implement the GPT-2 regex for splitting and tokenizing each chunk separately. ✅
# Play around with sentencepiece and use it to tokenize ✅
# Play around with tiktoken - specifically, reproduce the SolidGoldMagikarp issue  ✅
# Implement the GPT4 tokenizer and compare to tiktoken
# Try implementing adding a new token to an existing mini-GPT model and then finetuning it.


class Tokenizer:
    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # gpt-4 pattern
    pat = regex.compile(pat_str)

    def __init__(
        self,
        merges: list[tuple[int, tuple[int, int]]],
        vocab: Mapping[int, bytes],
    ):
        self.merges = merges
        self.vocab = vocab

    @classmethod
    def train(cls, input_text: str, target_vocab_size: int):
        """Train the tokenizer on a given input text
        at initialization."""
        vocab_size = 256
        merges = []
        vocab = {i: bytes([i]) for i in range(vocab_size)}
        text_chunks = cls.pat.findall(input_text)
        tokens_per_chunk = [
            [int(b) for b in chunk.encode("utf-8")] for chunk in text_chunks
        ]
        while vocab_size < target_vocab_size:
            top_tuple = cls.find_most_common_pair(tokens_per_chunk)
            if top_tuple is None:
                print("No more tokens to merge, aborting prematurely")
                break
            tokens_per_chunk = [
                cls._substitute(chunk, list(top_tuple), [vocab_size])
                for chunk in tokens_per_chunk
            ]
            merges.append((vocab_size, top_tuple))
            vocab_size += 1

        # Finish building the vocabulary, for faster decoding.
        for token, original_pair in merges:
            vocab[token] = vocab[original_pair[0]] + vocab[original_pair[1]]

        return cls(merges=merges, vocab=vocab)

    @classmethod
    def _get_pair_counts(
        cls,
        tokens: list[int],
        existing_pair_counts: dict[tuple[int, int], int] | None = None,
    ) -> dict[tuple[int, int], int]:
        pair_counts = existing_pair_counts or defaultdict(int)
        for a, b in zip(tokens, tokens[1:]):
            pair_counts[(a, b)] += 1
        return pair_counts

    @classmethod
    def find_most_common_pair(
        cls, tokens_per_chunk: list[list[int]]
    ) -> tuple[int, int] | None:
        """Find the most common pair of tokens"""
        pair_counts = defaultdict(int)
        for chunk in tokens_per_chunk:
            pair_counts = cls._get_pair_counts(chunk, existing_pair_counts=pair_counts)
        tuples_sorted = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        top_tuple = next(iter(tuples_sorted))[0]
        return top_tuple

    @classmethod
    def _substitute(
        cls, tokens: list[int], pattern: list[int], replacement: list[int]
    ) -> list[int]:
        i = 0
        output_tokens = []
        while i < len(tokens):
            if (
                i < len(tokens) - len(pattern) + 1
                and tokens[i : i + len(pattern)] == pattern
            ):
                output_tokens.extend(replacement)
                i += len(pattern)
            else:
                output_tokens.append(tokens[i])
                i += 1
        return output_tokens

    def encode_chunk(self, text_chunk: str) -> list[int]:
        tokens = [int(b) for b in text_chunk.encode("utf-8")]
        for token, original_pair in self.merges:
            tokens = self._substitute(tokens, list(original_pair), [token])
        return tokens

    def encode(self, text: str) -> list[int]:
        text_chunks = self.pat.findall(text)
        tokens = []
        for chunk in text_chunks:
            tokens.extend(self.encode_chunk(chunk))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token] for token in tokens)
        return token_bytes.decode("utf-8", errors="replace")


def try_sentencepiece():
    options = dict(
        # input spec
        input="blogpost.txt",
        input_format="text",
        # output spec
        model_prefix="tok500",  # output filename prefix
        # algorithm spec
        # BPE alg
        model_type="bpe",
        vocab_size=500,
        # normalization
        normalization_rule_name="identity",  # ew, turn off normalization
        remove_extra_whitespaces=False,
        input_sentence_size=200000000,  # max number of training sentences
        max_sentence_length=4192,  # max number of bytes per sentence
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.99995,
        byte_fallback=True,
        # merge rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0,  # the UNK token MUST exist
        bos_id=1,  # the others are optional, set to -1 to turn off
        eos_id=2,
        pad_id=-1,
        # systems
        num_threads=os.cpu_count(),  # use ~all system resources
    )
    spm.SentencePieceTrainer.train(**options)
    sp = spm.SentencePieceProcessor()
    sp.load("tok500.model")
    print([(sp.id_to_piece(i), i) for i in range(sp.get_piece_size())])
    ids = sp.encode("hello 안녕하세요")
    print(ids)
    print([(sp.id_to_piece(i), i) for i in ids])


def main():
    input_text = Path("blogpost.txt").read_text()
    tokenizer = Tokenizer.train(input_text, 300)

    text_to_encode = input_text
    encoded = tokenizer.encode(text_to_encode)
    compression_ratio = len(encoded) / len(input_text)
    print(
        f"Encoded text has length: {len(encoded)}, compression ratio: {compression_ratio * 100:.0f}%"
    )
    decoded = tokenizer.decode(encoded)
    assert text_to_encode == decoded

    print(tokenizer.decode([129]))


def compare_tiktoken():
    enc = tiktoken.get_encoding("cl100k_base")  # this is the GPT-4 tokenizer
    test_text = "hello world!!!? (안녕하세요!) ZOINK ✅"
    ids = enc.encode(test_text)
    text = enc.decode(ids)  # get the same text back
    print(text)

    train_text = Path("blogpost.txt").read_text()
    tokenizer = Tokenizer.train(train_text, target_vocab_size=500)
    ids = tokenizer.encode(test_text)
    text = tokenizer.decode(ids)
    # print(enc._mergeable_ranks)
    print(text)


if __name__ == "__main__":
    compare_tiktoken()
