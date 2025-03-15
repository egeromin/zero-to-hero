import json
from collections import defaultdict
from pathlib import Path
import os
from typing import Mapping

import regex
import sentencepiece as spm
import tiktoken
from tqdm import tqdm
import pickle


# TODOs:
# Simplify the implementation to use `merges` and `vocab` dictionary.  ✅
# Fix decoding invalid tokens using the "replace" mechanism  ✅
# Implement the GPT-2 regex for splitting and tokenizing each chunk separately. ✅
# Play around with sentencepiece and use it to tokenize ✅
# Play around with tiktoken - specifically, reproduce the SolidGoldMagikarp issue  ✅
# Implement the GPT4 tokenizer and compare to tiktoken ✅
# Try implementing adding a new token to an existing mini-GPT model and then finetuning it. ✅
#     To do this, we have to extend the embedding table, and then also extend the logits.
#     Can freeze the original weights and then train only the new ones.
#     To test this, first re-train the mini-Shakespeare dataset, but using the tokenizer and a large-ish
#     target vocab size. And then, add a special token between different plays or otherwise breaking
#     up the input corpus. Fine tune on that.


class Tokenizer:
    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # gpt-4 pattern
    pat = regex.compile(pat_str)

    def __init__(
        self,
        merges: list[tuple[int, tuple[int, int]]],
        vocab: dict[int, bytes],
        byte_mapping: Mapping[int, int],
        special_tokens: dict[str, int] | None = None,
    ):
        # To ensure ordering, we require `merges` as a list.
        self.merges = {a: tuple(b) for a, b in merges}
        self.merges_inverted = {tuple(b): a for a, b in merges}
        self.vocab = vocab
        self.byte_mapping = byte_mapping
        self.special_tokens = special_tokens or {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def n_vocab(self) -> int:
        # For compatibility with tiktoken
        return self.vocab_size

    def save(self, target_dir: Path):
        if not target_dir.exists():
            target_dir.mkdir()
        elif not target_dir.is_dir():
            raise RuntimeError(f"{str(target_dir)} exists and is not a directory")
        merges_ordered = list(self.merges.items())
        (target_dir / "merges.json").write_text(json.dumps(merges_ordered))
        (target_dir / "vocab.bin").write_bytes(pickle.dumps(self.vocab))
        (target_dir / "byte_mapping.bin").write_bytes(pickle.dumps(self.byte_mapping))
        (target_dir / "special_tokens.json").write_text(json.dumps(self.special_tokens))

    @classmethod
    def load(cls, model_dir: Path):
        merges = json.loads((model_dir / "merges.json").read_text())
        vocab = pickle.loads((model_dir / "vocab.bin").read_bytes())
        byte_mapping = pickle.loads((model_dir / "byte_mapping.bin").read_bytes())
        special_tokens = json.loads((model_dir / "special_tokens.json").read_text())
        return cls(merges, vocab, byte_mapping, special_tokens)

    @classmethod
    def train(cls, input_text: str, target_vocab_size: int):
        """Train the tokenizer on a given input text
        at initialization."""
        vocab_size = 256
        merges = []
        byte_mapping = {i: i for i in range(vocab_size)}
        vocab = {i: bytes([i]) for i in range(vocab_size)}
        text_chunks = cls.pat.findall(input_text)
        tokens_per_chunk = [
            [int(b) for b in chunk.encode("utf-8")] for chunk in text_chunks
        ]
        start_vocab_size = vocab_size
        for _ in tqdm(range(target_vocab_size - start_vocab_size)):
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

        return cls(merges=merges, vocab=vocab, byte_mapping=byte_mapping)

    @classmethod
    def from_tiktoken(cls, enc: tiktoken.Encoding):
        mergeable_ranks = enc._mergeable_ranks  # inverse of vocab

        def pair_rank(pair: tuple[bytes, bytes]) -> int | None:
            return mergeable_ranks.get(b"".join(pair))

        def bpe_on_token(token: bytes, max_rank: int) -> tuple[bytes, bytes]:
            # Perform byte-pair encoding on the token, constraining
            # the vocabulary of merges to be those with rank
            # strictly lower than max_rank.
            parts = [bytes([b]) for b in token]
            while len(parts) > 2:
                possible_pairs = list(enumerate(zip(parts[:-1], parts[1:])))
                pairs_sorted = sorted(
                    possible_pairs, key=lambda x: pair_rank(x[1]) or max_rank
                )
                min_idx, min_pair = pairs_sorted[0]
                min_rank = pair_rank(min_pair)
                if min_rank >= max_rank:
                    break
                parts = (
                    parts[:min_idx]
                    + [parts[min_idx] + parts[min_idx + 1]]
                    + parts[min_idx + 2 :]
                )
            assert len(parts) == 2
            return tuple(parts)

        byte_mapping = {}
        merges_dict = {}
        for token_str, rank in mergeable_ranks.items():
            if len(token_str) == 1:
                byte_mapping[token_str[0]] = rank
            else:
                # In this case, we have a byte-sequence of length >1,
                # and we must figure out how it splits into sub-chunks.
                # To do this, we can run byte-pair encoding on the byte-string,
                # observing that the split will the two sub-strings that are
                # left after performing byte-pair encoding using all merges
                # of lower rank. To perform this byte-pair encoding, we can merge
                # adjacent parts iteratively, starting with those of lowest rank.
                pair_as_bytes = bpe_on_token(token_str, rank)
                p_rank = pair_rank(pair_as_bytes)
                pair = (
                    mergeable_ranks[pair_as_bytes[0]],
                    mergeable_ranks[pair_as_bytes[1]],
                )
                merges_dict[pair] = p_rank

        pairs_sorted = sorted(merges_dict.keys(), key=lambda x: merges_dict[x])
        merges = [(merges_dict[pair], pair) for pair in pairs_sorted]
        vocab = {r: bytes([b]) for b, r in byte_mapping.items()}
        for token, original_pair in merges:
            vocab[token] = vocab[original_pair[0]] + vocab[original_pair[1]]

        return cls(merges=merges, vocab=vocab, byte_mapping=byte_mapping)

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
        if not pair_counts:
            return None
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
        tokens = [self.byte_mapping[b] for b in text_chunk.encode("utf-8")]
        # A naive approach would be to iterate through all of the pairs in
        # `merges` and replace tokens one by one. However, this is inefficient,
        # for a large vocabulary. So, instead, given that chunks are short,
        # we can restrict ourselves to iterate through the pairs that actually
        # occur in that chunk.
        while len(tokens) > 1:
            current_pairs = self._get_pair_counts(tokens).keys()
            mergeable_pairs = list(
                filter(lambda pair: pair in self.merges_inverted, current_pairs)
            )
            if not mergeable_pairs:
                # Already merged everything we can.
                break
            lowest_rank_pair = min(
                mergeable_pairs, key=lambda pair: self.merges_inverted[pair]
            )
            merge_token = self.merges_inverted[lowest_rank_pair]
            tokens = self._substitute(tokens, list(lowest_rank_pair), [merge_token])
        return tokens

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        if verbose:
            print("Encoding. Finding chunks...")
        # Split the text into chunks. First, split by special tokens.
        if self.special_tokens:
            escaped_candidates = [regex.escape(c) for c in self.special_tokens]
            special_pattern_str = r"(" + "|".join(escaped_candidates) + r")"
            special_pattern = regex.compile(special_pattern_str)
            chunk_by_special = special_pattern.split(text)
        else:
            chunk_by_special = [text]

        # Now, split using the GPT-4 pattern, the usual way
        text_chunks = []
        for special_chunk in chunk_by_special:
            if special_chunk:
                if special_chunk in self.special_tokens:
                    # Unless it's a special token, in which case, replace it.
                    text_chunks.append(special_chunk)
                else:
                    text_chunks.extend(self.pat.findall(special_chunk))
        if verbose:
            print("Done finding chunks.")
        tokens = []
        if verbose:
            text_chunks = tqdm(text_chunks)
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                tokens.append(self.special_tokens[chunk])
            else:
                tokens.extend(self.encode_chunk(chunk))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token] for token in tokens)
        return token_bytes.decode("utf-8", errors="replace")

    def add_token(self, token_str: str):
        if token_str in self.special_tokens:
            print(f"Token {token_str} already added, nothing to do.")
            return

        self.special_tokens[token_str] = self.vocab_size
        self.vocab[self.vocab_size] = token_str.encode("utf-8")


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
    print(text)

    tokenizer = Tokenizer.from_tiktoken(enc)
    encoded = enc.encode(test_text)
    print(tokenizer.decode(encoded))
    encoded = tokenizer.encode(test_text)
    print(enc.decode(encoded))


def train_shakespeare():
    """Train the tokenizer on the shakespeare dataset."""
    train_text = Path("tinyshakespeare.txt").read_text()
    tokenizer = Tokenizer.train(train_text, target_vocab_size=10000)
    tokenizer.save(Path("tokenizer"))
    test_text = "hello world!!!? (안녕하세요!) ZOINK ✅"
    ids = tokenizer.encode(test_text, verbose=True)

    t2 = Tokenizer.load(Path("tokenizer"))
    assert t2.decode(ids) == test_text
    assert t2.encode(test_text) == ids


def encode_shakespeare():
    """How long does it take to encode the full shakespeare dataset?"""
    tokenizer = Tokenizer.load(Path("tokenizer"))
    text = Path("tinyshakespeare.txt").read_text()
    ids = tokenizer.encode(text, verbose=True)
    print("Done encoding")
    decoded = tokenizer.decode(ids)
    print("Done decoding")
    assert decoded == text

    # Test adding a new token to the vocabulary.
    token_str = "<|endoftext|>"
    test_str = "Testing with " + token_str + " surrounded by extra text."
    print(tokenizer.encode(test_str))
    tokenizer.add_token(token_str)
    print(enc := tokenizer.encode(test_str))
    print(tokenizer.decode(enc))


if __name__ == "__main__":
    encode_shakespeare()
