from collections import defaultdict
from pathlib import Path


# TODOs:
# Simplify the implementation to use `merges` and `vocab` dictionary.  ✅
# Fix decoding invalid tokens using the "replace" mechanism  ✅
# Implement the GPT-2 regex for splitting and tokenizing each chunk separately.
# Play around with tiktoken - specifically, reproduce the SolidGoldMagikarp issue
# Play around with sentencepiece and use it to tokenize
# Try implementing adding a new token to an existing mini-GPT model and then finetuning it.


class Tokenizer:
    def __init__(self, input_text: str, target_vocab_size: int):
        """Train the tokenizer on a given input text
        at initialization."""
        self.vocab_size = 256
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(self.vocab_size)}
        tokens = [int(b) for b in input_text.encode("utf-8")]
        while self.vocab_size < target_vocab_size:
            top_tuple = self.find_most_common_pair(tokens)
            if top_tuple is None:
                print("No more tokens to merge, aborting prematurely")
                break
            tokens = self._substitute(tokens, list(top_tuple), [self.vocab_size])
            self.merges.append((self.vocab_size, top_tuple))
            self.vocab_size += 1

        # Finish building the vocabulary, for faster decoding.
        for token, original_pair in self.merges:
            self.vocab[token] = (
                self.vocab[original_pair[0]] + self.vocab[original_pair[1]]
            )

    @classmethod
    def _get_pair_counts(cls, tokens: list[int]) -> dict[tuple[int, int], int]:
        pair_counts = defaultdict(int)
        for a, b in zip(tokens, tokens[1:]):
            pair_counts[(a, b)] += 1
        return pair_counts

    @classmethod
    def find_most_common_pair(cls, tokens: list[int]) -> tuple[int, int] | None:
        """Find the most common pair of tokens"""
        if len(tokens) < 2:
            return None
        pair_counts = cls._get_pair_counts(tokens)
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

    def encode(self, text: str) -> list[int]:
        tokens = [int(b) for b in text.encode("utf-8")]
        for token, original_pair in self.merges:
            tokens = self._substitute(tokens, list(original_pair), [token])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token] for token in tokens)
        return token_bytes.decode("utf-8", errors="replace")


def main():
    input_text = Path("blogpost.txt").read_text()
    tokenizer = Tokenizer(input_text, 300)

    text_to_encode = input_text
    encoded = tokenizer.encode(text_to_encode)
    compression_ratio = len(encoded) / len(input_text)
    print(
        f"Encoded text has length: {len(encoded)}, compression ratio: {compression_ratio * 100:.0f}%"
    )
    decoded = tokenizer.decode(encoded)
    assert text_to_encode == decoded

    print(tokenizer.decode([129]))


if __name__ == "__main__":
    main()
