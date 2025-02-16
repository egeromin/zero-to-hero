from collections import defaultdict
from pathlib import Path


class Tokenizer:
    def __init__(self, input_text: str, target_vocab_size: int):
        """Train the tokenizer on a given input text
        at initialization."""
        input_bytes = [int(b) for b in input_text.encode("utf-8")]
        self.first_new_token = 256
        assert target_vocab_size >= self.first_new_token
        self.new_tokens = list(range(self.first_new_token, target_vocab_size))
        self.original_pairs = []

        def merge_input_bytes(input_bytes: list[int], next_token: int) -> list[int]:
            # Task 1: find the most common pair of bytes
            pair_counts = defaultdict(int)
            for a, b in zip(input_bytes, input_bytes[1:]):
                pair_counts[(a, b)] += 1

            tuples_sorted = sorted(
                pair_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_tuple = next(iter(tuples_sorted))[0]
            # try:
            #     top_tuple_str = bytes(top_tuple).decode("utf-8")
            # except (UnicodeDecodeError, ValueError):
            #     top_tuple_str = "undefined: decode error"
            count_top_tuple = pair_counts[top_tuple]
            # print(
            #     f"Most common tuple: {top_tuple}, "
            #     f"corresponding to {top_tuple_str}, "
            #     f"appears {count_top_tuple} times."
            # )
            self.original_pairs.append(top_tuple)

            # Task 2: replace the most common pair with a new token.
            merged_bytes = self._substitute(input_bytes, list(top_tuple), [next_token])
            # print(f"Merged bytes length: {len(merged_bytes)}")

            # Strict equality might not hold. For example, consider a string
            # where are characters are the same: aaaaa
            # len(input_bytes) == 5
            # count_top_tuple == 4
            # len(merged_bytes) == 3
            assert len(merged_bytes) >= len(input_bytes) - count_top_tuple
            return merged_bytes

        for next_token in self.new_tokens:
            input_bytes = merge_input_bytes(input_bytes, next_token)

        # print(f"Final list of input bytes starts with: {input_bytes[:10]}")

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
        output_tokens = [int(b) for b in text.encode("utf-8")]
        for token, original_pair in zip(self.new_tokens, self.original_pairs):
            output_tokens = self._substitute(
                output_tokens, list(original_pair), [token]
            )
        return output_tokens

    def decode(self, tokens: list[int]) -> str:
        rebuilt_tokens = list(tokens)
        for token, original_pair in reversed(
            list(zip(self.new_tokens, self.original_pairs))
        ):
            rebuilt_tokens = self._substitute(
                rebuilt_tokens, [token], list(original_pair)
            )
        return bytes(rebuilt_tokens).decode("utf-8")


def main():
    input_text = Path("blogpost.txt").read_text()
    tokenizer = Tokenizer(input_text, 300)

    text_to_encode = input_text[50:120]
    encoded = tokenizer.encode(text_to_encode)
    print(f"Encoded text has length: {len(encoded)}")
    decoded = tokenizer.decode(encoded)
    assert text_to_encode == decoded


if __name__ == "__main__":
    main()
