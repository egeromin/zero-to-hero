from collections import defaultdict
from pathlib import Path


def main():
    input_text = Path("blogpost.txt").read_text()
    input_bytes = [int(b) for b in input_text.encode("utf-8")]

    def merge_input_bytes(input_bytes: list[int], next_token: int) -> list[int]:
        # Task 1: find the most common pair of bytes
        pair_counts = defaultdict(int)
        for a, b in zip(input_bytes, input_bytes[1:]):
            pair_counts[(a, b)] += 1

        tuples_sorted = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        top_tuple = next(iter(tuples_sorted))[0]
        try:
            top_tuple_str = bytes(top_tuple).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            top_tuple_str = "undefined: decode error"
        count_top_tuple = pair_counts[top_tuple]
        print(
            f"Most common tuple: {top_tuple}, "
            f"corresponding to {top_tuple_str}, "
            f"appears {count_top_tuple} times."
        )

        # Task 2: replace the most common pair with a new token.
        merged_bytes = []
        i = 0
        print(f"Input byte length: {len(input_bytes)}")
        while i < len(input_bytes):
            if (
                i < len(input_bytes) - 1
                and (input_bytes[i], input_bytes[i + 1]) == top_tuple
            ):
                merged_bytes.append(next_token)
                i += 2
            else:
                merged_bytes.append(input_bytes[i])
                i += 1

        print(f"Merged bytes length: {len(merged_bytes)}")
        assert len(merged_bytes) == len(input_bytes) - count_top_tuple
        return merged_bytes

    next_token = 256
    num_mergers = 10
    for _ in range(num_mergers):
        input_bytes = merge_input_bytes(input_bytes, next_token)
        next_token += 1

    print(f"Final list of input bytes starts with: {input_bytes[:10]}")


if __name__ == "__main__":
    main()
