from collections import defaultdict


def main():
    input_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    input_bytes = [int(b) for b in input_text.encode("utf-8")]

    # Task 1: find the most common pair of bytes
    pair_counts = defaultdict(int)
    for a, b in zip(input_bytes, input_bytes[1:]):
        pair_counts[(a, b)] += 1

    tuples_sorted = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    top_tuple = next(iter(tuples_sorted))[0]
    top_tuple_str = bytes(top_tuple).decode("utf-8")
    print(
        f"Most common tuple: {top_tuple}, "
        f"corresponding to {top_tuple_str}, "
        f"appears {pair_counts[top_tuple]} times."
    )


if __name__ == "__main__":
    main()
