"""
Train GPT-2 from scratch.

Tasks:

1. Sample from huggingface transformers GPT2.
2. Re-implement GPT2 using the MiniGPT class, load the pretrained weights and reproduce the samples,
   ensuring that they're consistent.
"""

from transformers import pipeline, set_seed


def main():
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    results = generator(
        "Hello, I'm a language model,", max_length=100, num_return_sequences=3
    )
    for result in results:
        print(result["generated_text"])
        print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
