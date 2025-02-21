"""
Fine tune the mini-GPT model, adding a new token.
"""

from pathlib import Path

# import torch

from tokenizer import Tokenizer


# device = "cuda" if torch.cuda.is_available() else "cpu"


def fine_tune_model():
    tokenizer = Tokenizer.load(Path("tokenizer"))
    # context_size = 256
    # model = MiniGPT(
    #     vocab_size=tokenizer.vocab_size,
    #     embedding_size=384,
    #     context_size=context_size,
    #     head_size=384 // 6,
    #     num_heads=6,
    #     num_blocks=6,
    #     use_flash_attention=True,
    # )
    # model.load_state_dict(torch.load("model-minigpt.pth", map_location=device))

    # Now we fine tune, by adding an extra token
    # It will be the ' <|endoftext|>' token.
    # First, add it to the tokenizer.
    token_str = "<|endoftext|>"
    test_str = "Testing with " + token_str + " surrounded by extra text."
    print(tokenizer.encode(test_str))
    tokenizer.add_token(token_str)
    print(enc := tokenizer.encode(test_str))
    print(tokenizer.decode(enc))


if __name__ == "__main__":
    fine_tune_model()
