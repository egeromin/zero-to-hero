"""
Fine tune the mini-GPT model, adding a new token.
"""

from pathlib import Path

import torch
from torch import nn

from minigpt import MiniGPT
from tokenizer import Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"


def fine_tune_model():
    tokenizer = Tokenizer.load(Path("tokenizer"))
    context_size = 256
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_size=384,
        context_size=context_size,
        head_size=384 // 6,
        num_heads=6,
        num_blocks=6,
        use_flash_attention=True,
    )
    model.load_state_dict(torch.load("model-minigpt.pth", map_location=device))

    # Now we fine tune, by adding an extra token
    # It will be the '<|endoftext|>' token.
    # First, add new tokens to the tokenizer.
    tokens_to_add = ["<|endoftext|>"]
    for token_str in tokens_to_add:
        tokenizer.add_token(token_str)

    # Freeze the existing parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Augment the model. To use the existing modules, we can augment the
    # softmax by having a separate one-neuron output linear layer, which
    # maps to the logit for the new token. And then concatenate the logits,
    # before the softmax.
    # For the initial embedding, this does not work.
    # In both cases, we'd like an API for freezing part of the embedding
    # and part of the last linear layer.
    # A good solution seems to be *hooks* on tensors, which allow us to
    # run a hook on the gradient, after .`backward()` for a specific layer
    # has been called. In our case, we can use the hook to zero-out specific
    # parts of the tensor gradient.
    # https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html

    # Swap out the final linear layer and embedding layer, copying over the existing weights.
    num_tokens_to_add = len(tokens_to_add)
    old_vocab_size = model.vocab_size
    new_vocab_size = model.vocab_size + num_tokens_to_add

    # Copy over the embedding weights and register a hook for them.
    new_embedding = nn.Embedding(new_vocab_size, model.embedding_size)
    with torch.no_grad():
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        new_embedding.weight[:old_vocab_size] = model.embedding.weight

    def _embedding_hook(grad):
        new_grad = grad.clone()
        new_grad[:old_vocab_size] = 0
        return new_grad

    new_embedding.weight.register_hook(_embedding_hook)
    model.embedding = new_embedding

    # Same for the final linear layer
    new_linear = nn.Linear(model.embedding_size, new_vocab_size, bias=False)
    with torch.no_grad():
        nn.init.normal_(new_linear.weight, mean=0.0, std=0.02)
        new_linear.weight[:old_vocab_size] = model.linear.weight

    def _linear_hook(grad):
        new_grad = grad.clone()
        new_grad[:, :old_vocab_size] = 0
        return new_grad

    new_linear.weight.register_hook(_linear_hook)
    model.linear = new_linear

    # Update the model vocab size
    model.vocab_size = new_vocab_size

    # Finally, fine tune.
    # TODO: implement training loop and ensure optimizer updates only
    # the model params with requires_grad = True.


if __name__ == "__main__":
    fine_tune_model()
