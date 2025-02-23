"""
Train GPT-2 from scratch.

Tasks:

1. Sample from huggingface transformers GPT2.
2. Re-implement GPT2 using the MiniGPT class, load the pretrained weights and reproduce the samples,
   ensuring that they're consistent.
"""

import torch
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2TokenizerFast

from minigpt import MiniGPT, AttentionBlock, sample_from_model


def generate_samples():
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    results = generator(
        "Hello, I'm a language model,", max_length=100, num_return_sequences=3
    )
    for result in results:
        print(result["generated_text"])
        print("--------------------------------------------------------")


def init_model_from_state_dict(model: MiniGPT, hp_gpt2_sd: dict) -> MiniGPT:
    @torch.no_grad()
    def _copy_weights(source_tensor, target_tensor):
        target_tensor.copy_(source_tensor)

    # Copy over the embeddings.
    _copy_weights(hp_gpt2_sd["transformer.wte.weight"], model.embedding.weight)
    _copy_weights(
        hp_gpt2_sd["transformer.wpe.weight"], model.positional_encoding.weight
    )

    # Copy over the attention blocks.
    def _copy_attention_block(block_idx: int, block: AttentionBlock):
        prefix = f"transformer.h.{block_idx}"
        _copy_weights(hp_gpt2_sd[f"{prefix}.ln_1.weight"], block.norm_1.weight)
        _copy_weights(hp_gpt2_sd[f"{prefix}.ln_1.bias"], block.norm_1.bias)

        # Note the transpose of the source weights
        source_attn_weight = hp_gpt2_sd[f"{prefix}.attn.c_attn.weight"].T
        source_attn_bias = hp_gpt2_sd[f"{prefix}.attn.c_attn.bias"]

        # In the huggingface implementation, queries, keys, and values are stacked
        # into a single tensor. Original code unpacking these:
        # query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        source_query_weight, source_key_weight, source_value_weight = (
            source_attn_weight.split(model.embedding_size, dim=0)
        )
        source_query_bias, source_key_bias, source_value_bias = source_attn_bias.split(
            model.embedding_size
        )
        _copy_weights(source_query_weight, block.multi_head_attention.queries.weight)
        _copy_weights(source_query_bias, block.multi_head_attention.queries.bias)
        _copy_weights(source_key_weight, block.multi_head_attention.keys.weight)
        _copy_weights(source_key_bias, block.multi_head_attention.keys.bias)
        _copy_weights(source_value_weight, block.multi_head_attention.values.weight)
        _copy_weights(source_value_bias, block.multi_head_attention.values.bias)

        _copy_weights(
            hp_gpt2_sd[f"{prefix}.attn.c_proj.weight"],
            block.multi_head_attention.linear.weight,
        )
        _copy_weights(
            hp_gpt2_sd[f"{prefix}.attn.c_proj.bias"],
            block.multi_head_attention.linear.bias,
        )

        _copy_weights(hp_gpt2_sd[f"{prefix}.ln_2.weight"], block.norm_2.weight)
        _copy_weights(hp_gpt2_sd[f"{prefix}.ln_2.bias"], block.norm_2.bias)

        # Note the transpose of the source weights.
        _copy_weights(
            hp_gpt2_sd[f"{prefix}.mlp.c_fc.weight"].T,
            block.feed_forward.linear_1.weight,
        )
        _copy_weights(
            hp_gpt2_sd[f"{prefix}.mlp.c_fc.bias"], block.feed_forward.linear_1.bias
        )
        _copy_weights(
            hp_gpt2_sd[f"{prefix}.mlp.c_proj.weight"].T,
            block.feed_forward.linear_2.weight,
        )
        _copy_weights(
            hp_gpt2_sd[f"{prefix}.mlp.c_proj.bias"], block.feed_forward.linear_2.bias
        )

    for i, block in enumerate(model.attention_blocks.children()):
        _copy_attention_block(i, block)

    _copy_weights(hp_gpt2_sd["transformer.ln_f.weight"], model.final_layer_norm.weight)
    _copy_weights(hp_gpt2_sd["transformer.ln_f.bias"], model.final_layer_norm.bias)

    _copy_weights(hp_gpt2_sd["lm_head.weight"], model.linear.weight)
    return model


def main():
    hf_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    hp_gpt2_sd = hf_gpt2.state_dict()
    for k, v in hp_gpt2_sd.items():
        print(f"{k}: {tuple(v.shape)}")

    assert hf_gpt2.config.n_embd == hf_gpt2.config.hidden_size
    vocab_size = hf_gpt2.config.vocab_size
    embedding_size = hf_gpt2.config.n_embd
    context_size = hf_gpt2.config.max_position_embeddings
    num_heads = hf_gpt2.config.num_attention_heads
    head_size = embedding_size // num_heads

    print(f"""Model parameters:
    vocab_size: {vocab_size}
    embedding_size: {embedding_size}
    context_size: {context_size}
    num_heads: {num_heads}
    head_size: {head_size}
    """)

    model = MiniGPT(
        vocab_size=vocab_size,
        context_size=context_size,
        embedding_size=embedding_size,
        num_blocks=11,
        num_heads=num_heads,
        head_size=head_size,
        attention_bias=True,
        final_layer_bias=False,
        use_flash_attention=True,
        final_layer_norm=True,
    )

    print("Initialized the model. Copying over pretrained weights...")
    model = init_model_from_state_dict(model, hp_gpt2_sd)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    start_ctx = tokenizer.encode("I'm a language model,")
    tokens = list(sample_from_model(model, context_size, 100, vocab_size, start_ctx=start_ctx))
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
