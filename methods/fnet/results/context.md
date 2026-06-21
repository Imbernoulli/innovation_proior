# Context: the ground an attention-free encoder stands on

## Research question

The Transformer encoder has become the default backbone for language understanding, and its distinctive sublayer is self-attention. For a length-`N` sequence, self-attention lets each token form a representation from all other tokens, with the weights produced by query-key dot products and a softmax over an `N x N` score matrix.

That matrix is also the bottleneck. The standard encoder pays `O(N^2)` time and `O(N^2)` memory in the sequence length, and long-input encoders are often memory-bound. Efficient-attention variants try to reduce the cost by sparsifying the score matrix or by linearizing the attention computation, but these approaches still preserve attention as the object being approximated or constrained, and their favorable asymptotics can hide large constants.

The question is whether the encoder really needs token-dependent attention in every layer. Can the self-attention sublayer be replaced by a simpler token-mixing operation, while leaving embeddings, feed-forward sublayers, residual connections, layer normalization, and task heads essentially unchanged? A useful answer must mix information across positions, avoid quadratic length scaling when possible, and not add a complicated new parameterized mechanism in place of the old one.

## Background

The architectural template is BERT-style. Inputs are represented by word, absolute-position, and token-type embeddings, followed by a stack of post-LayerNorm encoder blocks. Each block has a mixing sublayer, a residual connection and layer norm, then a position-wise feed-forward sublayer, another residual connection, and another layer norm. The feed-forward width is typically `4 * d_model`, and the sentence-level pooler reads the first token through a dense layer and `tanh`.

This block separates two kinds of work. The feed-forward sublayer mixes the hidden dimension at each token independently. The self-attention sublayer is the part that mixes the sequence dimension, so downstream per-token transformations can use information that has already crossed positions. In that view, attention is one implementation of token mixing, not necessarily the only possible implementation.

Prior work already weakens the assumption that token mixing must be content-dependent dot-product attention. Synthesizer replaces query-key dot products with synthetic attention weights. Fixed-pattern and Gaussian attention variants show that some non-learned positional structure can carry useful signal. MLP-Mixer makes an analogous separation between token mixing and channel mixing in vision. These results do not solve the encoder problem, but they make it reasonable to ask for a simpler global mixer.

## Baselines

**BERT encoder with self-attention.** This is the reference architecture. Its self-attention sublayer computes `softmax(QK^T / sqrt(d_k))V`, giving a flexible, token-dependent all-pairs mixer. Its gap is the `N x N` score matrix, with quadratic time and memory in sequence length and a substantial parameterized projection stack.

**Dense learned token mixing.** One direct replacement is to use ordinary learned linear maps: one across sequence positions and one across hidden features. This preserves global mixing and removes the attention softmax and query-key interaction. Its gap is that a learned sequence mixer has an `N x N` matrix tied to a chosen maximum length, so it keeps quadratic sequence cost and does not naturally transfer to a different length.

**Efficient and long-sequence attention variants.** Sparse and linearized attention models reduce or restructure attention cost. Their gap is that they remain attention mechanisms: they either approximate the dense attention calculation or restrict its connectivity pattern, often adding implementation complexity and constants that matter at practical lengths.

The shared gap is that these baselines either keep the expensive all-pairs attention object, or replace it with a learned dense object that is still tied to the sequence length.

## Evaluation settings

The pre-training setting is the BERT recipe: masked-language modeling and next-sentence prediction over C4, with a 32k SentencePiece vocabulary and fixed Base/Large model sizes. Downstream language-understanding quality is measured on GLUE. Long-input behavior is measured on Long-Range Arena tasks, where sequence length and memory pressure are central.

Efficiency is measured directly rather than inferred only from asymptotic notation: training speed, inference speed, FLOPs, and peak memory are swept over hardware and sequence lengths. The comparison keeps the rest of the encoder template fixed so the empty slot is specifically the token-mixing sublayer.

## Code framework

```python
import flax.linen as nn
import jax.numpy as jnp


LAYER_NORM_EPSILON = 1e-12


class TokenMixingLayer(nn.Module):
    """The across-sequence mixing operation to be designed."""

    @nn.compact
    def __call__(self, inputs, padding_mask=None, deterministic=False):
        raise NotImplementedError


class FeedForwardLayer(nn.Module):
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        d_model = inputs.shape[-1]
        x = nn.Dense(self.d_ff, name="intermediate")(inputs)
        x = nn.gelu(x)
        x = nn.Dense(d_model, name="output")(x)
        return nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)


class EncoderBlock(nn.Module):
    mixing_layer: nn.Module
    feed_forward_layer: FeedForwardLayer

    @nn.compact
    def __call__(self, inputs, padding_mask=None, deterministic=False):
        mixed = self.mixing_layer(
            inputs, padding_mask, deterministic=deterministic
        )
        x = nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON, name="mixing_layer_norm"
        )(inputs + mixed)
        ff = self.feed_forward_layer(x, deterministic=deterministic)
        return nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON, name="output_layer_norm"
        )(x + ff)


class Encoder(nn.Module):
    blocks: list
    d_model: int

    def setup(self):
        self.pooler = nn.Dense(self.d_model, name="pooler")

    def __call__(self, x, padding_mask=None, deterministic=False):
        for block in self.blocks:
            x = block(x, padding_mask, deterministic=deterministic)
        return x, jnp.tanh(self.pooler(x[:, 0]))
```
