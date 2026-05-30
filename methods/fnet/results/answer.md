# FNet

## Problem

Self-attention is the expressive core of a Transformer encoder but costs `O(N²)` time and memory in the sequence length `N`, because it builds an `N×N` matrix of token–token scores. Efficient-attention methods reduce the asymptotics but hide large constants or merely approximate attention. The question: can the self-attention sublayer be replaced wholesale by a simpler, faster, ideally parameter-free token mixer, with limited accuracy loss?

## Key idea

An encoder block mixes the hidden dimension (the feed-forward sublayer) and the sequence dimension (attention). Attention's structural role is just to let every token see the others. Replace it with a **parameter-free 2D discrete Fourier transform** over the sequence and hidden axes, keeping only the real part — every output is a fixed-coefficient sum over all inputs (dense all-token mixing), computable in `O(N log N)` by the FFT, with zero learnable parameters and positional information baked into the transform kernel.

## Final method

The Fourier mixing sublayer, replacing self-attention:

`y = Re( ℱ_seq( ℱ_h(x) ) )`

— a 1D DFT along the hidden axis and a 1D DFT along the sequence axis (they commute, so the order is immaterial), with the real part extracted once, at the end, so the rest of the network stays real and unmodified. The DFT is `X_k = Σ_{n=0}^{N-1} x_n e^{-2πi nk/N}`.

Everything else is the standard BERT encoder block: residual + post-LayerNorm (`eps = 1e-12`) around the mixing sublayer, then around an unchanged position-wise feed-forward sublayer (`Dense(4d) → GELU → Dense(d)`). Embeddings are word + absolute-position + token-type; a pooler applies a dense layer and `tanh` to the first token.

Design points: the real part is taken only at the end (taking it midway, or taking the magnitude, is less accurate and less stable); the 2D transform (mixing hidden as well as sequence) beats a sequence-only 1D transform; the DFT beats DCT and Hadamard (both a few points worse) and ties the Hartley transform; adding learnable parameters to the mixer does not help. Implementation uses the FFT on GPU at all lengths, and a cached DFT-matrix multiply on TPU for shorter sequences (TPUs favor matmul). A hybrid that keeps a couple of self-attention sublayers (best placed in the top layers) recovers most of attention's accuracy while staying much faster.

## Code

```python
import flax.linen as nn
import jax, jax.numpy as jnp

class FourierTransformLayer(nn.Module):
    @nn.compact
    def __call__(self, x):                       # x: (batch, seq_len, d_model), real
        return jax.vmap(jnp.fft.fftn)(x).real     # 2D DFT over (seq, hidden); keep real part

class FeedForwardLayer(nn.Module):
    d_ff: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(self.d_ff, name="intermediate")(x)
        x = nn.gelu(x)
        x = nn.Dense(x.shape[-1], name="output")(x)
        return nn.Dropout(self.dropout_rate)(x, deterministic)

class FNetEncoderBlock(nn.Module):
    fourier_layer: FourierTransformLayer
    ff_layer: FeedForwardLayer
    @nn.compact
    def __call__(self, x, deterministic):
        mixing_output = self.fourier_layer(x)
        x = nn.LayerNorm(1e-12, name="mixing_layer_norm")(x + mixing_output)
        feed_forward_output = self.ff_layer(x, deterministic)
        return nn.LayerNorm(1e-12, name="output_layer_norm")(x + feed_forward_output)

class FNetEncoder(nn.Module):
    num_layers: int
    d_model: int
    d_ff: int
    dropout_rate: float
    def setup(self):
        self.blocks = [FNetEncoderBlock(FourierTransformLayer(),
                                        FeedForwardLayer(self.d_ff, self.dropout_rate),
                                        name=f"encoder_{i}")
                       for i in range(self.num_layers)]
        self.pooler = nn.Dense(self.d_model, name="pooler")
    def __call__(self, x, deterministic):
        for blk in self.blocks:
            x = blk(x, deterministic)
        pooled = jnp.tanh(self.pooler(x[:, 0]))
        return x, pooled
```
