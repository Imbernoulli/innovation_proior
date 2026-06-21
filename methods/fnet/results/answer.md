# FNet

## Problem

Self-attention gives a Transformer encoder flexible all-token communication, but it does so by forming an `N x N` token-token score matrix. That creates `O(N^2)` time and memory cost in the sequence length. The target is an encoder that keeps the BERT block around the mixer, but replaces self-attention with a simpler global mixing operation.

## Method

Replace each self-attention sublayer with a parameter-free two-dimensional discrete Fourier transform over the hidden and sequence axes:

`y = Re(F_seq(F_hidden(x)))`

with the forward DFT convention

`X_k = sum_{n=0}^{N-1} x_n exp(-2*pi*i*n*k/N)`.

The two one-dimensional transforms commute because they act on different axes. The real part is taken once, after both transforms, so the rest of the encoder remains real-valued. The mixer has no learned parameters and ignores the attention padding mask; pure FNet is trained and used at fixed padded lengths.

For code faithfulness, the FFT path and matrix path must use the same normalization. The canonical JAX implementation uses `jnp.fft.fftn` for the FFT path and `scipy.linalg.dft` for the cached matrix path; both are unnormalized forward DFTs. Do not insert a `1/sqrt(N)` factor into the matrix path unless the FFT path is rescaled too.

Everything else stays BERT-like: word, position, and type embeddings; post-LayerNorm residual blocks with epsilon `1e-12`; a `Dense(d_ff) -> GELU -> Dense(d_model)` feed-forward sublayer; and a dense plus `tanh` pooler over the first token. Hybrid variants optionally replace the final mixer layers with self-attention.

## Code

```python
import functools
import math
from typing import Callable

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from scipy import linalg


default_kernel_init = nn.initializers.normal(stddev=2e-2)
default_bias_init = nn.initializers.normal(stddev=2e-2)
LAYER_NORM_EPSILON = 1e-12


def two_dim_matmul(
    x,
    matrix_dim_one,
    matrix_dim_two,
    precision=lax.Precision.DEFAULT,
):
    # Equivalent to np.fft.fftn(x) when the matrices are scipy.linalg.dft(...)
    # with default, unnormalized scaling.
    return jnp.einsum(
        "ij,jk,ni->nk",
        x,
        matrix_dim_two,
        matrix_dim_one,
        optimize=True,
        precision=precision,
    )


def init_fourier_transform(max_seq_length, d_model, use_fft=True):
    if use_fft:
        if max_seq_length > 4096 and not math.log2(max_seq_length).is_integer():
            raise ValueError("For long FFT inputs, max_seq_length must be a power of 2.")
        return jnp.fft.fftn

    dft_mat_seq = jnp.asarray(linalg.dft(max_seq_length))
    dft_mat_hidden = jnp.asarray(linalg.dft(d_model))
    return functools.partial(
        two_dim_matmul,
        matrix_dim_one=dft_mat_seq,
        matrix_dim_two=dft_mat_hidden,
        precision=lax.Precision.DEFAULT,
    )


class FourierTransform(nn.Module):
    fourier_transform: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(self, inputs, padding_mask=None, deterministic=False):
        del padding_mask
        del deterministic
        return jax.vmap(self.fourier_transform)(inputs).real


class FeedForwardLayer(nn.Module):
    d_ff: int
    dropout_rate: float = 0.0
    intermediate_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        d_model = inputs.shape[-1]
        x = nn.Dense(
            self.d_ff,
            kernel_init=default_kernel_init,
            bias_init=default_bias_init,
            name="intermediate",
        )(inputs)
        x = self.intermediate_activation(x)
        x = nn.Dense(d_model, kernel_init=default_kernel_init, name="output")(x)
        return nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)


class EncoderBlock(nn.Module):
    mixing_sublayer: nn.Module
    feed_forward_sublayer: FeedForwardLayer

    @nn.compact
    def __call__(self, inputs, padding_mask=None, deterministic=False):
        mixing_output = self.mixing_sublayer(
            inputs, padding_mask, deterministic=deterministic
        )
        x = nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON, name="mixing_layer_norm"
        )(inputs + mixing_output)
        feed_forward_output = self.feed_forward_sublayer(
            x, deterministic=deterministic
        )
        return nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON, name="output_layer_norm"
        )(x + feed_forward_output)


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    d_ff: int
    dropout_rate: float
    max_seq_length: int
    use_fft: bool = True

    def setup(self):
        fourier_transform = init_fourier_transform(
            self.max_seq_length, self.d_model, self.use_fft
        )
        self.encoder_blocks = [
            EncoderBlock(
                mixing_sublayer=FourierTransform(
                    fourier_transform=fourier_transform,
                    name=f"fourier_transform_{layer}",
                ),
                feed_forward_sublayer=FeedForwardLayer(
                    self.d_ff, self.dropout_rate, name=f"feed_forward_{layer}"
                ),
                name=f"encoder_{layer}",
            )
            for layer in range(self.num_layers)
        ]
        self.pooler = nn.Dense(
            self.d_model, kernel_init=default_kernel_init, name="pooler"
        )

    def __call__(self, hidden_states, padding_mask=None, deterministic=False):
        for block in self.encoder_blocks:
            hidden_states = block(
                hidden_states, padding_mask, deterministic=deterministic
            )
        pooled_output = jnp.tanh(self.pooler(hidden_states[:, 0]))
        return hidden_states, pooled_output
```

## Design checks

The sequence and hidden transforms are both part of the final mixer; sequence-only mixing is faster but less accurate. Taking the real part midway or taking absolute values is less stable and less accurate than taking the real part once at the end. DCT and Hadamard are weaker alternatives; Hartley ties the reported Fourier result but does not improve it. Adding learned parameters to the transform sublayer does not improve the reported model and tends to slow it or destabilize training.
