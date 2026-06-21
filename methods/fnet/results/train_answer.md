Self-attention is the workhorse of the Transformer encoder, but its flexibility comes at a clear price. For a sequence of length N, it materializes an N x N token-token score matrix, which means the time and memory costs grow quadratically with sequence length. Efficient attention variants try to fix this by sparsifying that matrix or linearizing the softmax attention computation, yet they remain focused on approximating or constraining attention itself. The real architectural question is simpler: attention is just the mechanism the encoder uses to mix information across positions, so perhaps the position-mixing job can be done by something far less expensive.

A learned dense linear map across sequence positions is one natural replacement, and it confirms that query-key dot products are not essential. But a dense sequence matrix is still N x N, so it keeps the quadratic length cost and cannot transfer naturally to a new length. The goal is therefore a global, position-mixing operation that is structured enough to have a fast algorithm, fixed enough to be parameter-free, and simple enough to drop directly into a BERT-style encoder block without changing anything else.

The method is FNet. It replaces each self-attention sublayer with a two-dimensional discrete Fourier transform taken over the hidden and sequence axes. The exact mixer is y = Re(F_seq(F_hidden(x))), where F_hidden and F_seq are forward one-dimensional DFTs. With the standard forward convention, X_k = sum_n x_n exp(-2*pi*i*n*k/N), every output is a fixed-coefficient sum over every input position, so the transform is a dense global mixer with no learned parameters. The two one-dimensional transforms commute because they act on different axes, and the real part is taken only once, after both transforms, so the rest of the encoder stays real-valued. The feed-forward sublayers, residual connections, layer normalizations, embeddings, and pooler remain unchanged from the BERT template.

Because the Fourier transform has a fast algorithm, the mixer runs in O(N log N) time rather than O(N^2). It also needs no padding mask: it mixes every slot, including padded slots, and the released models are trained and evaluated at a fixed maximum length. This makes the sublayer deterministic and easy to implement with standard FFT kernels. For code faithfulness, the FFT path and any cached matrix path must use the same normalization; the canonical implementation uses the unnormalized forward DFT from jnp.fft.fftn and scipy.linalg.dft, without an extra 1/sqrt(N) factor. Design checks on this family of encoders show that transforming both the hidden and sequence axes works better than sequence-only mixing, that taking the real part at the end is more stable than taking it midway or using absolute values, and that adding learned parameters inside the transform sublayer tends to slow or destabilize training without improving quality. Alternatives such as DCT, Hadamard, or Hartley transforms are plausible, but the Fourier transform is the natural default because it is parameter-free, globally mixing, and backed by mature fast kernels.

The surprising empirical result is that this simple substitution is competitive. When the rest of the encoder is held fixed and trained with the standard BERT recipe on masked language modeling and next-sentence prediction, FNet reaches strong GLUE scores relative to its attention-based counterpart while training and running considerably faster on long sequences. On Long-Range Arena tasks, the reduced scaling becomes directly visible in wall-clock time, FLOPs, and peak memory. Hybrid variants that keep a few self-attention layers near the top of the stack can close any remaining quality gap, but the pure transform model already shows that content-dependent token-to-token attention is not a hard requirement for encoder-style language understanding.

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


def two_dim_matmul(x, matrix_dim_one, matrix_dim_two, precision=lax.Precision.DEFAULT):
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
        feed_forward_output = self.feed_forward_sublayer(x, deterministic=deterministic)
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
