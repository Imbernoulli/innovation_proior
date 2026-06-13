# Context: the ground an attention-free encoder stands on

## Research question

The Transformer encoder has become the dominant backbone for language understanding, and at its heart is self-attention: a sublayer that lets every token in a length-`N` sequence form a representation as a relevance-weighted combination of every other token, with the weights computed from token–token (query–key) dot products. This token-dependent, all-pairs mixing is widely credited as the source of the architecture's power.

It is also its main cost. Self-attention is `O(N²)` in both time and memory in the sequence length, because it materializes and uses an `N×N` matrix of pairwise scores; encoders are frequently memory-bound, and the quadratic blowup caps the sequence lengths that are practical to train on. A large body of "efficient Transformer" work attacks this — sparsifying the attention matrix or linearizing it — but these methods either hide large constants behind their improved asymptotics or only approximate the very attention they are trying to cheapen, and they add machinery rather than removing it.

The question is therefore whether the *flexibility and cost* of attention are actually necessary. Concretely: can the self-attention sublayer of an encoder be replaced wholesale by a **simpler token-mixing mechanism** — while keeping the rest of the encoder (the position-wise feed-forward sublayers, the embeddings, the residual/normalization structure) untouched, and losing only a limited amount of accuracy? A solution would have to mix information across all token positions (so that the downstream feed-forward sublayers have access to the whole sequence), scale sub-quadratically in `N`, and be simple enough that it shrinks the model and stabilizes training rather than complicating it.

## Background

The encoder being modified is BERT-style (Devlin et al. 2018): an embedding layer (word + absolute-position + token-type embeddings), then a stack of `N` identical encoder blocks, each a self-attention sublayer followed by a position-wise feed-forward sublayer, with a residual connection and layer normalization around each, and a pooler for sentence-level tasks. Pre-training is masked-language-modeling plus next-sentence-prediction; the feed-forward width is `4·d_h` and attention uses `d_h/64` heads.

The load-bearing observation is that an encoder block does two separable kinds of mixing. The feed-forward sublayer mixes the **hidden** dimension, transforming each token independently across its features. Self-attention's distinctive job is to mix the **sequence** dimension — to combine information across token positions — so that the feed-forward sublayers downstream can act on representations that have already "seen" the whole sequence. Framed this way, attention is the encoder's *token-mixing* sublayer, and the question becomes whether that mixing must be the specific token-dependent dot-product form.

Several prior results suggest it need not be. Tay et al. (2020, Synthesizer) replaced the query–key dot product with learned token-mixing weights and found that the dot product, while expressive, is not crucial for accurate NLP models. You et al. (2020) replaced attention weights with fixed, unparameterized Gaussian distributions with minimal degradation (provided learnable cross-attention was kept). Raganato et al. (2020) replaced all but one attention head per encoder layer with fixed, non-learnable positional patterns and saw little accuracy loss. Tolstikhin et al. (2021, MLP-Mixer) replaced attention with plain MLPs in vision with limited degradation. Together these say: the *content-dependent, all-pairs* nature of attention is not the only way to mix tokens well.

A piece of classical signal-processing machinery in the general background is the discrete Fourier transform. For a length-`N` sequence `{x_n}`, the DFT is `X_k = Σ_{n=0}^{N-1} x_n e^{-2πi nk/N}`, `0 ≤ k ≤ N-1`. It can be computed in `O(N log N)` by the Cooley–Tukey FFT (Cooley & Tukey 1965), or as a matrix multiply by the DFT matrix `W_{nk} = e^{-2πi nk/N}/√N` in `O(N²)`. By the convolution theorem, multiplication in the frequency domain corresponds to convolution in the time domain. Fourier transforms have been used to speed up convolutional and recurrent nets and to approximate dense linear layers, and the Performer (Choromanski et al. 2020) used random Fourier features to *approximate* the softmax attention kernel.

## Baselines

**BERT encoder with self-attention (Devlin et al. 2018; Vaswani et al. 2017).** The reference. Self-attention `softmax(QKᵀ/√d_k)V` mixes tokens with content-dependent, all-pairs weights, the most expressive token mixer. Gaps: `O(N²)` time and memory in sequence length; many learnable parameters in the mixing layer; the dominant accuracy ceiling but also the dominant cost.

**Dense linear token mixing ("Linear" baseline).** Replace attention with two ordinary learned matrix multiplications — one mixing the sequence dimension, one mixing the hidden dimension — no softmax, no dot products (a Synthesizer-/MLP-Mixer-style mixer). It mixes all tokens and trains faster than attention, and reaches close to BERT's accuracy. Gaps: a learned sequence-mixing matrix is `N×N`, so it scales as `O(N²)` in parameters and compute and is tied to a fixed maximum length (it cannot generalize across sequence lengths); it still carries many parameters in the mixing layer.

**Efficient/long-sequence Transformers (Longformer, ETC, BigBird, Performer, Linformer, Linear Transformer).** Sparsify or linearize attention to reach `O(N√N)` or `O(N)` asymptotic complexity. Gaps: the favorable asymptotics often hide large constants (e.g. attention linear in length but quadratic in the number of "global" tokens, which must be sizeable for good accuracy); and the linearizing methods *approximate* attention rather than removing it, adding machinery and approximation error.

The recurring gap: every baseline either keeps the quadratic cost (BERT, dense-linear) or buys lower asymptotics with large constants / approximation of attention (efficient Transformers) — each still pays in either compute, parameters, or added machinery.

## Evaluation settings

Pre-training follows the BERT recipe: masked-language-modeling and next-sentence-prediction objectives on the C4 corpus (Raffel et al. 2019), with a 32k SentencePiece vocabulary (Kudo & Richardson 2018), in fixed "Base" and "Large" configurations (Turc et al. 2019) where the feed-forward size is `4·d_h`. The downstream language-understanding yardstick is the GLUE benchmark (Wang et al. 2018), reporting per-task accuracy. For long-sequence behavior the yardstick is the Long-Range Arena (LRA) benchmark (Tay et al. 2020), a suite of tasks requiring long-range dependencies. Efficiency is measured directly: training speed (steps/s), inference speed (ms/batch), and peak memory (GB), swept across sequence lengths from 512 up to ~16k and across model sizes, on both GPU (V100) and TPU (v3) hardware. All of these datasets, metrics, and protocols predate any particular mixing choice.

## Code framework

A BERT-style encoder in JAX/Flax fixes everything except the token-mixing sublayer. Embeddings, the position-wise feed-forward sublayer, the residual + post-LayerNorm block structure, the pooler, and the loss are all standard; the body of the across-sequence mixing sublayer is the single empty slot.

```python
import flax.linen as nn
import jax, jax.numpy as jnp

class TokenMixingLayer(nn.Module):
    """Mixes information across the sequence dimension. To be designed."""
    @nn.compact
    def __call__(self, x):                 # x: (batch, seq_len, d_model)
        # TODO: the across-sequence mixing operator that replaces self-attention
        raise NotImplementedError

class FeedForwardLayer(nn.Module):
    d_ff: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(self.d_ff, name="intermediate")(x)
        x = nn.gelu(x)
        x = nn.Dense(x.shape[-1], name="output")(x)
        return nn.Dropout(self.dropout_rate)(x, deterministic)

class EncoderBlock(nn.Module):
    mixing_layer: nn.Module
    ff_layer: FeedForwardLayer
    @nn.compact
    def __call__(self, x, deterministic):
        mixing_output = self.mixing_layer(x)
        x = nn.LayerNorm(1e-12, name="mixing_layer_norm")(x + mixing_output)
        ff_output = self.ff_layer(x, deterministic)
        return nn.LayerNorm(1e-12, name="output_layer_norm")(x + ff_output)

class Encoder(nn.Module):
    num_layers: int
    d_model: int
    d_ff: int
    dropout_rate: float
    def setup(self):
        self.blocks = [EncoderBlock(TokenMixingLayer(),
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
