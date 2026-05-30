Let me start from what self-attention actually costs me in an encoder, and then ask whether I'm paying for something I need. A self-attention sublayer takes a `(N, d)` input — `N` tokens, `d` hidden features — and rebuilds each token as a relevance-weighted combination of all the others, with the weights coming from query–key dot products. To do that it forms an `N×N` matrix of pairwise scores. That matrix is the whole cost: `O(N²)` time and `O(N²)` memory in the sequence length. Encoders are routinely memory-bound, and that quadratic term is what caps the sequence lengths I can afford. The efficient-Transformer literature has been chipping at this for years — sparsify the attention matrix (Longformer, BigBird), or linearize it (Linformer, Performer, Linear Transformer) — but every one of those either hides a big constant behind its nice asymptotics (BigBird is linear in length but quadratic in the number of "global" tokens, and you need a lot of them) or *approximates* the attention it's trying to cheapen, which means more machinery, not less, plus an approximation error to worry about.

So instead of approximating attention faster, let me ask the blunt question: do I actually need attention's particular kind of mixing at all? What does the sublayer *do* for the rest of the network? Look at one encoder block: a token-mixing sublayer, then a position-wise feed-forward sublayer. The feed-forward sublayer runs independently on each token — it mixes the *hidden* dimension, transforming each token's `d` features. It never looks across positions. So the only sublayer that combines information *across tokens* is attention. Its job, structurally, is to mix the *sequence* dimension, so that when the feed-forward sublayer chews on a token it's chewing on something that has already absorbed the rest of the sequence. That's the function. The content-dependent, all-pairs dot-product machinery is *one way* to do that function — but the function itself is just "let every token see the others."

And there's evidence the specific machinery isn't sacred. The Synthesizer (Tay et al. 2020) threw out the query–key dot product and learned the token-mixing weights directly, and found the dot product expressive but not crucial for accurate NLP. You et al. (2020) replaced attention weights with *fixed* Gaussians and barely lost accuracy. Raganato et al. (2020) replaced nearly all the heads per encoder layer with fixed positional patterns, again little loss. And MLP-Mixer (Tolstikhin et al. 2021) replaced attention with plain MLPs in vision. The drumbeat is consistent: the across-token mixing matters, but it doesn't have to be token-dependent, and it doesn't even have to be learned.

So let me take the simplest possible replacement and see how far it goes. Drop attention; mix the sequence with a plain learned matrix multiply along the sequence axis, and (since I'm at it) a learned matrix multiply along the hidden axis too. No softmax, no dot products — just two dense linear maps, one per axis. This is the Synthesizer/MLP-Mixer move. And it works: a model with this "linear" mixer trains substantially faster than attention and comes within a few points of BERT on the pre-training task. That's already a strong hint that attention's flexibility is not the principal driver.

But I haven't actually solved my problem, and I should see exactly why. The sequence-mixing matrix is `N×N`. So this linear mixer is `O(N²)` in *parameters* and still `O(N²)` in compute — I've removed the dot products but kept the quadratic. Worse, an `N×N` learned matrix is welded to a fixed maximum length: train it at length 512 and it has no entries for length 1024, so it can't generalize across sequence lengths at all. I've traded a parameter-free quadratic for a parameter-*heavy* quadratic. What I actually want is a *structured*, fixed, parameter-free linear transform that still mixes every token into every output, and that I can compute in less than `O(N²)`.

What linear transform takes a sequence and produces, at each output position, a sum over *all* input positions, with fixed coefficients, fast? Stare at that description for a second — "each output is a fixed-coefficient sum over all inputs." The discrete Fourier transform is exactly that. `X_k = Σ_{n=0}^{N-1} x_n e^{-2πi nk/N}`. Each frequency component `X_k` is a weighted sum over *every* input token `x_n`, weighted by the twiddle factor `e^{-2πi nk/N}`. It's the densest possible all-to-all mixing — every output touches every input — and the coefficients are *fixed*, not learned: zero parameters in the mixer. It can be computed in `O(N log N)` by the FFT (Cooley–Tukey 1965), or, if I want, as a matrix multiply by the DFT matrix `W_{nk} = e^{-2πi nk/N}/√N` in `O(N²)` — and on some hardware the matmul form is actually faster for moderate lengths. So the Fourier transform gives me the all-token mixing of a dense linear map, with no parameters and sub-quadratic compute. That's the candidate.

Now, the input isn't a 1D sequence, it's a `(N, d)` array — sequence length by hidden dimension. I have two axes I could transform. The sequence axis is the one I need to mix to replace attention's role. But the feed-forward sublayer already mixes the hidden axis, so should I bother transforming hidden too? Let me try both and reason about it. If I apply a 1D DFT only along the sequence axis, I get the token mixing I'm after, and indeed that alone vastly outperforms a model with no token mixing at all — confirming that the across-token mixing is the load-bearing part. But applying a second 1D DFT along the hidden axis as well — a full 2D DFT — turns out to give the best accuracy, presumably because it lets the mixer also reorganize features before the feed-forward sublayer does its work. The two 1D transforms commute (the transform is separable), so the order doesn't matter; I'll write it as the hidden transform then the sequence transform:

`y = ℱ_seq(ℱ_h(x))`.

There's an immediate snag: the DFT produces *complex* output. If I let complex numbers flow into the rest of the network, I'd have to complexify the feed-forward sublayers, the output layer, everything — a mess. I want the mixer to take a real input and hand back a real output so the rest of the encoder is untouched. The obvious fix is to take the real part. But *where*? I could take the real part after each 1D transform, or once at the very end. Let me think about what each does. Taking the real part midway — `Re(ℱ_seq(Re(ℱ_h(x))))` — throws away the imaginary information after the first transform, before the second transform can use it, and empirically that's both less accurate and less stable in training. Taking the magnitude `|·|` instead of the real part is worse still. Keeping the imaginary part alive through *both* transforms and extracting the real part only once, at the end, preserves the most information and trains best:

`y = Re(ℱ_seq(ℱ_h(x)))`.

So the entire mixing sublayer is: take the real input, do a 2D DFT over the sequence and hidden axes, keep the real part. No parameters, nothing complex leaking downstream.

Why should a *fixed* transform with no token-dependence work as a language mixer at all? A couple of things make it less crazy than it first looks. The twiddle factors `e^{-2πi nk/N}` are functions of the position indices `n` and `k`, so the transform inherently encodes position — which means the model barely needs explicit positional embeddings (it works without them; I'd keep them only for a clean comparison to BERT). And there's a duality picture: alternating encoder blocks behave a bit like alternating forward and inverse Fourier transforms, hopping the representation between a "time" domain and a "frequency" domain; and since multiplying by the feed-forward weights in the frequency domain corresponds, by the convolution theorem, to *convolving* in the time domain, the whole stack acts loosely like alternating multiplications and large-kernel convolutions. That's only an intuition — the residual connections and the fact that keeping only the real part makes the transform non-invertible break the clean duality — but it makes "Fourier transform as a token mixer" feel like a sensible inductive bias rather than a random structured matrix.

Should the mixer be Fourier specifically, or some other structured transform? Worth checking the natural alternatives, because the argument so far only demanded "fixed, dense, all-token, fast." The Discrete Cosine Transform maps real to real (so no real-part dance needed) and is closely related to the DFT — but it underperforms by a few points. The Hadamard transform uses only `±1` roots of unity, so it's slightly faster, but it loses a couple of points of accuracy too. The Hartley transform, `ℋ = Re{ℱ} − Im{ℱ}`, also maps real to real and *matches* the Fourier transform's accuracy exactly. So Fourier isn't uniquely magical — Hartley ties it — but it's the natural, simplest choice and nothing beats it. I'll go with the DFT. (And out of curiosity I'd want to add learnable parameters back into the mixer — elementwise scales, or learnable entries in the DFT matrix — but every such attempt either hurts accuracy and stability or does nothing while slowing the model down, which says the plain DFT is locally optimal as a mixer. So: keep it parameter-free.)

The block structure stays exactly BERT's, because the whole point is to change only the mixer. The Fourier sublayer's output is added to its input and layer-normalized; then the position-wise feed-forward sublayer (a dense expansion to `4·d`, a GELU, a dense projection back) is added and layer-normalized again — post-LN, `eps = 1e-12`, residuals around both sublayers. Embeddings are BERT's word + position + token-type sum; a pooler takes the first token through a dense layer and a `tanh` for sentence-level tasks.

One implementation note, because it changes which form of the DFT I compute. On GPUs the FFT is the fastest way to get the DFT at every sequence length I care about (512 to ~8k), so the GPU path always uses the FFT. On TPUs, which are even more matmul-optimized and have a weaker FFT, it's actually faster to *cache the DFT matrix* and compute the transform as a matrix multiply for shorter sequences (up to ~4k), switching to the FFT only beyond that. Same mathematical operation, two computational backends chosen by hardware.

Here's the mixer and the block, grounded in the JAX/Flax implementation.

```python
import flax.linen as nn
import jax, jax.numpy as jnp

class FourierTransformLayer(nn.Module):
    """Replaces self-attention: a parameter-free 2D DFT over (seq, hidden), keep real part."""
    @nn.compact
    def __call__(self, x):                       # x: (batch, seq_len, d_model), real
        # vmap over the batch; fftn does the 2D DFT over (seq, hidden) at once;
        # .real extracts the real part once, at the end, so downstream stays real.
        return jax.vmap(jnp.fft.fftn)(x).real

class FeedForwardLayer(nn.Module):               # unchanged position-wise FFN (mixes hidden dim)
    d_ff: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(self.d_ff, name="intermediate")(x)
        x = nn.gelu(x)
        x = nn.Dense(x.shape[-1], name="output")(x)
        return nn.Dropout(self.dropout_rate)(x, deterministic)

class FNetEncoderBlock(nn.Module):               # same shape as a BERT block; only the mixer changed
    fourier_layer: FourierTransformLayer
    ff_layer: FeedForwardLayer
    @nn.compact
    def __call__(self, x, deterministic):
        mixing_output = self.fourier_layer(x)                       # token (and hidden) mixing
        x = nn.LayerNorm(1e-12, name="mixing_layer_norm")(x + mixing_output)   # post-LN residual
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

The causal chain: attention's `O(N²)` cost comes from its `N×N` score matrix, and its real function in the block is just to mix across tokens so the feed-forward sublayer sees the whole sequence. Prior work showed that across-token mixing need not be token-dependent or even learned, so I replaced attention with a fixed linear mixer — but a learned `N×N` matrix is still quadratic in parameters and compute and length-bound. The discrete Fourier transform is the structured, parameter-free linear map whose every output is a fixed-coefficient sum over all inputs, computable in `O(N log N)`; applied as a 2D DFT over sequence and hidden with the real part taken once at the end, it mixes every token, carries position in its kernel, leaves the rest of the encoder real and unchanged, and beats every other fixed transform I tried (with Hartley tying). Drop it into the BERT block in place of attention, keep everything else, and the encoder becomes attention-free, smaller, faster, and more stable to train.
