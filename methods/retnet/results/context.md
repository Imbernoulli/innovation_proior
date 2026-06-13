# Context

## Research question

The dominant architecture for large language models is the Transformer. It earned that
position by solving the training bottleneck of recurrent networks: because self-attention
compares every position to every other position in one matrix operation, a Transformer can
be trained teacher-forced over a whole sequence in parallel, saturating modern accelerators
in a way that a step-by-step recurrence never could.

That same design, however, makes *autoregressive inference* expensive. To generate token
\(n\), the model forms a query and compares it against the keys of all \(n-1\) previous
tokens, which is \(O(n)\) work for that single step; over a full sequence the decode cost is
quadratic. Worse, the standard remedy — caching the past keys and values so they are not
recomputed — produces a key–value cache whose size grows linearly with the sequence length.
Decoding becomes memory-bound: GPU memory, latency, and throughput all degrade as the
context grows, which is exactly the regime large language models are pushed into.

The goal, then, is a sequence-modeling primitive that holds three properties at once:

1. **Training parallelism** — the whole sequence processed in parallel, like attention, so
   training scales on GPUs.
2. **\(O(1)\) inference** — a constant amount of work and a constant amount of state per
   generated token, like a recurrent network, so decode cost is independent of how much has
   already been generated.
3. **Strong performance** — modeling quality on par with a Transformer of the same size,
   with favorable scaling as the model grows.

These three pull against each other. Parallel training wants an all-pairs operation that can
be evaluated at once; cheap inference wants a fixed-size summary of the past that is updated
one step at a time; and previous attempts to get the second have given up the third. A
solution would have to be one operator that admits both a parallel evaluation for training
and a recurrent evaluation for inference, *provably computing the same function*, without
conceding quality.

## Background

**Self-attention and where its cost comes from.** Given inputs packed as
\(X\in\mathbb{R}^{N\times d}\), attention projects \(Q=XW_Q,\,K=XW_K,\,V=XW_V\) and computes
\(\mathrm{softmax}(QK^\top/\sqrt{d})\,V\). The middle object \(QK^\top\) is an \(N\times N\)
array of pairwise scores. Forming it is \(O(N^2 d)\); applying the softmaxed matrix to \(V\)
is another \(O(N^2 d)\). For training this is acceptable — the whole sequence is known, so all
\(N\) rows are computed together in two big matrix multiplies. For autoregressive decode it is
the source of pain: the score \(\exp(q\cdot k)\) entangles query and key inside the
exponential, so it does not factor into a part depending only on the query times a part
depending only on the key. Consequently the past cannot be compressed into a fixed-size
summary; each new token must consult every old key, and the cache of those keys and values
grows without bound.

**Recurrent networks and their inverse trade-off.** A recurrence carries a fixed-size state
\(s_n\), folds the new input into it, and reads off an output: constant work per step,
constant memory. That is precisely the decode profile one wants. The cost is on the other
side: the dependence \(s_n = f(s_{n-1}, x_n)\) is inherently sequential, so training cannot be
parallelized across the time axis the way attention is.

**Relative position by rotation.** A line of work encodes position not by adding a vector to
the input but by *rotating* the query and key in a way that depends on their absolute
positions, so that the dot product depends only on the relative offset. Rotary position
embedding (RoPE, Su et al. 2021) multiplies \(q_n,k_m\) by phase factors \(e^{in\theta}\),
\(e^{im\theta}\); the score \((q_n e^{in\theta})(k_m e^{im\theta})^\dagger\) then depends on
\(n-m\). xPos (Sun et al. 2022) augments this rotation with a per-distance magnitude term so
that the relative encoding also has a length-extrapolation-friendly decay. These give a clean,
multiplicative, position-aware factorization of the query/key interaction.

**Normalization for multi-headed outputs.** When several parallel heads produce outputs with
differing variance statistics, normalizing them jointly is mismatched; group normalization
(Wu & He 2018), applied per head as in the Sub-LayerNorm / Magneto recipe (Wang et al. 2022),
normalizes each head's output separately. Group normalization is also scale-invariant:
multiplying a head's input by a scalar leaves the normalized output and its gradient unchanged.

**The diagnostic that frames the problem.** Laying the candidate architectures side by side
on the three axes makes the gap concrete: a Transformer has training parallelism and strong
performance but \(O(N)\)-per-step inference with an \(O(N^2)\) memory footprint over a long
sequence; a recurrent network has \(O(1)\) inference but no training parallelism and weaker
performance; the efficient attention variants each recover one missing corner while losing
another. No existing primitive occupies all three corners simultaneously.

## Baselines

**Linear attention (Katharopoulos et al. 2020, "Transformers are RNNs").** Replace the
exponential similarity with a factored kernel \(\phi(q)^\top\phi(k)\). Then
\((\phi(Q)\phi(K)^\top)V=\phi(Q)(\phi(K)^\top V)\) by associativity, and in the causal case the
key-side sum \(S_n=\sum_{m\le n}\phi(k_m)v_m^\top\) becomes a running state updated by a
rank-one addition each step. This delivers \(O(1)\) recurrent inference and \(O(N)\)-memory
parallel training, and it explicitly shows that a causal attention is a kind of RNN over time.
Its weaknesses: the output is a *normalized* average, \(\phi(q_n)^\top S_n / (\phi(q_n)^\top
z_n)\) with \(z_n=\sum_{m\le n}\phi(k_m)\), and that denominator is the awkward part — it
couples and dilutes the weights, the choice of feature map \(\phi\) matters a great deal, and
the resulting models trail Transformers in quality and encode position poorly.

**Recurrence with element-wise mixing (RWKV, Peng et al. 2023).** Returns to an explicitly
recurrent formulation with an exponential time-decay so that inference is \(O(1)\) and the
model is competitive in quality. To keep the recurrence cheap it mixes tokens with
*element-wise* (per-channel scalar) operations rather than a full outer-product state. That
keeps per-step cost low but limits the channel-mixing capacity of the state, and the
element-wise recurrence is not a matrix multiply, so it does not parallelize over time the way
an all-pairs operation does.

**State-space and long-convolution models (S4, Gu et al. 2021; H3; Hyena).** Replace attention
with a linear time-invariant state-space recurrence \(s_n = A s_{n-1} + B x_n\),
\(o_n = C s_n\), whose unrolled form is a long convolution that can be evaluated in parallel by
FFT, giving training parallelism and \(O(1)\) or \(O(N\log N)\) cost. They model long range
well, but the mixing kernel is *content-unaware*: \(A,B,C\) do not depend on the token being
processed, unlike a query/key score that is computed from the content.

**Attention-free / gated variants (AFT).** Simplify dot-product attention to element-wise
operations and move the normalization onto the keys. Cheap, but again element-wise mixing
caps the representational capacity relative to a high-dimensional interaction.

Each baseline occupies two corners of the triangle and gives up the third: linear attention
and the SSM line keep parallelism but lose quality (or content-awareness); the element-wise
recurrences keep quality and \(O(1)\) inference but lose training parallelism and full mixing.

## Evaluation settings

The natural yardstick is autoregressive language modeling. Models are trained from scratch at
several sizes (on the order of 1–7B parameters) on a large curated text mixture (web text,
books, and code corpora such as The Pile, C4, and The Stack), with a fixed token budget,
AdamW, warmup-then-decay learning-rate schedules, and a maximal training length of a couple of
thousand tokens. Quality is read off as validation perplexity and as zero-shot / few-shot
accuracy on standard commonsense and reasoning suites (HellaSwag, BoolQ, COPA, PIQA, Winograd,
Winogrande, StoryCloze). Because the whole point is the cost profile, the protocol also
measures, separately, **training** throughput and memory (against a standard Transformer and
against a fused-attention implementation) and **inference** GPU memory, decoding throughput,
and latency as functions of sequence length and batch size. Long-context behavior is probed by
reporting perplexity at increasing context lengths.

## Code framework

An autoregressive language-model scaffold. The data pipeline, optimizer, loss, and
block layout (pre-norm residual blocks of a token-mixing module plus a position-wise
feed-forward) already exist; the open slot is the token-mixing operator itself and the way it
is evaluated for training versus inference.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMixer(nn.Module):
    """The sequence-mixing operator — the slot the contribution fills.

    It must admit (at least) two evaluation modes that compute the SAME function:
      - a parallel mode over a whole sequence, for training;
      - a step-by-step mode carrying a fixed-size state, for O(1) inference.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # TODO: projections and any position-dependent factors
        pass

    def parallel_forward(self, x):
        # TODO: evaluate all positions at once (training)
        pass

    def recurrent_forward(self, x_t, state):
        # TODO: fold one token into a fixed-size state, read off output (inference)
        pass

    def forward(self, x, state=None):
        if state is not None:
            return self.recurrent_forward(x, state)
        return self.parallel_forward(x)


class Block(nn.Module):
    """Pre-norm residual block: token mixer, then position-wise FFN."""
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mixer = TokenMixer(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x, state=None):
        y = self.mixer(self.norm1(x), state=state) + x
        z = self.fc2(F.gelu(self.fc1(self.norm2(y)))) + y
        return z


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ffn_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [Block(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
```
