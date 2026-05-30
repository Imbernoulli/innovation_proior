# Context

## Research question

Large Transformer language models are now routinely deployed for generation:
summarization, translation, question answering. At generation time the model is
run *autoregressively* — one token is produced, appended to the sequence, and
fed back in to produce the next. This is inherently sequential and turns out to
be the dominant cost of serving these models.

The precise difficulty is not arithmetic. Generating one token requires only a
modest number of floating-point operations per parameter. The difficulty is
*memory bandwidth*: at every single decoding step the accelerator must read the
model weights and, critically, the entire history of attention keys and values
from high-bandwidth memory, while doing comparatively little compute with them.
The compute units sit idle waiting for data to arrive. As the generated
sequence grows, this stored history — the per-position keys and values that each
new query must attend over — grows with it, and reloading it comes to dominate
the per-step time.

The goal is to reduce the amount of data that must be streamed from memory at
each decoding step — in particular the size of the cached keys and values —
without giving up the modelling quality that makes the large model worth serving
in the first place, and ideally without having to train an entirely new model
from scratch to get the faster variant.

## Background

**The attention layer.** A Transformer layer mixes information across positions
through attention. For a single attention head, the input sequence of $n$
vectors of dimension $d$ is projected into queries $Q$, keys $K$, and values
$V$, each $n \times k$ for head dimension $k$. The output at each position is a
softmax-weighted average of value vectors:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{k}}\right) V.$$

A layer uses $h$ heads in parallel, each with its own projection matrices,
operating on a slice of the representation; the head dimension is typically
$k = d/h$, so the heads together cost about the same as one full-width
projection. The per-head outputs are concatenated and passed through an output
projection. Multiple heads let the layer attend to several kinds of relationship
at once.

**Autoregressive decoding and the cache.** During generation, position $i$
attends over all previous positions $j \le i$. Recomputing the keys and values
of every past position at every step would be wasteful, so they are computed
once and *cached*: at step $i$ the layer appends the new $K_i, V_i$ to a running
store and attends the single new query against the whole store. This makes the
arithmetic per step linear in the history length rather than quadratic, but it
introduces a stored tensor — the key-value cache — whose size grows with the
sequence.

**The memory-bandwidth diagnostic.** Accelerators have a fixed ratio between how
fast they can compute and how fast they can read memory (the *machine balance*);
a kernel is *memory-bound* when its arithmetic intensity — operations performed
per byte loaded — falls below that ratio (the roofline picture of
Williams, Waterman, and Patterson, 2009). Consider incremental decoding of a
self-attention layer over a sequence of length $n$, batch $b$, model dimension
$d$, $h$ heads, head dimension $k = d/h$:

- *Arithmetic.* The dominant cost is the query/key/value/output projections,
  $\Theta(d^2)$ per token, so $\Theta(b\,n\,d^2)$ over the whole sequence.
- *Memory access.* Two terms. The projection matrices, reloaded each step,
  contribute $\Theta(n\,d^2)$. The cached keys and values have shape
  $[b, h, i, k]$ at step $i$; summing the bytes touched over all steps gives
  $\sum_i b\,h\,i\,k = \Theta(b\,h\,n^2 k) = \Theta(b\,n^2 d)$ since $hk = d$.

Ignoring fixed constants such as the separate reads for keys and values, the
ratio of memory access to arithmetic is therefore

$$\frac{b\,n^2 d + n\,d^2}{b\,n\,d^2} = \frac{n}{d} + \frac{1}{b}.$$

When the sequence length $n$ approaches the model dimension $d$, or the batch
$b$ is small, this ratio approaches (or exceeds) one: every operation is paired
with roughly a full memory load, and the layer is squarely memory-bound. The
$n/d$ term — the cost of streaming the growing key-value cache — is the offender,
and it comes specifically from the cache carrying a separate key and value per
head. The conclusion follows directly from the tensor shapes.

**Why this matters more, not less, at scale.** The key-value cache scales with
the model dimension $d$, while the model's parameters and FLOPs scale with $d^2$.
So compute grows faster than cache as models get larger — large models are
relatively *less* dominated by the cache term — but they also use many more
heads, so the cache per layer is large in absolute terms and reloading it is a
real wall-clock cost during serving. Standard tensor-parallel sharding makes this
worse for the most aggressive remedies: a key/value tensor that has been shrunk
below the number of model partitions must be *replicated* across partitions,
wasting the saving.

## Baselines

**Multi-head attention.** The standard layer: $h$ query heads, and a matching
$h$ key heads and $h$ value heads, each with its own learned projection. During
decoding the cache stores all $h$ key heads and $h$ value heads per position —
tensors of shape $[b, h, n, k]$. This is the source of the $n/d$ term above:
full modelling capacity (every query head has its own key/value subspace to
attend over), but the cache, and hence the per-step memory traffic, is as large
as it can be. It is the quality reference — the configuration whose output one
would like to match — and simultaneously the thing whose decoding cost one wants
to cut.

**Multi-query attention** (Shazeer, 2019). The observation: the cache is large
*because* there are $h$ key heads and $h$ value heads. So keep the $h$ *query*
heads but collapse the key and value projections to a *single* head shared across
all query heads. The cache becomes $[b, n, k]$ — a factor of $h$ smaller. Redoing
the diagnostic: the cached key/value memory term drops from $\Theta(b\,n^2 d)$ to
$\Theta(b\,n^2 k) = \Theta(b\,n^2 d / h)$. Including the small per-token
activation/cache-write term $\Theta(b\,n\,d)$, the ratio becomes

$$\frac{1}{d} + \frac{n}{d\,h} + \frac{1}{b},$$

i.e. the offending $n/d$ term is reduced by a factor of $h$. Decoding becomes
dramatically faster.

The gap it leaves: a single shared key/value head is a large cut in capacity —
every query head must now attend through the *same* key/value subspace. This
tends to degrade quality relative to multi-head attention. It can also make
training less stable — models trained this way are prone to loss spikes during
pre-training and to divergence when fine-tuned on long-input tasks. And the cut
is all-or-nothing: it goes straight from $h$ key/value heads to one, with nothing
in between, even though larger models (with more heads, and relatively less
bandwidth pressure) might prefer a less drastic reduction. Finally, a model is
either built this way or not — existing high-quality multi-head checkpoints, of
which there are many publicly available, cannot benefit without retraining.

**Continued pre-training of an existing checkpoint** (e.g. sparse upcycling,
Komatsuzaki et al., 2022). Rather than train a structurally modified model from
random initialization, one can *initialize* the modified model from a trained
checkpoint — re-using its weights where the structure is unchanged and deriving
the new parts from the old — and then continue pre-training on the original
recipe for a small fraction of the original steps so the model adapts to its new
structure. This was demonstrated for converting dense Transformers into
mixture-of-experts models. It establishes that a structural change need not cost
a full pre-training run: a few percent of the original compute can suffice to
recover quality after a surgical edit to the architecture. What it does not yet
say is how to perform such an edit on the *attention* structure, nor how to
initialize the edited key/value projections.

**Other memory-reduction routes.** Several adjacent tools attack different
parts of the serving cost. FlashAttention (Dao et al., 2022) reorganizes the
attention computation to avoid materializing the full attention matrix, which is
crucial for training and prefill but does not by itself reduce the size of the
stored key/value cache that incremental decoding reloads. Quantization lowers
the precision of weights and activations, including cached keys and values, but
it is orthogonal to the number of cached vectors. Distillation trains a smaller
student from a larger model, reducing the whole model rather than surgically
changing the decoding cache. Layer-sparse cross-attention removes some expensive
cross-attention layers for long-input encoder-decoder systems, and speculative
decoding proposes multiple tokens with a helper model before verifying them with
the large model; both can improve serving, but neither answers whether the
standard attention layer is storing more key/value heads than it needs.

## Evaluation settings

The natural yardsticks are generation tasks where autoregressive decoding cost
matters and longer inputs/outputs stress the cache:

- *Summarization*: CNN/Daily Mail (short-form), and long-input datasets — arXiv,
  PubMed, MediaSum, Multi-News. Quality measured by Rouge.
- *Translation*: WMT 2014 English-to-German.
- *Question answering*: TriviaQA.

Input/output lengths are chosen per task (e.g. 512/256 for CNN/Daily Mail and
WMT; 2048/512 for the long summarization sets; 2048/32 for TriviaQA). Models are
fine-tuned to convergence with greedy decoding at inference. Classification
benchmarks (e.g. GLUE) are excluded because autoregressive inference is not the
operative cost there. Inference speed is measured as time per sample per
accelerator chip with a profiler, at the largest batch that fits, with
parallelization tuned per model. The base architecture is a standard
encoder-decoder Transformer (T5.1.1), trained with the Adafactor optimizer on the
original T5 pre-training recipe and dataset.

## Code framework

The existing machinery is a standard decoder attention module and the decoding
loop that caches keys and values. The unresolved slot is how to reduce the
cached key/value state while still presenting tensors that the ordinary
query-key score matmul can consume.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionConfig:
    def __init__(self, hidden_size, num_heads, attention_dropout=0.0, attention_bias=False):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        # TODO: choose the key/value cache layout.


def align_key_value_heads(key_states, value_states, num_query_heads):
    # TODO: map cached key/value tensors to the query-head layout required
    # by the score matmul.
    pass


class DecoderAttention(nn.Module):
    """Decoder attention with a key/value cache for autoregressive decoding."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        # TODO: choose the key/value projections and their cached shape.
        self.k_proj = None  # TODO
        self.v_proj = None  # TODO
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = None    # TODO: project and shape cached keys
        value_states = None  # TODO: project and shape cached values

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        key_states, value_states = align_key_value_heads(
            key_states, value_states, self.num_heads
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


def convert_attention_checkpoint(pretrained_attention, config):
    """Initialize a structurally modified attention layer from a trained
    multi-head checkpoint, re-using its weights."""
    # TODO: build the new key/value projections from the trained per-head
    # projections.
    pass


def continue_pretraining(model, steps_fraction):
    """Adapt a converted checkpoint to its new structure with a small
    fraction of the original pre-training budget."""
    # TODO: run the original pre-training recipe for `steps_fraction`
    #   of the original number of steps.
    pass
```
