## Research question

Multi-Query Associative Recall (MQAR) is the canonical stress test for a sequence model's ability to do
**in-context key→value lookup**: a stream of `(key, value)` token pairs is followed by query keys, and at
each query position the model must emit the value that was bound to that key earlier in the sequence. The
single thing being designed is one **sub-quadratic sequence-mixing module** — `CustomMixer` — that performs
that lookup at long context lengths. Everything else (embedding, residual stack, optimizer, data generator,
evaluation) is frozen, so any accuracy gain must come from the mixer's recall mechanism.

## Prior art before the first rung (sequence-mixing lineage)

The first rung — a recurrent mixer — is itself the resolution of a long line of memory mechanisms for
associative recall. These are the methods that precede the ladder; the fixed substrate below is what the
benchmark holds constant around whatever mixer fills the slot.

- **Neural Turing Machine (Graves et al. 2014).** Augments a recurrent controller with an external,
  content-addressable memory matrix and differentiable read/write heads, with associative recall as one of
  its headline synthetic tasks. Cracks recall when the memory is large enough, but the addressing machinery
  is intricate and brittle to train, and the controller still has to learn to drive the heads. Gap: heavy,
  fiddly external memory rather than a single drop-in mixer.
- **Recurrent nets with feedback memory (Elman 1990; BPTT/RTRL training).** Carry a fixed-size hidden state
  and fold each input into it, so per-step cost is constant — exactly the cost profile one wants at
  generation. But the gradient through the recurrence is a product of per-step factors, so it vanishes (or
  explodes) exponentially in the lag, and the fixed state has to compress the whole prefix into one vector.
  Gap: cannot reliably carry information across long lags, and the state is a fixed-size bottleneck.
- **Softmax self-attention (Vaswani et al. 2017).** At each query position reads a normalized
  exponential-weighted average over every earlier key-value pair, so a matching key a thousand tokens back
  spikes onto its value and is retrieved cleanly — the textbook recall machine, and the accuracy ceiling on
  MQAR. Gap: it forms an `N×N` score matrix, so its compute and KV-cache grow with the sequence; it is the
  ceiling, not a sub-quadratic candidate.

The open question the ladder probes is the one framed by the recall-throughput literature: which mixer
*closes the recall gap to full attention while staying sub-quadratic*, and where each one breaks as the
context length and the number of key-value pairs grow.

## The fixed substrate

A two-layer hybrid model is frozen and must not be touched. Tokens are embedded (`d_model=64`, vocab 8192,
**tied** input/output embeddings, GPT-2-style init); the first layer's mixer is a fixed **short causal
convolution** (depthwise, kernel size 3, left-padded so position `t` sees only `≤ t`); the second layer's
mixer is the editable `CustomMixer`. Each layer is a pre-norm residual block — `x + mixer(norm(x))` followed
by `x + mlp(norm(x))` with a 2× GELU MLP — and a final norm + tied head produces logits. Training is
`AdamW(lr=1e-3, weight_decay=0.1, betas=(0.9,0.95))` with cosine decay to `0.1·lr`, grad-clip 1.0, token-level
cross-entropy with `ignore_index=-100` so only the query positions contribute to the loss. The MQAR data
generator (distinct keys/values per example, power-law gap placement, random non-query filler) is ported
verbatim from the Zoology repo. The short-conv first layer exists so the editable mixer can lean on it for
the *local* shifts (lining a query token up against the key that preceded its value) and spend its own
capacity on the *global* lookup.

## The editable interface

Exactly one region is editable — the `CustomMixer` class — and every method on the ladder is a fill of the
same contract. `CustomMixer` must subclass `nn.Module`; its constructor is
`__init__(self, d_model, seq_len, ...)` (extra keyword args allowed with defaults); its
`forward(self, x)` takes and returns `[batch, seq_len, d_model]`; and it must be **causal** — the output at
position `t` may depend only on inputs at positions `≤ t`. Anything from `torch`, `torch.nn`,
`torch.nn.functional`, and `math` is allowed.

The starting point is the scaffold default: **full causal multi-head attention** wired into the slot — the
recall ceiling, included so the bar is visible from the first line. Each baseline replaces exactly this class
and nothing else.

```python
# EDITABLE region of custom_strategy.py — default fill (full causal attention)
class CustomMixer(nn.Module):
    """Causal multi-head self-attention (full softmax attention)."""

    def __init__(self, d_model: int, seq_len: int, num_heads: int = 2):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
```

## Evaluation settings

Three `(seq_len, num_kv_pairs)` settings of increasing difficulty, each trained from scratch on a single
seed (42): **`mqar-128`** (seq_len 128, 8 KV pairs, 20000 train examples, 32 epochs, batch 64),
**`mqar-512`** (512, 32 pairs, 20000 examples, 32 epochs, batch 32), and **`mqar-2048`** (2048, 128 pairs,
8000 examples, 16 epochs, batch 16). The longest setting is where a fixed-size recurrent state's
compression bottleneck is most pronounced. The metric is **`test_accuracy`** (reported also as `score`):
exact-match accuracy of the next-token prediction at every query position (where the label is not `-100`),
taking the best across all training epochs. Higher is better on all three.
