## Research question

Multi-Query Associative Recall (MQAR) tests whether a sequence model can perform **in-context key→value lookup**: a stream of `(key, value)` pairs is followed by query keys, and at each query position the model must emit the value bound to that key earlier in the sequence. The only component being designed is one **sub-quadratic sequence-mixing module** — `CustomMixer` — that performs that lookup at long context lengths. Everything else (embedding, residual stack, optimizer, data generator, evaluation) is frozen, so any accuracy gain must come from the mixer's recall mechanism.

## Prior art / Background / Baselines

- **Neural Turing Machine (Graves et al. 2014).** It augments a recurrent controller with an external content-addressable memory matrix and differentiable read/write heads. Gap: the addressing machinery is intricate and brittle to train, and the controller must learn to drive the heads, so the mechanism is not a single drop-in sequence mixer.

- **Recurrent nets with feedback memory (Elman 1990; trained by BPTT/RTRL).** They maintain a fixed-size hidden state and fold each input into it, giving constant per-step cost. Gap: gradients through the recurrence are products of per-step factors, so they vanish or explode exponentially with lag, and the fixed state must compress the whole prefix into one vector.

- **Softmax self-attention (Vaswani et al. 2017).** At each query position it reads a normalized exponential-weighted average over every earlier key-value pair, so a matching key far back can spike onto its value and be retrieved cleanly. Gap: it forms an `N×N` score matrix, so its compute and KV-cache grow with sequence length; it is not a sub-quadratic candidate.

The open question is how far a sub-quadratic mixer can push recall accuracy before it breaks as the context length and the number of key-value pairs grow.

## Fixed substrate / Code framework

A two-layer hybrid model is frozen and must not be touched. Tokens are embedded (`d_model=64`, vocab 8192, tied input/output embeddings, GPT-2-style init). The first layer's mixer is a fixed short causal convolution (depthwise, kernel size 3, left-padded so position `t` sees only `≤t`). The second layer's mixer is the editable `CustomMixer`. Each layer is a pre-norm residual block — `x + mixer(norm(x))` followed by `x + mlp(norm(x))` with a 2× GELU MLP — and a final norm + tied head produces logits. Training uses `AdamW(lr=1e-3, weight_decay=0.1, betas=(0.9,0.95))` with cosine decay to `0.1·lr`, grad-clip 1.0, and token-level cross-entropy with `ignore_index=-100` so only query positions contribute to the loss. The MQAR data generator (distinct keys/values per example, power-law gap placement, random non-query filler) is ported verbatim from the Zoology repo. The short-conv first layer handles local shifts, leaving the editable mixer to focus on the global lookup.

## Editable interface

Exactly one region is editable — the `CustomMixer` class. `CustomMixer` must subclass `nn.Module`; its constructor is `__init__(self, d_model, seq_len, ...)` (extra keyword args allowed with defaults); its `forward(self, x)` takes and returns `[batch, seq_len, d_model]`; and it must be causal — output at position `t` depends only on inputs at positions `≤t`. Anything from `torch`, `torch.nn`, `torch.nn.functional`, and `math` is allowed.

The starting fill is full causal multi-head attention, included as the visible recall ceiling.

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

Three `(seq_len, num_kv_pairs)` settings of increasing difficulty are trained from scratch on seed 42: **`mqar-128`** (seq_len 128, 8 KV pairs, 20000 train examples, 32 epochs, batch 64), **`mqar-512`** (512, 32 pairs, 20000 examples, 32 epochs, batch 32), and **`mqar-2048`** (2048, 128 pairs, 8000 examples, 16 epochs, batch 16). The metric is **`test_accuracy`** (reported also as `score`): exact-match accuracy of next-token prediction at every query position (where the label is not `-100`), taking the best across all training epochs. Higher is better on all three.
