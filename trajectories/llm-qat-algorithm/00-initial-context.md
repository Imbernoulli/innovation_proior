## Research question

Ship Pythia-1.4B with its transformer-block linear weights stored at 4, 3, or even 2 bits, and keep
the held-out perplexity as close to full precision as possible. The act that does the damage is a
single irreversible step at the end: snap every real weight onto a signed `B`-bit integer grid
(per-row, per-group of `group_size=128` columns, symmetric). At 8 bits that grid is fine and nobody
worries; at INT2 there are only four codes per group and the in-range reconstruction points are a
whole largest-weight apart, so a weight in the middle of a group can move by half the group's max.
The single thing being designed is the **training-side QAT algorithm**: how the fake-quant forward
behaves, how the gradient reaches the weight (and any extra learnable parameters) through the
non-differentiable round, and how the optimizer schedule is set — so that *after* the real
quantize-dequantize roundtrip the perplexity stays low, uniformly across INT4, INT3, and INT2. The
backbone, data, optimizer family, and the real-QDQ-then-eval harness are fixed.

## Prior art before the first rung (the lineage the ladder reacts to)

The first rung is the do-nothing control — pure round-to-nearest PTQ — and it reacts to the long line
of integer-inference work that established the format the whole task is built on.

- **Fixed-point CPU inference (Vanhoucke et al., 2011).** Take a trained fp32 net and run its linear
  layers in 8-bit integers with a per-tensor max-magnitude scale, no retraining. Established the
  affine code-to-real map and the max-abs scale. Gap: tuned for 8 bits, where the grid is fine; says
  nothing about what happens when only a handful of codes remain.
- **Integer-arithmetic-only inference (Jacob et al., CVPR 2018).** Formalized the affine scheme
  `r = S(q − Z)` with the zero-point that makes real 0 exactly representable (so zero-padding does not
  inject a bias), and showed the integer matmul payoff. Round-to-nearest, weights symmetric (`Z = 0`).
  Gap: still an 8-bit story; one-shot rounding has an error floor `S/2` that is harmless at 8 bits and
  fatal at 2.
- **Stochastic rounding for low-precision *training* (Gupta et al., 2015).** Round up with probability
  proportional to the residual, so the expectation is exact; needed because in training the *same*
  weight is rounded thousands of times and a per-step bias would accumulate. Gap: a one-shot PTQ
  conversion rounds each weight once, so stochastic rounding only buys variance for nothing.
- **The straight-through estimator (Bengio et al., 2013).** Treat the round/clamp as the identity on
  the backward pass, so a gradient can reach the weight that fed the quantizer. Gap on its own: it is
  only a *gradient* trick — it says nothing about *where* to place the grid; the step size is still a
  fixed statistic.

The format these converged to — per-row, per-group symmetric signed round-to-nearest, max-abs scale,
LM head and embeddings left full precision — is fixed for this task. The ladder is about what training
does on top of that fixed format.

## The fixed substrate

A QAT finetune-then-evaluate loop is frozen and must not be touched. It loads Pythia-1.4B
(GPTNeoX, 24 layers, hidden 2048) in fp32 with gradient checkpointing; samples random 1024-token
crops from WikiText-2 raw train; runs an AdamW finetune (`betas=(0.9,0.95)`, cosine schedule decaying
to 10% with linear warmup, gradient accumulation, global grad-norm clip); after training, walks every
wrapped linear and applies the **real, no-grad** per-group symmetric round-to-nearest to materialize
the genuine INT-`B` weights; then measures WikiText-2 raw test perplexity (sliding non-overlapping
1024-token blocks, exponentiated mean cross-entropy). The loop also sums, into the training loss, any
`aux_loss(step, total_steps)` a wrapped module exposes — the hook a method uses to add its own
objective. The optimizer is built over `model.parameters()` that `require_grad`, so any extra
learnable parameter a method registers (a per-group scale, a clipping factor) is trained automatically.

A control baseline `finetune_then_ptq` runs the *same* schedule with a pure-fp forward (no fake quant)
and then the same RTN as `no_qat`; a QAT method has to beat it, or its gains over `no_qat` are just the
in-domain finetune talking.

## The editable interface

Exactly one region is editable — lines 33–176 of `llm-qat-runtime/custom_qat.py`. Every method on the
ladder is a fill of this same contract: `CONFIG_OVERRIDES` (the per-method training schedule);
`fake_quantize_weight(weight, num_bits, group_size)` (the **differentiable** forward fake-quant —
gradient must flow back to the weight); `fake_quantize_activation(x, num_bits)` (optional, default
identity for weight-only QAT); `quantize_dequantize_weight(weight, num_bits, group_size)` (the
**real, no-grad** per-group QDQ used after training to materialize the integer model);
`class QATWrapper(nn.Module)` (wraps an `nn.Linear`, applies fake-quant in `forward`, may hold extra
learnable parameters and may expose `aux_loss`); and `prepare_qat_model(model, num_bits, group_size)`
(recursively replace every `nn.Linear` — and HF `Conv1D` — with a `QATWrapper`, then restore the LM
head, `embed_out` for Pythia, to a plain Linear so the output projection stays full precision).

The starting point is the scaffold default: **straight-through fake-quant with a fixed max-abs scale**
— the same per-group symmetric RTN in both the differentiable forward and the real eval QDQ, with the
weight as the only trainable parameter.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — default fill
CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1        # +2^{B-1} - 1
    qmin = -(1 << (num_bits - 1))           # -2^{B-1}
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    # differentiable per-group symmetric RTN; backward = straight-through
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)   # per-group max|w|
    scale = w_max / qmax                                          # fixed max-abs scale
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    w_dq = w + (w_q - w).detach()                                 # STE: grad passes to w
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x                                                      # weight-only QAT


def quantize_dequantize_weight(weight, num_bits, group_size):
    # real, no-grad per-group symmetric RTN, applied once after training
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = w_max / qmax
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        return w_q.reshape(out_features, in_features).to(weight.dtype)


class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        x = fake_quantize_activation(x, self.num_bits)
        w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
        return F.linear(x, w_q, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    from transformers.pytorch_utils import Conv1D

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
            elif isinstance(child, Conv1D):
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```

## Evaluation settings

The algorithm is run at three bit-widths, group size 128, on Pythia-1.4B, seed 42:
**`qat-1b-int4`** (INT4 — 16 codes, easy), **`qat-1b-int3`** (INT3 — 8 codes, medium), and
**`qat-1b-int2`** (INT2 — 4 codes, extreme). Primary metric `wikitext2_ppl`: WikiText-2 test
perplexity after the real QDQ roundtrip, **lower is better**. Secondary `degradation`:
`wikitext2_ppl − fp16_ppl`, with `fp16_ppl = 13.2033` the full-precision baseline measured before any
quantization. A method must work *uniformly* across all three bit-widths, and must beat the
`finetune_then_ptq` control to count as real QAT signal rather than in-domain finetune.
