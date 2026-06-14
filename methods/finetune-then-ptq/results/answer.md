# Finetune-then-PTQ, distilled

Finetune-then-PTQ produces a low-bit weight-quantized model by (1) running a plain
full-precision fine-tune of the trained model on in-domain text, with the forward pass kept
deliberately quantization-unaware (no fake-quant, pure fp32), and then (2) applying a single
irreversible per-group symmetric round-to-nearest (RTN) quantize-dequantize to every linear weight
to materialize the integer model for evaluation. It is the standard "train in floating point, then
quantize the trained weights" recipe: often adequate in high-bit or high-capacity regimes, and a
deliberate control baseline at very low bit-width because the fine-tune never sees the grid.

## Problem it solves

Store a trained transformer's linear weights at very low bit-width (`B in {4, 3, 2}`) while keeping
language-modeling perplexity close to full precision. The irreversible rounding onto a signed
integer grid is brutal at small `B`; the only leverage is to move the weights before rounding. The
right thing to lower is the *task loss* on the eval distribution, not the weight-space distance to
the originals.

## Key idea

- **Weight-space MSE is the wrong target; the task loss is the right one.** The pretrained weights
  minimize the pretraining loss in full precision; they are not the weights that, after being
  snapped to a coarse grid, minimize the next-token cross-entropy on the corpus actually being
  scored. A short fine-tune on in-domain text, driven by the real loss, lowers that loss directly.
- **A plain fine-tune buys two things.** (a) *Domain adaptation*: even with no quantization, fitting
  in-domain text lowers cross-entropy on the eval distribution. (b) *Robustness to rounding*: the
  loss-driven fine-tune tends to settle weights into a flatter basin, where the small per-weight
  perturbation that rounding inflicts costs less loss. Both come from the same cheap operation —
  ordinary fp32 SGD on the task loss, with fp32 master weights kept all the way through, collapsed
  to the grid exactly once at the end (echoing the low-precision-training lesson that the master copy
  must stay high-precision while the many small updates accumulate).
- **The fine-tune stays grid-unaware.** Folding the
  rounding into the forward pass (quantization-aware training) would let the optimizer place weights
  to sit well on the grid — but then a measured gain confounds "in-domain fine-tune" with
  "grid-awareness." Keeping the forward pure fp32 isolates the fine-tune signal.
- **It cannot reach under the RTN floor at low bit-width.** The one-shot RTN error is bounded by
  `S/2`; at `B = 2`, `qmax = 1` so `S = max|w_group|` and the floor is `max|w_group|/2`. The
  signed container has codes `-2, -1, 0, 1`, but with this scale the extra negative code is outside
  the observed no-clip range, so it does not reduce the in-range spacing. The grid-blind fine-tune
  does not target the documented one-shot failure modes (cross-channel weight ranges differing
  >100x; outlier weights coarsening a group's scale), because those cost loss only *through* the
  rounding the fine-tune never sees. So at 2-3 bits it still pays that floor.

## Final recipe

1. Wrap every transformer `nn.Linear` so the fine-tune can update it (convert GPT-2 `Conv1D` to
   `Linear`); restore the LM head to a plain `Linear` so the output projection stays full precision.
2. Fine-tune in full precision on in-domain text — AdamW, cosine LR with linear warmup, gradient
   accumulation, grad-norm clipping. Forward pass uses the fp32 weight unchanged (no fake-quant).
3. After the loop, apply per-group symmetric RTN once, no grad, to every wrapped weight. The affine
   map is `r = S(q - Z)`; for symmetric weight groups `Z = 0`, so `r = S q`. For each contiguous
   group of `group_size` columns, `S = max|w_group| / qmax` with `qmin = -2^(B-1)`,
   `qmax = 2^(B-1) - 1`, then `w_hat = S * clamp(round(w / S), qmin, qmax)`.
4. Evaluate perplexity on the held-out split.

## Working code

Fills the harness's three slots: `fake_quantize_weight` is the identity (pure fp32 forward),
`quantize_dequantize_weight` is the no-grad per-group symmetric RTN applied after training, and the
wrapper's forward is a plain linear on the (fine-tuned, then materialized) weight.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Finetune-then-PTQ control baseline: forward pass during training is pure FP
# (no fake quant), with the same training schedule as the QAT methods. After
# training, real RTN QDQ materializes the integer model.

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
    qmax = (1 << (num_bits - 1)) - 1     # +2^{B-1} - 1
    qmin = -(1 << (num_bits - 1))        # -2^{B-1}
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    # Identity: no fake quant in forward -- pure FP finetune.
    return weight


def fake_quantize_activation(x, num_bits):
    return x  # weight-only: activations stay full precision


def quantize_dequantize_weight(weight, num_bits, group_size):
    # Real (no-grad) per-group symmetric RTN, applied once after training.
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)       # rows -> column-groups
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)    # per-group max|w|, div-by-0 guard
        scale = w_max / qmax                                           # S = max|w| / qmax (no-clip cover)
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale  # round-to-nearest, clamp, dequant
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
        # Pure FP forward during training (no fake quant).  At eval time the
        # real QDQ has already been applied to linear.weight, so this still
        # produces the genuine INT-N output.
        return F.linear(x, self.linear.weight, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    from transformers.pytorch_utils import Conv1D

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name,
                        QATWrapper(child, num_bits=num_bits, group_size=group_size))
            elif isinstance(child, Conv1D):
                # GPT-2 Conv1D stores weight as (in, out); Linear wants (out, in).
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name,
                        QATWrapper(lin, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    # Keep the output projection (GPTNeoX embed_out / GPT-style lm_head) full precision.
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```

## Relation to prior approaches

- **One-shot RTN of the pretrained weights (data-free PTQ):** identical final conversion, but rounds
  the *pretrained* weights; finetune-then-PTQ rounds the *fine-tuned* weights, so it captures the
  in-domain fine-tune gain on top.
- **8-bit fixed-point inference of a trained net (Vanhoucke et al. 2011):** the same "train in fp,
  then quantize" stance and the same max-magnitude symmetric scale, but at 8 bits on large nets where
  the grid is fine enough that one-shot conversion can be adequate; at 2-4 bits the floor dominates.
- **Quantize-then-retrain the quantized representation (Han et al. 2015, Deep Compression):**
  fine-tunes *with* the quantization in the loop (quantization-aware) over a non-uniform codebook;
  finetune-then-PTQ fine-tunes in fp32 *before* a single uniform-affine RTN, staying grid-unaware so
  it serves as the control rather than the grid-aware method.
