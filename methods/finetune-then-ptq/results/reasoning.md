I have a transformer that was trained in fp32, and I have to ship it with its linear weights stored at 4, 3, or even 2 bits each, because that is the only way the thing fits in memory. The act that does the damage is a single irreversible step: snap every real weight onto a signed `B`-bit integer grid. At eight bits that grid is fine and nobody worries; at two bits `qmax = 1`, so the max-magnitude scale makes adjacent in-range reconstruction values a whole largest-weight apart, and a weight sitting in the middle of that spacing can move by half the largest weight in its group. I have a small pile of in-domain text I'm allowed to touch before I freeze the final low-bit model, and the only thing that's actually scored is perplexity on a held-out split — exponentiated mean next-token cross-entropy. So the question is narrow and concrete: given that the rounding is going to hurt and can't be undone, what can I do with the weights and that text *before* I round, to make the post-rounding perplexity as low as possible?

Let me first nail down the rounding itself, because I need to know exactly what the irreversible step is before I can reason about getting ahead of it. I want low-bit integers that I can still compute with, which forces the code-to-real map to be affine, `r = S(q - Z)`: anything nonlinear and the products and sums of codes stop tracking the products and sums of the reals, and I lose the whole reason for integers. The zero-point `Z` has to be there so that real 0 lands exactly on some code — otherwise zero-padding at layer boundaries contributes a fixed nonzero value every time, which is a bias, the one kind of error that compounds through depth rather than averaging out. For a weight tensor, though, the values are roughly zero-centered and symmetric, so the natural grid is centered at zero and `Z = 0` falls out, collapsing the map to `r = S q`. That leaves a single knob, the scale `S`. With no appetite for tuning the range against data, the honest choice is the one Vanhoucke and colleagues used years ago for 8-bit CPU inference — take the largest magnitude and normalize to the signed integer container. In the code range `qmin = -2^{B-1}`, `qmax = 2^{B-1} - 1`, the scale is tied to the positive endpoint, `S = max|w| / qmax`; the extra negative code is simply outside the observed no-clip range when the weights satisfy `|w| <= max|w|`. That choice covers the weights without clipping, and any smaller step would clip the positive maximum. And I round to nearest, because for a one-shot conversion round-to-nearest minimizes each element's error, `|error| <= S/2`, and under the usual uniform-within-a-cell model the residual variance is `S^2/12`. I should at least ask whether stochastic rounding is better here, since Gupta and colleagues needed it for low-precision *training*: between adjacent grid points `l` and `u = l + S`, round to `u` with probability `(x - l) / S` and to `l` otherwise, so the expectation is exactly `x`. But their unbiasedness only earns its keep when the *same* value is rounded over and over, because then a tiny per-step bias accumulates linearly while the extra variance averages down. I'm rounding each weight exactly once. There's no accumulation to fight, so I'd be paying stochastic rounding's variance for nothing; round-to-nearest it is. And I'll make the scale per-group — one `S` per contiguous block of, say, 128 columns — because a single tensor-wide scale is wrecked by the two known one-shot failure modes: weight ranges that differ by over a hundredfold across output channels, so a narrow channel gets a grid far too coarse for it, and lone outlier weights that inflate `max|w|` and coarsen everyone. Grouping localizes both. Fine. So the irreversible step is fixed: per-group symmetric round-to-nearest, `S_g = max|w_g| / qmax`.

Now here is the wall. That conversion, however carefully I place the grid, is *blind*. It looks only at the weights, never at the data or the loss, and once it rounds, the error is frozen. The error floor is `S/2`, and at two bits `qmax = 1` so `S = max|w_g|` and that floor is `max|w_g| / 2` — not a small perturbation. The signed container has codes `-2, -1, 0, 1`, but with this no-clip scale and weights inside `[-max|w_g|, max|w_g|]`, the in-range reconstruction points are spaced by `max|w_g|`; the extra negative code does not make the cells finer. Grouping localizes the outlier and range-spread damage but does nothing to the underlying coarseness; within a group of 128 weights snapped to that spacing, most weights are still being thrown onto a brutally coarse grid. I cannot make the grid finer — `B` is fixed by the memory budget. So if I'm going to lower the post-rounding perplexity, the leverage has to come from *moving the weights before I round*, not from a cleverer rounding rule. The grid is what it is; the weights are mine to adjust right up until the moment I freeze them.

So what do I adjust them *toward*? The naive instinct is "make the quantized weights close to the original weights" — minimize `||w_hat - w||`. But stare at that for a second and it's the wrong objective, and seeing why is the whole game. I don't ship the weights; I ship the *function* the network computes, and I'm scored on its loss on text, not on how near its weights are to some reference. Two weight sets that are far apart in MSE can compute nearly the same function, and conversely a weight set that's close in MSE can have noticeably worse loss — weight-space distance and loss-space distance are simply different things. The pretrained weights were placed to minimize the *pretraining* loss in *full precision*; they have no reason to be the weights that, after being smashed onto a coarse grid, minimize the loss I actually care about on the text I actually have. So the thing to minimize before rounding is the task loss — next-token cross-entropy on the in-domain text — and the original pretrained weights are not its minimizer in any sense relevant to me.

That immediately tells me there are two distinct things a pre-rounding adjustment can buy, and I should keep them separate in my head. The first is *domain*: my held-out evaluation is on a specific corpus, and even with no quantization at all, a short fine-tune on in-domain text from the same distribution lowers the cross-entropy on that distribution — that's just ordinary adaptation, nothing to do with bits. The model arrives knowing language broadly; a few hundred steps on this particular corpus sharpens it for this particular eval. The second is *robustness to rounding*: a fine-tune driven by the real loss tends to settle the weights into a flatter, lower region of the loss surface, and weights in a flat basin are, almost by definition, ones where moving each a little — which is exactly what rounding does — costs little loss. So a plain fine-tune does double duty: it pulls the loss down for the domain, and it tends to leave the weights somewhere the rounding hurts less. Both effects are obtained by the same cheap thing — run ordinary SGD on the task loss, in full precision, before I round. And I should let it run with the master weights kept in fp32 the entire time and round only at the very end, which is the right way to read Gupta again: keep a high-precision master copy through all the updates, so the thousands of small gradient steps actually accumulate, and collapse to the grid exactly once at the finish.

But now I have to be careful, because there's a tempting next move that would change the recipe. Han and colleagues showed that fine-tuning recovers accuracy lost to quantization — but in their pipeline the fine-tune happens *with the quantization in the loop*: they quantize first, then retrain the quantized representation against the loss, so the optimizer can see the grid and place weights to sit well on it. That's powerful, but it is no longer the plain train-in-floating-point-then-quantize recipe. If I want that recipe cleanly, the forward pass during training must *not* simulate the quantization at all. The weight goes into the matmul exactly as the fp32 weight. No fake-quant, no rounding, no grid in sight. The fine-tune is plain full-precision training, and the rounding is bolted on, once, only after the loop is over.

Let me make sure I see the limitation, because a plain fp32 fine-tune is not secretly optimizing the rounded model. The fine-tune is blind to the grid by construction. It can drive the task loss down and it tends to flatten the basin, but it has no term in its objective that says "place this weight so that *after rounding with step `S`* the loss is still low." It cannot target the named failure modes either: the cross-channel range spread of over a hundredfold and the outlier weights survive a fine-tune, because nothing in next-token cross-entropy on fp32 weights directly penalizes a wide per-channel range or a stray large value — those only become costly *through* the rounding, which the fine-tune never sees. So when I finally apply the one-shot RTN, I still pay the `S/2` floor; the fine-tune lowered the loss I'm rounding *from*, and put me somewhere a bit more robust, but it did not minimize the *post-rounding* error. At eight bits, this train-in-fp-then-quantize stance is often sufficient for large models with considerable representational capacity, which is exactly the regime where the older fixed-point inference work sits. At two and three bits the floor is high and the fine-tune cannot reach under it.

Good — the recipe is forced now, and it's almost embarrassingly simple. Run the ordinary fine-tune in fp32 with the fixed text-modeling schedule: AdamW, a smallish learning rate, a few hundred steps with gradient accumulation, cosine decay with a short warmup, gradient clipping for stability, all the master weights in fp32 throughout. Then, after the loop, walk every wrapped linear and apply the no-grad per-group symmetric round-to-nearest I pinned down at the start, materializing the genuine INT-`B` model. Evaluate. The only thing that distinguishes this from doing nothing-but-quantize is that the weights I round are the fine-tuned ones, not the pretrained ones.

Let me translate that into the slots the harness leaves me. There are three. The first is the forward-pass treatment of the weight during training, `fake_quantize_weight`. For this recipe it must be the identity — the weight enters the matmul untouched, pure fp32, no grid simulated:

```python
def fake_quantize_weight(weight, num_bits, group_size):
    # Identity: no fake quant in forward -- pure FP finetune.
    return weight
```

The activation path is identity too, since this is weight-only and I'm not quantizing activations:

```python
def fake_quantize_activation(x, num_bits):
    return x
```

The second slot is the real conversion, `quantize_dequantize_weight`, applied once after the loop with no gradient — this is the irreversible step I analyzed, per-group symmetric RTN with the max-magnitude scale, the `1e-12` floor on `max|w|` so an all-zero group doesn't divide by zero, and the reshape that carves each row into groups of `group_size` columns:

```python
def quantize_dequantize_weight(weight, num_bits, group_size):
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)       # rows split into column-groups
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)    # per-group max|w|, div-by-0 guard
        scale = w_max / qmax                                           # S = max|w| / qmax (no-clip cover)
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale  # round-to-nearest, clamp, dequant
        return w_q.reshape(out_features, in_features).to(weight.dtype)
```

where the range is the symmetric signed two's-complement one:

```python
def _qrange(num_bits):
    qmax = (1 << (num_bits - 1)) - 1     # +2^{B-1} - 1
    qmin = -(1 << (num_bits - 1))        # -2^{B-1}
    return qmin, qmax
```

The third slot is the wrapper's forward. During training it just runs a plain linear on the fp32 weight — there's nothing to fake-quant, because the recipe keeps the training computation in floating point. And it's the *same* plain-linear call that's correct at eval time, because by then the harness has already overwritten `linear.weight` with the materialized INT-`B` value via the no-grad conversion above, so a plain linear on that already-quantized weight produces the genuine low-bit output with none of any train-time grid noise:

```python
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
```

And the hyperparameters come from the fixed text-modeling schedule — same learning rate, step count, batch size, accumulation, clip, warmup, and weight decay for every run using this harness:

```python
CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}
```

The wrap itself replaces every `nn.Linear` in the transformer blocks, converts any GPT-2-style `Conv1D` to a `Linear` first (its weight is stored transposed, `(in, out)` instead of `(out, in)`, so I transpose on the copy to keep the wrapping uniform across architectures), and then restores the output projection — `embed_out` for the GPTNeoX-family backbone, `lm_head` for GPT-style — to a plain `Linear` so the LM head stays full precision, since it's accuracy-critical and cheap to leave alone:

```python
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

So the causal chain is this. The grid is fixed and coarse and the rounding is irreversible, so the only leverage is moving the weights before rounding. Minimizing weight-space distance is the wrong target; the right target is the task loss on the in-domain text, which the pretrained weights do not minimize. A plain full-precision fine-tune on that loss buys two things — domain adaptation that lowers cross-entropy on the eval distribution, and a flatter basin where rounding hurts less — and it buys them with the master weights kept in fp32 throughout, collapsing to the grid only once at the end. I keep the forward pass deliberately grid-unaware because that is the canonical train-in-floating-point-then-quantize recipe; it cannot target the cross-channel range spread or the outlier weights, because those only cost loss *through* the rounding it never sees, so it still pays the `S/2` floor at two and three bits. The implementation is therefore an identity forward-pass weight transform during a standard fp32 fine-tune, followed by the one-shot per-group symmetric round-to-nearest from the start, with the LM head left full precision.
