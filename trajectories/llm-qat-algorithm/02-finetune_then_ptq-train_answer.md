The PTQ control told me in numbers exactly how blind rounding scales with bit-width, and the shape is the one the $S/2$ floor predicted: INT4 landed at 15.01 perplexity (only 1.81 above the 13.2033 full-precision baseline), INT3 jumped to 42.54, and INT2 detonated to 104422 — every weight smashed onto a four-code grid, the model effectively destroyed. The diagnosis is sharp: this is a *no-repair* problem, and the only leverage is to move the weights before I round, which needs the loss and optimizer the control switched off. But before I credit any quantization-aware method, I have to confront a confound the harness was built to expose: QAT here finetunes on WikiText-2 train and is scored on WikiText-2 test, the same domain, so a plain in-domain finetune lowers test cross-entropy all by itself with no bits involved. If a QAT method beats `no_qat`, I cannot tell how much was the *quantization-aware* part and how much was just the finetune talking.

So I propose `finetune_then_ptq`, the **full-precision finetune control**: run the standard text-modeling finetune in pure fp32, with no grid ever simulated during training, then apply the identical no-grad per-group RTN as `no_qat` once at the end. It is not a contender — its job is to set the bar a real QAT method must clear, so that later I can read any method's improvement over `no_qat` and know how much was finetune and how much was QAT signal. What makes this a clean control is that the forward pass during training is pure fp32: `fake_quantize_weight` is the *identity*, `return weight`, full stop. The weight enters the matmul exactly as the fp32 value — no fake-quant, no rounding, no grid in sight — so the optimizer never sees the quantizer, and whatever this rung buys is, by construction, *not* QAT.

The load-bearing insight is what to adjust *toward*. The naive instinct is to make the quantized weights close to the originals, minimizing $\lVert \hat w - w\rVert$, and seeing why that is wrong is the whole game. I do not ship the weights; I ship the *function* the network computes, and I am scored on its loss on text, not on how near its weights sit to some reference. Two weight sets far apart in MSE can compute nearly the same function, and a weight set close in MSE can have noticeably worse loss — weight-space distance and loss-space distance are simply different things. The pretrained weights were placed to minimize the *pretraining* loss in *full precision*; they have no reason to be the weights that, after being smashed onto a coarse grid, minimize the test perplexity I care about. So the thing to minimize before rounding is the task loss, and the pretrained weights are not its minimizer in any sense relevant to me.

That tells me a plain finetune buys two distinct things, which I keep separate. The first is *domain*: even with no quantization at all, a short finetune on text from the eval distribution lowers cross-entropy on that distribution — ordinary adaptation, exactly the confound I am measuring. The second is *robustness to rounding*: a finetune driven by the real loss tends to settle the weights into a flatter, lower basin of the loss surface, and weights in a flat basin are almost by definition ones where moving each a little — which is what rounding does — costs little loss. Both effects come from the same cheap thing: run ordinary AdamW on the task loss, in full precision, before I round. I keep the master weights in fp32 the entire time and collapse to the grid exactly once at the finish — the right reading of the low-precision-training lineage, where a high-precision master copy lets thousands of small updates actually accumulate.

I have to resist a tempting next move that would quietly turn this control into a method. The accuracy-recovery line finetunes *with the quantization in the loop* — quantize first, then retrain the quantized representation against the loss so the optimizer sees the grid. That is powerful, but it is no longer the plain train-in-fp-then-quantize recipe, and it would defeat this rung's purpose. To keep the control clean the training forward must not simulate the quantization *at all*. The very next rung, STE QAT, does the opposite — it flips `fake_quantize_weight` back to a straight-through quantizer so the grid *is* in the forward — and because both rungs share the identical 500-step schedule, the gap between them is the genuine QAT contribution with the in-domain finetune held constant.

It is important to be honest that this finetune is *not* secretly optimizing the rounded model: it is blind to the grid by construction. It drives the task loss down and tends to flatten the basin, but it has no term saying "place this weight so that *after rounding with step $S$* the loss is still low." It cannot target the named failure modes either — cross-channel range spread of over a hundredfold, and outlier weights, both survive a finetune, because next-token cross-entropy on fp32 weights never penalizes a wide per-channel range or a stray large value; those become costly only *through* the rounding, which the finetune never sees. So when I finally apply the one-shot RTN I still pay the $S/2$ floor: the finetune lowered the loss I am rounding *from* and left me somewhere a bit more robust, but it did not minimize the post-rounding error. The falsifiable expectation that follows is exact — beat `no_qat` comfortably at INT4 (possibly under 13.2033, a negative degradation), improve clearly at INT3 but stay well above full precision, and *stay collapsed* at INT2 — and that surviving INT2 gap is precisely the bar the real QAT methods must clear.

The recipe is therefore almost embarrassingly simple. Run the ordinary finetune in fp32 with the same fixed schedule every QAT rung will use — `learning_rate=2e-5`, `num_steps=500`, `batch_size=2`, `gradient_accumulation_steps=4`, `max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0` — with master weights in fp32 throughout, then let the fixed harness apply the no-grad per-group symmetric RTN once and evaluate. The wrapper's forward is a plain linear on `self.linear.weight`, correct both during training (pure fp) and at eval (the harness has already overwritten the weight with the materialized INT-$B$ value). This differs from the scaffold default in exactly one place — the default's `fake_quantize_weight` simulates the grid, this control's is the identity — and that single change is the whole definition of "no QAT signal." The `quantize_dequantize_weight` and the wrap-and-restore-the-head logic are identical to `no_qat`'s.

```python
# EDITABLE region of custom_qat.py (lines 33-176) — step 2: finetune_then_ptq (control)

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
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    # Identity: no fake quant in forward -- pure FP finetune.
    return weight


def fake_quantize_activation(x, num_bits):
    return x


def quantize_dequantize_weight(weight, num_bits, group_size):
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
        # Pure FP forward during training (no fake quant).  At eval the real
        # QDQ has already been applied to linear.weight, so this still
        # produces the genuine INT-N output.
        return F.linear(x, self.linear.weight, self.linear.bias)


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
