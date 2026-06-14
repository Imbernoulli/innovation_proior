I have a network that was trained in 32-bit floating point, and I have to run it on a machine where floating point is the slow path — a mobile CPU, an edge DSP — and where what's cheap is integer arithmetic and what's scarce is memory bandwidth. So the question in front of me is concrete: how do I take a tensor of real-valued weights and stand it up as a tensor of low-bit integers, such that an integer-friendly evaluation of each layer still produces nearly the same numbers the floating-point layer would, and such that the conversion costs me essentially nothing — no retraining, no data, just arithmetic on the weights I already have. I need to start from what an integer code even *means*.

An integer code `q` is just an index. By itself it's nothing; it only becomes a number because I attach a rule that says "code `q` stands for real value `r`." So the first thing I have to choose is that rule, the correspondence `q ↦ r`. And here's the constraint that pins down its shape almost immediately: I don't want the integers as a lookup table, I want to *compute* on them. A layer is a matrix multiply — sums of products of weights and activations. If I store weights as codes and activations as codes, I'd like the products and sums of the *codes* to track the products and sums of the *real values*, so that I can do the whole accumulation in integers and only convert back at the very end. For that to work, the map from code to real value can't be anything exotic. Suppose `r = g(q)` for some function `g`. A real product is `r1·r2 = g(q1)·g(q2)`, and a real sum is `g(q1) + g(q2)`. The only way these compose back into something I can read off the integer operations cleanly — the only `g` under which "multiply the reals" becomes "do integer work on the codes, then one fix-up" — is when `g` is *affine*: `r = A·q + B`. Anything nonlinear and the product `g(q1)g(q2)` stops being expressible through `q1·q2` plus cheap corrections, and the whole point of working in integers evaporates. So the correspondence is affine. Good, that's forced, not chosen.

Now, `r = A·q + B` has two constants. Let me look harder at `B`, because there's a domain fact nagging me. Convolutions and poolings pad their borders with zeros. In the quantized layer, that padding has to be *some* integer code, and whatever real value that code denotes is what actually gets added at the border. If the real value 0 is *not* exactly representable — if no integer code maps to exactly 0 — then I'm forced to pad with a code that stands for some small nonzero number, and every padded position contributes that same small error to its output. Same error, every time: that's not noise, that's a *bias*, a non-zero-mean error term, and biases are exactly what compound across a deep net. So I need a hard guarantee: real 0 is exactly some code. Let me see what that costs in the form `r = A·q + B`. Set `r = 0` at the special code `q = Z`: `0 = A·Z + B`, so `Z = −B/A`. That's awkward — it demands that the real ratio `−B/A` come out to be exactly an integer, which is a fragile thing to arrange. Let me reparameterize. Write the same affine map as `r = C·(q + D)`. Plug `r = 0, q = Z`: `0 = C·(Z + D)`, and now `C` cancels entirely, leaving `D = −Z`. So if I write

  r = S·(q − Z),

with `S` a positive real (the *scale*, the step size) and `Z` an integer (the *zero-point*, the code that maps to real 0), then "0 is exactly representable" is satisfied by construction — `q = Z` gives `r = 0` no matter what `S` is. That's the right parameterization. The same map, but the zero-exact constraint is free instead of a knife-edge. So my correspondence is `r = S(q − Z)`, two knobs: a positive real step size and an integer zero-point.

Now the rule that picks a code for a given real `r`. Inverting the affine map, `q = r/S + Z`, but `r/S + Z` is generally not an integer, so I have to round it to the nearest representable code and clamp it into the allowed range. Which range? For `B` bits I have `2^B` codes. Two honest sub-questions: where do I put the grid (choose `S` and `Z`), and how do I round. Let me do rounding first because it's the cleaner of the two.

When I replace a real weight `w`, the general affine map says the code before clamping is `q = round(w/S + Z)`, and the dequantized value is `ŵ = S·(q − Z)`. So the error is `e = ŵ − w = S·(round(x) − x)` with `x = w/S + Z`. Since `Z` is an integer, it shifts `x` by whole grid cells and does not change the fractional part. Let `u = x − ⌊x⌋ ∈ [0,1)`. If I round to nearest, I pick `⌊x⌋` when `u ≤ 1/2` and `⌊x⌋+1` when `u > 1/2`, so the rounding residual `round(x) − x` is `−u` in the first case and `1 − u` in the second, and in both cases its magnitude is `≤ 1/2`. Multiply back by `S`: `|e| ≤ S/2`. So the worst-case per-weight error is half a step, and adding a bit halves `S`, which halves that bound — the textbook "one bit ≈ halve the error." Fine. But the worst case isn't what hurts a deep net; the *bias* is. Across a whole tensor, if the values `x` are spread over many grid cells — which they are when `S` is small relative to the spread of the weights — then the fractional parts `u` are roughly uniform on `[0,1)`. Under that, the rounding residual `round(x) − x` is roughly uniform on `[−1/2, 1/2)`: it has *mean zero* and variance `1/12`. So round-to-nearest gives me, per element, an error of mean ≈ 0 and variance ≈ `S²/12`. Mean zero is the property I want — no systematic shift of the layer output, just zero-centered jitter that partially averages out across the sum in a matmul.

Compare the alternatives. If I truncate (always round down), the residual is `−u`, always negative, mean `−1/2`, so the real error has mean about `−S/2`: a flat downward bias on every weight, exactly the compounding error I'm trying to avoid. What about stochastic rounding — round up with probability `u`, down with probability `1−u`? Its expected value is `⌊x⌋·(1−u) + (⌈x⌉)·u = ⌊x⌋ + u = x` exactly, so it's *unbiased* for any single `x`, not just on average over a uniform tensor. That's a stronger guarantee than round-to-nearest's. So why not use it? Because I have to look at the variance and at *what I'm actually doing*. Stochastic rounding's residual is `1−u` with probability `u` and `−u` with probability `1−u`; in real units its variance is `S²·u(1−u)`, which is up to `S²/4` at `u = 1/2` — up to three times the `S²/12` of round-to-nearest under the uniform-fractional-part model. Stochastic rounding *buys* pointwise unbiasedness by *spending* variance. When is that trade worth it? When I round the *same quantity over and over*, because then a small per-rounding bias accumulates linearly into a large drift, while the extra variance averages down. That's precisely the training setting — a weight gets nudged and re-rounded thousands of times, and there a deterministic bias is fatal, which is the whole reason stochastic rounding was introduced for low-precision training. But I'm not training. I'm doing a *one-shot* conversion: each weight is rounded exactly once, to a fixed target, and then it's done. There's no accumulation to worry about, so the anti-bias benefit of stochastic rounding has nothing to bite on, and I'd be paying its variance for free. For a one-shot conversion the right objective is just "make each `ŵ` as close to `w` as possible," and the rule that minimizes `|ŵ − w|` element by element is, by definition, round-to-nearest. So: round-to-nearest. The repeated-rounding regime that makes stochastic rounding necessary simply isn't present here.

Now the grid placement — `S` and `Z`. Start with the weights' distribution. A trained weight tensor is, to good approximation, centered near zero and roughly symmetric — positive and negative weights in comparable numbers, no strong one-sided skew (unlike, say, post-ReLU activations, which are all `≥ 0`). For a symmetric, zero-centered quantity, the natural grid is symmetric about zero, and then the zero-point falls out: I want code 0 to denote real 0, so `Z = 0`. And look what `Z = 0` does to the affine map — it collapses to `r = S·q`, a pure scaling. That's not just tidy; it's cheaper downstream. In the integer matmul, the general scheme `r = S(q − Z)` expands `(q1 − Z1)(q2 − Z2)` into the core `q1·q2` plus cross terms in `Z1` and `Z2` that I'd have to subtract off. With `Z = 0` on the weights, all of those cross terms vanish. So for weights, symmetric quantization with `Z = 0` is both the statistically natural choice and the computationally lean one. I'll keep the general zero-point in my back pocket for one-sided activations, where the data isn't centered and a nonzero `Z` genuinely buys range — but here, weights, `Z = 0`.

With `Z = 0`, the only remaining knob is `S`, and the bit-width fixes the rest of the range. I have `2^B` signed codes. The signed integer range for `B` bits is `[−2^{B−1}, 2^{B−1} − 1]`: call `qmin = −2^{B−1}` and `qmax = 2^{B−1} − 1`. There is one extra negative two's-complement code, so if I want a symmetric weight grid around zero I should make `+qmax` stand for `+w_max` and `−qmax` stand for `−w_max`; `qmin` remains the extra negative rail used only by the clamp. Now where to put `S`. The defining stance of a no-tuning conversion is: don't optimize the range against any data, just *cover* the weights. Let `w_max = max |w|` over the tensor (or, as I'll refine in a second, over a group). I want the positive endpoint to land on the positive code without being clipped, so `w_max / S = qmax`, i.e.

  S = w_max / qmax.

This is the no-clip extreme of a tradeoff I should name out loud, because it's the crux of the "no tuning" stance. I could pick a *smaller* `S` — clip the range tighter than `w_max` — which would shrink the rounding error `S/2` on the bulk of the weights, but at the cost of *clipping* the few weights beyond the tighter range, replacing their value with the rail. That's trading rounding error for clipping error, and finding the sweet spot requires looking at the data and optimizing — exactly the calibration step I'm refusing to do. The max-magnitude scale is the choice that needs no data: it is the smallest `S` that covers both `+w_max` and `−w_max` with the symmetric usable codes `±qmax`, hence the smallest rounding error achievable *without* introducing clipping error. So `S = w_max / qmax`, per the round-to-nearest, no-calibration philosophy. (One guard: if a whole group is all zeros, `w_max = 0` and `S = 0` divides by zero. Floor `w_max` at a tiny constant, `1e-12`, so the degenerate group just maps to all-zero codes instead of NaN.)

Let me pause on "per the tensor" versus "per a group," because this is where I can do real good cheaply. If I use one `S` for the entire weight matrix, I'm vulnerable to two things I know break one-shot quantization. First, different output channels of the same layer can have weight ranges that differ by more than a hundredfold; with a single `S` sized to the widest channel, the narrow channels see a `S` that's enormous relative to their own weights, so almost all their precision is wasted and their relative error explodes. Second, a single outlier weight inflates `w_max` for the *whole* tensor, blowing up `S`, and now every ordinary weight is being rounded on a needlessly coarse grid because of one lonely large value. Both are the same disease: one scale forced to serve very different local scales. The cure is to make the scale *local*. Partition each weight matrix into contiguous groups of columns — say `group_size = 128` — and give each group its own `w_max` and its own `S`. Now an outlier only coarsens the 128 weights in its own group, not the millions in the tensor, and a narrow channel's group gets a scale matched to its own magnitude. It costs me one extra scale per group of 128 weights — negligible storage — and it directly attacks both failure modes. So: per-group symmetric round-to-nearest, `S_group = max|w_group| / qmax`.

Let me make sure I see *why this pays off at all*, the integer-matmul payoff, because that's the entire justification for going to integers rather than just compressing storage. Take two real matrices, quantize both with the affine scheme. An entry of the product is `Σ_j r1_ij · r2_jk = Σ_j S1(q1_ij − Z1)·S2(q2_jk − Z2) = S1·S2·Σ_j (q1_ij − Z1)(q2_jk − Z2)`. The only real-valued thing left outside the sum is the constant `S1·S2`; the sum itself is *pure integer*. To express the result back as a quantized value `r3 = S3(q3 − Z3)`, divide by `S3`: `q3 = Z3 + (S1·S2/S3)·Σ_j (q1−Z1)(q2−Z2)`. The single non-integer is the multiplier `M = S1·S2/S3`, a constant I can compute offline. The implementation can represent that multiplier as fixed-point arithmetic: normalize it as `M = 2^{−n}·M0` with `M0 ∈ [0.5,1)` after choosing the shift `n`, store `M0` as a high-precision integer multiplier, and make `2^{−n}` a rounding right-shift. So the whole layer becomes: an integer multiply-accumulate into a 32-bit accumulator, one fixed-point multiply, one shift, and an add of `Z3`. No floating point in the hot loop. And with the weights symmetric (`Z1 = 0`), the `(q1 − Z1)(q2 − Z2)` cross terms in `Z1` drop, so I'm left essentially with the bare `Σ q1·q2` plus the activations' zero-point bookkeeping — the leanest possible kernel. *That's* why integer quantization earns its keep: the expensive `O(N³)` part runs entirely on small integers, and everything else is `O(N²)` with tiny constants. The bias vectors, note, I keep at high precision — quantize them as int32 with zero-point 0 and scale `S1·S2` (the accumulator's own scale) — because a bias is added to *many* outputs, so any error in it acts as a non-zero-mean term across all of them, the one place I cannot tolerate a bias.

So I have the scheme. Let me now ask the uncomfortable question: how far does this *actually* get me, especially at very low bit-widths — 4, 3, 2 bits? At `B = 4` there are 16 codes; at `B = 3`, 8; at `B = 2`, just 4. Round-to-nearest's error is bounded by `S/2 = w_max/(2·qmax)`. At `B = 2`, `qmax = 1`, so `S = w_max` and the rounding error can reach `w_max/2` — half the largest weight in the group. That's no longer a small perturbation for weights near the middle of the range. And there's no mechanism in this scheme to *repair* it: I round once, and whatever error I incur is frozen into the model. The two failure modes I cured with grouping are mitigated but not eliminated — grouping localizes range spread and outliers, but within a group of 128 weights at 4 codes, most weights are still being snapped to a very coarse grid. I can feel the wall: one-shot rounding, however carefully I place the grid, has a floor set by the grid's coarseness, and at 2-3 bits that floor is high. To go below that floor I would need to let weights move to compensate for the rounding of their neighbors, which means data, gradients, and optimization. Here, the whole premise is *no* training, so I should keep the method honest: it is the cheapest conversion, the zero-data reference point, and its value is precisely that nothing except the fixed rounding rule is allowed to change the trained weights.

Let me also be careful about one implementation subtlety so the *same* quantization function can serve both a pure-PTQ run and, if someone wanted, a train-through-it run, without two code paths diverging. The mapping `ŵ = S·clamp(round(w/S), qmin, qmax)` is non-differentiable: `round` has zero gradient almost everywhere and undefined gradient at the half-integers, so if I ever wanted a gradient to reach `w` through this, autograd would hand me zero and nothing upstream would learn. The standard fix is the straight-through identity: compute the quantized value `w_q`, but route the gradient as if the operation were the identity. Concretely, write `w_dq = w + (w_q − w).detach()`. In the forward pass this is exactly `w_q` (the `w` and `−w` cancel numerically). In the backward pass, the `.detach()` term contributes no gradient, so `∂w_dq/∂w = 1` — the gradient passes straight through to `w`. For a pure post-training conversion this is moot — there are zero training steps, so no gradient is ever taken — but writing the fake-quant this way means the conversion used in any forward pass is identical to the one used at eval, and the function is ready to drop into a training loop unchanged. The eval-time materialization, by contrast, I write under `no_grad` and return the hard `w_q` directly — no need for the straight-through wrapper when nothing will differentiate it.

Now let me set the run up to actually *be* no-training, because the method's identity is "PTQ only." The harness has an optimizer and a training loop, but I want it to do nothing: set `num_steps = 0` and `learning_rate = 0`, so the loop is a no-op and the FP weights are untouched. Then the harness's fixed post-step does the real work: it calls the no-grad quantize-dequantize once on every wrapped linear's weight, which snaps the FP weights to their per-group integer grid, and evaluates. Because I've turned training off, the wrapper's forward doesn't need to fake-quant inside the loop — by eval time the weight has already been replaced by its dequantized version, so the wrapper just calls a plain linear on the already-quantized weight. And I have to be deliberate about what I *don't* quantize: keep the activations in full precision (this is weight-only quantization — the activation fake-quant is the identity), and keep the output projection — the LM head — at full precision, because it's accuracy-critical and cheap to leave alone. The wrapper-swap walks the model replacing every `nn.Linear`, then restores the head module to a plain linear so it escapes quantization.

Let me write it as the code I'd actually drop into the harness, filling the empty slots with exactly the per-group symmetric RTN I derived.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PTQ-only baseline: round each trained weight to its per-group integer grid
# once, with no fine-tuning.

CONFIG_OVERRIDES = {
    "learning_rate": 0.0,            # no fine-tune
    "num_steps": 0,                  # training loop is a no-op
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    # symmetric signed two's-complement range: one extra negative code
    qmax = (1 << (num_bits - 1)) - 1     # +2^{B-1} - 1
    qmin = -(1 << (num_bits - 1))        # -2^{B-1}
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    # differentiable per-group symmetric RTN quantize-dequantize (forward path)
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    w = weight.float().reshape(out_features, -1, group_size)   # split columns into groups
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # per-group max|w|, div-by-0 guard
    scale = w_max / qmax                                        # S = max|w| / qmax  (no-clip cover)
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale  # round-to-nearest, clamp, dequant
    w_dq = w + (w_q - w).detach()                              # straight-through: grad passes to w
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    return x   # weight-only quantization: activations stay full precision


def quantize_dequantize_weight(weight, num_bits, group_size):
    # no-grad version, applied once after (zero-step) training to materialize the model
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
        # PTQ-only: by eval time the real QDQ has already replaced linear.weight,
        # so just run the plain linear on the already-quantized weight. (With zero
        # training steps there is nothing to fake-quant in the loop.)
        return F.linear(x, self.linear.weight, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name,
                        QATWrapper(child, num_bits=num_bits, group_size=group_size))
            else:
                _replace(child)

    _replace(model)
    # keep the output projection (LM head) at full precision
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)
    return model
```

I needed a trained net to run cheaply on integer-friendly hardware, so I needed a code-to-real correspondence; demanding that integer arithmetic approximate real arithmetic forced the map to be affine, `r = A·q + B`. Demanding that real 0 be exactly representable — to avoid a biased padding error — forced the reparameterization `r = S(q − Z)`, where the zero-exact constraint comes free. The rule for picking a code is invert-and-round: `q = round(r/S + Z)`, followed by clamping, and the dequantized value is `S(q − Z)`. Analyzing the rounding error showed round-to-nearest minimizes the per-element error and has variance `S²/12` under a uniform-fractional-part model, while stochastic rounding's pointwise unbiasedness can cost as much as `S²/4` variance and only earns its keep under *repeated* rounding, which a one-shot conversion doesn't do. Weights being zero-centered and symmetric set `Z = 0` (which also kills the weight zero-point cross-terms in the matmul), leaving a single scale; the no-tuning, no-data stance fixed that scale at `S = max|w|/qmax`, with `qmin = −2^(B−1)` and `qmax = 2^(B−1)−1`. Making the scale per-group instead of per-tensor localizes the damage from outliers and from cross-channel range spread, the two known one-shot failure modes, at the cost of one scale per 128 weights. The affine scheme is what makes the integer matmul pay off: the heavy accumulation runs in integers, and the remaining multiplier is `M = S1·S2/S3`, represented by fixed-point multiply and shift. The code fills the slots with that per-group symmetric round-to-nearest, in both a straight-through forward version and a no-grad eval version, with training switched off and the head left in full precision.
