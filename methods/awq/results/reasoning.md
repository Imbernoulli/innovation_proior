I'm trying to make a large language model fast on a single device at batch size 1 — a chatbot on one GPU. When I profile it, generation is the bottleneck, and generation is memory-bound: the arithmetic intensity is about 1 FLOP per byte, miles below the roofline, and the memory traffic is dominated by hauling the weights out of DRAM one token at a time. The FLOPs are fixed and tiny; the only knob that moves the ceiling is reading *less weight memory*. So I want the weights in 4-bit (W4A16): activations stay FP16, weights drop to 4-bit, weight traffic falls ~4×, peak performance rises ~4×. The trouble is accuracy — round-to-nearest at 3–4 bits loses a lot. And whatever I do, it has to be post-training: no backprop, no per-layer regression, and ideally not even much dependence on the calibration set, so it doesn't overfit and still works across domains.

Let me start from a question that turns out to matter: are all the weights equally important? I quantize a layer to INT3 but keep some fraction of the weight channels in FP16, and see how much accuracy I get back per channel kept. Keeping ~1% in FP16 recovers almost all the lost accuracy. So there's a small "salient" set doing the heavy lifting. The real question is which 1%.

The obvious guess is the large weights — keep the channels with the biggest weight magnitude / $L_2$-norm in FP16. I try it. It barely beats keeping a *random* 1%. So weight magnitude is not what makes a weight important. Let me try the other thing in the room: the activations. Keep the weight channels that multiply the *largest-magnitude input features* in FP16 — select by activation magnitude, not weight magnitude. That works dramatically: same 1% budget, large accuracy recovery. So saliency is set by the activation side. The reading is intuitive — an input feature with large magnitude contributes a lot to the output, so the weights that process it matter a lot; protect those weights and you protect the features that carry the signal.

So I have a working recipe: keep the activation-salient 1% of weight channels in FP16, quantize the rest. But it's a mixed-precision data type — some channels FP16, most INT3 — and that's a nightmare on hardware: irregular layout, special kernels, scattered memory access. The whole point was to be hardware-friendly and actually fast. I need to protect those salient weights *without literally storing them in a different precision*. Everything has to stay a uniform low-bit grid.

Let me look hard at where the quantization error in a single weight actually comes from, and whether I can shrink it without changing the bit-width. Take a group of weights $\mathbf w$ with $y = \mathbf w x$, quantized as $Q(\mathbf w) = \Delta\cdot\mathrm{Round}(\mathbf w/\Delta)$, $\Delta = \max(|\mathbf w|)/2^{N-1}$. The error in one element's contribution is $\mathrm{Err}(Q(w)x) = \Delta\cdot\mathrm{RoundErr}(w/\Delta)\cdot x$, where $\mathrm{RoundErr}$ is the residual of rounding to the nearest integer — uniform on $[0,0.5]$, so about $0.25$ on average, and it does not depend on the magnitude of $w$.

Now suppose I scale a single salient weight up by $s>1$ before quantizing, and scale its input down by the same $s$ to keep the product the same: I compute $Q(w\cdot s)\cdot(x/s)$. Write it out: $Q(ws)\cdot(x/s) = \Delta'\cdot\mathrm{Round}(ws/\Delta')\cdot x\cdot(1/s)$, where $\Delta'$ is the new group step after scaling. The error of this scaled-and-compensated version is

$$\mathrm{Err}\big(Q(ws)(x/s)\big) = \Delta'\cdot\mathrm{RoundErr}(ws/\Delta')\cdot x\cdot\frac{1}{s}.$$

Compare to the original error $\Delta\cdot\mathrm{RoundErr}(w/\Delta)\cdot x$. Three facts. The expected $\mathrm{RoundErr}$ is still ~0.25 either way — rounding to an integer has the same average residual regardless of the argument, so that factor is unchanged. Scaling *one* element up by a modest $s$ usually doesn't change the *group's* maximum (the group max is set by other, larger elements), so $\Delta'\approx\Delta$. And $\Delta$ and $x$ are FP16, no quantization error of their own. So the ratio of new error to old error is

$$\frac{\Delta'}{\Delta}\cdot\frac{1}{s} \approx \frac{1}{s}.$$

For $s>1$, the salient weight's error drops by a factor of $s$. I have protected it — given it effectively finer resolution — purely by scaling, with no FP16 storage, no change in bit-width. That's the escape from the mixed-precision data type.

I sanity-check it: take OPT-6.7B, scale the 1% salient channels by $s$, watch the perplexity. At $s=1$ (plain RTN) it's bad, ~23.5. At $s=2$ it drops to ~11.9 — a huge improvement, and I confirm that for $s<2$ the fraction of groups whose $\Delta$ actually changed is small (<5%), so the $\Delta'\approx\Delta$ assumption holds and the $1/s$ error reduction is real.

But the best perplexity is at $s=2$, not at the largest $s$ I try. Why doesn't more scaling keep helping? Because the assumption $\Delta'\approx\Delta$ breaks when $s$ gets large. Push $s$ high enough and the scaled salient weight starts to *become* the group maximum, so $\Delta'$ grows. And the error of every *non-salient* weight in that group is proportional to $\Delta$ — so when $\Delta'/\Delta$ rises above 1, all the non-salient weights' errors get *amplified* by that ratio (at $s=4$, more than 20% of channels see $\Delta$ grow). I've been writing the salient error as $(\Delta'/\Delta)(1/s)$, which keeps shrinking, but I forgot the other side: the non-salient errors scale as $\Delta'/\Delta$, which keeps growing. Protecting the salient weights by overscaling quietly damages everyone else. So there's a real trade-off, and I can't just crank $s$ — I have to choose it accounting for *both* the salient and the non-salient channels.

So I shouldn't pick $s$ to minimize one weight's error in isolation; I should pick the per-channel scale vector $\mathbf s$ to minimize the *whole layer's* output error after quantization. The honest objective is

$$\mathbf s^\* = \arg\min_{\mathbf s}\ \big\lVert Q(\mathbf W\,\mathrm{diag}(\mathbf s))\,(\mathrm{diag}(\mathbf s)^{-1}\mathbf X) - \mathbf W\mathbf X\big\rVert,$$

where $Q$ is the group-wise INT3/INT4 quantizer, $\mathbf X$ is calibration activations, and $\mathrm{diag}(\mathbf s)^{-1}\mathbf X$ can be folded into the previous layer just like any equivalence-transform scaling. The problem is $Q$ has a $\mathrm{Round}$ in it and isn't differentiable. I could reach for straight-through or learned-step approximate gradients — but those are exactly the unstable, backprop-dependent things I'm trying to avoid, and I find they converge badly here. I don't want to optimize $\mathbf s$ freely in $\mathbb R^{C_i}$ anyway; that's a huge, ill-conditioned, non-differentiable search.

I already know the one thing that determines saliency: the activation magnitude per channel. So I don't need a free search — I need a one-parameter family that dials "how much do I scale the high-activation channels," and I can grid-search that single knob cheaply. Let $\mathbf s_X$ be the average activation magnitude per input channel (from a tiny calibration pass — average, not max, because I only need the typical importance of a channel and averaging avoids overfitting to a few calibration samples). Define the search space

$$\mathbf s = \mathbf s_X^{\alpha},\qquad \alpha^\* = \arg\min_{\alpha\in[0,1]}\ \mathcal L(\mathbf s_X^{\alpha}).$$

At $\alpha=0$, $\mathbf s = \mathbf 1$ — no scaling, plain RTN. At $\alpha=1$, the scaling tracks the activation magnitude most aggressively — maximum protection of salient channels (and maximum risk of inflating $\Delta$ for the rest). $\alpha$ slides between "protect nobody" and "protect the salient channels hard," which is exactly the trade-off I diagnosed. Because it's one scalar, I just sweep $\alpha$ over a fine grid in $[0,1]$, and for each candidate I quantize the layer with that $\mathbf s$, run the calibration input through, and measure the actual output MSE against the FP16 output — the true objective, evaluated directly, no gradient needed. Take the $\alpha$ with the lowest loss. I also fold a small weight-clipping step in (clip the group range to minimize the quantization MSE), which trims the worst rounding outliers. To keep the scales numerically tame I normalize $\mathbf s$ by the geometric mean of its max and min before applying it, so it neither blows up the weights nor collapses them.

That's the method, and notice what it is *not*: no backprop, no second-order Hessian, no error-feedback regression over the calibration set. I only ever measure a per-channel *average* activation magnitude and grid-search one scalar by directly evaluating output error — so I barely depend on the calibration set and don't overfit to its distribution, which means the quantized model keeps the LLM's knowledge outside the calibration distribution and generalizes across domains and modalities. And since the output is a uniform group-wise 4-bit weight tensor (just pre-scaled), it packs into a regular layout that a fast on-device kernel can dequantize and multiply.

The causal chain: on-device generation is memory-bound, so I want W4A16; RTN at 3–4 bits is too lossy because weights aren't equally important; the salient ~1% is identified by *activation* magnitude, not weight magnitude; keeping them in FP16 works but is a hardware-hostile mixed-precision type; scaling a salient weight by $s>1$ (and its input by $1/s$) cuts its quantization error by $\sim1/s$ at no extra bits — but overscaling inflates the group step $\Delta$ and amplifies the non-salient errors, so the scale must balance both; that balance is captured by a single knob over the activation-derived scale $\mathbf s = \mathbf s_X^{\alpha}$, grid-searched against the layer's real output MSE, giving an activation-aware, training-free, hardware-friendly 4-bit weight quantizer.

```python
import torch

@torch.no_grad()
def get_act_scale(x):
    # per-input-channel AVERAGE magnitude (average, not max -> robust, low overfit)
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def search_scale(block, fcs, inp, w_bit=4, group_size=128, n_grid=20):
    org_out = block(inp)                                  # FP16 reference
    x_max = get_act_scale(inp)                            # s_X, per input channel
    org_w = [fc.weight.data.clone() for fc in fcs]

    best_loss, best_scales = float("inf"), None
    for i in range(n_grid):
        ratio = i / n_grid                                # alpha in [0,1)
        scales = x_max.pow(ratio).clamp(min=1e-4)
        # normalize by geometric mean of extremes so weights neither explode nor vanish
        scales = scales / (scales.max() * scales.min()).sqrt()

        for fc, w0 in zip(fcs, org_w):
            fc.weight.data = w0 * scales.view(1, -1)      # W <- W diag(s)
            fc.weight.data = pseudo_quantize(fc.weight.data, w_bit, group_size)
            fc.weight.data = fc.weight.data / scales.view(1, -1)  # Q(W diag(s)) diag(s)^-1, equivalent to scaling X by 1/s

        loss = (org_out - block(inp)).float().pow(2).mean().item()  # true output MSE
        for fc, w0 in zip(fcs, org_w):
            fc.weight.data = w0
        if loss < best_loss:
            best_loss, best_scales = loss, scales

    return best_scales                                    # s = s_X^{alpha*}, then quantize for real
```
