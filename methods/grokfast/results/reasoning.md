Let me start from the thing that actually bothers me, which is the training curve. I train this little two-layer Transformer on `a · b (mod 97)` and within maybe a thousand steps the training accuracy is pinned at 100%. The network has memorized the table. And then for tens of thousands, sometimes hundreds of thousands of steps, the validation accuracy just sits near chance. And then — late, much later — it climbs, and the thing generalizes, suddenly. The delay between "overfit" and "generalize" is one, two, sometimes three orders of magnitude. As a phenomenon it's fascinating. As something I have to wait for, it's intolerable. I don't want to explain why grokking happens; I want to make the late part happen sooner. The dataset is fixed, the loss is fixed, the split is fixed, the architecture is what it is. The only thing I really get to touch is the optimizer, and even there I'd love not to rip it out — people are using AdamW with weight decay for a reason here, weight decay is known to shorten this delay, and I don't want to throw that away. So whatever I do has to be a small thing bolted onto a first-order optimizer that already works.

Stare at the curve a bit longer. The training loss falls off a cliff early; the validation loss barely moves for ages and then drops. Two things are happening on wildly different timescales in the *same* run. And both of those loss curves are just downstream of one underlying process: the parameters moving. Every step, the optimizer takes the gradient and changes `θ`. The fast collapse of the training loss is the parameters doing one kind of thing; the slow eventual drop of the validation loss is the parameters doing another kind of thing. But it's the *same* parameter trajectory producing both. So the parameter motion must contain both a fast-changing part and a slow-changing part, mixed together, and the fast part is what overfits while the slow part is what eventually generalizes. That's a hypothesis, but it's a sharp one: if I could separate the slow part of the parameter motion from the fast part, the slow part is the friend I want to encourage.

How do I make "fast part" and "slow part" precise? The parameter update each step is `u(t) = θ(t+1) − θ(t)`. If I just look at one scalar parameter, then `u(t)` is a sequence indexed by training step `t` — a sequence of little nudges. A sequence indexed by time is a *signal*. And the most natural way to split a signal into "slow-varying" and "fast-varying" is the language I already have for exactly that: frequency. Take the discrete-time Fourier transform of the update sequence, `U(ω) = Σ_t u(t) e^{-iωt}`. Then "the slow-varying component of the parameter motion" is literally the low-frequency part of `U(ω)`, and "the fast-varying component" is the high-frequency part. My hypothesis, restated, is that the low-frequency content of `U(ω)` is what drives generalization and the high-frequency content is what drives overfitting. If that's right, then accelerating generalization means *boosting the low-frequency content* of the parameter motion.

Now I have to be a little careful, because I don't act on `u(t)` directly — the optimizer produces `u(t)` from the gradient `g(t)`. But for a first-order optimizer the update is built linearly out of the gradient stream (plain SGD: `u = −η g`; momentum: a running average of `g`; Adam: an EMA of `g` rescaled). The gradient signal `g(t)` and the update signal `u(t)` are tied together. So instead of trying to reshape `u`, I can reshape `g` — boost the low frequencies of the *gradient* signal `g(t)` and let that propagate into `u`. I'll have to come back and check that "boost low frequencies of `g`" really does mean "boost low frequencies of `u`," because if the optimizer does something frequency-dependent in between, the equivalence could break. Hold that thought. For now the working plan is: amplify the low-frequency component of the per-parameter gradient sequence.

Amplifying low frequencies of a signal is a standard operation. I want a transform that multiplies `G(ω)` by something larger near `ω = 0` than near `ω = π`. The cleanest way to get an *amplifier* (rather than a filter that throws information away) is to take a low-pass-filtered copy of the signal and *add it back* to the original. If `h(t)` is a low-pass filter and `*` is convolution, then

  `ĝ(t) = g(t) + h(t) * g(t)`.

In the frequency domain convolution is multiplication, so

  `Ĝ(ω) = G(ω) + H(ω) G(ω) = (1 + H(ω)) G(ω)`,

with `H(ω) = F{h}(t)` the filter's transfer function. The effective gain applied to the gradient at frequency `ω` is `1 + H(ω)`. If `H` is low-pass — big at low `ω`, small at high `ω` — then `1 + H` is `≈ 1 + (something big)` at low frequencies and `≈ 1 + (something tiny) ≈ 1` at high frequencies. So the slow component gets multiplied up while the fast component is left essentially untouched. That's exactly the asymmetry I wanted, and it's important that the high frequencies are *kept*, not killed — I'm not trying to denoise the gradient, I'm trying to *emphasize* its slow part relative to its fast part. The whole job is now reduced to: pick a low-pass filter `h`, and a way to dial how hard I amplify.

Before I commit, the obvious alternative — why not just *replace* `g` with its low-pass version, `ĝ = h * g`, i.e. throw the fast part away and train only on the slow gradient? That's tempting and it's simpler. But think about what the slow gradient *is*: a moving average over a window of recent gradients is, near enough, the gradient you'd get from a bigger, overlapping minibatch. Training purely on that is just training with a giant smoothed gradient and no stochastic, high-frequency content at all. My hypothesis says the fast part *causes overfitting*, sure, but the network still needs to actually fit the data — the rapid early descent is the fast part doing useful work. If I delete it I'd expect training to get slow and shaky, because I've removed the very signal that drives the quick descent and kept only a sluggish averaged one. So replacing is wrong; *adding* is right. I want both components present, with the slow one turned up. (I'd want to confirm this empirically — kill the fast part and watch it go slow and unstable — but the reasoning already tells me which way it should fail.)

Now design `h`. Start with the most transparent low-pass filter there is: a windowed moving average. Keep the last `w` gradients in a queue, average them, and that average is the slow gradient. As an impulse response,

  `h(t) = (λ / w)` for `0 ≤ t < w`, and `0` otherwise

— a rectangular window of width `w`, scaled by a gain `λ` and normalized by `w` so it's a genuine average. Two knobs, both meaning something concrete: `λ` is how strongly I amplify the slow component, and `w` is the window length, which sets *how slow* "slow" has to be — a long window only averages in genuinely persistent, low-frequency structure, so `w` is effectively the cutoff. The update rule per parameter is then

  `ĝ_t = g_t + λ · mean(Q)`,    `Q` = queue of the last `w` gradients,

and I hand `ĝ_t` to whatever optimizer I'm already using. Call this the moving-average version. It's almost nothing to implement: maintain a fixed-capacity queue of gradients for every parameter, append the new gradient, add `λ` times the queue mean back into the gradient, step. A grid over `λ ∈ {1,2,5,10}` and `w ∈ {2,5,…,200}` should find a sweet spot — and there *should* be a sweet spot, because if `w` is too small the "slow" gradient isn't actually slow (the window doesn't reach down to the low frequencies that matter) and if `w` is huge the average is so inert it stops tracking anything; somewhere in between is the cutoff that lines up with the generalizing component.

So far so good, and on the modular task this kind of low-frequency boost does cut the delay substantially. But there's a problem I can't ignore if I want this to be usable: memory. The moving-average filter literally stores `w` copies of the gradient *for every parameter*. At `w = 100` that's a hundred extra full-model-sized tensors. On the toy Transformer it's annoying; on anything real it's a non-starter, and copying all those tensors each step also slows the iteration down measurably. The moving average is a finite-impulse-response filter, and FIR filters pay for their memory in stored taps. I need a low-pass filter with the same low-frequency-boost behavior but a tiny, fixed memory footprint.

The way you get a low-pass filter with `O(1)` state is to make it recursive — an infinite-impulse-response filter that carries a single running summary instead of a window. The canonical one-pole IIR low-pass is the exponential moving average: keep one buffer `μ` and update it as

  `μ(t) = α μ(t−1) + (1 − α) g(t)`.

This is exactly the same recursion that lives inside momentum and inside Adam, which is reassuring — it's a known, stable smoother. Its impulse response is geometric: a unit impulse at the input at time 0 produces `μ(t) = (1−α) α^t` for `t ≥ 0`. Folding in my amplification gain `λ`, the equivalent filter is

  `h(t) = λ (1 − α)(1) α^t`,    `t ≥ 0`,

i.e. `h(t) = λ(1−α) Σ_{τ≥0} α^τ δ(t − τ)`. One buffer per parameter, the size of the model — done with the memory blowup. The window length of the moving average is replaced by `α`: a decay close to 1 has effective memory `≈ 1/(1−α)` steps, so `α` plays the role `w` did, it sets the cutoff, while `λ` is still the gain.

I should check that this recursion really is low-pass, and by how much, because the whole premise is "boost low, leave high alone," and I want the numbers. Transfer function — DTFT of the geometric impulse response:

  `H(ω) = Σ_{t≥0} λ(1−α) α^t e^{-iωt} = λ(1−α) Σ_{t≥0} (α e^{-iω})^t = λ(1−α) / (1 − α e^{-iω})`,

since `|α e^{-iω}| = α < 1` so the geometric series converges. Now evaluate the amplifier `1 + H(ω)` at the two ends. At DC, `ω = 0`:

  `H(0) = λ(1−α) / (1 − α) = λ`,

so the low-frequency gain is `1 + H(0) = 1 + λ`. The slowest component gets multiplied by `1 + λ` — directly, that's what `λ` buys me. At the Nyquist frequency, `ω = π`, `e^{-iπ} = −1`:

  `H(π) = λ(1−α) / (1 + α)`,

which for `α` near 1 is tiny — at `α = 0.98` it's `λ · 0.02 / 1.98 ≈ 0.01 λ`. So the fastest component is essentially passed through with gain `≈ 1`. Low frequencies up by `(1+λ)`, high frequencies barely touched, and `α` controls where the rolloff between the two sits. This is precisely the high-boost-of-the-slow-part shape I wanted, now with constant memory. The per-step rule is

  `μ ← α μ + (1 − α) g`,    `ĝ = g + λ μ`,

hand `ĝ` to the optimizer. Compared to the moving-average version this is `w`× smaller in memory — one buffer instead of a hundred — which is the whole reason to prefer it. I'll default the gain modest and the decay long: `λ = 2`, `α = 0.98`, with sensible ranges `λ ∈ [0.1, 5]`, `α ∈ [0.8, 0.99]`. `λ` shouldn't be enormous — I'm *emphasizing* the slow part, not drowning the step in it — and `α` wants to be high enough that the EMA reaches down to genuinely slow structure but not so high it never moves.

Now I have to pay off the debt I left earlier. I designed everything as a filter on the *gradient* `g(t)`, but my hypothesis was about the *parameter update* `u(t)` — the slow part of the *motion* is what generalizes. I waved my hand that they're linearly related, but the optimizer sits in between, and if the optimizer is itself frequency-dependent then a low-pass filter on `g` might not produce a low-pass-shaped change in `u` at all. I need to actually verify that filtering the input gradient by `h` has the same spectral effect as filtering the output update by the same `h`. Let me write a generic first-order optimizer as a linear time-invariant system with a scalar state `x` (the momentum buffer, say):

  `x(t) = A x(t−1) + B g(t)`,    `u(t) = C x(t) + D g(t)`,

with scalar coefficients `A, B, C, D`. This covers the optimizers I care about. Plain SGD-with-momentum keeps `m(t) = μ m(t−1) + (1−τ) g(t)` and steps `u(t) = −η m(t)`, so `A = μ`, `B = 1−τ`, `C = −η`, `D = 0`. Nesterov's look-ahead steps `u(t) = −η(g(t) + μ m(t))`, so `A = μ`, `B = 1−τ`, `C = −ημ`, `D = −η`. For stability the state can't blow up, so `0 < A < 1`.

Move to the frequency domain. The state recursion `x(t) = A x(t−1) + B g(t)` becomes, using the shift property `F{x(t−1)} = e^{-iω} X(ω)`,

  `X(ω) = A e^{-iω} X(ω) + B G(ω)`  ⟹  `X(ω) = B G(ω) / (1 − A e^{-iω})`,

and the output relation gives the input-to-output transfer function of the optimizer:

  `H_io(ω) = U(ω) / G(ω) = C · X(ω)/G(ω) + D = BC / (1 − A e^{-iω}) + D`.

Now run the filtered gradient `ĝ = g + h * g`, i.e. `Ĝ(ω) = (1 + H(ω)) G(ω)`, through the *same* optimizer. The optimizer's coefficients `A, B, C, D` haven't changed — I only changed its input — so it has the *same* transfer function: `Û(ω) / Ĝ(ω) = H_io(ω)`. Therefore

  `Û(ω) / U(ω) = [H_io(ω) Ĝ(ω)] / [H_io(ω) G(ω)] = Ĝ(ω) / G(ω) = 1 + H(ω)`.

That's the punchline. The ratio of filtered-to-unfiltered *update* spectra equals `1 + H(ω)`, which is exactly the ratio of filtered-to-unfiltered *gradient* spectra. In other words, if I define the equivalent post-optimizer filter `ĥ` by `û = u + ĥ * u`, then `1 + Ĥ(ω) = 1 + H(ω)`, so `ĥ = h`. Filtering the gradient by `h` is identical to filtering the update by `h`, for any linear optimizer. The `H_io` term — whatever the optimizer does, momentum, Nesterov, the lot — cancels top and bottom because it's linear and unchanged. So my hypothesis about the slow component of the *motion* is faithfully served by acting on the *gradient*. (The cancellation needs `H_io` to be the same on both runs, which is just "the optimizer is linear and I didn't change its coefficients," and it needs the filter not to be degenerate — as long as `1 + H` genuinely depends on `H` there's a real correspondence.)

This is more than a sanity check; it tells me *where* to put the code. I could have implemented the slow-component boost the "honest" way — intercept the optimizer's update `u` and add a filtered copy of it. But that means writing a custom optimizer object that exposes its updates, which is fiddly and has to be redone per optimizer. The equivalence says I can instead act on the gradients, which in any autograd framework are sitting right there in `p.grad` after `backward()` and before `step()`, and get *exactly the same effect* on the parameter motion regardless of which optimizer consumes them. So the implementation is: between `loss.backward()` and `optimizer.step()`, update the per-parameter EMA buffer and add `λ` times it back into `p.grad`. A few lines, optimizer-agnostic, no new optimizer class. The theorem is what licenses the cheap implementation.

Let me also be precise about how this differs from just turning up momentum, because the recursion `μ ← αμ + (1−α)g` looks like a momentum buffer and I don't want to be reinventing momentum. The difference is structural. Ordinary momentum *consumes* the smoothed gradient *as* the update — the average becomes the step. Here I add the smoothed gradient as a *residual* on top of the raw gradient, `ĝ = g + λμ`, and only *then* feed it to the optimizer, which still does its own thing. Formally it's closer to Nesterov, which also mixes the current gradient with the momentum buffer before stepping — but Nesterov-style mixing (as in NAdam) happens *inside* the optimizer, whereas here the mixing happens *before* the optimizer, on the gradient, independent of it. And because of the equivalence I just proved, "before the optimizer, on the gradient" is legitimate: it imposes the low-frequency boost on the update regardless of what the optimizer is. That independence is the point — the same hook works with SGD, with Adam, with AdamW, without touching any of them.

So the final form is the EMA gradient filter. For each parameter, maintain a running EMA `μ` of its gradient initialized to the first gradient seen; every step, decay it by `α`, mix in `(1−α)` of the current gradient, and add `λμ` back into the current gradient before the optimizer reads it. The moving-average version is the conceptual stepping stone — same idea, rectangular instead of geometric window — that I keep around as the easy-to-explain sibling, but the EMA is what I actually use because it's the same low-pass boost at `1/w` of the memory. Let me write both.

```python
from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


# Stepping-stone: windowed moving-average (FIR) low-pass boost.
# Stores the last `window_size` gradients per parameter (w x memory).
def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
) -> Dict[str, deque]:
    if grads is None:
        # one fixed-capacity queue per parameter
        grads = {n: deque(maxlen=window_size)
                 for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())        # push this step's gradient

            # add lambda * (slow gradient) back onto the raw gradient: g_hat = g + lambda*mean(Q)
            if not warmup or len(grads[n]) == window_size:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


# Final form: one-pole exponential-moving-average (IIR) low-pass boost.
# One buffer per parameter (1 x memory). alpha = cutoff, lamb = low-freq gain.
def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        # initialize each EMA buffer to the first gradient seen
        grads = {n: p.grad.data.detach()
                 for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            # EMA of the gradient:  mu <- alpha*mu + (1-alpha)*g     (low-pass on g)
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            # high-boost the slow part:  g_hat = g + lambda*mu  ->  spectral gain 1 + H(w)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads


# Plugs into the existing loop with two added lines: keep a running `grads`,
# and call the filter after backward() and before the optimizer reads p.grad.
grads = None
for batch in dataloader:
    model.zero_grad()
    loss = criterion(model(batch))
    loss.backward()                                  # fills p.grad for every parameter
    grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)   # boost slow gradients
    optimizer.step()                                 # any first-order optimizer, untouched
```

Pulling the chain together: the training and validation losses move on two timescales because the parameter motion mixes a fast component (overfitting) and a slow component (generalization); reading the per-parameter update sequence as a time signal lets me name those as high- and low-frequency content, so accelerating generalization means amplifying the low frequencies of the motion; I amplify by adding a low-pass-filtered copy of the gradient back onto itself, which gives spectral gain `1 + H(ω)` — big at DC, ≈1 at Nyquist; a windowed moving average does this but costs `w`× memory, so I switch to a one-pole EMA, whose transfer function `λ(1−α)/(1−αe^{-iω})` gives low-frequency gain `1+λ` and near-unit high-frequency gain with a single buffer and `α` as the cutoff; and because for any linear optimizer filtering the gradient is provably equivalent to filtering the update (the optimizer's transfer function cancels), I can implement the whole thing as a two-line hook on `p.grad` that works with whatever optimizer is already in place.
