The vanilla floor came in healthy — validation loss 2.2763, WikiText-2 44.28, LAMBADA 70.09, ARC-Easy 54.12, HellaSwag 33.82 — so the failure I am attacking is not a failure of *training* but of *use*. The Pre-LN stream is a fixed unit-weight accumulator, and the floor's left-on-the-table capacity has a concrete mechanism. Tracing the recursion $x_{l+1} = x_l + F(\mathrm{LN}(x_l))$, the normalization fixes the scale of the *branch input* but does nothing to the *stream* I keep adding onto, so the stream variance accumulates with depth — $\sigma^2_{x_l} \propto \prod_{k<l}(1 + 1/\sigma_{x_k})$, every layer a factor above one, linear growth in the benign case and worse in the bad one. The block Jacobian is $\partial(x + F(\mathrm{LN}(x)))/\partial x = I + (\partial F/\partial \mathrm{LN})\cdot(\partial \mathrm{LN}/\partial x)$, and the second term carries a $1/\sigma_{x_l}$ from the normalization dividing by the stream's own growing scale; when the stream variance is large that term is tiny, the block Jacobian collapses toward $I$, and a block whose Jacobian is $\approx I$ is a local identity map that barely transforms its input. So the deepest layers are pushed toward doing nothing. The question is sharp: keep those deep layers alive without touching attention, the MLP, or the norm — only the residual flow.

The standard counters all share one defect, and naming it precisely is what points to the fix. Static down-weighting (divide the branch at layer $l$ by $\sqrt{l}$) flattens the variance curve, but the factor depends only on depth and is frozen for the whole run — helpful early when it controls the explosion, but late in training, when everything has stabilized and I *want* the deep layers learning at full strength, it is still throttling them, trading "dead because the variance exploded" for "dead because I clamped them." Bounded-update designs (Fixup, T-Fixup, DeepNorm) constrain the model update with constants derived at initialization and held uniformly for all of training — the right protection for the chaotic warm-up, over-conservative once I am in the long stable phase. And the scalar-on-the-branch family (ReZero, SkipInit), $x_{l+1} = x_l + \alpha_l F(\mathrm{LN}(x_l))$ with every $\alpha_l$ initialized to 0, gives a lovely identity start but then learns the $\alpha_l$ independently by gradient descent — and nothing says the shallow branches turn on before the deep ones. The optimizer is free to grow a deep $\alpha$ first, and a deep branch with freshly-random weights firing early does two bad things at once: it injects noise that all the upper layers must consume, and it corrupts the gradient the shallow layers below are using to find their footing. The common failure is that every one of these levers sets up the *first step* and then either freezes or hands the wheel to the optimizer — none is *training-phase-aware*. But I have two trajectory facts: training is staged (chaotic warm-up, then stable, then decay), and layers converge unevenly (shallow settle earlier than deep). The real question is not only *how much* each branch contributes but *when*, and in *what order across depth*.

I propose ProRes, progressive residual warmup: multiply each layer's residual *branch* by a **predefined, non-learnable** scalar $\alpha(l,t)$ that starts at 0 — exact identity at init — and ramps to 1 over training, with deeper layers warming up more slowly, so the branches switch on in a shallow-to-deep wave. The residual write is

$$x_{l+1} = x_l + \alpha(l,t)\, F(\mathrm{LN}(x_l)),$$

and the schedule is a clipped linear ramp,

$$\alpha(l,t) = \min\!\left(\frac{t}{T\cdot l},\, 1\right), \qquad T = 1000,$$

where $T$ is the first layer's warm-up length, so layer $l$ reaches full strength at step $T\cdot l$ and the deepest layer at $T\cdot L$. Each design choice falls out rather than being asserted. *Identity at init*: $\alpha(l,0) = \min(0,1) = 0$, so $x_{l+1} = x_l$ exactly — ReZero's clean start, made exact and parameter-free, no init variance blowup. *Bounded update across depth and time*: early on only shallow layers have nonzero $\alpha$, so only a few layers can move the function and they move it by a fraction $\alpha < 1$; the constraint relaxes itself layer by layer, tight in the chaotic warm-up and fully released in the stable phase — the temporal awareness the constant init-time bounds and the static $1/\sqrt{l}$ both lack. *Shallow-before-deep ordering*: a deep branch's $\alpha$ stays near 0 while the shallow layers below do their large early updates, so by the time it climbs enough to matter the foundation beneath it has largely settled, and it is not injecting garbage back through the gradient path. And crucially the constraint is real early and *gone* late — as $t \to \infty$ every $\alpha \to 1$ and the block runs *exactly* vanilla, so I lose nothing in the limit, which is precisely the property the static methods cannot have.

The specific shape is load-bearing, not arbitrary. If every layer warmed up together ($\min(t/T,1)$, no $l$-dependence) I would delay the chaotic init updates but ignore the order — all deep branches switch on simultaneously, just postponed, so deep layers still fire while shallow representations are unstable. If I went the *wrong* way, deep-first ($\tau_l = T(L-l+1)$), I would actively let the deep randomly-initialized branches dominate early while the shallow layers starve — exactly the divergence Post-LN is notorious for. So the order, shallow before deep, is doing real work, and the linear-in-$l$ stagger is the simplest schedule that encodes it. The one free number $T$ is bracketed by two failure modes: too small and I barely delay anything past the chaotic init; too large and $T\cdot L$ eats the budget so the deep layers spend most of training artificially weakened. At 13,535 steps over 24 layers, $T = 1000$ is the natural default — the first layer finishes in 1000 steps, the shallow half within a few thousand, and the wave sweeps most of the stack before the cosine decay sets in — set once and not tuned. I scale only the branch and leave the skip at weight 1, because the skip is the identity highway and the gradient route: scaling it would attenuate the signal and at $\alpha = 0$ collapse the network to zero output instead of to the identity. Keeping the skip at 1 and dialing the branch from 0 to 1 makes $\alpha = 0$ *exactly* identity and $\alpha = 1$ *exactly* vanilla, a clean interpolation — and it costs zero learnable parameters, so the optimizer groups and `CONFIG_OVERRIDES` stay default.

There are two implementation wrinkles in this edit surface. The scalar $\alpha(l,t)$ depends on the global step $t$, which the `Block` does not see, and the contract wants `Block.forward(x) → x` unchanged — so I keep the block vanilla and put the schedule in the `GPT.forward` block loop, carrying the step as a non-trainable `_prores_step` buffer. And the block already folds the skip in and returns `block_out = x + delta`, the full Pre-LN residual, so rather than reach inside it I recover the branch as `delta = block_out − x` and write `x ← x + alpha·(block_out − x)`, which scales exactly the branch and leaves the skip at 1; once a layer's $\alpha$ has reached 1 I take `block_out` directly, bit-identical to vanilla. I read the step *before* the loop so the first training forward is truly $t = 0$ with every branch off, then advance the buffer after. Against the 2.2763 floor I expect a *modest* drop into the low-2.27s — this is a conditioning fix, not new capacity — clearest on the perplexities, with LAMBADA (the long-range task leaning hardest on the deep layers) the more sensitive of the two.

```python
# EDITABLE regions of custom_pretrain.py — step 2: ProRes (progressive residual warmup)
# (each region shown inside its enclosing method exactly as spliced into the file)

# Block: unchanged — vanilla Pre-LN residual.

class GPT(nn.Module):
    def _init_prores(self, config):  # GPT.__init__ residual region:
        # ── ProRes: progressive residual warmup ──
        # T controls the warmup period; deeper layers take T*layer_idx steps
        # to reach full contribution.  step counter is a non-parameter buffer.
        self.prores_T = 1000
        self.register_buffer('_prores_step', torch.zeros(1, dtype=torch.long))

    def _forward_block_loop(self, x):  # GPT.forward block loop:
        # ── ProRes: progressive residual warmup per block ──
        # Increment step counter once per forward (training only).
        if self.training:
            self._prores_step += 1
        step = self._prores_step.item()
        T = self.prores_T
        for i, block in enumerate(self.transformer.h):
            block_out = block(x)
            if self.training and step < T * (i + 1):
                # alpha ramps from 0 to 1 over T * layer_idx steps
                layer_idx = i + 1
                alpha = min(step / (T * layer_idx), 1.0)
                # block_out = x + delta (Pre-LN residual), so delta = block_out - x
                x = x + alpha * (block_out - x)
            else:
                x = block_out
        return x

# GPT.configure_optimizers: unchanged (no new learnable params).
# CONFIG_OVERRIDES = {}   (no override).
```
