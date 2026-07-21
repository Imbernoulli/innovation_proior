The schedule levers — the dynamic effective-batch controller and the sequence-length growth — have squeezed
the run well down from the ~3.5 minutes the batch controller reached. Two parts of the *network itself* are
still at textbook defaults: the MLP nonlinearity and the precision policy. Both are worth re-examining now
rather than earlier, because on the slow baseline the schedule savings were so large that a per-step-quality
change would have been partly masked by them — the binding constraint was scheduling. Now that the batch
controller and the length schedule are spent, the network's per-step quality *is* the binding constraint. One
change makes each step *learn* more (fewer steps to the bar), one makes each step *cost* less.

Start with the MLP. The baseline is expand 4×, GELU pointwise, project back. GELU is a *fixed* gate: the same
curve applied elementwise, the only learned thing being the linear map feeding it — a GELU hidden unit computes
GELU(w·x), a fixed scalar function of one learned linear feature. What if the gate itself were *learned* — if,
instead of one projection through a fixed nonlinearity, the layer produced two projections and used one to gate
the other? Split the widened activation into halves a and b and output a ⊙ σ(b): now each hidden unit is a
*product* of two learned linear features, one through a sigmoid-shaped gate. That product is the thing a plain
MLP can't form in a single layer — a multiplicative, AND-like interaction ("pass feature a only where feature b
is active") that a fixed pointwise curve only approximates through extra depth, so a plain MLP needs a second
layer to build it. SiGLU forms it at depth-1 cost. With a SiLU gate (x·sigmoid(x), smooth like GELU) this is
the SiLU-gated linear unit. That "more per parameter" lands exactly where I need it: this is a step-limited run
— a few tokens per parameter, so I'm never short on capacity, I'm short on *steps* — and a layer that extracts
more structure per gradient update spends the scarce currency better.

Is gating the right axis? Three ways to make an MLP do more: make it *bigger* (8× plain GELU) — scales
parameters and FLOPs for capacity I'm not short on, so brute width buys the wrong thing at the highest cost;
swap the *activation* (squared-ReLU) — cheaper per element, sometimes a little better, but still a fixed
pointwise curve with no input-dependent interaction; or make the gate *learned*, which is SiGLU. Only the third
changes the *kind* of function the layer can represent rather than just its size or curve. And a SiLU gate
rather than a hard sigmoid keeps the block of a piece with the smooth GELU/LayerNorm net around it, no kink in
the landscape.

I have to keep the budget honest or I've handed the gain back. The plain 4× GELU MLP is expand d→4d (4d²) plus
project 4d→d (4d²), so 8d² parameters, 16d² FLOPs per token. A SiGLU needs *two* projections into the hidden —
value and gate — so to *match* 8d² the gated hidden should be (2/3)·4d = 8d/3 ≈ 2.67d. I round up to a clean
3×: expand to 2·3d = 6d (two stacked 3d halves), gate down to a 3d hidden, project 3d→d, giving 6d² + 3d² = 9d²
— 12.5% more than 8d². At the whole-model level that's smaller than it sounds: the MLP is ~24% of the forward
FLOPs, so +12.5% on it is ~+3% per step overall. So the bet is sharp: SiGLU must cut the step count by more
than ~3% for a net win, and the representational argument clears that comfortably. And the trade is better than
+3% looks, because of where per-step cost lives: the vocab head alone is ~62% of the forward FLOPs, a fixed
cost paid once per step whatever the MLP does. SiGLU's payoff is *fewer steps*, and fewer steps means fewer
passes through that dominant fixed head — so cutting steps divides down the whole expensive fixed part of each
step. A change that attacks step count is multiplied by the fact that the biggest per-step cost is amortized
over exactly those steps. (SiGLU unit and rebuilt block in the answer; expand→2·3·width produces both halves
from one matmul, split on the last dim.)

One property of this arrangement worth noting: the *value* half is passed through no nonlinearity — the output
is (linear value) ⊙ SiLU(gate), an ungated linear path through the block. That's good for gradient flow:
∂out/∂value = σ(gate) ∈ (0,1), which doesn't vanish the way the gradient through a saturated nonlinearity
would, so the value channels keep receiving clean gradient even when the gate is nearly closed. The
nonlinearity is spent on *deciding how much of each value to pass*, not on mangling the value itself. And the
gating machinery is nearly free: the split returns two views into the single expand output, no copy, so the
only added arithmetic over a plain MLP is one SiLU and one elementwise multiply over the 3d hidden — negligible
next to the 6d² + 3d² of matmul, and the 12.5% premium is all in the linear layers, where I want the cost, on
the tensor cores rather than in bandwidth-bound pointwise ops.

One consistency check with the baseline init, because `project` is a residual projection I scaled by
1/√(2·num_blocks) to bound the residual-stream variance. Does that still hold when the sublayer output is a
gated product? At init the gate half b is near zero, so σ(b) ≈ 0.5 and the SiGLU output a·σ(b) is roughly the
value half scaled by ~0.5 — a *smaller* effective magnitude than a plain GELU MLP's activation, not larger. So
the same 1/√(2·num_blocks) scaling is, if anything, slightly conservative for SiGLU: the per-block contribution
starts a touch gentler and the depth-stability argument holds without re-tuning the init. The new MLP doesn't
quietly reintroduce the exploding-residual-stream problem I paid that init factor to prevent.

Now precision. The baseline runs *mixed* precision via autocast: the heavy matmuls happen in bf16 but the
tensors live in fp32 and get cast on the fly, keeping a high-precision master copy. Autocast exists in the
first place for fp16, whose 5 exponent bits and ~65504 ceiling force an fp32 master and loss-scaling to keep
small gradients from underflowing. But as I established in the baseline, bf16 carries fp32's full exponent
range and gives up only mantissa — it has no *range* problem. So the fp32 master autocast maintains is a
safeguard designed for a precision I'm not using. Which makes me ask: on this hardware, is that master buying
me anything, or is it paying memory and cast overhead for a margin bf16's range already provides?

Naming the options so I'm choosing rather than drifting: keep autocast (the status quo, paying cast overhead
and an fp32 footprint for an fp16-shaped safeguard); go to fp8, which would speed the *arithmetic* but has only
2–3 mantissa bits, needs careful per-tensor scaling, and changes the matmul math itself — a much larger
correctness bet than I want mid-ladder; or go pure bf16, keeping the exact bf16 matmuls autocast already runs
but dropping the fp32 master and the cast boundaries. The middle option removes a safeguard built for a
precision I'm not using, so the risk is bounded and known — the honest next step.

Casting the entire net to bf16 and dropping the autocast context changes three things: the weights halve (30M
params, ~120MB→~60MB), the activations halve, and every autocast boundary disappears, so I stop paying the
fp32→bf16 and bf16→fp32 casts at each region — pure memory-bandwidth traffic, gone. What does *not* change is
the matmul math: under autocast the heavy matmuls *already* ran in bf16, so going pure-bf16 speeds up no matmul
FLOP. That's the honest scope — a memory-bandwidth and footprint win, not an arithmetic one, so I should expect
a *modest* increment, not another halving.

There's a compounding win with the previous rung, though. The sequence-length schedule runs its early phase at
large batches (16× the reference at length 32) — exactly where activation memory becomes the binding constraint
on how big a batch I can fit. The accounting matters: the weights (~120MB fp32) and AdamW's two moment tensors
(~240MB) are fixed per parameter and don't grow with the batch; what grows with batch×length is the
*activation* memory. So it's specifically the halved activations that give the short-length phase more headroom
to push its batches larger before hitting the 40GB ceiling — the weight halving is a nice side effect, the
activation halving is the one that compounds with rung 3, buying the batch room the schedule spends. (One-line
cast in the answer.)

The risk is real and worth locating precisely. The danger of a 7-bit mantissa isn't range — it's *swamping* in
small updates. Late in training the optimizer applies tiny steps, weight ← weight − lr·update, and when the
weight is O(1) while lr·update is ~1e-4, bf16's ~0.008 relative precision can't represent the sum — the update
rounds away and the weight sits still. That's precisely the late, small-gradient phase where the run is settling
onto the bar, so if bf16 hurts, it hurts by *stalling* the final descent above 3.8, not by blowing up. A second
place the coarser mantissa could bite: the grad-norm batch controller reads a global gradient norm, and summing
hundreds of per-tensor squared norms in bf16 would round badly enough to corrupt the signal it steers on — but
`get_grad_norm` builds its accumulator as an fp32 scalar and adds into it, so the reduction stays fp32 even when
every gradient tensor is bf16. The controller's readout is protected by construction; I checked this rather than
assumed it, because a silently-noisier grad norm would destabilize the batch size with no obvious symptom.

Two things make me willing to take the mantissa risk. First, this is a *short* run to a flattening-point loss —
I'm not chasing the last fraction of a nat where swamped updates matter most, so the regime is where bf16 is
most affordable. Second, bf16's fp32 range is a genuine hedge against the overflow/underflow that would make
this reckless in fp16. The other precision-sensitive spot is LayerNorm's mean/variance reduction over 384
features, which I rely on being carried in higher precision internally; if the late loss stalls, that reduction
and the swamped weight updates are the first two places I'd look before blaming the architecture. One thing must
not silently defeat the cast — the positional-bias base and the −∞ mask fill have to be bf16-typed too, or an
fp32 constant would force an upcast back to mixed precision on the attention path. The −∞ fill is fine: because
bf16 shares fp32's exponent field it has a genuine infinity, so a bf16 `-inf` is representable and
softmax(…,−∞) still evaluates e^(−∞) = 0 exactly — the causal mask survives the cast intact, no leak.

The two changes are independent but land together. Writing the wall-clock as (steps to bar) × (FLOPs per step)
× (seconds per FLOP) keeps the factors separate: SiGLU raises FLOPs-per-step ~3% but is meant to cut
steps-to-bar by more, so total FLOPs to the bar falls; pure bf16 leaves both of those alone and shrinks
seconds-per-FLOP by lightening the memory traffic each FLOP drags. Different factors, no double-counting, both
pointing down — which is why I'm willing to make two changes at once: if the run slows, I can tell from *which*
factor moved whether SiGLU cost me steps or bf16 cost me stability. If both hold, the loss-per-step curve should
be visibly steeper than the GELU baseline's at matched step counts, and if pure bf16 is over its skis the
failure is a *stall* — the late loss creeping to a floor a little above 3.8 — not a blow-up. The hedges are the
short-run regime and bf16's fp32 range, and the test is the same: does it still land at ~3.8.
