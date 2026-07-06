The schedule levers — the dynamic effective-batch controller and the sequence-length growth — have squeezed
the run well down from the ~3.5 minutes the batch controller reached. I've been leaving two parts of the
*network itself* at their textbook defaults: the MLP nonlinearity and the precision policy. Both are worth
re-examining now, because at this point the per-step *quality* of the network is what gates how few steps I
need, and the per-step *cost* is dominated by the matmuls — and I priced those in the baseline, so I know
where they sit. The MLP is 16d² FLOPs per token per layer, ~24% of the whole forward pass once the vocab head
is counted; it's the biggest single chunk of block compute. If I can make each MLP FLOP *learn* more, fewer
steps reach the bar; if I can make each step *cost* less to run, the wall-clock drops directly. One change for
each. And there's a reason both come *now* rather than earlier: on the slow baseline, the schedule savings
were so large that a per-step-quality change like SiGLU would have been partly masked by them — the binding
constraint was scheduling, not the network. Now that the batch controller and the length schedule are spent,
the network's per-step quality *is* the binding constraint, so this is exactly when sharpening the MLP and
trimming the precision overhead pays the most visible return.

Start with the MLP. The baseline is the standard one: expand the channels 4×, apply GELU pointwise, project
back. GELU is a smooth gate — it multiplies each pre-activation by a smooth approximation of "is this value
positive." But it's a *fixed* gate: the gating function is the same nonlinearity applied elementwise, the
same curve in every channel and every input, and the only thing the layer learns is the linear map feeding
into it. Concretely, a GELU hidden unit computes GELU(w·x): a fixed scalar function of one learned linear
feature. What if the gate itself were *learned* — if, instead of one linear projection passed through a fixed
nonlinearity, the layer produced two linear projections and used one to gate the other? Split the widened
activation into two halves a and b and output a ⊙ σ(b): now each hidden unit is a *product* of two learned
linear features, one passed through a sigmoid-shaped gate. That product is the thing a plain MLP can't form in
a single layer — a multiplicative, AND-like interaction ("pass feature a only where feature b is active") that
a fixed pointwise curve can only approximate through extra depth. With a SiLU gate (x·sigmoid(x), smooth like
GELU) this is the SiLU-gated linear unit, SiGLU. The gate is now a function of a *separate learned projection*
of the input, so the layer can modulate each value channel by an input-dependent amount it controls. It's
worth being concrete about why that's more than cosmetic: to form a product of two input features, a plain
MLP needs *depth* — one linear-plus-nonlinearity can't multiply two of its inputs, so it takes a second layer
to approximate the interaction. SiGLU forms that product in a single layer, so it delivers a depth-2-flavoured
interaction at depth-1 cost. That's the mechanism behind "learns more per parameter," and it lands exactly
where I need it: this is a step-limited run — I noted earlier the corpus is only a few tokens per parameter,
so I'm never short on capacity, I'm short on *steps* — and a layer that extracts more structure per gradient
update spends the scarce currency (steps) better, even at a small premium in the abundant one (FLOPs). Gated
MLPs of this family consistently learn more per parameter than a plain activation MLP, which in a step-limited
run means each step makes more progress — fewer steps to the bar.

Before I commit to gating, let me make sure it's the right axis to spend on. There are three ways to make an
MLP do more. I could make it *bigger* — an 8× plain GELU instead of 4× — but that scales parameters and FLOPs
linearly for capacity I'm not obviously short on; at this scale the run is *step-limited*, not
capacity-limited, so brute width buys the wrong thing and costs the most. I could swap the *activation* for
something like a squared-ReLU: cheaper per element, sometimes a little better than GELU, but it's still a
fixed pointwise curve — it doesn't add the input-dependent, multiplicative interaction that's the whole point.
Or I could make the gate *learned*, which is SiGLU. Only the third option changes the *kind* of function the
layer can represent rather than just its size or its curve, and it does so at a 12.5% budget premium rather
than the 100% a width doubling would cost. That's why gating is the axis. And within the gated family, a SiLU
gate rather than a hard sigmoid is the natural pick here: SiLU (x·sigmoid(x)) is smooth and self-gating, of a
piece with the smooth GELU/LayerNorm net around it, so I'm not injecting a kink into an otherwise smooth
optimization landscape.

I have to keep the parameter and FLOP budget honest, though, or I've just made each step more expensive and
handed the gain straight back. Let me count it out. The plain 4× GELU MLP is expand d→4d (4d² params) plus
project 4d→d (4d²), so 8d² parameters and 16d² FLOPs per token. A SiGLU needs *two* linear projections into
the hidden width — one for the value, one for the gate — so if I naïvely kept a 4× hidden and doubled it for
the two halves, I'd blow the budget. If I want to *match* the 8d² of the plain MLP exactly, the arithmetic
says the gated hidden should be (2/3)·4d = 8d/3 ≈ 2.67d: expand d→2·(8d/3), project (8d/3)→d gives
2·(8/3)d² + (8/3)d² = 8d². So a parameter-matched SwiGLU-style block wants a hidden of ~2.67d. I round that up
to a clean 3×: expand to 2·3d = 6d (produced as two stacked 3d halves), SiGLU-gate them down to a 3d hidden,
project 3d→d. That's 6d² + 3d² = 9d² parameters — 12.5% more than the plain MLP's 8d², not budget-neutral but
close, and I'm choosing to pay that 12.5% because a 3d gated hidden is a rounder, slightly richer working
width than the matched 2.67d. What that 12.5% costs at the whole-model level is smaller than it sounds: the
MLP is ~24% of the forward FLOPs, so +12.5% on it is ~+3% on the per-step compute overall. So the bet is
sharp and quantifiable — SiGLU must cut the step count by more than ~3% for the trade to be a net win, and the
representational argument says gated MLPs clear that bar comfortably. And the trade is better than that ~3%
makes it look, because of where the per-step cost actually lives. I found in the baseline that the vocab head
alone is ~62% of the forward FLOPs and every layer's projections are fixed costs paid *once per step* whatever
the MLP does. SiGLU's payoff is fewer *steps*, and fewer steps means fewer passes through that dominant,
step-count-bound vocab head — so the real lever isn't the +3% I add to the MLP, it's that cutting steps
divides down the whole expensive fixed part of each step. A change that attacks step count is multiplied by
the fact that the biggest cost per step is amortized over exactly those steps.

```python
class SiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.SiLU()
    def forward(self, x, dim=-1):
        x = x.split((x.shape[-1]//2, x.shape[-1]//2), dim=dim)
        return x[0] * self.activation(x[1])      # value half, gated by SiLU of the gate half

class MLPBlock(nn.Module):
    def __init__(self, num_channels, expansion_factor=3):
        super().__init__()
        self.norm    = LayerNorm(num_channels, bias=False)
        self.expand  = nn.Linear(num_channels, num_channels*2*expansion_factor, bias=False)  # 2x for the two halves
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.siglu   = SiGLU()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.project(self.siglu(self.expand(x)))
        return x + residual
```

Let me trace the shapes at d=384 to be sure the split lines up, because an off-by-a-factor in the split would
silently halve the model's width. `expand` maps 384 → 2·3·384 = 2304. SiGLU's `x.split((2304//2, 2304//2))`
cuts that into two halves of 1152 = 3d each; the value half is 1152 wide, the gate half is SiLU'd and is 1152
wide, their elementwise product is 1152 = 3d. Then `project` maps 3d = 1152 → 384. So the hidden the projection
sees is 3d, exactly the 3× gated width I sized for, and the two 1152 halves come out of the single 2304-wide
expand — one matmul producing both value and gate, which is why the split is on the last dim of the expand
output. Shapes close.

One property of this arrangement I want to note is that the *value* half x[0] is passed through no
nonlinearity — the output is (linear value) ⊙ SiLU(gate), so there's an ungated linear path through the block.
That's good for gradient flow: ∂out/∂value = σ(gate), which sits in (0,1) and doesn't vanish the way the
gradient through a saturated nonlinearity would, so the value channels keep receiving clean gradient even when
the gate is nearly closed. The nonlinearity is spent on *deciding how much of each value to pass*, not on
mangling the value itself — a cleaner division of labour than a plain MLP, where every hidden unit's gradient
is at the mercy of where GELU happens to be on its curve. And the gating machinery itself is nearly free: the
`split` returns two views into the single expand output, no copy, so the only added arithmetic over a plain
MLP is one SiLU and one elementwise multiply over the 3d hidden — negligible next to the 6d² + 3d² of matmul.
The 12.5% premium I counted is all in the linear layers, which is where I want the cost to be, on the tensor
cores rather than in bandwidth-bound pointwise ops.

One consistency check with the baseline's initialization, because `project` is a residual projection and I
scaled those by 1/√(2·num_blocks) to keep the residual stream's variance bounded with depth. Does that
scaling still do its job when the sublayer output is a gated product rather than a GELU? At init the gate
half b is near zero, so σ(b) ≈ σ(0) = 0.5, and the SiGLU output a·σ(b) is roughly the value half scaled by
~0.5 — a *smaller* effective magnitude than a plain GELU MLP's activation, not a larger one. So the same
1/√(2·num_blocks) residual-projection scaling that bounded the GELU stream is, if anything, slightly
conservative for SiGLU: the per-block contribution starts a touch gentler, and the depth-stability argument
from the baseline still holds without re-tuning the init. Good — the new MLP doesn't quietly reintroduce the
exploding-residual-stream problem I paid that init factor to prevent.

Now precision. The baseline runs *mixed* precision: an autocast context where the heavy matmuls happen in
bf16 but the tensors live in fp32 and get cast on the fly. Autocast is the safe default — it keeps a
high-precision master copy and only drops to bf16 inside the cast region — but it costs something, and I want
to be precise about *what*, because the answer to "can I drop it" depends entirely on why it's there. Autocast
exists in the first place for fp16: fp16 has only 5 exponent bits and tops out near 65504, so training in it
needs an fp32 master copy and loss-scaling to keep small gradients from underflowing. bf16 is a different
animal — it carries fp32's *full 8 exponent bits*, so it covers the same dynamic range (~3.4e38), and the only
thing it gives up relative to fp32 is mantissa: 7 bits, ~2 decimal digits of relative precision (2⁻⁷ ≈ 0.008).
So the fp32 master copy that autocast maintains is a safeguard designed for fp16's tiny *range* — and bf16
doesn't have a range problem. Which makes me ask: on this hardware, is the fp32 master buying me anything, or
is it paying memory and cast overhead for a safety margin bf16's range already provides?

Let me name the precision options so I'm choosing rather than drifting. I could keep autocast — the status
quo, safe, and it pays cast overhead plus an fp32 footprint for a safeguard that exists to protect fp16's
range, a problem bf16 doesn't have. I could go all the way to fp8, which the newest kernels support and which
would genuinely speed the *arithmetic* — but fp8 has only 2–3 mantissa bits, needs careful per-tensor scaling
to not fall apart, and unlike bf16 it would change the matmul math itself, so it's a much larger correctness
bet than I want to make mid-ladder on this hardware. Or I could go pure bf16: keep the exact bf16 matmuls
autocast already runs, but drop the fp32 master and the cast boundaries. That middle option is the one where
the safeguard I'm removing was built for a precision I'm not using — the risk I take on is bounded and known,
not a new arithmetic regime — so it's the honest next step.

If I cast the entire net to bf16 — weights, activations, everything — and drop the autocast context, three
things change. The weights halve, 30M params going from ~120MB in fp32 to ~60MB in bf16; the activations
halve too; and every autocast boundary disappears, so I stop paying the fp32→bf16 cast going into each region
and the bf16→fp32 cast coming out — pure memory-bandwidth traffic, gone. What does *not* change is the matmul
math: under autocast the heavy matmuls *already* ran in bf16 on the tensor cores, so going pure-bf16 does not
speed up a single matmul FLOP. That's the honest scope of this change — it's a memory-bandwidth and footprint
win, not an arithmetic one, so I should expect a *modest* increment, the kind of thing that shaves seconds by
making each step lighter to move through memory, not the kind that halves a run. Being clear about that keeps
me from over-attributing.

There's a compounding win with the previous rung that makes the footprint halving more than a bandwidth
detail, though. The sequence-length schedule runs its early phase at large batches — 16× the reference batch
at length 32, to hold the token budget — and large batches at short lengths are exactly where activation
memory becomes the binding constraint on how big a batch I can actually fit. Halving the activation footprint
gives that phase more headroom, so bf16 doesn't only make each step lighter to move; it lets the seqlen
schedule push its short-length batches larger before hitting the 40GB ceiling, which feeds back into the same
throughput the schedule was chasing. The precise accounting matters here: the weights are ~120MB in fp32 and
AdamW's two moment tensors another ~240MB, but those are fixed per parameter and don't grow with the batch —
what grows with batch×length is the *activation* memory, so it's specifically the halved activations, not the
halved weights, that buy the schedule its extra batch room. The weight halving is a nice side effect; the
activation halving is the one that compounds with rung 3. The two changes aren't independent on the memory axis — bf16 buys the
batch room that the schedule spends.

```python
net = net.to(hyp['misc']['device'], torch.bfloat16)   # pure-bf16 net, no autocast region
```

The risk is real and worth locating precisely, because "bf16 is fine" is exactly the kind of claim that's true
until it isn't. The danger of 7-bit mantissa isn't range — bf16's range is fp32's — it's *swamping* in small
updates. Late in training the optimizer applies tiny steps: weight ← weight − lr·update, and when the weight
is O(1) while lr·update is ~1e-4, bf16's ~0.008 relative precision can't represent the sum — the update rounds
away and the weight sits still. That's precisely the late, small-gradient phase where the run is trying to
settle onto the bar, so if bf16 is going to hurt, it hurts by stalling the final descent above 3.8, not by
blowing up. There's a second place the coarser mantissa could bite that I want to rule out: the grad-norm batch controller
from two rungs ago reads a global gradient norm, and summing hundreds of per-tensor squared norms in bf16
would round badly enough to corrupt the signal the controller steers on. But `get_grad_norm` builds its
accumulator as an fp32 scalar and adds into it, so the reduction stays in fp32 even when every gradient tensor
is bf16 — the controller's readout is protected by construction, and the pure-bf16 net doesn't degrade the
batch schedule underneath it. That's the one interaction I actively checked rather than assumed, because a
silently-noisier grad norm would have destabilized the batch size without any obvious symptom.

Two things make me willing to take the mantissa risk on the weights. First, this is a *short* run to a
flattening-point loss — I am not chasing the last fraction of a nat where swamped updates would matter most,
so the regime is the one where bf16 is most affordable. Second, bf16's fp32 range is a genuine hedge against the overflow/underflow that
would make this reckless in fp16. The other precision-sensitive spot is LayerNorm's mean/variance reduction
over the feature dimension — a sum of 384 squared terms is exactly the kind of reduction bf16 handles worst —
and there I'm relying on the layernorm reduction being carried out in higher precision internally rather than
accumulating in bf16; if the late loss stalls, that reduction and the swamped weight updates are the first two
places I'd look before blaming the architecture. I do have to make sure nothing silently defeats the cast — the positional-bias
base matrix and the −∞ fill in the mask have to be bf16-typed too, or an fp32 constant would force an upcast
and I'd be back to mixed precision by accident on the attention path. The −∞ fill is worth one check: because
bf16 shares fp32's exponent field it *has* a genuine infinity, so a bf16 `-inf` is representable and
`softmax(… , −∞)` still evaluates e^(−∞) = 0 exactly — the causal mask survives the cast intact, no leak, no
spurious tiny weight where there should be none. If bf16 lacked infinities I'd have had to substitute a large
finite negative and worry about it; it doesn't, so the masked-softmax path needs no special handling beyond
the type.

The two changes are independent but land together, and they attack the two different gates I named at the
start: SiGLU makes each step *learn* more (fewer steps to the bar) at ~3% more compute per step, and pure bf16
makes each step *cheaper to run* (no autocast casts, half the memory traffic) at no change to the arithmetic.
If both mechanisms hold, the A100 seconds should fall — SiGLU by contracting the step count, bf16 by lightening
each step — while the val loss still lands at ~3.8 and the perplexity at ~44.7. The falsifiable tells are
specific: if SiGLU is doing its job, the loss-per-step curve should be visibly steeper than the GELU baseline's
at matched step counts (that's the "learns more per step" claim made checkable); if pure bf16 is over its
skis, the failure won't be a blow-up but a *stall* — the late loss creeping to a floor a little above 3.8 as
swamped updates stop registering. The hedges are the short-run regime and bf16's fp32 range, and the test is
the same one every rung answers to: does it still land at ~3.8. If it does, the two gains — more-per-step ×
cheaper-per-step — compose, and the run gets faster on both axes at once.

Writing the wall-clock as (steps to bar) × (FLOPs per step) × (seconds per FLOP) makes the composition
precise and keeps me honest about which factor each change touches. SiGLU raises FLOPs-per-step by ~3% but is
meant to cut steps-to-bar by more, so the product of the first two factors — total FLOPs to the bar — should
fall; pure bf16 leaves both of those alone and shrinks the *third* factor, seconds-per-FLOP, by lightening the
memory traffic each FLOP drags along. Different factors, no double-counting, and both pointing down — which is
exactly the shape of a change I'm willing to make two of at once, because if the run slows I can tell from
*which* factor moved whether SiGLU cost me steps or bf16 cost me stability.
