SwiGLU's row confirms exactly the suspicion I closed step 2 with: varying the gate inside the GLU family
has run its course. Read both rows together, because the pattern across the two gates is the real datum. On
seed 42 SwiGLU lands `val_loss` 2.2923 against GeGLU's 2.2952 — an improvement of 0.0029 nats, in the
predicted direction and precisely the tiny magnitude near-identical `Φ`/`σ` curves should produce
(`exp(2.2952)=9.93` → `exp(2.2923)=9.90`, about a third of a percent of validation perplexity). A real move,
directionally correct, but a third of a percent from swapping one smooth gate for another.

The secondary columns each carry a mechanism clue now that I have two rows to difference. LAMBADA fell most
in relative terms, 68.73 → 66.81 (−2.8%) — the long-range completion metric moved cleanly, where I aimed the
smoother SiLU gate. HellaSwag rose 32.90 → 33.40 and PIQA 64.15 → 64.64, the two gate-sensitive commonsense
columns lifting as bet. WinoGrande 50.36 → 50.43 is a +0.07 wiggle on its chance floor, read as nothing.
ARC-Easy dipped 54.88 → 54.71 (−0.17), within single-seed noise. And `elapsed` came down 21098 → 20750
(−1.6%), consistent with SiLU's one sigmoid-and-multiply being a hair cheaper than GELU's erf.

One column did *not* move with the others, and it is the one to sit with. WikiText-2 went the *wrong* way,
44.13 → 44.33 (+0.20 ppl). The gate swap that helped `val_loss` and long-range LAMBADA slightly *hurt* clean
in-domain word-level perplexity. On a single seed 0.20 ppl is within noise and I won't call it a regression
— but across *both* GLU gates WikiText-2 has not budged out of a 0.2-ppl band (44.13 ↔ 44.33), and it even
drifted the wrong way. That is a clue with teeth: if the gate axis cannot move WikiText-2 at all, then
whatever headroom WikiText-2 still has lives on a *different* axis, and a change that finally moves it *down*
would be the tell that I've found that axis. I file it as the sharpest falsifiable target for what I do next.

So what do GeGLU and SwiGLU actually share that I've held fixed across both steps? Both keep the
nonlinearity *on the gate* of a product of two linear projections; both carry the value path linearly; both
sit at 8/3 width with three matrices; and both use a *smooth, bounded-rate, sigmoid-derived* gate — `Φ(z)`
or `σ(z)`. Every move so far has lived inside one axis (the gate's smooth activation) of one structure (the
gated product). I've been optimizing the gate and treating the *activation primitive itself* as furniture.
The flatness of the GeGLU→SwiGLU step, joined by the pinned WikiText-2, says that axis is nearly spent.

What's genuinely on the table: changing GLU *width* off 8/3 breaks the matched-budget equality — out.
Spending the third matrix on serial *depth* forces `h=2d` at matched budget (the arithmetic from step 1)
and still computes a composition of degree-one bends — out. A *gated* squared-ReLU, `(ReLU(xW))² ⊗ (xV)`,
conflates two changes at once (leaving the smooth gate AND introducing the quadratic), keeps the narrowed
8/3 width I now suspect ate some headroom, and multiplies a linear value by an unbounded squared gate — the
unbounded-gate instability I flagged against Bilinear, worse; a real idea but not a controlled one, deferred.
That leaves the move that changes exactly one thing, recovers the full 4d width, and targets the axis the
pinned WikiText-2 says is live: leave gating entirely, return to the plain two-matrix 4d FFN, and reshape the
*pointwise function itself* — the "option one" I set aside at step 1.

Which shape? Not another smooth linear-asymptote curve — Mish, a re-tuned Swish-β, a shifted GELU — because
those are more of the same family that just left WikiText-2 pinned. I need a different *kind* of curve. Go
back to the anatomy: ReLU is `z·1[z>0]`, GELU `z·Φ(z)`, SiLU `z·σ(z)` — every activation I've touched has
the form `z·g(z)`, and in every one the gate `g` *saturates to at most 1* (`1[z>0] ≤ 1`, `Φ → 1`, `σ → 1`).
That saturation is exactly why they all grow *linearly* for large positive `z`: `z·g(z) → z·1 = z`. I've
varied *which* saturating gate sits in the slot; I've never asked what happens if the gate does *not*
saturate.

So make the gate unbounded. The simplest monotone gate that keeps ReLU's hard off-floor for `z ≤ 0` but does
not cap at 1 for `z > 0` is `ReLU(z)` itself — the identity ramp on the positives. Drop it into the `z·g(z)`
slot: `z·ReLU(z) = z·max(0,z)`, which is `0` for `z ≤ 0` and `z²` for `z > 0` — exactly `max(0,z)² =
(ReLU z)²`, squared-ReLU, the Primer-EZ activation (`z·max(0,z)` and `max(0,z)²` agree on both branches). So
this is not a new gadget: it is the member of the `z·g(z)` family where the gate is the *unbounded* ramp in
place of a saturating sigmoid. The value-times-gate motif carries straight through — GeGLU and SwiGLU untied
the gate onto a second projection; squared-ReLU keeps it tied to `z` pointwise (no third matrix) but lets it
grow without a ceiling — and *gate saturation*, the one axis I've never moved, is what this step turns. A
change of *kind*, degree-one-asymptotic to degree-two.

That framing also fixes *why degree two and not three*. `z·ReLU(z)` is the `p=2` member of `max(0,z)^p`:
`p=1` is plain ReLU (the linear-asymptote base I'm leaving), `p=2` the minimal super-linear integer power,
`p=3` and up squaring their dynamic range again at every step — a `z=3` unit fires `27` under a cube against
`9` under the square, a far more violent map to hand 24 stacked layers with only `grad_clip` to catch it.
The minimal step off the linear-asymptote family is `p=2`; the cube is a lever for after I know the square
helps.

Tabulate GELU, ReLU, ReLU² at `z = −1, 0, 0.5, 1, 2, 3`: GELU `−0.159, 0, 0.346, 0.841, 1.954, 2.996`; ReLU
`0, 0, 0.5, 1, 2, 3`; ReLU² `0, 0, 0.25, 1, 4, 9`. The crossover is at `z=1`, where ReLU² and ReLU agree
(both `1`); below it ReLU² sits *under* ReLU, above it it pulls *away*. So the square's effect is a see-saw
around `z=1`: it dims the weakly-matched half of the positive units (below `z=1`) and sharpens the
strongly-matched ones (above), redistributing the layer's response toward its most-activated units. Quantify
the selectivity `c_proj` sees: a strongly-matched unit at `z=2` against a weakly-matched one at `z=0.5` reads
`4/0.25 = 16` under ReLU², `2/0.5 = 4` under ReLU, `1.954/0.346 = 5.65` under GELU. So the down-projection
reads a hidden vector where the strong unit outweighs the weak by `16×` under squared-ReLU against about
`5.7×` under GELU — a far peakier, sparser input. That is soft, learned feature selection baked into the
activation: ReLU zeroes the negative half, squaring additionally de-emphasizes the small-positive half. A
*different* mechanism from the GLU steps — they added a multiplicative interaction across two projections;
this sharpens the response of a single projection.

The selectivity only matters if units actually land where the square bites, so pin the scale of `z` at init.
The preactivation is `z_j = Σ_i (ln x)_i W_ij` with `c_fc.weight ~ N(0, 0.02²)` on a LayerNormed input whose
components have variance ≈ 1; with `d = 1024` roughly-independent terms, `Var(z_j) ≈ 1024 · 0.02² = 0.4096`,
so `std(z) ≈ 0.64`. A typical unit sits at `|z| ≈ 0.64`; a 2σ "strong" unit at `z ≈ 1.28`, a 0.5σ "weak" one
at `z ≈ 0.32`. At those scales `ReLU²(1.28)/ReLU²(0.32) = 1.638/0.102 = 16.0` and `GELU(1.28)/GELU(0.32) =
1.152/0.200 = 5.75` — the same ~16-versus-6 gap holds at the actual operating scale, not just at round
numbers, and only sharpens as training pushes matched units further out.

The same init computation answers the stability worry from the other side. For zero-mean symmetric `z`,
`E[(ReLU z)²] = ½·Var(z) = 0.205`, so the mean squared-rectified activation at init is about `0.2` and its
RMS about `0.45` — *smaller* than the preactivation's own RMS of `0.64`, because squaring a sub-unit number
shrinks it. The layer starts *gentle*; the quadratic only becomes a large-magnitude map on the units that
train their way past `z=1`, exactly the sparse selective set I want it to sharpen. So "faster-than-linear
growth" is a training-time property of the few strongly-firing units, not an init-time blow-up.

Does going back to a rectified activation reintroduce ReLU's corner — the very thing GELU was chosen to
smooth? Differentiate: `d/dz max(0,z)² = 2·max(0,z) = 2·ReLU(z)`, which is `0` from both sides at the origin,
so squared-ReLU is C¹ — *smoother* at the origin than ReLU, whose derivative jumps `0 → 1` there. It is not
C² (the second derivative jumps `0 → 2`), but that constant positive curvature for `z>0` is precisely the
faster-than-linear growth I'm after. So the move off the smooth family does not drag ReLU's worst property
back with it.

The budget bookkeeping is the cleanest so far, and it changes what the comparison means. Squared-ReLU
adds *nothing*: it keeps the default's exact two matrices — `c_fc` `d×4d`, `c_proj` `4d×d` — the full `4d`
width, bias-free (`config.bias=False`), the only difference from the default MLP being `max(0,z)²` in place
of `GELU(z)`. Parameter count `2·d·(4d) = 8,388,608`, identical to the default and, by the 8/3 construction,
to GeGLU/SwiGLU's `8,454,144` within the same `+0.78%` nudge; FLOPs identical to the default, the square a
single fused elementwise op. So it is matched-budget to everything before it — and it is the only fill that
keeps the full `4d = 4096` width, recovering the `4096 − 2752 = 1344` units (a third) the GLU variants traded
for their third matrix. That matters more than tidiness: if any of the GLU family's ceiling was set by that
33% narrowing rather than by gating, this is the fill that reveals it, because squared-ReLU buys a sharper
per-unit response *and* keeps the wide layer. The GLU steps and this one disagree on where the extra
expressivity should go — multiplicative interaction at narrower width, or sharper activation at full width —
and holding the budget fixed is what lets `val_loss` arbitrate.

The worry that faster-than-linear growth destabilizes training over 12,030 iterations is buffered on three
sides. At init the squared activation is already gentler than the preactivation (above). Along depth, the
residual-scaled `c_proj` init `N(0, 0.02/√(2·n_layer)) = N(0, 0.00289)` — a weight std three-hundredths of
the up-projection's — keeps the residual stream's variance from growing regardless of activation gain. And
`grad_clip = 1.0` bounds the gradient spikes a strongly-firing quadratic unit produces (its derivative
`2·ReLU(z)` is unbounded, but the clip is the backstop). So I expect no instability and, crucially, *not* to
need any `CONFIG_OVERRIDES` — leaving the schedule where the substrate sets it keeps any `val_loss` change
attributable to the activation shape alone.

The edit goes back to the default's two-matrix skeleton — `c_fc = Linear(n_embd, 4·n_embd)`, `c_proj =
Linear(4·n_embd, n_embd)`, both bias-free, dropout on the output — with the single substantive change
`F.relu(x).square()` where the default had `self.gelu(x)`; I write `.square()` rather than `**2` as the
cheaper fused elementwise op that reads as the intent under `torch.compile`. `CONFIG_OVERRIDES` stays empty.
Shape-preserving drop-in, touching nothing else. The literal scaffold edit is in the answer.

On wall-clock, this is where the matched-budget story could actually improve rather than merely hold:
squared-ReLU does two *full-width* matmuls plus a square, and two wide matmuls launch fewer kernels over
larger, better-utilized tiles than the GLU variants' three narrow ones, with `.square()` far cheaper than a
sigmoid-multiply or erf. So I expect `elapsed` below SwiGLU's 20750 — plausibly the fastest of the three, by
more than the 348s the activation-only SiLU change bought, because this also collapses three matmuls into
two. An efficiency artifact of shape, not of doing less work.

This is a change of *axis*, not of gate. I expect `val_loss` below SwiGLU's 2.2923, and by a *wider* margin
than the 0.0029 the gate swap gave, because it combines a change of kind (activation shape) with a recovery
of the full 4d width — two sources of headroom, not one gate curve nudged. The sharpest falsifiable
prediction is WikiText-2: the gate axis left it pinned in a 0.2-ppl band, so if the sharper selective
activation is doing what I claim, WikiText-2 ppl should finally come *down* from 44.33 — the cleanest test of
the selectivity story available. I also expect throughput up. The exposure: on a single seed the downstream
accuracies are close enough that squared-ReLU could land any of them either way, so I claim only the primary
loss and WikiText-2 direction. And a residual risk — the quadratic's larger dynamic range could *hurt* the
smooth long-range LAMBADA even as it sharpens the selective columns; a LAMBADA regression alongside a
`val_loss` and WikiText-2 improvement would say sharpness helps local prediction but costs long-range
smoothness, and the follow-up would be a *gated* squared-ReLU to get both — but that is for past the strongest
baseline, not this step.
