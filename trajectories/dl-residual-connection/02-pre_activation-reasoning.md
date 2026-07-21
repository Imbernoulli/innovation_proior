The gated block landed where its low bandwidth said it would. On ResNet-20/CIFAR-10 it hit 92.96 — a strong
vanilla ResNet-20 number, the tell that the scalars settled near 1 and the block behaved as the baseline;
there was little structural slack at depth 20 on a 10-class problem and the gate found none, exactly the
"match the baseline, best on the easy case" outcome. On the deeper CIFAR-100 settings the gate ran out of
road: 71.98 (ResNet-56) and 73.46 (ResNet-110). Their *relationship* is the diagnostic. Doubling the
residual depth — 27 blocks to 54, twice the representational budget — bought only `73.46 − 71.98 = 1.48`
points. A 110-layer net genuinely using its extra 27 blocks should pull clear of a 56-layer one, not crawl
1.48 past it; the flat gap says the marginal blocks contribute almost nothing. And the gate could not have
fixed this: a per-block multiplier rescales how *loud* each block is but cannot change how cleanly signal and
gradient pass *through* the stack, and at 110 layers "through the stack" is the binding constraint. So the
limit is not residual *magnitude* (which the gate already let SGD tune, to no avail) but residual *wiring* —
what sits on the identity path and in what order the block applies BN and ReLU. That is the lever I held
fixed last time.

First rule out the cheaper explanation. If the deep stall were *capacity*, the fix would be more parameters,
not a reorder — but the 110-net already carries twice the blocks and parameters of the 56-net and still
edges it by only 1.48, so on this substrate more depth-worth-of-parameters is demonstrably not converting
into accuracy. Widening would lift both nets together and leave the depth-utilization gap where it is; the
evidence is specifically that the extra 27 blocks are wasted, a through-stack symptom, not a too-small-model
one.

So look hard at the piece I froze. The default (and gated) block is post-activation: `ReLU(BN(conv1(x)))`,
`BN(conv2(out))`, add, then a **final ReLU on the sum**. That last ReLU sits *after* the addition, so the
value carried from one block to the next is `ReLU(...)` — every block's output is forced non-negative before
the next block and its shortcut see it. Follow the within-stage shortcut across a stage: the map is
`x ↦ ReLU(x + F(x))`, and chained across the stage the clean additive identity `x_out = x_in + Σ F` is
interrupted at *every* block by a rectifier.

Quantify the damage. Differentiate `x_{ℓ+1} = ReLU(x_ℓ + F_ℓ(x_ℓ))`: the block Jacobian is
`diag(m_ℓ)·(I + ∂F_ℓ/∂x)` with `m_ℓ ∈ {0,1}` the ReLU mask. Across a stage of `k` blocks the "clean" term
that should carry gradient to the bottom undiminished is `diag(m_k)·…·diag(m_1)`, a product of `k` binary
masks. For a roughly zero-mean pre-activation each mask passes a coordinate with probability ~1/2, so a
coordinate survives all `k` clamps with probability ~`2^{−k}`: `2^{−18} ≈ 3.8×10⁻⁶` for a ResNet-110 stage,
`2^{−54}` across all 54 blocks. The identity path's whole promise — a top gradient reaching the bottom along
a route of all 1's — is erased coordinate by coordinate; gradient still reaches the bottom through the `∂F`
terms, but that is exactly the diminishable route the residual reformulation was meant to make optional. At
depth 20 the same product is `2^{−6}` per stage and the clamps barely matter. That depth dependence matches
where the gated numbers left headroom.

This sharpens into an ordering prediction, not just a lift. The per-stage clamp survival is `2^{−9}` at
ResNet-56 against `2^{−18}` at ResNet-110 — a ratio of `2^9 = 512`, so the deepest net's identity path is
~500× more thoroughly clamped per stage, and removing the clamps should help ResNet-110 far more than
ResNet-56. And in the quiet-block regime the deep-refining stall lives in (`f' ≈ 0`), the post-activation
two-block Jacobian collapses to `m₂·m₁`, zero wherever either pre-ReLU sum went negative, while the mask-free
path gives `(1+f₂')(1+f₁') = 1`, full transmission. At exactly the quiet late blocks the deep net most needs
to pass gradient through, post-activation can carry none and the clean path carries all.

What I want is a block whose identity path is genuinely unobstructed — no nonlinearity on the through-line —
so the stage Jacobian is `I + Σ ∂F_ℓ` with no mask. The obstruction is the activation order: the
post-activation block does the nonlinearity after the addition because the convs come first. Move the ReLU
*inside* the branch, ahead of the convs — **pre-activation**, BN-ReLU-Conv twice:
`F(x) = conv2(ReLU(BN(conv1(ReLU(BN(x))))))`, and the output is just `shortcut(x) + F(x)` with **no
activation after the addition**. Each block's nonlinearity acts only on the *copy* it pulls into its branch;
the block-to-block signal along the shortcut is the raw additive stream, never clamped.

Before accepting the full reorder, ask whether a smaller edit suffices. The tempting half-measure keeps
post-activation order but just *deletes the final ReLU*: `shortcut(x) + BN(conv2(ReLU(BN(conv1(x)))))`. That
alone gives the clean backward path — masks gone, through-line `I + ∂F`. But then `conv1` reads the *raw*
additive stream directly, with no normalization at the branch entrance, and that stream grows without bound
in a mask-free residual net (variance below), so `conv1` sees an increasingly badly-scaled input as depth
grows. Full pre-activation puts BN at the *entrance* of every branch, so *every* convolution in the network
gets a normalized input. The half-measure buys the backward win and leaves the forward-conditioning win on
the table; since both effects scale with depth and the whole point is the deep-net payoff, I take the full
reorder — that is why it is a reorder, not merely a deletion.

The forward benefit bites specifically against the *fixed* optimizer. The recipe hands every layer one
global `lr = 0.1`, and a conv's effective update scales with the magnitude of the input it convolves.
Post-activation `conv1` consumes the raw through-line whose variance compounds with depth (the ~19×-per-stage
growth below), so different convs see systematically different input scales and experience one global lr very
differently — too large for the deep large-input layers, too small for the shallow ones. Pre-activation pins
every `conv1`'s input to unit variance via `bn1`, so one lr is simultaneously right for all of them — and it
costs nothing, only relocating BN layers already present.

Now the substrate-specific fill, which differs from the textbook block. First, the shortcut on the
dimension-changing transitions cannot be a bare identity — the shapes do not match — so it stays a 1×1
stride-2 Conv-BN projection applied to the *raw* input `x`. The clean unobstructed identity therefore holds
only on within-stage blocks; at the two transitions per net the shortcut is a small Conv-BN projection, which
is fine — BN there keeps those two points well-scaled, and the overwhelming majority of blocks keep the clean
additive highway.

Second, a fixed-small residual scale whose value I have to derive. In pre-activation nothing normalizes the
within-stage shortcut, so the stream accumulates: `x_out = x_in + Σ_ℓ alpha·F_ℓ`, each fresh `F_ℓ` carrying
~unit per-channel variance. Treating a stage's 18 branches as roughly independent, they add ~`18·alpha²` to
the stream. At `alpha = 1` that is ~18, so a stream entering at variance ~1 leaves at ~19 — a 19× blow-up
within a single stage, before the next transition's BN can renormalize, exactly the scale explosion that
makes the first epochs of a very deep pre-activation net jittery. At `alpha = 0.1` the added variance is
~0.18, so the stream grows from ~1 to ~1.18 — the identity path stays dominant and the residuals are a gentle
perturbation the convs grow into. The neighbors bracket it: at `alpha = 0.2` the residuals add ~0.72 (a 72%
per-stage swell, jittery again), at `alpha = 0.05` only ~0.045 (under 5% of the running variance, so timid
against the decay pulling `alpha` down, meaning the residual must climb a long way before doing real work).
0.1 is the interior point where the added variance is a real fraction of the stream yet the identity term
stays clearly dominant.

And 0.1 carries the earlier lesson — not 0, which would waste the early budget on an effectively-shallow net
— but note this scalar means the *opposite* of the gate's: the gate at 1 said "behave like the baseline,
adjust"; the scale at 0.1 says "start near identity, grow." The climb is fast, since the drive `<g, F(x)>`
is persistently positive for a block the loss wants and momentum 0.9 amplifies the effective step, so `alpha`
climbs from 0.1 toward order 1 within the first handful of epochs — a front-loaded cost repaid across the
rest of training. As `alpha → 0` the within-stage blocks collapse to a clean identity, so starting near that
limit is safe, and because `alpha` is learnable SGD walks it up wherever a block wants more residual; I fix
only the starting point.

This is a reorder, not smuggled capacity: pre-activation moves `bn1` from `planes` onto `in_planes`,
identical for within-stage blocks and a hair *fewer* BN params at the transitions where `bn1` sits on the
smaller incoming width; the convs and dimension logic are unchanged, and the only genuine addition is one
scalar per block. So the parameter budget is essentially the last one's — what changed is the *order*,
precisely what the gated numbers pointed to as the deep-net limit. The full module is in the answer.

The payoff is depth-dependent and the three settings are a depth sweep. On ResNet-20/CIFAR-10 the clean
highway buys little — 20 layers barely feel the clamps (`2^{−6}` per stage) — and the 0.1 warm-up costs a
few epochs, so I expect to land around the gated 92.96, possibly a hair under. The real test is the deep
CIFAR-100 nets: if their limit was the obstructed identity path rather than magnitude, pre-activation should
lift ResNet-110 most and pull it decisively above ResNet-56, with ResNet-56 improving less (27 blocks
suffered the clamps less than 54); if instead the problem had been magnitude, all three would move together.
One risk stands out: pre-activation fixes how signal flows *through* depth but does nothing to make the deep
net *exploit* or *regularize* its 100-class fit — so if flow improves and the very deep net still
under-generalizes, the next lever is regularization across depth, which a reordering can never provide.
