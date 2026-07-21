Block AttnRes is the strongest result so far, and its number tells me both that the depth-flow idea was
right and exactly where it stopped short — so let me read the whole table before deciding what is left to
do. Validation loss fell to 2.2544, the first sub-2.26 result. Set that step against the two before it:
prores→learned-scaling was `−0.0027`, learned-scaling→attnres is `−0.0136`. The attnres step is five
times the last scalar step and larger than the two scalar steps *combined* (`0.0056 + 0.0027 = 0.0083`).
That is the fingerprint of *capacity added*, not conditioning tuned — the scalar family was decelerating,
and going full-rank broke the deceleration. The perplexities moved with it the way the "per-token choice
of which deep block to read" story predicted: WikiText-2 dropped to 41.82 (from 43.91, a `−2.09` move, by
far the biggest so far) and LAMBADA fell to 64.32, finally below *both* prores's 67.21 and
learned-scaling's 68.76 — so the attention did the deep-layer-conditioning *and* the token-identity job in
one mechanism, exactly where the two scalar schemes each had to trade one channel for the other. One
column went the other way and I should not skip it: ARC-Easy slipped to 55.01, `−0.84` from
learned-scaling's best-so-far 55.85. That is a small, real regression, and it has a plausible reading —
when `x0` stopped being a dedicated additive injection with its own gain and became just one of ~7
attention sources competing for softmax mass, the knowledge-recall edge the tied-embedding route was
feeding softened. HellaSwag ticked up to 34.05 (`+0.15`), the first move it has made at all, as I
guessed a richer depth flow might finally allow. So the jump from a rank-one scalar mix to a
content-dependent, full-rank mix over depth was the largest single move so far, and it added
genuine capacity — with a faint tell that it did so partly at the expense of the clean token-identity
route.

But read what it cost to fit the budget, because that cost is the seam. To survive a fixed 2-GPU,
micro-batch-32 run, Block AttnRes had to *coarsen* the depth axis: it kept ordinary unweighted residuals
*within* each block of 4 layers and only ran the learned attention at the 6 block boundaries. So inside
every block of 4 I am right back to the rigid unit-weight accumulator everything so far has been fighting —
the dynamic routing acts at 5 seams plus a readout, and the fine scale stayed dumb. That alone is a reason
the number is not the last word. And there is a deeper structural fact Block AttnRes never addresses, one
that sits underneath *every* step so far, including the ARC regression I just read. No matter how I weight
or attend over the *single* residual stream, that stream is forced to serve two conflicting jobs at once.
It must stay a clean identity highway so gradients reach the shallow layers, *and* it must carry each deep
layer's output strongly enough to keep that layer's representation distinct. Those two demands pull the one
stream's coefficient in opposite directions, and no step gave the network *more than one* stream to write
into. That is the move I have not made, and it attacks the conflict at its root rather than at one scale of
it.

Let me name the conflict precisely, because the fix has to be derived from it, and I can ground it in a
number I already have. Trace Pre-Norm again: `h_k = h_{k−1} + T(Norm(h_{k−1}))`, branch input normalized,
raw output added to an unnormalized stream whose norm climbs with depth. As it climbs, a fresh layer's
output is a *shrinking fraction* of the total, so deep layers' contributions wash out and adjacent deep
features collapse toward each other — the representation redundancy I first suspected at the vanilla floor
and which Block AttnRes only relieves at boundaries. The naive counter is Post-Norm — normalize after the
addition so each output stays a meaningful fraction — but that puts a normalization Jacobian back on the
highway and the gradients vanish with depth. This is a seesaw, and the key realization is that it is
*structural for a single stream*: one coefficient governs both "how much of the past survives" (the
gradient route) and "how much of this layer enters" (the depth influence), and one number cannot satisfy
both. I do not have to take that on faith — learned-scaling's `resid_lambda` *is* that single coefficient,
one learned number per layer sitting on exactly this seesaw, and its result was precisely what the seesaw
predicts: a small val-loss gain that could not hold both channels, buying WikiText-2/ARC while giving
LAMBADA back. A single scalar per stream can pick *one point* on the seesaw; it cannot win it. Every scalar
fix I have tried — ProRes's schedule, learned-scaling's carry, and the ReZero scalar and depth-aware inits
I rejected along the way — slides along this seesaw without tipping it off. So the structural escape is not
a better scalar; it is to stop having one stream.

What if the residual stream carried `n` parallel copies instead of one? Replicate the embedding into `n`
copies to form a hyper-hidden *matrix* `H ∈ R^{n×d}` — here, in this task's tensor shapes, `(B, T, n, D)` —
and let every layer operate on the whole matrix, summing the `n` copies back into one vector only at the
very top, right before the final LayerNorm and the head. The point of `n` copies is decoupling: now one
copy can act as the clean Pre-Norm gradient highway while another carries a strongly-written, distinct deep
output, so I no longer have to make a single coefficient serve both jobs. With `n > 1` the two demands live
in different copies, and the network can *reserve multiple patterns* of connecting to preceding layers
simultaneously. I can sanity-check that this is structural and not just extra parameters by looking at the
degenerate case: with `n = 1` there is still exactly one stream, the seesaw is untouched, and any dynamic
routing I add is just noise on that single stream — so `n = 1` should be *no better than, and plausibly
worse than, baseline*. It is the step from 1 to 2 copies, not the routing machinery, that breaks the
seesaw. That is the difference between sliding it and tipping it off.

Now derive the routing rule rather than guess it, because the overhead has to stay negligible or this
cannot run at GPT-2 Medium scale on a fixed budget. A layer reads the matrix `H`, must form a single input
for its sublayer, run the sublayer, and write the result back into the `n` copies. There are two distinct
axes of connection. *Depth*: how the new sublayer output is distributed into the copies, and how each copy
carries forward — the generalization of the residual skip. *Width*: how the copies exchange information
within a layer, and how they are mixed to form the sublayer's single input. Collect every weight into one
small `(n+1)×(n+1)` matrix where index 0 is the sublayer-output slot and `1..n` are the copies. Its first
row `β` distributes the single output back into the copies (the depth write); its first column `A_m` mixes
the copies into the sublayer input `h_0 = A_m^T H` (the width read); its `n×n` block `A_r` carries the
copies to the new copies `H' = A_r^T H` (the stream's own recombination). The whole layer is then: read the
input via `A_m`, carry the streams via `A_r`, run the sublayer on `h_0`, and write its output back via `β`
plus the carry. Two structural bonuses fall out for free, and they are why this is more than a
re-parameterization. The matrix *contains* Pre-Norm and Post-Norm as the non-trainable `n = 1` special
cases, so I lose nothing relative to the floor. And at `n = 2`, specific integer matrices express both the
ordinary sequential residual *and* a parallel-block arrangement — so learning the matrix learns a soft,
even dynamic, blend of sequential and parallel depth that the fixed residual can never reach.

I want the routing to depend on the input, not be a fixed learned constant, because the right depth/width
mix surely differs token to token — the same instinct that made Block AttnRes's per-token attention beat
the static scalars. So make the matrix entries functions of `H`: keep the static matrix as a base and
*add* a small dynamic correction predicted from `H` — normalize `H`, take a linear projection, squash with
`tanh`, scale by a small learnable factor, add to the static base, separately for `β`, `A_m`, `A_r`. Each
piece earns its place. The norm-before-projection keeps the depth-growing stream scale out of the routing
predictor — the disease I am curing must not re-enter through the router, the same reason Block AttnRes
RMS-normed its keys. The `tanh` bounds the correction so a runaway logit cannot blow up the connection
weights and destabilize a 13.5k-step run I cannot afford to lose. The small init scale (`0.01`) means the
dynamic part starts negligible and the network has to *earn* its way off the static matrix rather than
being handed a random routing at step zero.

Initialization is the part I cannot get wrong, and it is the cleanest argument for trying this here — but I
want to *trace* it rather than assert it, because the last step taught me that "starts at the floor" is a
claim I have to check. Two requirements. The dynamic projections start at zero, so `tanh(0) = 0` and the
dynamic correction is exactly nothing at init — the layer begins as pure static hyper-connection. And the
static matrix encodes Pre-Norm-on-`n`-copies: `β = 1` writes the full output into every copy (as Pre-Norm
adds the full branch), `A_r = I` carries each copy through unchanged (each stream a clean identity
highway), and `A_m = e_{k mod n}` is a one-hot that reads the sublayer input from a single copy, rotated by
site index `k` so the copies are used round-robin and none is privileged. Now trace `n = 2` at step zero.
Both copies start equal to the lifted embedding, `h_1 = h_2 = x`. The width read `A_m = e_{k mod 2}` picks
one copy, but both are `x`, so `h_0 = x`; the sublayer runs `T(Norm(x))`, call it `t`. The depth write adds
`t` into every copy (`β = 1`) and the carry `A_r = I` leaves each copy in place, so each copy becomes
`x + t` — and they stay equal. Iterate across all sublayers and every copy tracks the *same* value, which
is exactly the single Pre-Norm stream `x_prenorm`. At the top I sum the `n` copies: `Σ = n · x_prenorm`.
That is not literally Pre-Norm — it is `n` times it — but the sum feeds `ln_f`, and LayerNorm is invariant
to an overall scale, so it divides the `n` straight back out and the head sees *exactly* the Pre-Norm
logits. So the honest statement is: at init, with `n` identical copies, the model computes `n·x_prenorm`
and `ln_f` normalizes it to bit-for-bit Pre-Norm. Unlike Block AttnRes — which started at a uniform
block-average, a genuinely different operating point than vanilla — hyper-connections start at the vanilla
operating point and bend away only as the dynamics learn. That is the safest possible starting point on
this budget, and it means the ARC regression Block AttnRes paid for its non-vanilla start is a tax this
step does not owe.

Now the budget check, because it decides the expansion rate, and the arithmetic is friendlier than I
feared. Static parameters per site are the `(n+1)×(n+1)` matrix, `O(n²)` scalars — nothing. Dynamic
parameters per site are projections of size `O(nd)` plus two scalars and a norm — tiny against the `O(d²)`
of attention and the MLP. The main new compute is the width matmul `A^T H`, an `(n+1)×n`-by-`n×d` per
token, a rounding error for small `n`. Memory is the real cost: the `n` streams are the working state,
`(B, T, n, D)`, carried through every layer. At `n = 2`, that is `2 · 67 ≈ 134 MB` of residual activation
against vanilla's single `67 MB` — and, tellingly, *less* than the up-to-`0.47 GB` source list Block
AttnRes had to hold, because hyper-connections keep no list of past block outputs, only the `n`-stream
present state. So the thing I am trying to beat was actually *heavier* on activation memory than `n = 2`
would be; the finale fits more comfortably than the step below it. `n = 4` doubles that to `≈ 268 MB`, the
reach if memory allows. The known ablation picture matches the seesaw argument: `n = 4` is where dynamic
clearly beats static, and `n = 1` is *worse* than baseline (a single dynamic stream has no room to reserve
patterns and the dynamic noise just hurts). On *this* 24-layer, micro-batch-32, 2-GPU budget, where Block
AttnRes already had to coarsen to fit, I will not assume `n = 4` is free — but `n = 2` is comfortably
affordable and is exactly the point where the seesaw provably breaks, so it is the honest default to land
for this task, with `n = 4` the reach if memory allows. The streams sum to one vector before `ln_f`, so
the head and the loss are untouched.

Place it in the edit surface, because the fit is exact and worth being precise about. The contract keeps
`CausalSelfAttention`, `MLP`, `LayerNorm` fixed and asks `Block.forward(x) → x` and
`GPT.forward(idx, targets) → (logits, loss)`. Hyper-connections need two residual sites per layer
(attention and MLP), each with its own static matrix, dynamic projections, and norm — and they operate on
the `n`-stream tensor, not the single `x`. So I keep the `Block` as the vanilla container of
`ln_1, attn, ln_2, mlp` but do not call its `forward`; instead, as Block AttnRes already did, I drive the
sublayers directly from the `GPT.forward` loop. In `GPT.__init__` I build `2·n_layer` `HyperConnection`
modules (one per site, the static one-hot indexed by site id `k mod n`), set the expansion rate `n`, and
that is the only structural addition. In the forward loop I lift the embedding `x` into `n` copies (`H` of
shape `(B, T, n, D)`), then for each layer run the attention site — the width connection gives me the layer
input in row 0 and the carried streams in the rest, I norm row 0 with `ln_1`, run `attn`, and the depth
connection writes the output back via `β` plus the carry — then the same for the MLP site with `ln_2` and
`mlp`. After all layers I sum the `n` streams into one vector and pass it to `ln_f` and the head. In
`configure_optimizers` the new parameters (static matrices, dynamic projections, scales, site norms) are
gains, not weight matrices, so — following the pattern established earlier for leveraged routing
parameters (Block AttnRes's `0.1×` query group) — I route the hyper-connection parameters into their own
no-decay group, because weight decay would just pull them back toward the Pre-Norm init I deliberately
chose. The LR schedule and `CONFIG_OVERRIDES` stay default. The full scaffold module is in the answer.

So the delta from the strongest prior step is precise, and it is the move I have not yet made: Block
AttnRes kept one residual stream and got smarter about *attending over its coarsened history*;
hyper-connections keep that per-token dynamic routing but apply it to a fundamentally richer object — `n`
parallel streams with a full static-plus-dynamic `(n+1)×(n+1)` routing matrix at *every* sublayer,
fine-grained where Block AttnRes was coarse, and breaking the gradient-vs-collapse seesaw at its root
instead of relieving it at 5 seams. The number to beat is 2.2544 validation loss (WikiText-2 41.82, LAMBADA
64.32, ARC-Easy 55.01, HellaSwag 34.05). The mechanism predicts hyper-connections should clear it: they add
the fine-grained dynamic depth routing Block AttnRes gave up to fit the budget, they start at exactly the
Pre-Norm operating point (so they never pay the start-from-a-different-point tax the ARC regression hints
Block AttnRes paid), and they directly target the adjacent-deep-layer feature collapse the per-block
routing leaves untouched inside each block. The falsifiable test of the whole story is one measurement: if
hyper-connections are working, the cosine similarity between adjacent layers' features should *drop*
relative to Pre-Norm — the direct readout of "deep layers made distinct," which distinguishes a real
seesaw-break from a mere reshuffle of the loss. On the headline metrics I would expect the
deep-layer-sensitive ones to move most, LAMBADA and WikiText-2 below their current values with validation
loss into the low-2.25s, and ARC-Easy possibly recovering toward its 55.85 high since a per-copy identity
highway can carry the clean token identity learned-scaling's `x0` route did. Two honest risks: at `n = 2`
the seesaw breaks but the capacity gain is smaller than `n = 4` would give, so the win over Block AttnRes
may be narrow (and `n = 1` would be worse than vanilla, so the rate must stay above 1); and
hyper-connections' reported advantage grows over long training, so at 13.5k steps the measured margin may
be smaller than large-scale reports — the test is a clean beat of 2.2544 plus a bent similarity curve, not
a large convergence speedup at this scale.
