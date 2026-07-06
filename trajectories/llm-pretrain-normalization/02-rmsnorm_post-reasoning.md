The parallel rung came back and it told me what I wanted to know, with one number and a fistful of witnesses.
Validation loss 2.3112; WikiText-2 perplexity 45.98, LAMBADA 70.96; ARC-Easy 54.76, HellaSwag 32.93; and the
fastest wall-clock of anything I will run, 19,747s training against 406s eval. The first thing I read off this
is the half I most needed cleared: the run was *stable to completion*, no blow-up, no divergence, so dropping
the mean-subtraction and the bias cost me nothing catastrophic — exactly as I argued from the spread-control
angle — and I will carry RMSNorm as the substrate norm into every rung from here without re-litigating it. The
init check I did on paper held up: two writes per block, `1/√48` still matching, no variance pathology from
summing the two branches. So the RMSNorm half is vindicated and set aside as settled.

Before I act on the headline number I want to read the whole feedback row for what it says, because a clean
diagnosis is what tells me where to spend next. The `val_loss` `2.3112` is a FineWeb cross-entropy;
exponentiate it and the in-distribution perplexity is `e^2.3112 ≈ 10.09`, a different scale entirely from the
held-out `45.98` on WikiText-2 and `70.96` on LAMBADA — four to seven times higher — exactly as it should be,
since the model trained on FineWeb and is scored out-of-distribution on the other two. What matters is not the
absolute gap but that all three point the same way, so I read the perplexities as *direction* witnesses for
`val_loss`, not as commensurable numbers. The wall-clock is `19747`s of training against `406`s of eval: eval
is under `2.1%` of the run, so any speed comparison between rungs is dominated by the `19747`s training loop
and I can treat the lm-eval time as bookkeeping — per iteration that is `19747/12030 ≈ 1.64` s/iter, my first
fixed point on what a step of this model costs. The downstream row I read against chance, so I know which
columns even carry signal at 355M: ARC-Easy `54.76` sits well clear of its `~25%` random floor; HellaSwag
`32.93` is only about eight points over `25%`, a task a 355M model barely does; PIQA `64.42` is a healthy `14`
over the two-way `50%`; WinoGrande `50.2` is *at* chance and carries no information at this scale. So on later
rungs the downstream witnesses worth reading are ARC-Easy and PIQA; HellaSwag and WinoGrande are too close to
their floors to move meaningfully. And the discipline the setup forces on me: this is one seed, `42`, so I
have no run-to-run variance estimate at all — a `val_loss` move on the order of a thousandth could sit inside
the noise a second seed would reveal, and nothing here lets me bound it, so I will not over-read small gaps.

Now the number itself. 2.3112 is the *worst* validation loss on the board, and I predicted it would be —
this was the rung built to pay the small-scale parallel tax. Let me confirm the diagnosis reads consistently
across the witnesses before I act on it, because a clean diagnosis is what tells me *where* to spend next. The
perplexities are both on the high side (45.98, 70.96) and they agree in direction with the high `val_loss` —
in-distribution and held-out language modeling both saying "slightly worse language model," which is the
signature of a uniform quality shift, not some stranger interaction. The downstream numbers (54.76, 32.93) are
unremarkable, which is what I expected from the coarsest witnesses. And the wall-clock confirms the other half
of my prediction: this *was* the cheapest rung, and I had guessed the win would be modest — a few percent, the
memory traffic of one dropped norm, not a halving, because the fused input projection that carries the
large-scale critical-path win is frozen outside the edit surface. I cannot yet check the *magnitude* of that
speed claim against a second architecture, but the sign is right. So the whole picture is coherent: the
parallel wiring traded a small slice of quality for a small slice of speed; the speed came, the quality bill
is the 2.3112, and — crucially — the bill is *not* about the normalization rule, which trained stably and read
neutral. The signal I take from "fastest but worst" is that I have spare quality budget, and the lever that
spent it is the *wiring*. So I stop simplifying the block.

Here is the structural problem the parallel rung made vivid by summing two un-normalized branch outputs
straight into the residual, and it is the thing I want to attack. In a pre-norm residual stream, nothing ever
normalizes the *output* of a sublayer before it is added back. `x ← x + Attn(LN(x))` normalizes the *input*
to attention, but whatever magnitude attention's projection produces is written into the residual stream raw,
and the same for the MLP. Across 24 blocks those raw writes accumulate and the residual-stream variance grows
monotonically with depth — this is the well-known cost of pure pre-norm, the reason it lands at a slightly
higher final loss than a successfully-trained post-norm model: the later blocks are writing their contribution
onto a stream whose magnitude has drifted far from what the norm-then-project pipeline was implicitly
calibrated for, and the relative size of each new contribution shrinks as the stream grows. The parallel rung
made this *worse*, not better, because it summed two raw branch outputs per block with a non-recentering norm
in front and nothing behind. So the failure I measured and the failure this pre-norm family carries are the
same family of failure — un-normalized writes into a growing stream — and the parallel rung just turned the
dial up on it.

What are my actual options for spending the recovered budget? Let me walk the design space honestly, because
more than one move is defensible and I want to choose for a reason I can defend, not a reflex. Option one:
minimally undo the wiring — put the sublayers back in sequential order, restoring the intra-block cross-talk
the parallel rung lost, and stop there. That fixes the cross-talk half of the parallel failure but does
*nothing* about the un-normalized-write / variance-growth half, which the parallel rung just made vivid and
which pure pre-norm carries anyway; it is the conservative revert, and it leaves a known problem on the table.
Option two: go to pure post-LN, `x ← LN(x + Attn(x))`, the norm *after* the residual add. That controls the
variance growth beautifully — it renormalizes the whole stream at every block — but it puts the norm back on
the *main residual path* and destroys the clean identity shortcut that gives pre-norm its depth-stable,
warmup-free gradients. I am running a 24-layer stack and relying on that identity path; and the substrate's
warmup is only 4% of training, `0.04 × 12,030 ≈ 481` iterations, a schedule chosen for pre-norm's forgiving
early gradients, not for post-norm's fragile ones that historically need a longer, more careful warmup. To
make post-LN train cleanly here I would almost certainly have to reach into `CONFIG_OVERRIDES` and lengthen
warmup or drop the learning rate — and I have no diagnostic that justifies touching the optimizer; the
parallel rung's problem was wiring, not optimization. So pure post-LN is out: it fixes variance by sacrificing
the depth stability I refuse to give up, and it drags the optimizer into the change. Option three: get *both*
— keep pre-norm's input normalization for the gradient, and separately add a normalization on each sublayer's
*output* before it is added to the residual. That is the move that fixes the variance-growth half without
sacrificing the depth-stable-gradient half, and it *subsumes* option one, because it puts the sublayers back
in sequential order too. So I am not choosing between "revert" and "add control" — the third option is the
revert *plus* the control, and the only question is whether the extra output norm earns its keep.

There is a fourth move I owe a hearing before I commit, because it also attacks un-normalized writes and it is
not a norm at all: scale each branch output by a learnable per-channel diagonal initialized near zero — a
LayerScale-style gate, `x ← x + diag(λ)·Attn(LN(x))` with `λ ≈ 1e-4`. It starts every branch's contribution
near zero, so the residual barely grows early and the optimizer dials each branch up only as it earns its
place, which sounds like exactly the variance discipline I want. But walk what it actually guarantees and it is
the wrong tool for the failure I measured. A learned diagonal is a *fixed* multiplier once trained; it does not
renormalize. If a sublayer's `c_proj` grows large during training, `diag(λ)` scales that output by whatever
constant the optimizer happened to settle on, not back toward a controlled magnitude — so it puts *no bound* on
trained variance growth, which is precisely the quantity I am trying to bound. It is a gating/capacity change
(a new length-1024 vector per branch the optimizer will drive wherever lowers loss), not a scale-pinning
normalization, and its near-zero init trades the stream-growth problem for a slow-start problem I have no
schedule budget to absorb — the warmup is only `0.04 × 12030 ≈ 481` iterations, tuned for a network whose
branches are live from step one, not one that must first learn to switch them on. So LayerScale treats the
*symptom*, early stream growth, without the *mechanism* I want: a per-block operation that re-pins each
contribution no matter how the projection drifts. The post-branch norm gives that; the diagonal does not. I set
it aside.

That third option is the sandwich. Keep the pre-norm on each sublayer's input — `LN_pre(x)` going into
attention, which preserves the well-conditioned gradient at initialization — and add a second norm on the
sublayer's *output*, applied before it is added to the residual: `x ← x + LN_post(Attn(LN_pre(x)))`, and
identically for the MLP, `x ← x + LN_post2(MLP(LN_pre2(x)))`. Read what each of the four norms is now doing.
`LN_pre` controls the distribution the sublayer *sees*, which is what makes the gradient bounded and
depth-independent — the pre-norm property I refuse to lose. `LN_post` controls the magnitude of what the
sublayer *contributes* to the residual, so each block writes a quantity of controlled scale into the stream
regardless of how large the sublayer's internal projection has grown during training. That directly attacks
the monotonic variance growth: the residual stream still adds a fresh contribution per block, but each
contribution is normalized, so the growth is bounded the way post-norm bounds it — *without ever putting a
norm on the main path between blocks*. The identity shortcut `x ← x + (normalized branch)` is intact; the
gradient still flows straight down the residual without passing through a normalization, so I keep the
pre-norm depth stability *and* I buy back the post-norm variance control. This is precisely the failure mode I
just watched the parallel rung exaggerate — raw branch outputs summed into the stream — turned around by
normalizing the branch output before the add.

I want to be careful this is the block I actually mean and not a different animal that shares the name.
"Post-LN" in the loose sense means `x ← LN(x + Attn(x))`, the norm on the main path after the residual add —
that is option two, the one I rejected. This rung normalizes the *branch output before the add* and keeps the
residual path itself norm-free between blocks — two norms per sublayer, one in and one out, with the residual
add sitting *between* them. That is the sandwich placement (the CogView-style arrangement): pre-LN for the
gradient, an additional post-LN on the sublayer output for the variance, the residual connection threaded
between. Each of the four is an RMSNorm carried verbatim from the first rung — gain only, no bias, one
reduction — so I am spending normalization, not saving it, the exact opposite trade from the parallel rung, on
the bet that controlled branch contributions are worth more at 355M than the half-norm speed saving was.

Let me account for what the extra norms cost so I know what I am buying. In parameters: four gain vectors of
length 1024 per block, 24 blocks, is 98,304 gains — double the plain sequential block's 49,152, and quadruple
the parallel rung's 24,576. But 98,304 parameters against a 355M model is 0.028% of the model; the extra norms
add essentially *no capacity*. That matters for how I read the eventual number: this rung is not testing
"more parameters," it is testing *placement* — whether normalizing the branch output helps, holding capacity
fixed. In wall-clock: four norms per block is the most normalization arithmetic on the ladder, four times the
parallel rung's one. I only have a single datapoint to extrapolate from — the parallel rung's 19,747s at one
norm per block — so I cannot yet pin down the per-norm cost (that needs a second point at a different norm
count). But I established last rung that a norm is a small, memory-bound kernel, a few percent of a step, so
three extra norms per block over the parallel rung should push the training time up by something on the order
of several percent into the low-20,000s — the *slowest* rung, a near-mirror of the parallel rung's speed win
run in reverse. That is the price of the variance control, and I am consenting to it.

Now the initialization, and here I have to slow down because my first instinct is wrong and I only catch it by
doing the algebra. My reflex is: `LN_post` gains start at 1 and the `c_proj` init already scales the branch
output down by `1/√48`, so the initial contribution to the residual should be *small and conservative*. But
that is not what happens, because `LN_post` is applied *after* `c_proj` and `LN_post` is *scale-invariant*:
`LN_post(y) = y/RMS(y) · γ`, and if `c_proj`'s `1/√48` scales `y` down, it scales `RMS(y)` down by the same
factor, and the two cancel in the ratio. Let me verify that cancellation is exact and not approximate, because the
whole reversal hinges on it: for any scalar `α`, `LN_post(αy) = αy/RMS(αy)·γ = αy/(α·RMS(y))·γ = y/RMS(y)·γ =
LN_post(y)`, so the `α` divides straight out and the `1/√48 ≈ 0.1443` that `c_proj` applies is erased with it;
concretely, the raw branch output `y` sits at `RMS(y) ≈ 0.1443` relative to a unit-RMS input — the small
conservative write the substrate intends — and `LN_post` re-pins it to `RMS = |γ| = 1`. So `LN_post` *washes
out* the `c_proj` init entirely — at
initialization each branch contribution lands at RMS equal to the gain, which is 1. That is *not* small; it is
`√48 ≈ 6.9×` *larger* than the plain block's `1/√48`-scaled write. Summed over 48 writes, the residual stream
RMS at the top of the stack grows to roughly `√48` at init, a much steeper climb than the plain block's near-
doubling. So my reflex was backwards: the sandwich starts by writing *bigger*, not smaller, contributions. Is
that a pathology? I do not think so, and here is why: `LN_pre` re-normalizes each sublayer's *input* to unit
RMS at every block regardless of how large the residual has grown, so no sublayer ever sees the inflated
stream directly, and the final `ln_f` normalizes before the LM head, so the growing residual is absorbed
downstream. The block writes twice per block, 48 writes, so the `1/√48` bookkeeping on the residual writes is
untouched in *count*, even though `LN_post` overrides what that scaling would have done to each write's
magnitude. So the honest picture is: this is not variance control *at initialization* — at init it grows the
stream faster. The real value of `LN_post` is a bound on the *trained* growth: as `c_proj` learns and its
output magnitude drifts, `LN_post` re-pins each contribution back toward the gain, so the residual cannot
inflate the way plain pre-norm's can when its projections grow. That reframing matters for what I should
expect, because it tells me the benefit is about *late* training, not the start.

And it exposes the caveat I have to be honest about. `LN_post`'s bound is only as tight as its learned gain:
the gain starts at 1 but the optimizer is free to grow it, and if it does, the "controlled" contribution
re-inflates and the variance control leaks. So this is a *soft* prior, not a hard cap — the network can learn
to partially undo it wherever undoing it lowers the loss. At 24 layers — deep enough for variance growth to be
a real effect but not so deep that pure pre-norm is in trouble — the benefit of a soft variance bound may be
modest: it should clearly help relative to the parallel rung, which had *no* output control and lost cross-talk
on top, but whether the output norm earns its keep — whether it beats simply swapping in RMSNorm with no
output control added, cross-talk intact and the clean pre-norm gradient — is the genuinely open question this
rung answers. My
prior is that the output norm is more stability/variance insurance than a final-loss lever at this depth, so I
expect the sandwich to *recover* most of the gap the parallel rung opened but not necessarily to surpass what
the bare norm swap alone would give. I run it anyway because I cannot know the sign of that effect without the number,
and the variance-growth argument is strong enough that it could genuinely go either way. I touch no
`CONFIG_OVERRIDES`: the learning rate and schedule stay the substrate's, since the parallel rung's problem was
wiring, not optimization, and I have no diagnostic pointing at the optimizer.

So this rung is a clean fill of the same two-region surface: keep the first rung's RMSNorm class verbatim (it
earned its place), and replace the `Block` with the sandwich — `ln_pre1`, attention, `ln_post1`; `ln_pre2`,
MLP, `ln_post2`; the residual adds threaded between, both sublayers sequential again. The distilled module and
the literal scaffold code are in the answer.

The falsifiable expectations, stated against the parallel rung's actual numbers so I can read the verdict
directly. First and most confident: the sandwich should *beat* the parallel rung's 2.3112 validation loss,
because it restores the intra-block cross-talk the parallel rung gave up *and* adds output-variance control —
two improvements in the quality direction with no quality regression I can name. If it does not beat 2.3112,
the whole "spend structure on more normalization" thesis is wrong and the next move should abandon block
restructuring entirely. Second: the perplexities should move down with the validation loss — WikiText-2 below
45.98 and LAMBADA below 70.96 — since they read the same language-modeling quality on held-out text, and I
would want all three to agree in direction. Third, and this is the one I am genuinely unsure of: I do *not*
expect the sandwich to be the *best* rung overall; my prior is that the extra output norm is insurance more
than a final-loss lever at 24 layers, so I expect it to sit *between* the parallel floor and what
the bare norm swap alone would give, recovering most of the parallel gap but leaving a sliver the added output
norm does not actually reclaim. If it instead clears the bar the bare norm swap would set, that tells me
output-variance control is a real final-loss lever even at this modest depth, and the next move would be to push it harder. Fourth, the cost:
the sandwich should be the *slowest* rung, into the low-20,000s, a near-mirror of the parallel rung's speed,
because four norms per block is the most normalization arithmetic on the ladder.

The causal chain in one breath: the parallel rung came in fastest but worst (val_loss 2.3112) because it
summed two un-normalized branch outputs and lost intra-block cross-talk at small scale, and the perplexities
agreed it was a uniform quality shift, not a norm-rule problem → so undo the simplification and spend the
recovered budget on the un-normalized-write half of the failure: keep pre-norm's input norm for the depth-
stable gradient and add a post-branch RMSNorm on each sublayer's output *before* the residual add, which
subsumes the sequential revert (bringing cross-talk back) and adds variance control → this is the sandwich
placement, which buys post-norm's variance control without a norm on the main path, unlike the pure-post-LN
option I rejected because it breaks the identity gradient and would drag the 481-iter warmup schedule into the
change → the four norms add negligible capacity (0.028% of the model) so this tests placement not parameters,
and cost the most wall-clock (low-20,000s, slowest); the init algebra corrects my reflex — `LN_post` is
scale-invariant so it washes out the `c_proj` `1/√48` init and at initialization writes *larger*, unit-RMS
contributions, so the value of `LN_post` is a soft bound on *trained* growth, not small init writes → so I
expect it to beat the parallel rung's 2.3112 and lower the perplexities, but I am unsure it beats what the
bare norm swap alone would give, because the output norm's learned gain makes the variance control a soft prior that
may be insurance more than a final-loss lever at 24 layers.
