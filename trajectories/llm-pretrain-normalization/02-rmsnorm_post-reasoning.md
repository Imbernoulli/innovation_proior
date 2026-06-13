The parallel rung told me what I wanted to know, and it told me with one number: validation loss 2.3112,
against WikiText-2 perplexity 45.98 and LAMBADA 70.96, ARC-Easy 54.76 and HellaSwag 32.93, in the fastest
wall-clock of anything I will run (19,747s training, 406s eval). The run was stable to completion — no
blow-up, no divergence — so the RMSNorm half of the bundle is vindicated: dropping the mean-subtraction and
the bias cost me nothing catastrophic, exactly as I argued, and I will keep RMSNorm as the substrate norm for
every rung from here. But 2.3112 is the *worst* validation loss on the board so far, and I expected that:
this was the rung that paid the small-scale parallel tax. The two sublayers could not see each other within a
block, and at 355M that lost cross-talk shows up as loss left on the table. The perplexities tell the same
story consistently — WikiText-2 at 45.98 and LAMBADA at 70.96 are both on the high side — and the downstream
numbers (ARC-Easy 54.76, HellaSwag 32.93) are unremarkable. So the diagnosis is clean and it is *not* about
the normalization rule: the parallel wiring traded quality for speed, the speed came (this was the cheapest
rung), and the quality bill is the 2.3112. The signal I read from the fast-but-worst result is that I have
spare quality budget to spend, and the structural lever to spend it on is the wiring — undo the parallel
simplification, restore the sequential per-block cross-talk, and then go *further* than the default block by
spending structure on *more* normalization rather than less.

So the move is to keep the cheaper RMSNorm and stop simplifying the block — instead, ask whether adding
normalization back, in a smarter place, buys quality the parallel rung threw away. Here is the structural
problem the parallel rung made vivid by summing two un-normalized branch outputs straight into the residual:
in a pre-norm residual stream, nothing ever normalizes the *output* of a sublayer before it is added back.
`x ← x + Attn(LN(x))` normalizes the *input* to attention, but whatever magnitude attention's projection
produces is written into the residual stream raw. Across 24 blocks those raw writes accumulate, and the
residual-stream variance grows monotonically with depth — this is the well-known cost of pure pre-norm, the
reason it lands at a slightly higher final loss than a successfully-trained post-norm model. The parallel rung
made this worse, not better, because it summed two raw branch outputs per block with a non-recentering norm in
front and nothing behind. Pure post-norm, `x ← LN(x + Attn(x))`, is the opposite extreme: it normalizes
*after* the add, which controls the variance growth beautifully, but it puts the norm back on the main
residual path and breaks the clean identity shortcut that gave pre-norm its depth-stable gradients. I do not
want to give up the pre-norm gradient behavior — the substrate is 24 layers deep and I am relying on the
identity path staying clean — but I do want post-norm's variance control. The two desiderata are not actually
in conflict if I am willing to place *two* norms per sublayer instead of one.

That is the sandwich. Keep the pre-norm on the input to each sublayer — `LN_pre(x)` going into attention,
which preserves the well-conditioned gradient at initialization — but add a second norm on the sublayer's
*output*, applied before it is added to the residual: `x ← x + LN_post(Attn(LN_pre(x)))`, and identically for
the MLP, `x ← x + LN_post2(MLP(LN_pre2(x)))`. Read what each norm is now doing. `LN_pre` controls the
distribution the sublayer *sees*, which is what makes the gradient bounded and depth-independent — the
pre-norm property I refuse to lose. `LN_post` controls the magnitude of what the sublayer *contributes* to
the residual, so each block writes a quantity of controlled scale into the stream regardless of how large the
sublayer's internal projection grew during training. That directly attacks the monotonic variance growth: the
residual stream still adds a fresh contribution per block, but each contribution is normalized, so the growth
is bounded the way post-norm bounds it — without ever putting a norm on the main path between blocks. The
identity shortcut `x ← x + (normalized branch)` is intact; the gradient still flows straight down the residual
without passing through a normalization, so I keep the pre-norm depth stability *and* I buy back the post-norm
variance control. This is precisely the failure mode I just watched the parallel rung exaggerate — raw branch
outputs summed into the stream — turned around by normalizing the branch output before the add.

I want to be careful that this is the block this task actually builds, and not a different animal that shares
the name. "Post-LN" in the loose sense means `x ← LN(x + Attn(x))`, the norm on the main path after the
residual add. That is *not* what this rung does. This rung normalizes the *branch output before the add* and
keeps the residual path itself norm-free between blocks — two norms per sublayer (one in, one out), the
residual add sitting *between* them. That is the sandwich placement, the CogView-style arrangement (Ding et
al. 2021): `Pre-LN` for the gradient, an additional `Post-LN` on the sublayer output for the variance, with
the residual connection threaded between. The distinction matters for what I expect, because true post-norm
would risk the depth-stability I am keeping, whereas the sandwich is engineered to keep it. So when I write
the block it is four norms: `ln_pre1`/`ln_post1` around attention, `ln_pre2`/`ln_post2` around the MLP, each
an RMSNorm carried over from the first rung (gain only, no bias, one reduction). I am spending normalization,
not saving it — the exact opposite trade from the parallel rung — on the bet that controlled branch
contributions are worth more at 355M than the half-norm speed saving was.

Now I have to be honest about the cost and about why I expect this rung to land *between* the parallel floor
and a plain sequential RMSNorm block, not at the top. The sandwich runs four RMSNorms per block where the
plain sequential block runs two and the parallel rung ran one. So it is the *most expensive* rung in
normalization arithmetic — I should expect the slowest, or close to it, in wall-clock, a near-mirror of the
parallel rung's speed win. That is the price of the extra variance control. But there is a subtler concern
about whether the extra output norm actually helps *final loss* or just *stability*. The thing `LN_post`
gives me is a controlled-scale contribution per block; but the learned gain `γ` on `LN_post` can re-inflate
that contribution back toward whatever magnitude the optimizer wants, so the variance control is a
*soft* prior, not a hard cap — the network can learn to undo some of it. At 24 layers, which is deep enough
for variance growth to matter but not so deep that pre-norm is in trouble, the sandwich's benefit may be
modest: it should help relative to the parallel rung (which had *no* output control and lost cross-talk on
top), but whether it beats a plain sequential RMSNorm block — which already has the cross-talk back and the
clean pre-norm gradient, just without the output norm — is the open question this rung answers. My prior is
that the extra output norm is more of a stability/variance insurance than a final-loss lever at this depth,
so I expect the sandwich to *recover* most of the gap the parallel rung opened but not necessarily to surpass
the simplest sequential block. The reason I run it anyway is that I cannot know the sign of that effect
without the number, and the variance-growth argument is strong enough that it could go either way.

There is also an initialization interaction to think through, because the substrate's `1/√(2·n_layer)`
residual-scaling was tuned for the default block's *two* residual writes with *no* output norm. The sandwich
still writes to the residual twice per block (one normalized attention contribution, one normalized MLP
contribution), so the count of writes is unchanged and the `1/√(2·n_layer)` factor still matches. But the
`LN_post` gains start at 1, so at initialization each branch's contribution is normalized to unit RMS and
then scaled by the `c_proj` init *and* the `1/√(2·n_layer)` factor — which means the initial per-block
contribution is, if anything, *smaller* and better-controlled than the default block, not larger. So I do not
expect an init pathology; the sandwich starts conservative and lets the `LN_post` gains grow the
contributions if the optimizer wants them. That is the right direction for stability. It also means the early
training dynamics may be a touch slower to get moving (smaller initial residual writes), which the cosine
schedule and 4% warmup should absorb. I touch no `CONFIG_OVERRIDES`: the learning rate and schedule are still
the substrate's, and I have no diagnostic from the parallel rung that would justify moving them — the parallel
rung's problem was wiring, not optimization.

So this rung is a clean fill of the same two-region edit surface: keep the first rung's RMSNorm class
verbatim (it earned its place), and replace the `Block` with the sandwich — `ln_pre1`, attention, `ln_post1`;
`ln_pre2`, MLP, `ln_post2`; residual adds threaded between, both sublayers sequential again. The distilled
module and the literal scaffold code are in the answer.

The falsifiable expectations, stated against the parallel rung's actual numbers so I can read the verdict
directly. First and most confident: the sandwich should *beat* the parallel rung's 2.3112 validation loss,
because it restores the intra-block cross-talk the parallel rung gave up *and* adds output-variance control —
two improvements in the quality direction with no quality regression I can name. If it does not beat 2.3112,
the whole "spend structure on more normalization" thesis is wrong and the next rung should abandon block
restructuring entirely. Second: the perplexities should move down with the validation loss — WikiText-2 below
45.98 and LAMBADA below 70.96 — since they read the same language-modeling quality. Third, and this is the
one I am genuinely unsure of: I do *not* expect the sandwich to be the *best* rung overall. My prior is that
the extra output norm is variance insurance more than a final-loss lever at 24 layers, so I expect it to sit
*between* the parallel floor and a plain sequential RMSNorm block — recovering most of the parallel gap but
leaving a sliver for the simplest block to claim. If the sandwich instead beats a plain sequential block,
that would tell me output-variance control is a real final-loss lever even at this modest depth, and the next
move would be to push it harder. Fourth, the cost: the sandwich should be the *slowest* rung, near a mirror of
the parallel rung's speed, because four norms per block is the most normalization arithmetic on the ladder.

The causal chain in one breath: the parallel rung came in fastest but worst (val_loss 2.3112) because it
summed two un-normalized branch outputs and lost intra-block cross-talk at small scale → so undo the
simplification and spend structure the other way, on *more* normalization placed where pre-norm leaves a gap:
add a post-branch RMSNorm on each sublayer's output, applied *before* the residual add, keeping the pre-norm
input norm for the depth-stable gradient → this is the sandwich placement (CogView-style), which buys
post-norm's variance control without putting a norm on the main residual path, turning around exactly the
raw-branch-sum failure the parallel rung exaggerated → at four RMSNorms per block it is the most expensive
rung, so I expect it to beat the parallel rung's 2.3112 and lower the perplexities, but I am unsure it beats a
plain sequential RMSNorm block, because the output norm's learned gain makes the variance control a soft prior
that may be insurance more than a final-loss lever at 24 layers.
