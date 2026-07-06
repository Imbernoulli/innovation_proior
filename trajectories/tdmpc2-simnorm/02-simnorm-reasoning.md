The RMSNorm run told me exactly what I half-expected, and it told me in the one number that
discriminates. On walker-walk all three seeds sat at the top, mean 976.92, and on cartpole-swingup the
same, mean 873.39 — both tight, both saturated. Those two tasks do not stress the latent: bounding the
magnitude is sufficient and the lack of structure costs nothing, exactly as I predicted. The honest test
was cheetah-run, and there RMSNorm landed at mean 680.96 with seeds {712.15, 699.55, 631.18} — not a
high-variance lucky-seed story, but a uniformly *low* band; even the best seed (712) is below where I
want to be, and the worst (631) drags. So this is not noise I can average away. It is a representational
ceiling: group-wise RMSNorm bounds the spread of each group and nothing else, and on the one task whose
running gait is rich enough to need fine latent structure, a bounded-but-dense code gives the value head
too little to read. The diagnosis I wrote going in held — bounding the magnitude is necessary but not
sufficient — and now the cheetah-run gap puts a price on the missing ingredient: roughly the difference
between 681 and whatever a *structured* latent can reach. The next move has to add structure, not just
re-bound the scale.

Let me read the three feedback tables more quantitatively before I trust that story, because "cheetah is
the discriminating task" should be a measured fact, not a hunch. On walker-walk the seeds are 979.32,
973.84, 977.61 — a spread of 5.48 points around 976.92, a coefficient of variation near 0.2%. On
cartpole-swingup, 879.86, 864.74, 875.58 — spread 15.12 around 873.39, still under 2%. On cheetah-run the
seeds deviate from the 680.96 mean by +31.19, +18.59, −49.78; the standard deviation is
`√((31.19² + 18.59² + 49.78²)/3) ≈ √1266 ≈ 35.6`, a coefficient of variation around 5.2% — more than
twenty times walker's. And the raw spread is telling on its own: cheetah's best-to-worst gap of 81 points
is larger than walker-walk's *entire* range many times over. Two conclusions fall out of that arithmetic.
First, the two easy tasks are genuinely saturated — a fraction of a percent of seed-to-seed wobble means
the ceiling is the task, not the representation, so nothing I do to the latent geometry can move them and
I should not expect it to. Second, cheetah-run is where the latent is the bottleneck: the variance there
is not lucky seeds scattering around a good mean, it is the representation failing to reliably support the
value loop, so the whole informational content of this experiment lives in that one column. That is where
I have to spend the next rung's bet.

Let me be precise about what RMSNorm failed to provide, because the fix has to provide exactly that and
no more. RMSNorm divides each group by a single scalar — its root-mean-square — and applies a gain. A
single scalar per group means every coordinate is rescaled identically: there is no *competition*
between the eight entries of a group, no pressure to prioritize a few directions over the rest, so the
learned latent stays dense, every coordinate generically nonzero. And the gain can re-inflate the scale
per coordinate, so even the bound is soft. What a value/reward readout actually likes — what makes
bootstrapped value learning across a dynamic gait stable — is a *sparse, structured* code: a
representation where each group commits to a few active directions, so the readout is reading a clean
combinatorial pattern rather than a noisy dense blob. RMSNorm has no mechanism to produce that. So the
question for this rung is: can I keep the boundedness I got from RMSNorm but *add* a within-group
competition that biases the latent toward sparse structure — without bolting on an L1 penalty I would
have to weight per task?

That last clause is worth pausing on, because the obvious fix is exactly the L1 penalty I just waved off,
and I should reject it with a reason, not a preference. I could keep group-wise RMSNorm and add a term
`λ‖z‖₁` to the objective to push coordinates toward zero. Three things go wrong. It introduces a knob `λ`
that trades reward against sparsity and almost certainly wants a *different* value on cheetah-run than on
the two saturated tasks — and the entire point of this exercise is a normalization that is stable across
DMControl *without* per-task tuning, so a task-dependent `λ` defeats the premise. It also fights the bound
rather than cooperating with it: RMSNorm pins the root-mean-square to one, so an L1 penalty pushing mass
down is opposed by the normalization pushing the overall scale back up, and the two settle at some
compromise I do not control. And it is soft in the wrong way — an L1 penalty *encourages* small
coordinates but guarantees nothing about the latent's norm, so I would be paying a tuning cost to *coax*
a structure I would rather get *for free* from the geometry of the activation itself. So the move is not
"bound, then penalize toward sparsity"; it is to find a single map that bounds *and* forces the
competition intrinsically, with no extra loss term and no `λ`.

Let me think about what a *good* latent geometry would be and let the constraint fall out of it. There is
an old intuition I trust: sparse, overcomplete codes — more components than the input dimension, most of
them inactive — are stable under noise and are the representations that downstream linear readouts like.
My value and reward heads *are* essentially linear readouts on the latent. So I want the latent bounded
*and* sparse. The cleanest realization of "sparse and structured" is a vector of categoricals: split the
latent into groups and let each group commit to one active entry out of its eight. That is exactly the
group partition I already have from the RMSNorm fill — 16 groups of 8 — but instead of dividing each
group by its RMS, I want each group to behave like a soft one-hot. A hard one-hot is bounded (norm 1),
maximally sparse within the group, and 16 independent groups form an overcomplete code with up to `8¹⁶`
configurations. The geometry is right. But the hard one-hot is an argmax — non-differentiable — and
training through it would need a straight-through estimator and a codebook, dragging codebook collapse
and dead codes along. For a recurrent world model whose gradients must flow cleanly through the dynamics
net step after step, that fragile machinery is the wrong tool. So I want the *geometry* of a
vector-of-categoricals but the *soft, differentiable* version.

The smooth relaxation of argmax is softmax. If I take each group of eight raw values and apply a softmax
*within the group*, I get eight nonnegatives summing to one — a point on the 7-simplex. Do that for all
16 groups and concatenate. That is the whole method: reshape the 128-vector to `(*batch, 16, 8)`,
softmax over the last axis, reshape back. It is the natural successor to the RMSNorm fill because it
reuses the *exact same partition* — the reshape to 16 groups of 8 is identical — and only changes what
happens inside each group: from "divide by the RMS scalar" to "softmax." The delta from the previous
rung is one operation, and that operation is precisely the within-group competition RMSNorm lacked. In
code it is one line moving — the `rms = …; x = x/rms * weight` pair collapses to `x = F.softmax(x,
dim=-1)`, the two learnable-gain parameters vanish, and everything around it (the reshape in, the reshape
out, the placement in both stacks) is untouched. A one-line change to the activation is the entire
intervention, which is what I want: it means any difference in the cheetah-run number is attributable to
softmax-vs-rescale and to nothing else.

Let me verify it solves the original problem first — does it bound the latent? Within a group, softmax
gives entries in `(0,1)` summing to one, so `‖gᵢ‖₁ = 1` exactly, and since each entry is in `[0,1]`,
`‖gᵢ‖₂² = Σ gᵢⱼ² ≤ Σ gᵢⱼ = 1`, so `‖gᵢ‖₂ ≤ 1`. Over all 16 groups, `‖z‖₁ = 16` exactly and
`‖z‖₂ ≤ 4`. A hard, parameter-free upper bound on the latent norm that depends on *nothing* about the
input or the task — only on the number of groups. This is *stronger* boundedness than RMSNorm gave me:
RMSNorm pinned the spread but the learnable gain could re-inflate it; here there is no gain, the bound is
structural and exact. So I keep everything the previous rung bought on the magnitude problem, with a
tighter guarantee, and I have not even gotten to the structure yet.

Let me put a number on that bound with a concrete group so I trust the inequality rather than just the
algebra. Take logits `[2, 1, 0, 0, 0, 0, 0, 0]`. Exponentiating: `e² = 7.389`, `e¹ = 2.718`, and six ones,
summing to `16.107`, so the softmax is `[0.459, 0.169, 0.062, 0.062, 0.062, 0.062, 0.062, 0.062]`. The
L1 is `1` by construction. The L2 squared is `0.459² + 0.169² + 6·0.062² = 0.211 + 0.029 + 0.023 =
0.262`, so `‖g‖₂ = 0.512 ≤ 1` — the inequality holds with room to spare, and it would tighten toward `1`
only as the group sharpens to one-hot. The Shannon entropy is
`H = −Σ pⱼ ln pⱼ = 0.358 + 0.300 + 6·0.173 ≈ 1.69` nats out of a maximum of `ln 8 = 2.079` — so even
these mild logits already sit below the uniform ceiling, at about 81% of maximum entropy, and any real
pressure from the loss will drive them further down. That is the knob I get to watch: `H(gᵢ) ∈ [0, ln 8]`,
zero for one-hot, `ln 8` for uniform, and the whole mechanism is a bias toward the low-entropy end.

And I should check the other end of that knob, the end the loss will actually push toward, to be sure the
competition really does produce sparsity rather than settling somewhere mushy. Sharpen the same group to
logits `[4, 0, 0, 0, 0, 0, 0, 0]`: `e⁴ = 54.6` against seven ones sums to `61.6`, so the softmax is
`[0.886, 0.0162, …]` — the entropy has fallen to `−0.886 ln 0.886 − 7·0.0162 ln 0.0162 ≈ 0.108 + 0.469 =
0.58` nats, barely a quarter of the `2.079` ceiling, and the L2 has risen to `√(0.886² + 7·0.0162²) =
√0.787 = 0.887`, climbing toward the one-hot value of `1`. So as the network raises one logit the group
does exactly what I want: entropy drops, mass concentrates, the L2 tightens up toward the bound. The
competition is real and monotone — sharpening a single coordinate simultaneously buys lower entropy and a
larger (but still bounded) norm — which is the signature of a code that will commit to a few active
directions under pressure rather than hedging across all eight. Nothing about this required a penalty; it
is the shape of the softmax simplex doing the work.

It is worth saying plainly how this closes the runaway loop I opened in the previous rung, because the
tighter bound is not just cosmetic. There the value-target triangle — larger latents → larger targets →
larger gradients → larger weights — was cut by pinning each group's root-mean-square to one, but the
learnable gain left a crack: a coordinate whose gain drifted up re-inflated its magnitude, so the leash
had slack. Here there is no gain at all. `‖z‖₂ ≤ 4` holds for every input, every task, every training
step, with equality only in the impossible fully-one-hot limit, and *nothing* the network can learn moves
that ceiling — there is no parameter in the layer to move it with. So the feedback triangle is not merely
damped, it is severed at a fixed radius: no matter how the upstream weights evolve, the latent the value
target reads can never exceed norm four. That is a strictly stronger guarantee than the soft, gain-leashed
bound of the RMS rescale, and it costs me nothing, since the same softmax that supplies the structure is
what supplies the hard bound.

Now the structure, which is the whole reason I expect to clear the cheetah-run gap. Softmax inside a
group is a zero-sum competition: the outputs sum to one and are all nonnegative, so to raise one
component the others must collectively give up the same amount. To drive the consistency and value
losses down, the network cannot make everything large; it has to *prioritize* — push mass onto a few
entries per group at the expense of the rest. So the learned groups drift toward approximately one-hot,
sparse-but-soft, with no L1 term and no hard argmax. This is exactly the competition RMSNorm structurally
could not provide: there a single scalar rescaled all eight entries together; here the softmax pits them
against each other. That is the sparse overcomplete code the value head wants, emergent from the geometry
rather than imposed — precisely the "add structure, don't just re-bound scale" the cheetah-run gap
demanded.

Why *groups* and not one big softmax over all 128 entries? This matters, and it is also why I keep the
16-of-8 partition rather than collapsing it. A single softmax over 128 entries forces the whole vector to
sum to one, so on average each coordinate is about `1/128`, vanishingly small, and only one coordinate
can be appreciably active at a time — the latent can pick out essentially one direction and carries on
the order of `log₂ 128 ≈ 7` bits. That is a brutal bottleneck for a world-model latent that must encode
the full state for dynamics, reward, and value. Splitting into 16 independent simplices fixes exactly
this: each group independently chooses which of its eight entries to activate, so I get 16 near-
independent decisions, up to `8¹⁶` configurations, roughly `16·log₂ 8 = 48` bits, while every group is
*still* individually bounded. The arithmetic is stark — 48 bits against 7, nearly a seven-fold increase
in the code's addressable capacity — and it comes at *zero* cost to the boundedness, because a per-group
simplex bounds each group whether or not the other fifteen are also constrained. Boundedness and
expressivity stop fighting. The group structure is not optional — it is what lets me bound the latent
without crushing it to a single direction, and it is the same structure RMSNorm used, now carrying a
sparsity bias instead of just a rescale.

The group size is the other lever the partition exposes, and even though the harness fixes
`cfg.simnorm_dim = 8`, I want to know that 8 is a sensible place to sit rather than an accident, because
it controls both terms of the boundedness-versus-capacity trade at once. With group size `d`, the latent
splits into `128/d` groups, the L2 bound is `‖z‖₂ ≤ √(128/d)`, and the code carries about
`(128/d)·log₂ d` bits. At `d = 4`: 32 groups, bound `√32 ≈ 5.66`, capacity `32·2 = 64` bits — more
addressable configurations, but a looser magnitude bound and only two options per group, so each simplex
is nearly binary and the sparsity "commit to a few of eight" degenerates to "pick one of two." At
`d = 16`: 8 groups, bound `√8 ≈ 2.83`, capacity `8·4 = 32` bits — a tighter bound but a third less
capacity and only eight groups, so the code is coarser. At `d = 8`: 16 groups, bound `4`, capacity `48`
bits — it sits in the middle of both curves, giving each group a genuine eight-way choice (enough room
for real within-group sparsity) while keeping sixteen independent decisions and a bound that is tight
enough to starve the runaway loop. So `d = 8` is not just the default I inherited; it is the group size
that keeps both the bound and the capacity in a healthy range rather than sacrificing one for the other,
and it is exactly the partition the RMSNorm rung used, which keeps this a clean one-variable swap.

There is a temperature knob hiding in the softmax, `gᵢ = softmax(zᵢ/τ)`, and I want to choose it on
purpose. As `τ → 0` the softmax sharpens to a hard argmax — each group becomes one-hot, recovering the
discrete vector-of-categoricals continuously and differentiably. As `τ → ∞` every logit washes out and
each group goes uniform, `1/8`, carrying no information and blocking gradient flow through the latent.
So `τ` dials between "hard discrete code" and "trivial uniform code," and the useful regime is between.
Very small `τ` is risky in *this* model for the same reason vanishing gradients are: the softmax Jacobian
is `J = diag(g) − ggᵀ`, and near one-hot `g → eₖ` every entry of `J` collapses toward zero — the active
coordinate has derivative `gₖ(1−gₖ) → 0` and every off-diagonal `−gⱼgₖ → 0` — so almost no gradient
reaches the encoder and dynamics. At the uniform point `g = 1/8` the Jacobian has diagonal
`(1/8)(1 − 1/8) = 7/64 ≈ 0.11` and healthy off-diagonals `−1/64`, gradients flowing freely. Gradient flow
through the dynamics over a multi-step rollout is the whole ballgame, so I do not want to sit near the
degenerate `τ→0` end. I also have no reason to push toward the uniform end, and the point of this exercise
is a representation stable *without* per-task tuning, so I take the neutral choice `τ = 1`, a plain
softmax: a healthy sparsity bias with the Jacobian well away from its degenerate limit. This is the
scaffold default's exact form, and it is the honest default.

Where does it go? The consistency loss compares `d(z,a)` against `sg(h(s'))`. For that comparison to be
meaningful the prediction and the target must live in the *same* space — same geometry, same bound. So
the simplicial normalization must be the final activation of *both* the dynamics net and the encoder. If
I normalized only the encoder, the dynamics could emit an unbounded next-latent and I would reintroduce
the exact magnitude drift, plus a geometry mismatch between predictor and target. Worse, the consistency
residual would then be comparing a bounded target against an unbounded prediction, so the runaway loop I
cut in the previous rung would sneak back in through the half that stayed free. The task already wires the
same `CustomSimNorm` instance into both stacks, so this is satisfied by construction — the same
requirement held for RMSNorm, and it holds identically here.

So the falsifiable bar against the previous rung is clear. On walker-walk and cartpole-swingup I expect
SimNorm to stay saturated, statistically indistinguishable from RMSNorm's 976.92 and 873.39 — those
tasks do not see the difference, and if SimNorm somehow *dropped* there it would mean the softmax is
hurting an easy case, which I do not expect. The whole bet rides on cheetah-run: I expect SimNorm to beat
RMSNorm's 680.96 mean by a clear margin, because the within-group competition supplies precisely the
sparse structure RMSNorm's scalar rescale could not. If the cheetah-run mean does *not* move above ~681,
my diagnosis was wrong — the gap would not have been about structure after all — and I would have to look
elsewhere. I also expect the per-seed band to behave like RMSNorm's: cheetah-run is the variable task, so
I would not be surprised by one softer seed even if the mean clears the bar, since the simplex code still
has to be learned through a bootstrapped value loop, and RMSNorm's own 35-point seed standard deviation
tells me this column is noisy by nature. The mechanism in one breath: RMSNorm's measured cheetah-run
ceiling is a *structure* problem — a scalar rescale gives a bounded but dense, competition-free latent →
so replace the within-group RMS divide with a within-group softmax, projecting each group onto a simplex
→ which keeps a tighter, parameter-free magnitude bound *and* adds the zero-sum competition that biases
the latent toward a sparse, overcomplete code the value head can read → expecting the easy tasks to stay
saturated and cheetah-run to clear 681. The full scaffold module — the reshape, the within-group softmax
at `τ=1`, the reshape back — is in the answer.
