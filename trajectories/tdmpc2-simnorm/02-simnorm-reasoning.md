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
rung is one operation, and that operation is precisely the within-group competition RMSNorm lacked.

Let me verify it solves the original problem first — does it bound the latent? Within a group, softmax
gives entries in `(0,1)` summing to one, so `‖gᵢ‖₁ = 1` exactly, and since each entry is in `[0,1]`,
`‖gᵢ‖₂² = Σ gᵢⱼ² ≤ Σ gᵢⱼ = 1`, so `‖gᵢ‖₂ ≤ 1`. Over all 16 groups, `‖z‖₁ = 16` exactly and
`‖z‖₂ ≤ 4`. A hard, parameter-free upper bound on the latent norm that depends on *nothing* about the
input or the task — only on the number of groups. This is *stronger* boundedness than RMSNorm gave me:
RMSNorm pinned the spread but the learnable gain could re-inflate it; here there is no gain, the bound is
structural and exact. So I keep everything the previous rung bought on the magnitude problem, with a
tighter guarantee, and I have not even gotten to the structure yet.

Now the structure, which is the whole reason I expect to clear the cheetah-run gap. Softmax inside a
group is a zero-sum competition: the outputs sum to one and are all nonnegative, so to raise one
component the others must collectively give up the same amount. To drive the consistency and value
losses down, the network cannot make everything large; it has to *prioritize* — push mass onto a few
entries per group at the expense of the rest. So the learned groups drift toward approximately one-hot,
sparse-but-soft, with no L1 term and no hard argmax. This is exactly the competition RMSNorm structurally
could not provide: there a single scalar rescaled all eight entries together; here the softmax pits them
against each other. I can even measure how sparse a group is by its entropy `H(gᵢ) ∈ [0, ln 8]` — zero
is one-hot, `ln 8` is uniform — and the competition biases it toward the low-entropy end. That is the
sparse overcomplete code the value head wants, emergent from the geometry rather than imposed.

Why *groups* and not one big softmax over all 128 entries? This matters, and it is also why I keep the
16-of-8 partition rather than collapsing it. A single softmax over 128 entries forces the whole vector to
sum to one, so on average each coordinate is about `1/128`, vanishingly small, and only one coordinate
can be appreciably active at a time — the latent can pick out essentially one direction and carries on
the order of `log₂ 128 ≈ 7` bits. That is a brutal bottleneck for a world-model latent that must encode
the full state for dynamics, reward, and value. Splitting into 16 independent simplices fixes exactly
this: each group independently chooses which of its eight entries to activate, so I get 16 near-
independent decisions, up to `8¹⁶` configurations, roughly `16·log₂ 8 = 48` bits, while every group is
*still* individually bounded. Boundedness and expressivity stop fighting. The group structure is not
optional — it is what lets me bound the latent without crushing it to a single direction, and it is the
same structure RMSNorm used, now carrying a sparsity bias instead of just a rescale.

There is a temperature knob hiding in the softmax, `gᵢ = softmax(zᵢ/τ)`, and I want to choose it on
purpose. As `τ → 0` the softmax sharpens to a hard argmax — each group becomes one-hot, recovering the
discrete vector-of-categoricals continuously and differentiably. As `τ → ∞` every logit washes out and
each group goes uniform, `1/8`, carrying no information and blocking gradient flow through the latent.
So `τ` dials between "hard discrete code" and "trivial uniform code," and the useful regime is between.
Very small `τ` is risky in *this* model for the same reason vanishing gradients are: near one-hot the
softmax Jacobian collapses and almost no gradient reaches the encoder and dynamics — and gradient flow
through the dynamics over a multi-step rollout is the whole ballgame. I have no reason to push toward
either extreme, and the point of this exercise is a representation stable *without* per-task tuning, so I
take the neutral choice `τ = 1`, a plain softmax: a healthy sparsity bias with gradients still flowing.
This is the scaffold default's exact form, and it is the honest default.

Where does it go? The consistency loss compares `d(z,a)` against `sg(h(s'))`. For that comparison to be
meaningful the prediction and the target must live in the *same* space — same geometry, same bound. So
the simplicial normalization must be the final activation of *both* the dynamics net and the encoder. If
I normalized only the encoder, the dynamics could emit an unbounded next-latent and I would reintroduce
the exact magnitude drift, plus a geometry mismatch between predictor and target. The task already wires
the same `CustomSimNorm` instance into both stacks, so this is satisfied by construction — the same
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
has to be learned through a bootstrapped value loop. The mechanism in one breath: RMSNorm's measured
cheetah-run ceiling is a *structure* problem — a scalar rescale gives a bounded but dense, competition-
free latent → so replace the within-group RMS divide with a within-group softmax, projecting each group
onto a simplex → which keeps a tighter, parameter-free magnitude bound *and* adds the zero-sum
competition that biases the latent toward a sparse, overcomplete code the value head can read →
expecting the easy tasks to stay saturated and cheetah-run to clear 681. The full scaffold module — the
reshape, the within-group softmax at `τ=1`, the reshape back — is in the answer.
