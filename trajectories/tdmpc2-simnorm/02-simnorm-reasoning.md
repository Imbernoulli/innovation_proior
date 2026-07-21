The RMSNorm run told me exactly what I half-expected, and in the one number that discriminates. On
walker-walk all three seeds sat at the top, mean 976.92, and on cartpole-swingup the same, mean 873.39 —
both tight, both saturated. Those two tasks do not stress the latent: bounding the magnitude is sufficient
and the lack of structure costs nothing. The honest test was cheetah-run, and there RMSNorm landed at mean
680.96 with seeds {712.15, 699.55, 631.18} — not a lucky-seed story but a uniformly *low* band; even the
best seed (712) is below where I want to be. So this is not noise I can average away. It is a
representational ceiling: group-wise RMSNorm bounds each group's spread and nothing else, and on the one
task whose running gait needs fine latent structure, a bounded-but-dense code gives the value head too
little to read.

The arithmetic backs the claim that cheetah is the discriminating column, not a hunch. On walker-walk the
seeds spread 5.48 points around 976.92, a coefficient of variation near 0.2%; on cartpole-swingup, 15.12
around 873.39, still under 2%. On cheetah-run the seeds deviate by +31.19, +18.59, −49.78, a standard
deviation of `√((31.19² + 18.59² + 49.78²)/3) ≈ 35.6`, CV around 5.2% — more than twenty times walker's,
and the best-to-worst gap of 81 points is larger than walker-walk's entire range. Two conclusions: the
easy tasks are genuinely saturated, so nothing I do to the latent geometry can move them; and on
cheetah-run the variance is the representation failing to reliably support the value loop, so the whole
informational content of this experiment lives in that one column. That is where the next rung's bet goes.

The fix has to supply exactly what RMSNorm lacked and no more. RMSNorm divides each group by a single
scalar and applies a gain, so every coordinate is rescaled identically: no *competition* between the eight
entries, no pressure to prioritize a few directions, the latent stays dense — and the gain can re-inflate
scale, so even the bound is soft. What a value/reward readout likes is a *sparse, structured* code where
each group commits to a few active directions, so the readout reads a clean combinatorial pattern rather
than a dense blob. So the question is: can I keep RMSNorm's boundedness but *add* a within-group
competition that biases the latent toward sparse structure — without an L1 penalty I would tune per task?

The obvious fix is exactly that L1 penalty, and I reject it with a reason. Adding `λ‖z‖₁` to the objective
introduces a knob `λ` that trades reward against sparsity and almost certainly wants a *different* value on
cheetah-run than on the saturated tasks — but the whole point is a normalization stable across DMControl
*without* per-task tuning, so a task-dependent `λ` defeats the premise. It also fights the bound rather
than cooperating: RMSNorm pins the root-mean-square to one, so an L1 push-down is opposed by the
normalization pushing scale back up, and the two settle at a compromise I do not control. And it only
*coaxes* small coordinates while guaranteeing nothing about the norm. So the move is not "bound, then
penalize toward sparsity"; it is a single map that bounds *and* forces the competition intrinsically.

The constraint should fall out of what a good latent geometry is. Sparse, overcomplete codes — more
components than the input dimension, most inactive — are stable under noise and are what downstream linear
readouts like, and my value and reward heads are essentially linear readouts. So I want the latent bounded
*and* sparse. The cleanest realization is a vector of categoricals: split the latent into groups and let
each group commit to one active entry out of eight — exactly the 16-groups-of-8 partition I already have,
but instead of dividing each group by its RMS, let each group behave like a soft one-hot. A hard one-hot is
bounded (norm 1), maximally sparse within the group, and 16 independent groups form an overcomplete code
with up to `8¹⁶` configurations. The geometry is right, but the hard one-hot is an argmax — the same
non-differentiable machinery (straight-through, codebook, dead codes) I set aside for the recurrent world
model. So I want the *geometry* of a vector-of-categoricals but the soft, differentiable version.

The smooth relaxation of argmax is softmax. Take each group of eight raw values, softmax *within the
group*, and get eight nonnegatives summing to one — a point on the 7-simplex. Do that for all 16 groups and
concatenate. That is the whole method: reshape the 128-vector to `(*batch, 16, 8)`, softmax over the last
axis, reshape back. It reuses the exact partition of the RMSNorm fill and changes only what happens inside
each group — from "divide by the RMS scalar" to "softmax," the `rms = …; x = x/rms * weight` pair
collapsing to `x = F.softmax(x, dim=-1)`, the gain parameters vanishing. A one-line change to the
activation is the entire intervention, so any difference in the cheetah-run number is attributable to
softmax-vs-rescale and nothing else.

Does it still bound the latent? Within a group, softmax gives entries in `(0,1)` summing to one, so
`‖gᵢ‖₁ = 1` exactly, and since each entry is in `[0,1]`, `‖gᵢ‖₂² = Σ gᵢⱼ² ≤ Σ gᵢⱼ = 1`, so `‖gᵢ‖₂ ≤ 1`.
Over all 16 groups, `‖z‖₁ = 16` and `‖z‖₂ ≤ 4`: a hard, parameter-free upper bound depending on nothing
but the number of groups. This is *stronger* than RMSNorm gave me — there the learnable gain could
re-inflate the pinned spread; here there is no gain, the bound is structural and exact, with equality only
in the impossible fully-one-hot limit. So the value-target triangle is not merely damped but severed at a
fixed radius: no matter how the upstream weights evolve, the latent the value target reads can never
exceed norm four, and there is no parameter in the layer to move that ceiling. I keep everything the
previous rung bought on magnitude, with a tighter guarantee, before even reaching the structure.

Now the structure, the reason I expect to clear the cheetah-run gap. Softmax inside a group is a zero-sum
competition: the outputs sum to one and are nonnegative, so raising one component forces the others to
collectively give up the same amount. To drive the consistency and value losses down the network cannot
make everything large; it has to *prioritize* — push mass onto a few entries per group at the expense of
the rest — so the learned groups drift toward approximately one-hot, sparse-but-soft, with no L1 term and
no hard argmax. That is exactly the competition RMSNorm's single scalar could not provide, and it is the
sparse overcomplete code the value head wants, emergent from the geometry rather than imposed.

Why *groups* and not one softmax over all 128 entries? A single softmax forces the whole vector to sum to
one, so each coordinate averages `1/128`, only one can be appreciably active at a time, and the latent
carries on the order of `log₂ 128 ≈ 7` bits — a brutal bottleneck for a latent that must encode the full
state. Splitting into 16 independent simplices lets each group independently choose which of its eight
entries to activate: 16 near-independent decisions, up to `8¹⁶` configurations, roughly `16·log₂ 8 = 48`
bits, while every group is still individually bounded. Nearly a sevenfold increase in addressable capacity
at zero cost to boundedness, because a per-group simplex bounds each group whether or not the others are
constrained. The group structure is what lets me bound the latent without crushing it to a single
direction — the same structure RMSNorm used, now carrying a sparsity bias instead of just a rescale.

The group size trades the two off, and although the harness fixes `cfg.simnorm_dim = 8` I want to know 8
is a sensible place to sit. With group size `d` the L2 bound is `‖z‖₂ ≤ √(128/d)` and the code carries
about `(128/d)·log₂ d` bits: `d = 4` gives 64 bits but a loose bound `√32 ≈ 5.66` and only a binary choice
per group; `d = 16` gives a tight bound `√8 ≈ 2.83` but a third less capacity and only eight coarse
groups; `d = 8` sits in the middle — bound 4, 48 bits, a genuine eight-way choice with room for real
within-group sparsity across sixteen independent decisions. So 8 keeps both the bound and the capacity in
a healthy range, and it is the partition the RMSNorm rung used, keeping this a clean one-variable swap.

There is a temperature hiding in the softmax, `gᵢ = softmax(zᵢ/τ)`. As `τ → 0` it sharpens to a hard
argmax (one-hot groups, the discrete code recovered continuously); as `τ → ∞` every group washes out to
uniform `1/8`, carrying no information. Very small `τ` is risky here because the softmax Jacobian
`J = diag(g) − ggᵀ` collapses toward zero near one-hot — the active coordinate's derivative `gₖ(1−gₖ) → 0`
— so almost no gradient reaches the encoder and dynamics, and gradient flow through the dynamics over a
multi-step rollout is the whole ballgame. I have no reason to push toward the uniform end either, and the
premise is stability without per-task tuning, so I take `τ = 1`, a plain softmax: a healthy sparsity bias
with the Jacobian well away from its degenerate limit, which is the scaffold default's exact form.

Placement follows from the consistency loss comparing `d(z,a)` against `sg(h(s'))`: for that comparison to
be meaningful prediction and target must live in the same space, so the normalization must be the final
activation of *both* nets — the same requirement that held for RMSNorm, and the task already wires the same
`CustomSimNorm` instance into both stacks. Normalizing only the encoder would let the dynamics emit an
unbounded next-latent and sneak the runaway loop back in through the half that stayed free.

So the bar against the previous rung is clear. I expect walker-walk and cartpole-swingup to stay
saturated, indistinguishable from RMSNorm — those tasks do not see the difference. The bet rides on
cheetah-run: the within-group competition supplies precisely the sparse structure RMSNorm's scalar rescale
could not, so I expect the cheetah-run mean to clear RMSNorm's 680.96 by a clear margin. If it does not,
the gap was not about structure and I would have to look elsewhere. I would not be surprised by one softer
seed even if the mean clears the bar — the simplex code still has to be learned through a bootstrapped
value loop, and RMSNorm's own 35-point seed spread says this column is noisy by nature. The full scaffold
module — the reshape, the within-group softmax at `τ=1`, the reshape back — is in the answer.
