The baseline is a GPT-2 trained with AdamW, and after tuning the learning rate and swapping in rotary
position embeddings it clears 3.28 FineWeb val loss in about half an hour on the eight H100s. That's
already a big cut from the 45-minute llm.c reference — call it 45 → ~31 minutes, roughly a 30% reduction —
and I want to be honest about where that reduction came from, because it tells me where the *next* one
won't. Almost all of it came from the schedule and the optimizer's learning rate: a better warmup, a cosine
warmdown, a higher peak LR. Those are the cheap levers, and the fact that they already bought 30% means
they're mostly spent — I can keep fiddling with the LR schedule and scrape another percent or two, but the
schedule lever has been pulled about as hard as it goes. So the obvious next lever is the optimizer itself,
not its hyperparameters. Most of the wallclock is the optimization: thousands of AdamW steps grinding the
loss down, and when I profile a step the forward and backward matmuls plus the AdamW update *are* the step;
there's no fat systems overhead left to trim. If I can make each step buy *more* loss reduction, I reach the
bar in fewer steps and the wallclock falls proportionally, because wallclock ≈ steps × time-per-step and I'm
not about to make the step slower. The question is whether AdamW is leaving something on the table.

Before I commit to redesigning the update rule, let me lay out what's actually on the menu, because "improve
the optimizer" has several branches and most of them are traps in a speedrun. One branch is: don't change
the optimizer, just push AdamW harder — bigger batch, more aggressive betas, tuned weight decay. But I've
argued the schedule/LR lever is largely tapped, and batch-size scaling trades memory and communication for a
worse loss-per-token, which is the wrong direction when the token budget is what I'm trying to shrink. A
second branch is the genuinely second-order family — Shampoo, K-FAC, full-matrix Adagrad. These are actually
right in *spirit*: unlike AdamW they do see a weight matrix's structure, because they build Kronecker-factored
preconditioners out of the Gram matrices GGᵀ and GᵀG and precondition the gradient with (roughly) their
inverse fourth roots. That's the thing AdamW can't do. But the canonical implementations pay a brutal price:
maintaining an m×m and an n×n factor per matrix, and periodically taking an eigendecomposition or matrix
inverse-root of those factors. For a 3072-wide factor that's a 3072³ eigendecomposition, sequential, not
tensor-core-friendly, and it lands exactly on the wall I'm trying to avoid — if the preconditioner costs
more than the step it improves, the wallclock goes *up*. A third branch is the cheap sign-based methods, Lion
and signSGD — but those are per-coordinate again (sign of the momentum, entrywise), so they inherit the exact
blindness I'm about to indict AdamW for. So the shape of the problem is: I want what the second-order family
sees, at the price of the sign family. Hold that thought.

Let me look at what AdamW actually does to a weight *matrix*. A linear layer is a matrix W ∈ ℝ^{m×n};
its gradient G is also m×n. Adam keeps a running estimate of the gradient's first moment m and second
moment v *per scalar entry*, and the update for each entry is roughly −lr · mᵢⱼ / (√vᵢⱼ + ε). So Adam
treats the matrix as mn independent scalars: every entry is rescaled by its own gradient magnitude and
nothing else. That's the whole mechanism — a diagonal preconditioner, normalizing each coordinate so
that flat and steep directions get comparable step sizes. It's robust precisely because it commits to
nothing about the structure of W. And that robustness is exactly the blindness. The gradient of a matrix
is not a bag of scalars; it has *matrix* structure that the coordinate-wise view throws away. Write the
singular value decomposition G = U S Vᵀ. The directions in which the gradient is large (big singular
values) and the directions in which it is small (tiny singular values) are encoded in the singular
*vectors*, not in individual entries. A coordinate-wise rescale does nothing about a badly *conditioned*
G — if the gradient's energy is concentrated in one or two singular directions, Adam's per-entry √v
normalization doesn't fix that; the update still points mostly along the dominant singular directions, and
the model effectively learns along a low-rank slice each step while the small-singular-value directions get
starved. Concretely: suppose G has singular values (10, 1, 0.1). Adam's √v is a per-entry statistic that
doesn't line up with these singular directions at all — it can't rotate into the U, V frame — so the update
it produces still carries roughly the same 10:1:0.1 imbalance across singular directions. The subspace with
σ=0.1 gets a step a hundred times smaller than the σ=10 subspace, and stays under-trained. For a transformer,
where the interesting structure lives in how the matrix maps subspaces (which query subspace reads which key
subspace, etc.), that anisotropy is exactly what I'd want the optimizer to flatten out.

So the move that answers the design question is: instead of rescaling entries, rescale *singular values*.
Take the momentum-smoothed gradient G, compute G = U S Vᵀ, and replace S with the identity — step along
U Vᵀ. That is the "nearest orthogonal matrix" to G; it keeps the *directions* the gradient is pushing
(the singular vectors) but equalizes how hard it pushes in each of them. Every direction gets a step of
the same scale, so the small-singular-value directions are no longer starved and the dominant ones no
longer dominate. In the (10, 1, 0.1) example, all three singular directions come out at 1 — the σ=0.1
subspace now gets the *same* step as the σ=10 subspace, which is precisely the anisotropy I wanted to kill.
Geometrically I'm replacing "steepest descent in the Euclidean entrywise metric" (Adam-ish) with a step
that is uniform across the singular spectrum — a spectral, condition-number-blind update. And notice this
is the same object the second-order family is groping toward: U Vᵀ = G (GᵀG)^{−1/2} is exactly the
whitened/preconditioned gradient, the polar factor. The difference is that I don't need the full inverse
fourth root Shampoo maintains; I need only the zeroth power, the orthogonal factor, and that turns out to
be much cheaper to approximate. Stack standard SGD momentum in front of it so the gradient I orthogonalize
is a smoothed estimate, not a single noisy minibatch gradient — orthogonalizing raw minibatch noise would
just equalize the scale of the noise, which is not what I want.

The wall, as always, is cost. An SVD of a 768×3072 matrix, every layer, every step, on a GPU, is brutal —
SVD is sequential, dislikes bf16, and would erase whatever per-step savings the better direction buys. This
is the same wall that sinks Shampoo, and if I hit it I've lost the race no matter how good the direction is.
Put a number on the stakes: the body of this model is twelve blocks, each with four attention matrices and
two MLP matrices; per block that's 4·768² + 768·3072 + 3072·768 ≈ 2.36M + 2.36M + 2.36M ≈ 7.08M parameters,
so ≈ 85M across the twelve blocks — a matrix operation I'd do on every one of them, every step. If that
operation is a full SVD I've added a fat sequential kernel to a step that currently runs in a couple hundred
milliseconds. I need the orthogonal factor U Vᵀ *without* an SVD.

What do I actually need? Given G, I want the matrix that has the same singular vectors but all singular
values equal to one — i.e. G (GᵀG)^{−1/2}, the "matrix sign" / zeroth-power / polar factor. And I don't
need it exactly. I need it cheaply, in bf16, on a GPU, accurate *enough* that the update direction is roughly
spectrum-flat — the argument above works whether the small singular values come out at exactly 1 or merely
"not starved." There's a classic tool for applying a scalar function to the singular values of a matrix using
only matrix multiplies: a polynomial iteration. If p(x) is an odd polynomial that drives the interval (0, 1]
toward 1, then applying p to the matrix — built only from G, GGᵀ, and products — applies p to each singular
value while leaving the singular vectors fixed. That last fact is the crux: if G = U S Vᵀ, then
G Gᵀ G = U S Vᵀ · V S Uᵀ · U S Vᵀ = U S³ Vᵀ, and likewise (GGᵀ)² G = U S⁵ Vᵀ,
so any odd polynomial a·X + b·XXᵀX + c·(XXᵀ)²X equals U (a S + b S³ + c S⁵) Vᵀ — the U and V ride through
untouched and the polynomial acts on the diagonal S entry-by-entry. So a matmul-only iteration is exactly a
way to apply a scalar function σ ↦ a σ + b σ³ + c σ⁵ to the whole singular spectrum at once. Newton–Schulz
is the canonical such iteration for the inverse square root / sign function. Matrix multiplies are exactly
what H100 tensor cores are fastest at, and they run fine in bf16. So I trade an expensive exact SVD for a
handful of cheap matmuls that give an *approximate* orthogonalization — the second-order family's insight at
the sign family's price, which is what I said I wanted.

Concretely: first normalize so the top singular value is ≤ 1 (divide G by its norm), which puts every
singular value into (0, 1] where the iteration is well-behaved. Then iterate

  X ← a·X + b·(X Xᵀ X) + c·(X Xᵀ)² X

with fixed scalar coefficients (a, b, c). Each step is a degree-5 (quintic) polynomial in the singular
values: σ ↦ a·σ + b·σ³ + c·σ⁵. I want this polynomial to pull the whole interval (0, 1] up toward 1 in
as few steps as possible. The naive choice is the coefficients that make the iteration *converge* to the
exact sign function — the textbook (a, b, c) = (1.5, −0.5, 0), which has p(1) = 1 exactly and slope p′(0) =
1.5. But I don't care about exact convergence — I care about getting close in ~5 steps. So choose (a, b, c)
to maximize the *slope at zero*, p′(0) = a: the steeper the slope, the faster the tiny singular values (the
starved directions) get yanked up toward 1 in the first couple of steps. Pushing the slope at zero even past
the point where the iteration stops converging to exactly 1 everywhere is fine — it overshoots and leaves
the singular values not at 1 but scattered around it. The coefficients that do this come out to roughly
(a, b, c) = (3.4445, −4.7750, 2.0315). Slope at zero 3.4445 — more than double the safe iteration's 1.5.

Run this on a couple of singular values by hand, to see both that it pulls starved directions up fast and
how badly it overshoots. Take a
starved singular value σ = 0.1. One step: p(0.1) = 3.4445(0.1) − 4.7750(0.001) + 2.0315(1e-5) = 0.34445 −
0.004775 + 0.00002 ≈ 0.340. So a value at 0.1 is pulled up to 0.34 in a single step — a 3.4× amplification,
which is just the slope-at-zero doing its job. Second step: p(0.340) = 3.4445(0.340) − 4.7750(0.0393) +
2.0315(0.00454) ≈ 1.1711 − 0.1877 + 0.0092 ≈ 0.993. So 0.1 → 0.34 → 0.99 in *two* steps — the tiny direction
is essentially rescued. Now compare the safe (1.5, −0.5, 0) iteration on the same start: 0.1 → 0.1495 →
0.2226 → 0.3284 → 0.4805 → 0.6853 → 0.9457 → … it takes about seven steps to climb from 0.1 to ≈1, because
its per-step amplification near zero is only 1.5×. That's the whole justification for over-steepening the
slope: it's the difference between a 5-step budget and a 7+-step budget, and per-step matmul count is what
I'm paying for.

Now the price of the aggressive coefficients — the overshoot — traced on a value that's already near the
top. p(0.993) = 3.4445(0.993) − 4.7750(0.9791) + 2.0315(0.9655) ≈ 3.420 − 4.675 + 1.961 ≈ 0.706. So a value
near 1 gets knocked *down* to 0.71. Next: p(0.706) ≈ 2.432 − 1.680 + 0.356 ≈ 1.107, back up past 1. And
p(1.107) ≈ 3.813 − 6.478 + 3.378 ≈ 0.713. So the upper end of the spectrum doesn't settle at 1 at all — it
oscillates in a band roughly between 0.70 and 1.11. (I can see why: the map's fixed points solve a σ + b σ³
+ c σ⁵ = σ, i.e. 2.0315 u² − 4.775 u + 2.4445 = 0 with u = σ², giving σ ≈ 0.868 and σ ≈ 1.264, neither of
them 1, with p(1) = 3.4445 − 4.775 + 2.0315 = 0.701 sitting between them.) So after five iterations the
singular spectrum has been squashed from its original spread across (0, 1] into a tight band scattered
around 1 — flat *enough* — rather than pinned exactly at 1. That is precisely the US′Vᵀ with S′ ≈ Uniform-ish
about 1 that I'm willing to accept: my whole argument only needed "no starved directions, no dominant
directions," and a band of [0.7, 1.1] delivers that. Exact orthogonality was never the point; equalized
scale was, and the trace confirms I get it in five steps.

A couple of details fall out. The iteration as written needs G to be "tall-ish" for the X Xᵀ products to
be the smaller of the two Gram matrices; if G has more columns than rows I transpose it first and
transpose back at the end, so I always multiply the smaller Gram matrix. (For a 768×3072 MLP matrix, XXᵀ is
768×768 rather than 3072×3072 after the transpose — a 16× smaller Gram matrix to form and multiply, which is
exactly the kind of constant factor that decides whether "nearly free" is true.) And after orthogonalizing,
the update U Vᵀ has all singular values ≈ 1, so its overall scale is fixed regardless of the original
gradient magnitude — I need to put a sensible scale back. I set the update so its RMS entry is order one:
a matrix with orthonormal-ish columns and unit singular values has squared-entry-mean ≈ 1/max(m, n), so
multiplying by max(m, n)^{1/2} restores update.square().mean() ≈ 1. (For a fused QKV block, where three
n×n matrices are stacked into a 3n×n parameter, I split it and orthogonalize each n×n piece separately —
otherwise the iteration would try to orthogonalize the *stack* as one 3n×n matrix, mixing the Q, K, V row
spaces, which is meaningless — then scale by n^{1/2}.) Then the actual parameter step is p ← p − lr · scale
· (orthogonalized momentum).

Which parameters get this? The orthogonalization only makes sense for 2D matrices — the hidden weight
matrices of the transformer body, the attention and MLP projections. It is the wrong tool for the
embedding table and the final classifier (an embedding is a lookup whose rows are updated one-per-token,
with no cross-row subspace structure the way a projection has, and orthogonalizing the head across the
50304-way vocabulary axis conflates directions that shouldn't be equalized) and meaningless for 1D things
like biases and norm gains (a vector has no nontrivial singular structure — its "orthogonalization" is
just its sign). So this is not a drop-in replacement for the whole optimizer: I run it only on the
transformer block matrices — that ≈85M of the ≈124M total parameters — and keep AdamW for the
embedding/head, the remaining ≈39M. Two optimizers, side by side, each stepping the parameters it's suited
to. I'll name the new one after what it does — momentum, orthogonalized by Newton–Schulz: Muon.

Now the arithmetic behind the "nearly free" claim. One Newton–Schulz
step on an n×n matrix is three matmuls (XXᵀ, then A@X, then A@B), ≈ 3n³ multiply-adds; for n = 768 that's
≈ 3·4.5e8 ≈ 1.4e9 MACs, and five steps ≈ 7e9. Across the body's ≈14 matrices per model that's on the order
of 1e11–1e12 FLOPs of orthogonalization per optimizer step. An H100 does bf16 matmul at ~10^15 FLOP/s, so
this is on the order of a millisecond of tensor-core time — against a step that currently runs ~200 ms of
forward+backward. Under ~1%. The exact-SVD alternative, by contrast, is not bounded by FLOPs but by the
sequential, non-tensor-core nature of the factorization, which in practice runs orders of magnitude slower
in wall time and doesn't even like bf16. So the trade is real: five bf16 matmuls per matrix buy me the
better direction for a per-step cost I can barely measure.

Why should this beat Adam on *this* race? Two reasons converge. First, the better-conditioned update
should reduce loss more per step, so I can shave iterations off the schedule and the wallclock drops.
Second — and this is what makes it a *speedrun* method rather than just a better optimizer — the
orthogonalization is those five bf16 matmuls, nearly free on H100 tensor cores next to the forward and
backward pass; so I get the better direction at almost no per-step cost. The risk is that the imperfect,
overshooting Newton–Schulz makes the updates noisy enough to need a smaller learning rate or to destabilize
training, eating the gain — the band [0.7, 1.1] means the update isn't exactly orthogonal, and if that
residual anisotropy interacts badly with a high LR the run could get twitchy. But the whole point of
normalizing G by its norm up front and of momentum-smoothing before orthogonalizing is to keep the
iteration in the stable regime; I'll set the Muon learning rate well below the AdamW head's (0.1× to start)
and warm into it. If the mechanism is right the step count to reach 3.28 falls below the baseline's while
the per-step time barely moves, so train_time drops roughly in proportion to the step-count cut — not
through a cheaper step, but through fewer of them; if the step time balloons instead, the "nearly free"
story is wrong and I back off the coefficients or the LR.

I run the new optimizer on the transformer body's 2D matrices — that ≈85M of the ≈124M total — and keep
AdamW for the embedding/head; the full `zeropower_via_newtonschulz5` iteration and `Muon` class are in the
answer.
