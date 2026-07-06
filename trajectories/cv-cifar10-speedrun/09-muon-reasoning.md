Back to the 94% bar, which is where the headline record lives, and where I left it at airbench94: 3.83
seconds, 3.29 compiled. The last two rungs climbed the 96% bar and taught me something I want to carry
back — that removing waste from *how* training spends its compute (data filtering) can move a point left
at constant accuracy — but the record everyone quotes is the 94% one, and that is the number I want to
beat as the closing move. So the frame is: airbench94 is the standing record at 3.83 / 3.29, and I ask
what structural inefficiency is left that a single change could remove.

Every algorithmic lever I pulled to reach airbench94 — whitening, Dirac init, the 64× bias learning rate,
Lookahead, multi-crop TTA, alternating flip — was about *initialization*, *data*, or *inference*. Lay
them out and there is a conspicuous gap: none of them touched the *update rule itself*. The conv weights,
which are the vast bulk of the parameters and the part that actually learns the features, are still being
updated by plain Nesterov SGD — take the momentum-smoothed gradient, step in its direction. Initialization
decides where the weights start; the optimizer decides how they move; I have optimized the start
exhaustively and barely questioned the move. So the question for this rung is whether SGD is leaving speed
on the table in *how* it turns a gradient into a step, specifically for these weight *matrices*.

Here is the geometric issue, and it is specific to matrix parameters. A convolution's weight, reshaped to
two dimensions, is a matrix mapping input features to output features. Its gradient is therefore also a
matrix, and a matrix has a *spectrum* — singular values that, for a typical gradient, are wildly unequal:
a few large ones capturing the dominant direction of change, a long tail of small ones. SGD steps in the
raw gradient direction, which means the update is dominated by the gradient's top singular directions: the
few directions where the gradient happens to be large get most of the step, and the many directions where
it is small barely move. So early training over-updates a handful of dominant directions in each weight
matrix and *starves* the rest. In a long run the starved directions eventually get their turn as the loud
ones saturate; in a short run they may never catch up. And this run is very short — 9.9 epochs at ~49
steps per epoch is about 485 optimizer steps total. Across 485 steps, if a matrix's gradient spectrum
spans, say, two orders of magnitude, the smallest-singular-value directions receive on the order of a
*hundredth* the cumulative motion of the largest, so they arrive at the finish line essentially untrained.
The features those directions would encode are simply never learned. That is the inefficiency: in a 485-
step budget, SGD spends almost all its motion on the few loud directions of each weight matrix.

What I want instead is an update that treats every direction of the weight matrix *evenly* — that moves
the same distance along every direction the gradient points in, regardless of how large or small that
direction's gradient magnitude is. The mathematical object that does exactly this is the
*orthogonalization* of the gradient. If the momentum-smoothed gradient matrix G has singular value
decomposition G = U S Vᵀ, then the matrix U Vᵀ — G with all its singular values flattened to one — points
along the same singular directions as G but gives every one of them equal step length. Updating with U Vᵀ
instead of G is a steepest-descent step under the *spectral* (operator) norm rather than the Euclidean
norm, and the spectral norm is the natural geometry for a matrix parameter — it measures a matrix by its
largest singular value, so steepest descent under it equalizes the singular directions instead of chasing
the loudest. This is the same *kind* of correction I made with the 64× bias learning rate: there I
equalized the learning rate between two *kinds* of parameter (loud scalar biases vs redundant weights);
here I equalize it across the *singular directions within* each weight matrix. Same theme — do not let a
few well-resolved directions hog the step budget — on a different axis.

Before I chase the SVD-free route, let me place this against the obvious alternative, because there is a
well-known optimizer that also "equalizes learning across directions" and I should say why it is not the
right tool here. Adam (and its kin) rescales each *coordinate* of the gradient by its own running
magnitude, so it equalizes learning across the individual weight *entries*. But the entries are not the
natural directions of a matrix parameter — the *singular* directions are, and a matrix can have every
coordinate at similar magnitude while its singular spectrum is wildly unequal (the imbalance lives in the
correlations between entries, which a per-coordinate scaler cannot see). So Adam fixes the wrong basis: it
would flatten the coordinate-wise spectrum and leave the singular-value starvation untouched, and it would
add per-parameter state I do not want. The correct object is spectral, not coordinate-wise, which points at
orthogonalization. The other alternative — an exact SVD to get U Vᵀ literally — has the right geometry but
the wrong cost.

The obstacle is cost. Computing the SVD of every conv weight every step, 485 times, is far too slow for a
sub-3-second budget; the record would drown in eigendecompositions. So I need an SVD-*free* way to compute
(approximately) U Vᵀ. The trick is a *Newton–Schulz iteration*: a fixed low-degree polynomial in G,
applied a few times, that drives the singular values toward 1 while leaving the singular vectors alone
(because an odd polynomial in G acts on G's singular values through the same U, V — it only reshapes S).
Starting from G normalized so its top singular value is at most 1, I iterate a degree-5 update X ← aX +
(b·XXᵀ + c·(XXᵀ)²)X with coefficients (a, b, c) = (3.4445, −4.7750, 2.0315), and it is all matmuls, which
the GPU eats for breakfast.

Let me actually check what this polynomial does to a singular value, because the coefficient choice is the
heart of it and I should verify rather than trust. On a single singular value x the iteration applies p(x)
= a·x + b·x³ + c·x⁵ (the odd polynomial induced by the matrix update). Two evaluations tell the story. The
slope at zero is p′(0) = a = 3.4445 — steep, well above 1 — so a *small* singular value is amplified by
~3.4× on the first pass: the coefficients are chosen precisely to maximize this slope, which is what makes
the starved small directions shoot up fast. Trace one: x = 0.1 → p(0.1) = 0.344 − 0.005 + 0.00002 ≈ 0.34,
then p(0.34) = 1.171 − 0.188 + 0.009 ≈ 0.99 — so a singular value of 0.1 climbs to ~0.34 and then to ~0.99
in just two iterations. Meanwhile p(1) = 3.4445 − 4.7750 + 2.0315 = 0.701, *not* 1 — so the iteration does
*not* have a clean fixed point at one; a large singular value gets pulled down toward ~0.7 and the values
oscillate in a bounded band rather than converging exactly. That is deliberate: by maximizing the slope at
zero I give up exact convergence to U Vᵀ, and after three iterations the spectrum is left compressed into
roughly (0.5, 1.5) — something like U S′ Vᵀ with S′ spread over that band, not a clean orthogonalization.
The check confirms the design: small singular values are lifted hard and fast (the important part —
un-starving the weak directions), large ones are held near one, and the residual spread of S′ is bounded
and, in practice, harmless. Three iterations suffice.

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)            # ensure top singular value <= 1
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X
```

Two implementation details in that function are load-bearing and worth verifying. The normalization `X /=
(X.norm() + eps)` divides by the Frobenius norm, and since the largest singular value of a matrix is always
at most its Frobenius norm (‖X‖_F² = Σ σᵢ² ≥ σ_max²), this guarantees the top singular value of the
normalized X is ≤ 1 — which is exactly the precondition the iteration needs, because the polynomial only
behaves (amplifies small, holds large near one) on the interval where the singular values start below 1.
Feed it a matrix with σ_max > 1 and the quintic term would blow up. Second, the transpose: `if G.size(0) >
G.size(1): X = X.T` flips the matrix so the iteration runs on the *smaller* dimension. The cost of the
iteration is dominated by X @ Xᵀ, which is (min-dim)² × (max-dim) — computing it on the smaller side is
cheaper — and orthogonalizing Xᵀ then transposing back gives the same U Vᵀ up to the transpose, so it is a
free efficiency win. Running the whole thing in bfloat16 is the other speed lever: the iteration is
robust to low precision (it is pushing singular values into a band, not computing anything delicate), so
bfloat16 halves the matmul cost with no meaningful loss.

That is the update direction; the optimizer around it is Muon — MomentUm Orthogonalized by Newton-schulz.
Each step: accumulate momentum on the gradient, orthogonalize the momentum buffer via Newton–Schulz, then
step. The order there is not arbitrary — I orthogonalize the *momentum buffer*, not the raw minibatch
gradient, and that ordering matters. A single minibatch gradient is noisy, and its small singular
directions are dominated by that noise; if I orthogonalized the raw gradient I would flatten the spectrum
of the *noise*, amplifying the tiny noise-directions up to unit step length and injecting garbage into the
weights. Momentum averages the minibatch noise down first, so by the time I orthogonalize, the small
singular directions of the buffer are real accumulated signal, not per-batch noise — and *those* are the
starved directions worth lifting. So momentum-then-orthogonalize is the right composition: smooth the
noise, then equalize the directions of what remains. The `buf.mul_(momentum).add_(g)` accumulates the
buffer and `g = g.add(buf, alpha=momentum)` forms the Nesterov look-ahead combination before it is
orthogonalized. Momentum here is 0.6, lower than the usual 0.9 — which fits, because the orthogonalization
already stabilizes the update direction, so I need less momentum smoothing on top, and a lower momentum
keeps the buffer responsive over a run that is only ~485 steps long.

The reshape that feeds the iteration is worth a shape check. A conv filter is 4-D, (out, in, kh, kw), and
`g.reshape(len(g), -1)` collapses it to 2-D as (out, in·kh·kw) — `len(g)` is the output-channel count and
`-1` folds the input channels and the 3×3 spatial taps into one axis. So a 512→512 3×3 filter becomes a
(512, 512·9) = (512, 4608) matrix, and the transpose trick then runs the iteration on min(512, 4608) = 512,
the small side. The "matrix mapping input features to output features" I have been reasoning about is
exactly this (out) × (in·kh·kw) reshaping — each output channel's full 3×3×in receptive filter is one row —
so the singular directions I am equalizing are the genuine input-feature-to-output-feature directions of
the convolution, which is the object the spectral argument was about.

There is a symmetry here that tells me this is the *right* closing move, not just an available one. The
ladder opened, at the very first rung, by *whitening the input*: estimating the input's covariance and
flattening its eigenvalue spectrum so every input direction carried equal variance and the first layer saw
a well-conditioned signal. It closes by *whitening the update*: taking the gradient matrix and flattening
its singular spectrum so every direction of every weight matrix gets an equal-length step. The code comment
even calls it that — `# whiten the update`. Same operation (flatten a spectrum so no direction dominates),
applied at the two ends of the pipeline: to the data going in, and to the gradient coming back. The first
whitening conditioned *where the signal enters*; this one conditions *how the weights move*. That the
entire speedrun is bookended by the same spectral-flattening idea, once on the input covariance and once on
the gradient's singular values, is the kind of structural rhyme that makes me think the update geometry is
genuinely the last unconditioned spectrum in the system. Two scoping choices make it work in this network. First, I apply Muon only to the 4-D conv filters
(reshaped to 2-D) — the matrices where the spectral argument actually bites — and I keep the cheap,
well-behaved scalars (the whitening bias, the norm biases, the linear head) on plain SGD, because
"orthogonalize the singular values" is meaningless for a vector or a scalar; there is no matrix spectrum to
equalize. Second, one piece pairs naturally with orthogonalized updates and I have to add it or the scheme
misbehaves: because the update direction is now *scale-free* (its singular values are ≈ 1, so the update
has a fixed spectral size independent of the gradient magnitude), the *magnitude* of each weight matters
independently and is no longer regulated by the update. So before each orthogonalized step I renormalize
the weight to a fixed norm, `p.data.mul_(len(p.data)**0.5 / p.data.norm())`, which sets its norm to
√(len(p)) — i.e. root-mean-square 1 per element — keeping every filter matrix at a controlled scale so the
equal-length updates land consistently rather than drifting the weight norm around.

```python
buf.mul_(momentum).add_(g)
g = g.add(buf, alpha=momentum) if nesterov else buf
p.data.mul_(len(p.data)**0.5 / p.data.norm())                 # normalize the weight
update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update
p.data.add_(update, alpha=-lr)                                 # take a step
```

The conv filters go to Muon (`filter_params = [p for p in model.parameters() if len(p.shape) == 4 and
p.requires_grad]`, at lr 0.24, momentum 0.6, Nesterov) and everything else stays on the SGD optimizer from
the earlier rungs — a clean two-optimizer split by parameter shape.

I have to be honest about the cost ledger, because Muon is not free per step the way Lookahead or a
per-group lr was. The Newton–Schulz iteration is three passes, each a couple of matmuls of the reshaped
filter — a 512×512-ish filter matrix means matmuls on the order of 512³ ≈ 10⁸ FLOPs each, several per
iteration, three iterations, across the ~10 conv weights, so on the order of 10¹⁰ extra FLOPs per step
layered on top of the forward+backward. That per-step surcharge is real and it must be *repaid* by needing
fewer steps: if orthogonalizing the updates lets the run clear 94% in enough fewer steps that the total
(fewer steps × more-expensive steps) beats airbench94's (more steps × cheaper steps), Muon wins; if not,
it loses. Let me put a rough break-even on it. The forward+backward of this ~2M-parameter net at batch 1024
is on the order of 10¹¹ FLOPs per step; the Newton–Schulz surcharge I estimated at ~10¹⁰, so the per-step
cost rises by very roughly ~10%. That sets the bar cleanly: Muon has to cut the step count by more than
about 10% just to break even, and it profits on everything beyond that. Given that the mechanism is
"un-starve directions that a 485-step SGD run leaves essentially untrained" — directions that are not
slightly behind but *hundreds of times* behind on the loud/quiet spectrum — a step-count reduction well
past 10% is a plausible payoff, which is why the ledger looks favorable rather than marginal. The two hedges I already accepted feed into this same ledger — the iteration is approximate (it
yields U S′ Vᵀ, not U Vᵀ, per the trace above) and it is these extra matmuls — and both are bets that the
per-step cost buys back more than its worth in saved steps.

So this is the closing move, and I will state the bar it has to clear and why I believe it can. The
standing 94% record at the top of this ladder is airbench94 at 3.83 seconds (3.29 compiled). The bet is
that replacing SGD with Muon on the conv weights equalizes the per-direction learning of every weight
matrix — so the starved low-singular-value directions, which a 485-step SGD run never finishes training,
all learn at once from the first steps — and that this lets the same architecture clear 94% in *fewer
steps still*, dropping the record below 3.29 seconds. The falsifiable prediction is sharp: seconds-to-94%
below the airbench94 record with mean accuracy held at the bar, and if instead the Newton–Schulz surcharge
is not repaid — if the per-step cost rises more than the step count falls — I would see the seconds go *up*
despite fewer epochs, which is the clean signature of a losing ledger. The wager of this whole ladder has
been that the right *structural* insight beats brute force, and the update geometry is the last structure
left to fix: initialization, learning rates, averaging, inference, augmentation, and data selection are all
spent, and the way a gradient matrix becomes a step is the one untouched lever. If orthogonalizing the
matrix updates buys back more steps than the iteration costs, Muon is the fastest way known to train a net
to 94% on CIFAR-10, and that is where the ladder ends.
