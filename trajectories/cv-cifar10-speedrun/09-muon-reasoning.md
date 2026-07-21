Back to the 94% bar, which is where the headline record lives and the number everyone quotes. The last
two changes climbed the 96% bar and taught me something I want to carry back — that removing waste from
*how* training spends its compute can move a point left at constant accuracy — but airbench94 at 3.83 s
(3.29 compiled) is the standing record I want to beat as the closing move, so I ask what structural
inefficiency is left that a single change could remove.

Every algorithmic lever I pulled to reach airbench94 — whitening, Dirac init, the 64× bias rate,
Lookahead, multi-crop TTA, alternating flip — was about *initialization*, *data*, or *inference*. There
is a conspicuous gap: none of them touched the *update rule itself*. The conv weights, which are the vast
bulk of the parameters and the part that actually learns the features, are still updated by plain Nesterov
SGD — take the momentum-smoothed gradient, step in its direction. Initialization decides where the weights
start; the optimizer decides how they move; I have optimized the start exhaustively and barely questioned
the move. So: is SGD leaving speed on the table in *how* it turns a gradient into a step, specifically for
these weight *matrices*?

The geometric issue is specific to matrix parameters. A convolution's weight, reshaped to two dimensions,
is a matrix mapping input features to output features, so its gradient is also a matrix, with a
*spectrum* — singular values that for a typical gradient are wildly unequal: a few large ones capturing
the dominant direction of change, a long tail of small ones. SGD steps in the raw gradient direction, so
the update is dominated by the top singular directions: the few directions where the gradient is large get
most of the step, the many where it is small barely move. Early training over-updates a handful of
dominant directions in each weight matrix and *starves* the rest. In a long run the starved directions
eventually get their turn as the loud ones saturate; in a short run they may never catch up — and this run
is very short, 9.9 epochs at ~49 steps is about 485 optimizer steps. Across 485 steps, if a matrix's
gradient spectrum spans two orders of magnitude, the smallest-singular-value directions receive on the
order of a *hundredth* the cumulative motion of the largest, arriving essentially untrained. The features
those directions would encode are simply never learned.

What I want instead is an update that moves the same distance along every direction the gradient points
in, regardless of that direction's gradient magnitude. The object that does this is the
*orthogonalization* of the gradient: if the momentum-smoothed gradient G has SVD G = U S Vᵀ, then U Vᵀ —
G with all singular values flattened to one — points along the same directions but gives every one equal
step length. Updating with U Vᵀ is steepest descent under the *spectral* (operator) norm rather than the
Euclidean norm, and the spectral norm is the natural geometry for a matrix parameter — it measures a
matrix by its largest singular value, so steepest descent under it equalizes the singular directions
instead of chasing the loudest. This is the same *kind* of correction as the 64× bias rate: there I
equalized the rate between two *kinds* of parameter (loud scalar biases vs redundant weights); here I
equalize it across the *singular directions within* each weight matrix.

Adam is the obvious alternative that also "equalizes learning across directions," and it is the wrong tool
here: it rescales each *coordinate* of the gradient by its own running magnitude, equalizing across the
individual weight *entries*. But the entries are not the natural directions of a matrix parameter — the
singular directions are — and a matrix can have every coordinate at similar magnitude while its singular
spectrum is wildly unequal, because the imbalance lives in the correlations between entries, which a
per-coordinate scaler cannot see. So Adam flattens the wrong basis and leaves the singular-value
starvation untouched, while adding per-parameter state I do not want. The correct object is spectral.

The obstacle is cost: an exact SVD of every conv weight every step, 485 times, would drown the sub-3-second
budget in eigendecompositions. So I need an SVD-*free* route to U Vᵀ — a *Newton–Schulz iteration*: a
fixed low-degree polynomial in G, applied a few times, that drives the singular values toward 1 while
leaving the singular vectors alone (an odd polynomial in G acts on its singular values through the same U,
V — it only reshapes S). Starting from G normalized so its top singular value is ≤ 1, I iterate a degree-5
update X ← aX + (b·XXᵀ + c·(XXᵀ)²)X with (a, b, c) = (3.4445, −4.7750, 2.0315), all matmuls.

The coefficient choice is the heart of it, so I check what the induced odd polynomial p(x) = a·x + b·x³ +
c·x⁵ does to a single singular value. Its slope at zero is p′(0) = a = 3.4445, well above 1, so a *small*
singular value is amplified ~3.4× on the first pass — the coefficients are chosen to maximize this slope,
which is what makes the starved small directions shoot up fast. Trace one: x = 0.1 → p(0.1) ≈ 0.34, then
p(0.34) ≈ 0.99 — so 0.1 climbs to ~0.99 in two iterations. Meanwhile p(1) = 3.4445 − 4.7750 + 2.0315 =
0.701, *not* 1, so the iteration has no clean fixed point at one: a large singular value is pulled down
toward ~0.7 and the values oscillate in a bounded band rather than converging exactly. That is deliberate
— by maximizing the slope at zero I give up exact convergence to U Vᵀ, and after three iterations the
spectrum is left compressed into roughly (0.5, 1.5), a U S′ Vᵀ with S′ spread over that band. The
important part (un-starving the weak directions) is done hard and fast, the large directions are held near
one, and the residual spread is bounded and in practice harmless. Three iterations suffice. Two
implementation points make it work (the function is in the answer): the normalization divides by the
Frobenius norm, and since ‖X‖_F² = Σ σᵢ² ≥ σ_max², this guarantees the top singular value of the
normalized X is ≤ 1 — the precondition the polynomial needs, since it only behaves on the interval where
singular values start below 1. And a transpose runs the iteration on the *smaller* matrix dimension, since
the cost is dominated by X @ Xᵀ ((min-dim)² × max-dim), and orthogonalizing Xᵀ then transposing back gives
the same U Vᵀ — a free efficiency win, and bfloat16 halves the matmul cost with no meaningful loss since
the iteration is only pushing singular values into a band.

That is the update direction; the optimizer around it is Muon — MomentUm Orthogonalized by Newton-schulz.
Each step: accumulate momentum on the gradient, orthogonalize the *momentum buffer*, then step — and
orthogonalizing the buffer, not the raw minibatch gradient, matters. A single minibatch gradient is noisy,
and its small singular directions are dominated by that noise; orthogonalizing the raw gradient would
flatten the spectrum of the *noise*, amplifying tiny noise-directions up to unit step length and injecting
garbage into the weights. Momentum averages the minibatch noise down first, so by the time I orthogonalize,
the small singular directions of the buffer are real accumulated signal — and *those* are the starved
directions worth lifting. Momentum here is 0.6, lower than the usual 0.9, which fits: the orthogonalization
already stabilizes the update direction, so I need less momentum smoothing, and a lower momentum keeps the
buffer responsive over a ~485-step run. The reshape that feeds the iteration collapses a 4-D conv filter
(out, in, kh, kw) to 2-D as (out, in·kh·kw), so a 512→512 3×3 filter becomes (512, 4608) and the transpose
runs on min(512, 4608) = 512 — and each output channel's full 3×3×in receptive filter is one row, so the
singular directions I equalize are the genuine input-feature-to-output-feature directions of the
convolution.

There is a structural rhyme that makes me think this is the *right* closing move. The ladder opened by
*whitening the input* — flattening the input covariance's eigenspectrum so every input direction carried
equal variance — and it closes by *whitening the update* — flattening the gradient matrix's singular
spectrum so every direction of every weight matrix gets an equal-length step. Same operation at the two
ends of the pipeline: the data going in, and the gradient coming back. That the update geometry is the
last unconditioned spectrum in the system is why it is the lever left to pull. Two scoping choices make it
work: I apply Muon only to the 4-D conv filters (the matrices where the spectral argument bites) and keep
the scalars — the whitening bias, the norm biases, the linear head — on plain SGD, since "orthogonalize
the singular values" is meaningless for a vector. And because the update is now *scale-free* (its singular
values are ≈ 1, so its spectral size is independent of the gradient magnitude), the *magnitude* of each
weight is no longer regulated by the update, so before each step I renormalize the weight to a fixed norm
√(len(p)) — root-mean-square 1 per element — keeping every filter matrix at a controlled scale so the
equal-length updates land consistently. The conv filters go to Muon at lr 0.24, momentum 0.6, Nesterov;
everything else stays on the earlier SGD (the split is in the answer).

I have to be honest about the cost ledger, because Muon is not free per step the way Lookahead was. The
Newton–Schulz iteration is three passes, each a couple of matmuls of the reshaped filter — a 512×512-ish
matrix means ~512³ ≈ 10⁸ FLOPs per matmul, several per iteration, three iterations, across ~10 conv
weights, so on the order of 10¹⁰ extra FLOPs per step layered on the forward+backward. Against a
forward+backward of this ~2M-parameter net at batch 1024 (~10¹¹ FLOPs per step), the surcharge is very
roughly ~10%. That sets the bar cleanly: Muon must cut the step count by more than ~10% just to break even,
and it profits on everything beyond. Since the mechanism un-starves directions that a 485-step SGD run
leaves *hundreds of times* behind, a step-count reduction well past 10% is plausible, which is why the
ledger looks favorable rather than marginal.

So this is the closing move. The bet is that replacing SGD with Muon on the conv weights equalizes the
per-direction learning of every weight matrix — so the starved low-singular-value directions, which a
485-step SGD run never finishes training, all learn from the first steps — and that this clears 94% in
*fewer steps still*, dropping the record below 3.29 seconds. The prediction is sharp: seconds-to-94% below
the airbench94 record with mean accuracy held at the bar, and if the Newton–Schulz surcharge is not repaid
— if the per-step cost rises more than the step count falls — I would see the seconds go *up* despite
fewer epochs, the clean signature of a losing ledger. The wager of this whole ladder has been that the
right *structural* insight beats brute force, and the update geometry is the last structure left to fix.
If orthogonalizing the matrix updates buys back more steps than the iteration costs, Muon is the fastest
known way to train a net to 94% on CIFAR-10, and that is where the ladder ends.
