The bonus of the whole task is feature learning, but the floor — the thing every adaptive method has to
beat — is the strategy that learns *no* features at all: freeze the first layer at its random
initialization and let only the readout move. That is the lazy / NTK regime, and it is the right place to
start, because if I can name precisely what a fixed feature map can and cannot do on these three targets,
I have the yardstick that every later rung is measured against. So I start here not because I think it
will win, but because it is the most honest possible baseline: it is the network with its capacity to
adapt deliberately switched off.

Let me set up what "no feature learning" means concretely inside this scaffold. The contract gives me
`build_model`, `get_optimizer`, `train_step` on a two-layer net, with `d=100`, `M=100`, batch `b=150`,
`T=4000` one-pass steps, so the total sample budget is `n = b·T = 150·4000 = 6·10^5`. The default fill is
the mean-field network whose weights are free to move; to turn it into a fixed-feature machine I keep the
same two-layer shape but *freeze the first layer*. The first layer becomes a bank of `M` random features
`phi_j(x) = sigma(<w_j, x> + c_j)` with `w_j, c_j` sampled once and never updated, and the model output
is a linear combination of those features, `fhat(x) = (1/sqrt(M)) sum_j a_j phi_j(x)`, with only the
readout `a` trainable. Training the readout by square loss on fresh batches is exactly online
least-squares regression onto a fixed `M`-dimensional feature map — there is nothing nonlinear left in the
optimization, the loss in `a` is convex, and the whole object is a random-features approximation to kernel
ridge regression. This is the canonical lazy baseline, and the reason it deserves a rung of its own: it is
the regime a two-layer network *falls into* if you parametrize it the wrong way, so I want to see exactly
how badly it does before I argue for the feature-learning parametrization.

Before I commit to random ReLU features I should ask whether they are even the right embodiment of "no
feature learning," because there is a small design space here and picking the wrong member would make the
floor either dishonestly weak or off-contract. The bluntest option is a pure linear model,
`fhat(x) = <a, x>`: train a linear readout directly on the raw coordinates. That is genuinely a fixed
feature map — the identity features — and it is convex and trivial to fit. But it can represent *only*
degree-1 functions of `x`; it cannot even in principle put mass on a degree-2 or degree-3 monomial, so a
linear model would conflate two very different failures, "the machine can't adapt its features to `I`" and
"the machine has no nonlinearity at all." I want the floor to be the *strongest* non-adaptive machine, so
that whatever gap I later attribute to feature learning is honestly the value of adaptation and not just
the value of adding a nonlinearity. A second option is to widen the feature bank far past `M=100` — throw
`M' = 10^4` random ReLU features at it so the fixed kernel can in principle resolve degree-2 pairs. But the
harness fixes the width at `config.width = 100`, and more to the point, matching the width the
feature-learning rungs will get is exactly what makes this a fair control: I want the ceiling of the
`M=100` fixed-feature machine, not of an unboundedly wide one. So I reject both extremes and take the
middle member: keep the two-layer nonlinear shape at `M=100`, freeze layer one, train the readout. That is
the lazy version of the *actual* architecture, which is precisely the object I need to compare against.

Now the choices inside this frozen-feature fill, because they are not arbitrary even though nothing
adapts. First, the activation and the initialization scale. The default mean-field net uses a
shifted-sigmoid with `w ~ N(0, I_d)`; for the lazy baseline I switch to a ReLU random-feature stack with
`w ~ N(0, 1/d)` and a bias `c ~ N(0,1)`. The reason for the `1/d` variance is worth computing rather than
asserting. With `x in {+1,-1}^d`, the pre-activation `<w,x> = sum_i w_i x_i` is, for a fixed weight
vector, a sum whose variance over the data is `sum_i w_i^2 x_i^2 = ||w||_2^2`, and with `w ~ N(0,1/d)`,
`E||w||_2^2 = d·(1/d) = 1`. So `<w,x>` has standard deviation `≈ 1`, and adding `c ~ N(0,1)` makes the
total pre-activation `≈ N(0,2)`, standard deviation `sqrt(2) ≈ 1.41`. That is exactly the range where a
ReLU spends real time near its kink: some inputs on, some off, the feature genuinely nonlinear across the
data cloud. Contrast the unit-variance `w ~ N(0, I_d)` the scaffold ships with: then `||w||_2^2 ≈ d = 100`
and the pre-activation swings with standard deviation `≈ 10`. Each feature is then dominated by one
enormous, nearly-linear response direction, and the `M` random features become a badly ill-conditioned
basis with essentially a single loud coordinate — a poor sketch of the input space. Scaling to `N(0,1/d)`
is the standard NTK / He-style choice precisely because it puts every feature at `O(1)` and keeps the
random-feature Gram well-conditioned. The nonzero bias `c ~ N(0,1)` shifts each ReLU's kink off the origin
so the features tile the input distribution rather than all firing on the same half-space.

Second, the output normalization: `1/sqrt(M)` rather than the mean-field `1/M`. This is the load-bearing
difference between this rung and the next. With `1/sqrt(M)` and unit-scale readout, each of the `M`
features contributes `O(1/sqrt(M))` to the output and the readout can move an `O(1)` amount while the
*first-layer* weights, if they were trainable, would barely move — that is precisely the
linearization-around-init that defines the lazy regime. I am choosing the parametrization that keeps the
network a fixed kernel machine; that choice is the whole point of the baseline. Third, the readout
initialization: I start `a` at zero. Since the features are fixed and the objective is convex in `a`, the
init only affects the transient, and starting at zero means the network outputs `0` before training and
the readout grows toward the least-squares solution — clean and reproducible.

The optimizer and step size follow from the same logic. The readout sees a fixed feature Gram matrix
`Phi^T Phi / b` whose scale is set by the feature second moment; with `1/sqrt(M)` normalization the
effective feature vector is `phi/sqrt(M)`, each entry `O(1/sqrt(M))`, `M` of them, so its squared norm is
`(1/M) sum_j phi_j^2 ≈ E[phi^2] = O(1)`, and the square-loss curvature in `a` is an `O(1)`-scale operator
with `M=100 ≈ d` coordinates. A learning rate of `0.5` — the scaffold's value, tuned for the `1/M`
mean-field output — is far too large here and the online least-squares iterate would oscillate. The
dimension-aware fix is to shrink the readout step by the feature dimension, matching the standard
random-features / kernel-regression practice of respecting the Gram conditioning at `M ≈ d = 100`, so I
use `lr = 0.5 / d = 0.005` on the trainable parameters only. `get_optimizer` filters to `requires_grad`
parameters (just the readout, since the first layer is frozen), and plain SGD with no momentum keeps the
iterate a faithful online gradient step on the convex readout problem. The `train_step` is the unchanged
square-loss-on-a-fresh-batch step: zero the gradients, forward, square error, backward, step, return the
float loss. There is no auxiliary loss, no alternation, nothing — the simplicity is the baseline. And one
worry I can lay to rest immediately: the readout has only `M = 100` free parameters, fit on `n = 6·10^5`
fresh samples, a `6000:1` ratio, so the online least-squares solution is massively over-determined and
converges tightly to the ridgeless optimum. Whatever floor I see is therefore the *approximation* ceiling
of the fixed feature map, not an optimization or estimation artifact — the numbers I read off will be the
honest representational limit, which is exactly what I want a baseline to report.

Now the part that actually matters: reasoning about what this floor can and cannot do, because that is the
entire purpose of running it. A fixed feature map `phi` defines a kernel `K(x,x') = E_{w,c}[phi_w(x)
phi_w(x')]`, and online least-squares onto `M` random features approximates ridgeless regression in the
RKHS of `K`. On the hypercube with isotropic random weights this kernel is diagonal in the Fourier
(parity) basis, with an eigenvalue that depends only on the *degree* of the monomial: every degree-`k`
character shares one eigenvalue `lambda_k`, and there are `C(d,k) ≈ d^k / k!` of them. That degeneracy is
the whole story. Two independent budgets bound what I can resolve. The first is the feature count: with
only `M` random features I can span at most an `M`-dimensional slice of the RKHS, and a degree-`k` monomial
over an *unknown* subset needs the features to individually correlate with it. The expected squared
correlation of a single random ReLU feature with a specific degree-2 monomial `x_i x_j` is the feature's
total degree-2 Fourier mass, an `O(1)` constant, spread by symmetry roughly uniformly over the
`C(d,2) ≈ d^2/2` pairs — so `O(1/d^2)` lands on the latent pair. Summing `M` independent features, the
represented squared norm on that pair is `≈ M/(d^2/2) = 2M/d^2 = 200/10^4 = 0.02`. So with `M=100`
features I can capture at most about `2%` of a degree-2 monomial's norm — essentially nothing — and to
represent a unit-coefficient degree-2 monomial I would need `M = Omega(d^2) = 10^4` features. The second
budget is samples: the standard fixed-feature lower bound says a degree-`k` sparse parity over an unknown
subset requires `min(n, q) = Omega(d^k)` for any feature map of dimension `q`. With `d=100` the thresholds
are `d^1 = 100`, `d^2 = 10^4`, `d^3 = 10^6`, and my `n = 6·10^5` sits *below* `d^3`.

It is worth a limiting check that these two budgets agree, because if the feature-count bound and the
sample bound disagreed I would not trust either. Push `M -> infinity` at fixed `n`: the random-feature
regression converges to ridgeless regression against the exact kernel `K`, the first-order arccosine
kernel of ReLU. On the hypercube that kernel is diagonal in the parity basis with the degree-`k`
eigenvalue `lambda_k` shared across all `C(d,k)` characters of that degree. To fit one specific degree-`k`
monomial by kernel regression I must resolve its coordinate against the `C(d,k) ≈ d^k/k!` competitors that
share `lambda_k`, and the number of samples the kernel needs to separate one direction out of a
`d^k`-fold-degenerate eigenspace is exactly `n = Omega(d^k)` — the same threshold the fixed-feature sample
bound gave. So the `M -> infinity` machine, with *unlimited* features, would still be blocked at degree-3
by samples alone (`d^3 = 10^6 > 6·10^5`), and would clear degree-2 (`d^2 = 10^4 < 6·10^5`). My machine has
the *extra* handicap of only `M = 100` features, which — by the `2M/d^2 = 0.02` capture computation —
knocks degree-2 out too. The two bounds are consistent and my finite-`M` version is strictly the weaker of
the pair, exactly as a floor should be. This also tells me which knob is binding where: at degree-2 I am
*feature-limited* (samples would suffice, features do not), at degree-3 I am limited on both counts.

One more design check, on the features themselves, before I read off predictions. I use ReLU features with
a nonzero bias rather than reusing the scaffold's shifted sigmoid, and rather than solving the readout in
closed form. The activation choice barely matters for a *frozen* bank — any non-polynomial nonlinearity
gives a universal kernel and the same degree-degeneracy structure — but ReLU with `c ~ N(0,1)` gives the
best-conditioned tiling of half-spaces and is the textbook random-feature choice, so I take it and move on;
the interesting activation questions belong to the rungs where the features actually move. The closed-form
question is more tempting: with `M=100` fixed features I could just solve the `100×100` normal equations
and skip SGD entirely. But the harness hands me a `train_step` that must take one optimizer step per fresh
batch, and I would rather stay inside that contract than special-case it — and it costs me nothing, because
with `n/M = 6000` the online iterate converges to the same ridgeless solution the normal equations would
give. So plain SGD on the readout it is, and the ceiling I measure is the true representational floor.

So the prediction is concrete and per-degree. The degree-1 component (the lone `z1` term in `h1`) is
reachable — a linear function of a single coordinate is in the span of even a modest random-feature basis,
and `n = 6·10^5 >> d = 100` samples resolve it. The degree-2 components (the `z1z2` term in `h1`, all three
terms of `h2`) sit at `d^2 = 10^4`: my sample budget clears that threshold, but my feature count `M=100` is
two orders of magnitude short of the `Omega(d^2)` features needed to resolve a degree-2 product over an
unknown pair, and the `2%`-capture computation above says the random basis simply does not contain them.
So even the degree-2 pieces are out of reach at `M=100` — bottlenecked by features, not by samples. The
degree-3 components (the `z1z2z3` term in `h1`, all of `h3`) need `d^3 = 10^6` on *both* counts — above my
sample budget and far above my feature count — so they are flatly unreachable. The net prediction: the
lazy baseline recovers at best the degree-1 piece of `h1` and is essentially at the trivial predictor on
everything else.

What does "the trivial predictor" score? Each target is a sum of distinct parity monomials, each with unit
magnitude and zero mean, and distinct parities are orthogonal on `{+1,-1}^d`, so the constant predictor
`0` has `test_mse = E[h*(x)^2] = (number of monomials)`. For `h1 = z1 + z1z2 + z1z2z3` that is `1+1+1 = 3`;
if the lazy machine grabs the degree-1 term with coefficient `c` it removes `2c - c^2` and lands at
`test_mse = 3 - (2c - c^2)`. For `h2` (three degree-2 monomials) the constant predictor gives
`test_mse = 3`, and since no degree-2 piece is resolvable the lazy machine should sit right there. For `h3`
(a single degree-3 monomial) the variance is `1`, so `test_mse ≈ 1`, `score = exp(-1) ≈ 0.37` — note that
`h3`'s score looks the *highest* of the three precisely because its target is the *smallest* in magnitude,
not because the method learns anything; it is the trivial-predictor score of a unit-variance target. This
is the trap the geometric-mean aggregate is built to expose, and it is worth naming now so I read the later
rungs correctly: a high `score_h3` is *not* evidence of learning `h3`; only a `score_h3` close to `1`
(i.e. `test_mse` near zero) would be.

The Fourier-recovery metric is the honest tell — it measures `mean |hat_S(model) - hat_S(h*)|` over the
latent monomials directly. For a trivial predictor `hat_S(model) = 0`, so recovery equals the mean of the
true coefficients, `= 1` per monomial: recovery near `1` means learned nothing. I can even predict the
small `h1` deflection quantitatively. If the frozen features capture a fraction of the `z1` coefficient,
say `c`, then `test_mse_h1 = (1-c)^2 + 1 + 1` and `recovery_h1 = (|1-c| + 1 + 1)/3`. A modest degree-1
pickup that pulls `test_mse_h1` down by only `~0.2` corresponds to `(1-c)^2 = 0.8`, i.e. `c ≈ 0.11` — so I
should see `test_mse_h1 ≈ 2.8` and `recovery_h1 ≈ (0.89 + 2)/3 ≈ 0.96`, a hair below the trivial `1.0`.
That is the signature of a regression onto features that resolve almost nothing: a whisker of the degree-1
term and nothing else.

It is worth translating this into the aggregate the leaderboard actually ranks on, a geometric mean of the
three per-environment `score`s, because that geometry decides what "beating the floor" even means. Writing
the aggregate as `exp(-(mse_h1 + mse_h2 + mse_h3)/3)`, it is the exponential of the *average* test MSE
across environments — so it rewards driving down whichever environment is worst, and a catastrophic
failure on any single target caps the whole score no matter how well the others do. For the trivial-floor
numbers `(≈3, ≈3, ≈1)` the average MSE is `≈2.33` and the aggregate is `exp(-2.33) ≈ 0.10`. That is the
number to beat, and the arithmetic tells me where the leverage is: the degree-1 nibble on `h1` moves the
`h1` term by a fraction of a unit and barely touches the mean, whereas any method that could pull one of
the stuck-at-`3` environments down toward `0` would move the average by a full unit. So the floor is not
just low, it is low in a way that pins responsibility on the two frozen degree-2/degree-3 environments — a
useful thing to have quantified before I start unfreezing features. I deliberately add no ridge penalty and
no weight decay to the readout: a ridge term would only bias the already-over-determined readout away from
the least-squares optimum and make the floor *look* worse than the feature map truly is, muddying the very
comparison this rung exists to provide. The cleanest floor is the unregularized ridgeless one.

So I expect the lazy baseline to be the floor by construction, and a particular *shape* of floor:
near-trivial `test_mse` on all three targets, Fourier recovery close to its worst possible value, and the
deceptively-high `score_h3` that comes from `h3`'s small target magnitude rather than from learning. The
one place it might show life is the degree-1 piece of `h1`, which should pull `test_mse_h1` slightly below
the full constant-predictor value of `3` — a small, real signal that this is a regression onto features,
not pure noise. Everything else should be stuck, and stuck for the structural reason the fixed-feature
bounds name: the feature map is frozen *before* it knows which coordinates are latent, so it cannot put its
limited capacity where the signal is. And because the readout is over-determined `6000:1`, I expect the
three seeds to land nearly on top of each other — a stable floor, not a noisy one.

The diagnosis this rung is built to deliver is therefore already pointed at the next step. The failure is
not a learning-rate or an optimizer problem — the readout regression is convex, over-determined, and
converges. It is that the *features cannot move toward the latent subset `I`*. The fix is not a better
optimizer on the same frozen features; it is to *unfreeze* the features — to leave the lazy regime and put
the network in the parametrization where the first-layer weights travel an `O(1)` distance and can rotate
toward the signal coordinates. That is exactly the mean-field scaling, and it is the next rung: keep the
two-layer shape but swap the `1/sqrt(M)` lazy normalization for the `1/M` mean-field one, unfreeze the
first layer, and use plain SGD with a large step so the weights actually move. I expect that to do strictly
better than the floor on `h1`, because feature learning can climb the leap-1 staircase that no fixed
feature map can reach — and I expect it to *still* fail on `h2` and `h3`, because feature learning alone
climbs only one coordinate at a time. But that is the next rung's question. Here the only claim is the
floor: with the features frozen, the network is a kernel machine that cannot adapt to the hidden subset,
and on high-degree sparse targets a non-adaptive kernel machine cannot escape the trivial predictor at this
budget. The distilled fill — frozen ReLU random features, `w ~ N(0,1/d)`, `c ~ N(0,1)`, `1/sqrt(M)`
output, readout-only SGD at `lr = 0.5/d` — is in the answer.
