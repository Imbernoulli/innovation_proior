The bonus of the whole task is feature learning, but the floor — the thing every adaptive method has to
beat — is the strategy that learns *no* features at all: freeze the first layer at its random
initialization and let only the readout move. That is the lazy / NTK regime, and it is the right place to
start, because if I can name precisely what a fixed feature map can and cannot do on these three targets,
I have the yardstick every later method is measured against. I start here not because I think it will win,
but because it is the most honest possible baseline: the network with its capacity to adapt deliberately
switched off.

Concretely, inside this contract — `build_model`, `get_optimizer`, `train_step` on a two-layer net,
`d=100`, `M=100`, batch `b=150`, `T=4000` one-pass steps, so total sample budget `n = b·T = 6·10^5`. The
default fill is the mean-field network whose weights move freely; to turn it into a fixed-feature machine
I keep the two-layer shape but *freeze the first layer*. It becomes a bank of `M` random features
`phi_j(x) = sigma(<w_j, x> + c_j)`, `w_j, c_j` sampled once and never updated, with output
`fhat(x) = (1/sqrt(M)) sum_j a_j phi_j(x)` and only the readout `a` trainable. Training the readout by
square loss on fresh batches is exactly online least-squares onto a fixed `M`-dimensional feature map —
nothing nonlinear left in the optimization, the loss in `a` convex, the whole object a random-features
approximation to kernel ridge regression. This is the regime a two-layer network *falls into* if you
parametrize it the wrong way, so I want to see how badly it does before arguing for the feature-learning
parametrization.

A pure linear model `fhat(x) = <a,x>` is also a fixed feature map, but it represents only degree-1
functions and would conflate "the machine can't adapt to `I`" with "the machine has no nonlinearity";
widening the bank to `M' = 10^4` random features to resolve degree-2 pairs is off-contract
(`config.width` is fixed at 100), and matching the width the feature-learning methods get is exactly what
makes this a fair control. So I take the middle member — the two-layer nonlinear shape at `M=100`, layer
one frozen, readout trained: the lazy version of the *actual* architecture, which is precisely the object
I need to compare against.

The choices inside the frozen bank are not arbitrary. I switch from the scaffold's shifted sigmoid to a
ReLU random-feature stack with `w ~ N(0, 1/d)` and bias `c ~ N(0,1)`. The `1/d` variance: with
`x in {+1,-1}^d`, the pre-activation `<w,x> = sum_i w_i x_i` has variance `sum_i w_i^2 = ||w||_2^2`, and
`E||w||_2^2 = d·(1/d) = 1`, so `<w,x>` has standard deviation `≈ 1`; adding `c ~ N(0,1)` makes the total
pre-activation `≈ N(0,2)`, std `≈ 1.41` — exactly the range where a ReLU spends real time near its kink,
some inputs on, some off. The scaffold's `w ~ N(0, I_d)` instead gives `||w||_2^2 ≈ 100` and pre-activation
std `≈ 10`: each feature dominated by one enormous near-linear direction, the `M` random features a
badly ill-conditioned basis. The `N(0,1/d)` scaling is the standard NTK / He choice that keeps every
feature `O(1)` and the random-feature Gram well-conditioned; the nonzero bias shifts each kink off the
origin so the features tile the input distribution.

The output normalization is `1/sqrt(M)`, not the mean-field `1/M` — the load-bearing difference between
this baseline and the next. With `1/sqrt(M)` and unit-scale readout, each of the `M` features contributes
`O(1/sqrt(M))` to the output and the readout can move `O(1)` while the first-layer weights, were they
trainable, would barely move — the linearization-around-init that *defines* the lazy regime. I start the
readout at zero: features fixed and objective convex in `a`, so the init only sets the transient, and the
network outputs `0` before training then grows toward the least-squares solution.

The step size follows the same logic. With `1/sqrt(M)` normalization the effective feature vector is
`phi/sqrt(M)`, squared norm `(1/M) sum_j phi_j^2 ≈ E[phi^2] = O(1)`, so the square-loss curvature in `a`
is an `O(1)` operator over `M=100 ≈ d` coordinates. The scaffold's `lr=0.5` — tuned for the `1/M`
mean-field output — would oscillate the online least-squares iterate; the dimension-aware fix is to shrink
the readout step by the feature dimension, `lr = 0.5/d = 0.005`, on the trainable parameters only.
`get_optimizer` filters to `requires_grad` (just the readout), plain SGD, no momentum. `train_step` is the
unchanged square-loss-on-a-fresh-batch. The readout has only `M=100` free parameters fit on `n = 6·10^5`
samples — `6000:1`, massively over-determined — so the online solution converges tightly to the ridgeless
optimum, and the three seeds should land nearly on top of each other. Whatever floor I read off is the
*approximation* ceiling of the fixed feature map, not an optimization or estimation artifact.

Now the part that matters: what this floor can and cannot do. A fixed feature map `phi` defines a kernel
`K(x,x') = E_{w,c}[phi_w(x) phi_w(x')]`, and online least-squares onto `M` random features approximates
ridgeless regression in its RKHS. On the hypercube with isotropic random weights this kernel is diagonal
in the Fourier (parity) basis, its eigenvalue depending only on the *degree* of the monomial: every
degree-`k` character shares one `lambda_k`, and there are `C(d,k) ≈ d^k/k!` of them. That degeneracy is
the whole story, and two budgets bound what I can resolve. First, feature count: `M` features span at most
an `M`-dimensional slice of the RKHS, and a single random ReLU's expected squared correlation with a
specific degree-2 monomial `x_i x_j` is its degree-2 Fourier mass, an `O(1)` constant, spread by symmetry
roughly uniformly over the `C(d,2) ≈ d^2/2` pairs — so `O(1/d^2)` lands on the latent pair. Summing `M`
independent features, the represented squared norm on that pair is `≈ 2M/d^2 = 200/10^4 = 0.02`: with
`M=100` I capture at most `~2%` of a degree-2 monomial's norm, and resolving one would need
`M = Omega(d^2) = 10^4` features. Second, samples: the standard fixed-feature bound says a degree-`k`
sparse parity over an unknown subset needs `min(n,q) = Omega(d^k)` for any feature map of dimension `q`.
The thresholds are `d^1 = 100`, `d^2 = 10^4`, `d^3 = 10^6`, and my `n = 6·10^5` sits *below* `d^3`.

The two budgets agree, and their agreement locates the binding knob. Pushing `M -> infinity` at fixed `n`
gives ridgeless regression against the exact arccosine kernel, still diagonal by degree, and separating one
direction out of a `d^k`-fold-degenerate eigenspace needs exactly `n = Omega(d^k)` — the same thresholds.
So even with *unlimited* features I would clear degree-2 (`10^4 < 6·10^5`) but stall at degree-3 by samples
alone (`10^6 > 6·10^5`); my `M=100` machine additionally loses degree-2 to the `2%` capture. At degree-2 I
am *feature-limited* (samples would suffice); at degree-3 I am limited on both counts. The activation choice
barely matters for a frozen bank — any non-polynomial nonlinearity gives the same degree-degeneracy — so
ReLU-with-bias is simply the best-conditioned textbook pick; and I skip the closed-form `100×100` normal
equations only because the contract wants one optimizer step per batch and `n/M = 6000` lands the online
iterate on the same ridgeless solution anyway.

So the prediction is per-degree. Degree-1 (the lone `z1` in `h1`) is reachable — a linear function of a
single coordinate, and `n = 6·10^5 >> d = 100` resolves it. Degree-2 (the `z1z2` term of `h1`, all three
terms of `h2`) sits at `d^2 = 10^4`: samples clear it, but `M=100` is two orders short of the `Omega(d^2)`
features needed and the `2%`-capture says the basis simply lacks them — out of reach, feature-limited.
Degree-3 (the `z1z2z3` term of `h1`, all of `h3`) needs `d^3 = 10^6` on both counts — flatly unreachable.
Net: at best the degree-1 piece of `h1`, trivial predictor everywhere else.

What does the trivial predictor score? Each target is a sum of distinct unit-magnitude parity monomials,
orthogonal on `{+1,-1}^d`, so the constant predictor `0` has `test_mse = E[h*(x)^2] = ` the number of
monomials. `h1 = z1 + z1z2 + z1z2z3` gives `3`; grabbing the degree-1 term with coefficient `c` removes
`2c - c^2` and lands at `3 - (2c - c^2)`. `h2` (three degree-2 monomials) gives `3`, and since no degree-2
piece is resolvable it should sit right there. `h3` (a single degree-3 monomial) has variance `1`, so
`test_mse ≈ 1`, `score = exp(-1) ≈ 0.37` — the *highest* of the three scores precisely because its target
is the *smallest* in magnitude, not because anything is learned. That is the trap the geometric-mean
aggregate exists to expose: a high `score_h3` is not evidence of learning `h3`; only `test_mse` near zero
would be.

The Fourier-recovery column is the honest tell — it measures `mean |hat_S(model) - hat_S(h*)|` over the
latent monomials. For a trivial predictor `hat_S(model) = 0`, so recovery equals the mean true coefficient,
`= 1` per monomial: recovery near `1` means learned nothing. On `h1` the small degree-1 pickup should pull
recovery a hair below `1` and `test_mse` a fraction below `3` — the signature of a regression onto features
that resolve almost nothing but not literally zero.

Translating to what the leaderboard ranks: the geometric mean of the three scores is
`exp(-(mse_h1 + mse_h2 + mse_h3)/3)`, the exponential of the *average* test MSE. For the trivial floor
`(≈3, ≈3, ≈1)` the average is `≈2.33` and the aggregate is `exp(-2.33) ≈ 0.10`. That is the number to beat,
and the arithmetic pins responsibility on the two frozen degree-2/degree-3 environments: the degree-1
nibble on `h1` barely moves the average, whereas pulling a stuck-at-`3` environment toward `0` would move it
a full unit. I add no ridge penalty or weight decay — either would only bias the already over-determined
readout away from the least-squares optimum and make the floor *look* worse than the feature map truly is.
The cleanest floor is the unregularized ridgeless one.

The diagnosis this floor delivers is already pointed at the next step. The failure is not a learning-rate
or optimizer problem — the readout regression is convex, over-determined, and converges. It is that the
*features cannot move toward the latent subset `I`*. The fix is to *unfreeze* them: leave the lazy regime
for the parametrization where the first-layer weights travel an `O(1)` distance and rotate toward the
signal coordinates — the mean-field `1/M` scaling, plain SGD, a large step. I expect that to beat the floor
on `h1`, where feature learning can climb the leap-1 staircase that no fixed feature map reaches, and to
*still* fail on `h2` and `h3`. But that is the next question. Here the only claim is the floor: with the
features frozen the network is a kernel machine that cannot adapt to the hidden subset, and on high-degree
sparse targets at this budget a non-adaptive kernel machine cannot escape the trivial predictor. The
distilled fill — frozen ReLU random features, `w ~ N(0,1/d)`, `c ~ N(0,1)`, `1/sqrt(M)` output,
readout-only SGD at `lr = 0.5/d`.
