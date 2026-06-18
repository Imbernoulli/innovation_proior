The bonus of the whole task is feature learning, but the floor — the thing every adaptive method has to
beat — is the strategy that learns *no* features at all: freeze the first layer at its random
initialization and let only the readout move. That is the lazy / NTK regime, and it is the right place to
start, because if I can name precisely what a fixed feature map can and cannot do on these three targets,
I have the yardstick that every later rung is measured against. So I start here not because I think it
will win, but because it is the most honest possible baseline: it is the network with its capacity to
adapt deliberately switched off.

Let me set up what "no feature learning" means concretely inside this scaffold. The contract gives me
`build_model`, `get_optimizer`, `train_step` on a two-layer net, with `d=100`, `M=100`, batch `b=150`,
`T=4000` one-pass steps. The default fill is the mean-field network whose weights are free to move; to
turn it into a fixed-feature machine I keep the same two-layer shape but *freeze the first layer*. The
first layer becomes a bank of `M` random features `phi_j(x) = sigma(<w_j, x> + c_j)` with `w_j, c_j`
sampled once and never updated, and the model output is a linear combination of those features,
`fhat(x) = (1/sqrt(M)) sum_j a_j phi_j(x)`, with only the readout `a` trainable. Training the readout by
square loss on fresh batches is exactly online least-squares regression onto a fixed `M`-dimensional
feature map — there is nothing nonlinear left in the optimization, the loss in `a` is convex, and the
whole object is a random-features approximation to kernel ridge regression. This is the canonical lazy
baseline, and the reason it deserves a rung of its own: it is the regime a two-layer network *falls into*
if you parametrize it the wrong way, so I want to see exactly how badly it does before I argue for the
feature-learning parametrization.

Now the choices inside this frozen-feature fill, because they are not arbitrary even though nothing
adapts. First, the activation and the initialization scale. The default mean-field net uses a
shifted-sigmoid with `w ~ N(0, I_d)`; for the lazy baseline I switch to a ReLU random-feature stack with
`w ~ N(0, 1/d)` and a bias `c ~ N(0,1)`. The reason for the `1/d` variance is that with `x in {+1,-1}^d`
the pre-activation `<w,x>` is a sum of `d` independent terms, so with unit-variance weights it would have
standard deviation `sqrt(d) ≈ 10` — the ReLU would be essentially linear-or-dead over the data and the
features would be a poor, ill-conditioned basis. Scaling the weights to `N(0, 1/d)` keeps `<w,x>` at
`O(1)`, which is the standard NTK / He-style scaling that makes the random features a sensible, isotropic
sketch of the input space. The nonzero bias `c ~ N(0,1)` shifts each ReLU's kink off the origin so the
features tile the input distribution rather than all firing on the same half-space. Second, the output
normalization: `1/sqrt(M)` rather than the mean-field `1/M`. This is the load-bearing difference between
this rung and the next. With `1/sqrt(M)` and unit-scale readout, each of the `M` features contributes
`O(1/sqrt(M))` to the output and the readout can move an `O(1)` amount while the *first-layer* weights, if
they were trainable, would barely move — that is precisely the linearization-around-init that defines the
lazy regime. I am choosing the parametrization that keeps the network a fixed kernel machine; that choice
is the whole point of the baseline. Third, the readout initialization: I start `a` at zero. Since the
features are fixed and the objective is convex in `a`, the init only affects the transient, and starting at
zero means the network outputs `0` before training and the readout grows toward the least-squares solution
— clean and reproducible.

The optimizer and step size follow from the same logic. The readout sees a fixed feature Gram matrix
`Phi^T Phi / b` whose scale is set by the feature variance; with `1/sqrt(M)` normalization and `M=100`
features, a learning rate of `0.5` on the readout would be far too large and the online least-squares
iterate would oscillate or diverge. The right scale is to shrink the readout step by the feature
dimension, so I use `lr = 0.5 / d` on the trainable parameters only — `get_optimizer` filters to
`requires_grad` parameters (just the readout, since the first layer is frozen), and plain SGD with no
momentum keeps the iterate a faithful online gradient step on the convex readout problem. The
`train_step` is the unchanged square-loss-on-a-fresh-batch step: zero the gradients, forward, square
error, backward, step, return the float loss. There is no auxiliary loss, no alternation, nothing — the
simplicity is the baseline.

Now the part that actually matters: reasoning about what this floor can and cannot do, because that is the
entire purpose of running it. A fixed feature map `phi` defines a kernel `K(x,x') = E_{w,c}[phi_w(x)
phi_w(x')]`, and online least-squares onto `M` random features approximates ridgeless regression in the
RKHS of `K`. The question for each target is whether `h*` lies in the span that `M=100` random features
plus `n = 6·10^5` samples can resolve. The fixed-feature lower bounds answer it sharply: a degree-`k`
sparse parity over an *unknown* subset `I` requires `min(n, q) = Omega(d^k)` for any feature map of
dimension `q`. Here `q` is bounded by the number of features `M=100` *and* by the sample budget `n`, and
`d=100`, so the relevant thresholds are `d^1 = 100`, `d^2 = 10^4`, `d^3 = 10^6`. My budget `n = 6·10^5`
sits *below* `d^3`, and my feature count `M = 100` sits at `d^1`. So the prediction is concrete and
per-degree. The degree-1 component (the lone `z1` term in `h1`) is reachable — a linear function of a
single coordinate is in the span of even a modest random-feature basis with `O(d)` samples. The degree-2
components (the `z1z2` term in `h1`, all three terms of `h2`) sit right at the `d^2 = 10^4` threshold: my
sample budget clears it, but with only `M=100` features the random basis cannot *resolve* a degree-2
monomial over an unknown pair of coordinates, because a generic random-feature map needs `Omega(d^2)`
features to have a degree-2 product over a specific unknown pair in its effective span. So even the
degree-2 pieces are out of reach at `M=100`. The degree-3 components (the `z1z2z3` term in `h1`, all of
`h3`) need `d^3 = 10^6` — both my sample budget and my feature count are below that, so they are flatly
unreachable. The net prediction: the lazy baseline recovers at best the degree-1 piece of `h1` and is
essentially at the trivial predictor on everything else.

What does "the trivial predictor" score? Each target has unit-magnitude monomials and zero mean, so the
constant predictor `0` has `test_mse = E[h*(x)^2] = (number of monomials)`. For `h1` that is
`||z1||^2 + ||z1z2||^2 + ||z1z2z3||^2 = 3`; if the lazy machine grabs the degree-1 term it removes one
unit and lands near `test_mse ≈ 2`, with `score = exp(-2) ≈ 0.13`. For `h2` (three degree-2 monomials)
the constant predictor gives `test_mse = 3` and `score = exp(-3) ≈ 0.05`, and since no degree-2 piece is
resolvable the lazy machine should sit right there. For `h3` (a single degree-3 monomial) the variance is
`1`, so `test_mse ≈ 1` and `score = exp(-1) ≈ 0.37` — note that `h3`'s score looks the *highest* of the
three precisely because its target is the *smallest* in magnitude, not because the method learns anything;
it is the trivial-predictor score of a unit-variance target. This is the trap the geometric-mean
aggregate is built to expose, and it is worth naming now so I read the later rungs correctly: a high
`score_h3` is *not* evidence of learning `h3`; only a `score_h3` close to `1` (i.e. `test_mse` near zero)
would be. The Fourier-recovery metric is the honest tell — it measures `|hat_S(model) - hat_S(h*)|`
directly, so a lazy machine that recovers nothing will show recovery close to the full coefficient
magnitude (near `1` per monomial), regardless of how flattering the `score` looks.

So I expect the lazy baseline to be the floor by construction, and a particular *shape* of floor:
near-trivial `test_mse` on all three targets, Fourier recovery close to its worst possible value, and the
deceptively-high `score_h3` that comes from `h3`'s small target magnitude rather than from learning. The
one place it might show life is the degree-1 piece of `h1`, which should pull `test_mse_h1` slightly below
the full constant-predictor value — a small, real signal that this is a regression onto features, not pure
noise. Everything else should be stuck, and stuck for the structural reason the fixed-feature bounds name:
the feature map is frozen *before* it knows which coordinates are latent, so it cannot put its limited
capacity where the signal is.

The diagnosis this rung is built to deliver is therefore already pointed at the next step. The failure is
not a learning-rate or an optimizer problem — the readout regression is convex and converges. It is that
the *features cannot move toward the latent subset `I`*. The fix is not a better optimizer on the same
frozen features; it is to *unfreeze* the features — to leave the lazy regime and put the network in the
parametrization where the first-layer weights travel an `O(1)` distance and can rotate toward the signal
coordinates. That is exactly the mean-field scaling, and it is the next rung: keep the two-layer shape but
swap the `1/sqrt(M)` lazy normalization for the `1/M` mean-field one, unfreeze the first layer, and use
plain SGD with a large step so the weights actually move. I expect that to do strictly better than the
floor on `h1`, because feature learning can climb the leap-1 staircase that no fixed feature map can
reach — and I expect it to *still* fail on `h2` and `h3`, because feature learning alone climbs only one
coordinate at a time. But that is the next rung's question. Here the only claim is the floor: with the
features frozen, the network is a kernel machine that cannot adapt to the hidden subset, and on
high-degree sparse targets a non-adaptive kernel machine cannot escape the trivial predictor at this
budget. The distilled fill — frozen ReLU random features, `1/sqrt(M)` output, readout-only SGD — is in
the answer.
