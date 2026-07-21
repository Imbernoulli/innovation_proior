The recipe is the whole point, but it fills the empty slots of a fixed MLP harness, and with the most
naive fill that harness *is* the floor — so the place to start is the textbook recipe and the question
of whether plain gradient descent, left to its own devices, moves the first layer toward `theta*` at
all. I have a two-layer ReLU net, a single-index target `y = g(<theta*, x>) + noise`, isotropic
Gaussian inputs, and four callbacks to fill: how to initialise the two layers, what optimiser to
build, what one mini-batch update does, and an optional post-hoc refit. The default fill answers all
four in the most ordinary way, and I want to understand precisely what that ordinary choice can and
cannot do, because that is what every later rung will have to beat.

Initialisation first. The first layer is a `Linear(d, W)` with `W = 256`, `d = 100`. The standard
choice for a layer feeding a ReLU is Kaiming-normal: draw each weight `~ N(0, 2/d)` so the
pre-activation variance is preserved through the ReLU (the factor 2 compensates for ReLU killing half
its inputs). Biases at zero. The scale matters later, so pin it now: each row has `d = 100` entries of
variance `2/d = 0.02`, so `E||w_j||^2 = 2` and every row sits at norm `sqrt(2) ~ 1.414`. The readout
`fc2` is a `Linear(W, 1)`; I give it a small uniform init in `[-1/sqrt(W), 1/sqrt(W)]`, the usual
`1/sqrt(fan_in)` scale, so the network output starts at `O(1)` variance rather than exploding. There
is nothing single-index-aware here — the rows of `fc1` point in `W` independent random directions on
roughly the sphere of radius `sqrt(2)`, each essentially orthogonal to `theta*`: a random unit vector
has correlation `<w_j/||w_j||, theta*> ~ N(0, 1/d)`, so the typical overlap at init is `1/sqrt(d) =
0.1` — every row starts about eighty-four degrees away from the truth. At init the network has no idea
where `theta*` is, and the entire burden of finding the direction falls on the optimiser.

I need to know what "found nothing" looks like on the scorer, because the harness reads the direction
off a fixed estimator and its chance level is my zero. The estimator is `theta_hat = normalize(sum_j
|a_j| w_j)`. At init the along-`theta*` component of the un-normalised sum is `sum_j |a_j| <w_j,
theta*>`, a sum of `W` mean-zero terms with total standard deviation `sqrt(sum_j |a_j|^2 ||w_j||^2 /
d)`, while the whole vector `sum_j |a_j| w_j` has squared norm `~ sum_j |a_j|^2 ||w_j||^2` because the
rows are near-orthogonal. The ratio is `~ sqrt(1/d) = 0.1`. So a network that has learned nothing
scores `direction_recovery ~ 0.1`, and anything the optimiser earns has to climb from that floor
toward `1`. That single number is my yardstick for every link: recovery near `0.1` means the rows
never left the equator, near `1` means they reached the pole.

The optimiser is the second slot. Plain SGD with momentum `0.9`, learning rate `1e-2`, no weight decay
— the config defaults. The third slot, `training_step`, is the canonical supervised loop: zero the
grads, forward the mini-batch, mean-squared error against the noisy targets, backward, step. Batch size
`256`, a fresh i.i.d. index each of `8000` steps, so over a run the optimiser sees `8000 x 256 =
2,048,000` sample-uses from `n_train = 32768` — about `62.5` passes over the data. That budget is
generous in epochs but, as I am about to argue, the epoch count is not the binding constraint on the
hard link; the per-step signal-to-noise is. The optional `finalize` does nothing — vanilla SGD has no
closed-form refit stage.

Now the part that actually decides the outcome: does this find `theta*`? The honest answer depends
entirely on the information exponent `k` of the link, and I can see why from the gradient. For one
first-layer row `w_j`, the gradient under the squared loss is schematically `grad_{w_j} L = E[ 2 (f(x)
- y) a_j x 1{<w_j,x> + b_j >= 0} ]`. The piece carrying direction information is the correlation of the
target with the gated input, `E[ g(<theta*,x>) x 1{...} ]`; expand it against the Gaussian in Hermite
polynomials and the leading term along `theta*` is governed by the first non-vanishing Hermite
coefficient of `g` — the information exponent `k`. The whole difficulty ordering falls out of those
coefficients. By Stein's lemma the first Hermite moment is `mu_1 = E[g(u) u] = E[g'(u)]`. For ReLU,
`g'(u) = 1{u > 0}`, so `mu_1 = 1/2` — nonzero, `k = 1`. For `sign`, `mu_1 = E[|u|] = sqrt(2/pi) ~
0.798` — nonzero and even larger, `k = 1`. For `g = He_3/sqrt(6)` with `He_3(u) = u^3 - 3u`, the first
moment `E[(u^3 - 3u) u] = E[u^4] - 3 E[u^2] = 3 - 3 = 0` vanishes, the second vanishes by parity, and
the first non-vanishing term is the third, `c_3 = E[He_3^2]/sqrt(6) = 6/sqrt(6) = sqrt(6) ~ 2.449`, so
`k = 3`. That single split — `mu_1` present for relu and sign, `mu_1 = 0` for hermite — is the entire
story.

For `k = 1` the first Hermite moment `E[y x] = mu_1 theta*` is already nonzero, so the very first
gradient on every row has an `O(1)`-detectable component along `theta*`, scaled by `mu_1`. For `k = 3`
the first two moments vanish; the direction does not appear until the third-order term, which after
contracting against a row that overlaps `theta*` by only `1/sqrt(d)` is of size `~ d^{-(k-1)/2} =
d^{-1} = 0.01` — buried under the per-mini-batch sampling noise, which for a batch of `256` is `~
sqrt(d/256) = sqrt(100/256) ~ 0.625`. That is the crux, and it is worth making quantitative across all
three links. On the easy links the per-step direction signal is `O(mu_1)` — `0.5` for relu, `0.8` for
sign — against `~ 0.625` noise, so the SNR is order one every single step, and the row drifts steadily
toward the pole; momentum's smoothing on top of an already-visible signal is the well-conditioned
regime it was built for. On hermite the signal is `~ 0.01` against `~ 0.625`, an SNR of about `0.016`
— forty times smaller than the noise it is buried in. Two orders of magnitude of SNR separate the easy
and hard links, and it is not a tuning gap; it is set by `k` and `d` and nothing in the default recipe
touches it.

Seen as a per-step walk, the asymmetry is stark. On relu a single step lands the drift somewhere in
`0.5 +- 0.625` — positive more often than not, so averaged over a handful of steps the overlap
ratchets upward and the row homes on the pole. On hermite the step lands in `0.01 +- 0.625` — the sign
of the update is essentially a coin flip, and the next step draws a fresh independent coin. Forty steps
of a coin flip do not add up to a nudge of size `0.01`; they add up to a random walk whose standard
deviation grows like `sqrt(steps) x 0.625` while the signal grows like `steps x 0.01`, and the walk
dominates until `steps` is enormous. That is the `n ~ d^2` wall of Ben Arous, Gheissari & Jagannath,
now as a per-step trace: the crossover where accumulated signal overtakes accumulated noise is far past
the budget when each step carries only `256` samples.

There is a second metric in the harness — the noise-free `test_mse` — and it separates "found the
direction but cannot represent the link" from "found nothing," so I should predict it too. The
reference is the trivial predictor that outputs the target mean, with test MSE `Var(y)`. For relu
`Var(ReLU(u)) = 0.5 - (1/sqrt(2 pi))^2 ~ 0.341`; for sign `Var(sign(u)) = 1`; for the `He_3` link
`Var(y) = E[g^2] = 1`. On relu, once the direction is found the wide `W = 256` ReLU bank easily
represents the (itself ReLU) link, so I expect `test_mse` pressed toward the small residual set by
`noise_std = 0.1` — well under `0.341`. On sign there is a representation obstruction on top of the
direction problem: `sign` is a step, and approximating a discontinuity with a finite bank of ReLU ramps
leaves an irreducible Gibbs-like residual near the jump, so even a run that recovers the direction
shows a `test_mse` floor above relu's, and a run that misses it sits near `Var = 1`. On hermite,
capacity is not the issue — `256` features are ample for a cubic-ish link — so a high `test_mse` there
is a *direction* failure telegraphing itself through the loss. The two metrics move together on the
hard links: low recovery with high `test_mse` is the signal-aggregation failure, not a capacity failure
I could fix by widening the net.

I should rule out the obvious knobs before I accept this floor, or I would be under-tuning the control
rather than measuring an honest ceiling. A larger `base_lr` multiplies the whole gradient, signal and
noise alike; on hermite that scales the `0.016` SNR by exactly one — it never improves, and what a big
step does buy is faster diffusion of the rows and a head that overfits the noisy targets, which pushes
recovery down. Running longer is the same story: each of the `8000` steps is an independent draw of
essentially the same noise around a sub-threshold mean, so more steps accumulate more walk, not more
signal. Momentum is the one part that integrates across steps: an exponential average with `beta = 0.9`
has an effective window of `1/(1 - 0.9) = 10` batches, `2560` samples' worth of gradient. But the
third-order signal needs `n ~ d^2 = 10^4` samples to clear its own empirical noise, and `2560 < 10^4`,
so momentum averages the noise down a little but falls short by a factor of four. None of the in-recipe
knobs move hermite.

The one knob that would actually help is the batch size — precisely the knob the harness locks. The
third-order signal clears its empirical noise `~ sqrt(d/B)` only when `B ~ d^2 = 10^4`; a batch of
`256` is nearly two orders of magnitude short, and raising it to `10^4` per step would surface the
signal. Summing the gradient over the full `n_train = 32768` would cut the noise to `~ sqrt(d/32768) ~
0.055`, finally below the `0.01`-scale signal once the dimension factor is folded in, since the
third-order term needs `n ~ d^2 = 10^4 < 32768` to be seen. But `batch_size = 256` is fixed in the
config, and `training_step` is handed a `256`-row mini-batch with no say in its size. The only place
the full `n_train` sits in one tensor is the `finalize` callback, which vanilla SGD leaves empty. That
is the structural reason the aggregation remedy cannot live in `training_step` and must wait for a rung
that does its work in `finalize` — the harness has partitioned the data exactly so mini-batch SGD sees
too little at a time.

There is a second failure mode beyond the raw signal size, and it is why sign is a lottery rather than
a uniform stall. The first layer and the readout are trained jointly, so the high-dimensional direction
search and the one-dimensional link fit happen at once and interfere. Picture a sign run whose rows
carry a little early overlap: the head, trained jointly, immediately fits the current features to the
discontinuous target as best it can, putting large weights on whichever rows happen to straddle `u = 0`
and locking those rows into place through the loss even though their directions are still mostly
off-`theta*`. The rows that would have rotated toward the truth are now anchored by a head already
exploiting them, and the joint dynamics settle into a basin that fits the training targets without the
rows ever aligning. Nothing in the recipe says "find the direction first, fit the link second"; the
faster one-dimensional fit freezes the slower high-dimensional search in place, and whether a run
escapes depends on whether its lucky rows carried enough overlap before the head locked on.

I could instead have *frozen* the first layer entirely — the random-feature / kernel reading of Rahimi
& Recht and Jacot et al., train only the head on fixed random features. Worth ruling out precisely
because it is the opposite extreme. If `fc1` never moves, the rows stay at their random init forever,
and the estimator `normalize(sum_j |a_j| w_j)` reads a readout-weighted sum of *fixed random* rows —
pinned at overlap `~ 0.1`, the chance floor, no matter how well the head fits. A random-feature model
can drive `test_mse` down by memorising but structurally cannot recover `theta*`, because recovery
lives in the first layer and the first layer is frozen. So training `fc1` is not incidental; it is the
only way the direction can be found at all, and the control keeps it. The interesting failure is not
"the rows are frozen" but "the rows are free yet the signal that should move them is below the noise."

So at step 1 my edit is the trivial one: leave the scaffold at its default. Kaiming first layer, small
uniform readout, SGD with momentum on all parameters, mean-squared-error mini-batch updates, no
finalize (the distilled module is in the answer). It is the floor by construction — no single-index
structure is used, the direction is left entirely to whatever the joint SGD dynamics extract from
`256`-sample gradients. I am not sandbagging the control: its two live knobs, step size and step count,
cannot surface the hard-link signal, and the opposite extreme of freezing `fc1` cannot recover the
direction at all.

What I will watch is the `k`-split, read against the `0.1` chance floor and the `1.0` ceiling. I
expect `relu-d100` near-perfect (the easy regime where vanilla SGD is already good), `sign-d100`
middling and high-variance (`k=1` but non-smooth, the direction present but the landscape rough), and
`hermite-d100` clearly the weakest (`k=3`, the third-order signal drowned at SNR `~ 0.016`), so the
mean score is dragged down by the two hard links while the easy one pins near `1`. Whatever the precise
numbers, the diagnosis they support is already a *landscape and signal-aggregation* problem, not a
learning-rate problem: the first structural fix is to stop letting the first layer and the head fight
each other, and the deeper fix is to aggregate the weak per-step signal across the whole dataset rather
than one `256`-sample slice at a time.
