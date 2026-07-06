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
its inputs). Biases at zero. Let me make the scale concrete, because it matters later: each row has
`d = 100` entries of variance `2/d = 0.02`, so `E||w_j||^2 = d * 2/d = 2` and every row sits at norm
`sqrt(2) ~ 1.414`. The readout `fc2` is a `Linear(W, 1)`; I give it a small uniform init in
`[-1/sqrt(W), 1/sqrt(W)]`, the usual `1/sqrt(fan_in)` scale, so the network output starts at `O(1)`
variance rather than exploding. There is nothing single-index-aware here — the rows of `fc1` point in
`W` independent random directions on roughly the sphere of radius `sqrt(2)`, each one essentially
orthogonal to `theta*`. Quantify that orthogonality: a random unit vector has correlation
`<w_j/||w_j||, theta*> ~ N(0, 1/d)`, so the typical overlap at init is `1/sqrt(d) = 1/10 = 0.1` — every
row starts about eighty-four degrees away from the truth. So at init the network has no idea where
`theta*` is; every feature is a random projection, and the entire burden of finding the direction falls
on the optimiser.

Before I even trust the optimiser, I should know what "found nothing" looks like on the scorer, because
the harness reads the direction off a fixed estimator and I need its chance level as my zero. The
estimator is `theta_hat = normalize(sum_j |a_j| w_j)`. At init the along-`theta*` component of the
un-normalised sum is `sum_j |a_j| <w_j, theta*>`, a sum of `W` mean-zero terms with total standard
deviation `sqrt(sum_j |a_j|^2 ||w_j||^2 / d)`, while the whole vector `sum_j |a_j| w_j` has squared
norm `~ sum_j |a_j|^2 ||w_j||^2` because the rows are near-orthogonal. The ratio is `~ sqrt(1/d) =
0.1`. So a network that has learned nothing scores `direction_recovery ~ 0.1`, and anything the
optimiser earns has to climb from that `0.1` chance floor toward `1`. That single number is my
yardstick for every link: recovery near `0.1` means the rows never left the equator, recovery near `1`
means they reached the pole.

The optimiser is the second slot. Plain SGD with momentum `0.9`, learning rate `1e-2`, no weight decay —
the config defaults. The third slot, `training_step`, is the canonical supervised loop: zero the grads,
forward the mini-batch, mean-squared error against the (noisy) targets, backward, step. Batch size
`256`, a fresh i.i.d. index each of `8000` steps, so over a run the optimiser sees `8000 x 256 =
2,048,000` sample-uses drawn from `n_train = 32768` — about `62.5` passes over the data. That budget is
generous in epochs but, as I am about to argue, the epoch count is not the binding constraint on the
hard link; the per-step signal-to-noise is. The optional `finalize` does nothing — vanilla SGD has no
closed-form refit stage.

Now the part that actually decides the outcome: does this find `theta*`? The honest answer depends
entirely on the information exponent `k` of the link, and I can see why from the gradient. Consider one
first-layer row `w_j`. Its gradient under the squared loss is, schematically,
`grad_{w_j} L = E[ 2 (f(x) - y) a_j x 1{<w_j,x> + b_j >= 0} ]`. The piece that carries information
about the direction is the correlation of the target with the gated input, `E[ g(<theta*,x>) x
1{...} ]`. Expand this against the Gaussian in Hermite polynomials and the leading term that points
along `theta*` is governed by the first non-vanishing Hermite coefficient of `g` — the information
exponent `k`. Let me actually compute those coefficients for the three links, because the whole
difficulty ordering falls out of them. By Stein's lemma the first Hermite moment is `mu_1 = E[g(u) u] =
E[g'(u)]`. For ReLU, `g'(u) = 1{u > 0}`, so `mu_1 = E[1{u>0}] = 1/2` — nonzero, `k = 1`. For `sign`,
`mu_1 = E[sign(u) u] = E[|u|] = sqrt(2/pi) ~ 0.798` — nonzero, even larger, `k = 1`. For the `He_3`
link `g = He_3/sqrt(6)` with `He_3(u) = u^3 - 3u`, the first moment `E[(u^3 - 3u) u] = E[u^4] - 3 E[u^2]
= 3 - 3 = 0` vanishes, the second Hermite moment vanishes by parity, and the first non-vanishing term
is the third, `c_3 = E[g He_3] = E[He_3^2]/sqrt(6) = 3!/sqrt(6) = 6/sqrt(6) = sqrt(6) ~ 2.449`, so
`k = 3`. That single split — `mu_1` present for relu and sign, `mu_1 = 0` for hermite — is the entire
story.

For `k = 1` the first Hermite moment `E[y x] = mu_1 theta*` is already nonzero, so the very first
gradient on every row has an `O(1)`-detectable component pointing along `theta*`, scaled by `mu_1`. For
`k = 3` the first *two* Hermite moments vanish; the direction does not appear in the gradient until the
third-order term, which after contracting against a row that overlaps `theta*` by only `1/sqrt(d)` is of
size `~ d^{-(k-1)/2} = d^{-1}` — a hundredth — buried under the per-mini-batch sampling noise, which for
a batch of `256` is `~ sqrt(d/256) = sqrt(100/256) ~ 0.625`. That is the crux: for the hard link the
per-step gradient signal is essentially noise, and chasing it with a small learning rate just rattles
the rows around their random init.

Let me put the two `k = 1` links through the same noise arithmetic so the asymmetry is quantitative, not
just qualitative. On the easy links the per-step direction signal is `O(mu_1)` — `0.5` for relu, `0.8`
for sign — against the same `~ 0.625` mini-batch noise, so the signal-to-noise ratio is order one, near
or above unity every single step, and the row drifts steadily toward the pole; momentum's smoothing on
top of an already-visible signal is exactly the well-conditioned regime it was built for. On hermite the
signal is `~ 0.01` against noise `~ 0.625`, an SNR of about `0.016` — the direction pull is roughly
forty times smaller than the noise it is buried in. Two orders of magnitude of SNR separate the easy
and hard links, and it is not a tuning gap; it is set by `k` and `d` and nothing in the default recipe
touches it.

There is a second, subtler reason vanilla SGD underperforms on the hard link even when it does find
*some* direction: the first layer and the readout are coupled the whole way through. Every row is free
and trained jointly with the head, so the high-dimensional direction search and the one-dimensional
link fit happen at once and interfere — the head chases whatever the rows currently encode while the
rows are still near the equator, and the joint dynamics can stall in configurations that fit the noisy
targets locally without aligning to `theta*`. Nothing in the default recipe decouples these two jobs,
and nothing aggregates the weak per-step direction signal across the whole dataset; each step sees only
its `256`-sample slice.

Let me trace a single row through one step to feel the difference rather than assert it. Take a row at
init with overlap `m = 0.1`, unit-scaled for the trace. On relu the population drift along `theta*` is
`~ mu_1 = 0.5` and the mini-batch estimate of it fluctuates by `~ 0.625`, so a single step lands the
drift somewhere in `0.5 +- 0.625` — positive more often than not, and averaged over even a handful of
steps the row's overlap ratchets upward; after tens of steps the `m`-dependent restoring geometry takes
over and the row homes on the pole. On hermite the population drift is `~ m^2 = 0.01` inside the same
`+- 0.625` fluctuation, so a single step lands in `0.01 +- 0.625` — the sign of the update is
essentially a coin flip, `0.01` is lost in the second decimal of the noise, and the next step draws a
fresh independent coin. Forty steps of a coin flip do not add up to a nudge of size `0.01`; they add up
to a random walk whose standard deviation grows like `sqrt(steps) x 0.625` while the signal it is
supposed to reveal grows like `steps x 0.01`, and the walk dominates until `steps` is enormous. That is
the same `n ~ d^2` wall again, now as a per-step trace: the crossover step count where accumulated
signal overtakes accumulated noise is far past the budget when each step only carries `256` samples.

There is a second signal in the harness beyond direction recovery — the noise-free `test_mse` — and I
should predict it too, because it separates "found the direction but cannot represent the link" from
"found nothing." The reference point is the trivial predictor that outputs the target mean, whose test
MSE equals `Var(y)`. For relu `Var(ReLU(u)) = E[ReLU^2] - E[ReLU]^2 = 0.5 - (1/sqrt(2 pi))^2 ~ 0.5 -
0.159 = 0.341`; for sign `Var(sign(u)) = 1`; for the `He_3` link `Var(y) = E[g^2] = 1`. A `test_mse`
far below these means the network genuinely fit the link, not just its mean. On relu, once the direction
is found the wide `W = 256` bank of ReLU features easily represents the (itself ReLU) link, so I expect
`test_mse` pressed toward the small residual set by the `noise_std = 0.1` targets seen in training —
well under `0.341`. On sign there is a representation obstruction on top of the direction problem: `sign`
is a step, and approximating a discontinuity at `u = 0` with a finite bank of ReLU ramps leaves an
irreducible residual near the jump — a Gibbs-like error no finite-width spline removes — so even a run
that recovers the direction should show a `test_mse` floor visibly above relu's, and a run that misses
the direction should sit up near `Var = 1`. On hermite, capacity is not the issue — `256` features are
ample for a cubic-ish link — so a high `test_mse` there would be a *direction* failure telegraphing
itself through the loss, exactly the `k = 3` stall. So the two metrics should move together on the hard
links and give me a cross-check: low recovery with high `test_mse` is the signal-aggregation failure I
am predicting, not a capacity failure I could fix by widening the net.

So I can predict the split before running it. On `relu-d100` (`k=1`) the first moment is large, the
direction signal is present every step at SNR near one, and ordinary SGD should drive
`direction_recovery` close to `1` and `test_mse` to near the noise floor — this is the regime where the
textbook recipe genuinely works, and any later method will at best tie it here. On `sign-d100` (`k=1`
but non-smooth) the first moment is actually *larger* (`0.8 > 0.5`), so the direction signal is if
anything stronger, but the link is a discontinuous step and the readout has to approximate that step
from a bank of ReLU features whose thresholds are wherever the joint dynamics leave the trained biases;
the squared-loss landscape is rougher, and I expect partial recovery — better than the `0.1` chance
floor, sometimes much better, but not reliably near `1`, with high seed-to-seed variance depending on
whether a given run's random rows happened to carry enough early overlap for the head to lock onto. On
`hermite-d100` (`k=3`) I expect the worst: at SNR `~ 0.016` the rows barely move toward `theta*`, and
recovery should land well short of `1` — this is exactly the `n = Theta(d^{k-1})` wall of Ben Arous,
Gheissari & Jagannath, seen from the per-mini-batch side rather than the sample-count side.

I should be disciplined about the obvious knobs before I accept this floor, because if a larger step or
a longer run rescued hermite I would be under-tuning the control rather than measuring an honest
ceiling. Take the learning rate first. A larger `base_lr` multiplies the whole gradient, signal and
noise alike; on hermite the signal is `~ 0.01` and the noise `~ 0.625`, so scaling the step by any
factor scales that `0.016` SNR by exactly one — it never improves. What a big step does buy is faster
diffusion of the rows around their random init and a head that overfits the noisy targets in the
non-convex wide-ReLU landscape, which pushes recovery down, not up. So the learning rate is not a lever
on the hard link. Running longer is the same story from a different angle. Each of the `8000` steps is
an independent draw of essentially the same noise around a sub-threshold mean, so the row's trajectory
is a random walk on the sphere with a per-step drift far below the step-to-step jitter; more steps
accumulate more walk, not more signal, because a fresh i.i.d. batch each step means there is no fixed
sub-threshold signal being integrated — the mean pull is below the floor and stays there. Momentum is
the one part of the default that does integrate across steps, so I should check it too: an exponential
average with `beta = 0.9` has an effective window of `1/(1 - 0.9) = 10` batches, i.e. it aggregates
about `10 x 256 = 2560` samples' worth of gradient. But the third-order signal needs `n ~ d^2 = 10^4`
samples to clear its own empirical noise, and `2560 < 10^4`, so even momentum's smoothing falls short by
a factor of four on the hard link — it averages the noise down a little but never far enough to surface
a `k = 3` signal. So none of the in-recipe knobs move hermite; the control is at its honest ceiling.

The one knob that would actually help hermite is the batch size — and it is precisely the knob the
harness locks. My own noise arithmetic says the third-order signal `~ 0.01` clears its empirical noise
`~ sqrt(d/B)` only when `B ~ d^2 = 10^4`; a batch of `256` is nearly two orders of magnitude short, and
raising it to `10^4` per step would surface the signal. But `batch_size = 256` is fixed in the config,
and `training_step` is handed a `256`-row mini-batch with no say in its size, so within the per-step
loop I simply cannot assemble the `10^4` samples the signal needs. The only place the full `n_train =
32768` sits in one tensor is the `finalize` callback, which vanilla SGD leaves empty. That is the
structural reason the aggregation remedy cannot live in `training_step` and must wait for a rung that
does its work in `finalize` — the harness has partitioned the data exactly so that mini-batch SGD sees
too little at a time, and the full set is reachable only through the refit hook the control declines to
use.

I should also name concretely how the coupling stalls a run that has *some* signal, because it is not
just the hermite case. Picture a sign run whose rows carry a little early overlap. The head, trained
jointly, immediately fits the current features to the discontinuous target as best it can — and because
the target is a step, the least-squares head puts large weights on whichever rows happen to straddle
`u = 0`, locking those rows into place through the loss even though their directions are still mostly
off-`theta*`. The rows that would have rotated toward the truth are now anchored by a head that is
already exploiting them, and the joint dynamics settle into a local basin that fits the training targets
without the rows ever aligning. Nothing in the recipe says "find the direction first, fit the link
second"; the two happen at once and the faster one-dimensional fit freezes the slower high-dimensional
search in place. That is the mechanism behind the seed lottery I expect on sign: whether a run escapes
depends on whether its lucky rows carried enough overlap *before* the head locked on.

I could also have gone the other way and *frozen* the first layer entirely — the random-feature / kernel
reading of Rahimi & Recht and Jacot et al., train only the head on fixed random features. That is worth
ruling out precisely because it is the opposite extreme of the default. If `fc1` never moves, the rows
stay at their random init forever, and the direction estimator `normalize(sum_j |a_j| w_j)` reads off a
readout-weighted sum of *fixed random* rows — which, by the same chance-level computation as at init, is
pinned at overlap `~ 1/sqrt(d) = 0.1` no matter how well the head fits. A random-feature model can drive
`test_mse` down by memorising, but it structurally *cannot* recover `theta*`, because recovery lives in
the first layer and the first layer is frozen. So the default's choice to train `fc1` is not incidental;
it is the only way the direction can be found at all, and the control keeps it. The interesting failure
is therefore not "the rows are frozen" but "the rows are free yet the signal that should move them is
below the noise" — which is the hermite case.

Let me make the noise count fully concrete, because it is the whole reason the hard link is hard and it
sets the bar for everything that follows. The relevant order parameter for any row is its correlation
with the truth, `m = <w_j/||w_j||, theta*>`. At a random init `m ~ 1/sqrt(d) ~ 0.1` for `d = 100` —
every row starts microscopically close to the equator `{m = 0}`. Near the equator the population
gradient that would move a row toward `theta*` scales like `m^{k-1}`: for `k = 1` that is `m^0 = O(1)`
(present from the first step), but for `k = 3` it is `m^2 ~ d^{-1} ~ 0.01`. Against that I have the
empirical gradient noise of a mini-batch of `B = 256` samples, `~ sqrt(d/B) = sqrt(100/256) ~ 0.625`.
So on the hard link the signal-to-noise ratio per step is roughly `0.01 / 0.625 ~ 0.016` — the
direction signal is essentially invisible inside the mini-batch gradient, and momentum cannot
manufacture a signal that is not there; it only averages noise, and only over `2560` samples at that.
The remedy implied by the same arithmetic is to *aggregate*: summing the gradient over the full
`n_train = 32768` samples cuts the noise to `~ sqrt(d/n_train) = sqrt(100/32768) ~ 0.055`, finally
below the `0.01`-scale signal once the dimension factor is folded in, since the third-order term needs
`n ~ d^2 = 10^4 < 32768` samples to be seen. But vanilla SGD never does that — it crawls in `256`-sample
steps — so it leaves the hard-link signal on the table. That is the gap a later rung will have to close
by aggregating the weak signal across the whole dataset instead of one slice at a time; and there is a
second gap, the coupling, which a later rung will attack by decoupling the high-dimensional direction
search from the one-dimensional link fit so the gradient on each row at least points cleanly along
`theta*` when it does carry signal.

So at step 1 my edit is the trivial one: leave the scaffold at its default. Kaiming first layer, small
uniform readout, SGD with momentum on all parameters, mean-squared-error mini-batch updates, no
finalize. It is the floor by construction — no single-index structure is used, the direction is left
entirely to whatever the joint SGD dynamics extract from `256`-sample gradients, and the harness's own
direction estimator `normalize(sum_j |a_j| w_j)` reads off whatever alignment those rows happen to
acquire (the distilled module is in the answer). I am not sandbagging the control; I have checked that
its two live knobs — step size and step count — cannot surface the hard-link signal, and that the
opposite extreme of freezing `fc1` cannot recover the direction at all. This is the textbook recipe at
its genuine ceiling.

What I will watch is exactly the `k`-split, read against the `0.1` chance floor and the `1.0` ceiling. I
expect `relu-d100` near-perfect (the easy regime where vanilla SGD is already good), `sign-d100`
middling and high-variance (`k=1` but non-smooth, the direction present but the landscape rough), and
`hermite-d100` clearly the weakest (`k=3`, the third-order direction signal drowned at SNR `~ 0.016`).
The mean score is then dragged down by the two hard links while the easy one pins near `1`. Whatever the
precise numbers, the diagnosis they will support is already pointed at step 2: this is a *landscape and
signal-aggregation* problem, not a learning-rate problem, and the first structural fix is to stop
letting the first layer and the head fight each other — to decouple the non-parametric link fit from the
high-dimensional direction search so the search collapses onto the one direction that matters.
