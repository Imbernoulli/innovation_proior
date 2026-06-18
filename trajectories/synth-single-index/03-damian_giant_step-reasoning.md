Freezing the biases did exactly what the asymmetric prediction said, and the numbers confirm the
diagnosis rather than refute it. On `relu-d100` recovery stayed pinned at `0.998` — the easy `k=1`
regime is saturated and indifferent to the bias treatment, as expected. On `sign-d100` the mean moved
from vanilla's `0.210` to `0.467`, and the structure shifted: seed 456 jumped to `0.848` where the
clean landscape pulled it cleanly to a pole. But the spread did *not* tighten the way I hoped — seeds
came out {0.439, 0.113, 0.848}, with seed 123 actually *lower* than its vanilla value. So the
landscape collapse helps on average but the wide net's many independent rows, still driven by noisy
256-sample gradients, do not reliably converge to the same place; the benign scalar-flow picture is
realised only loosely when 256 rows each chase their own noisy overlap. And on `hermite-d100` the mean
went from `0.656` to `0.614` — essentially flat, even slightly down. That is the decisive reading.
Freezing the biases collapses the landscape, but on the `k=3` link the per-mini-batch direction signal
is `~ d^{-1}`, far below the 256-sample gradient noise `~ sqrt(d/256)`, and a cleaner landscape does
nothing about a signal that is buried below noise. The bias freeze was never going to fix hermite,
because the missing ingredient is not the landscape shape — it is *signal aggregation*. So the
hermite result is the one that dictates step 3: I have to stop crawling with noisy small-batch
gradients and instead extract the weak third-order signal by summing it over the entire training set.

Let me reason carefully about what that means, because it changes every callback. Go back to the
first-layer gradient at initialisation. With a symmetric/zero-output init the gradient on a row `w_j`
loses its self-interaction term and is the clean correlation `grad_{w_j} L = -2 a_j E[ f*(x) x
1{<w_j,x>+b_j >= 0} ]`. Expand `E[f*(x) x sigma'(<w,x>)]` against the Gaussian using Stein's lemma and
Hermite orthogonality, and it becomes an asymptotic series in `d^{-1/2}` whose terms shrink by a factor
`sqrt(d)` each: the leading informative term points each row *into* the relevant subspace — for a
single index, along `theta*` — and the first term that does so is governed by the link's information
exponent `k`. For `k=1` the first Hermite moment `E[y x] = mu_1 theta*` is the signal, of size `O(1)`
along `theta*` once aggregated. For `k=3` the first two moments vanish and the signal first appears in
the third-order term, of size `~ d^{-(k-1)/2} = d^{-1}` after contracting against a random probe that
overlaps `theta*` by only `1/sqrt(d)`.

Now the two facts that decide the algorithm. First, the empirical gradient fluctuates around the
population one with noise `~ sqrt(d/n)`; to see the leading `O(d^{-1/2})`-scale signal above its own
empirical noise the third-order term needs `n ~ d^2` samples. With `d = 100` that is `~10^4`, and a
single mini-batch of 256 has `256 << 10^4` — so a per-step mini-batch gradient on the hard link is
*essentially pure noise*, which is precisely the `0.614` hermite stall I just measured. The signal only
emerges if I aggregate the *whole* training set into one gradient: `n_train = 32768 > d^2`, comfortably
enough. So the update should be one **full-batch** step, not a long crawl of tiny noisy ones. Second,
the rows are unit vectors and I want to rotate them an `O(1)` amount toward `theta*`, but the gradient I
am stepping along has norm only `~ d^{-(k-1)/2}` for the hard link. A normal `O(1)` learning rate would
move the rows imperceptibly; to get an `O(1)` rotation the step has to *grow with the dimension* — the
"giant" step — with learning rate `eta_1 ~ d^{(k-1)/2}`. For `k=1` that is `O(1)` in `d` (relu, sign);
for `k=3` it grows like `d` (hermite). This is the same information-exponent story I have been tracking,
now seen from the step-size side: the deeper the direction is hidden in the series, the bigger the step
needed to surface it.

So the recipe inverts the whole loop. Instead of `8000` noisy mini-batch SGD steps, I want the
mini-batch loop to do *nothing* — a single noisy batch-256 first step is exactly the failure mode I
measured — and do the real work once, full-batch, in `finalize`, where the harness hands me the entire
`x_train, y_train`. So `make_optimizer` returns an SGD with learning rate zero (the loop changes
nothing), `training_step` is a no-op that only logs the batch target energy, and `finalize` carries the
giant step plus the readout fit.

Now I need the right *direction estimator* per link, read straight off the first non-vanishing term of
the gradient series. For `k=1` (relu, sign) the first moment is already informative:
`E[y x] = mu_1 theta*` by Stein on the first Hermite, so the empirical estimator is simply
`normalize( (1/n) sum_i y_i x_i )` — one matrix-free average over the full batch. The sign link is
`k=1` too (sign has a nonzero first Hermite coefficient), and crucially the construction only ever
touches `g` through its low-order Hermite *moments*, never through derivatives of `g` — so it is robust
to sign's non-smoothness, which is exactly why it should rescue the sign link where frozen-bias was
erratic. For `k=3` (hermite) the first moment estimates `mu_1 theta* = 0` and is useless; I must build
the third-order term. The multivariate third Hermite tensor contracted twice against a probe `v` gives
the vector `x <x,v>^2 - x ||v||^2 - 2 v <x,v>`, and `E[ y (that) ] ∝ C_3(v,v) = mu_3 <theta*,v>^2
theta*` — a vector along `theta*` whose strength grows with the overlap `<theta*,v>`. The empirical
estimator is the same average over the full batch. But the signal is `~ <theta*,v>^2`, so a fresh
random probe (`<theta*,v> ~ d^{-1/2}`) gives a weak contraction; I sharpen it with tensor power
iteration — once I have a rough direction, contract again *using that estimate as the probe*. The map
`v -> C_3(v,v)/||.||` has `theta*` as its attracting fixed point, so a couple of refinement passes
drive the overlap toward 1. Concretely: form a first contraction averaged over the current random rows
(probes), get a coarse direction, then refine with two power-iteration passes.

Now what do I do to the network with that direction? The theory says step the first layer along the
giant gradient, after which each row equals the (scaled) gradient feature — a vector along `theta*`.
In effect each row should be overwritten toward the estimated direction, by an amount set by the giant
learning rate: the larger `eta`, the more completely the random probe is replaced by the signal. I
write that as a convex mix with weight `mix = eta_1 / (eta_1 + 1)` — small `eta_1` leaves the row
mostly itself, large `eta_1` (the giant regime) pushes it almost entirely onto the direction. Here I
should be honest about how this grounds in the harness versus the cleanest analysis. The analysis pairs
the giant step with a matching weight decay `lambda_1 = eta_1^{-1}` so the `-W^{(0)}` term *exactly
cancels* the random init and the first layer becomes *purely* gradient features; the harness instead
realises the same "overwrite the random probe with the signal" idea through the convex mix, which for
the giant `eta_1` is numerically almost the same (with `eta_1 = sqrt(W) d^{(k-1)/2}` and `d=100`, the
hermite mix weight is essentially 1, so the rows are set almost entirely to the direction). It is the
same mechanism, expressed without the exact-cancellation bookkeeping.

Two refinements the construction needs, and both are in the harness. First, if I set *every* row to
exactly the same direction, the features degenerate — all neurons compute `ReLU(<theta*,x> + b_j)`,
which is a good basis only if the thresholds `b_j` differ and the rows are not literally identical. So
(a) I keep a small *orthogonal* jitter on each row — a random component projected off the direction and
renormalised — so the rows are a tight spread around `theta*` rather than identical, keeping the feature
matrix well-conditioned without re-adding off-subspace noise; and (b) I re-initialise the biases to a
spread of thresholds. With the rows now aligned to `theta*`, the functions `x -> ReLU(<theta*,x> + b_j)`
for a spread of `b_j` form a one-dimensional random-feature basis that can approximate any smooth
univariate link. The original biases (zero, from the init) would give a degenerate basis. A
deterministic spread `b_j = linspace(-2.5, 2.5, W)` cleanly covers the range of `u = <theta*,x> ~
N(0,1)`. Then renormalise each row to the sphere — ReLU is positively homogeneous, so only directions
and biases matter, and unit rows keep the readout problem well-scaled.

The giant learning rate carries one more harness-specific factor I must get right. The first-layer
gradient I derived has a factor `a_j` out front. The scaffold's readout normalisation puts
`a_j = 1/sqrt(W)` at init, so the *effective* first-layer gradient is shrunk by `1/sqrt(W)`; to keep
the same effective step size as the theory (which uses `a_j ~ +-1`) I multiply the learning rate by
`sqrt(W)`. So the full giant rate is `eta_1 = sqrt(W) * d^{(k-1)/2}` — `sqrt(W)` undoing the readout
normalisation, `d^{(k-1)/2}` the dimension scaling the information exponent demands. This is why
`init_two_layer` must set `fc2.weight` to the constant `1/sqrt(W)` (and zero biases): the giant rate is
calibrated to exactly that readout scale.

Last, the readout. After the giant step the first layer is frozen and the only thing left is to fit the
head on the fixed features `phi(x) = ReLU(W^{(1)} x + b)`. That is a convex linear least-squares
problem, and the theory fits it with norm-constrained / weight-decayed regression — i.e. ridge, which
has a closed form, so I do not run GD on the head at all. I solve
`beta = (Phi^T Phi + lambda I)^{-1} Phi^T y` where `Phi` is `[n, W+1]` (the ReLU features plus a
constant column for the head bias) and `lambda = max(weight_decay, 1e-4)` to keep the Gram matrix
invertible; set `fc2.weight = beta[:-1]`, `fc2.bias = beta[-1]`. This is exactly the harness's
`finalize` hook earning its keep — the closed-form refit that frozen-bias deliberately left as a no-op.
The whole method, then, lives entirely in `init_two_layer` (random probes, constant readout) and
`finalize` (estimate the direction, giant-overwrite the rows, spread the biases, ridge-fit the head),
with the mini-batch loop neutralised.

Now the falsifiable expectations against the frozen-bias numbers, which is the whole reason to run this.

On `relu-d100` (`k=1`) I expect to *match* the `0.998` ceiling, not beat it — the easy link is
saturated for every method, and the first-moment estimator is just the clean way to get there. If
giant-step came out below `0.998` on relu, something in the `k=1` first-moment path or the ridge fit
would be wrong.

On `sign-d100` (`k=1`, non-smooth) I expect the *largest* gain over frozen-bias's erratic `0.467`.
Sign is `k=1`, so the first-moment estimator `normalize(mean(y x))` aggregated over all 32768 samples
gives a high-overlap direction directly, and the construction never differentiates `g`, so the
non-smoothness is a non-issue. I expect recovery near `0.99` on every seed and, critically, the
seed-to-seed spread to *collapse* — no more {0.439, 0.113, 0.848} lottery, because the full-batch
first moment is a deterministic, low-variance estimate of `theta*`. That tightening is the cleanest
test that the failure was signal aggregation, not landscape.

On `hermite-d100` (`k=3`) this is the make-or-break. Frozen-bias was flat at `0.614` because the
third-order signal was below the mini-batch noise. The giant step aggregates that signal over `n_train
= 32768 > d^2 = 10^4` samples and refines it with tensor power iteration, so the third Hermite
contraction should now surface a high-overlap direction where SGD could not. I expect hermite recovery
to jump from `0.614` to near `0.98`+ — the single largest improvement on the ladder — with `test_mse`
falling well below frozen-bias's `0.28`. The power iteration is the part I would watch: if the random
probes' initial overlap is too small for the first contraction to catch (a `d^{-1/2}` overlap raised to
the `2` is `d^{-1}`, then averaged over 256 probes), two refinement passes should still pull it in, but
this is where the method could underperform on an unlucky seed. If hermite lands near 1 across all
three seeds, the diagnosis is confirmed end to end: the easy link was always solved, the sign link
needed only clean full-batch signal, and the hard link needed exactly the full-batch giant step plus
power iteration that the mini-batch crawl could never provide. Since direction recovery near 1 across
all three links is the ceiling of this benchmark, I do not expect a further published method to clearly
beat a giant step that already saturates every link — this rung is the endpoint, and the bar it must
clear is precisely frozen-bias's per-link numbers: match `0.998` on relu, sharply beat `0.467` on sign,
and convert the `0.614` hermite stall into near-perfect recovery.
