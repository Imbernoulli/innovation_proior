Freezing the biases did exactly what the asymmetric prediction said, and the numbers confirm the
diagnosis rather than refute it. Let me read the three links in order and turn each into a mechanism
conclusion before I decide what step 3 has to be. On `relu-d100` recovery stayed pinned at `0.998`
(seeds `0.998`, `0.998`, `0.998`) with `test_mse ~ 0.0016` — the easy `k=1` regime is saturated and
indifferent to the bias treatment, exactly as I expected; there is no signal to read from a link every
recipe already solves. The interesting information is entirely in the two hard links, and it is not
where I hoped it would be.

Take `sign-d100` first because it is where the frozen-bias move was supposed to pay off. The mean rose
from vanilla's `0.210` to `0.467` — more than a doubling — and seed 456 jumped all the way to `0.848`
where the collapsed landscape pulled that run cleanly toward a pole. So on average the landscape
argument was right: `k=1` sign only ever needed a benign landscape, and freezing the biases supplied
one. But I predicted the *spread* would tighten, and it did not. The per-seed values came out {`0.439`,
`0.113`, `0.848`}, a range of `0.735` — *wider* than vanilla's `0.377`-wide {`0.049`, `0.156`,
`0.426`}, with seed 123 at `0.113` actually *lower* than its vanilla value. So the coefficient of
variation barely moved (`0.64` now versus `0.75` before) while the absolute range grew. That is the
precise failure mode I flagged at the previous rung as the risk of applying a tied-architecture analysis
to a wide net: the benign scalar-flow picture is realised only *loosely* when `256` independent rows
each chase their own noisy `256`-sample overlap, so the collapsed landscape lifts the mean but cannot
force every run's rows onto the same pole. The mechanism is real but the wide net dilutes it — and
crucially, the residual randomness is still driven by the `256`-sample gradient noise, not by the
landscape shape.

Now `hermite-d100`, which is the decisive reading. The mean went from vanilla's `0.656` to `0.614` —
essentially flat, if anything slightly *down* — and the per-seed band {`0.496`, `0.671`, `0.674`} is a
tight cluster of mediocrity, standard deviation `0.083`, coefficient of variation `0.14`. So on the
hard link the freeze changed nothing about the outcome and nothing about the spread: every run lands in
the same low band, run after run. That consistency is itself the tell. Sign was a lottery (wild spread,
some runs win); hermite is a *uniform* wall (tight spread, every run stuck at the same height). A
lottery is a landscape symptom; a uniform wall is a signal-size symptom. Freezing the biases addressed
the landscape, so it moved the lottery link (sign, on average) and left the wall link (hermite) exactly
where it was. That is the clean separation the two rungs of numbers now hand me: the sign gain plus the
hermite flatness together say the bias freeze was a landscape fix, and hermite's problem is not the
landscape.

Let me put the SNR arithmetic on paper because it is the whole justification for what comes next. On the
`k=3` link the per-mini-batch informative gradient along `theta*` scales as `m^2` where `m ~ 1/sqrt(d)`
is the current overlap, so near the equator the signal is `~ d^{-1} = 0.01`. The empirical gradient
noise from a batch of `n` samples is `~ sqrt(d/n)`; for `n = 256`, `d = 100` that is `sqrt(100/256) =
0.625`. So the per-step signal-to-noise on hermite is `0.01 / 0.625 ~ 0.016` — the direction push is
forty times below the noise on *every* mini-batch step, which is exactly why hermite sits flat at
`0.614` regardless of how clean the landscape is. A cleaner landscape reshapes *where* the gradient
points; it does nothing about *how loud* it is. So the missing ingredient is not landscape shape — it is
signal aggregation, and that dictates step 3 entirely: I have to stop crawling with noisy small-batch
gradients and instead extract the weak third-order signal by summing it over the entire training set.

Before I build that, let me be disciplined and walk the design space, because there is more than one way
to "get more signal" and the tempting ones fail on arithmetic I should do now rather than after a wasted
run. There are four moves on the table. First, keep the mini-batch loop but raise the batch size — say
from `256` to a few thousand. The noise falls as `sqrt(d/n)`, so to drag the hermite SNR from `0.016`
up to order `1` I would need `n` such that `sqrt(d/n) ~ 0.01`, i.e. `n ~ d / 0.0001 = 10^6` — four
thousand times the current batch, and thirty times larger than the *entire* `n_train = 32768`. A bigger
*mini*-batch cannot reach it; only using the whole set at once even comes close. Second, raise the
learning rate to amplify the tiny push. But a larger `eta` multiplies signal and noise together — the
SNR is scale-invariant — so a big step on a sub-noise gradient just takes a big *random* step, which is
precisely the erratic sign lottery I want to kill, not cure. Third, train longer: the budget is fixed at
`8000` steps, and even unlimited steps of a zero-mean-dominated update is a random walk whose drift `~
0.016` per step is swamped by diffusion `~ 0.625 sqrt(steps)`; averaging `8000` such independent batches
gives an SNR gain of `sqrt(8000) ~ 89`, lifting `0.016` to `~1.4` — but only because the `8000` batches
are independent draws of the *same population gradient*, so this is really just a slow, high-variance way
of computing the full-batch average I could compute in one shot. That observation is the key: the loop
is already trying to average, badly. So the fourth move is the honest one — compute the population
gradient directly by aggregating the *whole* training set into a single full-batch step, where `n_train
= 32768`.

Is `32768` enough? The third-order term surfaces above its own empirical noise when `n ~ d^2 = 10^4`.
Here `n_train = 32768 = 3.3 d^2`, so `sqrt(n_train / d^2) = sqrt(3.28) ~ 1.8`: the full-batch signal
sits about `1.8x` above its aggregation-noise floor — comfortably detectable, not marginal. Contrast the
mini-batch: `sqrt(256 / d^2) = 0.16`, sixfold *below* the floor. The ratio between them is `sqrt(32768 /
256) = sqrt(128) = 11.3` — one full-batch pass is worth about eleven mini-batches' worth of SNR, and it
is the only single-pass way to clear the detection threshold at all. So the recipe should invert the
whole loop: the `8000` mini-batch steps should do *nothing*, and the real work should happen once,
full-batch, in `finalize`, where the harness hands me the entire `x_train, y_train`. Concretely
`make_optimizer` returns an SGD with learning rate zero so the loop changes nothing, `training_step` is
a no-op that only logs the batch target energy, and `finalize` carries the giant step plus the readout
fit. I cannot delete the loop — the driver always runs it — so neutralising it with `lr = 0.0` is the
faithful way to say "spend no update budget on noise".

Now I need the right *direction estimator* per link, read straight off the first non-vanishing term of
the first-layer gradient series. Go back to the gradient at a symmetric/zero-output init, where the row
gradient loses its self-interaction term and is the clean correlation `grad_{w_j} L = -2 a_j E[ f*(x) x
sigma'(<w_j,x>+b_j) ]`. Expand `E[f*(x) x sigma'(<w,x>)]` against the Gaussian with Stein's lemma and
Hermite orthogonality and it becomes an asymptotic series in `d^{-1/2}` whose terms shrink by `sqrt(d)`
each: the leading informative term points each row *into* `theta*`, and the order of that first term is
the link's information exponent `k`. For `k=1` the first Hermite moment `E[y x] = mu_1 theta*` is already
the signal, of size `O(1)` along `theta*` once aggregated. For `k=3` the first two moments vanish and
the signal first appears in the third-order term, of size `~ d^{-(k-1)/2} = d^{-1}` after contracting
against a random probe that overlaps `theta*` by only `1/sqrt(d)`.

So for `k=1` (relu, sign) the estimator is simply `theta_hat = normalize( (1/n) sum_i y_i x_i )` — one
matrix-free average over the full batch. The sign link is `k=1` too — its first Hermite coefficient
`mu_1 = E[sign(u) u] = E[|u|] = sqrt(2/pi) ~ 0.8` is nonzero, in fact *larger* than relu's — and
crucially the construction only ever touches `g` through its low-order Hermite *moments*, never through
derivatives of `g`. That is exactly why it should rescue the sign link where frozen-bias was erratic:
the sign discontinuity is invisible to a moment estimator, so the non-smoothness that made the SGD
landscape rough is simply not in the picture. The sign lottery was never a signal problem — `mu_1 ~ 0.8`
is a loud first moment — it was an aggregation-and-landscape problem, and a deterministic full-batch
first moment is a low-variance estimate that removes both the lottery and the wide-net dilution I saw at
`0.467`.

For `k=3` (hermite) the first moment estimates `mu_1 theta* = 0` and is useless, so I must build the
third-order term explicitly. The multivariate third Hermite tensor contracted twice against a probe `v`
gives the vector `x <x,v>^2 - x ||v||^2 - 2 v <x,v>`, and its `y`-weighted expectation is proportional
to `mu_3 <theta*,v>^2 theta*` — a vector along `theta*` whose strength grows with the overlap
`<theta*,v>`. Let me verify this by hand on the one case I can check in closed form, the aligned probe
`v = theta*`. Then `<x,v> = u` and `||v||^2 = 1`, so the contracted vector is `x u^2 - x - 2 theta* u`.
Project it on `theta*`: the coefficient is `u * u^2 - u - 2u = u^3 - 3u`, which is exactly the third
probabilist's Hermite polynomial `He_3(u)`. So the along-`theta*` expectation is `E[ g(u) He_3(u) ] =
mu_3` by Hermite orthogonality, and the components orthogonal to `theta*` vanish by the isotropy of the
Gaussian (they are odd in `x_perp`, which is independent of `u = <theta*,x>`). The contraction really
does return a vector along `theta*` with strength `mu_3` when `v = theta*` — the estimator is correct,
not just plausible.

But a *random* probe overlaps `theta*` by only `<theta*,v> ~ d^{-1/2} = 0.1`, and the signal scales as
`<theta*,v>^2 ~ d^{-1} = 0.01`, so a single fresh-probe contraction is weak — right at the edge of the
`1.8x` full-batch margin. I sharpen it with tensor power iteration: once I have a rough direction,
contract again *using that estimate as the probe*. The map `v -> normalize(C_3(v,v))` has `theta*` as
its attracting fixed point because the returned vector's overlap with `theta*` is monotone increasing in
the input overlap — feed in overlap `m`, get out a vector dominated by the `m^2 theta*` term plus a
perpendicular residual, whose normalised overlap exceeds `m` once `m` clears the noise floor. Starting
from `m_0 ~ 0.1`, one contraction already lifts the coherent part quadratically, and two refinement
passes drive the overlap toward 1. I average the first contraction over the `256` current random rows as
probes to beat down the per-probe variance, get a coarse direction, then run two power-iteration passes
on that. I do not want more than two: each pass costs a full `[n, d]` contraction, and once the overlap
is near 1 the map is essentially the identity, so further passes buy nothing but compute. Two is the
point where the coarse `0.1`-overlap start has been pulled in but I have not started paying for
diminishing returns.

Now what do I do to the network with that direction? The theory says step the first layer along the
giant gradient, after which each row equals the scaled gradient feature — a vector along `theta*`. In
effect each row should be overwritten toward the estimated direction, by an amount set by the giant
learning rate: the larger `eta`, the more completely the random probe is replaced by the signal. Two
facts fix the size of that rate. First, the rows are unit vectors and I want to rotate them an `O(1)`
amount toward `theta*`, but the gradient I am stepping along has norm only `~ d^{-(k-1)/2}` for the hard
link. A normal `O(1)` learning rate would move the rows imperceptibly; to get an `O(1)` rotation the
step has to *grow with the dimension* — the "giant" step — with `eta_1 ~ d^{(k-1)/2}`. For `k=1` (relu,
sign) that is `O(1)` in `d`; for `k=3` (hermite) it grows like `d`. This is the same information-exponent
story I have tracked all along, now seen from the step-size side: the deeper the direction is hidden in
the series, the bigger the step needed to surface it. Second, the derived first-layer gradient carries a
factor `a_j` out front, and the scaffold's readout normalisation puts `a_j = 1/sqrt(W)` at init, so the
*effective* first-layer gradient is shrunk by `1/sqrt(W)`; to keep the theory's effective step size
(which uses `a_j ~ +-1`) I multiply by `sqrt(W)`. So the full giant rate is `eta_1 = sqrt(W) *
d^{(k-1)/2}` — `sqrt(W)` undoing the readout normalisation, `d^{(k-1)/2}` the dimension scaling the
information exponent demands. This is exactly why `init_two_layer` must set `fc2.weight` to the constant
`1/sqrt(W)` and zero the biases: the giant rate is calibrated to precisely that readout scale, and if I
initialised the head any other way the calibration would be off.

I write the overwrite as a convex mix with weight `mix = eta_1 / (eta_1 + 1)` — small `eta_1` leaves the
row mostly itself, large `eta_1` pushes it almost entirely onto the direction — and I should compute the
actual numbers to be sure the giant regime does what the words claim. With `W = 256`, `d = 100`:
hermite's rate is `eta_1 = sqrt(256) * 100 = 16 * 100 = 1600`, so `mix = 1600/1601 = 0.99938` — the rows
are set to essentially the estimated direction, the random probe fraction is `0.06%`. The `k=1` links
have `eta_1 = sqrt(256) * 1 = 16`, so `mix = 16/17 = 0.941` — still dominated by the signal but with a
`6%` residual of the original probe. Let me carry the `k=1` case one step further and check what
per-row overlap with `theta*` that actually produces, because it feeds the width amplifier. A row is
`mix * dir + (1 - mix) * probe + 0.05 * jitter`, with `probe` overlapping `dir` by `~ 0.1` and `jitter`
purely orthogonal. The along-`dir` part is `0.941 + 0.059 * 0.1 = 0.947`; the perpendicular magnitude is
`sqrt(0.059^2 * 0.99 + 0.05^2) = 0.077`; so the per-row overlap is `0.947 / sqrt(0.947^2 + 0.077^2) =
0.997`. That is already at the relu ceiling on a *single* row — and the harness estimator `normalize(
sum_j |a_j| w_j)` then aggregates `256` such rows, sharpening it further via the width amplifier
`m / sqrt(m^2 + (1 - m^2)/256)`, which sends even a modest per-row `m = 0.5` to `0.994`. So the giant
mix is self-consistent: `k=1` lands near `0.998`, and hermite's near-unit mix sets the rows to the
estimated direction almost exactly, so the only thing that can hold hermite back is the *estimator*
quality, which the power iteration is built to guarantee.

Here I should be honest about how this grounds in the harness versus the cleanest analysis. The analysis
pairs the giant step with a matching weight decay `lambda_1 = eta_1^{-1}` so the `-W^{(0)}` term
*exactly* cancels the random init and the first layer becomes *purely* gradient features. The harness
instead realises the same "overwrite the random probe with the signal" idea through the convex mix,
which for the giant `eta_1` is numerically almost identical — with `mix = 0.99938` on hermite the
residual init is `0.06%`, indistinguishable from the exact `-W^{(0)}` cancellation, just expressed
without the weight-decay bookkeeping. It is the same mechanism; I am matching the task's edit surface,
which has no separate `lambda_1` knob, by folding the erase into the mix.

Two refinements the construction needs, and both are in the harness. First, if I set *every* row to
exactly the same direction, the features degenerate — all neurons compute `ReLU(<theta*,x> + b_j)`,
which is a good basis only if the thresholds `b_j` differ and the rows are not literally identical. So I
keep a small *orthogonal* jitter on each row — a random component projected off the direction and
renormalised — so the rows are a tight spread around `theta*` rather than identical, keeping the feature
matrix well-conditioned without re-adding off-subspace noise (the `0.05` jitter I already used in the
overlap arithmetic, small enough that the per-row overlap stays `0.997`). Second, I re-initialise the
biases to a spread of thresholds. With the rows now aligned to `theta*`, the functions `x ->
ReLU(<theta*,x> + b_j)` for a spread of `b_j` form a one-dimensional random-feature basis that can
approximate any smooth univariate link; the original zero biases from the init would give a degenerate
basis where every neuron switches at the same threshold. A deterministic spread `b_j = linspace(-2.5,
2.5, W)` cleanly covers the range of `u = <theta*,x> ~ N(0,1)` — `[-2.5, 2.5]` holds about `98.8%` of
the standard-normal mass, so the `256` knots blanket essentially the whole support of `u` with spacing
`~ 5/256 ~ 0.02 sigma`, far finer than any smooth or step link needs. Then renormalise each row to the
sphere — ReLU is positively homogeneous, so only directions and biases matter, and unit rows keep the
readout problem well-scaled.

Last, the readout. After the giant step the first layer is frozen and the only thing left is to fit the
head on the fixed features `phi(x) = ReLU(W^{(1)} x + b)`. That is a convex linear least-squares
problem, and the theory fits it with norm-constrained / weight-decayed regression — i.e. ridge, which
has a closed form, so I do not run GD on the head at all. I solve `beta = (Phi^T Phi + lambda I)^{-1}
Phi^T y` where `Phi` is `[n, W+1]` (the `256` ReLU features plus a constant column for the head bias),
so the Gram matrix is `[257, 257]` and the solve is a single small linear system, and `lambda =
max(weight_decay, 1e-4)`. The config hands `weight_decay = 0`, so `lambda = 1e-4` — just large enough to
keep the Gram invertible if two aligned features become nearly collinear (which the bias spread is
designed to prevent, but the floor costs nothing), and small enough not to bias the fit. Then set
`fc2.weight = beta[:-1]`, `fc2.bias = beta[-1]`. This is exactly the harness's `finalize` hook earning
its keep — the closed-form refit that frozen-bias deliberately left as a no-op. The whole method, then,
lives entirely in `init_two_layer` (random probes, constant readout) and `finalize` (estimate the
direction, giant-overwrite the rows, spread the biases, ridge-fit the head), with the mini-batch loop
neutralised.

Now the falsifiable expectations against the frozen-bias numbers, which is the whole reason to run this.

On `relu-d100` (`k=1`) I expect to *match* the `0.998` ceiling, not beat it — the easy link is saturated
for every method, and the first-moment estimator is just the clean way to get there; the per-row overlap
arithmetic above already put a single row at `0.997`, so the aggregate should sit right at `0.998`. If
giant-step came out below `0.998` on relu, something in the `k=1` first-moment path or the ridge fit
would be wrong.

On `sign-d100` (`k=1`, non-smooth) I expect the *largest reliability gain* over frozen-bias's erratic
`0.467` with its `0.735`-wide spread. Sign is `k=1`, so the full-batch first moment `normalize(mean(y
x))` over all `32768` samples gives a high-overlap direction directly, and the construction never
differentiates `g`, so the non-smoothness is a non-issue. I expect recovery near `0.99` on every seed
and, critically, the seed-to-seed spread to *collapse* — no more {`0.439`, `0.113`, `0.848`} lottery,
because the full-batch first moment is a deterministic, low-variance estimate of `theta*` rather than
`256` rows each chasing a noisy overlap. That tightening is the cleanest test that the sign failure was
signal aggregation plus wide-net dilution, not something the landscape freeze alone could reach.

On `hermite-d100` (`k=3`) this is the make-or-break. Frozen-bias was flat at `0.614` in a tight band
because the third-order signal was forty times below the mini-batch noise on every seed alike. The giant
step aggregates that signal over `n_train = 32768 > d^2` samples — clearing the detection threshold at
`1.8x` — and refines it with two tensor power-iteration passes, so the third-Hermite contraction should
surface a high-overlap direction where SGD could not. I expect hermite recovery to jump from `0.614` to
near `0.98`+ — the single largest improvement on the ladder — with `test_mse` falling well below
frozen-bias's `0.28`. The power iteration is the part I would watch: the random probes' initial overlap
is `~ 0.1`, and squared through one contraction and averaged over `256` probes it starts weak, but two
refinement passes should still pull it in; this is where an unlucky seed could underperform if the first
contraction fell below the aggregation floor. If hermite lands near 1 across all three seeds, the
diagnosis is confirmed end to end: the easy link was always solved, the sign link needed only clean
full-batch signal, and the hard link needed exactly the full-batch giant step plus power iteration that
the mini-batch crawl could never provide. Since direction recovery near 1 across all three links is the
ceiling of this benchmark, I do not expect a further published method to clearly beat a giant step that
already saturates every link — this rung is the endpoint, and the bar it must clear is precisely
frozen-bias's per-link numbers: match `0.998` on relu, sharply beat and tighten `0.467` on sign, and
convert the `0.614` hermite stall into near-perfect recovery.
