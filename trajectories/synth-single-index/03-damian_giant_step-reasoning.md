Freezing the biases did what the asymmetric prediction said on average, but the spreads confirm the
diagnosis in a way I did not fully expect, and reading the three links in order tells me exactly what
step 3 has to be. On `relu-d100` recovery stayed pinned at `0.998` with `test_mse ~ 0.0016` — the easy
`k=1` regime is saturated and indifferent to the bias treatment, no signal to read from a link every
recipe already solves.

Take `sign-d100` next, because the frozen-bias move was supposed to pay off there. The mean rose from
vanilla's `0.210` to `0.467` — more than a doubling — and seed 456 jumped to `0.848` where the
collapsed landscape pulled that run cleanly toward a pole. So on average the landscape argument was
right: `k=1` sign only needed a benign landscape. But I predicted the *spread* would tighten, and it did
not. The per-seed values were {`0.439`, `0.113`, `0.848`}, a range of `0.735` — *wider* than vanilla's
`0.377`, with seed 123 at `0.113` actually below its vanilla value. That is the precise failure mode I
flagged as the risk of applying a tied-architecture analysis to a wide net: the benign scalar-flow
picture is realised only *loosely* when `256` independent rows each chase their own noisy `256`-sample
overlap, so the collapsed landscape lifts the mean but cannot force every run's rows onto the same pole.
The mechanism is real but the wide net dilutes it, and the residual randomness is still driven by the
`256`-sample gradient noise, not by the landscape shape.

Now `hermite-d100`, the decisive reading. The mean went from `0.656` to `0.614` — essentially flat, if
anything slightly down — and the per-seed band {`0.496`, `0.671`, `0.674`} is a tight cluster of
mediocrity, standard deviation `0.083`. On the hard link the freeze changed nothing about the outcome
*and* nothing about the spread: every run lands in the same low band. That consistency is the tell.
Sign was a lottery (wild spread, some runs win); hermite is a *uniform* wall (tight spread, every run
stuck at the same height). A lottery is a landscape symptom; a uniform wall is a signal-size symptom.
Freezing the biases addressed the landscape, so it moved the lottery link on average and left the wall
link exactly where it was. Together the sign gain and the hermite flatness say the bias freeze was a
landscape fix, and hermite's problem is not the landscape.

That problem is signal size, and the arithmetic is the one I have tracked since the control: on `k=3`
the per-mini-batch informative gradient along `theta*` scales as `m^2 ~ d^{-1} = 0.01` near the
equator, against empirical noise `~ sqrt(d/256) = 0.625`, an SNR `~ 0.016` — forty times below the
noise on every mini-batch step, which is exactly why hermite sits flat regardless of how clean the
landscape is. A cleaner landscape reshapes *where* the gradient points; it does nothing about *how
loud* it is. So the missing ingredient is signal aggregation, and that dictates step 3: stop crawling
with noisy small-batch gradients and extract the weak third-order signal by summing it over the entire
training set.

Before building it I should walk the design space, because there is more than one way to "get more
signal" and the tempting ones fail on arithmetic worth doing before a wasted run. Four moves are on the
table. First, keep the mini-batch loop but raise the batch size. The noise falls as `sqrt(d/n)`, so to
drag hermite's SNR from `0.016` to order `1` I need `sqrt(d/n) ~ 0.01`, i.e. `n ~ 10^6` — four thousand
times the current batch, and thirty times larger than the entire `n_train = 32768`. A bigger *mini*-
batch cannot reach it. Second, raise the learning rate: a larger `eta` multiplies signal and noise
together, the SNR is scale-invariant, so a big step on a sub-noise gradient just takes a big *random*
step — the erratic sign lottery I want to kill. Third, train longer: the budget is fixed at `8000`
steps, and even unlimited steps of a zero-mean-dominated update is a random walk whose drift `~ 0.016`
is swamped by diffusion `~ 0.625 sqrt(steps)`; averaging `8000` independent batches gives an SNR gain
of `sqrt(8000) ~ 89`, lifting `0.016` to `~1.4` — but *only* because the `8000` batches are independent
draws of the *same population gradient*. That observation is the key: the loop is already trying to
average, badly. So the fourth move is the honest one — compute the population gradient directly by
aggregating the whole training set into a single full-batch step.

Is `32768` enough? The third-order term surfaces above its own empirical noise when `n ~ d^2 = 10^4`.
Here `n_train = 32768 = 3.3 d^2`, so `sqrt(n_train / d^2) ~ 1.8`: the full-batch signal sits about `1.8x`
above its aggregation-noise floor — detectable, not marginal. The mini-batch, `sqrt(256/d^2) = 0.16`,
is sixfold *below* the floor; one full-batch pass is worth `sqrt(32768/256) = 11.3` mini-batches' worth
of SNR and is the only single-pass way to clear detection at all. So the recipe inverts the loop: the
`8000` mini-batch steps do *nothing*, and the real work happens once, full-batch, in `finalize`, where
the harness hands me the entire `x_train, y_train`. `make_optimizer` returns an SGD with `lr = 0.0` so
the loop changes nothing, `training_step` is a no-op that only logs the batch target energy, and
`finalize` carries the giant step plus the readout fit. I cannot delete the loop — the driver always
runs it — so neutralising it with `lr = 0.0` is the faithful way to say "spend no update budget on
noise".

Now the direction estimator per link, read off the first non-vanishing term of the first-layer gradient
series. At a symmetric/zero-output init the row gradient loses its self-interaction term and is the
clean correlation `grad_{w_j} L = -2 a_j E[ f*(x) x sigma'(<w_j,x>+b_j) ]`. Expanding `E[f*(x) x
sigma'(<w,x>)]` against the Gaussian gives an asymptotic series in `d^{-1/2}` whose leading informative
term points each row into `theta*`, of order the information exponent `k`. For `k=1` the first Hermite
moment `E[y x] = mu_1 theta*` is the signal, `O(1)` along `theta*` once aggregated. For `k=3` the first
two moments vanish and the signal first appears in the third-order term, of size `~ d^{-1}` after
contracting against a random probe of overlap `1/sqrt(d)`.

So for `k=1` (relu, sign) the estimator is `theta_hat = normalize( (1/n) sum_i y_i x_i )` — one
matrix-free average over the full batch. Sign is `k=1` too — `mu_1 = E[|u|] = sqrt(2/pi) ~ 0.8`,
nonzero and larger than relu's — and crucially the construction touches `g` only through its low-order
Hermite *moments*, never through derivatives. That is why it should rescue the sign link where
frozen-bias was erratic: the discontinuity is invisible to a moment estimator, so the non-smoothness
that roughened the SGD landscape is simply not in the picture. The sign lottery was never a signal
problem — `mu_1 ~ 0.8` is a loud first moment — it was aggregation and landscape, and a deterministic
full-batch first moment is a low-variance estimate that removes both the lottery and the wide-net
dilution I saw at `0.467`.

For `k=3` (hermite) the first moment estimates `0` and is useless, so I build the third-order term. The
multivariate third Hermite tensor contracted twice against a probe `v` gives `x <x,v>^2 - x ||v||^2 - 2
v <x,v>`, and its `y`-weighted expectation should be proportional to `mu_3 <theta*,v>^2 theta*`. Check
the aligned probe `v = theta*` in closed form: then `<x,v> = u`, `||v||^2 = 1`, and the contracted
vector is `x u^2 - x - 2 theta* u`; projecting on `theta*` gives coefficient `u^3 - u - 2u = u^3 - 3u =
He_3(u)`, so the along-`theta*` expectation is `E[g(u) He_3(u)] = mu_3` by orthogonality, and the
components orthogonal to `theta*` vanish by isotropy (odd in `x_perp`, independent of `u`). The
contraction returns a vector along `theta*` with strength `mu_3` — the estimator is right.

A *random* probe overlaps `theta*` by only `~ 0.1`, and the signal scales as `<theta*,v>^2 ~ 0.01`, so
a single fresh-probe contraction is right at the edge of the `1.8x` full-batch margin. I sharpen it with
tensor power iteration: once I have a rough direction, contract again *using that estimate as the
probe*. The map `v -> normalize(C_3(v,v))` has `theta*` as an attracting fixed point because the
returned vector's overlap with `theta*` is monotone increasing in the input overlap — feed in overlap
`m`, get out a vector dominated by the `m^2 theta*` term whose normalised overlap exceeds `m` once `m`
clears the noise floor. I average the first contraction over the `256` current random rows as probes to
beat down per-probe variance, get a coarse direction, then run two power-iteration passes on it. Not
more than two: each pass costs a full `[n, d]` contraction, and once the overlap nears `1` the map is
essentially the identity, so further passes buy only compute.

Now what to do to the network with that direction. The theory steps the first layer along the giant
gradient, after which each row equals the scaled gradient feature — a vector along `theta*`. Two facts
fix the step size. First, the rows are unit vectors and I want to rotate them an `O(1)` amount toward
`theta*`, but the gradient has norm only `~ d^{-(k-1)/2}` for the hard link; to get an `O(1)` rotation
the step must *grow with the dimension* — the "giant" step — with `eta_1 ~ d^{(k-1)/2}`. For `k=1` that
is `O(1)` in `d`; for `k=3` it grows like `d`. This is the information-exponent story from the
step-size side: the deeper the direction is hidden, the bigger the step to surface it. Second, the
first-layer gradient carries a factor `a_j` out front, and the scaffold's readout normalisation puts
`a_j = 1/sqrt(W)` at init, shrinking the effective gradient by `1/sqrt(W)`; to keep the theory's
effective step (which uses `a_j ~ +-1`) I multiply by `sqrt(W)`. So the full rate is `eta_1 = sqrt(W)
d^{(k-1)/2}` — `sqrt(W)` undoing the readout normalisation, `d^{(k-1)/2}` the dimension scaling. This
is why `init_two_layer` sets `fc2.weight` to the constant `1/sqrt(W)` and zeros the biases: the giant
rate is calibrated to precisely that readout scale.

I write the overwrite as a convex mix with weight `mix = eta_1 / (eta_1 + 1)` — small `eta_1` leaves
the row mostly itself, large `eta_1` pushes it almost entirely onto the direction. Compute the numbers.
With `W = 256`, `d = 100`: hermite's rate is `eta_1 = 16 * 100 = 1600`, so `mix = 1600/1601 = 0.99938`
— the rows are set to essentially the estimated direction, the random-probe fraction `0.06%`. The `k=1`
links have `eta_1 = 16`, so `mix = 16/17 = 0.941`, still dominated by the signal with a `6%` residual
probe. Carrying the `k=1` case one step further: a row is `mix * dir + (1 - mix) * probe + 0.05 *
jitter`, with `probe` overlapping `dir` by `~ 0.1` and `jitter` orthogonal, so the along-`dir` part is
`0.941 + 0.059 * 0.1 = 0.947`, the perpendicular magnitude `sqrt(0.059^2 * 0.99 + 0.05^2) = 0.077`, and
the per-row overlap `0.947 / sqrt(0.947^2 + 0.077^2) = 0.997` — at the relu ceiling on a single row.
The width amplifier `m/sqrt(m^2 + (1-m^2)/256)` from the previous rung then sharpens the `256`-row
aggregate further. So the mix is self-consistent: `k=1` lands near `0.998`, hermite's near-unit mix
sets the rows to the estimated direction almost exactly, and the only thing that can hold hermite back
is the *estimator* quality, which the power iteration is built to guarantee.

I should be honest about how this grounds in the harness versus the cleanest analysis. The analysis
pairs the giant step with a matching weight decay `lambda_1 = eta_1^{-1}` so the `-W^{(0)}` term
*exactly* cancels the random init and the first layer becomes pure gradient features. The harness has no
separate `lambda_1` knob, so I realise the same "overwrite the random probe with the signal" idea
through the convex mix, which for the giant `eta_1` is numerically almost identical — with `mix =
0.99938` on hermite the residual init is `0.06%`, indistinguishable from the exact cancellation, just
without the weight-decay bookkeeping.

Two refinements the construction needs, both in the harness. First, if I set every row to exactly the
same direction the features degenerate — all neurons compute `ReLU(<theta*,x> + b_j)`, a good basis only
if the thresholds differ and the rows are not literally identical. So I keep a small *orthogonal* jitter
on each row (a random component projected off the direction and renormalised), so the rows are a tight
spread around `theta*` rather than identical — the `0.05` jitter already in the overlap arithmetic,
small enough that the per-row overlap stays `0.997`. Second, I re-init the biases to a spread of
thresholds: with the rows aligned to `theta*`, `{ReLU(<theta*,x> + b_j)}` for spread `b_j` is a
one-dimensional basis that can approximate any smooth univariate link, whereas the original zero biases
would give a degenerate basis where every neuron switches at the same point. A deterministic `b_j =
linspace(-2.5, 2.5, W)` covers the range of `u ~ N(0,1)` — `[-2.5, 2.5]` holds `~98.8%` of the mass, so
the `256` knots blanket the support with spacing `~ 0.02 sigma`. Then renormalise each row to the
sphere (ReLU is positively homogeneous, so only directions and biases matter).

Last, the readout. After the giant step the first layer is fixed and only the head remains to fit on
`phi(x) = ReLU(W^{(1)} x + b)` — a convex least-squares problem, so I solve it in closed form rather
than run GD on the head: `beta = (Phi^T Phi + lambda I)^{-1} Phi^T y`, where `Phi` is `[n, W+1]` (the
`256` ReLU features plus a constant column for the head bias), the Gram matrix is `[257, 257]`, and
`lambda = max(weight_decay, 1e-4)`. The config hands `weight_decay = 0`, so `lambda = 1e-4` — just
enough to keep the Gram invertible if two aligned features become nearly collinear, small enough not to
bias the fit. Then `fc2.weight = beta[:-1]`, `fc2.bias = beta[-1]`. This is the `finalize` hook earning
its keep — the closed-form refit frozen-bias deliberately left as a no-op. The method lives entirely in
`init_two_layer` (random probes, constant readout) and `finalize` (estimate the direction, giant-
overwrite the rows, spread the biases, ridge-fit the head), with the mini-batch loop neutralised.

So the falsifiable expectations against the frozen-bias numbers. On `relu-d100` (`k=1`) I expect to
*match* the `0.998` ceiling, not beat it — the per-row overlap arithmetic already put a single row at
`0.997`, and the first-moment estimator is just the clean way there; a value below `0.998` would mean
something in the `k=1` path or the ridge fit is wrong. On `sign-d100` (`k=1`, non-smooth) I expect the
largest reliability gain over frozen-bias's erratic `0.467`: the full-batch first moment over all
`32768` samples gives a high-overlap direction directly and never differentiates `g`, so I expect
recovery near `0.99` on every seed and, critically, the seed spread to *collapse* — no more {`0.439`,
`0.113`, `0.848`} lottery, because the moment is a deterministic low-variance estimate rather than `256`
rows chasing a noisy overlap. That tightening is the cleanest test that the sign failure was
aggregation plus wide-net dilution. On `hermite-d100` (`k=3`) this is make-or-break: frozen-bias was
flat at `0.614` because the third-order signal was forty times below the mini-batch noise, and the giant
step aggregates it over `n_train > d^2` (clearing detection at `1.8x`) and refines with two power-
iteration passes, so I expect hermite to leap from `0.614` toward near-perfect recovery with `test_mse`
well below `0.28`. The power iteration is the part I would watch — the random probes' `~ 0.1` overlap,
squared through one contraction, starts weak, and an unlucky seed could underperform if the first
contraction fell below the aggregation floor. Since the score is bounded by `1` and I expect this
construction to saturate every link, there is little headroom left above it; the bar to clear is
frozen-bias's per-link numbers — match `0.998` on relu, sharply beat and tighten `0.467` on sign, and
convert the `0.614` hermite stall into near-perfect recovery.
