I have to start somewhere on the ladder, and the cleanest place to start is the resource accounting,
because that is the thing the three editable knobs actually move. The target is a hidden `K`-sparse
parity, `y = chi_S(x)` with `|S| = 8`, and statistically the problem is nothing: telling one subset
apart from the `C(N, 8)` candidates needs only `log C(N, 8) = Theta(K log N)` labels. For `N = 32`
that is a few hundred examples in principle; for `N = 64` a few thousand. So the information to specify
`S` fits in a tiny dataset. What is expensive is not the labels but the *correlation*. Parity gives no
partial credit: by orthonormality of the parities under the correlation inner product, a subset that
overlaps seven of the eight right indices has exactly the same (zero) correlation with `y` as one that
overlaps none. There is no smooth signal to follow toward `S`; a correlation-only learner is forced to
check subsets essentially one at a time, and the statistical-query lower bound turns this into a hard
floor: any noise-tolerant correlational learner answering `T` queries to tolerance `tau` must satisfy
`T / tau^2 >= Omega(N^K)`. Gradient descent is such a learner — a stochastic gradient is an expectation
estimated to a precision set by its minibatch noise — so this `N^{Omega(K)}` computational price is not
something I can engineer away with initialization or hyperparameters. It is the floor.

That floor is the lens through which I have to read the harness, because the harness couples two
resources that the floor keeps separate. The loop is `while steps < max_steps`, and inside each epoch
it draws a fresh permutation of whatever training set `make_dataset` returned and steps through it in
minibatches of 128. So the number of gradient steps is fixed by the budget (`max_steps = 30_000`), but
the number of *independent* training examples I get to spend is whatever `make_dataset` puts on the
table. If I return a fresh, maximal random dataset, every step consumes new data and the run is
effectively single-pass online SGD: each step is an unbiased sample of the same population gradient,
which is the cleanest possible setting, but it pays the `N^{Omega(K)}` cost in *fresh samples*. In the
window `K log N << m << N^K`, that coupling is exactly the blocker. The dataset can hold enough
information about `S` to specify it many times over, yet the online recipe runs out of fresh samples —
runs out of *steps*, since steps and samples are the same thing online — long before it has spent enough
compute to extract `S` from that information. The budget is `30_000` steps times `128` = under four
million example-visits; the online cost at the SQ floor is `N^{Omega(8)}`, which for `N = 32` is already
astronomically larger. So pure online training under this budget should land at chance on all but the
smallest `N`. The first thing I want to try is the move that breaks the coupling.

The move is to separate compute from evidence: draw a finite dataset *once* and keep optimizing on it.
Then the number of gradient steps is no longer the number of fresh examples — I can spend compute by
revisiting the same `m` labels across many reshuffled epochs. Concretely, because the loop reshuffles
and re-passes whenever the dataset is exhausted, returning a small `m` forces `epochs = max_steps *
batch_size / m` passes over those examples. With `m = 10_000` and the `30_000`-step, `128`-batch budget,
that is `30_000 * 128 / 10_000 ~ 384` epochs at `N = 32` — hundreds of passes over a fixed table
instead of one pass over a fresh stream. I am not creating new evidence; I am converting the fixed step
budget into repeated optimization on one informative empirical distribution. That is the entire content
of this first rung: keep the standard initialization and the default AdamW settings, and change only
the dataset size from "maximal" to a small fixed `m`.

Before I commit, I have to be honest about exactly what reuse can and cannot buy, because it is easy to
fool myself here. An online minibatch gradient is an unbiased estimate of the *population* gradient. A
minibatch drawn from a fixed dataset is only an estimate of the *empirical* gradient of that dataset.
Reuse does not manufacture new independent samples, and it cannot make the empirical distribution closer
to the population than the original draw of `m` examples already made it. So the only way many passes can
help is if the fixed sample already preserves enough trace of the hidden parity, and the repeated steps
give the optimizer time to integrate that trace. What is the trace I am hoping survives? It lives in the
population gradient, which is where the sparse feature first becomes visible. Take a single ReLU neuron
and the correlation view of the loss: at initialization, the coordinate-`i` gradient has the form
`E[-y x_i 1[w.x + b > 0]]`. The indicator, after flipping coordinates by the signs of `w`, is
`1/2 + 1/2 Maj(x)`, so multiplying by `y x_i` reads off a Fourier coefficient of majority — an
order-`(K-1)` coefficient when `i in S` (the `x_i` cancels one parity bit) and an order-`(K+1)`
coefficient when `i not in S` (it adds one bit). The relevant indices are separated from the irrelevant
ones precisely because majority's degree-`(K-1)` mass dominates its degree-`(K+1)` mass. The exact
majority formula gives `|xi_{K-1}| = ((N-K)/(K-1)) |xi_{K+1}|`, so the gap is
`gamma_Maj >= 0.03 (N-1)^{-(K-1)/2}` for `N >= 4K`. The signal is real, but its scale is
`Theta(N^{-(K-1)/2})` — faint, and shrinking polynomially with `N`. That single exponent is why I
expect this rung to behave very differently across the three configurations: the same `m` and the same
budget meet a signal that is far weaker at `N = 50, 64` than at `N = 32`.

Now I can see the wall that reuse creates on the other side, and it is the wall I expect this rung to
hit. A finite dataset is not only reusable; it is also *memorizable*. The MLP can fit a sample-specific
labeling of `m` points with a dense interpolating function, and that route can be far faster than
resolving the tiny parity gap, because fitting `10_000` arbitrary labels with a width-`512` network is
easy. If I optimize the empirical loss and the loop early-stops the moment windowed train accuracy hits
`0.999`, I may have learned only the sample. The early-stop criterion in the fixed loop is on *training*
accuracy, so a memorizer trips it: once the network interpolates the table, training accuracy is one,
the stable-window counter fills, and the run stops — having generalized to nothing. So the trajectory I
should expect in the informative-but-small window is not smooth population learning. It is fit-the-sample
first, and only *then*, if the sparse trace keeps accumulating across passes, recover the rule. That
delayed-generalization shape is the whole reason this is interesting and also the whole reason it is
fragile: whether the sparse circuit ever overtakes the memorizer depends on a competition I am not
directly controlling, and the early-stop can cut the run off in the memorization phase before the
generalization phase ever arrives.

What would tilt that competition toward the right competitor? A dense memorizer spreads dependence
across many coordinates and many degrees of freedom; the true parity rule concentrates dependence on
eight coordinates. A norm penalty is the natural bias toward the lower-complexity explanation, and the
harness already gives me one for free: AdamW's decoupled weight decay, default `wd = 1e-2`, applies a
multiplicative shrink `theta <- (1 - lr * wd) theta` outside the adaptive normalization. The decay does
not know which weights are relevant, so it is a blunt instrument, but it changes the competition in the
direction I want: weights that are only useful for a brittle sample fit are continually eroded, while
weights that keep receiving coherent reinforcement from the parity trace across passes can survive and
grow. So for this rung I keep the default `wd = 1e-2` on, betting that it makes "fit the sample and stay
there" less stable and gives the sparse solution room to win. I am keeping every other hyperparameter at
its default too — `lr = 1e-3`, `(beta1, beta2) = (0.9, 0.999)` — because the only variable I want to
isolate on the weakest rung is the dataset-size change; the optimizer settings are the control.

I also have to make the repeated passes behave like honest stochastic optimization on the empirical
distribution rather than like a deterministic artifact of one fixed batch order. The fixed loop already
handles this: it reshuffles via `torch.randperm` at every epoch boundary, so each step is closer to a
fresh minibatch from the empirical distribution and the batch order does not itself become an object to
memorize. That costs me nothing — it is in the substrate — but it matters, because without it the
optimizer would see the identical temporal pattern at every pass and the reuse would degrade into
overfitting a single trajectory. With reshuffling, the reused-gradient stream is well mixed, and the
`epochs = max_steps * B / m` passes act like `30_000` mildly-correlated steps on the empirical objective
rather than `384` repetitions of the same `78`-step cycle.

There is a tension in the choice of `m` itself, and I want to be explicit that I am picking a single
point in a window rather than an optimum. If `m` is extremely large (the maximal allowed dataset), each
example is seen about once and I am back to single-pass online training — no reuse, no leverage from the
budget. If `m` is near or below the statistical floor `Theta(K log N)`, the sample may not even identify
`S`, and no amount of repeated optimization can reconstruct information that is not in the table. The
useful regime is in between: enough examples that the fixed sample carries the sparse trace, far fewer
than the online fresh-sample cost, and enough passes that compute can extract what is already present.
`m = 10_000` is my first guess at a point in that window — comfortably above the floor at all three `N`,
small enough to force hundreds of passes. I am not claiming it is the best `m`; I am claiming it is the
simplest concrete instantiation of "convert the budget into reuse," and it is the cleanest probe of
whether reuse alone, with everything else at default, moves the needle.

Reading the construction forward, here is what I expect and where I am uneasy. At `N = 32` the Fourier
gap is largest, so this is where reuse has the best chance: if the sparse circuit can win the competition
against the memorizer within `~384` passes, `N = 32` is where I will see it first. But I am genuinely
worried that the early-stop will fire on memorization before generalization arrives — a width-`512`
network can interpolate `10_000` points well inside the `30_000`-step budget, and once training accuracy
hits `0.999` the run halts. If that happens, the mean test accuracy at `N = 32` will sit just above
chance, dragged up only by the handful of runs where the sparse solution happened to emerge before the
stop. At `N = 50` and `N = 64` the gap is weaker by the `N^{-(K-1)/2}` factor and `10_000` examples is a
smaller fraction of `C(N, 8)`, so I expect the memorizer to win essentially every time and the means to
sit at chance. The falsifiable expectation, then: this rung should land near `0.50` at `N = 50, 64`, and
only marginally above `0.50` at `N = 32` — and crucially, the `mean_steps` should come in *well below*
the `30_000` budget at every `N`, because the early-stop on training accuracy is exactly what a
memorizer trips. If the steps are short and the test accuracy is at chance, that is the signature that
the small dataset taught the network to memorize and stop, not to generalize — and it would tell me the
next rung must remove the thing that lets memorization win the race: the finite dataset itself. The full
scaffold fill for this rung is in the answer.
