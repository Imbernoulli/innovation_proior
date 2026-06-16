The multi-epoch run told me, in numbers, exactly which side of the competition won. At `N = 32` the
three seeds landed at `0.506`, `0.508`, `0.516` for a mean of `0.510` — a hair above chance, and the
`mean_steps` came in around `18_600`, well short of the `30_000` budget. That short-step signature is
the tell I was dreading: the loop's early-stop fires on *training* accuracy, so a run that halts early
has hit `0.999` train accuracy on the `10_000`-example table — it memorized the sample and stopped. The
held-out accuracy near `0.51` confirms it generalized to essentially nothing; the tiny excess over
`0.50` is just the handful of seeds where the sparse circuit started to emerge before the stop cut it
off. At `N = 50` and `N = 64` it is worse and cleaner: means of `0.4996` and `0.5017`, dead chance, with
`mean_steps` of only `~8_900` and `~6_700`. The larger `N`, the faster the memorizer interpolates and
the earlier the run dies — the steps *shrink* with `N`, which is the opposite of what learning the rule
would look like. So the diagnosis is unambiguous: the finite dataset created a memorization competitor,
the competitor won every time at large `N` and nearly every time at small `N`, and the early-stop on
training accuracy guaranteed the run never lived long enough for the sparse trace to take over. Reuse
did not buy me generalization; it bought me fast overfitting.

That diagnosis points directly at the variable to remove. The memorizer wins because there is a finite
table to memorize. If I take the table away — make the dataset so large that within the budget the loop
never re-passes — then there is nothing to interpolate, every step sees data it has not seen before, and
the only way to drive the training loss down is to learn something that holds on the population. So the
next rung is the opposite extreme of the dataset knob: return a *maximal* random dataset (up to the
`max_train_examples = 12_800_000` cap) so that `epochs = max_steps * batch_size / m` is below one and the
run is single-pass online SGD. Keep the initialization and the AdamW settings exactly where they were —
this is the same controlled comparison, flipping only the dataset size from small to maximal, so any
change in the result is attributable to the online-versus-reuse axis and nothing else.

Let me re-derive why online single-pass is the *clean* regime, because the multi-epoch failure was
fundamentally about the gradient being an estimate of the wrong thing. When I draw a fresh i.i.d. batch
every step, the minibatch gradient is an unbiased estimate of the *population* gradient — the gradient of
the true risk over the uniform distribution on `{0,1}^N`. There is no empirical-vs-population gap to
exploit and no fixed table to memorize, so the trajectory is governed entirely by the population
gradient's structure. And that structure is exactly the Fourier story. For a single ReLU neuron at
standard initialization, coordinate `i` of the population gradient under the correlation view is
`E[-y x_i 1[w.x + b > 0]]`, which reads a Fourier coefficient of majority: order `K-1` when `i in S`
(the `x_i` cancels one parity bit) and order `K+1` when `i not in S`. Majority's spectrum dominates the
lower degree, `|xi_{K-1}| = ((N-K)/(K-1)) |xi_{K+1}|`, giving a gap
`gamma >= 0.03 (N-1)^{-(K-1)/2}` for `N >= 4K`. So the relevant coordinates of the population gradient
are systematically larger than the irrelevant ones, by an amount that is strictly positive but
polynomially faint in `N`. Online SGD's whole hope rests on this gap: each step adds a noisy push whose
expectation is this population gradient, so the relevant weights experience a small *drift* on top of
the minibatch noise, and over many steps the relevant features amplify.

Concretely, weight `i` does a biased random walk: drift `beta_i * t` plus diffusion `sigma * sqrt(t)`,
with `beta_i` of order `|xi_{K-1}|` (large) on the relevant coordinates and `|xi_{K+1}|` (smaller)
elsewhere. The relevant coordinates pull monotonically away from the rest once `t` is large enough that
the linear drift beats the `sqrt(t)` noise — past about `t ~ (sigma / max beta_i)^2`. That is the
plateau-then-jump: the loss stays flat at chance while the weights climb, because the Sigmoid classifier
does not change its labels until the relevant weights actually *overtake* the irrelevant ones, and only
then does accuracy snap upward. Nothing is searching; a fixed signal is being amplified. The multi-epoch
run never got to run this amplification cleanly, because its gradient was the empirical gradient of a
small table — which the memorizer drove to zero by fitting idiosyncratic labels long before the
population drift could separate the eight relevant coordinates. Online training removes that shortcut:
there is no idiosyncratic table to fit, so the only persistent signal in the gradient stream *is* the
Fourier gap.

I should also pin down *why* this amplification is genuine feature learning and not some trivial readout
fit, because if it were the latter the dataset knob would be irrelevant and I would be wasting the rung.
The lazy/kernel worry is that the weights barely move, the network is effectively a linear model over its
initial random-feature map, and all the "learning" is in the second layer. But that regime cannot even
*represent* parity at width `512`. The parities are orthonormal, so any fixed `D`-dimensional feature map
with bounded norm and a readout of norm `R` has only `D R^2` worth of total squared correlation to spread
across the `C(N, K)` size-`K` parities; to fit all of them with margin you need `D = Omega(N^K)` features.
Width `512` is nowhere near `N^8`, so the network *must* leave its initialization — the relevant
first-layer weights have to physically grow — and the success, when it comes, is the relevant features
emerging, exactly the drift mechanism above. This matters for the dataset choice: feature learning is a
property of the population gradient, so it only runs cleanly when each step samples that population
gradient afresh, which is precisely what the maximal one-pass dataset delivers and what the small reused
table destroyed by substituting an empirical gradient the memorizer could game.

This also re-frames the early-stop, which sabotaged the previous rung, into a non-issue here. Online,
training accuracy on a fresh batch is essentially the population accuracy, so it cannot read `0.999`
until the network actually generalizes. The early-stop will not fire on memorization because there is
nothing memorized; if it fires at all it fires because the run genuinely solved the parity. So I expect
the online runs to use the *full* `30_000`-step budget unless they truly crack the rule — which, read
against the multi-epoch `mean_steps`, is itself a diagnostic: if the default sits at `30_000` steps at a
given `N` while multi-epoch died early there, that is direct evidence the online run is grinding on the
population gradient rather than collapsing onto a table.

Now I have to be honest about the cost I am paying to get this clean regime, because it is the same SQ
floor I started from. Online single-pass means steps and fresh samples are the same resource, so the
budget caps me at `30_000 * 128 ~ 3.8` million example-visits. The amplification needs the drift to
outrun the noise, and the drift scale is the gap `gamma ~ N^{-(K-1)/2}`. The number of steps to resolve
that gap scales like `gamma^{-2} ~ N^{K-1}`. For `K = 8`: at `N = 32` that is `32^7 ~ 3.4e10` in the
crude reading, but the constant and the width help — the width-`512` network runs `512` neurons climbing
the same population gradient in parallel, and with a lucky fraction of them landing useful sign patterns
the effective number of steps to first separation is far smaller than the worst-case bound. Empirically
this is precisely the regime where wide MLPs *do* crack `N = 32` parity within tens of thousands of
steps. But the same exponent says `N = 50` and `N = 64` are out of reach: `50^7` and `64^7` blow past the
budget by orders of magnitude, and the gap `gamma` is smaller there too, so the drift-beats-noise
crossover happens later than `30_000` steps allow. The `N^{-(K-1)/2}` exponent that made multi-epoch fail
worse at large `N` is the *same* exponent that will cap the online run at small `N`.

So the falsifiable expectations against the multi-epoch numbers are sharp. At `N = 32`, the default
should *clearly beat* multi-epoch's `0.510` — removing the memorizer should let the population
amplification run, so I expect the mean to climb into the `0.7`-plus range, with high per-seed variance
(some secrets and orderings cross the threshold within budget and hit near-`1.0`, others do not and stay
at `0.5`, so the mean is a mixture and the `test_accuracy_std` should be large, on the order of `0.2`).
The `mean_steps` at `N = 32` should rise toward the full `30_000`, the opposite of multi-epoch's
`~18_600`, because online runs grind the budget unless they solve it. At `N = 50` and `N = 64`, I expect
the default to stay at chance — around `0.50` — because the budget cannot cover `N^{K-1}` steps there, and
the `mean_steps` should sit at the full `30_000` (no early-stop, no solve). If instead the default also
collapsed to chance at `N = 32`, that would falsify the whole "memorizer was the problem" reading and
send me back to the optimizer knob rather than the dataset knob. And if the default beats multi-epoch at
`N = 32` but the high-variance mixture suggests the threshold is being crossed *just barely* within
budget — many seeds sitting at `0.5`, a few at near-`1.0` — then the next lever is obvious: the default
weight decay is shrinking the relevant weights while they are still tiny, slowing the very amplification
I need, and the rung after this should test turning it off. The full scaffold fill for this rung is in
the answer.
