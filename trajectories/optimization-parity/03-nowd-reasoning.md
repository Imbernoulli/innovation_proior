The default run did what I bet it would, and the numbers say exactly where the remaining slack is. At
`N = 32` the online single-pass default jumped to `0.747`, `0.768`, `0.799` for a mean of `0.771` —
a decisive improvement over multi-epoch's `0.510`, and the `test_accuracy_std` of `~0.21` confirms the
mixture picture I predicted: within the `30_000`-step budget some secrets and orderings cross the
amplification threshold and shoot to near-`1.0`, while others stall at `0.5`, and the mean is the
average of the two populations. The `mean_steps` came in around `29_000`, essentially the full budget,
which is the signature I wanted: online runs grind the budget because training accuracy tracks population
accuracy, so the early-stop only fires when a run genuinely solves the rule (the slightly-below-30k mean
is exactly the few seeds that solved early). So removing the memorizer worked, and the population
amplification ran. At `N = 50` and `N = 64` the default sat at chance — `0.4996` and `0.5000` — with the
full `30_000` steps every time, precisely the `N^{K-1}`-budget wall I expected: at those `N` the gap
`gamma ~ N^{-(K-1)/2}` is too faint and the steps-to-separation too large for the budget. The lever for
this rung therefore has to live where there is still slack, which is `N = 32`: the means there are high
but not saturated, and the large variance says many seeds are crossing the threshold *just barely*, or
not quite, within budget. If I can make the amplification a little faster, those marginal seeds tip over
and the mean climbs.

Where is the amplification being slowed? Re-read the drift picture against the one hyperparameter I have
been holding at its default the whole time: weight decay. AdamW applies a decoupled multiplicative shrink
`theta <- (1 - lr * wd) theta` every step, outside the adaptive normalization, with `wd = 1e-2`. On the
multi-epoch rung I *wanted* that shrink, because it eroded the dense memorizer and tilted the competition
toward the low-norm sparse rule. But online there is no memorizer to erode — the gradient stream carries
no idiosyncratic table, so the only thing weight decay can act on is the very feature signal I am trying
to grow. And the feature signal starts out tiny: the relevant first-layer weights begin near their
Xavier-scale initialization and have to climb, under a drift of order `gamma ~ N^{-(K-1)/2}` per step,
until they overtake the irrelevant coordinates. During that long climb the weights are small, so the
*relative* effect of a multiplicative shrink is exactly the same fraction `lr * wd` per step on the
relevant and irrelevant coordinates alike — it does not know which is which. The drift is adding
`~gamma` to the relevant weights each step while the decay is subtracting `~ lr * wd * w_i` from them.
While `w_i` is small the decay is small, but it is a *constant fraction* working against a drift that is
already polynomially faint, and it acts every single step over tens of thousands of steps. So weight
decay is a steady headwind on the amplification: it does not stop the relevant weights from growing, but
it slows the rate at which they pull away from the irrelevant ones, pushing the threshold-crossing later
in the run — sometimes past the `30_000`-step budget, which is exactly how a marginal seed ends up at
`0.5` instead of `1.0`.

This is the danger sentence read in the direction that now matters. On a finite dataset, weight decay was
load-bearing because the competition was sparse-rule versus dense-memorizer and decay broke the tie in
the right direction. Online, the competition is gone — there is no memorizer — so the only role left for
decay is to fight the feature growth. The regularization that helped generalization in the finite-sample
regime becomes a pure optimization headwind in the online regime. So the rung is the cleanest possible
ablation: keep the maximal one-pass dataset and the standard initialization exactly as in the default,
and set `wd = 0`. Everything else — `lr = 1e-3`, `(beta1, beta2) = (0.9, 0.999)` — stays put, so any
change is attributable to the single decay knob and nothing else.

Let me make sure I am not removing something I will miss. The argument for keeping decay online would be
that the irrelevant weights also diffuse under minibatch noise (`sigma * sqrt(t)`), and decay caps that
diffusion, keeping the irrelevant coordinates from wandering large enough to drown the relevant signal.
That is a real effect, but the sizes do not favor it: the irrelevant coordinates have *zero* population
drift (their gradient reads the smaller `xi_{K+1}` coefficient, and over the family it averages toward
nothing), so their growth is pure `sqrt(t)` diffusion, which is *sub-linear*, while the relevant
coordinates have a linear-in-`t` drift. Linear beats `sqrt(t)` eventually regardless of decay; the
question is only *when*, and decay pushes the crossover later by shrinking the relevant drift's
accumulated lead. The diffusion-capping benefit would matter if the run were long enough for the
irrelevant walks to blow up, but at `30_000` steps the relevant drift is the binding constraint, not the
irrelevant diffusion. AdamW's per-coordinate normalization already rescales each coordinate's effective
step by its own gradient history, which damps the raw diffusion of the dead coordinates without any help
from decay. So I expect dropping decay to be net positive: I lose a diffusion cap I do not need at this
horizon and I remove a headwind on the signal I do need.

There is a subtlety in how AdamW's decay interacts with the adaptive step that makes me even more
confident the headwind is real. The decoupled decay shrinks *every* weight by the same multiplicative
factor regardless of its gradient magnitude — that is the whole point of decoupling, and it is correct
behavior for ordinary regularization. But for parity the relevant first-layer weights are precisely the
ones receiving the largest, most coherent gradients (the `xi_{K-1}` signal), so they are the weights I
most want to *grow*, and decay shrinks them at the same rate as the dead coordinates that are receiving
near-zero coherent gradient. Decay treats the signal-carrying weights and the noise-carrying weights
identically, when the entire mechanism of solving parity is to make the former pull away from the latter.
Turning decay off lets that separation happen at its natural drift rate, unimpeded.

It is worth being precise about how the *adaptive* part of AdamW interacts with all this, because it is
the reason I can drop decay without the dead coordinates exploding. AdamW divides each coordinate's step
by `sqrt(v_i) + eps`, where `v_i` is the running second moment of that coordinate's gradient. The relevant
coordinates carry a coherent gradient of magnitude `~ |xi_{K-1}|`, so their `v_i` is dominated by that
signal and their effective step is the signal divided by roughly the signal's own scale — a normalized,
order-one push in the right direction every step. The irrelevant coordinates carry near-zero coherent
gradient and only minibatch noise, so their `v_i` is set by the noise variance and their effective step
is noise divided by the noise scale — an order-one *random* push with zero mean. So the per-coordinate
normalization already equalizes the raw magnitudes, and what distinguishes relevant from irrelevant after
normalization is purely that the relevant push is *biased* (it always points the same way) while the
irrelevant push is unbiased (it cancels in expectation). The relevant coordinate accumulates a coherent
drift `~ t`; the irrelevant one accumulates a random walk `~ sqrt(t)`. This is the same drift-versus-noise
crossover as before, but now I can see that decay is the *only* term that touches the relevant drift's
lead symmetrically with the irrelevant noise — and since the normalization has already handled the
magnitude equalization, decay's diffusion-capping contribution is redundant while its drag on the
coherent drift is not. Removing it leaves the relevant coordinates with a clean normalized drift and the
irrelevant ones with a clean normalized random walk, which is the most favorable possible setting for the
amplification to win within the budget.

I also want to keep honest about what this rung does *not* fix, because it sets the ceiling. The SQ floor
says resolving the gap needs `~ N^{K-1}` steps, and weight decay was never the thing standing between me
and that floor — it only shifted the constant in front of it. Turning decay off lowers that constant at
`N = 32`, where the budget is already in the right ballpark, so a few more seeds tip over. But at `N = 50`
and `N = 64` the floor itself is above the `30_000`-step budget by orders of magnitude, and no setting of
a knob that merely changes the constant — not the init scale, not the dataset size, not the decay — can
buy the missing factor of `N`. The only knob in this edit surface that changes the *per-step progress* by
more than a constant is the learning rate, and even that runs into the noise: a larger `lr` makes each
amplification step bigger but also amplifies the minibatch noise the normalization is fighting, so it
trades crossover-time for stability rather than beating the exponent. So I expect this rung to be the top
of what initialization, dataset construction, and weight decay alone can reach on this task.

So the falsifiable expectations against the default numbers are specific. At `N = 32`, `wd = 0` should
beat the default's `0.771`: removing the headwind should let more of the marginal seeds cross the
threshold within budget, so I expect the mean to climb into the high-`0.7`s or beyond, with at least one
seed pushing into the high-`0.8`s where the default topped out near `0.80`. The variance should stay
large — this is still a mixture of solved and unsolved runs, and removing decay shifts a few from the
unsolved pile to the solved pile rather than making every run succeed, so I do not expect a tight
high-accuracy cluster. The `mean_steps` at `N = 32` should stay near the full budget, because the runs
that still do not solve grind all `30_000` steps and the ones that do solve early were already solving
under the default. At `N = 50` and `N = 64`, I expect essentially no change — both should remain at chance
(`~0.50`) with full-budget `mean_steps`, because the binding constraint there is the `N^{K-1}` step count,
which weight decay never touched; removing a headwind on a signal that cannot cross the threshold within
budget anyway changes nothing. If instead `wd = 0` *hurt* `N = 32` — dropped it back toward chance — that
would falsify the "decay is a pure online headwind" reading and tell me the diffusion-capping role was
actually load-bearing at this horizon, sending me to tune `wd` to a small positive value rather than zero.
And if `wd = 0` helps `N = 32` but the means are still a mixture far from `1.0`, the remaining gap is not
in any of these three knobs at their current settings — it is the budget itself meeting the `N^{K-1}`
floor, and the only further lever inside this edit surface would be the optimizer's learning rate, to
make each amplification step larger and reach the threshold in fewer steps. That is the boundary of what
initialization, dataset, and weight decay alone can do here. The full scaffold fill for this rung is in
the answer.
