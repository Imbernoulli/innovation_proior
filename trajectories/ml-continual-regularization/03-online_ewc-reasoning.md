EWC's numbers confirmed the SI diagnosis and then quietly exposed its own ceiling. The collapse I traced
to SI's unbounded path integral is gone: on Permuted-MNIST EWC jumped from SI's 0.4474 to 0.8381 — a gain
of 0.3907, an 87% relative improvement, the biggest single recovery on the board — and the per-context
line {0.9325, 0.9258, 0.9166, 0.9009, 0.8737, 0.8501, 0.8091, 0.7463, 0.7066, 0.7199} is no longer a
monotone slide to chance. Its worst point, 0.7066, is five times the ten-way chance floor SI crashed
through; where SI lost a total of 0.6468 from first context to last, EWC loses only 0.2259 to its minimum.
That is exactly the EWC signature I predicted: a bounded, PSD, per-endpoint Fisher keeps all ten
permutations alive where SI's growing importance had killed them. Split-CIFAR100 came in at 0.5463 against
SI's 0.5363 — and reading the two per-context lines side by side, the mean absolute per-context difference
is 0.0126, so the curves are essentially the same curve, confirming what SI's difficulty-shaped line
already told me: on that 10-context CNN neither estimator over-accumulates and the choice barely matters.
And Split-MNIST landed at 0.9577, just under SI's 0.9852, also as predicted, and the *way* it lands under
is precise: four of the five contexts are {0.9916, 0.9903, 0.9814, 0.9933}, all above 0.98, averaging
0.98915 — the whole deficit against SI is one context, context 2 at 0.8317. If that single task had held
at its neighbors' level the Split-MNIST mean would be 0.98732, a dead tie with SI. So this is not a broad
EWC weakness on short tasks; it is one binary task the endpoint Fisher failed to protect, which is exactly
the `p(1-p) -> 0` under-weighting I derived — a cleanly solved task gets a near-zero Fisher, hence a slack
spring, and one later context happened to contest its trunk features. The aggregate verdict is clear:
EWC's bounded estimator fixed SI's catastrophic regime and pays only a small, localized tax on the short
clean one.

But look harder at that Permuted-MNIST line, because it is *also* EWC's tell. The decay is gentle but it is
real and monotone until the very end, and its step-to-step drops {0.0067, 0.0092, 0.0157, 0.0272, 0.0236,
0.0410, 0.0628, 0.0397, -0.0133} *accelerate*: the erosion per context grows from under a point early to
over six points by the seventh-to-eighth context, then the final step ticks *up* by 0.0133. Read both
features mechanically. The acceleration is the summed Fisher biting harder each context — `F-summed =
sum_t F_t` grows in every direction any context cared about, so the marginal rigidity added per new context
increases, and later permutations land progressively lower *while* the earliest contexts, re-anchored away
from their own optima toward the latest boundary, gradually un-protect. The recency uptick at the last
context is the flip side of the same mechanism: the most recent context is the one whose anchor is freshest
and whose Fisher has not yet been buried under later sums, so it is the least forgotten. The earliest
contexts are remembered *worse* than the recent ones — 0.9325 fading toward the 0.70s while context 10
sits at 0.7199 — and that ordering is not noise, it is structural. The harness's loop sums every context's
Fisher into one `_custom_importance` buffer and re-anchors `_custom_prev_params` to the *latest* boundary
each time. So across ten contexts the summed stiffness only ever grows, never shrinking, and on a fixed-
capacity net that accumulating rigidity is precisely what makes the later permutations land lower while the
early ones un-protect. EWC has converted SI's *explosion* into a *slow bleed*: bounded per context, yes,
but unbounded in the *sum* over contexts. So the next rung is not about the importance estimator anymore —
the Fisher is the right curvature, the CIFAR near-tie and the Permuted recovery both say so — it is about
*how importance accumulates across contexts*. I want EWC's protection at a cost that does not grow, and a
mechanism that frees capacity for the future instead of letting the summed springs rigidify the net.

Feel the wall first. EWC itself notes two quadratic penalties sum to a single quadratic penalty — a sum of
springs is a spring. So can I just merge them? For one coordinate,
`0.5 F_A (theta - theta*_A)^2 + 0.5 F_B (theta - theta*_B)^2` completes the square to a single spring
centered at `m = (F_A theta*_A + F_B theta*_B)/(F_A + F_B)` with stiffness `F_A + F_B`, up to a
`theta`-independent constant that vanishes under any gradient. So the merge is exact for the descent
dynamics: the *stiffness* merges into a running sum of Fishers — one parameter-sized object regardless of
context count, constant memory, good. But the merged anchor `m` is a Fisher-weighted blend of all past
optima; to update it incrementally I would carry `sum_t F_t` and `sum_t F_t theta*_t` and divide — still
constant memory, but this is just algebra on a pile of springs I have not questioned. The harness already
does the constant-memory thing for the stiffness (it sums Fishers). The real question is whether anchoring
a fresh penalty at *every* past optimum — or at a Fisher-weighted blend of them — is even what the
probabilistic story prescribes. If the springs are the wrong object, merging them faithfully just preserves
the error in a cheaper container.

So redo the derivation, slowly, without assuming the per-context stack. The weights are random; what I know
after contexts 1..k is the posterior `p(theta | T_{1:k})`. The contexts' data are conditionally
independent given `theta`, so the posterior builds recursively:
`p(theta | T_{1:k}) ∝ p(theta | T_{1:k-1}) p(T_k | theta)`. Stare at that. The posterior after all k
contexts is the posterior after the first k-1 — used as the *prior* — times the likelihood of just the
k-th. It is purely sequential, order-independent, and it never asks me to keep the individual context
likelihoods `p(T_t | theta)` around. The only object the recursion needs is the *running posterior*. That
is the tell: the per-context stack of springs is not in this equation at all. EWC's linear growth came not
from the Bayesian math but from how EWC *approximated* it.

How did the approximation go, and where is the branch point? The posterior over a deep net is intractable,
so Laplace: near the mode `theta*`, approximate `-log` of a distribution by its second-order Taylor
expansion, purely quadratic at a mode, with curvature the diagonal Fisher (folding in prior curvature).
Same for everyone. The branch point is *what* you Laplace-approximate. EWC applies the Gaussian to each
context's *likelihood* `p(T_t | theta)` separately — one Gaussian `N(theta*_t, F_t^{-1})` per context — and
their product is the product of springs, one anchored at each `theta*_t`. That is where the stack is born.
But the recursion says the object to approximate is the *whole running posterior*, not each likelihood.

Run the consistent version context by context, because the difference only shows from the third. Two
contexts, A then B: `log p(theta | D_A, D_B) = log p(D_B | theta) + log p(theta | D_A) + const`. Laplace
`p(theta | D_A)` around `theta*_A` gives one spring `0.5 sum_i F_{A,i} (theta_i - theta*_{A,i})^2`, so
learning B minimizes `L_B + 0.5 sum_i F_{A,i} (theta_i - theta*_{A,i})^2` — identical to EWC, no
disagreement yet. Now the third context C, after A and B. The exact log posterior is
`log p(D_C|theta) + log p(theta | D_A, D_B) + const`, and I must Laplace-approximate the running posterior
`p(theta | D_A, D_B)`. But I already threw D_A and D_B away — I do not have the true running posterior.
What I have is the approximation I built while learning B:
`log p(theta | D_A, D_B) ~ log p(D_B|theta) - 0.5 sum_i F_{A,i} (theta_i - theta*_{A,i})^2 + const`. So I
Taylor-expand *that* around its mode, which is `theta*_B` (the optimum of exactly this approximated
objective, so the first-order term vanishes there). Its curvature at `theta*_B` has two pieces: the NLL of
B contributes its Fisher `F_B`, and the spring around `theta*_A` contributes its own stiffness `F_A` (a
quadratic's second derivative is its coefficient, constant everywhere). So the curvature of the running
posterior at `theta*_B` is `F_A + F_B`, and the consistent Laplace approximation of `p(theta | D_A, D_B)`,
expanded at `theta*_B`, is a *single* quadratic `0.5 sum_i (F_{A,i} + F_{B,i})(theta_i - theta*_{B,i})^2`.
Learning C becomes `L_C + 0.5 sum_i (F_{A,i} + F_{B,i})(theta_i - theta*_{B,i})^2`.

Compare. EWC at context C carries *two* springs, at `theta*_A` and `theta*_B`. The consistent recursion
gives *one* spring, anchored at the *latest* optimum `theta*_B`, with stiffness the *sum* of past Fishers.
The number of penalty terms did not grow — it stayed one — and that fell out of just being consistent about
what to approximate. And once I see it, I can see *why* EWC's extra spring is not merely redundant but
actively wrong, which matters because the harness already half-does the right thing. `theta*_B` was not
found in a vacuum — it was found while *already* being pulled toward `theta*_A` by the spring
`0.5 F_A (theta - theta*_A)^2`. So information about A is already baked into *where `theta*_B` sits*; a
spring around `theta*_B` inherits A's pull. If on top of that I keep a *separate* spring still anchored at
`theta*_A`, I impose A's constraint a second time — I double-count A. Over a long sequence this is a
systematic bias toward the earliest contexts: they get re-asserted at every step through their surviving
anchors while later contexts are represented once. That is the harness's EWC behavior I diagnosed in the
feedback — the early Permuted contexts erode *and* the later ones land lower, the accelerating-drop
signature of an over-constraining, early-biased accumulation. The consistent fix is not to blend all the
old anchors (the merged `m` would preserve exactly this double-counted structure); it is to *drop* them and
anchor only at the latest optimum, because the latest optimum already encodes the cumulative pull of
everything before it. So the running state is two parameter-sized objects: the latest mean `theta*_{i-1}`
and a running sum of Fishers, `F-summed <- F-summed + F_i`. One spring, re-centered forward each context,
with an accumulating Fisher. Constant memory, constant per-step cost. This is, in fact, *exactly* what the
harness loop already does for EWC — sum the Fishers, re-anchor to the latest snapshot — which is why EWC
already had constant memory and yet still bled on Permuted-MNIST, its line accelerating from 0.9325 down
to 0.7066.

So is constant-memory single-spring EWC the end? No — and the remaining failure is precisely the one the
feedback showed. Two things. First, the fixed-capacity problem is only *compressed*, not solved: with
`F-summed = sum_t F_t` the stiffnesses only ever add, never shrink. Over ten Permuted contexts the summed
Fisher grows in every direction any context cared about, and a fixed-parameter net becomes so rigid that
later permutations cannot be learned — which is exactly why EWC's Permuted line decays, and why the decay
*accelerates* rather than settling. Second, in a lifelong setting contexts can recur, and the right thing is
to *revise* a context's contribution in light of fresh data, not pile a new redundant term on the stale one
— but I deliberately do not store per-context terms (that is what gives constant memory), so how do I
revise? Both problems point at the same missing capability: I need to *remove*, or down-weight, a context's
earlier contribution to the shared summary without ever having stored it separately.

Before reaching for the principled version, let me rule out the naive ways to bound the sum, because if a
crude cap worked I would take it. One: hard-clip `F-summed` at some ceiling. That bounds the magnitude but
the ceiling is arbitrary — there is no principle telling me where to put it — and clipping is discontinuous,
so a direction that any context cared about slams into the cap and stays there, losing all resolution among
the truly-stiff weights. Two: average the Fishers, `F-summed = (1/K) sum_t F_t`. That keeps the magnitude
bounded, but it treats a context a hundred boundaries ago as exactly as important as yesterday's, and as
`K` grows each new context's influence shrinks like `1/K`, so eventually the net stops protecting anything
new — and it needs the count `K` stored, and it still never fades a context that turned out to be
half-wrong. Three: keep only the latest context's Fisher and drop all history. That is bounded and
recency-weighted, but it forgets everything older than one boundary — it throws away the whole point of
consolidation. What I actually want interpolates between "keep everything" and "keep only the last" with a
single dial for how far back to remember, and it should come out of the inference, not be bolted on. None
of the three has that dial with a principle behind it, so I do not want a cap — I want a principled
fractional removal.

That phrasing names the move. Approximate-inference's expectation propagation refines one factor by
*dividing it out* of the product, recomputing, and multiplying back — but EP keeps a factor per context,
the linear memory I am fleeing. *Stochastic* EP relaxes exactly this: keep a *single shared* averaged
factor standing in for all of them, and treat any one factor's contribution as a *fraction* of the shared
one. To update, do not divide out a stored copy — down-weight the shared factor by a fraction `gamma < 1`
(a partial removal), refine on the new data, fold back in. That is the capability I need, and it costs
nothing: a scalar down-weighting of the one summary I already keep. Transcribe it into the Fisher
accumulation. The harness's EWC does `F-summed <- F-summed + F_i`; the stochastic-EP move is to *partially
remove* the previous shared contribution first, `F* <- gamma * F*_old + F_i`. That single change does both
jobs. For recurrence: scaling by `gamma` is the fractional removal of a context's previous presentation
before fresh data is added — no per-context factor, no task identity, only that a boundary occurred. For
capacity: `gamma < 1` turns the unbounded running sum into a *geometric* one. Unroll it,
`F*_i = sum_{t<=i} gamma^{i-t} F_t`, so a context that finished k boundaries ago contributes with weight
`gamma^k`, decaying geometrically; the stored summary settles to an effective sum on the order of
`1/(1-gamma)` recent contexts instead of climbing forever.

Put numbers on that decay so I pick `gamma` for a reason and not a habit. At `gamma = 0.9` the weights on a
context 1, 5, 10, 20 boundaries ago are `gamma^1 = 0.90`, `gamma^5 = 0.59`, `gamma^10 = 0.35`,
`gamma^20 = 0.12`, and the effective sum is `1/(1-0.9) = 10`. So the summary keeps roughly the last ten
contexts strongly represented — well matched to Permuted-MNIST's ten — while a context ten boundaries back
still contributes about a third of its weight rather than being erased: it fades gracefully, an old
half-wrong constraint softening rather than being dropped catastrophically. That is exactly the room to
keep learning new contexts on fixed capacity. The recurrence case gives a clean stability check: if one
context recurred at *every* boundary, the update `F* <- gamma F* + F` would settle to its fixed point
`F*_inf = F/(1-gamma) = 10 F` for `gamma = 0.9` — bounded, and about ten effective presentations' worth of
stiffness, which is the sensible stationary answer for a context seen over and over. The strict sum, by
contrast, would send that recurring context's stiffness to infinity, freezing the net; the decay keeps even
an infinitely-repeated context at a finite, ten-fold consolidation. The endpoints check out too: `gamma = 1` gives
`gamma^k = 1` for all k, recovering the strict undecayed sum (the harness EWC), and with the loss also
scaled by `gamma`, `gamma -> 0` leaves no protective penalty at all. A shorter `gamma = 0.8` would give an
effective window of `1/(1-0.8) = 5`, too short to hold ten Permuted contexts; a longer `0.95` gives 20,
drifting back toward EWC's unbounded accumulation. So `gamma = 0.9`, effective window ten, sits on the
sequence length, and `gamma` is simultaneously the EP partial-removal fraction and the explicit forgetting
knob — the same scalar wearing both hats, a sign it is the right scalar. I set `model.gamma = 0.9`,
overriding the framework default of `1.0` (at which Online EWC *is* EWC).

The penalty uses `0.5 * gamma * sum F* (theta - theta*)^2`: the `gamma` sits in the loss as well as the
accumulation, because the factor I hold the new context to is the partially-removed shared summary, not the
raw sum. There is a genuinely ambiguous reading where one regularizes against the undamped `F*` and only
decays the next update; I keep the two uses tied — applied stiffness `gamma F*_{i-1}`, post-context update
`gamma F*_{i-1} + F_i` — because the cavity-style reading says the object I regularize against is already
the down-weighted shared factor.

One harness-specific subtlety, and it is the difference between this and a from-scratch Online EWC. The
loop accumulates whatever I return *additively* into `_custom_importance` — `existing + my_return`. But I
want the buffer to land on `gamma * existing + F_new`, not `existing + F_new`. So inside
`estimate_importance` I compute the full decayed-plus-new Fisher `gamma * existing + F_new` myself, then
*subtract* the existing value and return the difference `(gamma - 1) * existing + F_new`, so the loop's
`existing + ((gamma-1)*existing + F_new)` lands exactly on `gamma * existing + F_new` (with `existing = 2,
F_new = 1, gamma = 0.9`: increment `-0.2 + 1 = 0.8`, buffer `2 + 0.8 = 2.8 = 0.9*2 + 1`). That subtraction
is purely an adapter to the harness's add-don't-replace convention, making the carried buffer hold the
geometric decayed sum the derivation prescribes. The
importance estimator itself is unchanged from EWC: the diagonal Fisher from softmax-weighted squared
gradients over all classes, eval mode, capped sample — the model's own expected curvature, first-order and
cheap. (The full module, with the subtraction adapter and the `gamma` in both places, is in the answer.)

So the delta from EWC is one scalar and one adapter, both surgical, and the expectations against EWC's
measured line are sharp. On Permuted-MNIST — the benchmark that decides the aggregate — I expect the
geometric decay to relieve the accumulating rigidity that pulled EWC's line from 0.9325 down through 0.7066
with an *accelerating* drop: the later permutations should land *higher* than EWC's {0.7463, 0.7066, 0.7199}
because `gamma F*` stops the summed Fisher from climbing without bound and frees capacity, and although the
very earliest contexts may fade slightly below EWC's {0.9325, 0.9258} (the honest cost of `gamma < 1`
favoring the recent past), the *average* across ten contexts should edge above EWC's 0.8381. If it does
not, my diagnosis of EWC's bleed — unbounded *summed* Fisher over-rigidifying — was wrong. On Split-MNIST I
expect a near-tie with or slight improvement over EWC's 0.9577: five short tasks well inside `1/(1-gamma) ~
10` contexts of effective memory, so the decay barely engages, and it might even relieve the context-2
anomaly (0.8317) by not over-constraining the trunk while that task's slack Fisher leaves it exposed. On
Split-CIFAR100 I expect essentially EWC's number, mid-0.5s — the estimator is identical, the two prior
lines already agreed to within 0.0126 per context, and ten contexts is right at the decay's effective
window, so `gamma F*` neither frees much nor costs much. The win, if it comes, is modest and concentrated
where the theory says it should be: the long sequence whose accumulating springs were the only thing left
over-constraining the net. The bar to clear is EWC's 0.9577 / 0.8381 / 0.5463, with the Permuted-MNIST
average the decisive number.
