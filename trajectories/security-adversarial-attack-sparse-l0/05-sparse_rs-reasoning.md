SparseFool came back with exactly the mixed signature I predicted, and the per-model spread is the whole
story. `asr = 0.167` on `Rebuffi-R18-L2`, `0.113` on `Augustin-L2`, `0.280` on `Engstrom-L2`, mean about
`0.187`. As counts over the `150`-sample runs that is `25`, `17`, `42` flips. So it beat OnePixel's `0.153`
mean — a real gain — but it *trailed* OnePixel on `Augustin` (`0.113` versus `0.18`, `17` flips versus
`27`) while leading clearly on `Engstrom` (`0.280` versus `0.18`, `42` versus `27`) and `Rebuffi` (`0.167`
versus `0.10`, `25` versus `15`). Look at the spread: SparseFool's three numbers span `0.113` to `0.280`,
a range of `0.167`, where OnePixel's spanned only `0.10` to `0.18`, a range of `0.08` — so the per-model
spread *doubled* going from the population search to the boundary-linearization search, exactly the
widening I expected from a method gated by boundary linearity rather than overall robustness. That
non-uniform result is the diagnosis written in numbers: SparseFool's engine is local-linear, and where the
boundary best fits a hyperplane it wins decisively (`Engstrom`, `42` flips), but on whichever model's
boundary is least linear (`Augustin` here, the one model where it actually *regressed* below the
gradient-free DE) twenty relinearizations chase a boundary the linear model keeps mis-estimating, and
`lam=3.0` overshoots a wrong target without crossing. The method's strength and its fragility are the same
assumption. And stepping back across all four rungs so far, every one is limited by *something other than
the budget itself*: Pixle guessed locations (mean `0.011`), JSMA committed greedily to a local first-order
signal (`0.047`), OnePixel ran out of generations (`0.153`), SparseFool leaned on a brittle local-linear
boundary (`0.187`). The mean has climbed by roughly `4x`, `3.3x`, `1.2x` across those steps — decelerating
hard, each new method fixing the previous one's flaw only to expose its own. None of them is *built for
the discrete `L0` set*. That is the gap the strongest baseline closes.

The realization that reframes everything: the `L0` budget is not a norm radius, it is a *combinatorial
support*. It says only a selected set `M` of `k` pixels may move at all; once a pixel is in `M`, its
channels can go anywhere in `[0,1]`. So the object to optimize is not a dense vector in a ball — it is a
*pair*, a support `M` and the colors on it. This is why every continuous method stalled. In an `L_inf` or
`L_2` attack you step and project back into a smooth convex ball; here projection means *choosing* a
support — a discrete top-`k` — so a small continuous change before projection causes a large support change
after, the iteration lurches, and the support settles poorly. That is precisely SparseFool's and JSMA's
trouble seen from the `L0` side. And in any gradient-based form the budget fights the gradient, which wants
to spread signal over many pixels even though only `k` survive. So I drop continuous optimization
altogether and use the one primitive that needs nothing but feasible candidates and scalar comparisons:
*random search*. Keep the best candidate, sample a new *legal sparse* candidate, query its loss, keep it
if it improves. The accept-if-improves loop is trivial; the entire problem is the *distribution over legal
sparse candidates*, and getting that distribution right is what makes this the strongest rung.

I split a candidate into support and color and design the proposal for each. The objective is the
untargeted margin `L(z) = f_y(z) - max_{r!=y} f_r(z)`, minimized; its *sign* is the misclassification
certificate, `L(z) < 0` is exactly a flip, so I never need a separate success test. For colors: because
the budget penalizes magnitude not at all — a pixel in `M` costs the same whether nudged or slammed to an
extreme — I use *corners* of the color cube, `{0,1}^c`, spending each precious pixel maximally. Let me
justify the corners choice against the obvious alternative, a continuous color search per pixel, because
it is the same trap OnePixel walked into. If I let each of the `k` pixels take a continuous color, the
proposal distribution has `k * c = 24 * 3 = 72` extra continuous dimensions to explore, and random search
over `72` continuous dimensions is hopeless in any realistic budget — that is precisely the color cube DE
had to search. Restricting to `{0,1}^c` collapses each pixel's color to one of `2^3 = 8` corners, so the
color search per pixel is a tiny discrete draw rather than a continuous hunt, and because magnitude is free
under `L0` the corners are also the *most effective* colors, not merely the cheapest. So corners are
doubly correct: they use the budget maximally and they keep the proposal low-dimensional enough for random
search to work. For the support, I have to pick a move, and the choice matters enough to enumerate. The naive move is to
*add* a pixel or *remove* a pixel from `M` — but adding pushes `|M|` to `k+1`, breaking the budget, and
removing wastes a pixel; either way I would have to repair feasibility afterward, which is the projection
lurch I am trying to escape. The move that never leaves the budget is a *swap*: pick `A` from the current
perturbed support `M` and `B` from the clean complement of equal size, restore `A` to clean, give `B`
fresh corner colors, so `M' = (M\A) union B` and `|M'| = k` always. A swap of size `s` holds `|M|` at
exactly `k` for any `s`, which is precisely why the schedule can vary the swap size freely without ever
touching the budget — a property no add/remove move has. Every candidate is feasible by construction — no
projection, no clipping, no rejected samples, which is the structural cleanliness all the
budget-bookkeeping pain on earlier rungs was groping toward.

The accept rule deserves the same scrutiny, because I could accept worse candidates (simulated annealing)
or only-better ones (greedy random search), and the choice interacts with the feasible-proposal design.
Annealing would let the search climb out of a locally-good support by occasionally accepting a worse one —
tempting, since escaping local optima is the whole theme of this ladder. But annealing's escape mechanism
is *needed* only when the proposal cannot itself reach distant supports; here the *large early swaps
already provide that escape* — a nine-pixel swap relocates more than a third of the support in one move, so
the proposal distribution, not the accept rule, is doing the exploration. Given that, accept-if-improves is
the right rule: it never throws away a good support, it needs no temperature schedule to tune, and it pairs
cleanly with the margin certificate — I keep a move if the loss strictly improves *or* the margin has gone
negative, so a flip is locked in the instant it happens. The exploration lives in the schedule, the
exploitation lives in the accept rule, and they are cleanly separated. That separation is why I do not need
the extra machinery annealing would add.

One more choice is load-bearing: the objective is the *margin* `L(z) = f_y(z) - max_{r!=y} f_r(z)`, not
cross-entropy. Two reasons. First, its sign is a certificate — `L(z) < 0` is exactly a misclassification —
so the accept rule and the success test are the same quantity and I never run a separate check. Second,
cross-entropy on the true class saturates: once the true-class probability is driven near zero its log-loss
gradient flattens, so a candidate that has *nearly* flipped and one that has *barely* dented the true class
can look similarly good to a saturating loss, blurring the ranking random search relies on. The raw margin
does not saturate — it keeps decreasing linearly as the runner-up logit overtakes the true one — so it
gives random search a clean, monotone signal all the way to the flip. On a robust model where most
candidates sit near the boundary, that unsaturated ranking is exactly what lets accept-if-improves make
steady progress instead of plateauing on a flat loss.

The swap *size* has to change over the run, and this is the heart of the method. Swap many pixels early and
I escape a poor random initial support fast; keep swapping many late and I destroy the good support I have
found; swap one from the start and I refine gently but spend far too many queries finding the right region.
So the swap fraction *decays* — large for exploration, small for refinement — which neither a constant-large
nor a constant-small schedule can do. There is a real reason this is query-efficient and not merely
plausible, and it is worth tracing because it tells me what the decay buys. Analyze single-pixel swaps on a
binary-linear model as a coupon-collector chain. In the white-box limit the optimal `k`-sparse attack is to
pick the `k` coordinates with the smallest entries of an effective weight vector; the black-box obstacle is
that this vector is hidden and reading it coordinate by coordinate costs `O(d)` queries — hopeless at
`d ~ 1000` on CIFAR, let alone ImageNet scale. But I do not need the *exact* top `k`: a relaxed goal is to
collect `k` coordinates among the `m` smallest, for some `m > k`. Model the support's progress toward that
goal as a Markov chain on the count of "good" coordinates currently in the support; a single swap improves
the count when it drops a bad coordinate and adds a good one, with a probability I can write down exactly,
and the expected time to fill the support is a sum of geometric waiting times. Working that sum out gives
the expected number of queries `E[t_k] < (d-k)k(ln k + 2)/(m-k)`, which is *sublinear* in the input
dimension `d` once `m-k` grows with `d` — beating the `O(d)` cost of coordinate-wise weight estimation that
sank any black-box gradient route.

Let me actually plug this task's numbers into that bound, because it tells me whether `10000` queries is
enough or a fantasy. Here `d = H*W = 1024` spatial positions and `k = 24`, so `d - k = 1000`,
`ln k + 2 = ln 24 + 2 ~= 3.18 + 2 = 5.18`, and the numerator is `1000 * 24 * 5.18 ~= 124300`. The bound
then depends entirely on the relaxation width `m - k`. If I aim for the `k` best among the `m = 2k = 48`
smallest coordinates, `m - k = 24` and `E[t_k] < 124300 / 24 ~= 5180` queries — comfortably inside the
`10000` budget. If I tighten the goal to `m = 1.5k = 36`, then `m - k = 12` and the bound roughly doubles
to `~10360`, right at the edge of the budget. So the arithmetic is not just decorative: it says `10000`
queries buys me the `k`-among-`2k` relaxation with margin to spare, and it is the decaying schedule that
*implements* that relaxation — broad swaps early to collect good coordinates into the support fast, single-
pixel swaps late to settle the exact `k` among the region already found. The relaxation from "exact top
`k`" to "`k` among the `m` smallest" is the whole trick: it is what converts a prohibitive `O(d)`
identification problem into a `k log k`-style hitting time, and the budget is sized for it. So the
strongest rung is the one whose proposal distribution is *derived from* the `L0` structure, not adapted to
it from a continuous method.

It is worth pausing on the alternative the bound rules out, because it is the obvious black-box move and
seeing *why* it loses cements why random search is the right primitive. A finite-difference or NES-style
gradient estimator would probe the model to reconstruct the effective per-coordinate importance, then pick
the top `k`. But reconstructing a `d = 1024`-dimensional importance vector to any useful fidelity costs on
the order of `d` queries — at least one probe per coordinate, realistically several — so `O(d)` is `~1000`
to a few thousand queries *just to estimate the ranking once*, before placing a single adversarial pixel,
and the estimate is noisy on a flattened surface where the true gradient is small. The coupon-collector
random search never estimates the ranking at all; it *samples* feasible supports and lets the accept rule
keep the good ones, reaching the `k`-among-`2k` goal in `~5180` queries with no explicit gradient. So the
comparison is `O(d)` estimation-then-exploitation versus a sublinear `(d-k)k(ln k+2)/(m-k)` direct search,
and on these robust models — where a black-box gradient estimate would be both expensive and noisy — the
direct feasible-support search strictly dominates. That is the deeper reason the strongest baseline is
gradient-*free*: not because the gradient is unavailable (the harness grants it), but because *estimating*
it black-box costs more than the search it would inform.

Now the part specific to *this* task, because the harness does not wrap a library here — `torchattacks` has
no Sparse-RS, so the fill *inlines* the full L0 attack, and the literal scaffold edit is that whole inlined
function. The configuration is `n_queries = 10000`, `p_init = 0.8`, `eps = pixels = 24`. Let me trace the
schedule concretely, because `eps_it = max(int(p_selection(it) * eps), 1)` turns the abstract "decay" into
an exact sequence of swap sizes and I want to see it explore-then-refine on paper. Early, for iterations in
`(0, 50]`, `p_selection = p_init/2 = 0.4`, so `eps_it = int(0.4 * 24) = int(9.6) = 9` — nine-pixel swaps,
broad exploration. Then `(50, 200]` gives `0.8/4 = 0.2 -> int(4.8) = 4`; `(200, 500]` gives
`0.8/5 = 0.16 -> int(3.84) = 3`; `(500, 1000]` gives `0.8/6 ~= 0.133 -> int(3.2) = 3`; `(1000, 2000]`
gives `0.8/8 = 0.1 -> int(2.4) = 2`; `(2000, 4000]` gives `0.8/10 = 0.08 -> int(1.92) = 1`; and every
interval past `4000` — divisors `12, 15, 20` — gives `int(1.6) = 1`, `int(1.28) = 1`, and finally
`0.8/20 = 0.04 -> int(0.96) = 0`, which the `max(., 1)` floors to `1`. So the realized swap-size sequence
is `9, 4, 3, 3, 2, 1, 1, 1, 1`: it opens by rearranging nine of the `24` pixels at once and closes with
pure single-pixel refinement — exactly the exploration-to-refinement profile the coupon-collector analysis
says to run, and the `max(., 1)` floor is what keeps the last several thousand queries doing useful
single-pixel work instead of degenerating to no-ops. The proposal maintains per image an index set `b` of
the `24` perturbed pixels and its complement `be`, maps a pixel index `p` to coordinates
`(p // W, p % W)` (matching the harness's channel-wise `L0` count — a pixel changes if *any* channel
differs), and swaps `eps_it` entries between `b` and `be` each step, redrawing binary corner colors on the
entrants. The accept rule keeps a move if the loss improves *or* the margin is already negative (locking in
a flip), and only spends queries on images whose current margin is still positive — so easy samples stop
early and the budget concentrates on the stubborn ones. The single-pixel case resamples the entering color
until it differs from the current one, so a one-pixel refinement step is never a wasted no-op query. This
is `10000` directed, *always-feasible* proposals per image with a schedule that provably explores-then-
refines — a different order of search entirely from SparseFool's twenty relinearizations or OnePixel's six
generations.

Two implementation details in the inlined function are not cosmetic; they are what make `10000` queries a
real budget rather than a nominal one. The first is the initialization: each image starts from a *random*
`eps`-pixel support with random binary corner colors, not from the clean image. That matters because the
clean image has margin strictly positive by construction (the harness only feeds correctly-classified
samples), so starting there wastes early queries just to leave zero; a random `24`-pixel corner
perturbation already probes a nontrivial support on query one, and the coupon-collector clock starts
counting from a support that is already `k` pixels large. The second is the active-set batching: the loop
computes `idx_to_fool = (margin_min > 0)` each iteration and only proposes moves for images still on the
wrong side of the boundary, breaking entirely once none remain. So an easy image that flips at query `200`
consumes no queries `201..10000` — those are silently redirected to the stubborn images that are still
positive. This is why the `10000` figure is a *per-image ceiling*, not a per-image expenditure: the total
model work concentrates on the hard tail, and the coupon-collector bound of `~5180` queries for the
`k`-among-`2k` relaxation is what most hard images actually need, well inside the ceiling. Together these
two details mean the budget is spent where the coupon-collector analysis says it pays, not uniformly across
already-solved samples.

Where does that leave my expectations against the SparseFool ceiling of `0.187`? Sparse-RS attacks the
*one* thing every prior rung got wrong — it is built natively for the discrete `L0` support, with a feasible
proposal, a provably query-efficient decaying schedule, and corner colors that use the budget maximally —
and it gets two-to-three orders of magnitude more queries to do it (`10000` versus `~15`–`120`). So I
expect not a modest gain but a *step change*: from the high-teens of the best continuous method into the
high-eighties-to-mid-nineties of percent across all three models. The per-model spread should also
*flatten* relative to SparseFool, because random search over feasible supports does not depend on the local
linearity of any one model's boundary — the property that made SparseFool swing from `0.113` to `0.280`.
My falsifiable expectation: Sparse-RS's mean ASR clears `0.187` decisively and lands above `0.85` on every
model, with `Engstrom` (the model SparseFool already found softest, `42` flips) at or near the top — and
the spread across models narrower than SparseFool's `0.167`-wide swing, because the method no longer hinges
on a boundary-linearity assumption. If it *failed* to dominate — if it landed merely in the SparseFool
range — that would mean `10000` queries were not enough on these robust models, and the only thing stronger
would be a method that uses the gradient the harness permits to *guide* the support choice rather than
searching it blind.

The delta from the previous rung, concretely: where SparseFool returned `attack(images, labels)` from a
local-linear boundary-following white-box search and scored `0.187`, this rung returns `x_best` from an
inlined random search over feasible `L0` supports — trading a continuous method adapted to `L0` for one
*derived from* `L0`, with binary corner colors, a provably efficient decaying swap schedule that runs its
swap size from `9` down to `1`, and `10000` always-feasible queries per image. The full scaffold module is
in the answer. I expect a step change to the high-eighties-plus across all three models, flattening the
per-model spread SparseFool's linearity assumption produced — the strongest baseline, and the bar a finale
would have to clear.
