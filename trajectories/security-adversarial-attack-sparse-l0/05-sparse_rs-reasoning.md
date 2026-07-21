SparseFool came back with the mixed signature I predicted, and the per-model spread is the story. `asr =
0.167` on `Rebuffi-R18-L2`, `0.113` on `Augustin-L2`, `0.280` on `Engstrom-L2`, mean `~0.187` — as counts
`25`, `17`, `42` flips. So it beat OnePixel's `0.153` mean but *trailed* OnePixel on Augustin (`17` versus
`27`) while leading clearly on Engstrom (`42` versus `27`) and Rebuffi (`25` versus `15`). SparseFool's three
numbers span `0.113` to `0.280`, a range of `0.167`, where OnePixel's spanned `0.10` to `0.18`, a range of
`0.08` — the per-model spread *doubled* going from population search to boundary-linearization, exactly the
widening I expected from a method gated by boundary linearity rather than overall robustness. Where the
boundary best fits a hyperplane it wins decisively (Engstrom), and on whichever model's boundary is least
linear (Augustin here, where it actually *regressed* below the gradient-free DE) twenty relinearizations
chase a boundary the linear model keeps mis-estimating and `lam=3.0` overshoots without crossing. The
method's strength and its fragility are the same assumption. Stepping back across all four rungs, every one
is limited by *something other than the budget itself*: Pixle guessed locations (`0.011`), JSMA committed
greedily to a local first-order signal (`0.047`), OnePixel ran out of generations (`0.153`), SparseFool
leaned on a brittle local-linear boundary (`0.187`). The mean climbed by roughly `4x`, `3.3x`, `1.2x` —
decelerating hard, each method fixing the previous flaw only to expose its own. None is *built for the
discrete `L0` set*. That is the gap the strongest baseline closes.

The realization that reframes everything: the `L0` budget is not a norm radius, it is a *combinatorial
support*. It says only a selected set `M` of `k` pixels may move; once a pixel is in `M`, its channels can go
anywhere in `[0,1]`. So the object to optimize is a *pair* — a support `M` and the colors on it — not a dense
vector in a ball. That is why every continuous method stalled: in an `L_inf`/`L_2` attack you step and
project back into a smooth convex ball, but here projection means *choosing* a support, a discrete top-`k`,
so a small continuous change before projection causes a large support change after, the iteration lurches,
and the support settles poorly (SparseFool's and JSMA's trouble seen from the `L0` side). And in any gradient
form the budget fights the gradient, which wants to spread signal over many pixels though only `k` survive.
So I drop continuous optimization and use the one primitive that needs only feasible candidates and scalar
comparisons: *random search* — keep the best, sample a new legal sparse candidate, query its loss, keep it if
it improves. The accept-if-improves loop is trivial; the entire problem is the *distribution over legal
sparse candidates*, and getting that right is what makes this the strongest rung.

I split a candidate into support and color and design the proposal for each. For colors: the budget penalizes
magnitude not at all, so I use *corners* of the color cube, `{0,1}^c`, spending each pixel maximally. Against
the obvious alternative — a continuous color per pixel — this is the same trap OnePixel walked into:
`k*c = 24*3 = 72` extra continuous dimensions, hopeless for random search in budget (the color cube DE had to
search). Restricting to `{0,1}^c` collapses each pixel's color to one of `2^3 = 8` corners, a tiny discrete
draw, and because magnitude is free the corners are also the *most effective* colors, not merely the
cheapest — doubly correct. For the support I have to pick a move. *Adding* a pixel pushes `|M|` to `k+1`,
breaking the budget; *removing* wastes a pixel; either way I would repair feasibility afterward, the
projection lurch I am escaping. The move that never leaves the budget is a *swap*: pick `A` from the
perturbed support `M` and `B` from the clean complement of equal size, restore `A` to clean, give `B` fresh
corner colors, so `M' = (M\A) union B` and `|M'| = k` always — for any swap size `s`, which is precisely why
the schedule can vary the swap size freely without ever touching the budget. Every candidate is feasible by
construction — no projection, no clipping, no rejected samples, the structural cleanliness all the earlier
budget-bookkeeping pain was groping toward.

The accept rule: I could accept worse candidates (annealing) or only better ones (greedy random search).
Annealing's escape mechanism is needed only when the proposal cannot itself reach distant supports, but here
the *large early swaps already provide that escape* — a nine-pixel swap relocates more than a third of the
support in one move — so the proposal distribution, not the accept rule, does the exploration. Given that,
accept-if-improves is right: it never throws away a good support, needs no temperature schedule, and pairs
cleanly with the objective — the untargeted margin `L(z) = f_y(z) - max_{r!=y} f_r(z)`, minimized. The margin
is load-bearing for two reasons: its *sign* is the misclassification certificate (`L(z) < 0` is exactly a
flip), so the accept rule and success test are the same quantity and I never run a separate check; and unlike
cross-entropy it does not *saturate* — once the true-class probability is near zero, log-loss flattens and a
nearly-flipped candidate looks similar to a barely-dented one, blurring the ranking random search relies on,
while the raw margin keeps decreasing linearly as the runner-up overtakes. On a robust model where most
candidates sit near the boundary, that unsaturated ranking is what lets accept-if-improves make steady
progress. So exploration lives in the schedule, exploitation in the accept rule, cleanly separated — no
annealing machinery needed.

The swap *size* changing over the run is the heart of the method. Swap many pixels early to escape a poor
random initial support fast; keep swapping many late and I destroy the good support I found; swap one from
the start and I refine gently but waste queries finding the right region. So the swap fraction *decays* —
large for exploration, small for refinement. There is a real reason this is query-efficient. Analyze
single-pixel swaps on a binary-linear model as a coupon-collector chain: the optimal `k`-sparse attack picks
the `k` coordinates with the smallest entries of an effective weight vector, but that vector is hidden and
reading it coordinate-by-coordinate costs `O(d)` queries — hopeless at `d ~ 1000`. I do not need the *exact*
top `k`, though; a relaxed goal is to collect `k` coordinates among the `m` smallest for some `m > k`. Model
the support's progress as a Markov chain on the count of good coordinates in the support; a swap improves the
count when it drops a bad coordinate and adds a good one, and the expected time to fill gives
`E[t_k] < (d-k)k(ln k + 2)/(m-k)`, *sublinear* in `d` once `m-k` grows with `d`, beating the `O(d)` cost of
coordinate-wise weight estimation that sank any black-box gradient route. Plugging this task's numbers:
`d = 1024`, `k = 24`, `ln 24 + 2 ~= 5.18`, numerator `1000*24*5.18 ~= 124300`. Aim for the `k` best among
`m = 2k = 48`, so `m-k = 24` and `E[t_k] < 124300/24 ~= 5180` queries — comfortably inside a `10000` budget;
tighten to `m = 1.5k = 36` and the bound roughly doubles to `~10360`, right at the edge. So `10000` queries
buys the `k`-among-`2k` relaxation with margin, and the decaying schedule is what *implements* it: broad
swaps early to collect good coordinates fast, single-pixel swaps late to settle the exact `k`. The relaxation
from "exact top `k`" to "`k` among the `m` smallest" is the whole trick — it converts a prohibitive `O(d)`
identification into a `k log k`-style hitting time, and the budget is sized for it. That is also the deeper
reason the strongest baseline is gradient-*free*: not because the gradient is unavailable, but because
*estimating* it black-box (a finite-difference or NES reconstruction of a `1024`-dim importance vector,
`O(d)` noisy probes before placing a single pixel) costs more than the direct feasible-support search it
would inform.

Now the task, because the harness has no Sparse-RS to wrap — the fill *inlines* the full `L0` attack, and the
literal edit is that whole function. The configuration is `n_queries = 10000`, `p_init = 0.8`,
`eps = pixels = 24`, and `eps_it = max(int(p_selection(it) * eps), 1)` turns the abstract decay into an exact
swap-size sequence. Running the schedule's intervals gives `9, 4, 3, 3, 2, 1, 1, 1, 1`: it opens by
rearranging nine of the 24 pixels at once and closes with pure single-pixel refinement — exactly the
explore-then-refine profile the coupon-collector analysis prescribes, with the `max(.,1)` floor keeping the
last several thousand queries doing useful single-pixel work instead of degenerating to no-ops. The proposal
keeps per image an index set `b` of the 24 perturbed pixels and its complement `be`, maps a pixel index `p`
to `(p // W, p % W)` (matching the harness's channel-wise count), swaps `eps_it` entries between `b` and `be`,
and redraws binary corner colors on the entrants; the single-pixel case resamples the entering color until it
differs, so a refinement step is never a wasted no-op. Two implementation details make `10000` a real budget
rather than nominal. Each image starts from a *random* 24-pixel corner support, not the clean image: the
clean image has strictly-positive margin by construction (the harness only feeds correctly-classified
samples), so starting there wastes early queries just to leave zero, whereas a random 24-pixel corner probes
a nontrivial support on query one and the coupon-collector clock starts from a support already `k` pixels
large. And the loop computes `idx_to_fool = (margin_min > 0)` each iteration, proposing moves only for images
still positive and breaking when none remain — so an image that flips at query `200` consumes no queries
`201..10000`, and the total model work concentrates on the hard tail. The `10000` figure is a *per-image
ceiling*, not an expenditure, and the `~5180`-query coupon-collector bound is what most hard images actually
need, well inside it.

So expectations against SparseFool's `0.187`. Sparse-RS attacks the *one* thing every prior rung got wrong —
it is built natively for the discrete `L0` support, with a feasible proposal, a provably query-efficient
decaying schedule, and corner colors that use the budget maximally — and it gets two-to-three orders of
magnitude more queries to do it (`10000` versus `~15`-`120`). So I expect not a modest gain but a *step
change*: from the high-teens of the best continuous method into the high success regime, on the order of
`90%` across all three models. And the per-model spread should *flatten* relative to SparseFool, because
random search over feasible supports does not depend on the local linearity of any one model's boundary — the
property that swung SparseFool from `0.113` to `0.280`. If instead it landed merely in the SparseFool range,
`10000` queries were not enough on these robust models, and the only thing stronger would be a method that
uses the gradient the harness permits to *guide* the support choice rather than searching it blind.
