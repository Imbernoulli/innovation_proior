SparseFool came back with exactly the mixed signature I predicted, and the per-model spread is the whole
story. `asr = 0.167` on `Rebuffi-R18-L2`, `0.113` on `Augustin-L2`, `0.280` on `Engstrom-L2`, mean about
`0.187`. So it beat OnePixel's `0.153` mean — a real gain — but it *trailed* OnePixel on `Augustin`
(`0.113` versus `0.18`) while leading clearly on `Engstrom` (`0.280` versus `0.18`) and `Rebuffi`
(`0.167` versus `0.10`). That non-uniform result is the diagnosis written in numbers: SparseFool's engine
is local-linear, and where the boundary best fits a hyperplane it wins decisively, but on whichever model's
boundary is least linear (`Augustin` here) twenty relinearizations chase a boundary the linear model keeps
mis-estimating, and `lam=3.0` overshoots a wrong target without crossing. The method's strength and its
fragility are the same assumption. And stepping back across all four rungs so far, every one is limited by
*something other than the budget itself*: Pixle guessed locations, JSMA committed greedily to a local
first-order signal, OnePixel ran out of generations, SparseFool leaned on a brittle local-linear boundary.
None of them is *built for the discrete `L0` set*. That is the gap the strongest baseline closes.

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
extreme — I use *corners* of the color cube, `{0,1}^c`, spending each precious pixel maximally. For the
support, the move that never leaves the budget is a *swap*: pick `A` from the current perturbed support `M`
and `B` from the clean complement of equal size, restore `A` to clean, give `B` fresh corner colors, so
`M' = (M\A) union B` and `|M'| = k` always. Every candidate is feasible by construction — no projection, no
clipping, no rejected samples, which is the structural cleanliness all the budget-bookkeeping pain on
earlier rungs was groping toward.

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
sank any black-box gradient route. The relaxation from "exact top `k`" to "`k` among the `m` smallest" is
the whole trick: it is what converts a prohibitive `O(d)` identification problem into a `k log k`-style
hitting time. The decaying schedule is exactly what runs the broad-support exploration
early and approaches the one-pixel refinement regime whose analysis gives that bound. So the strongest rung
is the one whose proposal distribution is *derived from* the `L0` structure, not adapted to it from a
continuous method.

Now the part specific to *this* task, because the harness does not wrap a library here — `torchattacks` has
no Sparse-RS, so the fill *inlines* the full L0 attack, and the literal scaffold edit is that whole inlined
function. The configuration is `n_queries = 10000`, `p_init = 0.8`, `eps = pixels = 24`. The proposal is
exactly the swap above: maintain per image an index set `b` of the `24` perturbed pixels and its complement
`be`, map a pixel index `p` to coordinates `(p // W, p % W)` (matching the harness's channel-wise `L0`
count — a pixel changes if *any* channel differs), and swap `eps_it = max(int(p_selection(it)*eps), 1)`
entries between `b` and `be` each step, redrawing binary corner colors on the entrants. The schedule
`p_selection` is the piecewise-constant decay with divisors `{2,4,5,6,8,10,12,15,20}` on the reference
intervals, rescaled to `n_queries`. The accept rule keeps a move if the loss improves *or* the margin is
already negative (locking in a flip), and only spends queries on images whose current margin is still
positive — so easy samples stop early and the budget concentrates on the stubborn ones. The single-pixel
case resamples the entering color until it differs from the current one, so a one-pixel refinement step is
never a wasted no-op query. This is `10000` directed, *always-feasible* proposals per image with a schedule
that provably explores-then-refines — a different order of search entirely from SparseFool's twenty
relinearizations or OnePixel's six generations.

Where does that leave my expectations against the SparseFool ceiling of `0.187`? Sparse-RS attacks the
*one* thing every prior rung got wrong — it is built natively for the discrete `L0` support, with a feasible
proposal, a provably query-efficient decaying schedule, and corner colors that use the budget maximally —
and it gets two-to-three orders of magnitude more queries to do it (`10000` versus `~15`–`120`). So I
expect not a modest gain but a *step change*: from the high-teens of the best continuous method into the
high-eighties-to-mid-nineties of percent across all three models. The per-model spread should also
*flatten* relative to SparseFool, because random search over feasible supports does not depend on the local
linearity of any one model's boundary — the property that made SparseFool swing from `0.113` to `0.280`.
My falsifiable expectation: Sparse-RS's mean ASR clears `0.187` decisively and lands above `0.85` on every
model, with `Engstrom` (the model SparseFool already found softest) at or near the top — and the spread
across models narrower than SparseFool's because the method no longer hinges on a boundary-linearity
assumption. If it *failed* to dominate — if it landed merely in the SparseFool range — that would mean
`10000` queries were not enough on these robust models, and the only thing stronger would be a method that
uses the gradient the harness permits to *guide* the support choice rather than searching it blind.

The delta from the previous rung, concretely: where SparseFool returned `attack(images, labels)` from a
local-linear boundary-following white-box search and scored `0.187`, this rung returns `x_best` from an
inlined random search over feasible `L0` supports — trading a continuous method adapted to `L0` for one
*derived from* `L0`, with binary corner colors, a provably efficient decaying swap schedule, and `10000`
always-feasible queries per image. The full scaffold module is in the answer. I expect a step change to the
high-eighties-plus across all three models, flattening the per-model spread SparseFool's linearity
assumption produced — the strongest baseline, and the bar a finale would have to clear.
