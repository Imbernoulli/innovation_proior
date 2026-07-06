OnePixel landed where I expected and confirmed the diagnosis. `asr = 0.10` on `Rebuffi-R18-L2`, `0.18` on
`Augustin-L2`, `0.18` on `Engstrom-L2`, mean about `0.153`. As counts over the `150`-sample runs that is
`15`, `27`, `27` flips — so the black-box population search more than tripled JSMA's `0.047` (mean
`0.153/0.0467 ~= 3.3x`), exactly because it escapes the local optima that trapped greedy saliency. The
per-model multipliers are worth reading: Rebuffi `4 -> 15` flips (`3.75x`), Augustin `7 -> 27` (`3.85x`),
Engstrom `10 -> 27` (`2.7x`). So the two models JSMA found softer and harder have converged toward each
other under DE — Augustin and Engstrom now tie at `27` — while Rebuffi remains distinctly the hardest at
`15`. That convergence is the signature of a *global* evaluative search: DE reads the whole-image
objective, so the per-model differences that a first-order signal exposed (Engstrom's less-suppressed
gradient) matter less, and what is left is a cruder ranking by overall robustness, on which Rebuffi is
simply the toughest. But the absolute level, mid-teens of percent, is the ceiling I warned about: six
generations of differential evolution over a `120`-dimensional encoding is enough to harvest the easy
fragile-pixel flips and not much more. The per-model pattern is telling — `0.10` on the hardest model
(`Rebuffi`) versus `0.18` on the two softer ones — which is what query starvation looks like: where the
surface is most flattened, DE needs *more* generations to evolve a working 24-pixel support, and with only
six it runs out before it solves the hard samples. DE found the right *tool* (search, not greedy local
gradient) but was given too few queries to express it. So the question for the next rung is not "search
versus gradient" again; it is how to spend the budget far more efficiently — and one route I deliberately
held in reserve back at JSMA is now the right move: go *back* to the gradient, but use it through a
*reconsiderable, boundary-aware* construction instead of a greedy saliency walk.

Let me lay out the alternatives on the "spend the budget more efficiently" axis before I commit, because
"go back to the gradient" is not the only option and I want the elimination on paper. Route one: keep
OnePixel but give it more generations — say sixty instead of six. That would let the self-adapting
difference-mutation actually reach its refinement radius, and it would raise the ASR; but it scales the
query cost roughly tenfold for a search that is *still* undirected, sampling around the population rather
than moving toward the boundary, and it inherits DE's blindness to the local geometry. It buys success by
brute force, not by using information the harness already grants. Route two: a dense-relaxation gradient
attack — `L1`-PGD — that steps on the smooth surrogate and clips. But clipping is exactly the failure I am
about to analyze: the high-magnitude coordinates an `L1` step relies on are the ones that leave the box, so
an end-clip guts the attack. Route three: the boundary-linearization method that keeps the box *inside*
the optimization, uses the gradient to point at the boundary, and relinearizes so no pixel choice is
permanent. Only route three fixes *both* prior failures at once — greedy commitment and query starvation —
without either brute-forcing queries or dying to the clip, so that is the rung I take.

Here is the reasoning. JSMA's gradient use failed because it was greedy and committed; OnePixel's
gradient-free use was query-starved. The synthesis is a white-box method that linearizes the *decision
boundary* (not the classifier values, and not a per-pixel saliency score), takes a *sparse* step toward
it, and *relinearizes* at the new point — so it both uses the gradient efficiently and reconsiders the
geometry every iteration instead of committing once. That is SparseFool, and it is genuinely the next rung.

Let me derive it from the constraint. I want minimal-`L0`, but `L0` is combinatorial and NP-hard, so I
relax to its convex surrogate `L1` — compressed sensing says `L1` recovers sparse solutions under linear
constraints. The bottleneck is then turning the label-change condition into a *linear* constraint. DeepFool
gives the engine: for an affine classifier the closest `L2` boundary point is an orthogonal projection,
and in the `Lp` form the dual exponent is `q = p/(p-1)`, so the `L1` case (`q = infinity`) puts *all* the
correction mass on the single coordinate with the largest `|w_j|` — sparse for free. That is `L1`-DeepFool,
and it is the closest prior idea to what I want. But it has a fatal flaw I have to design around, and it is
the *same* flaw that the budget bookkeeping kept biting earlier: validity. `L1`-DeepFool concentrates large
magnitude on a few coordinates, and those are exactly the coordinates most likely to exceed `[0,1]`;
clipping to the valid range afterward collapses the fooling rate (in the reference setting from near-100%
down to ~13%), because the clip removes the high-magnitude evidence the attack relied on. The box has to
live *inside* the optimization, not after it.

Before the fix, let me be sure I understand *why* the `L1` projection is sparse at all, because that is the
property I am trying to preserve while curing the validity failure, and I want the duality on paper rather
than as a slogan. For an affine constraint `w^T(z - x_B) = 0`, projecting a point onto that hyperplane in
the `Lp` sense minimizes `||z - x||_p` subject to lying on the plane; the Lagrangian optimum puts the
correction mass according to the *dual* norm `q = p/(p-1)`, spreading it in proportion to `|w_j|^{q-1}`
along the coordinates. Run the exponent for the three cases. For `L2` (`p = 2`, `q = 2`) the mass goes as
`|w_j|^{1}` — spread smoothly across all coordinates in proportion to the normal, which is dense, exactly
what DeepFool gives. For an intermediate `p = 1.5` (`q = 3`) the mass goes as `|w_j|^{2}`, already
concentrating on the large-`|w_j|` coordinates. And for `L1` (`p = 1`, `q = infinity`) the exponent runs
to infinity, so the mass collapses entirely onto the single coordinate with the largest `|w_j|`: the
projection is one-hot. That is sparsity for free, falling straight out of the `L1`-`L_inf` duality, and it
is why an `L1`-relaxed boundary projection is the natural sparse-attack primitive. The catch is precisely
that "all the mass on one coordinate" tends to drive that coordinate out of `[0,1]`: if the residual gap
to the plane needs, say, a `+1.4` correction and the whole `1.4` lands on one channel whose clean value is
`0.7`, the target `2.1` is a full `1.1` outside the box, and clipping it back to `1.0` recovers only
`0.3` of the `1.4` needed — the plane is not reached and the attack stalls. So I want to keep the
one-coordinate-at-a-time structure while forcing each coordinate to respect the box. That is the design
target.

One design decision inside this is subtle enough to state on its own: I linearize the *decision boundary*,
not the *classifier outputs*. JSMA linearized the outputs — it read `dF_j/dx_i` and treated each logit as
locally affine — and that is a poor model on a deep net whose outputs bend everywhere. The boundary,
by contrast, is a codimension-one surface, and near a natural image its *mean curvature* is empirically
low even when the individual logits are wildly nonlinear, because the curvatures of the competing logits
partially cancel along the surface where they are equal. So modeling the boundary as one hyperplane is a
much better local approximation than modeling the two logits separately, which is why the normal
`w = grad f_adv - grad f_true` (the gradient of the logit *difference*, i.e. of the surface itself) is the
right object rather than either logit's gradient alone. This is the specific sense in which SparseFool uses
the gradient "better" than JSMA: same first-order information, applied to the geometrically flatter object.

The fix is two moves. First, linearize the *boundary* rather than the classifier: near a natural image the
deep decision boundary has low mean curvature, so I can model it locally as one hyperplane. I find a
boundary point `x_B` by an `L2`-DeepFool step, and the oriented normal is the gradient of the logit
difference there, `w = grad f_adv(x_B) - grad f_true(x_B)`. The relaxed problem becomes
`min ||r||_1` s.t. `w^T(x + r - x_B) = 0` and `l <= x + r <= u`. Second, solve *that* with a
coordinate-greedy projection that *retires* any coordinate as it saturates against the box: pick the
largest remaining `|w_j|`, put the `L1` mass needed to close the residual gap on it, clip the image into
`[l,u]`, and if the chosen coordinate hit a box wall, drop it and solve the residual with the next-best
coordinate. Trace that against the stall above: the `+1.4` gap lands on the top coordinate, clips to `+0.3`
of usable travel, that coordinate retires against the `1.0` wall, and the *remaining* `1.1` of gap moves
to the next-largest `|w_j|`, and so on — so the mass spills onto a *few* coordinates instead of blowing one
out of range, every intermediate point is valid by construction, and sparsity is preserved because the
spill stops as soon as the residual is closed. A single hyperplane is not enough — flatness is only local,
and a sparse step leaves the neighborhood where I estimated the plane — so the outer loop *relinearizes*:
at the new iterate, run `L2`-DeepFool again, rebuild `x_B` and `w`, run the box-aware `L1` solver, and stop
only when the label flips. This relinearization is exactly the reconsideration JSMA lacked: the boundary
geometry is re-estimated every step, so the method tracks the moving boundary instead of committing to a
stale ranking.

Let me dimension-check the normal `w` once, because the whole method rests on it being a well-defined
sparse-projection direction and not a shape mismatch. The images are `(N, 3, 32, 32)`, so a single image
flattens to a `3072`-vector; `grad f_adv(x_B)` and `grad f_true(x_B)` are each gradients of a scalar logit
with respect to that input, hence each `3072`-long, and their difference `w` is `3072`-long — one signed
sensitivity per input feature, exactly the object the hyperplane constraint `w^T(x + r - x_B) = 0` needs.
The `L1` solver then ranks the `3072` entries of `|w|` and pours mass onto the largest first, so the
support it produces is a subset of those `3072` features; collapsed channel-wise, that is at most `24`
distinct spatial pixels when the early-stop fires in time. The shapes line up end to end, and the sparsity
enters precisely at the `argmax_j |w_j|` step — nowhere is there a dense projection hiding. Good; the
construction is coherent before I trust its numbers.

Now the part specific to *this* task, because the harness's configuration is the literal fill and one of
its knobs is the method's only real control. The fill is `SparseFool(model, steps=20, lam=3.0,
overshoot=0.02)`, then `attack(images, labels)`. Read the three numbers. `steps=20` is the outer
relinearization budget — twenty boundary re-estimations. Let me weigh that against OnePixel's six DE
generations honestly, because "far more effective" needs a reason. Each SparseFool step is a *directed
geometric move*: one DeepFool boundary-point computation plus one box-aware `L1` solve, spending on the
order of a handful of gradient and forward evaluations to advance the iterate along the estimated normal.
Six DE generations spent `~5760` forward evaluations to nudge a `120`-dim population; twenty SparseFool
steps spend a comparable order of model queries but each one *moves the iterate toward the boundary on
purpose* rather than sampling around it, so the same query budget travels much further in the direction
that matters. That is why twenty directed relinearizations can outrun six undirected generations even at
similar cost. `overshoot=0.02` is the small final nudge that pushes the accumulated perturbation just
across the actual classifier boundary before returning (separate from the boundary-target control). And
`lam=3.0` is the real parameter: it aims the linear solver's target *past* the estimated boundary,
`x + lam*(x_B - x)`, to absorb the boundary's curvature. The trade-off is exactly the one the budget cares
about — `lam` near 1 aims right at the boundary and keeps the perturbation sparsest but may fail to cross
and need more iterations; larger `lam` pushes farther across, raising the fooling rate and cutting
iterations at the cost of spending more coordinates. Let me sanity-check the extremes: at `lam = 1` the
target sits exactly on the estimated plane, so any curvature that bends the true boundary away leaves the
iterate short and it must relinearize again, burning steps; push `lam` very large and the target overshoots
so far that the `L1` solver spends many coordinates racing past a boundary it has already crossed, blowing
the `24`-pixel budget. `lam = 3.0` sits in between — far enough past a curved boundary to cross reliably in
few steps, not so far that the support explodes — and it is the standard CIFAR-10 setting, deliberately on
the aggressive side, because on *robust* models the curvature is larger and aiming right at the boundary
under-shoots; I want to cross reliably even if it costs a few of the 24 pixels. The harness validates the
`L0` count afterward, so as long as the box-aware solver keeps the support under 24 (which the budget and
the early stop tend to ensure on these images), the aggressive `lam` is the right call. The `pixels`,
`device`, and `n_classes` arguments are unused — SparseFool's own `num_classes` default and `lam` govern
the behavior, and the harness handles validity.

It is worth reasoning about the relinearization budget as a race against curvature, because that is what
`steps=20` is really buying and it tells me when the method runs out. Each relinearization corrects for the
fact that the previous hyperplane estimate was only locally valid; if the boundary were perfectly flat, one
step would suffice and the other nineteen would be idle. The more curved the boundary, the shorter the
neighborhood over which each linear estimate holds, so the more steps I burn creeping toward a boundary
that keeps bending away. Put a rough number on it: if a flat boundary needs one step and a mildly curved
one needs three or four to converge, then twenty steps comfortably covers moderate curvature — but on a
model where the boundary bends sharply in the attack-relevant directions, each step advances only a little
before the linear model is stale again, and twenty steps may still leave the iterate short of a crossing.
That is the exact regime where `lam=3.0`'s overshoot backfires: aiming `3x` past a *wrong* boundary
estimate sends the `L1` solver spending pixels in a direction that curves away, so the support grows
without the label flipping. So the method has a built-in failure mode that is *sharpest precisely where
the boundary is least linear* — and since I cannot know in advance which of the three models has the most
curved boundary, I should expect exactly the kind of non-uniform result where SparseFool wins big on the
near-linear models and possibly stalls on the curved one.

Where does that leave my expectations against the OnePixel ceiling of `0.153`? SparseFool addresses
OnePixel's exact weakness — query efficiency — by replacing undirected population sampling with a directed,
boundary-following geometric search that reconsiders every step, *and* it addresses JSMA's exact weakness —
greedy commitment — by relinearizing. So on both axes it is strictly better-motivated than either prior
rung. I expect it to clear `0.153`. But I am cautious about *how much*, and the caution is specific to
robust models. SparseFool's whole engine is local-linear: it assumes the boundary is locally a hyperplane
of low curvature. Adversarial `L2` training does two things that fight this — it flattens the loss surface
(weak gradients, so the DeepFool step and the normal `w` are small and noisy) and it can make the boundary
*more* curved in the directions an attack would exploit, so the one-hyperplane model is a worse fit
exactly where it matters. When the local-linear assumption is poor, even twenty relinearizations chase a
boundary the linear model keeps mis-estimating, and `lam=3.0` overshooting a mis-estimated boundary can
spend coordinates without crossing. Here is the crucial difference from every prior rung: because
SparseFool's success is gated by *boundary linearity*, and linearity varies from model to model
independently of overall robustness, I should expect its per-model numbers to *spread*, not converge the
way DE's did. A model whose boundary happens to be well-approximated by a hyperplane should jump well past
`0.153`; a model whose boundary is highly curved could actually *trail* OnePixel even while the mean rises.
My falsifiable expectation: SparseFool's mean ASR beats OnePixel's `0.153`, but not by a landslide —
somewhere in the high-teens — and it may even *trail* OnePixel on the single model whose boundary is least
linear, while leading clearly on the others, producing a *wider* per-model spread than OnePixel's tight
`0.10/0.18/0.18`. That mixed, modest gain would be the diagnosis for the final rung: every gradient-based
and population method so far is limited by either local commitment, query starvation, or a brittle
local-linear boundary model, and the strongest sparse attack is the one designed natively for the discrete
`L0` set — a random search whose proposal distribution is built specifically to sample *feasible supports*,
with a query schedule that explores then refines, so it neither commits early nor wastes queries nor leans
on a linear-boundary fiction.

The delta from the previous rung, concretely: where OnePixel returned `attack(images, labels)` from a
six-generation black-box population search and scored `0.153`, this rung returns `attack(images, labels)`
from a `SparseFool` instance with `steps=20`, `lam=3.0` — trading undirected, query-starved population
sampling for a directed, reconsiderable, boundary-linearizing white-box search that keeps the box inside
the optimization. The full scaffold module is in the answer. I expect a real but modest gain over `0.153`,
limited by how well a local-linear boundary model survives on adversarially-trained surfaces — the
limitation the final rung is built to escape.
