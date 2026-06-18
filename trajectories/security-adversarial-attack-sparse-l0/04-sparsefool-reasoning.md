OnePixel landed where I expected and confirmed the diagnosis. `asr = 0.10` on `Rebuffi-R18-L2`, `0.18` on
`Augustin-L2`, `0.18` on `Engstrom-L2`, mean about `0.153`. So the black-box population search more than
tripled JSMA's `0.047` — a clear, real gain — exactly because it escapes the local optima that trapped
greedy saliency. But the absolute level, mid-teens of percent, is the ceiling I warned about: six
generations of differential evolution over a `120`-dimensional encoding is enough to harvest the easy
fragile-pixel flips and not much more. The per-model pattern is telling — `0.10` on the hardest model
(`Rebuffi`) versus `0.18` on the two softer ones — which is what query starvation looks like: where the
surface is most flattened, DE needs *more* generations to evolve a working 24-pixel support, and with only
six it runs out before it solves the hard samples. DE found the right *tool* (search, not greedy local
gradient) but was given too few queries to express it. So the question for the next rung is not "search
versus gradient" again; it is how to spend the budget far more efficiently — and one route I deliberately
held in reserve back at JSMA is now the right move: go *back* to the gradient, but use it through a
*reconsiderable, boundary-aware* construction instead of a greedy saliency walk.

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
property I am trying to preserve while curing the validity failure. For an affine constraint
`w^T(z - x_B) = 0`, projecting a point onto that hyperplane in the `Lp` sense puts the correction mass
according to the dual norm `q = p/(p-1)`. For `L2` (`q=2`) the mass spreads smoothly across all
coordinates — dense, which is what DeepFool gives. For `L1` (`q=infinity`) the optimizer puts *all* the
mass on the single coordinate with the largest `|w_j|`: the projection is one-hot. That is sparsity for
free, falling straight out of the `L1`-`L_inf` duality, and it is why an `L1`-relaxed boundary projection
is the natural sparse-attack primitive. The catch is precisely that "all the mass on one coordinate" tends
to drive that coordinate out of `[0,1]`; so I want to keep the one-coordinate-at-a-time structure while
forcing each coordinate to respect the box. That is the design target.

The fix is two moves. First, linearize the *boundary* rather than the classifier: near a natural image the
deep decision boundary has low mean curvature, so I can model it locally as one hyperplane. I find a
boundary point `x_B` by an `L2`-DeepFool step, and the oriented normal is the gradient of the logit
difference there, `w = grad f_adv(x_B) - grad f_true(x_B)`. The relaxed problem becomes
`min ||r||_1` s.t. `w^T(x + r - x_B) = 0` and `l <= x + r <= u`. Second, solve *that* with a
coordinate-greedy projection that *retires* any coordinate as it saturates against the box: pick the
largest remaining `|w_j|`, put the `L1` mass needed to close the residual gap on it, clip the image into
`[l,u]`, and if the chosen coordinate hit a box wall, drop it and solve the residual with the next-best
coordinate. Every intermediate point is valid by construction, and sparsity is preserved because mass
still concentrates on few coordinates. A single hyperplane is not enough — flatness is only local, and a
sparse step leaves the neighborhood where I estimated the plane — so the outer loop *relinearizes*: at the
new iterate, run `L2`-DeepFool again, rebuild `x_B` and `w`, run the box-aware `L1` solver, and stop only
when the label flips. This relinearization is exactly the reconsideration JSMA lacked: the boundary
geometry is re-estimated every step, so the method tracks the moving boundary instead of committing to a
stale ranking.

Now the part specific to *this* task, because the harness's configuration is the literal fill and one of
its knobs is the method's only real control. The fill is `SparseFool(model, steps=20, lam=3.0,
overshoot=0.02)`, then `attack(images, labels)`. Read the three numbers. `steps=20` is the outer
relinearization budget — twenty boundary re-estimations, far more *effective* iterations than OnePixel's
six DE generations bought, because each step is a directed geometric move rather than a population sample,
so the same order of model queries goes much further. `overshoot=0.02` is the small final nudge that
pushes the accumulated perturbation just across the actual classifier boundary before returning (separate
from the boundary-target control). And `lam=3.0` is the real parameter: it aims the linear solver's target
*past* the estimated boundary, `x + lam*(x_B - x)`, to absorb the boundary's curvature. The trade-off is
exactly the one the budget cares about — `lam` near 1 aims right at the boundary and keeps the perturbation
sparsest but may fail to cross and need more iterations; larger `lam` pushes farther across, raising the
fooling rate and cutting iterations at the cost of spending more coordinates. `lam = 3.0` is the standard
CIFAR-10 setting, deliberately on the aggressive side, because on *robust* models the curvature is larger
and aiming right at the boundary under-shoots — I want to cross reliably even if it costs a few of the 24
pixels. The harness validates the `L0` count afterward, so as long as the box-aware solver keeps the
support under 24 (which the budget and the early stop tend to ensure on these images), the aggressive `lam`
is the right call. The `pixels`, `device`, and `n_classes` arguments are unused — SparseFool's own
`num_classes` default and `lam` govern the behavior, and the harness handles validity.

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
spend coordinates without crossing. My falsifiable expectation: SparseFool's mean ASR beats OnePixel's
`0.153`, but not by a landslide — somewhere in the high-teens — and it may even *trail* OnePixel on the
single model whose boundary is least linear, while leading clearly on the others. That mixed, modest gain
would be the diagnosis for the final rung: every gradient-based and population method so far is limited by
either local commitment, query starvation, or a brittle local-linear boundary model, and the strongest
sparse attack is the one designed natively for the discrete `L0` set — a random search whose proposal
distribution is built specifically to sample *feasible supports*, with a query schedule that explores then
refines, so it neither commits early nor wastes queries nor leans on a linear-boundary fiction.

The delta from the previous rung, concretely: where OnePixel returned `attack(images, labels)` from a
six-generation black-box population search and scored `0.153`, this rung returns `attack(images, labels)`
from a `SparseFool` instance with `steps=20`, `lam=3.0` — trading undirected, query-starved population
sampling for a directed, reconsiderable, boundary-linearizing white-box search that keeps the box inside
the optimization. The full scaffold module is in the answer. I expect a real but modest gain over `0.153`,
limited by how well a local-linear boundary model survives on adversarially-trained surfaces — the
limitation the final rung is built to escape.
