OnePixel landed where I expected. `asr = 0.10` on `Rebuffi-R18-L2`, `0.18` on `Augustin-L2`, `0.18` on
`Engstrom-L2`, mean `~0.153` — as counts `15`, `27`, `27` flips. The black-box population search more than
tripled JSMA's `0.047` (mean `3.3x`), exactly because it escapes the local optima that trapped greedy
saliency. The per-model multipliers: Rebuffi `4 -> 15` (`3.75x`), Augustin `7 -> 27` (`3.85x`), Engstrom
`10 -> 27` (`2.7x`), so the two models JSMA found softer and harder converged — Augustin and Engstrom tie at
`27` — while Rebuffi stays distinctly hardest at `15`. That convergence is the signature of a *global*
evaluative search: DE reads the whole-image objective, so the per-model differences a first-order signal
exposed matter less, and what is left is a cruder ranking by overall robustness on which Rebuffi is toughest.
But the absolute level, mid-teens, is the ceiling I warned about: six generations of DE over a `120`-dim
encoding harvests the easy fragile-pixel flips and not much more, and `0.10` on the hardest model versus
`0.18` on the two softer is what query starvation looks like — where the surface is most flattened, DE needs
more generations to evolve a working support and runs out. DE found the right *tool* but too few queries. So
the question is not "search versus gradient" again; it is how to spend the budget far more efficiently — and
the route I held in reserve at JSMA is now right: go *back* to the gradient, through a *reconsiderable,
boundary-aware* construction.

The alternatives on that axis, so the elimination is on paper. Give OnePixel sixty generations instead of
six: that reaches the refinement radius and raises ASR, but scales queries `~10x` for a search that is
*still* undirected and blind to local geometry — success by brute force, not by using information the harness
grants. A dense-relaxation `L1`-PGD steps on the smooth surrogate and clips — but clipping is exactly the
failure I am about to analyze. The boundary-linearization method keeps the box *inside* the optimization,
uses the gradient to point at the boundary, and relinearizes so no pixel choice is permanent — it fixes
*both* prior failures at once (greedy commitment and query starvation) without brute-forcing queries or dying
to the clip. That is SparseFool.

The derivation from the constraint: minimal-`L0` is combinatorial and NP-hard, so relax to its convex
surrogate `L1` — compressed sensing recovers sparse solutions under linear constraints. The bottleneck is
turning the label-change condition into a *linear* constraint, and DeepFool gives the engine: for an affine
classifier the closest boundary point is an orthogonal projection, and in the `Lp` form the dual exponent
`q = p/(p-1)` governs how the correction mass spreads as `|w_j|^{q-1}`. For `L2` (`q=2`) the mass spreads
smoothly across all coordinates — dense. For `L1` (`q=infinity`) the exponent runs to infinity and the mass
collapses entirely onto the single largest-`|w_j|` coordinate: the projection is one-hot, sparsity for free
straight out of the `L1`-`L_inf` duality. That is `L1`-DeepFool, the closest prior idea to what I want. But
it has a fatal validity flaw, the same one the budget bookkeeping kept biting: concentrating large magnitude
on a few coordinates drives exactly those coordinates out of `[0,1]`, and clipping back afterward collapses
the fooling rate (from near-total down toward the low teens of percent), because the clip removes the
high-magnitude evidence the attack relied on. If the residual gap needs a `+1.4` correction landing entirely
on one channel whose clean value is `0.7`, the target `2.1` clips to `1.0`, recovering only `0.3` of the
`1.4` needed — the plane is not reached and the attack stalls. The box has to live *inside* the optimization,
not after it.

One design decision is subtle enough to state: I linearize the *decision boundary*, not the *classifier
outputs*. JSMA linearized the outputs — read `dF_j/dx_i`, treated each logit as locally affine — a poor model
on a deep net whose outputs bend everywhere. The boundary is a codimension-one surface whose *mean curvature*
near a natural image is empirically low even when the individual logits are wildly nonlinear, because the
competing logits' curvatures partially cancel where they are equal. So one hyperplane is a much better local
fit than modeling the two logits separately, which is why the normal `w = grad f_adv - grad f_true` (the
gradient of the logit *difference*, i.e. of the surface itself) is the right object rather than either
logit's gradient alone. This is the specific sense in which SparseFool uses the gradient better than JSMA:
same first-order information, applied to the geometrically flatter object.

The fix is two moves. First, linearize the boundary: find a boundary point `x_B` by an `L2`-DeepFool step,
take the oriented normal `w = grad f_adv(x_B) - grad f_true(x_B)`, and pose `min ||r||_1` s.t.
`w^T(x + r - x_B) = 0` and `l <= x + r <= u`. Second, solve *that* with a coordinate-greedy projection that
*retires* any coordinate as it saturates against the box: pick the largest remaining `|w_j|`, put the `L1`
mass needed to close the residual gap on it, clip the image into `[l,u]`, and if the coordinate hit a wall
drop it and solve the residual with the next-best coordinate. Against the stall above: the `+1.4` gap lands
on the top coordinate, clips to `+0.3` of usable travel, that coordinate retires against the wall, the
remaining `1.1` moves to the next-largest `|w_j|`, and so on — the mass spills onto a *few* coordinates
instead of blowing one out of range, every intermediate point is valid by construction, and sparsity is
preserved because the spill stops as soon as the residual is closed. A single hyperplane is not enough —
flatness is only local and a sparse step leaves the neighborhood — so the outer loop *relinearizes*: at the
new iterate, rerun `L2`-DeepFool, rebuild `x_B` and `w`, rerun the box-aware `L1` solver, stop when the label
flips. That relinearization is the reconsideration JSMA lacked: the geometry is re-estimated every step,
tracking the moving boundary instead of committing to a stale ranking. (The normal is well-defined
dimensionally: each image flattens to `3072`, `grad f_adv` and `grad f_true` are each `3072`-long, their
difference `w` is `3072`-long — one signed sensitivity per feature, exactly what `w^T(x + r - x_B) = 0`
needs, and sparsity enters at the `argmax_j |w_j|` step, nowhere a dense projection.)

Now the task config, one knob of which is the method's only real control. The fill is
`SparseFool(model, steps=20, lam=3.0, overshoot=0.02)`. `steps=20` is the outer relinearization budget —
twenty boundary re-estimations. Weighing that against OnePixel's six DE generations: each SparseFool step is
a *directed geometric move* (one DeepFool boundary-point computation plus one box-aware `L1` solve, a handful
of gradient and forward evaluations) that advances the iterate along the estimated normal, where six DE
generations spent `~5760` forward evaluations to nudge a `120`-dim population; comparable order of model
queries, but each SparseFool step *moves the iterate toward the boundary on purpose* rather than sampling
around it, so the same budget travels much further in the direction that matters. `overshoot=0.02` is the
small final nudge that pushes the accumulated perturbation just across the actual boundary before returning.
And `lam=3.0` is the real parameter: it aims the linear solver's target *past* the estimated boundary,
`x + lam*(x_B - x)`, to absorb its curvature. The trade-off is exactly the budget's concern — `lam` near 1
aims right at the boundary and keeps the perturbation sparsest but may fail to cross and burn steps
relinearizing; larger `lam` crosses reliably in fewer steps but spends more coordinates racing past a
boundary already crossed, risking the 24-pixel budget. `lam=3.0` sits between — far enough past a curved
boundary to cross in few steps, not so far the support explodes — the standard CIFAR-10 setting, deliberately
aggressive because on *robust* models the curvature is larger and aiming right at the boundary undershoots; I
want to cross even if it costs a few of the 24 pixels, and the harness validates the `L0` count afterward so
the aggressive `lam` is safe as long as the box-aware solver keeps the support under 24. `steps=20` is really
a race against curvature: a flat boundary needs one step, a mildly curved one three or four, so twenty covers
moderate curvature — but where the boundary bends sharply in the attack-relevant directions, each step goes
stale fast and twenty may still fall short, and there `lam=3.0` backfires, aiming `3x` past a *wrong* estimate
and spending pixels in a direction that curves away.

So expectations against OnePixel's `0.153`. SparseFool addresses OnePixel's exact weakness (query efficiency,
via a directed boundary-following search) *and* JSMA's exact weakness (greedy commitment, via
relinearization), so on both axes it is better-motivated than either prior rung and should clear `0.153`. But
I am cautious about *how much*, and the caution is specific to robust models: SparseFool's whole engine is
local-linear, assuming the boundary is locally a low-curvature hyperplane, and `L2` training fights this both
ways — it flattens the loss surface (weak, noisy gradients, so `w` and the DeepFool step are small) and it
can make the boundary *more* curved in the directions an attack exploits, so the one-hyperplane model is a
worse fit exactly where it matters. Here is the crucial difference from every prior rung: because
SparseFool's success is gated by *boundary linearity*, and linearity varies model to model independently of
overall robustness, I expect its per-model numbers to *spread*, not converge the way DE's did — a
well-approximated boundary jumps well past `0.153`, a highly-curved one could actually *trail* OnePixel even
while the mean rises. My falsifiable expectation: mean ASR beats `0.153` but not by a landslide, a real but
modest gain, with a *wider* per-model spread than OnePixel's tight `0.10/0.18/0.18` and possibly one model
trailing OnePixel outright. That mixed, modest gain would be the diagnosis for the final rung: every
gradient-based and population method so far is limited by local commitment, query starvation, or a brittle
local-linear boundary model, and the strongest sparse attack is the one designed natively for the discrete
`L0` set — a random search whose proposal distribution samples *feasible supports* directly, with a schedule
that explores then refines.
