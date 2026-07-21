The attack is the whole point, but it runs inside a fixed oracle, and the crudest thing I can do with
that oracle is the floor I have to start from — turning forward-only score queries into a
misclassification at all, with no cleverness about *where* or *how big* the moves are. I am locked out
of the gradient: I send the oracle an image and read back logits, nothing else. The wrapper runs under
`torch.no_grad` and hands back only logits, so there is no backward path to steal and no differentiable
probability to climb. There is exactly one lever — propose a perturbed image, query, and decide whether
to keep it — and the floor uses that lever in the most naive way available.

Almost every judgement below is arithmetic on a few constants, so pin them first. A CIFAR image is
`3 x 32 x 32`, so the search space is `D = C*H*W = 3072` real coordinates, each confined to a width-`2eps`
interval around its clean value and to `[0,1]`. The radius is `eps = 8/255 = 0.03137`. The per-sample
budget is `n_queries = 1000`, evaluated at batch size 50 over 200 correctly-classified images — four
batches per `(model, dataset)` scenario, six scenarios, seed 42. Those numbers already determine most of
what the floor will and will not do.

For a correctly classified `(x, y)` I want `argmax_k f_k(x_adv) != y` with `||x_adv - x||_inf <= eps` and
`x_adv in [0,1]`. The cheapest scalar that tracks progress, using only the logits I already pay for, is
the **correct-class score** `f_y(x_adv)`: it starts high and the prediction flips once another class
overtakes it. But `f_y` is a lossy proxy for what actually matters. Say the clean logits are
`f = (2.0, 0.5, -1.0)` with `y = 0`. If a perturbation drives `f_0` down to `1.8` while `f_1` rises to
`1.9`, the argmax is now class 1 — a success — yet `f_y` fell only `0.2`, while the *event that mattered*
was `f_1` crossing above it. The true margin `J = f_0 - max_{k!=0} f_k` went `1.5 -> -0.1`, crossing zero
exactly at the flip; `f_y` alone can keep sliding while no competitor is catching up, and can miss a flip
driven purely by a rising competitor. The floor takes `f_y` anyway, because it is the single cheapest
number: one `gather`, no `max` over the other `K-1` classes (99 of them on CIFAR-100). Cheap and crude —
exactly the floor's brief. So the floor minimizes `f_y` by greedy accept-if-better: keep a candidate only
when it lowers the correct-class logit.

Greedy accept-if-better is pure hill-climbing — it never accepts a worse score, so it cannot escape a
local basin. The textbook escape is simulated annealing, but that needs a temperature schedule I would
have to tune per scenario with no gradient and no validation budget, and every accepted-worse move burns
one of my scarce queries moving backwards. For a tiny budget the expected value of escaping a basin is
small and the cost is a real query, so greedy is the correct floor policy: it wastes nothing on
exploration, and its only sin is a badly aimed proposal. So I keep greedy and put all the blame on the
proposal distribution.

The proposal draws uniform noise in a small box, `uniform(-step, step)` with `step = eps/2`, adds it to
the current best, then re-projects: `clamp(x + clamp(cand - x, -eps, eps), 0, 1)`. With `step = eps/2 =
0.01569`, a proposed component is uniform on `[-0.01569, 0.01569]`, expected magnitude `eps/4 = 0.00784`
— roughly a quarter of the radius the box allows. Successful `L_inf` perturbations want each touched
component pushed to `+/- eps`; the floor lands uniformly *inside* a half-interval, spending on average
about a quarter of its per-component allowance. A factor of roughly four of leverage left on the table at
the component level before I have even asked whether the *direction* is any good.

And the direction is the deeper problem. The proposal is isotropic over all `D = 3072` coordinates: no
notion that the model is convolutional, no notion that useful `L_inf` perturbations sit at corners, no
step decay, no concentration. In `3072` dimensions a uniformly random direction is almost orthogonal to
any particular target: for a random unit vector, `E|cos| ~ sqrt(2 / (pi * D)) = sqrt(2 / (pi * 3072)) =
0.0144`. So each blind proposal is on average about **1.4% aligned** with whatever direction actually
helps. The accept rule keeps a proposal roughly half the time — by symmetry half of all directions lower
the score — but the *size* of each accepted improvement carries that same `~1.4%` factor. The floor is a
downhill random walk with a genuine but minuscule drift per step.

The reprojection has a subtlety worth naming because it is the floor's only, accidental, push toward the
boundary. On the first step `adv = x`, so `cand - x = noise` is entirely inside `[-eps, eps]` and the
inner clamp does nothing — the move is interior. But `adv` accumulates: after a few accepts a component's
running delta can sit near `+/- eps`, and then `cand - x = (adv - x) + noise` exceeds `eps`, so the inner
clamp pins that component to the boundary. The only thing nudging components onto corners is this clamp
saturating after several lucky accepts in the same direction — weak, indirect pressure, nothing like
deliberately sitting at `+/- eps`. The outer `clamp(., 0, 1)` cuts the other way: for a clean pixel near
`0` or `1`, part of its `2eps` interval falls outside `[0,1]`, so those components have even less than the
nominal budget.

Chasing the accumulation dynamics a step further corrects a lazy conclusion — that the floor's core defect
is its *interior* moves. Over `64` steps at a `~50%` accept rate that is about `32` accepted updates, each
adding a fresh `uniform(-step, step)` vector. On a single coordinate the accumulated delta is a sum of
`~32` such draws, standard deviation `sqrt(32) * step / sqrt(3) = 0.0513`, which already *exceeds*
`eps = 0.0314`, so the inner clamp saturates most coordinates. Given enough accepts the floor *does* fill
in the `eps` radius. What it can never fix is the *sign pattern*: those saturated coordinates carry the
random signs of the accepted noise, so the perturbation converges toward a random-signed point at radius
`eps`, aligned with the useful direction at only the `~1.4%` level. The magnitude takes care of itself;
the *direction* does not, and no amount of extra steps repairs a direction drawn blind.

The budget bookkeeping is where the floor quietly throws away most of its allowance, and it decides the
number. The oracle exhausts the *whole batch* to failure the instant the running query count crosses
`batch_size * n_queries = 50000` — and when it flips it returns *zeros* for the rest of the batch and
scores the whole batch as a failure. So overshooting is not a soft cap; it converts still-working samples
into hard misses. The floor does not walk near that line: it runs a fixed `n_steps = min(n_queries, 64) =
64` iterations, each a `model(cand)` call costing 50 queries, plus one initial `model(adv)` call to seed
`best` — `65` model calls, `3250` queries against a `50000` ceiling. Per sample the reported cost is `65`
out of `1000`: the floor leaves `93.5%` on the table. The tiny cap buys absolute safety from the
batch-to-zero catastrophe, but at the price of most of the search — the right answer is to spend near the
full budget with a margin, not to hide at `6.5%`. That flat `avg_queries = 65` should hold on every row
regardless of difficulty, because the cap is a constant.

One piece of vectorized care the floor does get right, and every later rung inherits it: the accept rule
operates per sample. I query the candidate's correct-class score for the whole batch at once, form
`improve = cand_score < best`, update only the rows where it holds via a masked `torch.where`, and carry
`best` forward so I never re-query an image whose candidate lost. A single batched forward pass advances
every still-improving sample simultaneously; the 50 queries are shared across the batch, but the
keep/discard decision is independent per image. The strict-inequality accept (`cand_score < best`) makes
the incumbent's score monotone non-increasing across the `64` steps: the search can never regress, so a
sample that ends unflipped ended unflipped because `64` blind proposals never found a good enough downhill
direction, not because bookkeeping threw away a candidate. There is no query-noise complication here — the
models are undefended deterministic classifiers, so a drop in `f_y` is a real drop, not sampling jitter.

Seeing *what the floor declines to do* sets up the next rung. Four levers sit unbought. The economic one:
raise `n_steps` from `64` toward `1000` — more accepted moves, but each still `~1.4%` aligned and the
magnitude already saturates after `~32` accepts, so spending the budget is *necessary* but not
*sufficient*. The per-component one: move by `+/- eps` so the projection pins every touched component to a
corner from the first step, instead of the timid `eps/2` interior draw. The objective one: descend the
true margin `J = f_y - max_{k!=y} f_k` so a rising competitor counts as progress and the zero-crossing
coincides with the flip. And the dimensional one, most powerful: stop scattering across all `3072`
coordinates and concentrate the change into a `k`-coordinate block, raising alignment from `1/sqrt(3072)`
toward `1/sqrt(k)` — a 100-coordinate block sits near `0.1`, a `7x` better-aimed move, *if* I knew where
to spend it. The floor buys none of these.

That last framing is the cleanest statement of the disease. In the `D = 1` limit a random direction is
perfectly aligned (`cos = 1`) and greedy random search solves the problem by line search; the entire
penalty the floor pays is the `1/sqrt(D)` orthogonality of high dimension, `sqrt(3072) = 55.4`. A blind
proposal in this space is about `55x` worse-aimed than the same greedy rule on a one-dimensional problem,
and every rung after this is a contest over how much of that `55x` it can claw back by imposing structure
on the proposal. The feasibility clamps do their job — the inner guarantees `||adv - x||_inf <= eps`, the
outer guarantees `[0,1]` — so the harness's return-side re-check passes and every miss the floor records
is an honest failure to flip the model, not a validity artifact.

Now what the floor must do across the six scenarios. Greedy random search with isotropic interior noise in
`~3000` dimensions is the classic high-dimensional failure, and `64` timid steps barely move the
perturbation off the clean image. Each scenario is decided by how far the flip-inducing boundary sits
relative to how far a walk of `~32` accepted `~1.4%`-aligned moves can travel. On the easier pairs — close
boundary — the floor should flip a fair fraction; on a more robust backbone whose boundary sits farther
out, the same moves mostly fail and `asr` sags well below half.

Which way CIFAR-100 cuts is worth thinking about, because two intuitions point opposite ways. CIFAR-100
gives the correct class `99` competitors instead of `9`, so *any* of them crossing `f_y` is more likely —
a closer nearest boundary, which should *help* an untargeted attack. Against that, a 100-way model spreads
confidence over a more crowded logit geometry, which could make the crude `f_y` proxy noisier. For the
floor specifically — whose whole game is to get *some* competitor above `f_y` — the "more targets" effect
should dominate, so I would not be surprised if CIFAR-100 rows come out slightly *stronger* than their
CIFAR-10 siblings, not weaker. The axis I expect to actually separate the floor is *architecture*:
whichever backbone presents the farthest, cleanest boundary is where `~1.4%`-aligned noise collapses
hardest. A wide spread across scenarios and a modest mean, with `avg_queries` flat near `65` everywhere.

So the floor has two separable diseases. The *economic* one — it caps itself at `65` queries when `1000`
are available; curing it alone is easy but insufficient, since `15x` unaligned moves is still unaligned.
The *directional* one — unstructured interior nudges are `~1.4%`-aligned in `3072` dimensions — is the
real one. The next rung has to do both at once: stop leaving the budget unused, and, using the very
forward queries the floor throws away, reconstruct an actual descent direction instead of guessing one.
That is the finite-difference gradient-estimation primitive the background lists but the floor refuses to
touch. The distilled floor module is in the answer.
