The attack is the whole point, but it runs inside a fixed oracle, and the crudest thing I can do with
that oracle is the floor I have to start from — so the pain to begin with is just turning forward-only
score queries into a misclassification at all, with no cleverness about *where* or *how big* the moves
are. I am locked out of the gradient: I send the oracle an image and read back logits, nothing else.
The wrapper runs under `torch.no_grad` and hands back only logits, so there is no backward path to steal
and no differentiable probability to climb. There is exactly one lever — propose a perturbed image,
query, and decide whether to keep it — and the floor uses that lever in the most naive way available.

Let me pin the constants first, because almost every judgement below is really arithmetic on them. A
CIFAR image is `3 x 32 x 32`, so the search space is `D = C*H*W = 3072` real coordinates, each confined
to a width-`2eps` interval around its clean value and additionally to `[0,1]`. The radius is
`eps = 8/255 = 0.03137`. The per-sample budget is `n_queries = 1000`, evaluated at batch size 50 over
200 correctly-classified test images — four batches per `(model, dataset)` scenario, six scenarios,
seed 42. Those few numbers — `D = 3072`, `eps ~ 0.031`, budget 1000, batch 50 — already determine most
of what the floor will and will not do before I run a single query.

Let me write down the objective so I do not fool myself. For a correctly classified `(x, y)` I want
`argmax_k f_k(x_adv) != y` with `||x_adv - x||_inf <= eps` and `x_adv in [0,1]`. The cheapest scalar
that tracks progress toward that, using only the logits I already pay for, is the **correct-class
score** `f_y(x_adv)`: it starts high (the model is confident in `y`) and the prediction flips once some
other class overtakes it. Let me trace a three-class example to be sure it is a usable objective and to
see exactly where it is lossy. Say the clean logits are `f = (2.0, 0.5, -1.0)` with `y = 0`, so
`f_y = 2.0` and the runner-up is class 1 at `0.5`. The prediction flips the instant some `f_k` for
`k != 0` overtakes `2.0`. If a perturbation drives `f_0` down to `1.8` while `f_1` rises to `1.9`, the
argmax is now class 1 — a success — and `f_y` did indeed drop from `2.0` to `1.8`, so hill-climbing on
`f_y` was pulling in the right direction. But notice the looseness: `f_y` fell only `0.2` while the
*event that mattered* was `f_1` crossing above it. The true margin `J = f_0 - max_{k!=0} f_k` went
`2.0 - 0.5 = 1.5 -> 1.8 - 1.9 = -0.1`, crossing zero *exactly* at the flip; `f_y` alone can keep
sliding down while no competitor is actually catching up, and can miss a flip driven purely by a rising
competitor. So `f_y` is a lossy proxy for the margin. The floor takes it anyway, because it is the
single cheapest number available: one `gather` on the logits, no `max` over the other `K-1` classes
(which on CIFAR-100 is a max over 99 entries). Cheap and crude — exactly the floor's brief. So the
floor minimizes `f_y` by greedy accept-if-better: keep a candidate only when it lowers the correct-class
logit.

The accept policy itself is a decision, even at the floor, and it is worth pricing against the obvious
alternative before I move on. Greedy accept-if-better is pure hill-climbing: it never accepts a worse
score, so it cannot escape a local basin. The textbook escape hatch is to sometimes accept a worse
candidate — simulated annealing, with an acceptance probability that decays over the run. But that costs
me two things I cannot spare here. It needs a temperature schedule I would have to tune per scenario with
no gradient and no validation budget, and every accepted-worse move burns one of my scarce `~65` queries
moving *backwards*, breaking the monotonicity that makes the floor's failures interpretable. For a
`64`-query budget the expected value of escaping a basin is tiny and the cost is a real query, so greedy
is the correct floor policy: it wastes nothing on exploration and its only sin is the one I actually care
about, a badly aimed proposal. So I keep greedy and put all the blame where it belongs — on the proposal
distribution.

Now the only design questions left are the proposal and the budget bookkeeping, and the floor answers
both as bluntly as possible. The proposal draws uniform noise in a small box, `uniform(-step, step)`
with `step = eps/2`, adds it to the current best, then re-projects into the feasible set:
`clamp(x + clamp(cand - x, -eps, eps), 0, 1)`. Put numbers on it. `step = eps/2 = 0.01569`, so a single
proposed component is uniform on `[-0.01569, 0.01569]`, with expected magnitude `step/2 = eps/4 =
0.00784` — roughly a quarter of the radius the box actually allows. Successful `L_inf` perturbations
want each touched component pushed as far as the box permits, to `+/- eps`; the floor, using `eps/2` and
landing uniformly *inside* that half-interval, spends on average only about a quarter of its per-component
allowance. That is a factor of roughly four of leverage left on the table at the component level before
I have even asked whether the *direction* is any good.

And the direction is the deeper problem. The proposal is isotropic in all `D = 3072` coordinates at
once: no notion that the model is convolutional, no notion that useful `L_inf` perturbations sit at
corners of the box, no decay of the step size, no concentration of the change into any region. It
scatters tiny nudges across every coordinate simultaneously. In `3072` dimensions a uniformly random
direction is almost orthogonal to any particular target direction — say the direction that would most
efficiently lower `f_y`. For a random unit vector the expected absolute cosine with a fixed direction is
`E|cos| ~ sqrt(2 / (pi * D)) = sqrt(2 / (pi * 3072)) = sqrt(2.07e-4) = 0.0144`. So each blind proposal
is on average about **1.4% aligned** with whatever direction actually helps. Against a locally-linear
objective the accept rule keeps a proposal roughly half the time — by symmetry, half of all directions
have a negative inner product with the local gradient, so about `50%` of draws lower the score. But the
*size* of each accepted improvement is proportional to `step * |grad| * E[|cos| | cos < 0]`, i.e. it
carries that same `~1.4%` alignment factor. So the floor is a downhill random walk with a genuine but
minuscule drift per step: half the moves are kept, and each kept move makes about a percent-scale
fraction of the progress a well-aimed move of the same size would.

The reprojection has a subtlety worth naming because it is the floor's only, accidental, push toward the
boundary. On the very first step `adv = x`, so `cand - x = noise in [-step, step]`, entirely inside the
`[-eps, eps]` box, and the inner clamp does nothing — the move is purely interior. But `adv` accumulates:
after a few accepts a component's running delta can already sit near `+/- eps`, and then `cand - x =
(adv - x) + noise` can exceed `eps`, so the inner clamp *does* bite and pins that component to the
boundary. So the only thing nudging components onto corners is this clamp saturating after several lucky
accepts in the same direction — a weak, indirect pressure, nothing like deliberately sitting at `+/- eps`.
The outer `clamp(., 0, 1)` cuts the other way: for a clean pixel near `0` or `1`, part of its `2eps`
interval falls outside `[0,1]` and is unusable, so those components have even less than the nominal budget
to give. Net, the floor lives in the interior of the box almost the whole time.

It is worth chasing the accumulation dynamics a step further, because it corrects a lazy conclusion I was
about to draw — that the floor's core defect is its *interior* moves. Take the roughly `50%` accept rate
seriously: over `64` steps that is about `32` accepted updates, each adding a fresh `uniform(-step, step)`
vector to the running perturbation. On a single coordinate the accumulated delta is then a sum of `~32`
independent `uniform(-step, step)` draws (over accepted steps), whose standard deviation is
`sqrt(32) * step / sqrt(3) = 5.66 * 0.01569 / 1.732 = 0.0513`. That already *exceeds* `eps = 0.0314`, so
the inner clamp saturates most coordinates: after a few dozen accepts the floor's perturbation is, in
magnitude, sitting near the `L_inf` boundary on a large fraction of coordinates. So the interior-move
critique is a first-order effect, not the asymptotic one — given enough accepts the floor *does* fill in
the `eps` radius. What it can never fix is the *sign pattern*: those saturated coordinates carry the
random signs of the noise that happened to be accepted, so the perturbation converges toward a
random-signed point at radius `eps`, aligned with the useful direction at only the `~1.4%` level. That is
the durable lesson to carry forward: the magnitude takes care of itself, the *direction* does not, and no
amount of extra steps repairs a direction that is drawn blind.

The budget bookkeeping is where the floor quietly throws away most of its allowance, and this detail
decides its number. The oracle exhausts the *whole batch* to failure the instant the running query count
crosses `batch_size * n_queries = 50 * 1000 = 50000`, and — this is the sharp edge — when it flips it
returns *zeros* for the rest of the batch and scores the entire batch as a failure. So overshooting the
budget is not a soft cap that merely stops early; it converts still-working samples into hard misses. The
floor does not walk anywhere near that line: it runs a fixed `n_steps = max(1, min(n_queries, 64)) = 64`
iterations, each one `model(cand)` call that costs `batch_size = 50` queries, plus one initial
`model(adv)` call to seed `best`. That is `65` model calls, `65 * 50 = 3250` queries consumed against a
`50000` ceiling — about `6.5%` of the allowance used. Per sample the reported cost is `65` queries out of
`1000`: the floor deliberately leaves `935` of every `1000`, i.e. `93.5%`, on the table. That is the
`avg_queries = 65` I should expect to read back, flat across every scenario because the cap is a constant
that ignores whether a sample flipped early or never. Read charitably, the tiny cap does buy one thing:
absolute safety from the batch-to-zero catastrophe — there is no way `65` queries trips a `1000` budget.
But that safety is bought at the price of `93.5%` of the search, which is a terrible trade; the right
answer is to spend near the full budget with a margin, not to hide at `6.5%`. It is the weakest
configuration by construction — timid interior moves, no structure, and a self-imposed ceiling far below
the real budget.

There is one piece of vectorized care the floor does get right, and it is worth stating precisely because
every later rung inherits it. The accept rule operates per sample. I query the candidate's correct-class
score for every image in the batch at once, form `improve = cand_score < best`, and update only the rows
where it holds via a masked `torch.where`, carrying `best` forward so I never re-query an image whose
candidate lost. So a single batched forward pass advances every still-improving sample simultaneously, the
`50` queries are shared across the batch, but the keep/discard decision is independent per image. That is
the correct structure; the floor simply refuses to use it for more than `64` steps or with anything
smarter than uniform noise.

The strict-inequality accept (`cand_score < best`, kept only on a genuine drop) matters more than it
looks, and carrying `best` forward instead of re-querying the incumbent is what makes the query budget go
to *proposals* rather than to re-measuring what I already know. Because I never overwrite `best` on a tie
or a rise, the incumbent's score is monotone non-increasing across the `64` steps — the search can never
regress, so every one of the `65` queries either confirms the current point or improves it. That
monotonicity is the one genuinely sound thing about the floor and it is exactly why greedy random search
is a legitimate *floor* rather than a broken control: it will never do worse than its starting query, it
just converges glacially. It also means the failure mode is unambiguous — a sample that ends unflipped
ended unflipped because `64` blind proposals never found a downhill direction good enough, not because the
bookkeeping threw away a good candidate. There is no query-noise complication here to muddy that reading:
the models are undefended deterministic classifiers, so `f_y` is a fixed function of the input and a drop
is a real drop, not sampling jitter. On a *defended*, stochastic model the strict `<` rule would start
accepting on noise, but that is not this substrate, so the greedy rule is exactly as reliable as its
proposal distribution is aimed — which is to say, `~1.4%` aimed.

Before I accept this as the floor, let me walk the design space it is sitting in, because seeing *what it
declines to do* — and pricing each option — is what sets up the next rung honestly. Four levers are
worth pricing. First, the economic one: raise `n_steps` from `64` toward the full `1000`. That is `15.4x`
more candidate queries, so `15.4x` more accepted moves — but each accepted move still carries the same
`~1.4%` alignment, and I already showed the magnitude saturates the `eps` box after only `~32` accepts, so
the extra `~950` steps mostly reshuffle random signs at fixed radius; spending the budget is *necessary*
but plainly not *sufficient*. Second, the per-component one: move by `+/- eps` and let the projection pin
every touched component to a corner, so the first move already reaches full radius instead of the floor's
timid `eps/2` interior draw — this only buys the early steps, since accumulation eventually saturates
radius anyway, but it stops wasting the opening queries. Third, the objective one: descend the true margin
`J = f_y - max_{k!=y} f_k` instead of the lossy `f_y` proxy, so that a rising competitor counts as
progress and the zero-crossing coincides exactly with the flip — a strictly better signal for one extra
`max` per query. Fourth, and most powerful, the dimensional one: stop scattering across all `3072`
coordinates and concentrate the change into a small block, which raises the alignment from `1/sqrt(3072)`
toward `1/sqrt(k)` for a `k`-coordinate move — a `100`-coordinate block would sit near `1/sqrt(100) = 0.1`,
a `7x` better-aimed move — *if* I knew where to spend it. The floor buys none of these: it caps its own
budget, it moves interior, it descends the crude `f_y`, and it stays maximally spread out. That is
precisely why it is the floor, and these unbought levers are the map for what a stronger method has to
purchase.

Let me sanity-check the mechanism against its limiting cases rather than assert it. As `step -> 0` the
proposal vanishes, no candidate ever differs from the current point, nothing is ever accepted, and `asr`
would be zero — the floor's progress is monotone in step size, so its timid `eps/2` (interior, so really
`~eps/4` of effective move) is already leaving signal behind. At the other extreme, `D -> 1`: a
single-coordinate problem makes the random direction perfectly aligned (`cos = 1`), the `1/sqrt(D)`
penalty disappears, and greedy random search essentially solves it by line search. So the entire penalty
the floor pays is the `1/sqrt(D)` orthogonality of high dimension — the algorithm is not wrong, it is
just being run in `3072` dimensions with no structure to cut that factor. This is the cleanest possible
statement of the disease: `sqrt(3072) = 55.4`, so a blind proposal in this space is about `55x` worse-aimed
than the same greedy rule would be on a one-dimensional problem, and every rung after this one is really a
contest over how much of that `55x` it can claw back by imposing structure on the proposal. And the
feasibility clamps do their job: the inner clamp guarantees `||adv - x||_inf <= eps` exactly, the outer
guarantees `[0,1]`, so the harness's return-side re-check (`eps + 1e-6`, range) will pass and no sample is
lost to a validity violation — every miss the floor records is an honest failure to flip the model, not a
bookkeeping artifact. I can also check the units: `f_y` is a logit, dimensionless in the input's pixel
scale, and the accept test compares two logits directly, so there is no scale mismatch to worry about —
the only scales in play are `eps` and `step`, both in `[0,1]` pixel units, and both are handled by the
clamps.

Now reason about what this floor must do across the six scenarios, because running it is the entire point
of step 1, and I have no measured numbers yet — only the shape to predict. Greedy random search with
isotropic interior noise in `~3000` dimensions is the classic high-dimensional failure, and `64` timid
steps barely move the perturbation off the clean image. The mechanism that decides each scenario is how
far the flip-inducing boundary sits from the clean point relative to how far a random-walk of `~32`
accepted `~1.4%`-aligned moves can travel toward it. On the *easier* pairs — where the decision boundary
is close and even crude noise stumbles across it within `64` steps — the floor should flip a fair fraction
of images. On the *harder* pairs — a more robust architecture whose boundary sits farther out, or the
`CIFAR-100` setting — the same `64` moves should mostly fail, and `asr` should sag well below half.

I should be careful about which way `CIFAR-100` cuts, because the two obvious intuitions point opposite
ways and only one survives thought. On one hand `CIFAR-100` gives the correct class `99` competitors
instead of `9`, so there are many more classes that could overtake `f_y` — which makes *any* of them
crossing more likely, i.e. a *closer* nearest boundary, which should *help* an untargeted attack. On the
other hand a `100`-way model spreads its confidence and its logit geometry is more crowded, which could
make the crude `f_y` proxy noisier. For the floor specifically — whose whole game is just to get *some*
competitor above `f_y` — the first effect should dominate: more classes is more targets, so I would not
be surprised if `CIFAR-100` rows come out slightly *stronger* than their `CIFAR-10` siblings within the
same architecture, not weaker. The axis I expect to actually separate the floor is *architecture*, not
class count: whichever backbone presents the farthest, cleanest boundary is where `~1.4%`-aligned noise
should collapse hardest. I expect a wide spread across scenarios and a modest mean, precisely because
nothing in the floor adapts to the model or spends the budget it was handed; the flat `avg_queries` near
`65` should hold on every row regardless of difficulty, since the cap is a constant.

That diagnosis already sorts the floor's crippling choices into two clean diseases, and they are
separable. One is *economic*: it caps itself at `65` queries when `1000` are available, and my arithmetic
says curing that alone — running more steps — is easy but insufficient, because `15x` unaligned moves is
still unaligned. The other is *directional*: unstructured interior nudges are `~1.4%`-aligned in `3072`
dimensions, so even the steps it does take crawl, and on the hardest boundaries they crawl to nothing.
The directional disease is the real one. So the next rung has to do two things at once: stop leaving the
budget unused, and — using the very forward queries the floor throws away — reconstruct an actual descent
direction instead of guessing one. That is exactly the finite-difference gradient-estimation primitive
the background lists but the floor refuses to touch, and it is where I go next. The distilled floor
module — the literal scaffold edit — is in the answer.
