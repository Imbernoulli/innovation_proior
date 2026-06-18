JSMA cleared the Pixle floor, but only just, and the shape of its numbers is exactly the brittleness I
predicted. `asr = 0.0267` on `Rebuffi-R18-L2`, `0.0467` on `Augustin-L2`, `0.0667` on `Engstrom-L2`, mean
about `0.047`. So it roughly quadrupled the Pixle mean of `0.011` — confirming that *using* a per-pixel
importance signal beats guessing locations — but it is still in the low single-digit-percent range, a few
successes out of 150 on the hardest model and ten on the softest. The per-model ordering is informative:
JSMA does best on `Engstrom-L2` and worst on `Rebuffi-R18-L2`, the reverse of nothing in particular but
consistent with a first-order signal that is least suppressed on whichever model's local surface is least
flattened. The flat, low magnitude across all three is the tell. JSMA's saliency is a *local, first-order*
quantity read at the clean image, and the greedy walk commits to the best-scoring pair, saturates them,
and never reconsiders — and on `L2`-adversarially-trained models the first-order signal is precisely what
training was hardened against. So the both-signs gate finds few features that clear it, the greedy search
is choosing among weak candidates, and it stalls in local optima where no next pair looks good even though
some *other* support would have flipped the label. The gradient is honest but the surface lies to it
locally, and greedy with no backtracking has no way to recover.

That diagnosis points two ways. One is to keep the gradient but make the move richer and reconsiderable —
a direction I will hold in reserve. The other, which I take here, is to *drop the gradient entirely* and
search. JSMA failed not because gradients are useless but because a *greedy, local, first-order* use of
them gets trapped on a flattened surface; a method that can escape local optima by maintaining a diverse
population and only ever *evaluating* the objective sidesteps both the locality and the greediness. That
is the differential-evolution one-pixel attack, and it is genuinely the next rung up: a black-box,
gradient-free, diversity-keeping search that answers exactly JSMA's failure mode.

Let me derive it cleanly, because the encoding is where it lives or dies. I want to flip the label by
changing as few pixels as possible, amplitude free, but the `L0` constraint is *combinatorial*: choosing
*which* pixels is discrete, on top of choosing *what value* to write — continuous. A gradient is the wrong
instrument for selecting a sparse support (that is JSMA's wall, and Pixle dodged it only by giving up on
location choice). What fits a non-differentiable, multimodal objective I can only query is a metaheuristic,
and differential evolution in particular, because it has the two properties JSMA lacked. It keeps a
*population* and forms each child from a *scaled difference of two other members*
(`x_i' = x_{r1} + F*(x_{r2} - x_{r3})`), so the search radius self-adapts — large while the population is
spread (exploration), small as it converges (refinement) — and a child competes only with *its own
parent* one-to-one, which preserves diversity instead of letting a few strong candidates clone themselves
across the population. That diversity is exactly what lets DE jump out of the local optima a greedy method
dies in. And it only ever *evaluates* the objective, never differentiates it.

Let me be precise about *why* DE's two properties answer JSMA's two failures, because that is the whole
justification for switching tools rather than just retuning the saliency search. JSMA stalled in local
optima because it was greedy with no backtracking — once it saturated a pixel it could never undo that
choice, and on a flattened surface the locally-best pixel is often a globally-poor commitment. DE never
commits irrevocably: a candidate that looks good early can be displaced by its own descendants, and the
population holds *many* supports simultaneously, so a promising-but-different combination is not crushed by
the current leader. And JSMA leaned on a *local, first-order* signal that robust training specifically
suppressed; DE reads only the *global* objective value — the actual probability the model assigns the true
class on the full perturbed image — so it is immune to the gradient being small. It does not care that the
surface is locally flat; it cares only whether a candidate, as a whole, lowers the true-class probability.
That is exactly the right instrument for a surface engineered to defeat first-order attacks.

But DE optimizes a *real vector*, and my problem has a discrete part — which pixel. The unlock is the
encoding: one modified pixel is a 5-tuple `(x, y, R, G, B)` — a location and a color — and a candidate
with budget `d` is `d` such tuples concatenated, a real vector of length `5d`. Look at what that does to
the constraint: by writing exactly `d` tuples and leaving every other coordinate at zero, `||e||_0 <= d`
holds *by construction* — no penalty, no projection, no constrained optimization. The discrete "which
pixel" choice rides along as continuous `(x, y)` coordinates that get rounded to indices at apply-time.
DE never knows it is looking at an image; it searches `5d` reals and I decode. The fitness is the one
number the black box hands me: for the untargeted attack the harness runs, the true-class probability
`f_t(x + e)`, which DE *minimizes* — drive probability mass off the correct class until something else
wins. No surrogate loss; the exact quantity I care about, read straight off the softmax.

Now the part specific to *this* task, because the harness's configuration diverges sharply from the
textbook DE and it bounds what I should expect. The fill is the literal call
`OnePixel(model, pixels=pixels, steps=6, popsize=8, inf_batch=128)`, untargeted (`torchattacks.OnePixel`
defaults to untargeted, minimizing the true-class probability). Two things stand out. First, `pixels=24`:
the encoding is `24` five-tuples, a `120`-dimensional DE search per image — and that is a *large* search
space relative to the budget the optimizer is given. Second, that budget is tiny: `steps=6` generations
and a population multiplier `popsize=8`. The `torchattacks` implementation delegates to a scipy-derived
differential-evolution minimizer whose population size is `popsize` times the number of parameters per
member, so `popsize=8` over a `120`-dim problem is a real population, but `steps=6` means only six
generations of evolution before it stops. Six generations of DE over a `120`-dim space is a *coarse*
search — enough to find the obviously-fragile pixels, nowhere near enough to refine 24 well-placed ones on
a robust model. The `inf_batch=128` only batches the forward passes for speed; it does not buy more
search. So this rung is the black-box-search answer to JSMA's greediness, but it is *query-starved* in its
own way — a population that can escape local optima, given too few generations to fully exploit that.

There is a further subtlety worth naming, because it shapes how hard 24 pixels is for this configuration.
With the budget at 24, the optimizer must place *and color* 24 pixels well, but DE does not get to optimize
them one at a time — it evolves the whole `120`-dimensional vector jointly, and the difference-mutation
perturbs many coordinates of a candidate at once. Early generations, with the population spread, take large
exploratory steps that scatter all 24 pixels around; only as the population converges do the steps shrink
enough to refine individual placements. Six generations barely reaches the refinement phase, so the
realized attack tends to find a handful of pixels that matter and leaves the rest poorly placed. This is
the opposite failure from JSMA: JSMA placed each pixel carefully but greedily and locally; OnePixel places
all pixels by global search but coarsely. Neither has both careful placement and a global, reconsiderable
view — which is the gap the next rung starts to close.

Where does that leave my expectations against the JSMA floor of `0.047`? DE's structural advantage over
JSMA is real and on the exact axis JSMA failed: it does not commit greedily and it does not rely on a
local first-order signal, so it should not stall in the same local optima. With 24 pixels of budget and a
diversity-keeping population, it should find *more* flips than greedy saliency — I expect it to clear
`0.047` and land in the low-to-mid teens of percent, a clear step up. But I do not expect it to approach
the high success rates that the hardest sparse attacks reach, and the reason is the query budget: six
generations is too few to evolve a good 24-pixel support against a flattened surface, so DE will harvest
the easy fragile-pixel flips and run out of generations before it solves the harder samples. There is also
a subtle cost to the large search space — `120` dimensions with `8x` population and `6` steps spreads the
optimizer thin, so each pixel's location and color get only coarse optimization. My falsifiable
expectation: OnePixel's mean ASR clears JSMA's `0.047` comfortably — black-box population search beats
greedy first-order saliency on robust models — but stays well under the strongest sparse attacks, exposing
its own ceiling as *query starvation* rather than the wrong-tool problem JSMA had. If it comes back only a
little above JSMA, that would say six generations is simply too few for DE to express its advantage, and
the next rung should either spend its queries far more efficiently or go back to gradients with a smarter,
reconsiderable move.

The delta from the previous rung, concretely: where JSMA returned `attack(images, labels)` from a
saliency-guided greedy white-box pass and scored `0.047`, this rung returns `attack(images, labels)` from
a `OnePixel` instance — trading a local, greedy, first-order gradient search for a black-box,
diversity-keeping population search over a `5d`-real encoding in which the `L0` budget holds by
construction. The full scaffold module is in the answer. I expect it to beat the JSMA floor by escaping
the local optima that trapped greedy saliency, while its own six-generation budget caps how far it can
push — the ceiling the next rung must break.

The causal chain in one breath: JSMA's `0.047` is a *brittleness* result — greedy, local, first-order
saliency stalls on flattened robust surfaces → escape it by dropping the gradient for a population search
that only evaluates the objective and self-adapts its step via difference-mutation, keeping diversity to
jump out of local optima → encode each pixel as `(x,y,R,G,B)` and a candidate as `d` such tuples so the
`L0` budget is free, with fitness the raw true-class probability → wrapped in this task as untargeted
`OnePixel` with a `120`-dim search but only `steps=6`/`popsize=8`, so expect a clear gain over `0.047`
into the low-mid teens, capped by query starvation rather than the wrong tool — pointing the next rung at
spending queries far more efficiently.
