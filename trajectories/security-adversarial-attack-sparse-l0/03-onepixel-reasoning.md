JSMA cleared the Pixle floor, but only just, and the shape of its numbers is exactly the brittleness I
predicted. `asr = 0.0267` on `Rebuffi-R18-L2`, `0.0467` on `Augustin-L2`, `0.0667` on `Engstrom-L2`, mean
about `0.047`. Read as raw counts over the `150`-sample runs these are `4`, `7`, and `10` flips — still
single-digit-to-low-double-digit successes, but a real step off the floor. Let me take the ratios against
Pixle properly, because they carry mechanism. The mean went from `0.0111` to `0.0467`, about `4.2x`;
per model, Rebuffi went `1 -> 4` flips (`4x`), Augustin `3 -> 7` (`2.3x`), Engstrom `1 -> 10` (`10x`).
That spread in the multipliers is the interesting part. Pixle's softest model was Augustin (blind luck
paid off there); JSMA's softest is now Engstrom, and Engstrom is where the *saliency* signal bought the
most (`10x`). The per-model ordering *flipped* between a blind attack and a first-order one, which is
exactly what I would expect if the thing JSMA exploits — a usable first-order forward derivative — is
least suppressed on Engstrom's local surface and most suppressed on Rebuffi's, independent of where blind
luck happened to land. So confirming that *using* a per-pixel importance signal beats guessing locations
is not just the mean moving up `4.2x`; it is the ordering rearranging along the axis the new signal
actually reads. The flat, low magnitude across all three is still the tell, though. JSMA's saliency is a
*local, first-order* quantity read at the clean image, and the greedy walk commits to the best-scoring
pair, saturates them, and never reconsiders — and on `L2`-adversarially-trained models the first-order
signal is precisely what training was hardened against. So the both-signs gate finds few features that
clear it, the greedy search is choosing among weak candidates, and it stalls in local optima where no
next pair looks good even though some *other* support would have flipped the label. The gradient is honest
but the surface lies to it locally, and greedy with no backtracking has no way to recover.

That diagnosis points two ways, and I should name both and choose deliberately. One route keeps the
gradient but makes the move richer and *reconsiderable* — relinearize the boundary each step so no pixel
choice is permanent — a direction I will hold in reserve because it is the natural answer to *greediness*
specifically. The other, which I take here, is to *drop the gradient entirely* and search. Why this order?
Because JSMA's failure has two independent components — it is local (first-order, read at one point) *and*
it is greedy (no backtracking) — and I do not yet know which one dominates. A reconsiderable gradient
method fixes greediness but still leans on the first-order signal that robust training suppressed; if the
locality is the real killer, it would inherit the problem. A population search that only ever *evaluates*
the objective sidesteps *both* at once: it reads the global objective value, immune to the gradient being
small, and it holds many supports simultaneously, immune to greedy commitment. So the evaluative search is
the cleaner experiment — it isolates whether escaping locality-and-greediness together is enough, and if
it succeeds it tells me the gradient was never essential here; if it stalls too, that implicates the query
budget rather than the tool. That is the differential-evolution one-pixel attack, and it is genuinely the
next rung up: a black-box, gradient-free, diversity-keeping search that answers exactly JSMA's failure
mode.

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
holds *by construction* — no penalty, no projection, no constrained optimization. Let me verify that
constraint is truly free and not quietly violated at decode time, because it is the whole reason this
encoding is legal. The decoder rounds each tuple's `(x, y)` to an integer index and writes the color
there; two tuples *could* round to the same pixel, which only *lowers* the distinct-pixel count, never
raises it, so `d` tuples touch *at most* `d` distinct spatial pixels — the budget holds with slack, never
breach. And because the tuples are the only thing written, no other pixel moves. So the harness's channel-
wise `L0 <= 24` check can never reject a candidate this encoding produces; the constraint is structural,
exactly as with Pixle's rearrangement but now with a searchable color instead of a copied one. The
discrete "which pixel" choice rides along as continuous `(x, y)` coordinates that get rounded to indices
at apply-time. DE never knows it is looking at an image; it searches `5d` reals and I decode. The fitness
is the one number the black box hands me: for the untargeted attack the harness runs, the true-class
probability `f_t(x + e)`, which DE *minimizes* — drive probability mass off the correct class until
something else wins. No surrogate loss; the exact quantity I care about, read straight off the softmax.

Before I fix on DE, let me check the two nearby alternatives so the choice is reasoned and not reflexive.
One is a covariance-adapting evolution strategy, which learns a full search-covariance and would refine a
`120`-dim landscape more sharply per generation — but it pays for that with a per-generation cost that
grows with the *square* of the dimension to maintain and factor the covariance, and at `120` dimensions
with only six generations I would spend most of the budget estimating a covariance I never get to exploit;
DE's difference-mutation approximates an adaptive step *for free* from the population spread, which is
exactly what a starved budget wants. The other is a pure random search over the same `5d` encoding — but
that discards the one thing DE adds over Pixle, the *difference* signal that couples candidates so the
search radius tracks the population, and it would collapse this rung back toward the blind sampling I am
trying to climb past. So DE is the right point on this axis: cheaper per step than covariance adaptation,
smarter per step than random sampling, and its self-adaptation is the property that a six-generation
budget can actually use. That reasoning also tells me *not* to reach for the textbook `DE/rand/1` with
`F = 0.5` and a large population run to convergence — I have six generations, not sixty, so the
configuration the harness lands (`steps = 6`, `popsize = 8`, untargeted) is deliberately a *coarse*
instantiation, and I should read the result as "what does under-converged DE get," not "what does DE get."

Let me trace the difference-mutation's self-adapting radius with a number, because "large early, small
late" is the property I am counting on and I want to see it is not wishful. A child coordinate is
`x_{r1} + F * (x_{r2} - x_{r3})` with `F ~ 0.5`. In generation one the population is initialized spread
across the whole feasible range — a location coordinate spans `[0, 31]`, so two random members differ by
an expected `~10` in that coordinate, and the mutation step is `0.5 * 10 = 5` pixels of displacement: huge,
genuinely exploratory, scattering candidate locations across the image. As selection culls the losers and
the survivors cluster, the spread `|x_{r2} - x_{r3}|` shrinks; once the population has concentrated near a
promising support the same coordinate might differ by only `~2`, giving a step of `0.5 * 2 = 1` pixel —
refinement. So the *same* fixed `F` produces a step that decays from `~5` pixels to `~1` pixel purely
through population convergence, no schedule required. That is the mechanism I want. The problem is the
timeline: that decay from exploration to refinement takes many generations to complete, and I have six.
Six generations get the step down from `~5` toward maybe `~3` on the coordinates that have started
converging, nowhere near the `~1`-pixel refinement regime for all `24` pixels at once. So the self-
adaptation is real but *truncated* — I get the exploratory phase and the very start of refinement, which
is exactly why I expect the attack to place a few pixels well and leave the rest coarse.

Now the part specific to *this* task, because the harness's configuration diverges sharply from the
textbook DE and it bounds what I should expect. The fill is the literal call
`OnePixel(model, pixels=pixels, steps=6, popsize=8, inf_batch=128)`, untargeted (`torchattacks.OnePixel`
defaults to untargeted, minimizing the true-class probability). Let me do the search-space bookkeeping,
because it is what turns "coarse search" from a hunch into a count. With `pixels = 24` the encoding is `24`
five-tuples, a `5 * 24 = 120`-dimensional DE search per image — and that is a *large* search space
relative to the budget the optimizer is given. The `torchattacks` implementation delegates to a
scipy-derived differential-evolution minimizer whose population is `popsize` times the number of
parameters per member, so `popsize = 8` over a `120`-dim problem is a population on the order of
`8 * 120 ~= 960` candidates, and `steps = 6` means only six generations of evolution before it stops. So
the whole attack spends roughly `960 * 6 ~= 5760` fitness evaluations per image — orders of magnitude more
model queries than JSMA's `120` backward passes, yet against a `120`-dimensional continuous landscape it
is still coarse: six generations is barely enough for the difference-mutation to shrink from its initial
exploratory spread to a refinement radius. `inf_batch = 128` only batches those forward passes for speed;
it does not buy more search. So this rung is the black-box-search answer to JSMA's greediness, but it is
*query-starved* in its own way — a population that can escape local optima, given too few generations to
fully exploit that.

There is a further subtlety worth naming, because it shapes how hard 24 pixels is for this configuration.
With the budget at 24, the optimizer must place *and color* 24 pixels well, but DE does not get to optimize
them one at a time — it evolves the whole `120`-dimensional vector jointly, and the difference-mutation
perturbs many coordinates of a candidate at once. Early generations, with the population spread, take large
exploratory steps that scatter all 24 pixels around; only as the population converges do the steps shrink
enough to refine individual placements. Six generations barely reaches the refinement phase, so the
realized attack tends to find a handful of pixels that matter and leaves the rest poorly placed. This is
the opposite failure from JSMA: JSMA placed each pixel carefully but greedily and locally; OnePixel places
all pixels by global search but coarsely. Neither has both careful placement and a global, reconsiderable
view — which is the gap the next rung starts to close. It is worth noting the tension the budget creates:
a *smaller* `d` would give DE a lower-dimensional space it could refine well in six generations, but the
harness fixes the encoding at the full `pixels = 24`, so I am handed a `120`-dim search and six
generations and must live with the mismatch. That mismatch is precisely the query-starvation I expect to
read off the result.

The one-to-one tournament deserves a second look, because it is the specific mechanism that protects the
diversity JSMA lacked, and I want to be sure it is not undone by the small budget. In DE a child replaces
*only its own parent*, and only if it is fitter — so a single strong candidate cannot overwrite the whole
population the way rank-proportional selection would let it. With a population near `960`, that means up to
`960` distinct supports can coexist through all six generations, each refined against its own lineage.
Contrast that with JSMA, which carries exactly *one* support (the growing set of saturated pixels) and can
never fork it. So even six under-converged generations of DE explore hundreds of supports where JSMA
explored one greedy trajectory — that is the concrete sense in which DE "keeps diversity," and it is why I
expect it to find flips on images where JSMA's single greedy path stalled. The limit is not diversity but
*depth*: hundreds of supports, each only coarsely refined, so the attack finds the easy flips broadly but
polishes none of them deeply. That is the exact profile of a search that is wide but shallow, and it
predicts a clear gain over greedy saliency that nonetheless plateaus well below saturation.

Where does that leave my expectations against the JSMA floor of `0.047`? DE's structural advantage over
JSMA is real and on the exact axis JSMA failed: it does not commit greedily and it does not rely on a
local first-order signal, so it should not stall in the same local optima. With 24 pixels of budget and a
diversity-keeping population, it should find *more* flips than greedy saliency — I expect it to clear
`0.047` and land in the low-to-mid teens of percent, a clear step up, on the order of a few tens of flips
out of `150` rather than JSMA's four-to-ten. But I do not expect it to approach the high success rates
that the hardest sparse attacks reach, and the reason is the query budget: six generations is too few to
evolve a good 24-pixel support against a flattened surface, so DE will harvest the easy fragile-pixel
flips and run out of generations before it solves the harder samples. There is also a subtle cost to the
large search space — `120` dimensions with `~8x` population and `6` steps spreads the optimizer thin, so
each pixel's location and color get only coarse optimization. My falsifiable expectation: OnePixel's mean
ASR clears JSMA's `0.047` comfortably — black-box population search beats greedy first-order saliency on
robust models — but stays well under the strongest sparse attacks, exposing its own ceiling as *query
starvation* rather than the wrong-tool problem JSMA had. I would also expect the hardest model to remain
hardest under DE — where the surface is most flattened, more of the six generations are wasted before a
working support emerges — so if any model lags it should be the one JSMA found toughest. If it comes back
only a little above JSMA, that would say six generations is simply too few for DE to express its advantage,
and the next rung should either spend its queries far more efficiently or go back to gradients with a
smarter, reconsiderable move.

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
