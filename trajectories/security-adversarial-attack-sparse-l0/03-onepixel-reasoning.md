JSMA cleared the Pixle floor but only just. `asr = 0.0267` on `Rebuffi-R18-L2`, `0.0467` on `Augustin-L2`,
`0.0667` on `Engstrom-L2`, mean `~0.047` — as counts `4`, `7`, `10` flips over the `150`-sample runs, a real
step off the floor but still low. The mean went from `0.0111` to `0.0467`, about `4.2x`; per model Rebuffi
`1 -> 4`, Augustin `3 -> 7`, Engstrom `1 -> 10`. That spread in the multipliers is the interesting part:
Pixle's softest model was Augustin (blind luck), JSMA's softest is now Engstrom (`10x`), and the per-model
ordering *flipped* between a blind attack and a first-order one — exactly what I would expect if the thing
JSMA exploits, a usable first-order forward derivative, is least suppressed on Engstrom's surface and most on
Rebuffi's, independent of where blind luck landed. So *using* an importance signal beats guessing not just in
the mean but in the ordering rearranging along the axis the new signal reads. Still, the flat low magnitude
is the tell: JSMA's saliency is local and first-order, read at the clean image, and the greedy walk commits
and never reconsiders — on `L2` models the first-order signal is precisely what training hardened against, so
the both-signs gate finds few features, the search chooses among weak candidates, and it stalls in local
optima where a different support would have flipped the label. The gradient is honest but the surface lies to
it locally, and greedy with no backtracking cannot recover.

That points two ways. One keeps the gradient but makes the move *reconsiderable* — relinearize the boundary
each step — the natural answer to greediness specifically; I hold it in reserve. The other, which I take
here, drops the gradient entirely and searches. Why this order? JSMA's failure has two independent components
— it is local (first-order) *and* greedy (no backtracking) — and I do not know which dominates. A
reconsiderable gradient method fixes greediness but still leans on the first-order signal robust training
suppressed; if locality is the real killer it inherits the problem. A population search that only
*evaluates* the objective sidesteps both at once — it reads the global objective value, immune to the
gradient being small, and holds many supports simultaneously, immune to greedy commitment — so it is the
cleaner experiment: if it succeeds, the gradient was never essential here; if it stalls too, that implicates
the query budget rather than the tool. That is the differential-evolution one-pixel attack.

The encoding is where it lives or dies. The `L0` constraint is combinatorial — choosing *which* pixels is
discrete, on top of *what value* to write — and a gradient is the wrong instrument for selecting a sparse
support (JSMA's wall). What fits a non-differentiable, multimodal objective I can only query is a
metaheuristic, and differential evolution specifically, because it has the two properties JSMA lacked: it
keeps a *population* and forms each child from a scaled difference of two other members,
`x_{r1} + F*(x_{r2} - x_{r3})`, so the search radius self-adapts — large while the population is spread,
small as it converges — and a child competes only with *its own parent* one-to-one, preserving diversity
instead of letting a few strong candidates clone across the population. Those are exactly the answers to
JSMA's greedy commitment (a candidate can be displaced by its own descendants, and many supports coexist)
and to its reliance on a local signal (DE reads only the global true-class probability on the full perturbed
image, indifferent to the surface being locally flat).

But DE optimizes a real vector and my problem has a discrete part. The unlock: one modified pixel is a
5-tuple `(x, y, R, G, B)`, and a candidate with budget `d` is `d` such tuples concatenated, a real vector of
length `5d`. By writing exactly `d` tuples, `||e||_0 <= d` holds *by construction* — no penalty, no
projection. The decoder rounds each `(x, y)` to an integer index and writes the color; two tuples could
round to the same pixel, which only *lowers* the distinct-pixel count, so `d` tuples touch *at most* `d`
distinct pixels — the budget holds with slack, never breach, and the harness's channel-wise check can never
reject a candidate this encoding produces. The discrete "which pixel" rides as continuous `(x, y)` rounded
at apply-time; DE never knows it is looking at an image, it searches `5d` reals and I decode. Fitness is the
one number the black box hands me: the true-class probability `f_t(x + e)`, minimized.

Two nearby alternatives, so the choice is reasoned. A covariance-adapting evolution strategy learns a full
search-covariance and would refine a `120`-dim landscape more sharply per generation, but its per-generation
cost grows with the *square* of the dimension, and at `120` dimensions with six generations I would spend
the budget estimating a covariance I never get to exploit; DE's difference-mutation approximates an adaptive
step for free from the population spread, which is what a starved budget wants. Pure random search over the
same encoding discards the one thing DE adds over Pixle — the *difference* signal coupling candidates so the
radius tracks the population — collapsing back toward blind sampling. So DE is the right point: cheaper per
step than covariance adaptation, smarter per step than random sampling. That same reasoning warns against the
textbook `DE/rand/1` run to convergence — I have six generations, not sixty — so the harness's
`OnePixel(pixels=pixels, steps=6, popsize=8, inf_batch=128)` is deliberately a *coarse* instantiation, and I
should read the result as "what does under-converged DE get."

The self-adapting radius is the property I count on, and it is truncated here. A child coordinate is
`x_{r1} + F*(x_{r2}-x_{r3})` with `F ~ 0.5`; in generation one the population spans a location coordinate's
`[0,31]` range, so two members differ by an expected `~10` and the step is `0.5*10 = 5` pixels — genuinely
exploratory. As selection culls losers and survivors cluster, `|x_{r2}-x_{r3}|` shrinks; near a promising
support the same coordinate differs by `~2`, a step of `~1` pixel — refinement, with no schedule. But that
decay takes many generations and I have six: six get the step down from `~5` toward maybe `~3`, nowhere near
`~1`-pixel refinement for all `24` pixels at once. So I get the exploratory phase and the very start of
refinement — which is why I expect the attack to place a few pixels well and leave the rest coarse.

The search-space bookkeeping turns "coarse" into a count. With `pixels = 24` the encoding is `24` five-tuples,
a `120`-dimensional DE search per image. The scipy-derived minimizer sets population to `popsize` times
parameters per member, so `popsize = 8` over `120` dims is `~960` candidates, and `steps = 6` means six
generations — roughly `960 * 6 ~= 5760` fitness evaluations per image, orders of magnitude more model queries
than JSMA's `120` backward passes, yet still coarse against a `120`-dimensional landscape (`inf_batch = 128`
batches those forward passes for speed; it does not buy more search). So near-`960` supports coexist through
all six generations — versus JSMA's single, un-forkable support — each only coarsely refined: the attack
finds the easy flips broadly but polishes none deeply, wide but shallow. This is the opposite failure from
JSMA, which placed each pixel carefully but greedily and locally; OnePixel places all pixels by global search
but coarsely. Neither has both careful placement and a reconsiderable global view — the gap the next rung
starts to close. And the budget fixes the encoding at the full `24` pixels, so I am handed a `120`-dim search
and six generations and must live with the mismatch; a smaller `d` would give DE a space it could refine in
six generations, but that is not on offer.

Where does that leave expectations against JSMA's `0.047`? DE's advantage is on the exact axis JSMA failed —
no greedy commitment, no reliance on a local first-order signal — so it should not stall in the same local
optima and should find more flips. I expect it to clear `0.047` into the low-to-mid teens of percent, on the
order of a few tens of flips out of `150`. But not near the high rates the strongest sparse attacks reach,
because six generations is too few to evolve a good 24-pixel support against a flattened surface: DE harvests
the easy fragile-pixel flips and runs out of generations before the harder samples. So OnePixel's ceiling is
*query starvation*, not the wrong-tool problem JSMA had. I would also expect the hardest model to stay
hardest — where the surface is most flattened, more generations are wasted before a working support emerges —
so if any model lags it should be the one JSMA found toughest. If it comes back only a little above JSMA, six
generations is simply too few for DE to express its advantage, and the next rung should either spend queries
far more efficiently or go back to gradients with a smarter, reconsiderable move.
