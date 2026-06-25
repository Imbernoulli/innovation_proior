The hierarchical lifts have been paying — `0.381240` at `24` cells, `0.381076` at `120` — and each rung
ran the same loop: upscale the optimized profile, kick it to break the block plateau, refine with an
annealed soft-max search while keeping the best true overlap. The feedback says the binding cap is still
resolution: the published frontier lives at several hundred cells (AlphaEvolve's `95`-step `0.380924`,
AutoEvolver's `~600`-step `0.38086945`), found by large evolutionary searches. So the obvious next move is
to lift once more, to the `~600`-cell scale the records use, and grind there. Whether that grind actually
buys anything past the upscale is the question this rung has to answer honestly, not assume.

I start with the one step I trust completely: take the optimized `120`-cell profile and upscale `×5` to
`600`. I have been calling this "free" all the way up the ladder, but at the endpoint I want to know *why*
it is free rather than just that the previous rungs behaved that way, because the whole endpoint rests on
it. Work the smallest case, `v = [a, b]` upscaled `×2` to `[a, a, b, b]`. The score is `max_k Σ_i v_i(1 −
v_{i−k})`, rescaled by `2/n`. For `[a,b]` the cross-correlation against `1−v` has lags `[ab, a(1−a)+b(1−b),
(1−a)b]`; with `a=0.3, b=0.7` that is `[0.09, 0.42, 0.49]`, max `0.49`, so `C = 0.49·2/2 = 0.49`. For the
doubled `[0.3,0.3,0.7,0.7]` the correlation is `[0.09, 0.18, 0.51, 0.84, 0.91, 0.98, 0.49]`, max `0.98`,
so `C = 0.98·2/4 = 0.49`. The max value doubled and the length doubled, and `C = max·2/n` cancels the two
exactly. That is the mechanism: repeating each cell `r` times multiplies the peak correlation by `r` and
the length by `r`, leaving `C` invariant. I check it once more at `×5` on a random feasible `8`-cell vector
— `C(v) = 0.483283`, `C(upscale ×5) = 0.483283`, sums `4 → 20` against target `20` — identical to six
places. So the upscale genuinely transfers the structure at no cost in score and only adds degrees of
freedom. That part I can lean on.

The question is what to do with those degrees of freedom at `600` cells, and here I have to confront a
solver wall I have been deferring. The coarse and middle rungs leaned on constrained SLSQP against the
soft-max surrogate, the right tool at two dozen and a hundred-odd cells. Does it still work at `600`? I time
a single annealed-SLSQP solve at both sizes: `n=120` takes `0.8 s`, `n=600` takes `24.3 s` — a `~30×` jump
for `5×` the variables. That is the QP inside SLSQP being super-linear in the number of heights, exactly as
I feared; a full annealed ladder is several such solves, and a basin-hop wraps tens of restarts around the
ladder, so at `600` cells the SLSQP route is minutes per restart. Worse, from a good starting point the
surrogate optimum it chases has essentially coincided with where the profile already sits, so it barely
moves for that cost. So SLSQP does not scale to the endpoint, and I need a refiner whose per-step cost is
just a correlation, not a QP.

The natural replacement is a first-order method on a cheap, exact gradient. I do not want to differentiate
the soft-max numerically, so before committing I write down the gradient of a single shift's overlap `c_k =
Σ_i v_i(1 − v_{i−s})` by hand: `∂c_k/∂v_i` picks up `(1 − v_{i−s})` from the first factor and `−v_{i+s}`
from where `v_i` appears as the subtracted term, i.e. the assembly `grad[i] += 1 − v[j]; grad[j] += −v[i]`
over the overlapping index range. I check this against finite differences on a random `10`-cell vector at
the binding lag: max `|analytic − finite-difference|` is `3.5e-10`. The formula is right, so a first-order
refiner on the active shifts is well-founded and `O(n²)` per step rather than a QP.

Now the actual refinement. The soft-max and the true `max` are not the same function, and the surrogate's
peak can sit below the true peak when `β` is too small relative to the spread of correlation values, so the
honest objective to descend at the very end is the *true* minimax: at the current point find the binding
(near-worst) shifts and distribute a descent step across them using the analytic subgradient I just
verified. That is the polish. I run it from the upscaled `600`-cell point — `2000` iterations, `lr₀ =
0.005`, decaying `lr`, keeping the best true `C` ever seen — and watch what it does, because this is the
step that is supposed to carve the endpoint gain.

It does not carve a gain. The upscaled point starts at `C = 0.3810764`. The very first polish step jumps to
`0.382021`, and the trajectory stays *above* the start the whole way: `0.381932` at iter 10, `0.381712` at
100, `0.381385` at 1000, `0.381346` at the last iteration. The best true `C` over all `2000` iterations
never drops below the starting `0.3810764` — the delta is exactly `0`. Watching the active set explains it:
the upscaled point has only **one** binding lag, not the many closely-tied shifts I had been picturing for a
spiky `600`-cell profile. (I check directly: the `×5`-upscaled vector has `1` near-max lag out of `1199`,
the same single binding shift the `120`-cell parent had, just rescaled.) With one active shift the
subgradient is just that one shift's gradient, and stepping along it to lower that peak immediately raises a
different correlation value, which becomes the new max — so the worst overlap goes *up*, and the projection
back onto the simplex does not rescue it. The polish is descending the right object but from a point that is
already a sharp local minimum of the true minimax; there is no nearby feasible direction that lowers the
single binding peak without raising another to meet it.

So the honest construction is: build the `120`-cell profile, upscale `×5` to `600`, run the polish, and
keep whichever of the two is actually better under the frozen evaluator — which is the unpolished upscaled
point, because the polish never beats it. That last guard, `return polished if C(polished) < C(b600) else
b600`, is not decoration; on this run it is the line that fires, and the returned vector is the upscaled one
at `C = 0.3810764`. I leave the polish in the pipeline because it is the right tool whenever the upscale
lands on a point with a fat active set — at finer or differently-kicked profiles it would have room to move
— but on this profile it confirms, by running and failing to improve, that the upscaled point is already a
robust local optimum the subgradient cannot leave.

What this rung actually delivers, then, is the value the upscale transfers for free: `C = 0.3810764` at
`600` cells, identical to the `120`-cell parent because upscaling is exactly score-preserving and the
endpoint refinement finds no feasible descent past it. That is a hair above the published step-function
records — AutoEvolver `0.38086945`, AlphaEvolve `0.380924` — and the gap is exactly the part of the problem
that a single bounded gradient-refinement run does not close: those records were bought by evolutionary
coding-agent searches run to `n=750` over `~12` hours, orders of magnitude more compute, which find the
finer irregular structure that a free upscale plus a local polish cannot synthesize. White's provable lower
bound `0.379005` sits below as the floor the true constant `C5` cannot cross. The number I report is the one
the evaluator returns on the returned `600`-cell vector, `0.3810764`, and the remaining gap to `0.38087` is
the honest measure of how tightly squeezed — and how much still-contested at the fifth decimal — this
seventy-year-old problem remains.
