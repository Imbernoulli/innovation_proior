Let me start from where the success-history adaptive DE I already trust actually leaves performance on
the table, because I do not want to reinvent it — I want to find the smallest set of changes that
makes it strictly better. L-SHADE is the machine: each individual samples `F_i` from a Cauchy and
`CR_i` from a Normal, each centered on a slot of an `H`-entry memory; it mutates by current-to-pbest/1
with an external archive feeding the second difference term; it greedily keeps the better of trial and
parent; it summarizes the winners' parameters into one memory slot per generation by the weighted
Lehmer mean for both `F` and `CR`; and it shrinks the population linearly to `N_min`
over the budget, deleting the worst. It self-tunes `F` and `CR`, it manages exploration-to-refinement
through both the difference vectors and the population schedule, and it is robust to a single bad
generation because the memory has `H` slots. So the adaptation is genuinely good. Where, then, is the
slack?

The first place I keep noticing is the *early* run. The memory starts at a neutral `0.5` for both `F`
and `CR`. But I know something about the early phase that the neutral start ignores: when the
population is still spread across the box and I am exploring, I generally want a *high* `CR` — most
coordinates of the donor taken into the trial, so the population can make large coordinated moves —
and the high `CR` regime tends to come with a not-too-small `F`. Starting the memory at `0.5` means
the first many generations are spent re-discovering, on every new problem, that high `CR` helps early.
That is budget I could save by initializing the bias I already believe in. So: start all `CR` memory
slots high, at `0.8`, and start `F` conservatively at `0.3`. This is a free head start — if the problem turns out to
prefer low `CR` (a near-separable landscape), the adaptation will drive the slots down anyway, but the
default no longer wastes the early generations.

There is a sharper version of the same idea. The memory adapts toward whatever the winners report, and
on a hard multimodal problem the winners late in the run may drive every slot toward a quiet,
conservative setting — small `F`, the search going gentle exactly when it might be stuck. I want a
permanent reservoir of aggression that the adaptation can never extinguish. So I reserve one sampled
period (the last memory index) to use `M_F = M_CR = 0.9` no matter what is stored in the adaptive
arrays. Now, because each individual picks its
memory index uniformly, a fixed fraction of the population — about `1/H` of it — always samples around
an aggressive `(0.9, 0.9)` regardless of what the rest of the memory has decided. This is cheap
insurance against the whole population going quiet, and it costs me only one of `H` adaptable slots.

The third early-run fix is a guardrail, not a bias. The Cauchy on `F` has heavy tails, so even with a
moderate center it will occasionally throw an `F` near 1; early in the run, a near-1 `F` on a spread
population produces an enormous, destabilizing donor that almost always loses. And the Normal on `CR`
will occasionally propose a near-0 `CR` early, which is the one-coordinate-at-a-time policy — fine
late, but premature early when I still want coordinated moves. So I clamp by *phase*. For the first
quarter of the budget I floor `CR` at `0.7`; for the first half, floor it at `0.6`; and for the first
0.6 of the budget I cap `F` at `0.7`. These are not adapted quantities — they are hard rails that keep
the early sampling inside the regime I argued is right, and they relax as the run proceeds so the
adaptation takes over for the refinement phase. (The exact fractions are chosen empirically, but the
*shape* — high `CR`, capped `F` early, both relaxed later — is the principled part.)

And a fourth, subtle one about the memory update itself. L-SHADE writes a slot by overwriting it with
the weighted Lehmer summary of this generation's winners. But one generation is a noisy estimate, and
overwriting throws away the slot's accumulated history. I would rather *blend*: write
`M[k] <- (mean_WL(S) + M[k]_old)/2`, the average of the new summary and the old slot value. Now a
single generation can move a slot by at most half the distance to its summary, so a noisy generation
is damped and the slot retains memory of what worked before. This is a small change with a real
stabilizing effect on the adaptation trajectory.

Those four are refinements to the *parameter propagation*. Now the change I think matters most, which
is to the *mutation operator itself*. Look hard at current-to-pbest/1:
`v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)`. The same `F_i` multiplies two structurally
different things. The first term, `x_pbest - x_i`, is a *directed pull toward the elite* — it is
exploitation, dragging the individual toward one of the best solutions found. The second term,
`x_r1 - x_r2`, is the *self-scaling random difference* — it is the exploration, the perturbation that
carries the population's scatter. Using one `F_i` for both forces the strength of the elite-pull and
the strength of the random perturbation to move in lockstep. But I have just spent four refinements
arguing that the right balance between exploration and exploitation *changes over the run*: broad and
diversifying early, focused and elite-following late. So the elite-pull term should be *weaker* than
the perturbation early (do not commit to the current elite while still exploring) and *stronger* than
the perturbation late (follow the good solutions once the basin is found). A single `F_i` cannot
express that.

The fix is to give the elite-pull term its own factor — a *weighted* `F`. Replace `F_i*(x_pbest -
x_i)` with `Fw*(x_pbest - x_i)`, where `Fw` is `F_i` scaled by a phase-dependent multiplier, keeping
`F_i` on the random-difference term. Early in the run I want the elite pull weaker, so `Fw < F_i`;
late, stronger, so `Fw > F_i`. A simple three-step schedule on the budget fraction does it:
`Fw = 0.7*F_i` for the first 0.2 of the budget, `0.8*F_i` for the first 0.4, and `1.2*F_i` thereafter.
The donor becomes `v_i = x_i + Fw*(x_pbest - x_i) + F_i*(x_r1 - x_r2)` — current-to-pbest *weighted*.
Early, the individual is perturbed more than it is pulled toward the elite, preserving diversity;
late, it is pulled toward the elite more than it is perturbed, sharpening convergence. This decouples
the two roles `F` was forced to play, and it is the single most consequential change over the
unweighted operator: it is the difference between an algorithm whose exploration/exploitation balance
is one knob and one whose balance follows the run's phase by construction.

Let me also reconsider the greediness `p` of the pbest pool in light of the same phase argument.
Pulling toward the top `N*p` individuals: a *larger* `p` early means the elite pool is broad, so the
pull is toward a diverse set of good solutions, not one incumbent — that is the diversity I want
early. A *smaller* `p` late means the pool narrows toward the very best, sharpening the convergence.
So `p` should *decrease* over the run, linearly from a larger value to a smaller one:
`p = (p_max - p_min)*nfes/max_nfes + p_max`... no — I want it to *decrease*, so
`p = p_max - (p_max - p_min)*nfes/max_nfes`, starting at `p_max` and ending at `p_min`. A natural
setting is `p_max = 0.25`, `p_min = p_max/2 = 0.125`: early, draw the elite guide from the top quarter
(broad), late, from the top eighth (sharp). This matches the same exploration-to-exploitation arc as
the weighted mutation, applied to the pool the elite guide comes from.

Two edge cases I will pin down so the adaptation does not divide by zero. First, if a generation
produces no winners, no slot is updated — no evidence, no change. Second, `CR`: on some landscapes the
best policy is `CR = 0` (change one coordinate at a time). If a slot's successful `CR` values are all
zero, the weighted Lehmer mean is `0/0`; I recognize this as "this slot means `CR = 0`," mark it
terminal, and whenever an individual draws from a terminal slot it first sets `CR_i = 0`. The early
phase floors still apply afterward, so the terminal value only becomes literal zero once those rails
have relaxed. Once terminal, the slot stays terminal — it has discovered the one-coordinate regime,
which is slow but thorough on rugged functions.

The remaining constants follow standard adaptive-DE practice. Memory size `H = 5`: enough slots that
one bad generation cannot dominate, small enough that stale entries do not steer the search; and with
the last sampled period fixed at `0.9`, four ordinary periods carry learned centers at any instant. `N_init` scales with the problem,
`round(25*log(D)*sqrt(D))` — large enough early for exploration and good adaptation statistics,
knowing linear reduction will whittle it to `N_min = 4` (the minimum current-to-pbest/1 can run on).
The archive caps at `|A| = N` and rescales as `N` shrinks, random deletion on overflow. The
Cauchy/Normal spread `0.1` is tight enough that samples stay near the learned centers, loose enough
(with Cauchy's tail) to keep exploring parameter space. Bound violations are repaired by moving the
offending coordinate halfway between the violated bound and the parent's coordinate — the standard DE
repair that keeps the trial inside the box without snapping it hard to the wall.

Now let me assemble the whole machine and check it is exactly L-SHADE plus these changes, nothing more.
Initialize `N_init` random vectors and evaluate; set `H` memory slots to `M_F = 0.3`, `M_CR = 0.8`,
with the last sampled period overriding its center to `0.9, 0.9`; empty archive; memory counter `k = 0`. Each generation: sort
by fitness; compute the current `p` from the decreasing schedule and the elite pool from the top
`N*p`; for each individual pick a memory index `r`, draw `CR_i = randn(M_CR[r], 0.1)` (or 0 if
terminal) clamped to `[0,1]` then phase-floored, draw `F_i = randc(M_F[r], 0.1)` (resample if `<= 0`,
truncate to 1) then phase-capped, compute `Fw` from the budget fraction, pick `x_pbest` from the pool,
`x_r1` from `P`, `x_r2` from `P ∪ A`, form the weighted donor `v_i = x_i + Fw*(x_pbest - x_i) +
F_i*(x_r1 - x_r2)`, repair to bounds, binomial-cross to the trial, evaluate, and greedily select. A
strict improvement stores the losing parent in the archive and records `(F_i, CR_i, |Δf|)` as a
success. After the generation, if there were successes, blend the improvement-weighted Lehmer means of
`S_F` and `S_CR` into slot `k` (with the terminal-`CR` rule and the `(new+old)/2` averaging), advance
`k` round-robin; then apply the linear population reduction (delete the worst, rescale the archive)
and decrease the pbest fraction along its `0.25 -> 0.125` schedule.
That is the entire method: L-SHADE's reproduction and schedule, with the high/frozen/phase-rail memory
handling, the blended update, the decreasing `p`, and — the heart of it — the weighted mutation factor
that decouples the elite pull from the random perturbation and slides their balance from
exploration-heavy early to exploitation-heavy late.
