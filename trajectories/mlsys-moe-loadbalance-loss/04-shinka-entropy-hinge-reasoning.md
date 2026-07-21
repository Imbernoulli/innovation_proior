The global-batch loss landed where I expected, and the full table tells a layered story. Plain
global-batch LBL came in at `L_imb = 0.0561`, a hair better than Switch's `0.0587` — the marginal
separation I predicted, since the over-constraint tax never surfaced at this scale. Its cross-entropy
`3.7279` is a whisker under Switch's, its fitness `−3.7840` a touch above. So among the pure
differentiable losses the frontier sits at `0.0561`, and scope alone bought me almost nothing here
beyond a cleaner principle. The striking row is the other one: global-batch LBL *plus* the loss-free
selection bias drove imbalance to `0.0160`, a `3.5×` cut, at a cross-entropy of `3.7274` that is if
anything slightly better, for a fitness of `−3.7434`. But I have to read that for what it is. The bias
is a non-gradient hard-count controller that stacks with the loss rather than a loss competing with it,
so `0.0160` is not a loss result — it is a piece of information: it proves that in this exact model, at
this cross-entropy, an imbalance down near a sixtieth is physically reachable. There is `3.5×` of
headroom between what the best pure loss achieves and what a hard controller extracts, and the question
this rung asks is whether a smarter *loss* — one that reaches the router's own probabilities — can
recover part of that headroom without borrowing the count controller's machinery.

To know where the headroom is, I go back to what the global loss leaves undone. Its push on expert `j`
is `N P_j(f_j − m)`, magnitude scaling with `P_j`, so it equalizes the *average* usage but treats an
expert at a tenth of its fair share worse than one at nine-tenths — the nearly-dead expert with
`P_j ≈ 0` gets the *weakest* gradient precisely when its deficit is largest, and the two small factors
multiply so the rescue collapses in the tail. The count bias does not close this from the loss side: it
balances the counts by shoving the hard selection around while handing the router *no gradient* about
balance through `P`. So the gap I want a new loss term to fill is specific: single out the under-utilized
experts and push specifically on them, through `P`, without the smooth term's fatal habit of easing off
exactly where the expert is coldest. But the obvious ways to build such a term are exactly the ways to
wreck specialization, and I need to see those failures before I trust whatever survives them.

The shape a targeted under-use penalty must have: it should fire on an expert below some usage floor and
leave every expert at or above its fair share completely alone — a two-sided penalty would go right back
to pressuring healthy experts, the disease. That shape is a one-sided hinge: penalize `τ − f_i` only when
positive, zero otherwise. An expert above the floor `τ` contributes nothing; below it, in proportion to
how far below, so all the force concentrates on the cold tail — the exact opposite of the smooth `f·P`
term that thins out in the tail. The floor should be a small fraction of `1/N`, not `1/N` itself: I do
not want to demand every expert hit its fair share, only that none wither below a survival minimum.

But a bare hinge is dangerous. If it fires whenever an expert dips below the floor, it will fire even
when the router is healthy and the dip is nothing but the roughness of a finite batch — a momentary
sampling under-count. Pushing hard then does exactly what the micro-batch loss did: it flattens
legitimate structure and raises the cross-entropy, chasing noise as if it were collapse. So the hinge
needs a gate that answers one question — is this real collapse, or just normal variation? — and only
opens on collapse. Several ways to build that gate are traps. Cranking the coefficient `α` on the global
term just scales `N P_j(f_j − m)` including the `P_j` prefactor that is already nearly zero for the
expert I care about — uniform pressure on a non-uniform problem, pushing hardest where least needed. A
symmetric penalty on the spread of `f` pays equal attention to over- and under-users, spending half its
budget suppressing hot experts and still fading in the tail. The interesting temptation is entropy: the
router's entropy `H(P)` is a natural single number for "how collapsed is this router," so why not add a
term that maximizes it? Because maximizing router entropy pushes *every token's* probability toward
uniform — per-token flattening, the specialization-destroying move I have avoided since the micro-batch
rung. So entropy is the wrong thing to *optimize*. But that failure is the clue: entropy is an excellent
thing to *read*. A peaked, low-entropy router is where experts genuinely starve and the hinge should
fire hard; a near-uniform, high-entropy router is healthy and it should stay quiet. So I do not add
entropy to the loss — I use it as the gate, weighting the hinge by a function of entropy that is large
when the router is peaked and small when it is spread.

Concretely the gate is `s(P_ℓ) = 0.5 + (1 − H(P_ℓ) / log N_E)`. Normalizing `H` by `log N_E` — the
entropy of the uniform distribution — puts the ratio in `[0, 1]`, one for uniform and zero for fully
peaked; one minus it flips to a peakedness measure; the `0.5` offset lifts the whole thing so the floor
is always at least gently enforced. At a uniform router `s = 0.5`; at a two-expert collapse
`P = (0.5, 0.5, 0, …)`, `H = log 2 ≈ 0.693` against `log 8 ≈ 2.079`, so `s ≈ 0.5 + 0.667 = 1.167`; at
the top-2-unreachable one-expert limit `s → 1.5`. So `s` runs from `0.5` when healthy to `1.5` at total
collapse, a realistic partial collapse sitting around `1.1`–`1.2`: the hinge idles at half-strength when
the router is fine and surges to triple that as experts start dying. The gate also self-tracks in time
with no extra machinery: at initialization the router is near-uniform, `H` near `log N_E`, `s ≈ 0.5`, so
the hinge idles when nothing has collapsed yet; as the positive-feedback loop peaks the router, `H`
falls and `s` rises, arming the rescue in step with the very collapse it counters; and if the router
re-spreads, `s` relaxes and stands the hinge back down. That is a closed loop on the router's own
peakedness, not a fixed schedule I would have to tune against the run length. The floor is
`τ = 0.064 / N_E`, which for `N_E = 8` is `0.008` — `6.4%` of the uniform share `0.125`, a deliberately
low, survival-level line below which an expert counts as dying, not a uniformity target.

I should be candid about the provenance of these exact constants, because I did not derive the
half-offset, the floor at `0.064/N_E`, or the one-tenth hinge weight from first principles. They are the
artifacts of an evolutionary search over the loss function itself — a program-evolution run that evolved
the Python of the balancing loss, scored by this exact fitness `r = −(L_CE + L_imb)` on real MoE
pretraining, and this is the form it converged to. So the reasoning above is a reconstruction of *why*
the discovered form makes sense, not the path that found it. What I can verify is that each piece plays
the role the mechanism needs: the one-sided hinge targets the tail, the low floor is the survival
threshold, the entropy complement is the collapse gate, the small coefficient keeps it from overwhelming
the global term it sits on. That the gate regulates itself against the state of the very thing it is
trying to fix is the part I find most convincing as more than a lucky constant.

One implementation point decides whether the hinge does anything at all, and it is the same obstacle
that killed the naive count penalty at the first balancing rung. Written on the count `f_i`, the hinge
`max(0, τ − f_i)` is non-differentiable — its gradient in the router weights is the useless zero-or-jump
of a count. So I let `f` decide membership in the under-used set — `under = 1[τ − f_i > 0]`, detached —
and apply the differentiable pressure to the `P` of that set, penalizing `clamp(τ − P_i, 0)` so the
gradient pushes the router's probability mass *up* on exactly the experts the floor test flagged. The
entropy weight `s` is itself detached, modulating the strength without contributing a gradient. That is
what makes the hinge a real training signal.

The cleanest sign the pieces fit is that the endpoint degenerates to the previous rung where it should.
When the router is healthy — every expert above `τ` — the under-set is empty, every `max(0, τ − f_i)` is
zero, and the loss is *identically* the global-batch `f·P` term. So across the whole regime where balance
is already good, the endpoint is the loss from the last rung, and by construction cannot cost
cross-entropy relative to global-batch LBL there — there is no extra term active to cost anything. The
hinge only switches on once some expert has fallen below `0.008`, and even then it *adds* to the global
term rather than replacing it: `clamp` and `max` are both non-negative, so it is a one-way addition that
pushes cold experts up and never pulls healthy ones down. That conditionally-active, one-directional
structure is what lets it sharpen the tail without reopening the specialization wound.

Does it actually beat the global term where it counts, on the dying expert? The hinge's gradient with
respect to a flagged expert's probability is constant — `(0.1/L)·s` in magnitude, since `clamp(τ − P)`
has slope `−1` in `P` below the floor — which with `L = 2` is `0.05·s`. Take an expert genuinely dying
at `f = P = 0.005` in a mildly peaked router where `s ≈ 0.59`: the hinge delivers about `0.030`. The
global `f·P` term on that same expert delivers `8·0.005·|0.005 − m| ≈ 0.0064` with `m` near `0.16`. So
the hinge pushes about `4.6×` harder — and, more importantly than the ratio, its push is *constant* in
`P_j` while the global term's *vanishes* linearly as `P_j → 0`. As the expert nears death the global
term abandons it and the hinge holds its grip at `0.05·s`. The one-tenth coefficient makes the hinge's
*value* tiny — a few thousandths, negligible beside the global term's value of order one — but the value
is not what does the work; the gradient is, and it is concentrated on the two or three flagged experts
at constant magnitude, as large as the per-expert gradient the global term applies to a healthy one.

There is a division of labor between the two terms that is the real reason the low floor works. The
hinge does not try to carry a dying expert all the way to its fair share `1/N = 0.125`; it only lifts it
off the floor, and once the expert climbs above `τ = 0.008` the count test releases it. But that is not
abandonment, because at `P_j ≈ 0.008` the global term's gradient `N P_j(f_j − m)` is no longer vanishing
— about `0.008` and rising — whereas at `P_j ≈ 0.001` it was around `0.001`, useless. So the two terms
hand off cleanly at the floor: below it the smooth term's push has collapsed to nothing and the hinge's
constant `0.05·s` does the rescuing; at and above it the smooth term has recovered and integrates the
expert the rest of the way to its fair share. The hinge is a defibrillator, not a life-support machine —
it restarts an expert the global term let flatline, and then the global term, which works perfectly well
once there is a pulse, takes over. That is exactly why `τ` can sit so low: it only needs to be at the
boundary where the smooth gradient becomes usable again, and `6.4%` of fair share is comfortably inside
it.

So the rung is the discovered endpoint: keep the global-batch term unchanged and add the entropy-gated
under-utilization hinge on top, floor `τ = 0.064/N_E`, weight `s(P_ℓ) = 0.5 + (1 − H(P_ℓ)/log N_E)`,
hinge coefficient `0.1/L`, its gradient routed through the under-used experts' probabilities. My
falsifiable expectations are two. The cross-entropy should stay near `3.7279`, where the global-batch
loss had it — the hinge is idle unless an expert falls below `0.008`, so in a healthy router it barely
fires; if instead CE rose, the gate failed and the hinge is flattening a healthy router. And the
imbalance should fall below the loss frontier of `0.0561`, because the cold tail is now actively rescued
rather than merely averaged over — though I expect it to land *above* the `0.0160` the loss-free bias
reached, because a smooth gradient loss, even a targeted one, does not shove the hard counts around as
ruthlessly as a direct count controller. Somewhere below `0.0561`, not all the way to `0.0160`, at
unchanged CE, for the best fitness of any pure loss on the ladder. That endpoint is where the ladder
stops among hand-reachable losses; the one variant with still lower imbalance wins through an orthogonal
non-gradient count controller that stacks with this loss rather than competing with it. Among
load-balancing *losses* scored on `r`, the entropy-gated hinge is the endpoint.
