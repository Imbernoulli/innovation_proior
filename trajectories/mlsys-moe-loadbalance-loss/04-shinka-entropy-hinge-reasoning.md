The global-batch loss landed where I expected, and the full table now tells a layered story worth
reading carefully before I decide what to build. Plain global-batch LBL came in at `L_imb = 0.0561`,
a hair better than the Switch loss's `0.0587` — a `1.05×` edge, exactly the marginal separation I
predicted, since the over-constraint tax never surfaced at this scale to give the global scope
something visible to win back. Its cross-entropy, `3.7279`, is a whisker under Switch's `3.7281`, and
its fitness `−3.7840` a touch above. So among the pure differentiable losses the frontier now sits at
`0.0561`, and I should be honest that scope alone bought me almost nothing here beyond a cleaner
principle. The striking row is the other one: global-batch LBL *plus* the loss-free selection bias
drove imbalance to `0.0160`, a `3.5×` cut below plain global, at a cross-entropy of `3.7274` that is
if anything slightly better, for a fitness of `−3.7434` — the best point on the whole ladder so far.
That number is important, but I have to read it for what it is. The bias is a non-gradient hard-count
controller; it balanced the counts more aggressively than any smooth `f·P` gradient because it acts
directly on the hard selection, and it did so as a *complementary* mechanism that stacks with the
loss rather than a loss competing with it. So `0.0160` is not a loss result. The loss frontier is
still `0.0561`, and the bias's `0.0160` is really a piece of information: it proves that in this exact
model, at this cross-entropy, an imbalance down near a sixtieth is physically reachable. There is
`3.5×` of headroom between what the best pure loss achieves and what a hard controller extracts, and
the question this rung asks is whether a smarter *loss* — one that reaches the router's own
probabilities — can recover part of that headroom without borrowing the count controller's machinery.

To know where the headroom is, I go back to exactly what the global loss leaves undone, and I can be
quantitative because I derived the gradient two rungs ago. The push the `f·P` term puts on expert `j`
is `N P_j (f_j − m)`, with `m = Σ_i f_i P_i`; its magnitude scales with `P_j`. So the smooth term
equalizes the *average* usage but treats an expert at a tenth of its fair share the same as one at
nine-tenths — worse than the same, actually, because the `P_j` prefactor means the nearly-dead expert,
the one with `P_j ≈ 0`, gets the *weakest* gradient precisely when its deficit `f_j − m` is largest.
The two small factors multiply and the rescue collapses in the tail. And the count bias does not close
this from the loss side either: it balances the counts by shoving the hard selection around, but it
hands the router *no gradient* about balance through `P` at all — it treats the symptom in the
ranking while leaving the router's own probabilities uninstructed about the dying experts. So neither
existing mechanism resurrects the dying *through the router's probabilities*. That is the specific gap
I want a new loss term to fill: it should single out the under-utilized experts and push specifically
on them, through `P`, and it should do so without the smooth term's fatal habit of easing off exactly
where the expert is coldest. I have to be careful, though, because the obvious ways to build such a
term are exactly the ways to wreck specialization, and I need to see those failures before I trust
whatever survives them.

Let me start from the shape a targeted under-use penalty must have. It should fire on an expert that
has fallen below some floor of usage and leave every expert at or above its fair share completely
alone — a two-sided penalty would go right back to pressuring healthy experts, which is the disease.
That shape is a one-sided hinge: penalize `τ − f_i` only when it is positive, zero otherwise. An
expert comfortably above the floor `τ` contributes nothing; an expert below it contributes in
proportion to how far below, so all the force concentrates on the cold tail — the exact opposite of
the smooth `f·P` term that spreads its attention evenly and thins out in the tail. The floor `τ`
should be a small fraction of the uniform share `1/N`, not `1/N` itself: I do not want to demand every
expert hit its fair share exactly, only that none be allowed to wither below a survival minimum.
Something like a few percent of `1/N`.

But a bare hinge is dangerous, and I have to see the danger before I trust it. If the hinge fires hard
whenever an expert dips below the floor, it will fire even when the router is perfectly healthy and
the dip is nothing but the roughness of a finite batch — a momentary sampling under-count of an
expert that is fine on average. Pushing hard in that situation does exactly what the micro-batch loss
did: it flattens legitimate structure and raises the cross-entropy, chasing noise as if it were
collapse. So the hinge cannot fire unconditionally; it needs a gate that answers one question — is
this a real collapse, or just normal variation? — and only opens when the answer is collapse.

Here is where I have to think, because there are several ways to build that gate or to avoid needing
one, and most of them are traps. The first temptation is to skip the hinge entirely and just crank
the coefficient `α` on the global term — push harder on the same penalty. But I already know this
fails on its own terms: the global gradient on a dying expert is `N P_j(f_j − m)`, and multiplying `α`
scales that whole thing, including the `P_j` prefactor that is already nearly zero for the expert I
care about. Cranking `α` pushes hardest on the experts that need it least and still barely touches the
dying one, while raising the pressure on the healthy experts enough to start costing cross-entropy. It
is uniform pressure applied to a non-uniform problem. The second temptation is a symmetric penalty on
the spread of `f` — something like `Σ_i (f_i − 1/N)^2` routed through `P`. But symmetric means it
pays equal attention to over-users and under-users, so it spends half its budget suppressing hot
experts I do not need suppressed, and near the tail its gradient still fades the same way the linear
term's does. The third temptation is the interesting one, and it is about entropy. The router's
entropy `H(P)` is a natural single number for "how collapsed is this router" — low when the mass is
piled on a few experts, high when it is spread — so why not just add a term that *maximizes* it, i.e.
penalize low entropy directly? Because maximizing router entropy pushes *every token's* probability
toward uniform. That is per-token flattening, which is precisely the specialization-destroying move I
have been avoiding since the micro-batch rung; entropy-as-an-objective would manufacture balance by
making the router indecisive, the worst possible trade. So entropy is the wrong thing to *optimize*.
But that failure is also the clue. Entropy is an excellent thing to *read*: a peaked, low-entropy
router is the regime where experts genuinely starve and the hinge should fire hard; a near-uniform,
high-entropy router is healthy and the hinge should stay quiet. So I do not add entropy to the loss —
I use it as the gate. The hinge is weighted by a function of the router's entropy that is large when
the router is peaked and small when it is spread, so the targeted rescue is armed exactly when
collapse is happening and disarmed when it is not.

Concretely the gate is `s(P_ℓ) = 0.5 + (1 − H(P_ℓ) / log N_E)`. I want to read every piece of that.
Normalizing `H` by `log N_E` — the entropy of the uniform distribution, the maximum possible — puts
the ratio in `[0, 1]`, one for a uniform router and zero for a fully peaked one. Taking one minus that
flips it so peaked routers score near one and uniform routers near zero: a raw peakedness measure. The
`0.5` offset then lifts the whole thing so the weight is never quite zero — the floor is always at
least gently enforced even in a healthy router, rather than switched fully off. Let me check the range
by hand. At a uniform router `H = log N_E`, so `s = 0.5 + (1 − 1) = 0.5`. At a two-expert collapse,
`P = (0.5, 0.5, 0, …)`, `H = log 2 ≈ 0.693` against `log 8 ≈ 2.079`, so `H/log N_E ≈ 0.333` and
`s ≈ 0.5 + 0.667 = 1.167`. At the (top-2-unreachable) one-expert limit `H → 0` and `s → 1.5`. So `s`
runs from `0.5` when healthy to `1.5` at total collapse, a factor of three of modulation, with a
realistic partial collapse sitting around `1.1`–`1.2`. That is exactly the behavior I wanted: the
hinge, scaled by `s`, is a rescue that idles at half-strength when the router is fine and surges to
triple that as the router peaks and experts start dying.

The gate also has a temporal behavior I like, which falls out of the same formula with no extra
machinery. At initialization the router is essentially random, so its probability is near-uniform,
`H` near `log N_E`, and `s ≈ 0.5` — the hinge idles, which is right, because at the start no expert
has collapsed yet and there is nothing to rescue. As training proceeds and the positive-feedback loop
begins peaking the router, `H` falls and `s` rises toward `1.5`, so the hinge arms itself
automatically in step with the very collapse pressure it exists to counter. And if the combined loss
succeeds and the router re-spreads, `H` climbs back and `s` relaxes toward `0.5` again, standing the
hinge back down. So the gate is not a fixed schedule I would have to tune against the run length; it
is a closed loop on the router's own peakedness that ramps the rescue up exactly as fast as collapse
sets in and disarms it when balance returns. That self-tracking is something a hand-set constant on
the hinge could never do, and it is the part of the discovered form I find most convincing as more
than a lucky number — the search did not just find good constants, it found a term that regulates
itself against the state of the very thing it is trying to fix.

The floor itself the search fixed at `τ = 0.064 / N_E`, which for `N_E = 8` is `0.008` — eight
thousandths of the slots, i.e. `6.4%` of the uniform share `0.125`. So the hinge only flags an expert
once it has fallen below about a sixteenth of its fair share; anything healthier is left entirely
untouched. That is a deliberately low, survival-level floor, not a uniformity target — it is the line
below which an expert counts as dying, and it is what keeps the hinge silent in a healthy router where
no expert is anywhere near `0.008`.

I should be candid about the provenance of these exact constants, because I did not derive the
half-offset, the floor at `0.064/N_E`, or the one-tenth weight on the hinge from first principles.
They are the artifacts of an *evolutionary search over the loss function itself*: a program-evolution
run evolved the Python of the balancing loss, scored by the very fitness I am using here,
`r = −(L_CE + L_imb)`, on real MoE pretraining, and this is the form it converged to. So the reasoning
above is a reconstruction of *why* the discovered form makes sense, not the path that found it. What I
can verify is that each piece plays the role the mechanism needs — the one-sided hinge for targeting
the tail, the low floor for the survival threshold, the entropy complement for the collapse gate, the
small coefficient for not overwhelming the global term it sits on. The whole loss is the global-batch
`f·P` term, kept with its scale-free `N_E` factor and averaged over the layers, plus this
tenth-weighted, entropy-gated under-use hinge, also averaged over the layers.

One implementation point decides whether the hinge does anything at all, and it is the same obstacle
that killed the naive count penalty back at the first balancing rung. Written on the count `f_i`, the
hinge `max(0, τ − f_i)` is non-differentiable — its gradient in the router weights is the useless
zero-or-jump of a count. So as a literal function of `f` it would be a decorative zero. The count can
only *select* which experts are under the floor; the gradient that actually raises a cold expert's
usage has to flow through the differentiable probability `P_i` of those selected experts. So I let `f`
decide membership in the under-used set — `under = 1[τ − f_i > 0]`, detached — and apply the
differentiable pressure to the `P` of that set, penalizing `clamp(τ − P_i, 0)` so the gradient pushes
the router's probability mass *up* on exactly the experts the floor test flagged as dying. The
entropy weight `s` is itself detached, so it modulates the strength of the hinge without contributing
a gradient of its own. That is what makes the hinge a real training signal.

Before I trust the two-term loss I want to check its limits, because the cleanest sign that the
pieces fit is that the endpoint degenerates to the previous rung exactly where it should. When the
router is healthy — every expert above the floor `τ` — the under-set is empty, every `max(0, τ − f_i)`
is zero, and the hinge contributes nothing at all: the loss is *identically* the global-batch `f·P`
term. So across the entire regime where balance is already good, the endpoint is the loss from the
last rung, which means by construction it cannot cost cross-entropy relative to global-batch LBL in
that regime — there is simply no extra term active to cost anything. The hinge only switches on once
some expert has actually fallen below `0.008`, and even then it *adds* to the global term rather than
replacing it: `clamp` and `max` are both non-negative, so the hinge is a one-way addition that can
only push cold experts up, never pull healthy ones down. That conditionally-active, one-directional
structure is exactly what lets it sharpen the tail without reopening the specialization wound — in the
healthy regime it is invisible, and only in the collapse regime does it become the one term pushing
hard on the right experts.

Let me verify that this actually beats the global term where it counts, on the dying expert, rather
than just asserting it. The hinge's gradient with respect to a flagged expert's probability is
constant — `(0.1/L)·s` in magnitude, since `clamp(τ − P)` has slope `−1` in `P` below the floor — and
with `L = 2` that is `0.05·s`. Take an expert that is genuinely dying at `f = P = 0.005`, below the
floor, in a mildly peaked router where `s ≈ 0.59`: the hinge delivers a push of about `0.05·0.59 ≈
0.030`. The global `f·P` term, on that same expert, delivers `N P_j |f_j − m| ≈ 8·0.005·|0.005 − m|`,
which with `m` near `0.16` for such a router works out to about `0.0064`. So the hinge pushes about
`4.6×` harder on the dying expert than the global term does — and, more importantly than the ratio,
its push is *constant* in `P_j` while the global term's *vanishes* linearly as `P_j → 0`. As the
expert gets closer to death the global term abandons it and the hinge holds its grip at `0.05·s`. That
is the whole point of the construction in one comparison: a floor-restoring force that does not fade
where the smooth term fades.

It might look worrying that the hinge carries a coefficient of only one-tenth against the global
term's `N_E = 8`, as if it were too faint to matter. But the coefficient governs the penalty's
*value*, and the value is not what does the work. With a couple of experts flagged, the hinge's
contribution to the loss is about `0.1 · s · Σ_i clamp(τ − P_i)`, and since each flagged term is at
most `τ = 0.008`, that total is on the order of a few thousandths — utterly negligible beside the
global term's value of order one. What matters is the gradient, and the gradient is concentrated:
spread over only the two or three experts actually below the floor, at constant magnitude `0.05·s ≈
0.06` each, which is as large as or larger than the per-expert gradient the global term applies to a
healthy expert. So the hinge is value-tiny and gradient-sharp — it barely moves the number being
minimized while delivering a strong, focused push to a handful of dying experts. That is precisely how
a small-coefficient term can rescue the tail without perturbing the global balance it rides on, and it
is why the search could afford to keep its weight so low: a larger coefficient would not have rescued
the tail any better (the gradient is already decisive on the flagged set) and would only have risked
the value of the hinge starting to distort the loss landscape elsewhere.

There is a division of labor between the two terms that I think is the real reason the low floor
works, and it is worth spelling out. The hinge does not try to carry a dying expert all the way to
its fair share `1/N = 0.125`; it only lifts it off the floor. Once an expert climbs above `τ = 0.008`
the count test releases it and the hinge falls silent on it. But that is not abandonment, because at
`P_j ≈ 0.008` the global term's gradient `N P_j(f_j − m)` is no longer vanishing — it is about `0.008`
in magnitude and rising — whereas at `P_j ≈ 0.001`, near death, it was around `0.001`, useless. So the
two terms hand off cleanly at the floor: below it the smooth term's push has collapsed to nothing and
the hinge's constant `0.05·s` does the rescuing; at and above it the smooth term's push has recovered
and integrates the expert the rest of the way from the floor back up to its fair share. The hinge is
a defibrillator, not a life-support machine — it restarts an expert the global term has let flatline,
and then the global term, which works perfectly well once there is a pulse, takes over. That is
exactly why `τ` can be set so low: it does not need to be anywhere near `1/N`, it only needs to sit at
the boundary where the smooth gradient becomes usable again, and `6.4%` of fair share is comfortably
inside that boundary.

On the fitness itself, plain global sat at `r = −3.7840`. If the hinge holds the cross-entropy near
`3.7278` and presses imbalance down into the low hundredths, well under `0.0561`, then `r` climbs into
the `−3.75` neighborhood — a clear win over global as a loss — and the closer the hinge drives the
tail toward what the count controller managed, the closer `r` creeps toward the `−3.74`s. What it will
not do standalone is beat the bias-stacked `−3.7434`, because that point is a loss *plus* an
orthogonal controller and this rung is a loss alone; the fair comparison is against the other losses,
where the hinge should take the frontier. And the honest tell, as ever, is the decomposition: the gain
must live in `L_imb` with `L_CE` unmoved. If imbalance improved but cross-entropy crept up to pay for
it, the entropy gate would have failed to keep the hinge idle in the healthy directions, and I would
rather catch that than bank the headline `r`.

So the rung is the discovered endpoint: keep the global-batch term unchanged and add the
entropy-gated under-utilization hinge on top, with the floor `τ = 0.064/N_E`, the weight `s(P_ℓ) =
0.5 + (1 − H(P_ℓ)/log N_E)`, and the hinge coefficient `0.1/L`, its gradient routed through the
under-used experts' probabilities. My falsifiable expectations are two. The cross-entropy should stay
where the global-batch loss had it, near `3.7279` — the hinge is idle unless an expert falls below
`0.008`, so in a healthy router it barely fires and should cost no specialization; if instead CE rose,
the gate failed and the hinge is flattening a healthy router. And the imbalance should fall below the
loss frontier of `0.0561`, because the cold tail is now actively rescued rather than merely averaged
over — though I expect it to land *above* the `0.0160` the loss-free bias reached, because a smooth
gradient loss, even a targeted one, does not shove the hard counts around as ruthlessly as a direct
count controller does. Somewhere below `0.0561`, not all the way to `0.0160`, at unchanged CE, for the
best fitness of any pure loss on the ladder: that is the prediction. And this is where the ladder
stops among hand-reachable losses. This is the form a dedicated program-evolution search discovered
against this exact fitness, on a `556M`-parameter, `82M`-active MoE over two-billion-plus tokens; my
run reproduces its mechanism and its ordering, not its scale. The one variant with still lower
imbalance, the global loss paired with the loss-free bias, wins through a non-gradient hard-count
selection controller — an orthogonal mechanism that stacks with this loss rather than competing with
it. Among load-balancing *losses* scored on the fitness `r`, the entropy-gated hinge is the endpoint.
