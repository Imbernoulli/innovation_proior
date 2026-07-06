The lexicase numbers half-confirm the bet and half-refute it, and the split is exactly the place to push
next. The thing I most wanted is there: Nguyen-7's −1.0 catastrophe is gone. All three seeds are
positive — 0.986, 0.946, 0.972 — and the mean jumped from standard GP's 0.330 to 0.968. That is the
premature-convergence cure working as predicted: spreading selection pressure across cases kept
alternative forms alive, so no single early winner locked the population onto a form that diverges off
the test range. But I want to read the Nguyen-7 recovery more carefully than "it worked," because the
per-seed deltas against standard GP say something specific. The two already-solved seeds actually
slipped a hair under lexicase — 0.998 → 0.986 and 0.991 → 0.946, down 0.012 and 0.045 — while the
detonating seed recovered from −1.0 to 0.972, up nearly 2.0. So the entire mean gain is the one
catastrophic seed being rescued; on the seeds that had already found the right form, lexicase cost a
little. That small tax on good seeds is a thread I will pull in a moment. Koza-3 the control mostly held
— seed 42 at 0.99998 — but it actually *slipped* on the two harder seeds (0.908, 0.859, mean 0.922, down
from standard GP's 0.993), and the drops are large relative to Koza-3's own tightness: standard GP's
three seeds spanned only 0.012, whereas these two fell by 0.090 and 0.135, an order of magnitude past
that noise floor, so this is a real signal I should not wave away. And Nguyen-10, where I predicted
lexicase would preserve the sin/cos specialists and lift the mean above 0.588, is the disappointment:
seed 42 is a perfect 1.0, but 123 and 456 rose only to 0.643 and 0.500, and the mean came in at 0.714.
Every seed improved — good — but the two weak ones plateau well short, and the floor (a 0.500 seed) is
still low.

So I read three things together — Koza-3's real slip, Nguyen-7's small tax on its good seeds, and
Nguyen-10's capped recovery — because I think they have one cause. Lexicase's case-by-case filtering
does everything I asked of it on the *diversity* axis: it stops the population collapsing. But it does
nothing whatever about *size*, and it may make size worse. By rewarding specialists on individual cases
it actively keeps around large, contorted trees that happen to nail one nasty case, and a tree selected
through a long conjunction of cases is under no pressure to be small. Watch what that does over 50
generations. The mean tree size drifts upward generation after generation — programs accumulate inert
subexpressions, "introns," chunks that do not change the output — because once the population is near a
fitness plateau, larger trees are more likely to survive crossover unharmed (a random crossover point
lands in the padding, not the working core), so selection quietly rewards size for its own protective
value. Bloat does three bad things at once: it slows evaluation (surely part of why lexicase's
wall-clock blew up), it makes the final expression an unreadable monster, and — the part that shows up
in these numbers — it *hurts generalization*. A bloated tree overfits the training sample with
high-order wiggle and then fails on the held-out grid. Koza-3 trained on 20 points and tested on 100 in
the same range: a bloated tree that threads the 20 training points can still miss between them, exactly
the 0.908/0.859 slip. Nguyen-10 trained on 100 and tested on 400: same overfitting, worse because the
target is harder, exactly the 0.643/0.500 plateau. And even Nguyen-7's small tax on its good seeds fits:
the trees that found the right log form under lexicase were bulkier than they needed to be, so they
extrapolate a shade worse than a compact version would. One mechanism — unmanaged growth — explains the
control's slip, the sag on the hardest benchmark, and the little tax on the easy wins. The lever is not
selection-for-diversity anymore; it is *complexity control*.

Before I commit to a penalty, let me lay out the levers that are actually available and knock down the
ones that do not fit, because "control complexity" admits several moves and most are wrong here. The
first is the blunt one: tighten the depth cap. But bloat is a *size* problem, not a depth problem — a
tree at depth 8 already admits up to `2⁹ − 1 = 511` nodes, and I could halve the depth to 6 and still
carry 127 nodes of padding, while forbidding legitimately deep forms a hard target may need. A static
depth limit cannot tell productive depth from inert bulk, so it is the wrong axis. The second is a fixed
size penalty `c`, a constant I set by hand; I will come back to why the constant is a trap, but the
short version is that the right value is problem- and generation-dependent and I refuse to tune it three
ways. The third is genuinely tempting: a multi-objective, Pareto approach that treats size and error as
two objectives and selects on non-dominated fronts. It is the principled thing, but it does not fit this
contract cleanly — `evolve_one_generation` receives a flat scalar fitness list and carries no persistent
front state across generations, so I would rebuild a non-dominated sort from scratch every call and
replace the whole selection mechanism, when my diagnosis is specifically that *one* scalar nuisance,
size, needs canceling out of an otherwise-working selector. That is a heavy intervention where a
targeted one will do. The fourth is the one I have to think hardest about: keep lexicase, which just
cured convergence, and *add* parsimony on top. Here I hit a mechanical wall. Parsimony pressure is a
modification of the *scalar* fitness — I want to add `c·l` to each tree's number and let selection see
the penalized number. Lexicase selection does not consume a scalar at all; it filters on the per-case
error matrix, and there is nowhere to inject `c·l` into a per-case gate short of adding a size term to
every case's error, which turns lexicase into a different algorithm. So parsimony and lexicase are not
simply stackable in selection: to apply a size penalty I need a selector that consumes a scalar, and
that is the tournament. Returning to the size-7 tournament is therefore *forced* by the mechanism, not
merely chosen for speed — though it does also reclaim lexicase's `O(P²N)` cost down to `O(Pk)`, and it
turns this rung into a clean test of the bloat hypothesis: if diversity were still the binding
constraint, dropping lexicase would regress Nguyen-7 back toward a catastrophe and the numbers would
say so. One more class is quietly ruled out by the substrate itself: any scheme that needs persistent
per-individual state across generations — genotypic age of a lineage, an accumulated novelty score —
cannot be expressed, because `Node`'s `__slots__ = ('value','children')` blocks attaching fields and
`evolve_one_generation` sees only the current trees and their fitnesses. Whatever I do must be
recomputable from the current population alone.

So I start from the failure I can actually observe and make it precise, because "penalize big trees" is
easy to say and easy to get wrong. After a quiet opening where average size barely moves, the mean node
count starts climbing generation after generation while fitness stops improving. I must not confuse this
with legitimate growth — a hard target may genuinely need a larger expression. The bad case is growth
that buys *no* fitness. The old practical answer is to make large programs pay rent: select on a
penalized fitness instead of raw fitness. In a maximized-fitness convention that is `f_p(x) = f(x) −
c·l(x)`, where `l(x)` is program size and `c` is a pressure against length; the raw `f` still matters
for recognizing a solution, the penalty only steers selection. Sensible as a soft MDL-flavored
tradeoff, but it leaves the one question that matters: what is `c`? Too small and bloat continues; too
large and GP treats shrinking as the real objective and collapses toward tiny useless programs. Worse,
the right value changes with the problem, the primitive set, the population, the selection scheme, and
even the generation. Adapting `c` over time helps, but still does not tell me what value gives a
*specified* change in mean size. I want to *compute* the pressure from the current population, not tune
it — and especially not tune it three different ways for three benchmarks.

So I ask what actually determines the expected change in mean size. The size evolution equation is the
starting point. For selection followed by symmetric subtree crossover — crossover that does not favor
one parent order over the other — the expected next mean size depends only on which length classes
selection picks: `E[μ(t+1)] = Σ_l l·p(l,t)`, where `p(l,t)` is the probability a selection event
chooses a size-`l` program. If `Φ(l,t)` is the current population fraction at size `l`, the current
mean is `μ(t) = Σ_l l·Φ(l,t)`, so `E[Δμ] = Σ_l l·(p(l,t) − Φ(l,t))`. This already names the lever:
crossover, under the symmetry condition, does not move the mean in expectation — the mean moves *only*
because selection over- or under-samples length classes. If long programs are picked more than their
share, the mean grows; if short ones are favored, it shrinks. To stop bloat I have to make selection
*neutral* with respect to the length advantage it is currently exploiting.

To turn that into an equation for fitness I first use fitness-proportionate selection, because there
`p(l,t)` has a closed form: `p(l,t) = Φ(l,t)·f̄(l,t)/f̄(t)`, with `f̄(l,t)` the average fitness among
size-`l` programs and `f̄(t)` the population mean. Substituting gives `E[Δμ] = (1/f̄)·Σ_l l·(f̄(l,t) −
f̄(t))·Φ(l,t)`. This looks almost like a covariance, except a covariance wants `l − μ` not `l`. I check
whether inserting `l − μ` changes anything: split `l = (l − μ) + μ`. The first piece gives `Cov(l,f)`;
the second is `μ·Σ_l (f̄(l,t) − f̄(t))·Φ(l,t)`, and that population-weighted deviation of size-class
fitness from the population mean is `f̄ − f̄ = 0`. The cross-term vanishes for a precise reason, not by
handwaving, and I am left with `E[Δμ] = Cov(l,f)/f̄`. That is Price's theorem in the language of program
size: the expected one-generation change in the mean of a heritable feature is its covariance with
fitness over mean fitness. The operational reading is the whole point — a positive covariance between
length and fitness means selection has a length advantage to exploit, so the mean grows. That is bloat,
named exactly.

Let me pin the identity down on a concrete population, because I am about to lean my whole coefficient on
it and I want to see it hold, not just believe it. Take three programs with sizes `l = (2, 4, 6)` and
maximized fitnesses `f = (1, 2, 3)` — bigger trees fitter, the bloat setup. The mean size is `μ = 4` and
mean fitness `f̄ = 2`. Under proportionate selection the pick probabilities are `f/Σf = (1/6, 2/6, 3/6)`,
so the expected selected size is `(1/6)·2 + (2/6)·4 + (3/6)·6 = 0.333 + 1.333 + 3.0 = 4.667`, giving a
direct `E[Δμ] = 4.667 − 4 = +0.667`. Now the theorem: `Cov(l,f) = mean[(l−μ)(f−f̄)] = mean[(−2)(−1),
(0)(0), (2)(1)] = mean[2, 0, 2] = 1.333`, and `Cov(l,f)/f̄ = 1.333/2 = 0.667`. The two agree exactly —
the direct simulation of one selection step and Price's formula land on the same +0.667. The mean grows,
and the theorem says by how much. Now bring back the penalty and watch it cancel: with the linear
`g = c·l`, `Cov(l, f − c·l) = Cov(l,f) − c·Var(l)`, and `Var(l) = mean[(l−μ)²] = mean[4,0,4] = 2.667`,
so the covariance vanishes at `c = Cov(l,f)/Var(l) = 1.333/2.667 = 0.5`. Check it: the penalized
fitnesses are `f − 0.5·l = (1−1, 2−2, 3−3) = (0, 0, 0)`, whose covariance with length is trivially zero,
so `E[Δμ] = 0`. Setting `c` to that exact ratio flattened the size-fitness relationship to nothing and
froze the expected mean size. The toy did what the algebra promised.

That generalizes: bring the penalty back without assuming it is linear, `f_p(x,t) = f(x) − g(l(x),t)`
for any size function `g`. The same identity gives `E[Δμ] = Cov(l, f − g)/(f̄ − ḡ) = (Cov(l,f) −
Cov(l,g))/(f̄ − ḡ)`. For *no* expected growth I set `E[Δμ] = 0`, and (provided the denominator is
nonzero) the condition is simply `Cov(l,g) = Cov(l,f)`. The penalty must have exactly the same
covariance with size as the raw fitness currently has — the mean-fitness denominator drops out because I
am forcing the numerator to zero. Pick the traditional linear penalty `g = c·l`: then `Cov(l,g) =
c·Cov(l,l) = c·Var(l)`, and setting that equal to `Cov(l,f)` forces `c = Cov(l,f)/Var(l)`. That is the
coefficient for zero expected size change. It is dynamic — both pieces change during the run — and it has
a clean reading: it is the ordinary-least-squares slope of fitness against size, the per-node fitness
advantage selection currently sees. Subtracting that slope times length removes the linear size-fitness
advantage from the selection signal. No tuning, no per-benchmark constant — the population measures the
pressure it needs.

I keep the selection-scheme boundary honest, because this rung does not use proportionate selection. The
exact covariance derivation assumes fitness-proportionate `p(l,t)`. Tournament selection does not give
that closed form, so `c = Cov(l,f)/Var(l)` is not an exact theorem for a tournament. But the ratio still
*estimates* the linear size-fitness slope in the current population, and tournament selection amplifies
the same ordering — it just sharpens the preference for whoever has the better penalized value.
Cancelling that slope is therefore the natural practical coefficient, and it carries over without
changing the operator, which matters because I am keeping the size-7 tournament and the standard
uniform-point subtree crossover and mutation from the previous rungs. I am changing *only* what number
selection sees. There is one small numerical honesty check on the estimator: in the scaffold `np.cov`
divides by `n − 1` while `np.var` divides by `n`, so the ratio I compute is the true OLS slope times
`n/(n−1)` — at `pop_size = 500` a factor of `500/499 ≈ 1.002`, a 0.2% inflation that is far inside the
clamp I am about to add and beneath the run-to-run noise in the covariance itself, so I leave it.

Now translate to this scaffold's conventions, which is where the signs and the clamps live. The theory
used maximized fitness and wrote `f − c·l`; the scaffold gives MSE, where *lower* is better. Under that
convention, making large trees worse means selecting on `penalized = MSE + c·l` — the penalty is
*added*, and selection minimizes. I compute the coefficient exactly as the automatic rule: `auto_c =
Cov(length, raw_MSE)/Var(length)`, from the current lengths and the raw-MSE vector the function
receives, guarding the degenerate zero-variance case (`c = 0`). Then I apply two clamps to `[0, 0.001]`,
and the lower clamp is doing more than guarding a pathology — it is the switch that makes the penalty
fire only when it should. Read the sign of `auto_c`: it is positive exactly when `Cov(length, MSE) > 0`,
i.e. when larger trees carry *higher* MSE — growth that has stopped buying fit, which is bloat — and
there the added `c·l` leans selection off the larger trees. It is negative exactly when larger trees fit
*better*, growth that is earning its keep, and there flooring `c` to 0 declines to fight productive
growth (and never rewards it, since a negative `c` under lower-is-better would subtract from big trees'
penalized scores and actively subsidize bloat). So the floor at 0 turns the sign of the size-fitness
covariance into a bloat detector: brake when growth is inert, stand down when it is productive. The
upper clamp at 0.001 bounds the other tail. It is a mild ceiling: `c = 0.001` means a 100-node tree pays
at most `0.001·100 = 0.1` in added MSE, and on a benchmark like Koza-3 where fits are uniformly good and
the size-MSE covariance is weak, `auto_c` sits far below that cap and the ceiling never binds; it exists
only for the pathological generation where a spurious size-fitness correlation spikes `auto_c` and,
unclamped, would make selection ignore fit entirely and collapse the population to stubs. A quick
dimensional check confirms the object is sane: `c = Cov(l, MSE)/Var(l)` has units of MSE per node, so
`c·l` is in MSE units, addable to MSE, and `c` reads as the per-node MSE cost the population currently
shows. Critically, the penalty lives in **selection only**. The raw MSE stays the truth for elitism, for
the outer loop's best-tree tracking, and for the final report — `fitness_function` returns raw MSE
unchanged, the elite carried forward each generation is the raw-MSE best, and only the penalized vector
`MSE + c·l` is handed into the tournament. If I baked the penalty into the fitness, the loop would
preserve and report a tree chosen for being small rather than for fitting, which is exactly the collapse
I am clamping against. The per-generation loop is then: compute the lengths and the penalized vector,
keep the raw-MSE elite, and breed by tournament on the penalized vector plus the unchanged
crossover/mutation/reproduction split. The full scaffold module is in the answer.

Now the falsifiable expectations against the lexicase numbers, because this rung has to clear a specific
bar. The mechanism is the OLS slope of fitness on size cancelled out of selection, so the direct
prediction is that mean tree size stops drifting up — and the *downstream* prediction, the one the
leaderboard sees, is that controlling bloat recovers the generalization that bloat was costing. So I
expect Koza-3 to come *back up* from lexicase's 0.922 toward standard GP's 0.99 and beyond — the slip
was overfitting introns, and removing them should restore the clean polynomial fit; if Koza-3 does *not*
recover, the bloat story is wrong and the slip had another cause. I expect Nguyen-10 to improve on
lexicase's 0.714, because the 0.643 and 0.500 seeds were, on this reading, bloated overfits of the
sin·cos product; a parsimony-pressured search should find a tighter, more generalizable form and lift
the mean and the floor. Nguyen-7 is the interesting one: lexicase already cured its catastrophe to a
0.968 mean, and parsimony does *not* carry lexicase's diversity machinery, so the open question is
whether plain tournament + parsimony can hold Nguyen-7 near 0.97 without lexicase's per-case
protection. My expectation is yes — that the −1.0 blow-up was a depth/divergence problem as much as a
diversity problem, and that a compact, parsimony-controlled tree is *less* prone to diverging off the
test range than a bloated one, so Nguyen-7 should stay high (near or above 0.985) with the smaller trees
extrapolating more gracefully; the small tax lexicase levied on its good seeds should even reverse, since
those seeds now breed compacter logs. If all three hold, this rung should sit clearly above lexicase on
the overall mean — and it should do so *fast*, because parsimony keeps trees small and `O(Pk)` tournament
selection is orders of magnitude cheaper than lexicase's `O(P²N)`, so I expect both a higher mean and a
dramatically lower wall-clock than the previous rung. The one outcome that would refute the whole chain
is Nguyen-10 staying stuck near 0.7 while Koza-3 recovers: that would mean bloat was never the Nguyen-10
bottleneck, and the cap there is raw representational reach — the difficulty of assembling the right
two-variable product under this depth limit — which no selection-side fix can lift. That failure mode is
plausible on its own terms: Nguyen-10 needs a `sin(x)` subtree and a `cos(y)` subtree *co-present* in one
tree and joined under a `mul`, all inside the depth-8 cap, a simultaneous two-factor assembly strictly
harder than Nguyen-7's single univariate log, and parsimony only reshapes which trees selection prefers —
it cannot manufacture a factor the population never manages to build. So if Nguyen-10 stalls while
Koza-3 climbs, I will read it not as a failure of the parsimony coefficient but as a signal that the next
lever has to act on the operators or the representation, not on selection at all.
