The standard-GP numbers tell me exactly one story, and I want to read it precisely before I change
anything. Koza-3 is solved: 0.985, 0.998, 0.994 across the three seeds, mean 0.993 — the reachable
polynomial behaves exactly as I predicted, steady and high, no drama, with a seed-to-seed range of only
0.012. The drama is entirely on the two transcendental benchmarks, and it is *variance*, not a
uniformly low mean. Nguyen-7 has two seeds at 0.998 and 0.991 — essentially solved, they average 0.994 —
and then seed 456 at −1.0: a complete blow-up, the wrong form found early, diverging off the wider test
range badly enough that R² went negative and the floor caught it. The arithmetic makes the point sharp:
those two good seeds alone would put the mean at 0.994, and the single catastrophic seed drags it to
0.330, a cost of 0.665 in mean R² carried by one run in three. So Nguyen-7's problem is not that the
engine cannot fit logarithms — two thirds of the time it fits them almost perfectly — it is that one
seed in three detonates. Nguyen-10 is the same disease in a milder, more pervasive form: 0.884, 0.557,
0.322 — no single catastrophe, but a steady slide across seeds, mean 0.588, spanning a range of 0.56, as
if the search keeps locking onto partial fits of the sin·cos product and never recovers the rest. Two
different textures — one bimodal (solved or detonated), one a graded sag — but the same underlying
mechanism: with nothing but raw-MSE tournaments, one lineage that gets an early lead on the training
sample dominates the tournaments, the population homogenizes onto its *form*, and on the transcendental
targets that form is wrong often enough that the seed-to-seed outcome is a coin flip — solved, or stuck.
The lever the diagnosis points at is selection: the failure is *which* lineage wins, so I should
change what selection rewards.

Let me look hard at what tournament selection actually does, because the flaw is structural, not a
matter of tuning `k`. Each individual has a vector of per-case errors — one absolute error
`e_t(i) = |y_t − ŷ_t(i)|` for each of the `N` training points — and `fitness_function` crushes that
whole vector into a single scalar, the mean. Tournament then selects on that scalar. But the entire
premise of GP is that a full solution is *assembled* from partial solutions: subprograms that each get
some region of the problem right, recombined by crossover until one tree gets all of it right. The
instant I average the error vector into one number, I delete *which* regions a tree is good at. Two
trees can have the identical mean error with completely different profiles: one mediocre everywhere, the
other the population's *best* on a cluster of hard cases and bad on the rest. For harvesting a building
block the second is gold and the first is nothing — and tournament cannot tell them apart. Worse, it
actively buries the specialist: its few bad cases inflate its mean, so against a generalist with lower
mean it almost never wins. On Nguyen-10 this is exactly what I think is happening, and I can name the
specialist concretely. The target is `2·sin(x)·cos(y)`; a tree that has assembled the `sin(x)` factor
but not yet the `cos(y)` modulation is *excellent* on the slice of training points where `cos(y) ≈ 1`
(the `y ≈ 0` and `y ≈ 2π` rows) and *bad* where `cos(y) ≈ 0` (the `y ≈ π/2` rows). That is precisely a
case specialist — best-in-population on a subset of the 100 cases — and it is exactly the half-built
factor crossover needs to splice with a `cos(y)` specialist to finish the product. But its bad rows
inflate its mean, so tournament selects it out before crossover ever gets to combine the two. The
information I most need to escape premature convergence — the *shape* of the error vector, not its
average — is the information I throw away before selecting.

So the question is: what if I do not aggregate? What pressure do I actually want? I want to reward an
individual for being uniquely good on *some* part of the problem, especially the *hard* parts — the
cases few others get right — because those are where I am missing a building block. There is prior art
reaching for this. Implicit fitness sharing (McKay; Krawiec & Nawrocki) rewards solving cases few
others solve; historically-assessed hardness (Klein & Spector) scales each case's error by the
population's success on it; co-solvability (Krawiec & Lichocki) rewards solving *pairs* of cases
together. All have the right instinct — reward the rare and hard — but stare at what they do at the end:
they all *sum* the per-case rewards into one number and then select on that number. I have just chosen
a fancier scalar. I still cannot reward a unique *combination* of cases, because the combination is
gone the moment I add the terms up. The aggregation is the disease; these treat symptoms.

What would it mean to select on the combination without aggregating? Do not score-then-pick — *filter*.
Treat each case as a gate. Start with the whole population as the candidate pool. Look at one case and
throw out everyone who is not the best on it — keep only the elite on that single case. Then look at
another case and, among the survivors, throw out everyone not best on *that* case. Repeat. Each gate is
the simplest possible test — "are you the best left, on this case?" — no averaging anywhere, just a
chain of elitism filters. Run enough gates and the pool shrinks to one, and that is my parent. And the
individual that survives a long chain is elite on a *conjunction* of cases — exactly the combination no
sum could express. The order of the gates matters enormously: a fixed order makes the first case a
permanent dictator, so I shuffle the cases freshly for *every* parent-selection event. Then the
first gate — the one with the most filtering power, acting on the whole population — is a different case
each time, and over many selections every case gets its turn at being first. The beautiful part: a
case's filtering strength is proportional to its *difficulty*. An easy case almost everyone ties on
barely shrinks the pool; a hard case where only one or two individuals reach the best error slices the
pool to those few — and hands selection straight to the specialist on that hard case. Difficult cases
automatically exert more pressure, toward exactly the specialists I want to propagate. This is lexicase
selection, and there is a diversity bonus for free: because each parent comes down a different random
conjunction of cases, the population spreads across behavior space instead of collapsing onto one
lineage — which is the direct antidote to the premature convergence that produced standard GP's −1.0
seed and its Nguyen-10 slide.

A four-tree, three-case toy makes the specialist preservation concrete. Take error rows
`A = [0.1, 0.9, 0.5]`, `B = [0.9, 0.1, 0.5]`, `C = [0.5, 0.5, 0.1]`, `D = [0.4, 0.4, 0.4]` — specialists
on cases 0, 1, 2 and a mediocre generalist. Under the mean, `A = B = 0.50`, `C = 0.37`, `D = 0.40`:
tournament ranks `C` best, `D` next, and the two single-case specialists sit dead last — exactly the
trees it selects *out*, even though each is the population's unique best on its own case. Lexicase
instead returns whoever leads its first gate: case 0 first hands the parent to `A`, case 1 to `B`, case
2 to `C`, while `D`, never unique-best anywhere, wins a first gate essentially never. Over many events
lexicase rotates through all three specialists — precisely the half-built factors crossover needs —
where tournament kept only `C` and `D`. That is the `sin(x)`/`cos(y)`-specialist survival I need on
Nguyen-10.

So I am sold on the structure. Now drop it onto *this* problem — continuous symbolic regression on
real-valued, noisy data — and watch it break. Pool is the whole population, shuffle the cases, first
case: find the best error on it, keep only individuals whose error *equals* that best. And right there
is the wall. My errors are continuous floats. What is the chance two distinct expression trees produce
the *exact same* 64-bit floating-point error on a case? On the order of `2⁻⁵²` — essentially zero; it
only happens if they are literally the same model. So "best error on this case" is achieved by exactly
one individual, and the very first gate slams the pool from 500 to 1. Every selection event is then
decided by a single case, and the entire multi-case filtering chain — the conjunction mechanism that was
the whole point — is gone. And single-case selection is *worse* than the tournament I am replacing: it
picks a parent by its error at one of the 20 (Nguyen-7) or 100 (Nguyen-10) training points, discarding
everything the other points say — strictly less information than tournament's mean over all of them. The
mechanism is sound; the *pass condition* is too sharp for a continuous space.

Diagnose it precisely so I fix the right thing. The gate asks "is your error equal to the pool-best?"
In a discrete space many individuals share the best, so the gate keeps a meaningful subset and the
chain continues. In a continuous space, "equal" is measure-zero, so the gate keeps a singleton. I do
not want "exactly best." I want "near-best": close enough to the best on this case that I do not
disqualify a perfectly good near-elite for losing by `10⁻⁷`. So relax the gate — an individual passes
case `t` if `e_t(i) ≤ e*_t + ε`. Now near-elites survive, a single case no longer empties the pool,
multiple cases participate again, and I recover the conjunction behavior in continuous space.

What is `ε`? The lazy answer — a user-set constant — is a trap, and I can see why before coding it. A
fixed band is blind to how the population is doing: early in a run errors are huge and spread wide;
late, as the population converges, they are small and tight. A constant reasonable early becomes, late,
so wide relative to the now-tiny spread that everyone passes every case — the gates stop filtering,
selection goes random, and I lose all pressure exactly when I should be fine-tuning. And the right
value is problem-dependent — a band that filters sensibly on Koza-3's error scale would over- or
under-filter on Nguyen-10's. I refuse to hand-tune a knob per benchmark *and* have it be wrong at one
end of every run. So `ε` must set itself from the data. The natural idea: make `ε` track the *spread*
of the errors on each case. Tightly clustered errors need a small band to separate near-elites; a
widely spread (hard, contentious) case admits more contenders with a larger band; and as the population
solves a case and the errors compress, the band shrinks with them — the self-scheduling the constant
lacked.

Which dispersion measure? The textbook standard deviation fails immediately on a GP population, and the
reason is specific to GP. A population at any moment is full of *garbage* — freshly mutated junk,
expressions that divide by near-zero or blow up and produce enormous errors. Those are extreme outliers
in the error vector, and the standard deviation is dominated by them. Let me actually compute an example
so the failure is not hypothetical. Suppose on some case the six real contenders sit at errors `[0.20,
0.25, 0.30, 0.35, 0.40, 0.45]` and one blow-up tree sits at `10⁶`. The median error is 0.35, and the
median absolute deviation is `MAD = median|e − 0.35| = 0.10` — a sensible read of "the contenders live
about 0.1 apart." The standard deviation of the same seven numbers is ≈ `3.5·10⁵`, driven almost
entirely by the one blow-up, a factor of roughly three million larger than the MAD. Now play each as a
band: `ε = MAD = 0.10` sets the pass threshold at `0.20 + 0.10 = 0.30` and keeps `{0.20, 0.25, 0.30}` —
three of seven, a real filter that isolates the near-elite. `ε = σ ≈ 3.5·10⁵` sets the threshold at
essentially `3.5·10⁵` and passes all six real contenders (it excludes only the blow-up), so the σ-band
does no discriminating whatever among the trees I actually care about. The σ-band is set *by the junk*,
is enormous, and waves everyone through — back to no filtering. So I need a measure that ignores the
tails and reports the typical spread among the competitive individuals. That is the median absolute
deviation: `MAD = median_j |e_t(j) − median_k e_t(k)|`. The inner median is the typical error, the
outer the typical distance from it; a minority of arbitrarily huge outliers moves it by at most one rank
step — MAD has a 50% breakdown point, it simply does not see the junk. So `ε_t = MAD(e_t)` reports the
scale the real contenders live in, exactly where "near-elite" should be judged. It is parameter-free,
auto-scales to each case's difficulty, and shrinks as the population converges. That MAD-not-σ choice is
forced by the outlier-heavy nature of GP populations specifically.

Now the reference set: over which individuals do I compute the best error and the MAD? The cheap answer
is the whole population, once per generation — freeze `e*_t` and `ε_t` into a population-level pass
table. But it bothers me that by the third or fourth gate my pool is a handful of strong individuals
while I am still measuring "best" against the original 500, most already filtered out and irrelevant to
this event. So sharpen it: recompute *both* the pool elite `e*_t` and the MAD over the *current*
candidate pool at each gate. As the pool homogenizes, the pool-MAD collapses, so `ε` shrinks further and
the later gates keep filtering hard instead of waving everyone through — and the pool-best always clears
its own band, so the pool never empties. This pool-relative (dynamic) form is the one that keeps the
gates discriminating all the way down the chain, and it is what I land on. It also self-schedules the
endgame: as the pool winnows to near-tied contenders their errors cluster tightly, the pool-MAD
collapses toward zero, and the band `e*_t + ε_t` tightens back toward exact-best — so the last gates
decide by who is genuinely best among equals, wide when the pool is contested and narrow at fine
distinctions, without my writing any schedule.

There is a consistency point I should not gloss: selection no longer looks at the scalar MSE at all, yet
elitism and the outer loop still do. Is that coherent? It is, and deliberately so. The per-case filter
chooses *parents* — who gets to breed — and that is where I want behavioral diversity, specialists and
all. But the thing a run ultimately reports is a single tree, and "best" for reporting can only mean
lowest aggregate error on the training sample, which is raw MSE. So the two roles are genuinely
different: lexicase decides who reproduces, raw MSE decides who is remembered and carried forward
untouched. Keeping `fitness_function` as raw MSE is exactly what lets the elite stay the honest
best-fitter even while selection is breeding from specialists that would never top the MSE ranking — the
two mechanisms pull in complementary directions rather than fighting.

The price is real and worth quantifying against the tournament it replaces. Recomputing a minimum and a
MAD over the current pool at each gate is `O(PN)` worst case per selection, `O(P²N)` to fill the
population. But the pool winnows fast: if each contested gate keeps a fraction `ρ ≈ 0.3`, it reaches one
after `g ≈ ln 500 / ln(1/0.3) ≈ 5` gates, so a typical parent reads about `500 + 150 + 45 + … ≈ 700`
error entries against tournament's 7 — a factor near 100, heavy but a constant, since the winnowing
rescues the average. So I expect a large wall-clock jump, and the honest question is whether that
compute buys back the seeds standard GP threw away.

The contract here passes the per-case absolute-error matrix into `selection` from
`evolve_one_generation`, which builds it once per generation by evaluating every tree on the training
inputs; `selection` slices that matrix to the current candidate set at each gate and recomputes the
pool elite and pool MAD there. If the matrix is ever absent it falls back to a size-7 tournament, so the
function is safe to call either way. Crossover and mutation stay exactly the standard uniform-point
subtree operators — I am changing *selection only*, because the diagnosis was that selection is what
hands the population to the wrong lineage. `fitness_function` stays raw MSE,
because the outer loop still tracks the best-ever tree by it for elitism and reporting, and elitism
still carries the raw-MSE best forward unconditionally. The full scaffold module is in the answer.

Now the falsifiable expectations against the standard-GP numbers. The bet is that spreading selection
pressure across cases kills the premature convergence, so the cleanest prediction is that **Nguyen-7's
−1.0 seed disappears**: keeping alternative forms alive long enough that no single early winner locks the
population should turn all three seeds positive and pull the mean from 0.330 back toward the ~0.99 the
two good seeds already showed. On Nguyen-10, preserving the `sin(x)` and `cos(y)` specialists tournament
buried should lift the mean above 0.588, with seed 42 holding and the weaker seeds pulled up. Koza-3 is
the control — already solved by the plain engine, so case-based selection should not hurt it, staying
near 0.99. The refutation to watch: if Nguyen-10 does *not* improve, the bottleneck there is
representational reach, not selection diversity — the product is simply hard to assemble under the depth
cap — and the lever would have to move off selection.
