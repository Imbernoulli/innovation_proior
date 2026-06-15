The standard-GP numbers tell me exactly one story, and I want to read it precisely before I change
anything. Koza-3 is solved: 0.985, 0.998, 0.994 across the three seeds, mean 0.993 — the reachable
polynomial behaves exactly as I predicted, steady and high, no drama. The drama is entirely on the two
transcendental benchmarks, and it is *variance*, not a uniformly low mean. Nguyen-7 has two seeds at
0.998 and 0.991 — essentially solved — and then seed 456 at −1.0: a complete blow-up, the wrong form
found early, diverging off the wider test range badly enough that R² went negative and the floor caught
it. One bad seed drags the Nguyen-7 mean from "near-solved" down to 0.330. Nguyen-10 is the same disease
in a milder, more pervasive form: 0.884, 0.557, 0.322 — no single catastrophe, but a steady decline
across seeds, mean 0.588, as if the search keeps locking onto partial fits of the sin·cos product and
never recovers the rest. This is precisely the premature-convergence signature I worried about: with
nothing but raw-MSE tournaments, one lineage that gets an early lead on the 20-or-100-point training
sample dominates the tournaments, the population homogenizes onto its *form*, and on the transcendental
targets that form is wrong often enough that the seed-to-seed outcome is a coin flip — solved, or
stuck. The lever the diagnosis points at is selection: the failure is *which* lineage wins, so I should
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
mean it almost never wins. On Nguyen-10 this is exactly what I think is happening — the search needs a
tree that nails the `sin(x)` factor and another that nails the `cos(y)` factor, but a specialist on
one factor reads as mediocre on the mean and gets selected out before crossover can splice the two. The
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
seed and its Nguyen-10 decline.

So I am sold on the structure. Now drop it onto *this* problem — continuous symbolic regression on
real-valued, noisy data — and watch it break. Pool is the whole population, shuffle the cases, first
case: find the best error on it, keep only individuals whose error *equals* that best. And right there
is the wall. My errors are continuous floats. What is the chance two distinct expression trees produce
the *exact same* floating-point error on a case? Essentially zero — it only happens if they are the
same model. So "best error on this case" is achieved by exactly one individual, and the very first gate
slams the pool from 500 to 1. Every selection event is decided by a single case, and the entire
multi-case filtering chain — the conjunction mechanism that was the whole point — is gone. Worse,
single-case selection off one noisy case is *worse* than tournament. The mechanism is sound; the
*pass condition* is too sharp for a continuous space.

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
in the error vector, and the standard deviation is dominated by them: a handful of trees at error `10⁶`
while everyone real sits near `0.3` sends σ through the roof, so a σ-band would be set by the junk, be
enormous, and pass everyone — back to no filtering. I need a dispersion measure that ignores the tails
and reports the typical spread among the competitive individuals. That is the median absolute deviation:
`MAD = median_j |e_t(j) − median_k e_t(k)|`. The inner median is the typical error, the outer median
the typical distance from it; a minority of arbitrarily huge outliers moves it by at most one rank step
— MAD has a 50% breakdown point, it simply does not see the junk. So `ε_t = MAD(e_t)` reports the scale
the real contenders live in, exactly where "near-elite" should be judged. It is parameter-free,
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
gates discriminating all the way down the chain, and it is what I land on. The cost stays in lexicase's
class: recomputing a minimum and a MAD over the current pool at each gate is linear in the pool size per
case, so one selection is `O(PN)` worst case and filling the population is `O(P²N)` — the MAD adds no
asymptotic term, and in practice the pool winnows fast so wall-clock stays near tournament.

The contract here passes the per-case absolute-error matrix into `selection` from
`evolve_one_generation`, which builds it once per generation by evaluating every tree on the training
inputs; `selection` slices that matrix to the current candidate set at each gate and recomputes the
pool elite and pool MAD there. If the matrix is ever absent it falls back to a size-7 tournament, so the
function is safe to call either way. Crossover and mutation stay exactly the standard uniform-point
subtree operators from the previous rung — I am changing *selection only*, because the diagnosis was
that selection is what hands the population to the wrong lineage. `fitness_function` stays raw MSE,
because the outer loop still tracks the best-ever tree by it for elitism and reporting, and elitism
still carries the raw-MSE best forward unconditionally. The full scaffold module is in the answer.

Now the falsifiable expectations against the standard-GP numbers. The whole bet is that spreading
selection pressure across cases — and rewarding specialists on the hard ones — kills the premature
convergence that produced standard GP's worst outcomes. So the cleanest prediction: **Nguyen-7's −1.0
seed should disappear.** That catastrophe was one lineage converging early onto a form that diverges off
the test range; lexicase's case-by-case filtering keeps alternative forms alive long enough that no
single early winner can lock the population, so I expect all three Nguyen-7 seeds positive and the mean
to climb from 0.330 toward the ~0.97 the two good seeds already showed. On Nguyen-10, where the search
needs to splice a `sin(x)` specialist with a `cos(y)` specialist, lexicase should *preserve* exactly
those specialists that tournament buried — so I expect the Nguyen-10 mean to rise above standard GP's
0.588, with the seed-42 near-solved case holding and the weaker seeds pulled up. Koza-3 is the control:
it was already solved by the plain engine, so case-based selection should not hurt it — I expect it to
stay high, near 0.99. If instead Nguyen-10 *does not* improve, that would tell me the bottleneck there
is not selection diversity but raw representational search (the right product is simply hard to assemble
under this depth cap), and the lever would have to move off selection. And there is one cost I should
watch honestly: lexicase's `O(P²N)` selection is far heavier than tournament's `O(Pk)`, so I expect a
large jump in wall-clock per run — the question is whether that compute buys back the seeds standard GP
threw away.
