Random search told me exactly what it was always going to tell me, and it told me in the per-seed
columns. The means are respectable — 93.38 on CIFAR-10, 70.65 on CIFAR-100, 44.57 on ImageNet16-120 —
which is the volume argument coming true: thirty independent shots at a space whose good region is a
large fraction usually catch something decent, the best-of-thirty landing around the top few per cent in
expectation. But the spread is the message, and it is worth recomputing rather than eyeballing. CIFAR-10
sits at ±0.33 (seeds 92.83 to 93.74, a range of 0.91), CIFAR-100 at ±0.65 (69.92 to 71.54), and
ImageNet16-120 at ±0.58 on a lower base. Turning those into a scale-free comparison — dividing each spread
by its mean — gives coefficients of variation of about 0.35%, 0.93%, and 1.31% as I walk from CIFAR-10 to
CIFAR-100 to ImageNet16-120. So the relative dispersion roughly quadruples across the three settings, in
exactly the order the order statistic predicted: the same Beta(1, 30) quantile tail maps to a tight
accuracy band where the top of the distribution is compressed (CIFAR-10) and to a progressively wider one
where architecture quality falls off faster from the top (CIFAR-100, then ImageNet16-120). The floor
behaved as its own mechanism said it would, which means I can trust the diagnosis it hands me.

And the diagnosis is written most plainly in the ImageNet16-120 column. Its five seeds fall in a near-clean
monotone staircase — 45.47, 44.87, 44.60, 44.10, 43.80 — with no plateau, each seed a notch below the
last. That shape is the signature of pure luck-of-the-draw with no structure: five independent darts at the
Beta tail, sorted, look like a staircase precisely because there is nothing but the tail generating them.
The seed-3 CIFAR-100 of 69.92 and the seed-4 ImageNet of 43.80 are the tell within the tell: those are the
runs where thirty memoryless gambles simply never drew into the top region — the unlucky, top-9%-quantile
end of the Beta band realized — and random search has no way to *recover*, because it never looks at what
it already found. That is the structural failure I pointed at from the floor: not a bad mean, but a result
governed by the order-statistic tail, with the variance on the same order as any gap I could hope to open.
And the marginal-value curve told me where the slack is — the later of the thirty independent draws each
buy almost nothing, since the expected best quantile only crawls from `1/21 ≈ 0.048` at draw twenty to
`1/31 ≈ 0.032` at draw thirty. So the lever is obvious and narrow: stop throwing those low-value later
draws away as independent gambles. When a query reveals a strong architecture, spend the next queries
*near* it instead of uniformly, so a good early draw can be built on and a poor start can be climbed out of.

What is the simplest mechanism that turns "build on the best-seen" into a search? I do not want to train a
second model or carry a schedule — at 30 queries that overhead is not paid back, and the controller
machinery of RL-based search (an LSTM emitting tokens, updated by policy gradient) is exactly the
complexity I am trying to avoid, because it needs far more than thirty rollouts to say anything. The barest
possible exploit-the-good loop is evolution: keep a small population of evaluated architectures, repeatedly
pick a good one as a parent, mutate it by one small edit, evaluate the child, and let the population turn
over. No controller, no backprop through anything, nothing that couples the budget to a learner — just
selection and mutation, which is precisely the "spend later queries near the good draws" lever the random
search numbers asked for. The whole design question is what the turnover rule and the knobs should be in
the K = 30 regime, because the standard evolution recipes were tuned for thousands of evaluations and will
not transfer unchanged. Everything below is porting evolution to a budget two or three orders of magnitude
smaller than the one it was designed for.

Before committing to a population, I should check the one exploit mechanism simpler still — single-lineage
local search, hill-climbing or simulated annealing: hold one current cell, propose a single-edge neighbor,
accept it if better (or, for annealing, sometimes if worse, under a cooling temperature). It is even
barer than evolution — no population to carry — so why not it? Two reasons, both about the budget. First,
annealing's whole content is its temperature schedule, and a schedule is a knob (or several) that has to
be tuned to the horizon; at thirty queries I cannot spend runs tuning a cooling rate, and an untuned
schedule is as likely to hurt as help. Hill-climbing avoids the schedule but then has no escape from a
local optimum at all — worse than what I am replacing. Second, and decisively, a single lineage has no
diversity to hedge a bad start: if the one current cell sits in a mediocre basin, single-lineage search is
stuck there with nothing else in hand, which is precisely the random-search failure (cannot recover from a
poor draw) reappearing in a new costume. A population of P gives me P simultaneous footholds and a
selection rule to bet among them, and the tournament greed is a single interpretable knob rather than a
schedule. So evolution is not just simplest-that-works by taste; it is the cheapest mechanism that carries
a diversity hedge, and the hedge is exactly what the floor's failure said I need.

Start with the loop skeleton and make selective pressure precise, because that is the one thing that
governs whether 30 queries is enough to climb. Each cycle I run a tournament: sample S members of the
population at random, take the highest-accuracy one as the parent, mutate, evaluate the child, insert it.
S is the greed knob, and I can pin its effect down with the old takeover-time result — for an S-ary
tournament the best individual floods a population of size P in roughly `t* ≈ (ln P + ln ln P) / ln S`
cycles. The dependence on S runs through `1/ln S`, so let me actually evaluate it for P = 10 across the
candidate greed levels. With `ln 10 + ln ln 10 = 2.303 + 0.834 = 3.137` on top, S = 2 gives `3.137/0.693 ≈
4.5` cycles, S = 3 gives `3.137/1.099 ≈ 2.9` cycles, S = 5 gives `3.137/1.609 ≈ 1.9`, and S = 7 gives
`3.137/1.946 ≈ 1.6`. So the choice of S sets how many cycles the population takes to be swallowed by a
single lineage, and the numbers span 4.5 down to 1.6. With only about twenty cycles of evolution after the
population is seeded, this is the crux: S = 7 would flood the population in under two cycles, so the
remaining ~18 cycles would be a near-pure hill-climb on one lineage — fine if that lineage happens to sit
near the good region, fatal if the seeds were unlucky, which is exactly the random-search failure I am
trying to fix. S = 2 floods in 4.5 cycles, leaving room to explore, but is so slow to concentrate that
twenty steps may never really exploit a good seed before the run ends.

I can make the same tradeoff precise a second way — through the expected quality of the parent, not just
the takeover time — and it is worth doing because the two agree and that agreement is a check on the
reasoning. If the population members are ranked 1 (best) to P (worst), the best of an S-sample drawn
without replacement has expected rank `(P + 1)/(S + 1)`. For P = 10 that is 3.67 at S = 2, 2.75 at S = 3,
1.83 at S = 5, and 1.375 at S = 7. Two sanity checks on the formula: at S = 1 it gives `(P+1)/2`, the
expected rank of a single uniform pick, which is right; at S = P it gives 1, always the champion, also
right. So at S = 3 the parent is, on average, the 2.75-th best of ten — solidly in the top third but not
deterministically the global best, which keeps a little exploration alive in every selection. At S = 7 the
parent is the 1.4-th best, i.e. almost always the current champion, which is the same near-hill-climb the
takeover time flagged from the other direction. Both lenses point at moderate greed, and they point there
independently, so I fix S = 3.

Now the population size, and here the reasoning is budget arithmetic and one explicit bet. Thirty queries
split into a seeding phase and an evolution phase. I seed the population by drawing P random architectures
and evaluating each; that is the first P queries, and it is itself a random search of depth P. With P = 10
the seeding leaves ~20 queries to evolve, and the two phases balance: ten seeds hold genuine diversity
rather than collapsing immediately, twenty steps is enough for the S = 3 takeover time of ~3 cycles to
matter several times over. Compare the alternatives concretely. If I seed only P = 5, I free up 25
evolution steps, but a tournament of S = 3 from a population of 5 has expected parent rank `(5+1)/(3+1) =
1.5` — nearly the champion every time — so the search would concentrate almost instantly, the very
premature convergence I am trying to avoid. If I seed P = 15, I get a diverse population but only 15
evolution steps, cutting the exploitation budget by a quarter. P = 10 is the balance point: diverse enough
that S = 3 is moderate rather than greedy, small enough that twenty steps churn it meaningfully.

The bet hiding inside that split is worth stating in the order-statistic language from the floor, because
it is exactly what could make this rung *lose*. Ten random seeds give an expected best quantile of `1/11 ≈
0.091` — the top ~9% — before any evolution happens. Full thirty-draw random search reaches `1/31 ≈
0.032`, the top ~3%. So by spending only ten of my thirty queries on independent draws, I start the climb
from a *worse* expected position (top-9%) than random search would have finished at (top-3%), and I am
betting that the twenty directed single-edge mutations recover more than the twenty independent draws I
gave up. If the local landscape around a decent seed is climbable — if single edits reliably improve
accuracy — twenty directed steps beat twenty random gambles and REA passes random search. If the landscape
is rugged, or the seeds land in a mediocre basin with no climbable path to the good region, twenty
mutations refine a mediocre cell and REA *under*performs the floor it started from. That is the whole risk,
and it is not hedged away by any knob; it is inherent in trading exploration breadth for exploitation depth
at a fixed budget.

Now the eviction rule, which is where evolution either filters noise or accumulates it. The textbook greedy
thing is to remove the *worst* member each cycle — survival of the fittest, the population's average
ratchets up. On NAS-Bench-201 the tabular accuracy is deterministic (a query is a table lookup, not a
noisy training run), so the classic argument that kill-the-worst homesteads a *lucky* stochastic evaluation
does not bite the same way it would with real training. But there is a different reason to prefer removing
the *oldest* member even here, and it is about exploration rather than noise. Kill-the-worst keeps every
high-scoring architecture in the population permanently, so once a strong cell is found it sits there
forever, gets re-selected as parent over and over, and the population concentrates around it — the same
premature-convergence collapse the S knob was fighting, just driven now by genuine score instead of greed.
Let me make the failure concrete with a tiny trace. Suppose the seeding drew one cell at 46 and nine cells
clustered near 44. Under kill-the-worst, the 46 can never be evicted — it is never the worst — so it
persists across all twenty evolution cycles, is drawn into most tournaments, and wins them; every child is
a mutation of that one cell or of its descendants, and the twenty precious steps pile onto a single
lineage. If that lineage's basin tops out below the global good region, the search is stuck with no
mechanism to leave, because the anchor that traps it is exactly the member kill-the-worst refuses to
remove.

Kill-the-oldest breaks that trap. Every architecture gets a fixed bounded lifespan of exactly P cycles
regardless of its score — I append the child on the right of the population list and pop index 0, the
oldest — so a strong cell can only persist by being *re-discovered*: a parent carrying it must keep
producing children that also score well and re-enter the population before the original is aged out. In the
46-versus-44 trace, the 46 now leaves after ten cycles unless its descendants keep re-earning their place,
which forces the population to keep moving through new regions rather than homesteading the first good one.
The age-based turnover *is* the regularizer — it constrains the survivors to architectures the search keeps
re-finding, and it keeps the twenty evolution steps spread across the space instead of freezing on one
lineage. I can pin the turnover down numerically: with P = 10 and FIFO eviction, seeding occupies epochs 0
through 9, and each evolution cycle evicts the oldest survivor, so the ten original seeds are all gone from
the population by epoch 19 — the tenth evolution step — and epochs 20 through 29 then evict the first ten
children. Over the twenty evolution cycles the population turns over roughly twice completely; by the end,
not one of the original random seeds is still alive in the population. That is the point of the mechanism:
persistence is earned by rediscovery, not granted by a single lucky score. Encoding age is free — append
right, pop index 0 — so this costs no extra knob; P and S remain the only two.

That last fact forces a companion decision: because age can evict a strong cell from the population, I must
track the best-seen architecture *separately* from the population, updating it on every evaluation, seeding
and evolution alike, so that even after a great cell ages out I still return it. The metric rewards the best
architecture *ever found*, not the best currently alive, and the two diverge exactly because kill-the-oldest
deliberately lets good cells leave. So a small running best-and-its-score, updated on every query — the same
monotone-best bookkeeping the floor used — sits alongside the evolving population and is what
`get_best_architecture` returns. Without it, age-based eviction would throw away the very cell the search
exists to find.

The mutation has to match this task's edit surface, and here the cell space makes it trivial. An
architecture is a list of six op-indices in `[0, 4]`; a single-edge mutation picks one of the six edges and
changes its operation to a different one of the five — that is exactly the fixed helper
`mutate_architecture(parent)` the loop hands me. One edit is the right granularity for a computable reason:
it is the smallest move in the space, so a child stays close to a parent the tournament already judged
good, and I can count the neighborhood — six edges times four alternative operations each is exactly 24
single-edge neighbors of any architecture. Twenty-four is a small, local reach, and that locality is both
the strength and the danger. The strength: a child is one step from something already scored well, so
exploitation is cheap and reliable. The danger is a distance argument. The space has diameter six under
single-edge moves — any architecture differs from any other in at most six edges, so at most six mutations
connect any two — which sounds reassuring until I remember the moves are not directed. If the best seed
sits, say, Hamming-distance five from the good region and the intervening cells are *worse* (a fitness
valley), single-edge tournament mutation, which almost always accepts improving or lateral moves, cannot
pay the temporary cost to cross the valley in twenty steps. So the connectivity guarantees a path exists;
it does not guarantee twenty greedy local steps will walk it. That is the same risk the seeding bet exposed,
seen at the level of moves: REA can only ever be one edit from something it has already evaluated, so it
interpolates within a basin and cannot extrapolate across one.

I do not need crossover — there is no meaningful way to splice two six-edge cells that chained single
mutations cannot already reach, and crossover would add machinery for no gain at this budget. The one guard
is validity: a mutation could in principle produce an all-`none` degenerate cell, so I re-mutate until
`is_valid_arch` passes before spending a query on the child — that costs no budget, only a few cheap helper
calls, and it means every one of my thirty queries lands on a cell that can actually compute.

So the delta from random search is concrete and small. Where random search drew all 30 architectures
independently and kept the best, REA draws only the first 10 independently to seed a population, then spends
the remaining ~20 queries on children of tournament-selected parents (S = 3), mutating one edge at a time,
evicting the oldest member each cycle, and tracking the best-ever separately for the final return (the
distilled class is in the answer). The exploitation that random search structurally lacked is now the whole
point: a good seed gets built on through its descendants, and a poor seeding region gets climbed out of as
the population turns over toward whatever the tournaments keep re-finding — *if* the local landscape
permits it.

Reading the random-search numbers, here is what I expect this to fix and where I am unsure, stated so the
next feedback table can falsify it. The wins should show up where random search *failed to recover* — the
wide-spread settings. On ImageNet16-120, where random search drew that clean monotone fall to 43.80 on its
unlucky seed, the twenty evolution steps should lift the unlucky seeds by climbing local neighborhoods of
the best seed, so I expect the ImageNet16-120 mean to rise above 44.57 — that is the most falsifiable
prediction, because it is the setting with the most room between the floor and the ceiling. I can even name
the concrete targets: random search's two unlucky ImageNet seeds finished at 44.10 and 43.80, which are
1.37 and 1.67 points below its own best seed at 45.47. If twenty single-edge steps can climb a local
neighborhood by even a single point — a plausible ask if those seeds landed in a climbable basin — those
two runs alone would pull the mean up by roughly `(1.0 + 1.0)/5 ≈ 0.4` points, carrying 44.57 comfortably
past 45. So the mechanism has a clear, arithmetically-sized job to do on exactly the seeds where random
search had no recourse. On CIFAR-100 I
expect a modest lift or at worst a wash over 70.65: the benefit of exploitation is real but the budget for
it is only twenty steps, and if CIFAR-100's landscape does not reward single-edge climbs cleanly the
directed steps buy little over the twenty random draws they replaced. On CIFAR-10 I am genuinely unsure it
helps the mean at all and it might even nudge below 93.38: random search already nearly saturates the
compressed top of CIFAR-10's distribution — its 0.35% coefficient of variation says there is almost no room
above the floor — and concentrating queries via tournaments trades away a little of the per-axis coverage
that made random search strong there. If REA's CIFAR-10 mean lands around or just below random search's,
that is the narrow-band ceiling biting, not a bug.

The cleaner and more dangerous prediction is about *variance*, and it follows straight from the local-search
mechanism. Because REA is a local search that exploits beautifully when the seeds land near the good region
and can fail badly when they do not — and because ten random seeds plus twenty single-edge steps is thin
coverage for guaranteeing the former — I should expect its seed-to-seed spread on the low-saturation
settings to be *at least* as wide as random search's, and quite possibly wider, since a good-seed run and a
bad-seed run now diverge more than two memoryless runs would. So the signature I am looking for is REA
beating random search on the high-variance, low-saturation setting (ImageNet16-120 mean above 44.57) while
staying within noise on the near-saturated one (CIFAR-10); a widened ImageNet16-120 spread would be
*consistent* with the mechanism working on the lucky seeds and failing on the unlucky ones, not evidence
against it. But if instead REA *loses* on ImageNet16-120 outright, my diagnosis was wrong: the failure is
not lack of exploitation but that single-edge hill-climbing from random seeds cannot find the good region
in twenty steps — the distance-across-a-valley problem — and the next rung should not be more or better
evolution at all. It would have to be something that can reach architectures far from every seed instead
of only refining what luck placed within a few edits of a random draw — the one thing a mutation operator
that never reaches past its 24 neighbors can never do, no matter how the tournament and eviction knobs are
set.
