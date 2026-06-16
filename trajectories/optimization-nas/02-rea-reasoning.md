Random search told me exactly what it was always going to tell me, and it told me in the per-seed
columns. The means are respectable — 93.38 on CIFAR-10, 70.65 on CIFAR-100, 44.57 on ImageNet16-120 —
which is the volume argument coming true: thirty independent shots at a space whose good region is a
large fraction usually catch something decent. But look at the spread. CIFAR-10 is tight (±0.33, seeds
between 92.83 and 93.74), exactly as I expected from the compressed top of its accuracy distribution;
CIFAR-100 is wider (±0.65), and ImageNet16-120 widest (±0.58 on a lower base, and the seeds fall in a
clean monotone 45.47, 44.87, 44.60, 44.10, 43.80 — that is luck-of-the-draw, not signal). The seed-3
CIFAR-100 of 69.92 and the seed-4 ImageNet of 43.80 are the tell: those are runs where thirty memoryless
gambles simply never drew into the top region, and random search has no way to *recover*, because it
never looks at what it already found. That is the structural failure I pointed at — not a bad mean, but a
floor governed by luck, with the variance on the same order as any gap I could hope to open. So the lever
is obvious and narrow: stop throwing every draw away. When a query reveals a strong architecture, spend
the next queries *near* it instead of uniformly, so a good early draw can be built on and a poor start
can be climbed out of.

What is the simplest mechanism that turns "build on the best-seen" into a search? I do not want to train
a second model or carry a schedule — at 30 queries that overhead is not paid back, and the controller
machinery of RL-based search is exactly the complexity I am trying to avoid. The barest possible
exploit-the-good loop is evolution: keep a small population of evaluated architectures, repeatedly pick a
good one as a parent, mutate it by one small edit, evaluate the child, and let the population turn over.
No controller, no backprop through anything, nothing that couples the budget to a learner — just
selection and mutation, which is precisely the "spend later queries near the good draws" lever the random
search numbers asked for. The whole design question is what the turnover rule and the knobs should be in
the K = 30 regime, because the standard evolution recipes were tuned for thousands of evaluations and
will not transfer unchanged.

Start with the loop skeleton and make selective pressure precise, because that is the one thing that
governs whether 30 queries is enough to climb. Each cycle I run a tournament: sample S members of the
population at random, take the highest-accuracy one as the parent, mutate, evaluate the child, insert it.
S is the greed knob. There is the old takeover-time result — for an S-ary tournament the best individual
floods a population of size P in roughly `t* ≈ (1/ln S)·[ln P + ln ln P]` cycles — and the dependence on
S is through `1/ln S`, so bumping S from 2 to 4 roughly halves the takeover time. With only ~20 cycles of
evolution after the population is seeded, I cannot afford the leisurely takeover that S = 2 gives on a
large benchmark; but I also cannot afford to be so greedy that the search collapses onto the first decent
architecture and stops exploring. This is the crux of porting evolution to a tiny budget: the population
and tournament sizes that work at thousands of evaluations are wrong here, and I have to shrink both.

So fix the population at P = 10 and the tournament at S = 3. The reasoning is budget arithmetic. Thirty
queries split into a seeding phase and an evolution phase. I seed the population by drawing P = 10 random
architectures and evaluating each — that is the first 10 queries, and it is itself a random search of
depth 10, so the population starts already containing a reasonable best. That leaves ~20 queries to
evolve. P = 10 is small enough that those 10 seeds are a real fraction of the population's lifetime churn,
and large enough to hold genuine diversity rather than collapsing immediately. S = 3 out of 10 is
moderate greed: the parent is the best of a 3-sample, so it is usually one of the stronger members but
not deterministically the global best, which keeps a little exploration alive. A larger S (say 5 or 7 out
of 10) would make almost every parent the current champion and turn the 20 evolution steps into a near
hill-climb on one lineage — fine if that lineage is right, fatal if the seeds were unlucky, which is
exactly the random-search failure I am trying to fix. S = 3 is the low-budget tournament size the
sample-efficient NAS recipes converge on, and the takeover formula says it floods P = 10 in only a
handful of cycles, which is what I want: fast enough to exploit a good seed within the 20-step budget,
loose enough not to freeze on a bad one.

Now the eviction rule, which is where evolution either filters noise or accumulates it. The textbook,
greedy thing is to remove the *worst* member each cycle — survival of the fittest, the population's
average ratchets up. On NAS-Bench-201 the tabular accuracy is deterministic (a query is a table lookup,
not a noisy training run), so the classic argument that kill-the-worst homesteads a *lucky* evaluation
does not bite the same way it would with stochastic training. But there is a different reason to prefer
removing the *oldest* member even here, and it is about exploration rather than noise: kill-the-worst
keeps every high-scoring architecture in the population permanently, so once a strong cell is found it
sits there forever, gets re-selected as parent over and over, and the population concentrates around it —
the same premature-convergence collapse, just driven by genuine score instead of luck. With only 20
evolution steps, concentrating early means I stop exploring the rest of the space far too soon, and I am
right back to the random-search problem of being unable to recover from a mediocre early region.
Kill-the-oldest gives every architecture a fixed bounded lifespan of about P cycles regardless of its
score, so a strong cell can only persist by being *re-discovered* — a parent carrying it must keep
producing children that also score well — which forces the population to keep moving through new regions
rather than homesteading the first good one. The age-based turnover *is* the regularizer: it constrains
the survivors to architectures the search keeps re-finding, and it keeps the 20 precious evolution steps
spread across the space instead of piling onto one lineage. Encoding age is free — append the child on
the right of the population list, pop index 0 (the oldest) — so this costs no extra knob; P and S remain
the only two.

The mutation has to match this task's edit surface, and here the cell space makes it trivial. An
architecture is a list of six op-indices in `[0,4]`; a single-edge mutation picks one of the six edges
and changes its operation to a different one of the five — that is exactly the fixed helper
`mutate_architecture(parent)` the loop hands me. One edit is the right granularity: it is the smallest
move in the space, so a child stays close to a parent the tournament already judged good (local
exploitation), and chaining single edits can reach any architecture from any other (the space stays
connected), so I am not trapping the search in a neighborhood. I do not need crossover — there is no
meaningful way to splice two six-edge cells that single mutations cannot already reach, and crossover
would add machinery for no gain at this budget. The one guard is validity: a mutation could in principle
produce an all-`none` degenerate cell, so I re-mutate until `is_valid_arch` passes before spending a
query on the child — that costs no budget, only a few cheap helper calls. And I track the best-seen
architecture separately from the population (a simple `_update_best` on every evaluation, seeding and
evolution alike), so that even if age evicts a strong cell from the population I still return it: the
metric rewards the best architecture *ever found*, not the best currently alive.

So the delta from random search is concrete and small. Where random search drew all 30 architectures
independently and kept the best, REA draws only the first 10 independently to seed a population, then
spends the remaining ~20 queries on children of tournament-selected parents, mutating one edge at a time
and evicting the oldest member each cycle, while tracking the best-ever for the final return (the
distilled class is in the answer). The exploitation that random search structurally lacked is now the
whole point: a good seed gets built on through its descendants, and a poor seeding region gets climbed
out of as the population turns over toward whatever the tournaments keep re-finding.

Reading the random-search numbers, here is what I expect this to fix and where I am unsure. The wins
should show up where random search *failed to recover* — the wide-spread settings. On ImageNet16-120,
where random search drew a clean monotone fall to 43.80 on its unlucky seed, the 20 evolution steps
should lift the unlucky seeds by climbing local neighborhoods of the best seed, so I expect the
ImageNet16-120 mean to rise above 44.57 — that is the most falsifiable prediction, because it is the
setting with the most room. On CIFAR-100 I expect a modest lift or at worst a wash over 70.65: the
benefit of exploitation is real but the budget for it is only 20 steps, and CIFAR-100's landscape may not
reward single-edge climbs as cleanly. On CIFAR-10 I am genuinely unsure it helps the mean at all and it
might even nudge down from 93.38: random search already nearly saturates the compressed top of CIFAR-10's
distribution, so there is almost no room above the floor, and concentrating queries via tournaments
trades away a little of the per-axis coverage that made random search strong there — if REA's CIFAR-10
mean lands around or just below random search's, that is the narrow-band ceiling biting, not a bug. The
clean signature I am looking for is REA beating random search on the high-variance, low-saturation
setting (ImageNet16-120) while staying within noise on the near-saturated one (CIFAR-10); if instead REA
*loses* on ImageNet16-120, my diagnosis was wrong and the failure is not lack of exploitation but that
single-edge hill-climbing from random seeds cannot find the good region in 20 steps — which would point
the next rung not at evolution at all but at a method that *models* the accuracy surface to extrapolate
beyond the seeds, rather than only mutating them.
