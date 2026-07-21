The search is the whole point, and with the scaffold's default fill there is no search — so the floor I
start from is just getting the population to *evolve* at all. The default selection picks parents
uniformly at random, ignoring fitness entirely; the default crossover returns a copy of `parent1`; the
default mutation returns a copy of the parent. The loop's elitism dutifully carries forward whichever
random tree of generation zero happened to score lowest, but nothing ever improves on it, because no
operator moves anyone in program space and no selection prefers the good over the bad. The result of a
run is decided entirely by the luck of the initial ramped-half-and-half draw. So the pain to fix first
is the one Koza fixed: turn the empty contract into a genuine Darwinian loop over expression trees.

Let me write down what I am really searching, because the structure of the space is what forces every
later choice. I have a sample `(X, y)` and I want a symbolic formula that fits it. The objects I am
searching over — expression trees — have no fixed size: a candidate might be three nodes or three
hundred, a flat sum or a deeply nested composition. There is no fixed-dimensional parameter vector
here, so there is nothing to take a gradient of and nothing to descend over. What I *can* do is score
a candidate: run it on the data and measure error. That is exactly the interface a genetic algorithm
needs — a population, a fitness, and operators that make new candidates from old ones — and nothing
more. It never asks for smoothness or differentiability. So the engine is right; the only question is
what the individual is and how I recombine two of them, and the substrate has already answered the
first half: the individual *is* the tree, and `get_all_nodes()` hands me every subtree with its parent
pointer so I can splice. The load-bearing fact behind the whole representation is that **any subtree of
a valid expression is itself a valid expression** — so swapping subtrees between two parents always
produces a syntactically valid, evaluable child. That is why crossover on trees works where crossover
on flat strings shattered, and the protected operators in the substrate close the other half of the
problem (any subtree's output is a legal input to any operator).

The space is unenumerable, which is what forces the search to be stochastic. Offspring can reach depth
eight (a full binary tree there holds `2⁹ − 1 = 511` nodes), and even frozen at fifteen nodes each
internal slot picks one of eight operators `{add, sub, mul, div, sin, cos, log, exp}` and each leaf a
variable or a real constant from a continuum, so distinct labelled trees already number in the billions
before any constant varies. There is no template to invert and no grid I could sweep in fifty
generations. The only handle is the score, and the only way to turn scores into progress is to breed
preferentially from the ones that score well — precisely what a genetic algorithm asks for. Now I fill
the contract one function at a time.

**Fitness.** The task is a least-squares fit of the candidate's output to the targets over the sample
points. The natural error is the mean squared error: predict over the sample, subtract the targets,
square, average. Lower is better. Why squared and not absolute? Absolute error is also a clean total
order, but squared error weights a candidate's single worst points hardest, which is exactly what I
want here — a tree that fits most of the sample and goes wild on a handful should be punished for the
handful, not averaged into looking fine. And — the part that matters for selection — it gives a clean
total order on candidates. Selection only ever needs to ask "is A better than B," and "smaller MSE is
better" answers that directly. I keep `fitness_function` as raw MSE with no penalty term baked in,
because the outer loop tracks the best tree *by this number* for elitism, stopping, and the final
report; if I quietly mixed a complexity penalty into the fitness, the loop would report and preserve
the wrong tree. Whatever shaping a later strategy wants belongs in selection, not here.

**Selection.** The default's uniform random choice is the thing most obviously broken — it applies zero
fitness pressure, so the population drifts rather than improves. Holland's original answer is
fitness-proportionate selection: expected offspring proportional to fitness share. But I can feel two
failure modes before I even code it. Early in a run, one lucky tree with far lower error than the rest
grabs an enormous share of the reproductive opportunities and floods the next generation with its
children; the population collapses onto one family, crossover has nothing distinct left to recombine,
and the search converges prematurely onto a mediocre peak. Late in a run, as everyone improves, the
fitnesses compress into a narrow band, the shares become nearly equal, and proportionate selection
loses its ability to discriminate exactly when I still want to prefer the slightly-better trees. Both
failures come from the same root: proportionate selection cares about the *magnitudes* of fitness, and
those magnitudes are badly behaved at both ends of a run, and on three benchmarks whose error scales
differ by orders of magnitude one tuning could never serve all three.

What I actually want is for selection to care only about *which* tree is better, not by how much — to
be invariant to the scale and spread of the numbers. So: draw a small number `k` of individuals at
random and let the single lowest-error one win. A tournament of size `k`. It only ever *compares*
fitnesses, never sums or weights them, so a tree a thousand times better is treated like one barely
better — it just wins its tournaments — and no single super-fit individual can swamp the next
generation, because being selected still requires being drawn into a tournament and there are only so
many. That kills the early collapse. Late in the run, even with nearly-equal fitnesses, the better tree
still wins whenever it meets worse contenders, so selection pressure is set by `k` rather than by the
accidental spread of scores. `k` is the single greediness knob: `k=1` is random drift, large `k` is
nearly-always-take-the-best; `k=7` is the standard default — strong enough to drive improvement, loose
enough that average trees occasionally win and diversity survives. It is also cheap: `O(k)` per draw,
no global sum.

The strength of that cap is worth one number. Filling 499 slots at rates 0.9 / 0.05 / 0.05 costs about
1.9 parents per slot, ~950 selection events; each samples 7 of the 500 without replacement, so any
individual — the current best included — appears in about `950·7/500 ≈ 13` of them and wins whenever it
appears. The best therefore seeds on the order of 13 breeding events out of 950, ~1.4% of the
reproductive opportunity, and that ceiling holds *no matter how far ahead it is*. Fitness-proportionate
selection has no such cap: a tree's share scales with its lead, so a thousand-fold-better tree can
command hundreds of the 500 slots, the next generation is overwhelmingly its children, and crossover
has nothing distinct left to splice. The tournament's flat ~13 is set by `k` and the population size,
not by the spread of scores, which is why it defuses both the early collapse and the late-run
compression that guts proportionate selection. So selection becomes a size-7 tournament minimizing MSE,
returning copies so a parent can be reused without being mutated in place.

**Crossover.** The core operator, and the one the representation makes natural. To make a child: copy
`parent1` (so the original is never damaged — it may be selected again), copy `parent2` as the donor,
pick a node in each, and replace the copy's chosen subtree with the donor's. Because both endpoints are
valid subexpressions, the child is always valid. Now the question is how to pick the two points. The
honest, simple choice — the one I commit to here — is **uniform** over the nodes. I pick the
offspring's crossover point from `1 .. size−1` (excluding the root, index 0, so a single crossover
never wholesale-replaces the whole offspring with the donor in the common case) and the donor's point
from `0 .. size−1` (the donor may contribute its entire tree). There is a known refinement —
weighting function nodes over terminals — and its size tells me what I pay to decline it. By role, let
`b` be the binary operators, `u` the unary, `L` the leaves; a tree of `n = L + b + u` nodes has `n − 1`
edges while the operators supply `2b + u` child links, so `L + b + u − 1 = 2b + u`, i.e. `L = b + 1`:
leaves outnumber binary nodes by exactly one regardless of the unary mix. The leaf fraction
`(b+1)/(2b+1)` tends to one-half as trees grow, so a uniform crossover point lands on a leaf roughly
half the time, and a leaf swap merely trades one variable or constant for a subtree — a near-trivial
edit that shuffles no structure. That ~50% rate of low-information crossovers is the tax uniform
selection pays, and it is what a 90/10 function-node weighting buys back. I decline it deliberately: the
point is to establish the plainest Koza-style loop and see where it lands, and the weighting is a real
improvement held in reserve rather than spent on the floor.

The splice cannot produce an illegal tree — both endpoints are valid subexpressions and every operator
is protected — so the closure property carries the whole operator. The depth cap lives right here as the
bloat brake: if crossover produces a child deeper than the cap, I reject it and return a copy of
`parent1`.

**Mutation.** Crossover has a built-in ceiling: it can only ever shuffle material *already present* in
the population. If a primitive or a useful constant was never drawn in generation zero, or died out
when a subpopulation went extinct, no amount of recombination brings it back — crossover splices
existing pieces, it cannot invent one. So I need a source of fresh material. Subtree mutation, mirroring
crossover: pick a node in a copy of the parent, delete the subtree there, and graft in a *freshly
generated* random subtree via `generate_tree('grow', 3, n_features)`. At generation zero no primitive
is scarce — with about ten internal nodes drawn near-uniformly from eight functions, a given operator
like `log` is absent with probability only `(7/8)¹⁰ ≈ 0.26`, so most of the 500 fresh trees carry one.
Scarcity is a *dynamic* event: once tournaments and crossover pull the population toward one family, the
lineages that carried `log` can go extinct, and from then no recombination reintroduces it. That is the
hole mutation plugs — a shallow grow at depth 3 (at most 15 nodes), small enough not to detonate a
working tree but large enough to reinject a lost operator or constant. It is disruptive, throwing away a
working subtree for noise, so it is a repair valve against extinction rather than an engine of
improvement — the search leans overwhelmingly on crossover and fires mutation sparingly. Same depth cap:
if the mutated child is too deep, return the parent unchanged.

**The generation loop.** Assemble it: elitism first — copy the single best individual (by raw MSE)
forward unconditionally, because my operators are stochastic and could destroy the current best, and I
never want the best-of-generation to go backward. Then fill the rest by rolling a uniform random number
per child: below `crossover_rate` (0.9), do subtree crossover on two tournament-selected parents;
between 0.9 and 0.9+`mutation_rate` (0.95), mutate one tournament-selected parent; otherwise (the
remaining 0.05), reproduction — copy one tournament-selected parent unchanged. Crossover at 0.9 is the
workhorse; mutation at 0.05 is the diversity drip; reproduction lets a good tree occasionally pass
forward intact. Repeat until the new population reaches `pop_size`, return it. Over 50 generations on a
population of 500, with the best-ever tree tracked outside, the run reports the best tree it ever found.

The whole loop is now coherent: a representation that makes the space searchable and recombination
valid (trees + closure), an engine that needs no gradient (Darwinian selection), a recombination that
exploits structure (uniform-point subtree crossover), a scale-free selector (tournaments), a diversity
injector (subtree mutation), survival of the best (reproduction + elitism), a brake on bloat (the depth
cap), and a fitness that is just the least-squares error. The full scaffold module is in the answer.

Now what should this bare engine do on the three benchmarks, since that is the whole point of running
it. Koza-3's `x⁵ − 2x³ + x` is *reachable*: `x⁵ = x·x·x·x·x` is four `mul` nodes at depth 3, `2x³`
another depth-3 product, and `sub(add(x⁵, x), 2x³)` wraps them at around depth 5 with a dozen internal
nodes — comfortably inside the depth-8 cap, using only `{add, sub, mul}` and a constant `2` an ephemeral
draw supplies. The building blocks are simple and there is no transcendental to gamble into, so I expect
Koza-3 to cluster near 1.0 across all three seeds.

The two harder benchmarks are where the bare engine should show its seams. Nguyen-7 needs assembled
*logarithms* plus extrapolation to a wider test range; Nguyen-10 needs a *product* of sine and cosine in
two variables. Both hinge on finding and keeping a specific transcendental block, and the failure
tournament selection only *partly* fixes is premature convergence: with nothing but raw-MSE tournaments,
one lineage that leads early on the training sample dominates the tournaments, the population
homogenizes, and the search locks onto a tree that fits the training points but is the wrong *form*,
which generalizes badly.

Extrapolation punishes a wrong form hardest, and it tells me the shape of the failure. Nguyen-7 trains
on `x ∈ [0, 2]` but tests on `[−0.5, 2.5]`, into `x < 0` the search never sees, where the true target is
gentle (about 3.2 at `x = 2.5`, about −0.47 at `x = −0.5`). A wrong-machinery tree — an `exp` of a
growing argument, a division poling just outside `[0, 2]` — stays finite under the protected operators
but saturates hard: `protected_exp` clips to `[−10, 10]`, so a runaway still returns up to
`e¹⁰ ≈ 22026`, four orders past the target's ~3. One such test point contributes a squared residual of
order `10⁸` against a total variance of order 1, driving `R²` sharply negative (floored at 0). Which
form a run locks onto is decided by which lineage wins the early tournaments, so this should be a coin
flip across seeds, not a uniform sag: some seeds find the right form and score near 1.0, others blow up.
So the signature I expect is **high seed-to-seed variance on the transcendental benchmarks** — a good
mean dragged down by one or two prematurely-converged seeds — with Koza-3 steady and high. Nguyen-10
should show the same disease milder: its `2·sin(x)·cos(y)` is bounded, so a wrong form fits partially
rather than diverging, giving a scattered spread instead of a blow-up.

That diagnosis already names the next lever. If the problem is premature convergence — one lineage
taking over — then the fix is in *selection*: shape what it rewards so it stops handing the whole
population to a single early winner. This plainest engine has no defense against its own convergence
dynamics, so on the benchmarks where finding and keeping the right transcendental form is the whole
difficulty, that is exactly what should cost it.
