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

Before I fill anything, I want a feel for how large the space is, because that is what decides whether
the search even has to be stochastic. Generation zero seeds trees up to depth six, and the offspring
depth cap is eight — a full binary tree of depth eight holds up to `2⁹ − 1 = 511` nodes, so the search
roams over trees of up to a few hundred nodes with no floor I could enumerate. Even if I froze the
shape at a modest fifteen nodes, each internal slot chooses among the eight operators
`{add, sub, mul, div, sin, cos, log, exp}` and each leaf among the input variables or a fresh real
constant from a continuum, so the count of distinct labelled trees is already in the billions before I
let a single constant vary continuously. There is no template to invert in closed form and no grid I
could sweep in fifty generations. The only handle I hold is the score, and the only way to turn scores
into progress is to breed preferentially from the ones that score well. That is precisely — and only —
what a genetic algorithm asks for, which is why the engine is right and the entire design collapses to
what the individual is and how I recombine two of them. Now I fill the contract one function at a time,
forcing each choice to justify itself.

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
the wrong tree. Whatever shaping a later rung wants belongs in selection, not here.

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

Let me put a number on that "no single super-fit individual can swamp" claim, because it is the
load-bearing reason to prefer tournaments over the proportionate scheme and I should know how strong
the effect actually is. Each generation I fill 499 non-elite slots; a crossover slot draws two parents,
a mutation or reproduction slot one, so at rates 0.9 / 0.05 / 0.05 the expected parents per slot is
`0.9·2 + 0.05 + 0.05 = 1.9`, roughly 948 selection events. Each event samples 7 of the 500 individuals
without replacement, so any particular individual — the current best included — appears in about
`948 · 7/500 ≈ 13` of them, and whenever the best appears it wins its tournament. So the current best
seeds on the order of 13 breeding events out of 948, a ceiling near 1.4% of the reproductive
opportunity, and that ceiling holds *no matter how far ahead it is* — a tree a thousand times better
than the field still turns up in only 7-in-500 tournaments. Fitness-proportionate selection has no such
ceiling: a tree's share scales with how far ahead it is, so map that thousand-fold lead onto a
maximized share and it commands a slice of the 500 slots proportional to its fitness ratio — potentially
hundreds of them — and the next generation is overwhelmingly its children, the pool collapses onto one
family, and crossover has nothing distinct left to splice. The tournament's flat ~13 is exactly the cap
that defuses the early collapse, and the fact that the cap is set by `k` and the population size rather
than by the spread of scores is why it also survives the late-run compression that guts proportionate
selection. So selection becomes a size-7 tournament minimizing MSE, returning copies so a parent can be
drawn and reused without ever being mutated in place.

**Crossover.** The core operator, and the one the representation makes natural. To make a child: copy
`parent1` (so the original is never damaged — it may be selected again), copy `parent2` as the donor,
pick a node in each, and replace the copy's chosen subtree with the donor's. Because both endpoints are
valid subexpressions, the child is always valid. Now the question is how to pick the two points. The
honest, simple choice — and the one this rung commits to — is **uniform** over the nodes. I pick the
offspring's crossover point from `1 .. size−1` (excluding the root, index 0, so a single crossover
never wholesale-replaces the whole offspring with the donor in the common case) and the donor's point
from `0 .. size−1` (the donor may contribute its entire tree). There is a known refinement here —
weighting function nodes more heavily than terminals — and I can make the reason it exists precise,
which also tells me exactly what I am paying to decline it. Count any tree by role: let `b` be the
binary operators (`add, sub, mul, div`), `u` the unary ones (`sin, cos, log, exp`), and `L` the leaves.
Every edge is a child link, so the binary nodes contribute `2b` of them and the unary ones `u`; but a
tree of `n = L + b + u` nodes has exactly `n − 1` edges, giving `L + b + u − 1 = 2b + u`, hence
`L = b + 1`. The leaves always outnumber the binary internal nodes by exactly one, no matter how many
unary operators are mixed in. For a mostly-binary tree that means the leaf fraction is `(b+1)/(2b+1)` —
three-fifths for a five-node tree, tending to one-half as trees grow — so a uniform crossover point
lands on a leaf roughly half the time, and a leaf swap merely trades one variable or constant for a
subtree: often a near-trivial edit that shuffles no structure. That ~50% rate of low-information
crossovers is the tax the uniform rule pays, and it is precisely what a 90/10 function-node weighting
buys back. I am deliberately *not* taking it at this rung: the point of standard GP is to establish the
plainest possible Koza-style loop and see where it lands, uniform point selection is that plainest
version, and adding the weighting now would muddy the question of what the bare engine does. It is a
real improvement I am holding in reserve rather than spending on the floor.

Let me trace one crossover by hand, because the whole representation rests on the splice always being
valid and I want to watch it happen. Take `parent1 = add(x0, mul(x1, x0))`; its `get_all_nodes`
preorder is index 0 `add` (root, no parent), 1 `x0` (child 0 of `add`), 2 `mul` (child 1 of `add`), 3
`x1` (child 0 of `mul`), 4 `x0` (child 1 of `mul`), size 5. Take the donor `parent2 = sin(x0)`: index 0
`sin`, index 1 `x0`, size 2. Say the offspring point lands on 2 (the `mul` subtree) and the donor point
on 0 (the whole `sin(x0)`). I copy the donor subtree `sin(x0)`, read `off_nodes[2] = (mul, parent=add,
child_idx=1)`, and since the parent is not `None` I set `add.children[1] = sin(x0)`. The offspring is
`add(x0, sin(x0)) = x0 + sin(x0)` — a syntactically valid, evaluable expression, its depth 2 well under
the cap, accepted. Nothing about which node I picked could have produced an illegal tree, because both
endpoints were valid subexpressions and every operator is protected; the trace just confirms the
closure property is doing the work the representation promised. And the depth cap lives right here as
the bloat brake: if crossover ever produces a child deeper than the cap, I reject it and return a copy
of `parent1`.

**Mutation.** Crossover has a built-in ceiling: it can only ever shuffle material *already present* in
the population. If a primitive or a useful constant was never drawn in generation zero, or died out
when a subpopulation went extinct, no amount of recombination brings it back — crossover splices
existing pieces, it cannot invent one. So I need a source of fresh material. Subtree mutation, mirroring
crossover: pick a node in a copy of the parent, delete the subtree there, and graft in a *freshly
generated* random subtree via `generate_tree('grow', 3, n_features)`. It is worth checking whether the
gene pool really can lose a primitive, because if every primitive were permanently abundant the
mutation operator would be redundant. At generation zero the risk is low: if a tree's internal nodes
draw roughly uniformly from the eight functions, a tree with about ten internal nodes fails to contain
a given primitive — say `log` — with probability `(7/8)¹⁰ ≈ 0.26`, so about 74% of trees carry at least
one, and across 500 fresh trees I expect on the order of 370 with a `log` somewhere. The primitive is
not scarce at the start. Scarcity is a *dynamic* event: once tournaments and crossover pull the
population toward one family, the lineages that happened to carry `log` can go extinct, and from that
point no recombination reintroduces it. That is exactly the hole mutation plugs — a shallow grow at
depth 3, at most `2⁴ − 1 = 15` nodes, small enough not to detonate a working tree but large enough to
reinject a lost operator or a fresh constant. The low rate follows from the same accounting: mutation
is disruptive — it throws away a working subtree for random noise — so it is a repair valve against
extinction, not an engine of improvement, and the search should lean overwhelmingly on crossover to
assemble building blocks and fire mutation sparingly just to keep the pool from going extinct. Same
depth cap: if the mutated child is too deep, return the parent unchanged.

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

Now I reason about what this floor must do, because that is the entire point of running it — it is the
rung every later rung diagnoses. On a benchmark where the answer is *reachable* — a polynomial like
Koza-3's `x⁵ − 2x³ + x`, which is just repeated multiplication and addition of `x`, well inside the
`{add, sub, mul}` part of the function set and the depth cap — I expect standard GP to do well and
consistently. The building blocks are simple, crossover assembles them, and there is no transcendental
the search has to gamble its way into. Let me check the target is actually reachable inside the budget,
because "reachable" is easy to assert and worth one line of arithmetic. `x⁵` is `x·x·x·x·x`, which as a
balanced product `mul(mul(x, x), mul(x, mul(x, x)))` is four `mul` nodes at depth 3; `2x³` is
`mul(2, mul(x, mul(x, x)))`, depth 3; and the whole expression `sub(add(x⁵, x), 2x³)` wraps those in a
`sub` over an `add`, landing around depth 5 with roughly a dozen internal nodes — comfortably inside the
depth-8 offspring cap and using only `{add, sub, mul}`, no protected transcendental and no constant more
exotic than a `2` an ephemeral draw can supply. So the exact form sits well within what crossover can
assemble, and I would be surprised to see Koza-3 do anything but cluster near 1.0 across all three seeds. The two harder benchmarks are where I expect the bare engine to show its
seams. Nguyen-7 needs the search to assemble *logarithms* and then extrapolate to a wider test range
than it trained on; Nguyen-10 needs a *product* of a sine and a cosine in two variables. Both require
the search to find and protect a specific transcendental building block, and here the failure mode I
am most worried about is precisely the one tournament selection only *partly* fixes: premature
convergence. With nothing but raw-MSE tournaments and no pressure toward diversity or against bloat
beyond the depth cap, a single lineage that gets an early lead on the training sample can dominate the
tournaments, the population homogenizes, and the search gets stuck on a tree that fits the 20 training
points but is the wrong *form* — which then generalizes badly to the held-out grid.

The extrapolation is worth making concrete, because it is where a wrong form is punished hardest and it
tells me what shape the failure should take. Nguyen-7 trains on `x ∈ [0, 2]` but tests on `x ∈ [−0.5,
2.5]` — half a unit past each end and, crucially, into `x < 0`, which the search never sees. The true
target is gentle there: at `x = 2.5` it is `log(3.5) + log(7.25) ≈ 1.253 + 1.981 = 3.234`, and at
`x = −0.5` it is `log(0.5) + log(1.25) ≈ −0.693 + 0.223 = −0.470`. A tree that threads the twenty
training points with the wrong machinery — an `exp` of a growing argument, or a division that nearly
poles just outside `[0, 2]` — stays finite thanks to the protected operators but saturates hard:
`protected_exp` clips its exponent to `[−10, 10]`, so a runaway exponentiation still returns up to
`e¹⁰ ≈ 22026`, four orders of magnitude past the target's ~3. One such test point alone contributes a
squared residual of order `10⁸` to a total whose variance is of order 1, so `R² = 1 − SS_res/SS_tot`
goes sharply negative — and whether it reports as a large negative or the loop floors it at 0, that seed
is a catastrophe. Because which form the population locks onto is decided by which lineage happened to
win the early tournaments, this should be a coin flip across seeds rather than a uniform sag: some seeds
find the right form and score near 1.0, others lock onto a wrong form early and blow up. So I expect the
signature of this rung to be **high seed-to-seed variance on the transcendental benchmarks** — a good
mean dragged down by one or two seeds that converged prematurely onto the wrong expression — with
Koza-3 steady and high throughout. Nguyen-10 should show the same disease in a milder form: no single
`e¹⁰` blow-up (its target `2·sin(x)·cos(y)` is bounded, so a wrong bounded form fits partially rather
than diverging), but a scattered spread as the search keeps locking onto partial fits of the product.

That diagnosis already points at the next rung. If the problem is premature convergence and bloat —
one lineage taking over, trees growing without earning fitness — then the lever is *selection*: shape
what selection rewards so it stops handing the whole population to a single early winner or to ever-
larger trees. I expect standard GP to be the weakest rung by construction: it is the plain engine with
no defense against its own convergence dynamics, and on the benchmarks where finding and keeping the
right transcendental form is the whole difficulty, that lack of defense is exactly what should cost it.
