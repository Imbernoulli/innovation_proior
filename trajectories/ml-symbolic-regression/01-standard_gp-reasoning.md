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

Now I fill the contract one function at a time, forcing each choice to justify itself.

**Fitness.** The task is a least-squares fit of the candidate's output to the targets over the sample
points. The natural error is the mean squared error: predict over the sample, subtract the targets,
square, average. Lower is better. Why squared and not absolute? Because least squares is the objective
I actually care about, it penalizes large misses heavily, and — the part that matters for selection —
it gives a clean total order on candidates. Selection only ever needs to ask "is A better than B," and
"smaller MSE is better" answers that directly. I keep `fitness_function` as raw MSE with no penalty
term baked in, because the outer loop tracks the best tree *by this number* for elitism, stopping, and
the final report; if I quietly mixed a complexity penalty into the fitness, the loop would report and
preserve the wrong tree. Whatever shaping a later rung wants belongs in selection, not here.

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
no global sum. So selection becomes a size-7 tournament minimizing MSE, returning copies so a parent
can be drawn and reused without ever being mutated in place.

**Crossover.** The core operator, and the one the representation makes natural. To make a child: copy
`parent1` (so the original is never damaged — it may be selected again), copy `parent2` as the donor,
pick a node in each, and replace the copy's chosen subtree with the donor's. Because both endpoints are
valid subexpressions, the child is always valid. Now the question is how to pick the two points. The
honest, simple choice — and the one this rung commits to — is **uniform** over the nodes. I pick the
offspring's crossover point from `1 .. size−1` (excluding the root, index 0, so a single crossover
never wholesale-replaces the whole offspring with the donor in the common case) and the donor's point
from `0 .. size−1` (the donor may contribute its entire tree). There is a known refinement here —
weighting function nodes more heavily than terminals, because in a binary-operator tree most nodes are
leaves and a uniform point therefore usually swaps a single terminal rather than a meaningful
subexpression — but I am deliberately *not* taking it at this rung. The point of standard GP is to
establish the plainest possible Koza-style loop and see where it lands; uniform point selection is that
plainest version, and adding the 90/10 weighting now would muddy the question of what the bare engine
does. If crossover ever produces a child deeper than the cap, I reject it and return a copy of
`parent1` — the depth cap is the bloat brake and it lives right here.

**Mutation.** Crossover has a built-in ceiling: it can only ever shuffle material *already present* in
the population. If a primitive or a useful constant was never drawn in generation zero, or died out
when a subpopulation went extinct, no amount of recombination brings it back — crossover splices
existing pieces, it cannot invent one. So I need a source of fresh material. Subtree mutation, mirroring
crossover: pick a node in a copy of the parent, delete the subtree there, and graft in a *freshly
generated* random subtree via `generate_tree('grow', 3, n_features)`. A shallow grow (depth 3) keeps
the injected noise small, and it reintroduces operators and constants the gene pool may have lost.
Mutation is disruptive — it throws away a working subtree for random noise — so its rate must be small;
the search should lean overwhelmingly on crossover to assemble building blocks and use mutation
sparingly just to keep the pool from going extinct. Same depth cap: if the mutated child is too deep,
return the parent unchanged.

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
the search has to gamble its way into; I would be surprised to see Koza-3 do anything but cluster near
1.0 across all three seeds. The two harder benchmarks are where I expect the bare engine to show its
seams. Nguyen-7 needs the search to assemble *logarithms* and then extrapolate to a wider test range
than it trained on; Nguyen-10 needs a *product* of a sine and a cosine in two variables. Both require
the search to find and protect a specific transcendental building block, and here the failure mode I
am most worried about is precisely the one tournament selection only *partly* fixes: premature
convergence. With nothing but raw-MSE tournaments and no pressure toward diversity or against bloat
beyond the depth cap, a single lineage that gets an early lead on the training sample can dominate the
tournaments, the population homogenizes, and the search gets stuck on a tree that fits the 20 training
points but is the wrong *form* — which then generalizes badly to the held-out grid. Because that
stuck-point is a matter of which lineage happened to win the early tournaments, it should be highly
**seed-dependent**: some seeds find the right form and score near 1.0, others lock onto a wrong form
early and score far lower, and on the extrapolating Nguyen-7 a wrong form can blow up off the training
range badly enough that `r2_score` floors it at 0 — or, if the predictions diverge hard, the raw R²
goes sharply negative before the floor. So I expect the signature of this rung to be **high
seed-to-seed variance on the transcendental benchmarks**: a good mean dragged down by one or two seeds
that converged prematurely onto the wrong expression, with Koza-3 steady and high throughout.

That diagnosis already points at the next rung. If the problem is premature convergence and bloat —
one lineage taking over, trees growing without earning fitness — then the lever is *selection*: shape
what selection rewards so it stops handing the whole population to a single early winner or to ever-
larger trees. I expect standard GP to be the weakest rung by construction: it is the plain engine with
no defense against its own convergence dynamics, and on the benchmarks where finding and keeping the
right transcendental form is the whole difficulty, that lack of defense is exactly what should cost it.
