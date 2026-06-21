The object I want back from a multiobjective problem $f(x) = (f_1(x), \dots, f_M(x))$, with all objectives minimized, is not a single point but a whole set. When the objectives conflict, improving one forces another to get worse, so the meaningful answer is the Pareto-optimal set — the points that no feasible point can beat on every objective at once, where $x$ dominates $y$ iff $f_m(x) \le f_m(y)$ for all $m$ with strict inequality somewhere. I want to recover a good approximation of that entire front in one run, so a decision-maker can pick a trade-off afterward. A population-based evolutionary algorithm is the natural tool, because it already carries a hundred candidates at once; if I can spread them along the front I harvest the whole trade-off surface in a single sweep. That hands the algorithm two jobs that pull against each other: convergence, driving the population down onto the true front, and diversity, keeping the survivors spread out across the front and reaching its two extremes. Push only on convergence and everyone piles onto one easy corner and I get a useless dense blob; push only on diversity and the population fans out but never lands. Both, at once, cheaply, is the whole game.

The standard recipe in front of me ranks the population into Pareto layers (Goldberg's idea — peel off the non-dominated set as rank 1, then the non-dominated set of the rest as rank 2, and so on) and keeps it spread with Goldberg & Richardson's fitness sharing, which degrades an individual's fitness by how crowded its neighborhood is via a sharing kernel of radius $\sigma_{\text{share}}$, niche count $m_i = \sum_j sh(d_{ij})$, and shared fitness $F_i / m_i$. It works, but three things grind on me, and the fix has to hit all three. First, cost: finding rank 1 the naive way compares every solution against every other across $M$ objectives, $O(MN^2)$, and then I discard front 1 and repeat, in the worst case $N$ times — $O(MN^3)$, a cubic wall that caps how large $N$ can grow before a generation crawls. Second, no elitism: the scheme is generational, so I build children and the parents vanish, and the studies keep reporting that letting the best-so-far material compete in the next survival decision measurably speeds convergence and prevents regressions where a good front representative is lost merely because a generation boundary passed. Third, the $\sigma_{\text{share}}$ knob: the spread depends heavily on a distance scale I have to guess (along with an implicit guess at how many niches the front supports), and computing every niche count is itself $O(N^2)$. The elitist baselines that already exist buy me pieces of this but not the whole: SPEA and PAES get elitism through an external archive, but the archive is a second data structure with its own management — how big, when to prune, how to cluster, how to fold back into selection — and PAES's diversity rests on a hard grid with its own depth parameter; Rudolph's elitist GA proves convergence but has no mechanism for spread, so its solutions clump. Each fix must improve the algorithm without adding another fragile knob.

I propose NSGA-II, and it closes all three gaps with three interlocking moves. I attack elitism first, because it has the most leverage and reframes everything else. The reason I would want an archive is to stop good parents from disappearing before they can be compared with their children — but there is a cheaper way to get exactly that competition: do not throw the parents away. At generation $t$ I have parents $P_t$ of size $N$; I run selection, crossover, and mutation to make offspring $Q_t$ of size $N$; then I merge into one combined pool $R_t = P_t \cup Q_t$ of size $2N$ and choose the survivors from that pool. Now every parent sits in $R_t$ competing for a survivor slot on equal footing with every child. I sort $R_t$ by non-domination, take front $F_1$, then $F_2$, and fill $P_{t+1}$ front by front until I have collected $N$. Whole better-ranked fronts are carried forward when they fit; elitism becomes a property of the survival pool, with no separate archive subsystem at all. The price is that the thing I sort doubled from $N$ to $2N$, which makes the sorting cost matter even more.

So the cubic sort has to go, and the trick is to notice that the naive method recomputes facts that never change — the dominance relationship between any pair is fixed, yet peeling fronts re-examines it over and over. I compute each pair's relationship once. In a single $O(MN^2)$ pass over all pairs I record, for each solution $p$, a domination count $n_p$ (how many solutions dominate $p$) and a dominated set $S_p$ (the solutions $p$ dominates): for an unordered pair, if $p$ dominates $q$ I add $q$ to $S_p$ and increment $n_q$, and symmetrically the other way. Every solution with $n_p = 0$ is dominated by nobody and forms front 1. To peel the next front I walk each $p$ in the current front and, for each $q \in S_p$, decrement $n_q$ by one — removing the just-placed dominators — and any $q$ whose $n_q$ hits zero has lost its last dominator and joins the next front. This is genuinely quadratic, not secretly cubic: each solution sits in exactly one front, so the outer "for each $p$ in this front" body runs $N$ times total across all fronts combined, and the inner loop only walks stored $S_p$ entries, at most $N(N-1)$ visits over the whole run, each a counter decrement rather than a fresh $M$-objective comparison. The total is $O(MN^2 + N^2) = O(MN^2)$, traded against $O(N^2)$ storage for the $S_p$ sets — a trade I take gladly, because memory is cheap and the cubic was the real enemy.

The third move kills $\sigma_{\text{share}}$ entirely. Fitness sharing asks "how crowded is your neighborhood?" and answers with a kernel of radius $\sigma_{\text{share}}$ — the radius is the parameter. But within a single front I can read crowding straight off the neighbors without any radius: a solution whose immediate neighbors along the front are far away sits in a sparse, under-represented stretch and should be protected, while one whose neighbors hug it is in a dense cluster and is expendable. I turn this into a number called the crowding distance. For each objective $m$ I sort the front by $m$ and, for an interior solution $i$, take the gap its two neighbors leave around it, $f_m(i{+}1) - f_m(i{-}1)$ — the width of the slab $i$ occupies in that objective — then sum these gaps over all objectives. Geometrically this is the sum of side lengths of the cuboid whose opposite faces pass through $i$'s nearest neighbors in each objective direction: a parameter-free estimate of the empty space around $i$, where the front's own spacing sets the scale automatically. Two details make or break it. The objectives live on wildly different scales, so I normalize each objective's gap by that objective's spread across the front before summing — the interior update is

$$ i.\text{distance} \mathrel{+}= \frac{f_m(i{+}1) - f_m(i{-}1)}{f_m^{\max} - f_m^{\min}}, $$

with DEAP storing the *average* normalized side length, so its finite update carries the common factor $M$ in the denominator, $\frac{f_m(i+1) - f_m(i-1)}{M\,(f_m^{\max} - f_m^{\min})}$, a constant scaling that preserves the within-front ordering; if $f_m^{\max} = f_m^{\min}$ that objective contributes no finite gap. The second detail is the boundary solutions — the ones with the smallest and largest value of an objective have no neighbor on one side and an undefined gap, but these are precisely the extreme points of the front, the ends I most want to keep, so I assign them *infinite* crowding distance. That is not a hack to dodge an undefined value; it is the correct preference, because pinning the extremes is how the population reaches the ends of the front instead of curling inward. Assigning crowding distance costs only $O(MN \log N)$ — cheaper than the sort — and has no parameter.

These two numbers per individual, its non-domination rank and its crowding distance, meet in exactly one place: a single lexicographic comparison I call the crowded-comparison operator,

$$ i \prec_n j \quad\text{iff}\quad \text{rank}_i < \text{rank}_j, \;\text{ or }\; (\text{rank}_i = \text{rank}_j \;\text{ and }\; \text{distance}_i > \text{distance}_j). $$

Convergence speaks first — a better (lower) front always wins, because getting to the front is the primary objective — and only among equals in rank does diversity get a vote, preferring the less crowded (larger crowding distance) solution. This is a strict priority, never a weighted average with a coefficient I would have to tune, and it is the only place the two pressures interact. I use it in two roles. For environmental selection I non-domination-sort the $2N$ pool, add whole fronts to $P_{t+1}$ in order while they fit, and for the first front $F_l$ that would overflow the remaining slots I compute crowding distances within $F_l$, sort it by crowding distance descending, and take the top however-many I need to reach exactly $N$ — the least-crowded boundary-front members survive, the most-crowded are dropped. For mating, the code path uses DEAP's dominance-then-crowding tournament: draw candidates, and if one directly dominates the other it wins, otherwise the larger crowding distance wins with exact ties broken by a coin flip — so mating pressure already favors solutions both near the front and in sparse regions, seeding the next generation toward the under-represented parts.

Turning selected parents into offspring on real-coded variables uses simulated binary crossover and polynomial mutation, and both are chosen for a property that matches the front-spreading problem. SBX imitates single-point binary crossover in continuous space through the spread factor $\beta = |(c_2 - c_1)/(p_2 - p_1)|$, the ratio of child spread to parent spread, sampled from $P(\beta) = 0.5(\eta_c{+}1)\beta^{\eta_c}$ for $\beta \le 1$ and $0.5(\eta_c{+}1)/\beta^{\eta_c+2}$ for $\beta > 1$; drawing $u \sim U[0,1]$ and inverting the CDF gives $\beta_q = (2u)^{1/(\eta_c+1)}$ for $u \le 0.5$, else $(1/(2(1-u)))^{1/(\eta_c+1)}$, and the children land symmetrically around the parents' midpoint, $c_1 = 0.5[(1{+}\beta_q)p_1 + (1{-}\beta_q)p_2]$ and $c_2 = 0.5[(1{-}\beta_q)p_1 + (1{+}\beta_q)p_2]$. Two properties make it right: the density peaks at $\beta = 1$ so children land near their parents with high probability (a local search by default, refining good solutions), and because $\beta$ is a *ratio*, the spread of the children is proportional to the spread of the parents — when the population is scattered early, children scatter and explore; as it converges and parents bunch up, children bunch up and refine, so the operator self-adapts from exploration to refinement with nothing scheduled. Polynomial mutation applies the same philosophy to a single variable in bounded form: with normalized room to the bounds $\delta_L = (x - x_L)/(x_U - x_L)$, $\delta_R = (x_U - x)/(x_U - x_L)$, $u \sim U[0,1]$, and $\text{mut\_pow} = 1/(\eta_m+1)$, a left step uses $xy = 1 - \delta_L$, $\text{val} = 2u + (1{-}2u)\,xy^{\eta_m+1}$, $\delta_q = \text{val}^{\text{mut\_pow}} - 1$, and a right step uses $xy = 1 - \delta_R$, $\text{val} = 2(1{-}u) + 2(u{-}0.5)\,xy^{\eta_m+1}$, $\delta_q = 1 - \text{val}^{\text{mut\_pow}}$, giving $x' = x + \delta_q (x_U - x_L)$. Both use the bound-respecting forms — the bounded SBX recomputes its branches against each variable's distance to its $[x_L, x_U]$ box so no child escapes, with a final clamp only as a numerical guard — which is cleaner than rejection sampling, since rejection would bias the distribution and waste evaluations. I set $\eta_c = \eta_m = 20$ for moderately local variation, crossover probability $p_c = 0.9$ so recombination stays the main engine of progress, and mutation probability $p_m = 1/n$ so on average exactly one variable per individual mutates — the natural scale, because it makes the expected number of mutations per offspring independent of dimensionality.

One more thing folds in for free: constraints, handled without a penalty parameter so the parameter-free spirit survives. I extend dominance itself into constrained-domination — $i$ constrained-dominates $j$ if (1) $i$ is feasible and $j$ is not, or (2) both are infeasible but $i$ has the smaller total constraint violation, or (3) both are feasible and $i$ dominates $j$ in the ordinary Pareto sense. Any feasible solution outranks any infeasible one, less-violating infeasibles rank higher (pulling the population toward feasibility through the constraint boundaries), and feasibles fall back to ordinary Pareto sorting. It changes only the comparison primitive, slots straight into the same non-dominated sort with nothing downstream touched, and reduces to ordinary dominance when unconstrained. Accounting for one generation: the non-dominated sort of the $2N$ pool is $O(MN^2)$, crowding-distance assignment is $O(MN \log N)$, and the final front sort is $O(N \log N)$, so the whole algorithm runs at $O(MN^2)$ per generation, governed by exactly the piece I worked to bring down from cubic. The full loop initializes a random population in the bounds and evaluates it, sorts and assigns crowding so the first tournament has its information, then each generation runs the dominance/crowding tournament, applies SBX and polynomial mutation, evaluates, merges parents and offspring into the $2N$ pool, fast-non-dominated-sorts it, fills the next population front by front, and truncates the first overflowing front by crowding distance to reach exactly $N$.

Filling the single generation-strategy slot of the generational MOEA harness with DEAP's `sortNondominated`, `assignCrowdingDist`, `selTournamentDCD`, `cxSimulatedBinaryBounded`, and `mutPolynomialBounded`:

```python
from copy import deepcopy
import random
from operator import attrgetter

from deap import tools
from deap.tools.emo import assignCrowdingDist


class NSGA2Strategy:
    """One-generation strategy for a real-coded multiobjective EA."""

    def __init__(self, pop_size, n_var, bounds,
                 cx_prob=0.9, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        if pop_size % 4 != 0:
            raise ValueError("selTournamentDCD requires pop_size divisible by 4")
        self.pop_size = pop_size
        self.n_var = n_var
        self.bounds = bounds                                    # (low, up)
        self.cx_prob = cx_prob                                  # p_c = 0.9
        self.cx_eta = cx_eta                                    # eta_c = 20
        self.mut_eta = mut_eta                                  # eta_m = 20
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def _assign_selection_crowding(self, population):
        fronts = tools.sortNondominated(population, len(population),
                                        first_front_only=False)
        for front in fronts:
            assignCrowdingDist(front)

    def clone_and_vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        low, up = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.cx_prob:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=low, up=up)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=low, up=up, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    @staticmethod
    def evaluate_invalid(population, evaluate):
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)

    def generation(self, population, evaluate):
        self._assign_selection_crowding(population)
        parents = tools.selTournamentDCD(population, self.pop_size)
        offspring = self.clone_and_vary(parents)
        self.evaluate_invalid(offspring, evaluate)

        combined = population + offspring                       # elitist 2N pool
        fronts = tools.sortNondominated(combined, self.pop_size,
                                        first_front_only=False)
        for front in fronts:
            assignCrowdingDist(front)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
                continue
            remaining = self.pop_size - len(next_gen)
            front = sorted(front, key=attrgetter("fitness.crowding_dist"), reverse=True)
            next_gen.extend(front[:remaining])
            break
        return next_gen
```
