Let me start from what actually hurts. I have a black-box objective `f(x)` over a box of continuous reals, and I want to run a genetic algorithm on it because the population-based, crossover-driven search has a good track record. But that track record is on *discrete* problems, and the whole story for why it works there is single-point crossover propagating building blocks — short high-fitness substrings — between parents. When my variables are reals, the standard thing to do is binary-encode each variable in some fixed number of bits and run an ordinary binary GA on the concatenated string. And every time I do that I pay for it. The precision is capped by the bit length, so I can never get arbitrarily close to an optimum that sits between two representable values. The mapping from strings to reals is fixed and the algorithm can't bend it. Worst of all is the Hamming cliff: two reals that are neighbours in value can be maximally far apart in bits — `0111` and `1000` differ in every single position even though they're adjacent integers — so to nudge a variable across that boundary the crossover or mutation has to flip several bits at once, which it almost never does, and the search gets stuck on the wrong side of the cliff. I keep fighting the encoding instead of the problem.

So the obvious wish: work directly on the real numbers. No bits, no precision cap, no fixed mapping, no Hamming cliff. People have tried this — real-coded GAs exist. Wright's linear crossover takes two parents `p1, p2` and forms three points, the midpoint `(p1+p2)/2` and the two reflections `1.5·p1 − 0.5·p2` and `−0.5·p1 + 1.5·p2`, then keeps the best two. But stare at that: from any pair of parents it can only ever produce three candidate children. That's almost deterministic. If the good region happens to sit somewhere off those three points, the operator can't reach it. Whatever "search power" means precisely, this has almost none. Eshelman and Schaffer's BLX-α is better — for `p1 < p2` it draws the child uniformly from the widened interval `[p1 − α(p2−p1), p2 + α(p2−p1)]`, so with α = 0.5 it can land anywhere in a box a bit larger than the parents' span, and they reported BLX-0.5 working well. That's a genuinely stochastic real operator. But the offspring density is *flat* across that box — every point equally likely. And I have a nagging feeling that flat is wrong, because that's not how the binary operator I'm trying to replace actually behaves.

That's the thing I should pin down before I design anything: how does single-point crossover actually distribute its children, on the decoded real values, so I have a target to hit? Take two parent bitstrings, cut at a random site `k` from the right, swap the tails. Write a decoded value as `x = B·2^k + A`, where `A` is the integer from the right `k` bits and `B` from the left bits. Each child keeps its own parent's `B` and takes the other's `A`: `y1 = B1·2^k + A2`, `y2 = B2·2^k + A1`. Add them: `y1 + y2 = (B1+B2)·2^k + (A1+A2)`, and the parents sum to `x1 + x2 = (B1+B2)·2^k + (A1+A2)` — identical. So `(y1+y2)/2 = (x1+x2)/2`. The children's mean equals the parents' mean. And under a linear bit-to-real mapping `c = m·y + t` this carries straight over to the reals: the children sit symmetrically about the parents' midpoint, equidistant from it. That's a real, structural property of single-point crossover, and it feels important — it says the operator doesn't drift the population's per-variable centroid, it just spreads around it.

Now, *how much* does it spread? The natural dimensionless way to ask is to compare how far apart the children are with how far apart the parents are. Let me name that ratio. Call it the spread factor, `β = |c1 − c2| / |p1 − p2|`. If β < 1 the children are enclosed by the parents — a contracting crossover. If β > 1 the children enclose the parents — expanding. β = 1 is stationary, children land on the parents. The lovely thing about a *ratio* is that it's relative: the same β gives close children when the parents are close and distant children when the parents are far apart. So an operator built around a fixed distribution of β would automatically explore widely while the population is diverse and narrow its search as the population converges and parents get close — exploration that anneals itself, for free, just from the geometry. BLX-α also scales its interval with the parent span, so the geometry is not the problem there; the problem is the flat density inside that interval. It gives me width, but not the decoded single-point crossover *shape*. I want the shape.

So what shape does single-point crossover actually produce in β? I can nail this down exactly for a clean special case: take the two most different possible parents, all-ones and all-zeros, each `ℓ` bits long, and cross them at a site `k` bits from the right. Reusing the `x = B·2^k + A` decomposition, the all-ones parent has `A1 = 2^k − 1` (its low `k` bits) and `B1 = 2^{ℓ−k} − 1` (the rest); the all-zeros parent has `A2 = B2 = 0`. Swapping `A` gives the children `y1 = B1·2^k + A2 = 2^ℓ − 2^k` and `y2 = B2·2^k + A1 = 2^k − 1`, against parents `x1 = 2^ℓ − 1`, `x2 = 0`. So

  β(k) = |y1 − y2| / |x1 − x2| = (2^ℓ − 2^{k+1} + 1) / (2^ℓ − 1),

a closed form in `k` alone. For `ℓ = 20` this sits within a hair of 1 for almost every cut site and only plunges toward 0 in the last few sites near the most-significant end — cut low and the children barely move; cut high and the value changes a lot.

A cut site is uniform over the `ℓ − 1` positions, so the *density* of β is the density of `k` pushed through this map: `C(β) ∝ |dk/dβ| = 1/|dβ/dk|`. Differentiating the large-`ℓ` approximation `β(k) ≈ 1 − 2^{1−ℓ}·2^k` gives `dβ/dk ≈ −ln2·(1 − β)`, so

  C(β) ∝ 1/(1 − β),   0 ≤ β ≤ 1.

That's not merely rising toward β = 1 — it diverges there. Which is a problem, not a solution: `∫₀¹ 1/(1−β) dβ` doesn't converge, so this exact curve can't be renormalized into a probability distribution at all, and it's only valid for the single most extreme parent pair — other bit patterns will pull the true shape inward, less peaked than this. So I can't use `C(β) ∝ 1/(1−β)` directly as a sampling density. But it tells me exactly what shape I'm chasing: something that climbs toward β = 1 the way this does, without the blow-up, and general enough to cover ordinary, non-extreme parent pairs too. BLX-0.5's flat density is nowhere close — it makes a far-out child as likely as a near one, when the exact calculation says the near ones should dominate heavily.

And there's a symmetry I can exploit so I only have to model one side. If I take the two children of a contracting crossover and cross *them* at the same site, I get the original parents back. So every contracting crossover with spread β is in one-to-one correspondence with an expanding crossover of spread 1/β — they're the same event run backwards. That means the total probability mass on contracting crossovers equals the mass on expanding ones, each one half. And it gives me a transformation rule: if I write the contracting density as `C(β)` for 0 ≤ β ≤ 1, the matching expanding density must be `E(β) = (1/β²)·C(1/β)` for β > 1 — the 1/β² is the Jacobian of the `β ↦ 1/β` change of variables. So I only need to choose `C` on [0,1]; the expanding side is forced.

Now I get to design the operator. I want a `C(β)` that (1) increases toward β = 1, matching the rising shape I just derived; (2) is dead simple to sample from, because this runs on every variable of every mating every generation; and (3) has a knob to tune how concentrated it is. The simplest family that increases toward 1 is a power law. Let me try `C(β) = c·β^η` on [0,1] with η ≥ 0 a parameter I'll call the distribution index. Fix the constant by the half-mass requirement: `∫₀¹ c·β^η dβ = c/(η+1)` should equal 0.5, so `c = 0.5(η+1)`. So

  C(β) = 0.5(η+1)·β^η,    0 ≤ β ≤ 1.

Push it through the symmetry rule for the expanding side. `C(1/β) = 0.5(η+1)·(1/β)^η = 0.5(η+1)·β^{−η}`, times `1/β²`:

  E(β) = 0.5(η+1)·β^{−(η+2)} = 0.5(η+1) / β^{η+2},    β > 1.

The expanding-side mass comes out to the matching 0.5 automatically: `∫₁^∞ 0.5(η+1)·β^{−(η+2)} dβ = 0.5(η+1)·(1/(η+1)) = 0.5`, as it must, since the β ↦ 1/β change of variables preserves total probability by construction.

What does η do? `C(β) = 0.5(η+1)β^η` — for large η this is sharply peaked at β = 1, so children land very near the parents: focused, exploitative search. For small η it flattens. At η = 0, `C(β) = 0.5`, so the contracting half is uniform on [0,1], matching the inside-the-parents behaviour of BLX-0.0 / uniform crossover; the full operator still keeps the symmetry-forced expanding tail `0.5/β²`, so it is not just BLX-0.0. The role of η is the same role the inverse temperature 1/T plays in simulated annealing: small η is a hot, broad search, large η a cold, narrow one.

How concentrated should η actually be? The exact extreme-pair calculation says "arbitrarily" — `C(β) ∝ 1/(1−β)` blows up at β = 1 — but that curve isn't realizable as a distribution, and it only describes the single most extreme parent pair. To generalize I have to redo the accounting for ordinary bit patterns: cross every pair of strings at every possible site instead of just the two extremes, and histogram the resulting β. That gives a milder, finite version of the same rising shape, and a moderate η, in the 2-to-5 range, is what actually tracks that broader measurement most closely — push η much higher and the polynomial hugs β = 1 harder than single-point crossover itself does. Concretely, the mass `C` assigns a bin `[a,b]` is `0.5(β^{η+1})` from `a` to `b` (since `∫ 0.5(η+1)β^η dβ = 0.5 β^{η+1}`): at η = 2, the top tenth [0.9, 1.0) gets `0.5(1 − 0.9³) = 0.136` of the 0.5 contracting mass — 27% of it; at η = 20 it gets `0.5(1 − 0.9²¹) ≈ 0.446` — 89% of it, a far sharper concentration than the measured single-point shape actually calls for.

So a large η isn't a closer copy of the binary operator; it's a deliberate departure from it, toward an operator that leans harder on "stay near the parents" than single-point crossover typically does. That's a legitimate design point in its own right — still recovering the building-block instinct, just committing to it more — and because η is a single continuous knob I don't have to pick once and for all: η ≈ 2 stays closer to the measured binary behavior and is the more exploratory setting; η ≈ 20 trades that fidelity for a tighter, more exploitative search, useful once I care more about raw optimization performance than about literally reproducing the single-point statistic. (The ideal would anneal η across generations — broad early, focused late — but I'll fix one value for the whole run.) I'll ship η = 20 as the default, with η ≈ 2 in mind as the closer-to-original, more exploratory alternative.

Now the part that has to be cheap: given this density, how do I actually draw a β? Inverse-transform sampling — compute the cumulative distribution, set it equal to a uniform `u`, invert. And here's where the polynomial pays off, because both the CDF and its inverse are closed-form, no special functions. The contracting half: `∫₀^β 0.5(η+1)t^η dt = 0.5(η+1)·β^{η+1}/(η+1) = 0.5·β^{η+1}`. The total contracting mass is 0.5, so if I draw `u ∈ [0,1]` and it lands in the contracting half, `u ≤ 0.5`, I set the CDF equal to `u`: `0.5·β^{η+1} = u`, hence `β^{η+1} = 2u`, hence

  β = (2u)^{1/(η+1)},    u ≤ 0.5.

For the expanding half, `u > 0.5`, integrate the expanding density from 1 to β: `∫₁^β 0.5(η+1)t^{−(η+2)} dt = 0.5(η+1)·[t^{−(η+1)}/(−(η+1))]₁^β = 0.5·(1 − β^{−(η+1)})`. The total CDF at β is then `0.5 + 0.5(1 − β^{−(η+1)}) = 1 − 0.5·β^{−(η+1)}`. Set equal to `u`: `0.5·β^{−(η+1)} = 1 − u`, so `β^{−(η+1)} = 2(1−u)`, and

  β = (1 / (2(1−u)))^{1/(η+1)},    u > 0.5.

Two arithmetic operations and one power per coordinate. That's it — no Gaussian, no `erf⁻¹`, no rejection sampling. The polynomial wasn't an aesthetic choice; it's what makes the operator affordable at the rate I'll be calling it.

Now turn a spread β into actual children. I have two requirements already nailed down: the children must be equidistant from the parents' midpoint (the mean-preserving property I derived), and their separation divided by the parents' separation must equal β. Two equations. The symmetric linear combination that satisfies both is

  c1 = 0.5·[(1 + β)·x1 + (1 − β)·x2],
  c2 = 0.5·[(1 − β)·x1 + (1 + β)·x2].

By algebra: sum `c1 + c2 = 0.5·[(1+β+1−β)x1 + (1−β+1+β)x2] = x1 + x2`, so the midpoints coincide — mean preserved. Difference: `c1 − c2 = 0.5·[(1+β−1+β)x1 + (1−β−1−β)x2] = β(x1 − x2)`, so `|c1−c2|/|x1−x2| = β` — spread honored. Both requirements drop out of this one symmetric form. Concretely, with parents x1 = 2, x2 = 5 (sum 7, gap 3): at u = 0.10, β ≈ 0.926, giving c1 ≈ 2.111, c2 ≈ 4.889 — sum 7.000, `|c1−c2| = 2.778 = 0.926·3`; at u = 0.90, β ≈ 1.080, giving c1 ≈ 1.881, c2 ≈ 5.120, still summing to 7.000 but now straddling *outside* [2, 5]. So β < 1 pulls the children inward toward the midpoint, β > 1 pushes them outward past the parents, and in every case the centroid is pinned at the parents' mean. That is the two-property signature I extracted from single-point crossover — mean preservation plus a β-distribution concentrated near 1 — now reproduced on the raw reals. Call it simulated binary crossover.

I have a one-variable operator; I need it on a vector. The binary analogue tells me how. In single-point crossover on a concatenated multi-variable string, the cut lands inside one variable and that variable gets a real recombination while the variables on one side are swapped wholesale and the rest are untouched — so single-point crossover only disrupts a *subset* of variables, and which subset is tied to position, which introduces a positional bias. To avoid hard-wiring a positional bias, I can decide per variable, independently, whether to recombine it — a uniform-crossover-style choice. A natural setting is to apply the per-variable SBX with probability 0.5, mirroring the fact that single-point crossover recombines/swaps only part of the string; for a single-variable problem there's nothing to choose, so apply it with probability 1. For the concrete baseline code, though, I want to mirror the standard unbounded toolbox kernel exactly: once the outer crossover probability `p_c` selects a mating pair, the SBX formula is applied to every coordinate of that pair. If I need the 0.5 coordinate gate later, it is just a wrapper around the same kernel.

Crossover alone isn't a complete real-coded GA — I need a mutation operator to keep diversity and let the population escape a basin it has prematurely collapsed into, and I'd like that mutation to have the same character: perturb a variable to a nearby value, small perturbations more likely than large, and never step outside the box `[x_lo, x_hi]`. The same polynomial idea transplants cleanly. Instead of a spread between two parents, I want a *perturbation* `δ` of a single value, with a density peaked at δ = 0 and thinning toward ±1, controlled by its own index `η_m`. Take the same polynomial-shape density `∝ (1 − |δ|)^{η_m}` on `[−1, 1]`, peaked at 0. Draw `u ∈ [0,1]` and invert by the same trick. For `u < 0.5` the perturbation is negative: `δ = (2u)^{1/(η_m+1)} − 1`. For `u ≥ 0.5` it's positive: `δ = 1 − (2(1−u))^{1/(η_m+1)}`. Then I scale `δ` by the variable's available range and add it: `x' = x + δ·(x_hi − x_lo)`. Large `η_m` makes `δ` small (the perturbation is roughly of order `(x_hi − x_lo)/η_m`, a tiny local nudge); small `η_m` allows bigger jumps. The recommended range is `η_m` in [20, 100] for a controlled local search.

But a bare `±(range)` perturbation can step outside the box near a boundary — if `x` is already close to `x_lo`, a large negative `δ` overshoots below `x_lo`. I could just clip afterward, but clipping piles probability mass exactly on the boundary, which biases the search toward the walls. Better to bend the *density itself* so it can't produce an out-of-bounds value in the first place. Measure how much room there is on each side, normalized: `δ1 = (x − x_lo)/(x_hi − x_lo)` is the fraction of the range below `x`, `δ2 = (x_hi − x)/(x_hi − x_lo)` the fraction above. When I'm sampling a downward move I should only be able to reach down to `x_lo`, i.e. the negative perturbation is capped at `−δ1`; an upward move capped at `+δ2`. Fold those caps into the inversion by mixing the boundary-distance into the term being raised to the power. For `u < 0.5`,

  δ_q = [ 2u + (1 − 2u)·(1 − δ1)^{η_m+1} ]^{1/(η_m+1)} − 1,

and for `u ≥ 0.5`,

  δ_q = 1 − [ 2(1−u) + 2(u − 0.5)·(1 − δ2)^{η_m+1} ]^{1/(η_m+1)}.

Check the boundary behaviour: if `x` sits right at the lower bound, `δ1 = 0`, so for `u < 0.5` the bracket is `2u + (1−2u)·1 = 1`, giving `δ_q = 1^{1/(η_m+1)} − 1 = 0` — no downward move possible, exactly right, there's no room below. If `x` sits right at the upper bound, `δ2 = 0`, so the upward branch similarly gives `δ_q = 0`. At the other extreme of each branch, the cap is exact: as `u -> 0`, the downward branch gives `δ_q = -δ1`, so `x' = x_lo`; as `u -> 1`, the upward branch gives `δ_q = δ2`, so `x' = x_hi`. When the chosen side has the full normalized range available — `δ1 = 1` for a downward move or `δ2 = 1` for an upward move — the formula reduces to the unbounded polynomial perturbation; otherwise it smoothly compresses the tail before it hits the wall. The boundary-aware form interpolates between "no overshoot at the wall" and "the largest legal move reaches the opposite available bound," and `x' = x + δ_q·(x_hi − x_lo)`, then a final clip as a numerical safety net. Mutate each variable independently with a per-coordinate probability `indpb`. The binary world found `1/L` per bit best; the real-coded analogue is `indpb = 1/n`, so on average one variable per mutation pass is changed — enough to keep diversity alive without tearing good solutions apart.

The last piece is selection, and here I don't want to invent anything clever — I want robustness and no fitness scaling. Tournament selection: to pick one parent, grab `t` individuals uniformly at random and keep the best of them; repeat to fill the parent pool. It needs no sorting, no fitness normalization, no mapping of objective values to selection probabilities — just fitness comparisons. In the toolbox, "best" is whatever the fitness object says is best, so minimization is handled by a negative fitness weight rather than by rewriting the selection rule. The tournament size `t` *is* the selection-pressure knob: `t = 1` is random drift, large `t` is near-greedy, and `t = 3` is a mild pressure that keeps the population converging without collapsing diversity too fast — a good match for an operator suite (SBX + polynomial mutation) whose own job is to balance exploration and exploitation.

Now I can assemble the whole loop, real-coded end to end. Initialize a population of float vectors uniformly in the box and evaluate. Each generation: tournament-select a full pool of parents; clone them so I don't trample the current population; walk pairs and, with probability `p_c`, apply SBX to each pair; then walk individuals and, with probability `mut_prob`, send an individual through the mutation operator, whose own per-coordinate probability is `indpb = 1/n`; clip everyone back into the box as a safety step; re-evaluate whoever changed; replace the population with the offspring; record the best fitness. The defaults that match the operators' intent are the crossover index and the mutation index both at 20 (children and mutants hug their sources, the exploitative-but-not-frozen regime), per-coordinate mutation probability `1/n` inside the polynomial mutation kernel, and a tournament of size 3.

Let me write it as the code I'd actually run, filling the three operator slots, against the standard real-coded toolbox. The crossover is the unbounded SBX sampling and the symmetric offspring construction I derived, with the loop's clip step handling any crossover overshoot; mutation is the bounded polynomial kernel, with clipping left as a numerical guard.

```python
import random
import numpy as np
from typing import Callable, Tuple
from deap import base, tools


def custom_select(population: list, k: int, toolbox=None) -> list:
    # Tournament selection: best of 3 uniformly-sampled individuals, k times.
    # No fitness scaling/sorting; DEAP's Fitness weights define best.
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    # Simulated binary crossover, distribution index eta_c = 20.
    # Per coordinate: draw u, invert the polynomial CDF to a spread factor beta,
    # then build the two mean-preserving, beta-spread children.
    eta = 20.0
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        u = random.random()
        if u <= 0.5:                       # contracting half:  beta = (2u)^(1/(eta+1))
            beta = 2.0 * u
        else:                              # expanding half: beta = (1/(2(1-u)))^(1/(eta+1))
            beta = 1.0 / (2.0 * (1.0 - u))
        beta **= 1.0 / (eta + 1.0)
        # c1, c2 equidistant from the parents' midpoint, separation = beta*|x1-x2|
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    # Polynomial mutation, index eta_m = 20, per-variable rate 1/n.
    # Bounded form: the polynomial perturbation can never leave [lo, hi].
    eta = 20.0
    indpb = 1.0 / len(individual)
    for i in range(len(individual)):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - lo) / (hi - lo)         # room below x (normalized)
            delta_2 = (hi - x) / (hi - lo)         # room above x (normalized)
            u = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if u < 0.5:                            # downward perturbation, capped at -delta_1
                xy = 1.0 - delta_1
                val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)
                delta_q = val ** mut_pow - 1.0
            else:                                  # upward perturbation, capped at +delta_2
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)
                delta_q = 1.0 - val ** mut_pow
            x = x + delta_q * (hi - lo)            # scale by the variable range
            individual[i] = min(max(x, lo), hi)    # numerical safety clip
    return (individual,)


def run_evolution(
    evaluate_func: Callable, dim: int, lo: float, hi: float,
    pop_size: int, n_generations: int, cx_prob: float, mut_prob: float, seed: int,
) -> Tuple[list, list]:
    random.seed(seed)
    np.random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit

    fitness_history = []
    for gen in range(n_generations):
        offspring = custom_select(pop, len(pop), toolbox)        # tournament parents
        offspring = [toolbox.clone(ind) for ind in offspring]    # don't trample current pop

        for i in range(0, len(offspring) - 1, 2):                # SBX on pairs, prob cx_prob
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for i in range(len(offspring)):                          # outer mutation gate
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        for ind in offspring:                                    # enforce the box
            clip_individual(ind, lo, hi)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit                             # re-evaluate only the changed

        pop[:] = offspring                                       # generational replacement
        fitness_history.append(min(ind.fitness.values[0] for ind in pop))

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```

That's the whole loop: a real-coded GA whose crossover and mutation both reuse the same polynomial trick, so the population searches on the raw reals while still carrying single-point crossover's mean-preserving, near-the-parents-biased character — no bit string in sight, no precision cap, no Hamming cliff.
