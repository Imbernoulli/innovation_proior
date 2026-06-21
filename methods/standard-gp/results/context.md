# Context: discovering symbolic models from data (circa late 1980s)

## Research question

Across the empirical sciences and engineering the recurring task is: given a finite sample of
observations — pairs `(x, y)` of inputs and a measured output — find the *mathematical
relationship* that produced them, written as an explicit symbolic formula. Not a black-box
predictor that returns a number, but an expression a human can read, manipulate, integrate,
and interpret: `y = log(x+1) + log(x^2+1)`, `y = 2 sin(x) cos(z)`, a polynomial, a rational
function, an econometric identity. Conventional regression sidesteps the form question: linear,
quadratic, or higher-order polynomial regression each fix a functional template in advance and
only solve for the numeric coefficients that minimize squared error. The question is how to
discover *both* the functional form *and* its constants from data, without pre-committing to a
template family.

## Background

The field state offers one powerful, well-developed engine for searching difficult,
non-differentiable, combinatorial spaces by simulated evolution.

**Holland's genetic algorithm (Holland, *Adaptation in Natural and Artificial Systems*, 1975).**
A genetic algorithm maintains a *population* of candidate solutions and improves them by
imitating Darwinian natural selection. Each candidate is encoded as a fixed-length string of
characters — the "chromosome" — over some small alphabet (classically bits). A problem-specific
*fitness* function scores how good each string is. The population is then bred forward by two
operators applied to parents chosen with probability rising with fitness: **fitness-proportionate
reproduction**, which copies a string into the next generation with probability proportional to
its share of total fitness (the fitter the string, the more expected copies); and **crossover
(recombination)**, which takes two parent strings, picks a random cut point, and exchanges the
tail segments to produce two offspring strings. A small **mutation** rate occasionally flips a
character to maintain raw diversity. Iterating selection-plus-recombination over many
generations drives the population's average fitness upward.

Holland supplied the theory for *why* this works. A *schema* is a template matching a set of
strings (e.g. `1**0` matches all 4-bit strings starting with 1 and ending in 0). Holland's
schema theorem shows that short, low-order schemata whose average fitness exceeds the population
average receive an approximately *exponentially increasing* number of trials over successive
generations — the genetic algorithm implicitly allocates its sampling effort the way an optimal
gambler would across the arms of a multi-armed bandit, concentrating trials on the
above-average building blocks. The *building-block hypothesis* is the working picture: good
solutions are assembled by crossover splicing together short, high-fitness sub-pieces
("building blocks") discovered in different individuals. The genetic algorithm had been shown
to search large, non-linear, multidimensional spaces effectively where gradient information is
unavailable.

The predominant genetic-algorithm work builds on fixed-length character strings. The need to
let the structures undergoing adaptation grow in complexity beyond fixed strings was an actively
explored direction at the time (Cramer 1985; Fujiki & Dickinson 1987; Goldberg 1989, among
others).

At the implementation level, one can already execute a proposed formula on a matrix of input
values, compare its outputs with the measured targets, and reject or penalize candidates whose
numeric evaluation fails. That supplies the scoring interface: hand the system a candidate,
get back finite predictions and an error.

## Baselines

**Linear / polynomial least-squares regression.** Fix a template — `y = a₀ + a₁x` (linear),
`y = a₀ + a₁x + a₂x²` (quadratic), or a chosen higher-order polynomial — and solve for the
coefficients `aᵢ` that minimize the sum of squared residuals `Σ(y − ŷ)²`, in closed form via the
normal equations. Fast, optimal *for the chosen template*, and statistically well understood.

**The genetic algorithm on fixed-length strings (Holland 1975; Goldberg 1989).** A general,
representation-agnostic optimizer for combinatorial fitness landscapes: population, proportionate
selection, one-point crossover, mutation, run for generations. Each candidate is serialized into
a fixed-length chromosome, with a fixed meaning assigned to each locus. The schema theory that
justifies the method is stated over fixed-length strings.

**Earlier evolution of variable-size structures (Cramer 1985; Fujiki & Dickinson 1987).** Initial
explorations applied evolutionary operators to more structured, program-like objects rather than
flat strings, demonstrating that increasing representational complexity was an active direction.

## Evaluation settings

The natural yardstick is a set of symbolic-regression benchmark target functions, sampled on a
grid, with the model judged on held-out points it was not fit to:

- **A univariate transcendental target**, e.g. `log(x+1) + log(x²+1)` sampled on `x ∈ [0, 2]` —
  tests whether the search can assemble logarithms, not just polynomials.
- **A bivariate trigonometric target**, e.g. `2·sin(x)·cos(z)` on a box in `(x, z)` — tests
  multivariate composition of trigonometric primitives.
- **A univariate polynomial target**, e.g. `x⁶ − 2x⁴ + x²` on `x ∈ [−1, 1]` — a form the search
  *can* in principle represent exactly, testing whether it actually finds it.

Inputs are drawn on a grid over the training range; a separate held-out set (a finer or wider
grid) measures generalization. The error on the training sample is the sum or mean of squared
differences between the candidate's output and the target over the sample points (the "fitness
cases"). The reported quality metric is the coefficient of determination **R²** on the held-out
set (higher is better, 1.0 perfect), with root-mean-square error reported alongside. Runs fix a
random seed, a population size, a maximum number of generations, and depth limits, and are
repeated to average over the inherent stochasticity.

## Code framework

What already exists is a generic population-based evolutionary-search harness and a way to
evaluate a proposed symbolic candidate on numeric data. The evaluator takes a candidate and an
input matrix, returns predictions, and guards the outer loop from invalid numeric output. The
harness can initialize a population, score everyone, ask for a new generation, repeat, and keep
the best-so-far. The candidate representation and the rules that choose, combine, perturb, and
preserve candidates are the slots to be designed.

```python
import random
import numpy as np

def safe_evaluate(candidate, X):
    """Run a candidate on X and return finite predictions for scoring."""
    ...

def generate_random(strategy, limit, n_features):
    """Build one random candidate from whatever representation the search will define."""
    ...

def random_population(pop_size, limit, n_features):
    """Build the initial population of random candidates."""
    return [generate_random('default', limit, n_features)
            for _ in range(pop_size)]


# ---- the slots to design: how to score, select, recombine, mutate ----

def fitness_function(candidate, X, y) -> float:
    # TODO: define how the data-fit error is measured.
    pass

def selection(population, fitnesses, n_select, **kwargs) -> list:
    # TODO: define how parents are chosen from scored candidates.
    pass

def crossover(parent1, parent2, n_features, max_depth=17):
    # TODO: define how two selected parents produce a child.
    pass

def mutation(parent, n_features, max_depth=17):
    # TODO: define how one selected parent is randomly perturbed.
    pass

def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17) -> list:
    # Build the next generation from the current one, using selection + the operators above.
    # TODO: assemble the per-generation breeding logic.
    pass


# ---- fixed outer loop the design plugs into ----
def run(X_train, y_train, n_features, pop_size, generations, max_depth):
    population = random_population(pop_size, max_depth, n_features)
    best, best_fit = None, float('inf')
    for gen in range(generations):
        fitnesses = [fitness_function(c, X_train, y_train) for c in population]
        i = int(np.argmin(fitnesses))
        if fitnesses[i] < best_fit:
            best_fit, best = fitnesses[i], population[i]      # keep best-so-far
        if gen < generations - 1:
            population = evolve_one_generation(
                population, fitnesses, X_train, y_train,
                n_features, pop_size, max_depth=max_depth)
    return best
```

The outer loop scores the population and asks `evolve_one_generation` for the next one; the four
empty functions — fitness, selection, recombination, mutation — are where the method will live.
