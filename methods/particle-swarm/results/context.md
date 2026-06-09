# Context: Optimizing a multimodal black box with no gradients

## Research question

Suppose the only thing on hand is an evaluator: feed it a point x in a continuous, possibly high-dimensional space, get back a scalar quality value f(x). No derivative is available — f might be a noisy simulation, a neural network's training error as a function of its weights, or a hand-coded objective with no closed form. f may be wildly multimodal: many basins, many local optima separated by ridges, a global optimum hiding among them. The question is how to drive such an evaluator toward a good (ideally global) solution using nothing but repeated evaluations.

Anything that follows a gradient is out — there is no gradient, and even a numerical one would point straight into the nearest local minimum on a multimodal surface. What is wanted is a search that (a) treats f as an opaque box, (b) explores broadly enough to find the right basin out of many, and (c) refines a good region once found, all with only "primitive mathematical operators," modest memory, and a handful of parameters a user can set without intimate knowledge of f. The tension at the heart of it is explore-versus-exploit: spread out to avoid committing early to a local optimum, yet eventually concentrate to sharpen the answer.

## Background

**Derivative-free optimization of multimodal black boxes.** When f is non-differentiable, non-convex, and multimodal, a single trajectory (gradient descent, hill-climbing, Nelder-Mead from one start) gets trapped in whatever basin it starts in. The classical defenses either restart from many random seeds (wasteful, no information shared between starts) or carry a *set* of candidate solutions that sample the landscape in parallel and exchange information. Carrying a population is the structural answer to multimodality: different members can sit in different basins, and the basin containing the best-so-far can recruit the others.

**Social sharing of information has survival value.** Sociobiology (Wilson, 1975) observed that in a fish school or bird flock, "individual members of the school can profit from the discoveries and previous experience of all other members of the school during the search for food... whenever the resource is unpredictably distributed in patches." When good regions are sparse and scattered — exactly the multimodal case — an individual that can exploit the *group's* discoveries does far better than one searching alone. This is a hint that a search procedure should let candidates share where the good regions are.

**Flocking as a distributed, local model (Reynolds, 1987).** Reynolds' "boids" reproduced the fluid, synchronized motion of a flock from three purely local steering rules applied by each bird to its neighbors: separation (avoid crowding), alignment (match neighbors' heading), and cohesion (steer toward the local center of mass). No bird has global knowledge of the flock; coordinated global motion *emerges* from local interactions. The lesson for optimization: complex collective behavior can be produced by very simple per-agent update rules with no central controller.

**The cornfield / roost variant (Heppner & Grenander, 1990).** Heppner added a dynamic attractor to a flocking simulation — a "roost," a fixed point on the screen that birds are pulled toward until they land. A two-dimensional "cornfield vector" let each agent evaluate its current position, e.g. Eval = sqrt((x−100)²) + sqrt((y−100)²), zero at the target (100,100). With the attraction tuned high the flock was "sucked violently into the cornfield"; tuned low it swirled in and settled gently. So a flock-like simulation, given a position-quality function, can be made to *find* an unknown target — the bridge from animation to optimization. The open question Heppner's model leaves: real birds do not know where the food is in advance; how does a flock find an a-priori-unknown good region using only what its members discover?

**Nearest-neighbor velocity matching collapses, and needs perturbation.** A bare flocking simulation built only on each agent copying its nearest neighbor's velocity quickly settles onto a single unanimous, unchanging direction — the population collapses to one behavior and stops searching. Injecting a stochastic perturbation ("craziness") restores variety and a lifelike, non-collapsing motion. So pure imitation is not enough; randomness is required to keep a population from prematurely agreeing.

**Five principles of swarm intelligence (Millonas, 1994).** A swarm worth the name should obey: *proximity* (carry out simple space/time computations), *quality* (respond to quality factors in the environment), *diverse response* (not commit all activity to narrow channels), *stability* (not change behavior mode on every environmental fluctuation), and *adaptability* (change mode when it is worth the cost). Stability and adaptability are two sides of one coin: a good search must hold steady yet re-orient when something genuinely better appears.

**Explore-exploit and the cost of imbalance.** Any population search must balance wandering (explore) against converging (exploit). Lean too far toward dispersing the population and it never concentrates enough to sharpen an answer; lean too far toward pulling everyone to the single best-so-far and the population rushes prematurely into the first decent basin it stumbles on, before the landscape has been surveyed. Neither extreme searches a multimodal domain well, and there is no a-priori-known right balance for an arbitrary black box.

## Baselines

**Genetic algorithms (Holland; Davis 1991).** The dominant population-based black-box optimizer of the time. Encode candidate solutions as strings (a "chromosome"), assign each a scalar *fitness* = f(x), and evolve the population by selection (fitter strings reproduce more), crossover (recombine two parents' substrings), and mutation (random bit flips). GAs handle multimodal, non-differentiable f and maintain diversity through the population, and Holland's analysis of the "optimum allocation of trials" frames the explore/exploit trade-off precisely. Gaps: GAs operate by *recombining* discrete encodings rather than moving points through the continuous space; a GA individual is a string that gets spliced and discarded from generation to generation, with no persisting identity or trajectory of its own; and they require choosing an encoding, crossover and mutation operators, and selection pressure.

**Evolutionary programming / evolution strategies.** Mutation-driven population search over real-valued vectors with selection, no crossover. Heavily stochastic, like GAs, and likewise without any notion of a candidate that persists and moves through the space across generations.

**Hill-climbing with random restarts.** Local search from many seeds. Simple and gradient-free, but each restart is independent — no sharing of discoveries between starts — so on a sparse multimodal landscape it wastes most of its evaluations re-exploring.

**Multi-start Nelder-Mead / pattern search.** Direct-search methods that need no gradient and adapt a simplex or pattern to the local landscape. Effective for local refinement, but single-basin per run and again non-cooperative across restarts.

## Evaluation settings

The natural yardsticks are standard continuous black-box test functions, evaluated by the number of evaluations / iterations to reach a target value and by the best value attained. Highly multimodal cases stress the explore side: Schaffer's f6 (a very ridged, deceptive 2-D surface with many local optima), Rastrigin (a paraboloid studded with a regular lattice of local minima), Griewank, and Ackley; Rosenbrock's curved valley stresses the exploit/refinement side; the Sphere function is a unimodal sanity check. Typical protocol: a few tens of candidates, a fixed dimensionality (e.g. 10), a cap on iterations (e.g. 1000) or a "no-improvement for k iterations" stopping rule, averaged over many independent runs. Applied settings include training feedforward neural-network weights as a black box — the exclusive-or net (a 2-3-1 architecture, 13 weights) and a classifier on the Fisher Iris data — scored by sum-squared error or classification accuracy, to be compared against backpropagation. No outcomes are recorded here, only the settings.

## Code framework

The pieces that already exist: numpy for vectorized array math, a uniform random generator, and an objective function `f(X)` that maps a batch of candidate positions of shape `(n_particles, dimensions)` to a vector of costs of shape `(n_particles,)`. A container holds the per-candidate state, and a driver loop repeatedly evaluates the population and updates it. What bookkeeping each candidate keeps, and the rule by which the population is updated each step from the new evaluations, are the empty slots.

```python
import numpy as np

class Population:
    """Holds the per-candidate state for a population search."""
    def __init__(self, n_particles, dimensions, bounds, options):
        lo, hi = bounds
        self.position = np.random.uniform(lo, hi, (n_particles, dimensions))
        # TODO: whatever additional per-candidate state and population-level
        # bookkeeping the update rule turns out to need
        self.options  = options    # the few tunable parameters of the update rule

def update(pop, current_cost):
    """The update rule: given the new evaluation of every candidate's current
    position, move the population for the next step."""
    pass  # TODO

def optimize(f, n_particles, dimensions, bounds, options, iters):
    pop = Population(n_particles, dimensions, bounds, options)
    for _ in range(iters):
        cost = f(pop.position)
        update(pop, cost)
    return pop  # best-so-far, however the update rule chooses to track it
```
