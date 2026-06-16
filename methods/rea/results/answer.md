# Regularized Evolution (Aging Evolution), distilled

Regularized evolution — also called *aging evolution* (AE) — is a steady-state
evolutionary architecture search. It keeps a fixed-size population of trained
models and, each cycle, runs a tournament to pick a parent, mutates it into a
child, trains and evaluates the child, and then **removes the oldest model in the
population** rather than the worst. That single change — eviction by age instead of
by accuracy — is the contribution; it turns standard tournament selection into a
regularized search that resists overfitting to noisy validation scores. Run over
the cell-based NASNet search space, it searches for a reusable normal-cell and
reduction-cell pair without training a controller network.

## Problem it solves

Automatically discover a high-accuracy convolutional image classifier with a
method that is (1) competitive with both hand-designs and RL-based search, (2)
simple and low-maintenance (few meta-parameters, easy to reproduce), (3) cleanly
parallelizable at scale with high utilization, and (4) strong *early*, in the
resource-constrained regime where a run is stopped before convergence. Candidate
evaluation is expensive (each architecture is trained from scratch) and **noisy**
(re-training the same architecture gives a different accuracy), and that noise is
what defeats naive evolution.

## Key idea

Keep a population of P models as a FIFO queue. Each cycle:

1. **Tournament (exploitation):** sample S models uniformly at random (with
   replacement); the highest-accuracy one is the *parent*.
2. **Mutation (exploration):** apply one random mutation to the parent to make a
   *child*; train and evaluate it (a fresh, independent noise draw).
3. **Aging (the regularizer):** append the child (newest) and remove the **oldest**
   model — *not* the worst.

Why aging regularizes: under the standard "kill the worst" rule, a model that
scored high — possibly by luck on its one noisy training run — never loses a
tournament, so it persists and reproduces, and the search collapses onto a
fluctuation (premature convergence, overfitting to evaluation noise). Under "kill
the oldest," every model has the same fixed maximum lifespan (≈ P cycles)
regardless of accuracy, so a one-time lucky model ages out before it can dominate.
The *only* way an architecture survives for long is to be re-created through
parent→child inheritance, and each inheritance is a fresh training run — so an
architecture persists only if it **re-trains well repeatedly**, i.e. its quality is
a property of the architecture and not of one lucky evaluation. Constraining the
survivors to "architectures that re-train well" injects prior information that
suppresses fitting to noise — regularization in the broad mathematical sense.

It needs **no extra meta-parameter**: age is just position in the queue (append
right = newest, pop left = oldest). The only knobs are P (population size) and S
(tournament size). For S > 1, the takeover-time analysis of tournament selection
gives `t* ≈ (1/ln S)·[ln P + ln(ln P)]`, so S controls selective pressure: S=1
is random search, larger S is greedier. The loop is steady-state (one model in,
one out), so it parallelizes asynchronously across workers with no generational
barrier — fast machines never wait on slow ones.

## Search space and mutations (cell-based)

An architecture is a *normal cell* and a *reduction cell*, each built from five
*pairwise combinations*: a combination takes two existing hidden states, applies a
chosen op to each, and adds the results into a new hidden state; unused hidden
states are concatenated to form the cell output. N (cells per stack) and F
(filters) scale the model and are set by hand. Two mutations reach the whole space:

- **Hidden-state mutation:** pick a cell, a combination, one of its two inputs, and
  rewire it to another hidden state in the cell (no loops — keep it feed-forward).
- **Op mutation:** same selection, but replace the op with a random one from the
  fixed menu (none/identity; 3×3/5×5/7×7 separable conv; 3×3 avg pool; 3×3 max
  pool; 3×3 dilated separable conv; 1×7-then-7×1 conv).
- **Identity mutation:** no change (small fixed probability, ~0.05, untuned).

One mutation per cycle, chosen at random.

## Algorithm

```
population <- empty FIFO queue ; history <- {}
while |population| < P:                       # initialize
    arch <- RandomArchitecture()
    acc  <- TrainAndEval(arch)                # expensive, noisy
    append (arch, acc) to right of population ; add to history
while |history| < C:                          # C = total evaluated-model budget
    sample <- S models drawn uniformly at random (with replacement) from population
    parent <- highest-accuracy model in sample
    child.arch <- Mutate(parent.arch)
    child.acc  <- TrainAndEval(child.arch)
    append child to right of population ; add to history
    remove the OLDEST (leftmost) model from population        # aging = regularization
return highest-accuracy model in history
```

Meta-parameters used at scale: P = 100, S = 25 (lightly tuned over 5 settings).

## Working code

Grounded in the canonical implementation's structure and the final algorithmic
return value; a deque makes age = position, so no age field is stored. The toy
`train_and_eval` here (a bitstring whose fitness is its 1-bit fraction plus small
Gaussian noise, with a single optimum) stands in for the expensive, noisy network
training, and lets the whole loop run in milliseconds.

```python
import collections
import random

DIM = 100            # genome length (stands in for the architecture)
NOISE_STDEV = 0.01   # matches observed neural-net training noise


class Model:
    """An evaluated individual. 'Age' is NOT stored — it is the model's position
    in the FIFO population queue (leftmost = oldest)."""
    def __init__(self):
        self.arch = None       # the genome (here an int bit-string of length DIM)
        self.accuracy = None   # noisy validation accuracy (the 'fitness')

    def __str__(self):
        return '{0:b}'.format(self.arch)


def _sum_bits(arch):
    total = 0
    for _ in range(DIM):
        total += arch & 1
        arch >>= 1
    return total


def train_and_eval(arch):
    """Stand-in for building, training, and evaluating an architecture.
    NOISY: re-evaluating the same arch gives a different number."""
    accuracy = float(_sum_bits(arch)) / float(DIM)
    accuracy += random.gauss(mu=0.0, sigma=NOISE_STDEV)
    return min(1.0, max(0.0, accuracy))           # clip to [0, 1]


def random_architecture():
    """Sample a valid architecture uniformly over the search space."""
    return random.randint(0, 2 ** DIM - 1)


def mutate_arch(parent_arch):
    """One small random change. In the cell space this is a hidden-state rewire
    or an op relabel; in this toy space it flips one random bit."""
    position = random.randint(0, DIM - 1)
    return parent_arch ^ (1 << position)


def regularized_evolution(cycles, population_size, sample_size):
    """Aging / regularized evolution.

    Args:
        cycles:          total number of models to evaluate (the budget, C).
        population_size: P, models kept alive at once.
        sample_size:     S, tournament size = the selective-pressure knob.
    Returns:
        the highest-accuracy Model ever evaluated.
    """
    population = collections.deque()
    history = []

    # Initialize the population with random, evaluated models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.accuracy = train_and_eval(model.arch)
        population.append(model)
        history.append(model)

    # Steady-state evolution: each cycle produces one model and removes one.
    while len(history) < cycles:
        # Tournament selection: S random members; population is left untouched.
        sample = [random.choice(list(population)) for _ in range(sample_size)]
        parent = max(sample, key=lambda m: m.accuracy)      # exploitation

        child = Model()
        child.arch = mutate_arch(parent.arch)               # exploration
        child.accuracy = train_and_eval(child.arch)         # fresh noise draw
        population.append(child)                            # newest -> right
        history.append(child)

        population.popleft()                               # AGING: evict OLDEST

    return max(history, key=lambda m: m.accuracy)


if __name__ == '__main__':
    best = regularized_evolution(cycles=1000, population_size=100, sample_size=10)
    print('best accuracy:', best.accuracy)
```

In the real setting, `Model.arch` holds the normal- and reduction-cell genomes,
`train_and_eval` trains the network and returns validation accuracy, `mutate_arch`
applies the hidden-state / op / identity mutation, and the loop is distributed
across workers (the cycle loop is parallelized; sampling reads S random members),
which is exactly what keeps utilization high and the method simple to scale.

## Relation to prior methods

- **Tournament selection (Goldberg & Deb 1991):** the base loop; takeover time
  `t* ≈ (1/ln S)·[ln P + ln(ln P)]` gives S as the selective-pressure dial.
- **Large-scale neuro-evolution (Real et al. 2017):** same tournament loop but it
  removes the **worst** sampled model ("non-aging evolution"); high-accuracy models
  can then persist for the whole run and the search overfits to lucky evaluations.
  Aging removes the oldest instead, fixing that.
- **NASNet search space (Zoph et al. 2018):** the cell-based space searched here,
  originally explored with an RL controller; this method uses the same cell space
  and train/evaluate harness while replacing the controller with tournament
  selection, mutation, and age-based turnover.
- **ALPS (Hornby 2006):** also uses "age," but on *genes*, with age-layers and two
  extra meta-parameters; aging evolution attaches age to *individuals* as mere
  queue position and adds **no** meta-parameter.
