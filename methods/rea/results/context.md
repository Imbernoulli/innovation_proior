## Research question

State-of-the-art image classifiers have been hand-designed by experts over many
years of focused investigation — AlexNet (Krizhevsky et al. 2012), Inception
(Szegedy et al. 2015), ResNet (He et al. 2016), DenseNet (Huang et al. 2017). The
goal is to remove the human from this loop: to *automatically* discover a
convolutional architecture that classifies images as well as, or better than,
the best hand-crafted designs. Two broad camps of automated search exist.
The reinforcement-learning camp — today's most accurate ImageNet classifiers come
from RL search — trains a controller network (typically an LSTM) by policy-gradient.
The evolutionary camp uses random mutation plus selection, is trivially parallel,
and reached the published CIFAR-10 accuracy range with large-compute recent studies.
The question is how to design a population-based search method over neural
architectures that is accurate, simple to implement, and scales cleanly across
many accelerators.

## Background

By this time, automated architecture search ("neuro-discovery") is an active
field with a long history and a recent surge. The relevant landscape:

**Neuro-evolution.** Evolving neural-network topologies dates to Miller, Todd &
Hegde (1989) and was developed extensively by NEAT (Stanley & Miikkulainen 2002),
which simultaneously evolves weights and structure through three mutation kinds
(perturb a weight, add a connection, insert a node), recombines two genomes, and
maintains diversity by *fitness sharing*. The encoding question — *direct* (every
node and edge stored in the genome, as in NEAT) versus *indirect* (a compact
generative description, e.g. CPPNs) — has been a major theme. Recent large-compute
studies evolved CIFAR-10 classifiers into the published accuracy range.

**A standard fact about selection pressure.** In a genetic algorithm, a
*selection scheme* decides which individuals reproduce. Goldberg & Deb (1991)
analyzed the common schemes through the *birth-life-death* equation of population
dynamics, m_{i,t+1} = m_{i,t} + (births) - (deaths), and derived the *takeover
time* — the number of generations for the single best individual, starting from
one copy, to fill the whole population of size n under selection alone. For
*tournament selection* (draw s individuals at random, keep the best as a parent)
the takeover time for s > 1 is, asymptotically,

```
t* ≈ (1 / ln s) · [ ln n + ln(ln n) ],
```

so larger tournaments take over faster: selective pressure rises with the
tournament size s. Binary tournament (s = 2) is identical in expectation to linear
ranking. This gives a single, interpretable knob for "how greedy is the search."

**Generational vs. real-time evolution.** Classical (generational) GAs, NEAT
included, compute the whole next generation only after *every* current model has
finished training. In a distributed setting where each model trains on its own
machine, this is wasteful: a machine that finishes a fast model sits idle until
the slowest one is done. *Real-time* / *steady-state* variants (rtNEAT;
tournament/Genitor selection) instead replace one individual at a time, so workers
never wait — but they *retain* individuals by performance (or never discard them),
so a model can stay in the population for a very long time, even the whole run.

**An empirical fact about candidate evaluation: it is noisy.** Training a network
involves random initialization, data ordering, and stochastic regularization, so
the validation accuracy of a *fixed* architecture varies run to run — a spread on
the order of a percent has been observed for these neuro-evolution experiments.
The same architecture, re-trained, will not reproduce its score exactly.

**The aging idea exists, but heavyweight.** Hornby's ALPS (2006) introduces a
notion of *age*, assigned to *genes*, to split a population into "age-layers" so
that freshly introduced genes are protected from being immediately out-competed by
highly-selected older ones. It restricts competition to fight premature
convergence, and the age attaches to genes within a structured, multi-layer
population.

**The cell-based search space.** The dominant search space (Zoph et al. 2018)
fixes the network's *outer* skeleton and searches only a small repeated module.
The skeleton is a feed-forward stack of *cells*; there are two cell types, a
*normal cell* (preserves spatial resolution) and a *reduction cell* (strides by 2
to halve resolution), and the stack interleaves them. Every cell takes two input
tensors — the outputs of the previous two cells — treated as hidden states "0" and
"1". The cell is then built from exactly five *pairwise combinations* (blocks): a
block picks two existing hidden states, applies a chosen op to each, and *adds*
the two results to form a new hidden state, which becomes available to later
blocks. After five blocks, any hidden states never consumed are concatenated to
form the cell's output. The ops are a fixed menu of common convnet primitives
(separable convolutions of several sizes, pooling, dilated convolution, the
asymmetric 1×7-then-7×1 convolution, and identity/none). An architecture is thus
fully specified by the five blocks of the normal cell plus the five of the
reduction cell; two integers, the number of cells per stack N and the number of
filters F, are set by hand to scale the model. This space was originally explored
with an RL controller, but it is search-method-agnostic — random search over it is
a "difficult baseline to beat," which makes it an ideal arena for a controlled
comparison between search algorithms.

## Baselines

**Reinforcement-learning architecture search (Zoph & Le 2016; Zoph et al. 2018).**
A controller — typically an LSTM — emits an architecture token by token (for the
cell space: for each of the five blocks, pick two input hidden states and two ops
and how to combine them). The architecture is trained and evaluated, and its
validation accuracy is the reward used to update the controller by policy gradient
(REINFORCE, later PPO). Core idea: learn a parameterized *distribution* over
architectures that drifts toward high-reward regions. This camp holds the current
accuracy records.

**Large-scale neuro-evolution with tournament selection (Real et al. 2017).** Keep
a population of P trained models; repeatedly run a tournament — sample s models,
take the highest-accuracy one as parent, mutate it (mutation only; no
recombination, no fitness sharing), train and evaluate the child, and add it back.
To hold the population at size P, *each tournament also removes the worst of its
sampled models.* Weights can be inherited across mutations; the search starts from
trivial models and grows complexity. This reached the published CIFAR-10 accuracy
range with a simple, asynchronous method.

**Random search.** Construct each model independently and uniformly at random over
the search space (no mutation of existing models, no selection), train, evaluate,
keep the best seen. On a well-constructed cell space it is surprisingly strong and
a difficult baseline to beat, which is exactly why it must be included.

## Evaluation settings

- **Search dataset:** CIFAR-10 (Krizhevsky 2009), 32×32 color images, 10 classes;
  5k examples withheld for validation. Search is run over *small* models (small N,
  F — e.g. N = 3, F = 24, trained 25 epochs) so each candidate trains quickly.
- **Final-model datasets:** the searched cell is enlarged (increase N and F) and
  retrained for many epochs, on CIFAR-10 and on ImageNet (Deng et al. 2009; 1.2M
  images, 1000 classes, with a withheld validation split and the standard
  validation set used for test). For ImageNet a stem is prepended to downsize the
  input. This *model-augmentation* protocol (search small, then scale the winner)
  is the established recipe.
- **Metrics:** validation accuracy during the search (the fitness driving
  selection); test accuracy / error and model compute cost (FLOPs / multiply-adds)
  and parameter count for the final augmented models. For the search-algorithm
  comparison, the natural axis is *accuracy vs. number of models evaluated* (or vs.
  wall-clock), with several independent repeats to expose variance, and attention
  to the *early* part of the curve (the resource-constrained regime).
- **Controlled-comparison protocol:** to attribute differences to the *algorithm*
  rather than the space or the training code, run every algorithm in *identical*
  conditions — the same code for network construction, training, and evaluation,
  the same search space, the same budget — and tune each algorithm's
  meta-parameters independently before the head-to-head, reporting repeats without
  selection bias. Smaller-compute variants (gray-scale CIFAR / MNIST / gray-scale
  ImageNet, on CPU, with several search-space sizes) allow many repeats for
  statistical confidence.

## Code framework

The search plugs into a standard population-based-search harness over the cell
space. What already exists: an architecture is a small genome — the five blocks of
the normal cell and the five of the reduction cell, each block a pair of (input
hidden state, op); a `random_architecture()` that samples a valid genome
uniformly; a black-box `train_and_eval(arch)` that builds the network, trains it,
and returns a (noisy) validation accuracy — the expensive, rate-limiting step; and
a population data structure of evaluated models. The per-cycle search rule — how to
turn an existing population into the next candidate, and how to decide what the
population carries forward — is what must be designed.

```python
import random


class Model:
    """An evaluated candidate: its genome and its (noisy) fitness."""
    def __init__(self):
        self.arch = None        # the genome (normal-cell + reduction-cell blocks)
        self.accuracy = None    # noisy validation accuracy from train_and_eval


def random_architecture():
    """Sample a valid architecture uniformly over the search space."""
    # ... build five normal-cell blocks and five reduction-cell blocks,
    #     each block = (hidden-state input, op) pairs.
    raise NotImplementedError


def mutate(parent_arch):
    """Produce a child genome from a parent by a small random change."""
    # A valid local genome edit supplied by the search-space implementation.
    raise NotImplementedError


def train_and_eval(arch):
    """Build, train, and evaluate the architecture; return validation accuracy.
    Expensive (trains a network from scratch) and NOISY (re-evaluating the same
    arch gives a slightly different number). This is the rate-limiting step."""
    raise NotImplementedError


def search(cycles, population_size, sample_size):
    """Population-based search over the cell space.

    Initialize a population of `population_size` random, evaluated models, then
    run for `cycles` steps. Each step turns the current population into one new
    candidate and decides what the fixed-size population carries forward.
    """
    population = []
    history = []

    # Initialize: fill the population with random, evaluated models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.accuracy = train_and_eval(model.arch)
        population.append(model)
        history.append(model)

    while len(history) < cycles:
        step(population, history, sample_size)

    return history


def step(population, history, sample_size):
    # TODO: the per-cycle population rule we will design.
    pass
```

The expensive `train_and_eval` dominates the cost, so the search rule is judged by
how few calls it needs to reach a good model — and by how simply and
asynchronously it can be run across many machines.
