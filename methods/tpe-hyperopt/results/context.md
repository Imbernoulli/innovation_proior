# Context: hyper-parameter optimization over tree-structured configuration spaces (circa 2011)

## Research question

Models like deep belief networks, stacked denoising autoencoders, and convolutional nets have
anywhere from ten to fifty hyper-parameters — learning rates, layer sizes, pre-processing choices,
penalties, contrastive-divergence schedules — and their performance depends sharply on getting
these right. The trouble is threefold. First, each trial is *expensive*: a single hyper-parameter
configuration has to be turned into a fully trained model and evaluated, which can take hours, so
the budget is tens to a few hundred trials, not thousands. Second, the configuration space is
*heterogeneous*: some variables are continuous (a learning rate), some ordinal (number of hidden
units), some categorical (which pre-processing). Third, and most distinctively, the space is
*tree-structured* (conditional): a variable like "number of units in the 2nd layer" is only
well-defined when a parent variable, "number of layers," takes a value that makes a 2nd layer
exist. An optimizer here must not only choose values for variables but simultaneously decide
*which* variables are even in play for a given configuration.

The precise goal: find a configuration with low validation loss in as few trials as possible,
over a space that is mixed-type and conditionally structured, using the information from every
trial already run. A method that ignores its own history (re-sampling blindly), or that cannot
represent the conditional structure, or that needs a global optimizer over an opaque criterion in
ten-plus dimensions, is the wrong tool. Because hyper-parameter search has been mostly a human
art, making it a formal, automatic outer loop of the learning process is itself part of the
problem — reproducibility and scientific progress depend on it.

## Background

The field state and the load-bearing pieces a solution rests on:

- **Hyper-parameter tuning had become the bottleneck, and it was done by hand.** Several advances
  in image-classification benchmarks came not from new models but from more careful configuration
  of existing ones. Humans tune efficiently when only a handful of trials are possible, but
  clusters and GPUs now make many trials feasible, shifting the optimal allocation of compute
  toward more automated exploration. Hyper-parameter optimization is properly seen as a formal
  outer loop: a learning algorithm is a functional from data to classifier that includes a
  budgeting choice of how many cycles go to exploring configurations versus evaluating each one.

- **The configuration space is defined by a generative process.** A clean way to specify a
  conditional, mixed-type search space is to write down a *generative process* for drawing valid
  configurations: first draw the number of layers; conditioned on that, draw each layer's size and
  training parameters; draw the pre-processing choice; and so on. Valid samples are exactly the
  draws of this process. This turns "which variables are active" into a property of the draw,
  not a special case to be hand-coded.

- **Sampling configs from that generative prior is already a usable algorithm** — it is random
  search, which had recently been shown to beat grid search for one-layer neural-network
  classifiers (Bergstra & Bengio 2012), because grid search wastes trials on coordinates the loss
  is insensitive to while random search spreads its trials across all coordinates. Random search
  is the baseline any history-using method must beat.

- **Sequential Model-Based Optimization (SMBO)** is the established frame for optimizing an
  expensive black-box fitness. Replace the costly `f(x)` with a cheap *surrogate* fit to the
  history `H = {(x_i, f(x_i))}`, and at each step pick the next `x` by optimizing a cheap
  *criterion* `S(x, M)` derived from the surrogate, evaluate the true `f` there (the one expensive
  step), append to `H`, and refit. The loop is: `H ← ∅`; for `t = 1..T`: `x* ← argmin_x S(x, M_{t−1})`;
  evaluate `f(x*)`; `H ← H ∪ {(x*, f(x*))}`; fit `M_t` to `H`. SMBO algorithms differ only in
  *what surrogate* models `f` and *what criterion* `S` is optimized.

- **Expected Improvement (EI) is the criterion of choice.** Given a threshold `y*` and a model `M`
  of `f`, the expected improvement is the expectation of how far below the threshold the value
  lands:

  `EI_{y*}(x) = ∫_{−∞}^{y*} (y* − y) p_M(y | x) dy`.

  It weights each possible gain by both its size and its probability, balancing exploitation
  (where `f` is predicted low) against exploration (where `f` is uncertain) without a hand-set
  knob. EI is generally non-negative, zero at already-evaluated points, and multimodal — so
  iterating it searches globally.

- **Parzen-window (kernel) density estimation** (Parzen 1962) estimates a density from samples by
  placing a kernel — typically a Gaussian — at each observation and averaging. With per-point
  bandwidths it adapts resolution to the local density of data: narrow kernels where observations
  are dense, wide where they are sparse. It estimates a *generative* density `p(x)` from points,
  the inverse direction from a regressor that predicts `y` from `x`.

## Baselines

The prior methods a new optimizer would be measured against or built upon:

- **Grid search and manual / grid-assisted search.** Enumerate a grid of configurations (or have a
  human steer a grid-assisted search), train and evaluate each. The gaps: grid search is
  exponential in the number of hyper-parameters and wastes trials along insensitive axes; manual
  search does not scale, is unreproducible, and cannot exploit a cluster's parallelism
  systematically.

- **Random search** (Bergstra & Bengio 2012). Draw configurations from the generative prior over
  the space and evaluate them. It beats grid search because it never wastes resolution on
  irrelevant coordinates, and it matches careful manual tuning of neural nets within a few dozen
  trials. Its gap: it is *memoryless* — it never uses the losses it has already observed to steer
  future draws — so on harder problems (deep belief networks on certain datasets) it converges
  slowly or plateaus below what careful search reaches. The information in the history is thrown
  away.

- **Gaussian-process / kriging SMBO with EI** (the DACE/EGO line: Jones, Schonlau & Welch 1998;
  Mockus 1978 for EI). Model `p(y | x)` directly as a Gaussian process — a prior over functions
  closed under sampling, so the posterior given `H` is again a GP with closed-form mean and
  variance. The posterior mean and variance give EI a closed form, and the next point is the EI
  maximizer. The GP supplies both a prediction and an honest, data-scarcity-aware uncertainty,
  which is exactly what EI needs. Its gaps in *this* setting: the GP fit costs `O(|H|^3)`; the EI
  surface is multimodal and must be globally optimized over a ten-plus-dimensional mixed space,
  which calls for evolutionary / EDA / CMA-ES machinery and restarts; and the tree-structured
  conditional space does not fit one GP — it has to be carved into groups with an independent GP
  per group, an awkward retrofit. Modeling `p(y | x)` over a conditional space is the friction.

- **Probability-of-improvement and entropy criteria.** Score a candidate by `P(Y(x) < y*)`
  (Kushner 1964) or by the expected reduction in entropy of the minimizer's location. Probability
  of improvement counts *whether* you improve, not by how much, so it over-exploits near the
  incumbent; entropy criteria are principled but heavier to evaluate. EI is preferred as intuitive
  and well-behaved.

## Evaluation settings

The natural yardstick is hyper-parameter optimization of deep belief networks on the image-
classification datasets where manual and random search struggle — in particular the "convex" and
"MNIST rotated background images" (MRBI) tasks — with the search space given by a generative prior
over DBN configurations: pre-processing (raw or ZCA) and ZCA energy, a classifier learning rate
and annealing schedule and L2 penalty (log-uniform / mixture priors), one to three layers, batch
size, and, per layer, the number of hidden units (log-uniform), the weight-init scheme,
and the contrastive-divergence epochs / learning rate / annealing (log-uniform). A small
regression task (Boston Housing) serves as a sanity check for the surrogate-driven loop. The
metric is the best validation loss reached as a function of the number of trials (each trial = one
trained-and-evaluated model), compared against random search and against the previously reported
manual results at matched trial counts.

## Code framework

The pieces that already exist: a way to describe the search space as a generative process whose
draw decides which variables are active; a random-search loop that samples from that prior and
evaluates; a kernel-density primitive (Gaussian Parzen windows); and the generic SMBO loop. The
surrogate and the acquisition are the empty slots to fill.

```python
import numpy as np

# A configuration space is a generative process: each draw produces a valid
# configuration and implicitly decides which (conditional) variables are active.
class ConfigSpace:
    def sample(self, rng):
        """Draw one valid configuration from the prior generative process."""
        raise NotImplementedError

class Surrogate:
    """A cheap stand-in for the expensive loss, fit to the trial history H."""

    def fit(self, X, y):
        # TODO: build the model of the objective from past (config, loss) pairs.
        #       This is the slot the method fills.
        pass

    def suggest(self, rng):
        # TODO: propose the next configuration to evaluate, by optimizing a
        #       cheap criterion derived from the fitted model.
        pass

def expected_improvement_criterion(x, surrogate, y_star):
    # TODO: the cheap figure of merit S(x, M) the SMBO loop maximizes --
    #       expectation of improvement below the threshold y_star.
    pass

def smbo(objective, space, n_init, max_trials, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_init):                    # random warm-up
        x = space.sample(rng)
        X.append(x); y.append(objective(x))    # the expensive step
    surrogate = Surrogate()
    for _ in range(max_trials - n_init):
        surrogate.fit(X, y)
        x_next = surrogate.suggest(rng)        # optimize the cheap criterion
        y_next = objective(x_next)             # one expensive trial
        X.append(x_next); y.append(y_next)
    i = int(np.argmin(y))
    return X[i], y[i]
```
