## Research question

A scaling law predicts how a model's performance behaves as you scale it up — predicting the
training loss of a large run from a handful of cheap small-scale runs. It is the instrument that
decides how to spend a pre-training budget: how big a model, how many tokens, what learning rate
and batch size, how large a vocabulary, how many experts, how much data repetition. The form these
laws take is *symbolic*: a compact mathematical expression `f_theta(x) -> y_hat` mapping a few
descriptor variables `x` (parameter count `N`, tokens `D`, vocabulary `V`, learning rate `l`, batch
size `b`, number of experts `E`, unique tokens `U`, ...) to a target `y` (loss, perplexity, a Brier
score), with a small set of coefficients `theta` fit to observed runs. The reason a *formula* is
wanted rather than an arbitrary regressor is generalizability: a good law captures a mechanism, so
it extrapolates to scales never observed and transfers across related settings.

Today such laws are produced by hand. A human expert stares at a new regime, proposes a functional
form by intuition and analogy to prior laws, fits its coefficients, checks the fit, and iterates.
This is slow, labor-intensive, and bounded by what a person can reason about — especially once a law
must couple three, four, or five interacting variables. New regimes keep arriving faster than experts
can characterize them. The precise problem: given a fixed dataset of observed trials, automatically
produce (1) a single symbolic functional form shared across a family of experimental settings, and
(2) for each setting its own fitted coefficients, such that the law predicts *held-out, extrapolated*
points — larger models, more tokens, more extreme hyperparameters — accurately. Two structural
difficulties make this hard. First, **symbolism with open-endedness**: the space of candidate
expressions is infinite and has no a-priori-known form; the ground-truth law is unknown even to the
experts who study the regime, and any genuine improvement is itself a finding. Second, **abstraction
across contexts**: the goal is not to fit each experimental group independently but to discover *one*
form whose coefficients adapt per group — a single law that generalizes across dozens of disparate
contexts. A solution would have to search a vast, unstructured formula space under a clear but
unbounded objective, and do so robustly enough that the result extrapolates rather than memorizes.

## Background

**The objects being predicted.** Modern scaling laws span many axes. Model-size and data scaling
follow smooth power-law trends in `N` and `D` (Kaplan et al. 2020; Hestness et al. 2017). The
compute-optimal "Chinchilla" decomposition (Hoffmann et al. 2022) writes pre-training loss as an
additive sum of an irreducible floor and two power-law terms,

```
L(N, D) = E + A / N^a + B / D^b,
```

`E` the entropy floor, `A/N^a` the capacity-limited term, `B/D^b` the data-limited term. Beyond this
base case, regimes have multiplied: vocabulary scaling, mixture-of-experts, data-constrained training
with repetition, domain mixtures, supervised fine-tuning, inference-time parallelism,
learning-rate/batch-size selection, and non-monotonic ("U-shaped") capability curves. Each was
characterized separately, by hand.

**Extrapolation is the real test.** A law is only useful if, fit on small-scale runs, it predicts
large-scale behavior. So the natural evaluation holds out the *largest* models / datasets / most
extreme hyperparameters and asks how well a law fit on the rest predicts them. This is where
hand-crafted laws can quietly fail: a form that interpolates the observed region beautifully can bend
the wrong way just outside it. Learning-rate/batch-size prediction is especially exposed because the
available prior law predicts only the optimizer settings, not the full loss surface over the sweep.
Even for regimes with a canonical law, the human form may fit the known region while remaining brittle
under genuine extrapolation.

**Program search as scientific discovery.** A parallel line of work treats open-ended scientific and
algorithmic problems as a *search over programs*, scored by an automated, objective evaluator: faster
matrix-multiplication algorithms (Fawzi et al. 2022), improved combinatorial bounds (Campos et al.
2023, the diagonal-Ramsey improvement), faster sorting routines (Mankowitz et al. 2023). The shared
structure: a clear, continuous, machine-computable objective whose optimum is unknown, where progress
comes not from a single brilliant proposal but from *generations* of incremental improvement on prior
proposals. Pairing this with large language models — which can write and revise code — gave rise to
LLM-driven evolutionary program search (below). Scaling-law discovery has exactly this shape: the
objective (extrapolation `R^2`, target 1) is clear, continuous, and computed directly from data with
no learned reward model, while the best law is unknown.

**Quality-diversity search.** Mouret & Clune (2015), "Illuminating Search Spaces by Mapping Elites"
(MAP-Elites), is the population-management idea these systems rest on. Instead of keeping only the
single best solution, discretize a user-chosen multi-dimensional *feature* space (behavioral
descriptors of a solution) into a grid of cells and keep one elite — the best-fitness solution — per
cell. Local competition happens within each cell, so the population stays spread across the feature
space and the search "illuminates" which regions hold high performers, rather than collapsing onto one
local optimum. Combined with an *island* model — several sub-populations evolving in parallel with
periodic migration — this maintains diversity over long runs.

## Baselines

**Hand-crafted scaling laws (the prior art a new law is measured against under extrapolation).** Each regime has
a canonical human form, fit by standard optimizers (BFGS, SGD) per group:

- *Chinchilla* (Hoffmann et al. 2022): `L = E + A/N^a + B/D^b`, the additive power-law decomposition
  above. Limitation: it is a *single regime's* form; every new axis (vocabulary, experts, repetition,
  hyperparameters) has required a fresh hand-designed expression, and the purely additive structure
  cannot represent couplings between axes when they exist.
- *Vocabulary* (Tao et al. 2024): `L(N,V,D) = A/N^a + B/V^b + C/D^g + E`, defined on a
  *unigram-normalized* loss `Lossu` (the log-probability improvement of the model over a context-free
  unigram model, which can be negative) so that runs with different `V` are comparable. Larger `V`
  reduces tokenization cost but raises embedding difficulty; the `B/V^b` term captures the net effect.
  Limitation: additive across all three axes; the form is fixed by hand.
- *Data-constrained* (Muennighoff et al. 2023): when tokens are repeated (`D` exceeds the unique-token
  budget `U`), raw `D` and `N` are replaced by *effective* quantities via an exponential half-life,
  `D_eff = U + U R_D (1 - exp(-(D/U - 1)/R_D))` (and analogously `N_eff` with a base capacity `U_N`
  and its own decay `R_N`), then plugged into the Chinchilla form; at `D = U` it reduces to Chinchilla.
  Repeated tokens count progressively less. Limitation: the repetition-decay mechanism, the effective
  transforms, and the five-plus coefficients are all hand-designed; the form is frozen.
- *Learning-rate & batch-size*: no established full-loss law exists. The closest prior art (Li et al.
  2025, "Step Law") predicts only the *optima* `lr* = c N^a D^b`, `bsz* = d D^g`, fit from roughly 17
  best-performing configurations out of about 3,000 experiments — discarding the vast majority of the
  data. Limitation: it models the location of the optimum, not the loss surface, and throws away most
  observations.
- *Mixture-of-experts* (Krajewski et al. 2024): `log L = a log N + b log E_hat + c (log N)(log E_hat)
  + d`, an exponential of log-linear terms with an `N`-by-experts interaction (`E_hat` a stabilized
  transform of the expert count). Limitation: being an exponential of a log-bilinear form, its
  asymptotics are fragile — depending on the fitted sign of the interaction coefficient, the predicted
  loss can grow without bound as scale increases.
- *Supervised fine-tuning* (Lin et al. 2024): `L(D) = C + A/(D^a + B)`, a "rectified" power law with a
  pre-power phase at small `D` saturating to `C`. Limitation: the offset `B` enters additively next to
  `D^a`, so its units depend on the exponent and its interpretation is muddied.

Across all of these the common limitations are: the functional form is fixed by a human up front; the
parameter-fitting routine is an off-the-shelf optimizer applied uniformly; and on the harder regimes
the resulting laws extrapolate poorly, sometimes worse than a constant.

**Symbolic regression and genetic programming.** The classical automated alternative to hand-design.
Koza (1992) evolves expression trees under a small operator set `{+, -, *, /, pow, log, exp}`; Schmidt
& Lipson (2009) distill free-form natural laws from data; Brunton et al. (2016, SINDy) recover
governing equations via sparse regression over a candidate-function library; Udrescu & Tegmark (2020,
AI Feynman) use physics-inspired heuristics to prune the search. Limitations for the present problem:
the number of candidate expressions grows combinatorially, so exhaustive search is infeasible; these
methods carry weak domain priors; the standard benchmarks ask them to *rediscover* a pre-known
synthetic formula (essentially closed-form curve fitting), not to confront a genuinely open problem;
and they have no mechanism for the multi-context abstraction this task needs — discovering one form
whose coefficients adapt across many experimental groups.

**LLM-driven evolutionary program search.** Romera-Paredes et al. (2024, FunSearch) pair a pretrained
LLM with an automated evaluator and an island-based evolutionary loop. A database holds scored,
valid programs; each step samples several high-scoring programs into a "best-shot" prompt, asks the
LLM to write an improved candidate, executes and scores it, and inserts it back. The evaluator acts as
a guard against LLM confabulation: only programs that actually run and score well propagate. FunSearch
discovered new constructions for the cap-set problem and bin-packing heuristics that beat first-fit and
best-fit. Its structural limitation: it evolves a *single function body* embedded in a *fixed program
skeleton* — the surrounding harness, and crucially the routine that would *fit* any free parameters,
are hand-written and never searched. Novikov et al. (2025, AlphaEvolve) generalize this to evolving
whole files and multiple code blocks via diff edits, with an LLM ensemble, a MAP-Elites-plus-islands
database, and richer prompts; it is deliberately problem-agnostic, applying the same generic evolution
to scheduling, hardware design, and kernel optimization. Its generality is also its limitation for a
specialized regression-discovery task: a generic loop leaves scaling-law structure to be rediscovered
from the prompt and score alone.

## Evaluation settings

The natural yardstick is a benchmark of real LLM training experiments curated from the published
scaling-law literature (over 5,000 runs), turned into discovery tasks. Representative harder regimes:

- **Vocabulary** — inputs `(non_vocab_parameters N, vocab_size V, num_characters D)`; target the
  unigram-normalized loss `Lossu` (which can be negative — must not be clipped). Seen range
  `V in [4,100, 64,500]`; test fixes `V = 96,300`, outside the seen range. ~1,080 seen / 120 test.
- **Learning-rate & batch-size** — inputs `(lr l, bsz b, data_size D, non_embedding_param_size N)`;
  target `lm_loss`. Test restricted to the largest model (1B) and largest dataset (100B tokens).
  ~2,702 seen / 117 test. `lr in [2.4e-4, 2.2e-2]`, `bsz in [16, 2048]`, loss in `[2.1, 3.7]`.
- **Data-constrained** — inputs `(unique_tokens U, params N, tokens D)` with `D` possibly far
  exceeding `U` (repetition); target `loss`. Test holds out the largest one or two values of `N` or
  `D`. ~161 seen / 21 test.

Each task may carry several control groups (settings sharing the form but with their own coefficients).
The held-out split is always the *extrapolation* region — the largest scales — so the metric directly
measures extrapolation, not interpolation. Evaluation runs in a sandboxed terminal with a minimal
Python stack (`scikit-learn`, `pandas`, `datasets`, `numpy`, `scipy`), no network access, and the
agent must emit a script defining the law with a fixed signature. The primary metric is held-out
`R^2` per benchmark (`R^2 = 1 - sum(y - y_hat)^2 / sum(y - y_bar)^2`, closer to 1 is better, unbounded
below); secondaries are MAE, RMSE, NMAE. The test split is untouched during search and fitting.

## Code framework

The substrate is a generic LLM-driven evolutionary program-search harness — the kind FunSearch and
AlphaEvolve already provide — operating over executable candidate *programs* scored by an automated
evaluator. What already exists and is reused: a per-group data loader; an evaluator that runs a
candidate program against the seen split, fits free parameters, and returns a scalar fitness (here
derived from `R^2` / NMSE on the data, with failures mapped to a floor); a
quality-diversity program database (MAP-Elites cells over user-chosen feature dimensions, plus several
islands with periodic migration) with parent/inspiration sampling; and an LLM that, given a prompt,
returns a new candidate program. What is *not* settled — the open slots — are (a) what a candidate
*program* should contain and which part of it the LLM is asked to improve, and (b) how to construct the
mutation.

```python
import numpy as np


# --- existing: per-group data and the automated evaluator (reused as-is) ---

def load_seen_data(task):
    """Return {group_key: (X_group, y_group)} for the seen (training) split."""
    ...

def fitness(program_path, task):
    """Run a candidate program on the seen split: for each group, fit the program's
    free parameters, predict, and score the fit (R^2 / NMSE-derived). Return a
    scalar; a program that errors or times out gets the failure floor. The test
    split is never touched here."""
    ...


# --- existing: quality-diversity program database (MAP-Elites + islands) ---

class ProgramDatabase:
    """Keeps scored candidate programs. MAP-Elites: each program is binned by a few
    feature dimensions, one elite per cell. Several islands evolve in parallel with
    periodic migration, to keep the population diverse over a long run."""

    def add(self, program, score):
        ...

    def sample(self):
        """Return (parent_program, [inspiration_programs]) for the next step,
        balancing exploitation of high scorers, exploration for diversity, and
        elitism."""
        ...


# --- existing: the LLM that proposes a new candidate from a prompt ---

def llm_propose(prompt) -> str:
    """Return source code for a new candidate program."""
    ...


# --- open slots to fill ---

# What does a single candidate *program* consist of, and which region of it does the
# LLM change? The seed below is only a minimal runnable placeholder.
SEED_PROGRAM = """
# TODO: fill in a runnable candidate program.
"""

def build_prompt(parent_program, inspirations, task):
    # TODO: assemble a mutation request from the task context, parent program,
    #       and selected examples.
    pass

def discover(task, n_iterations):
    """The evolutionary loop the search runs (reused structure): seed the database,
    then repeatedly sample a parent, prompt the LLM for a child, score it, insert it;
    return the best program found."""
    db = ProgramDatabase()
    db.add(SEED_PROGRAM, fitness(SEED_PROGRAM, task))
    for _ in range(n_iterations):
        parent, inspirations = db.sample()
        prompt = build_prompt(parent, inspirations, task)
        child = llm_propose(prompt)
        db.add(child, fitness(child, task))   # test split untouched
    return db.best()
```
