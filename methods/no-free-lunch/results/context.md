# Context: matching algorithms to problems in black-box search and learning

## Research question

By the mid-1990s there is an abundance of general-purpose "black-box" optimizers — evolutionary
algorithms, simulated annealing, hill climbing, tabu search — and, in parallel, an abundance of supervised
learning algorithms — decision trees, nearest neighbors, neural nets, Bayesian methods. A practitioner
facing a new problem must pick one. The folklore is that some of these are simply *better*: that a
well-designed evolutionary algorithm beats random search "on average," that hill climbing beats hill
descending when you want a maximum, that simpler hypotheses (Occam's razor) generalize better.

The precise question is: **is any of this true with no assumption about the problem?** Given two search
algorithms a₁ and a₂, how does the set of cost functions on which a₁ beats a₂ compare to the set on which
a₂ beats a₁? Given two learning algorithms, is there an *a priori* reason — a reason that holds before you
say anything about which problems are likely — to prefer one over the other?

## Background

**The problem of induction (Hume, 1748).** Every inference from observed data to an unobserved case rests on
a Uniformity Principle: that "instances, of which we have had no experience, must resemble those, of which
we have had experience." Hume's dilemma shows this principle cannot be justified. A demonstrative (a priori)
argument fails because the negation involves no contradiction — "it implies no contradiction that the course
of nature may change, and that an object seemingly like those which we have experienced, may be attended
with different or contrary effects." A probable (empirical) argument fails because it already assumes the
principle — it "proceed[s] upon the supposition, that the future will be conformable to the past," which is
"taking that for granted, which is the very point in question." So there is no assumption-free justification
for generalizing from a training set to new points. This is a philosophical statement; it carries no
quantitative claim about specific algorithms.

**Conservation of generalization performance (Schaffer, 1994).** Define a learner's generalization
performance GP as its test accuracy minus the accuracy of random guessing — positive when it beats chance,
negative when it does worse. Schaffer argues that summed (averaged) over *all* possible learning tasks,
total GP is zero: "positive performance in some learning situations must be offset by an equal degree of
negative performance in others," so "no learning bias can outperform any other bias over the space of all
possible learning tasks." Whatever a simplicity bias like Occam's razor gains on a subset of problems it
pays for on the complement.

**Off-training-set error.** Generalization is about test points the learner has *not* seen. In-sample or
IID-test error mixes two different things: how well the learner reproduces the training labels (which can be
pure memorization) and how well it predicts unseen inputs (the only thing induction is about). A clean
statement about "distinctions between learners" must therefore be made on **off-training-set (OTS)** error —
the average loss restricted to inputs q outside the training set's input list d_X. With a noise-free target
f, a hypothesis h, a loss function L(h(q), f(q)), and a test point q ∉ d_X, OTS error is
Σ_{q∉d_X} L(h(q), f(q)) π(q) for some weighting π over test points. The zero-one ("homogeneous") loss — one
unit per disagreement — is the cleanest case.

**Bias and variance.** The standard decomposition of squared-error risk is
E[(y − ĥ)²] = bias² + variance + noise. A learner trades the two: a rigid, simple learner has high bias and
low variance; a flexible learner the reverse. Averaging several deterministic learners reduces variance
without changing bias, by the convexity identity (z − [α+β]/2)² ≤ ½(z−α)² + ½(z−β)².

**The prevailing wisdom.** Practitioners "match" algorithms to problems all the time, but on a heuristic
basis; benchmark papers report an algorithm winning on a few sample functions and implicitly generalize. The
amount of revisiting an algorithm does (re-evaluating points it has already seen) is a complicated,
algorithm-and-problem-dependent quantity that distorts oracle-based performance counts and cannot simply be
filtered out.

## Baselines

The prior methods a no-assumption result would be measured against (and would reframe):

- **Random search.** Sample distinct points of 𝒳 uniformly without replacement, keep the best. The trivial
  baseline everyone assumes is beatable. Its expected best-so-far depends only on the cost histogram of f,
  not on any cleverness.

- **Hill climbing / hill descending.** From the current point, examine neighbors and move to the best (lowest
  for descending, highest for climbing); iterate, with random restarts to escape local optima. Exploits
  local cost structure.

- **Simulated annealing (Kirkpatrick et al.).** A Metropolis walk with a cooling temperature, accepting
  uphill moves with probability ~exp(−Δ/T). Mimics statistical mechanics.

- **Evolutionary algorithms.** Maintain a population, select by fitness, recombine and mutate. Mimic natural
  selection. The most popular black-box optimizers of the period.

- **Tabu search (Glover).** Hill climbing with a memory of recently visited points, forbidden for a while, to
  avoid cycling and revisiting.

- **Branch and bound.** Uses the cost structure of partial solutions to prune. Deliberately outside the
  black-box class, because it relies on knowing more than the oracle returns.

- **Cross-validation and Occam's razor (as learning baselines).** Choose among candidate learners by held-out
  performance; prefer simpler hypotheses. The default model-selection and inductive-bias heuristics.

## Evaluation settings

The natural yardsticks that exist at the time. For optimization: a finite search space 𝒳 and a finite set of
cost values 𝒴 (as on any digital computer, 32- or 64-bit values), a cost function ("objective"/"energy")
f: 𝒳 → 𝒴, an algorithm run for m distinct function evaluations, and a performance measure read off the
ordered sequence of cost values seen — typically the best-so-far min_i d_m^y(i), or how fast the best-so-far
drops, or how the search tracks a moving optimum for time-varying problems. The oracle model: count function
evaluations (here, *distinct* evaluations) rather than wall-clock time. For learning: a finite input space
𝒳 and output space 𝒴, a target f (possibly a noisy input–output relation φ), a training set d of m examples,
a hypothesis h, a loss function L (zero-one the canonical homogeneous case), and the **off-training-set**
error as the metric — average loss over test inputs q ∉ d_X. The space of all problems is
ℱ = 𝒴^𝒳, of size |𝒴|^|𝒳|. The benchmark function suites and standard datasets are the existing comparison
substrate.
