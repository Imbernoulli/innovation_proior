# Context: learning a classifier from a large unlabeled pool under a label budget

## Research question

We have a large collection of inputs but almost none of them are labeled, and getting a label
is the expensive step. Concretely: a fixed pool of `N` inputs `x_1, ..., x_N` is available for
free (text, images, speech segments, molecules), but a human or laboratory oracle has to be
paid — in time, money, or expertise — to attach a label `y` to any one of them. We are allowed
to label only a small budget of them, in rounds: choose a batch of `n`, send them to the
oracle, fold the returned labels into the training set, retrain the model, and repeat. The
question is which `n` of the still-unlabeled inputs to send out on each round so that the model
reaches the best accuracy for the fewest labels.

This is a *selection* problem layered on top of ordinary supervised learning. The model
family, the training procedure, and the retraining loop are all fixed and not in question; the
only free choice is the rule that picks the next batch from the unlabeled pool. The pain point
is sharp because in many real settings unlabeled data is effectively unlimited while the label
budget is tiny — speech annotation can take 10x real time, entity annotation half an hour per
short document, biomedical labels require PhD-level annotators — so every label spent on an
uninformative input is a label not spent where it would have moved the model. A good selection
rule has to make the labeled training set *worth* its cost. But before any clever rule can be
trusted, there has to be a reference point: a selection rule whose behavior is fully understood
and whose statistical guarantees are unconditional, against which any cleverer rule is judged.

## Background

The field state is "learning from examples": a supervised learner is handed a training set and
fits a hypothesis to it. The dominant theoretical lens on *how many* examples this takes is the
PAC / VC framework (Valiant 1984; Blumer, Ehrenfeucht, Haussler & Warmuth 1989; Haussler
1992). It posits a fixed but unknown input distribution `P` and an unknown target labeling `t`,
defines the error of a hypothesis `c` as the probability it disagrees with the target on a
fresh input,

```
err(c) = Pr_{x ~ P}[ c(x) != t(x) ],
```

and asks: how many training examples `m`, each *drawn independently from `P` and labeled*, are
needed so that any hypothesis consistent with them has `err(c) <= eps` with confidence
`1 - delta`? For a class of VC dimension `d` the answer is

```
m = O( (1/eps) * ( d * ln(1/eps) + ln(1/delta) ) ),
```

and the entire guarantee rests on one premise: the training examples are an i.i.d. sample from
the same `P` the learner will be tested on. The sample is the bridge between training error and
test error; if the training set is distributed like deployment, low empirical error transfers
to low true error. This is the law-of-large-numbers / uniform-convergence backbone of
supervised learning — empirical risk converges to true risk because the training points are a
faithful, unbiased draw from `P`.

A second, older body of theory speaks directly to "which subset to label": survey sampling
(Cochran, *Sampling Techniques*). It studies how to pick a subset of a finite population so
that a quantity estimated on the subset is a good estimate for the whole. Its foundational
design is simple random sampling without replacement (SRS-WOR): pick `n` of the `N` units so
that every subset of size `n` is equally likely. Two properties of SRS-WOR are load-bearing.
First, every unit has the same inclusion probability `pi_i = n/N`; equivalently, unit `i`
appears in `C(N-1,n-1)` of the `C(N,n)` possible samples, so
`pi_i = C(N-1,n-1)/C(N,n) = n/N`. The Horvitz-Thompson estimator divides each sampled unit's
value by its `pi_i`; with equal `pi_i`, the unweighted sample mean is an unbiased estimate of the
population mean. Second, sampling without replacement has strictly lower estimator variance
than sampling with replacement. If `sigma^2` is the finite-population variance with denominator
`N`, the sample-mean variance is `Var_WOR = (sigma^2/n)(N-n)/(N-1)`, so the
finite-population correction `(N-n)/(N-1)` is exactly the variance reduction from not drawing
the same unit twice.

There is also a known, sharp distinction between learning from a *given* random sample and
learning where the learner gets to *choose* what to ask about. On the toy problem of locating a
threshold on the unit interval, achieving position error `eps` from random labeled examples
takes `O((1/eps) ln(1/eps))` of them, whereas a learner allowed to *query* points of its choice
(binary search) needs only `O(ln(1/eps))` — an exponential gap (Cohn, Atlas & Ladner 1994).
This says the *act of choosing* can, in principle, buy enormous label savings.

Two motivating observations about passive learning frame the whole problem:

- **The rare-class failure.** If the positive class is rare — say 1 in 1000 inputs is a class
  member — then a labeled subset of 500 inputs drawn blindly will, in expectation, contain ~500
  negatives and essentially no positives, which cannot train a classifier to recognize the
  positive class at all. The class imbalance is faithfully reproduced in the subset, which is
  exactly the problem when the rare class is the one you care about.
- **The diminishing-information argument** (Cohn, Atlas & Ladner 1994). Track the "region of
  uncertainty" `R` — the part of input space where two hypotheses still consistent with the
  labeled data disagree — and let `alpha = Pr_{x~P}[x in R]` be the probability that a fresh
  input lands in it. Only inputs in `R` can change the hypothesis; an input outside `R` is
  already determined and teaches nothing. As more points are labeled, `R` shrinks and `alpha`
  decreases toward zero, so the probability that a freshly *drawn-at-random* point falls in the
  still-useful region — and hence the information per labeled point — also decreases toward
  zero. Late in training, a blind draw spends most of its labels re-confirming regions the model
  has already settled.

## Baselines

The prior approaches a selection rule would be compared against or react to.

**Simple random sampling / survey-sampling estimation (Cochran).** Treat a finite population as
a set of units and draw a subset by SRS-WOR. Core idea and math: equal inclusion probability
`pi_i = n/N`, Horvitz-Thompson unbiased estimation, variance reduced by the finite-population
correction. **Gap:** the theory is built to estimate a fixed population quantity (a mean, a
total), where representativeness is the whole goal. A classifier's test error is a population
mean of a loss, but the loss is induced by a learned decision rule; the hard part is spending
labels so that the fitted rule improves where it is uncertain, often near a boundary or in a
rare class. Survey sampling by itself has nothing to say about concentrating labels where a
decision rule is hard.

**Passive "learning from examples" under PAC/VC (Valiant 1984; Blumer–Ehrenfeucht–Haussler–
Warmuth 1989; Haussler 1992).** Draw `m` i.i.d. labeled examples and fit a consistent
hypothesis; with high probability its error is `<= eps` once `m` meets the sample-complexity
bound above. Core idea: the i.i.d. premise makes empirical risk a faithful proxy for true risk,
so more random labeled data monotonically improves the guarantee. **Gap:** the bound is
worst-case and *non-adaptive* — it counts examples handed to the learner and never models the
possibility of choosing which inputs to label. The required `m` grows like `1/eps`, so reaching
small error demands many labels even when most of them land in regions the learner has already
resolved (the diminishing-information argument above quantifies exactly this waste).

**Membership-query learning (Angluin 1988).** Let the learner *synthesize* the input it most
wants labeled and query an oracle for it; on structured domains this can be exponentially more
label-efficient than random examples (the `ln(1/eps)` vs `1/eps` interval gap). **Gap:** the
synthesized queries need not correspond to any real, labelable input — a learner asked to
generate the most informative handwritten digit produces uninterpretable hybrids a human cannot
label — and it presumes an oracle that will label arbitrary points, which the pool setting (a
fixed list of real inputs to choose among) does not provide.

**Selective sampling (Cohn, Atlas & Ladner 1994).** Restrict labeling to inputs that fall in
the region of uncertainty `R(S^m)`, so each labeled point is one that can still change the
hypothesis; this keeps the per-label information from collapsing the way a blind draw's does.
Core idea: maintain (an approximation to) `R`, and only ask about points inside it. **Gap:** by
drawing preferentially from `R` rather than from `P`, the labeled set is no longer a faithful
i.i.d. sample of the deployment distribution, so the PAC guarantee that justified passive
learning no longer applies and any improvement is conditional, not certain; an early, poorly fit
model defines `R` badly and can steer the queries into an unrepresentative corner; and the rule
can over-concentrate on one part of the domain.

## Evaluation settings

The natural yardsticks already in use for this kind of selection problem:

- **Pool-based protocol.** Start from a small labeled seed and a large unlabeled pool; repeat
  for a fixed number of rounds: select a batch of `n` unlabeled inputs, obtain their labels,
  add them to the labeled set, retrain, evaluate. A fixed total label budget across all rounds.
- **Datasets.** Standard tabular classification benchmarks — e.g. OpenML's letter recognition,
  spambase, and splice — and text categorization corpora (newswire titles, 20-Newsgroups
  binary tasks such as baseball vs. hockey). These exist independently of any selection rule.
- **Metrics, higher is better.** Test accuracy after the final round (at the fixed budget); and
  the area under the learning curve — accuracy plotted against the number of labeled inputs over
  all rounds — which captures *sample efficiency*, how fast accuracy rises per label spent. A
  selection rule is judged by how its learning curve sits relative to others across the whole
  range of budgets.

## Code framework

The selection rule plugs into a fixed pool-based harness. A `Strategy` base class owns the pool
features `X`, a boolean mask `idxs_lb` marking which pool indices are already labeled, the pool
size `n_pool`, and the model together with hooks to train it and read its predictions — all of
which already exist and are not in question. The harness drives the rounds: it calls the
strategy to get a batch of indices, obtains their labels, flips their mask bits, and retrains.
The single empty slot is the `query` method: given a batch size `n`, return `n` indices, into
the pool, of currently-unlabeled inputs to send to the oracle. The rule that fills it is exactly
what is to be designed.

```python
import numpy as np


class Strategy:
    """Fixed pool-based active-learning harness. Owns the pool and the model;
    the only thing left to design is the rule that picks which inputs to label."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X                       # pool features, [n_pool, n_features]
        self.Y = Y                       # pool labels (only revealed once queried)
        self.idxs_lb = idxs_lb           # boolean mask: True where already labeled
        self.net = net                   # the model family (fixed)
        self.handler = handler           # data pipeline (fixed)
        self.args = args
        self.n_pool = len(Y)

    def query(self, n):
        # TODO: the selection rule we will design.
        #       Given the current labeled/unlabeled split, return n indices into
        #       the pool of currently-UNLABELED inputs to send to the oracle.
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb           # harness flips mask bits after labeling

    def train(self):
        ...                              # existing: retrain on all currently-labeled inputs

    def predict_prob(self, X, Y):
        ...                              # existing: model's softmax probabilities


# existing outer loop the strategy plugs into
def active_learning_loop(strategy, n_rounds, n_per_round, label_oracle):
    for _ in range(n_rounds):
        idxs = strategy.query(n_per_round)          # pick a batch to label
        new_mask = strategy.idxs_lb.copy()
        new_mask[idxs] = True                       # oracle labels them; mark labeled
        strategy.update(new_mask)
        strategy.train()                            # retrain on the expanded labeled set
```

The outer loop supplies the round structure and the oracle; `query` is where the selection rule
will live.
