## Research question

Text is cheap and arrives in floods; what is expensive is knowing which class each piece of
text belongs to. A statistical classifier can be learned automatically from labeled examples,
but only after a human annotator has hand-labeled them, and that annotation is the dominant
cost. The precise problem is sample efficiency under a labeling budget: given a large pool of
unlabeled examples and an oracle (a human annotator) who will label only a limited number of
them, choose *which* examples to send to the oracle so that a classifier trained on the
resulting labels reaches a target effectiveness with as few labels as possible.

The difficulty is sharpest exactly where it matters. Many useful categories are rare — a
target class might cover one example in a thousand. If examples to label are drawn uniformly
at random from the pool, then a random sample of 500 will, in expectation, contain about 500
negatives and essentially no positives, so it cannot teach a classifier to recognize the
positive class at all. Labeling effort is being spent on examples the classifier already
handles, while the informative ones — the ambiguous cases near the class boundary, and the
rare positives — are starved of labels. A solution would have to (1) draw the examples it asks
about from the real data rather than fabricating them, so a human can actually label them;
(2) work with whatever classifier is being trained, from very small training sets upward;
(3) turn "which examples are worth labeling" into a concrete, budgeted selection rule the
training loop can run each round; and (4) do so without assuming the data are noise-free or
that a perfect classifier exists. Each prior approach below meets some of these and breaks on
others.

## Background

By the early 1990s, casting text tasks — retrieval, routing, filtering, categorization, word
sense disambiguation, tagging — as supervised classification is established practice, because
it lets statistics and machine learning build the classifiers automatically (Hayes 1992)
rather than by hand-written rules. Building a classifier this way still requires manually
annotating training data with class labels, which "takes less skill and expense than building
classification rules by hand" but is the recurring bottleneck whenever a new category or
corpus appears. The amount of online text is increasing, so the demand for cheap classifiers
is increasing with it.

Two background facts about the world set up the problem:

- **Random labeling is inefficient when the informative examples are rare.** For a low-frequency
  category, a uniform random sample is dominated by negatives, so it provides little signal
  about the boundary or the positive class. The marginal value of one more random label falls
  as the training set grows — eventually almost every randomly drawn example is one the
  classifier already classifies correctly, and labeling it teaches nothing.
- **The cost asymmetry: unlabeled examples are abundant and free, labels are scarce and
  expensive.** This is what makes it worth spending computation to *choose* which examples to
  pay to have labeled.

The load-bearing theoretical concept is the **version space** (Mitchell 1982): given a
hypothesis class C and a set of labeled examples, the version space is the subset of
hypotheses in C consistent with all the labels seen so far. Learning, in this view, is
search — progressively shrinking the version space until it pins down the target concept. The
key derived object is the **region of uncertainty** (Cohn, Atlas & Ladner 1990, 1994): the set
of input points on which two consistent hypotheses still disagree,

```
R(S^m) = { x : there exist c1, c2 in C, both consistent with all m labeled examples, and c1(x) != c2(x) }.
```

Its probability mass `alpha = Pr[x in R(S^m)]` is what matters. Any hypothesis consistent with
the labels has error at most `alpha` (all disagreement among consistent hypotheses lives inside
R, so the true label can only differ from a consistent hypothesis on R), and `alpha` is also
the probability that a freshly drawn random example reduces the uncertainty at all. As more
examples are labeled, `alpha` shrinks. A point drawn from *outside* R leaves R unchanged and is
wasted; a point from *inside* R necessarily shrinks it. So a random draw over the whole domain
informs the learner with probability only `alpha`, which decays toward zero — its efficiency
decays with it. Restricting draws to within R keeps every example informative. This is the
**selective sampling** idea (Cohn et al.): treat learning as a sequence of draws, and choose
each draw to fall where the learner is still uncertain.

For probabilistic classifiers, the relevant background is Bayesian text classification: posterior
class probabilities via Bayes' rule, the two-class odds ratio, and the word-independence
("naive Bayes") decomposition that makes the likelihood tractable for word features, together
with its known failure — the independence assumption is always wrong for natural-language
features (words co-occur), so the raw posterior estimates are systematically miscalibrated.
Logistic regression (McCullagh & Nelder 1989) is the standard remedy for combining predictors
into a calibrated posterior, and smoothed likelihood-ratio estimators (Gale & Church 1990) are
the standard fix for the extreme estimates that arise when positive and negative training sets
are very different in size, as they are before any positives have been found.

## Baselines

**Membership-query learning (Angluin 1988).** The learner is allowed to *synthesize* any point
in the input space and ask the oracle for its label. Concept-learning theory shows this can
identify a target concept from far fewer queries than passive learning, because the learner
can probe exactly where it is confused. **Limitation:** a synthesized point is usually not a
natural example. Lang & Baum (1992) trained a network on handwritten characters with
synthesized membership queries and found that many generated query images were artificial
hybrids with no recognizable symbol — gibberish the human oracle could not label. For text the
problem is worse: a synthesized "document" is a meaningless string. So the theoretical
query-efficiency of synthesis cannot be cashed out when the oracle is a human and the inputs
are natural language.

**Selective sampling via the region of uncertainty (Cohn, Atlas & Ladner 1990, 1994).** Instead
of synthesizing, filter real examples: draw candidates from the actual distribution and only
query those that fall inside the region of uncertainty R(S^m). Each such query is guaranteed to
shrink R, so efficiency does not decay the way random sampling's does, and because the queries
are real data points the oracle can always label them. **Limitation:** R(S^m) is defined
through the entire version space, and computing it exactly means maintaining the set of all
consistent hypotheses and testing, for each candidate point, whether two of them disagree on
it — intractable for the rich model classes used on real tasks, and it must be recomputed after
every new label. It is also stated as a binary in/out test on points, not as a way to rank a
finite pool under a fixed labeling budget. Approximations are required.

**Query by committee (Seung, Opper & Sompolinsky 1992; Freund, Seung, Shamir & Tishby
1992).** Approximate the region of uncertainty without enumerating the version space: draw two
hypotheses at random from the version space, apply both to an incoming example, and query it
only when the two disagree — disagreement is a sample-based proxy for "inside R". Freund et al.
prove that, under their assumptions, after examining `m` random examples the learner needs
only logarithmically many labels, while generalization error still falls as `O(1/m)` in those
examined examples — equivalently, exponentially fast as a function of query count.
**Limitation:** the analysis assumes the data are
noise-free, that a perfect deterministic classifier consistent with all data exists
(realizability), and that one can draw hypotheses uniformly at random from the version space.
All three are problematic on real, noisy text tasks, where no consistent perfect classifier
exists and sampling the version space is not feasible. It also needs a committee of classifiers
rather than the single classifier one is already training.

**Relevance feedback / "relevance sampling" (Salton & Buckley 1990).** Also a sequential,
non-random selection: ask the oracle to label the examples the current classifier judges most
likely to be class members. In retrieval this is exactly what a user wants (more relevant
documents). **Limitation:** as a strategy for gathering *training* data it degrades as the
classifier improves — the examples it judges most likely positive become the ones it is already
most sure about, so the labels are increasingly redundant, and it tends to re-select
near-duplicates of known positives rather than the cases that would teach the classifier
something new. It biases the labeled set toward predicted positives rather than toward
informative examples.

## Evaluation settings

The natural yardstick for a sample-efficiency method is the effectiveness of the resulting
classifier as a function of the number of labeled examples, compared against the two obvious
sequential and non-sequential alternatives (relevance feedback and uniform random sampling)
across a range of sample sizes, plus a classifier trained on the full labeled corpus as a
ceiling.

- **A text-categorization corpus with rare categories.** News-wire story titles (lower-cased,
  punctuation stripped, whitespace-tokenized) split into a large training pool and a held-out
  test set, with a set of binary categories chosen to be low-frequency (on the order of 0.001
  to 0.005 of items) so that the rare-positive regime — the hard case for random sampling — is
  exercised. Each category is treated as a separate binary classification task. A small seed of
  a few positive examples plus a few random examples forms the initial labeled set; the protocol
  repeats over many rounds, retraining after each round, and is run multiple times from
  different seed subsamples because the seed can strongly affect the trajectory.
- **Metrics.** Recall and precision on the test set, combined into a single effectiveness number
  via van Rijsbergen's E-measure / its complement F (with recall and precision equally
  weighted). Effectiveness is plotted against the number of labeled examples; sample efficiency
  is read off as how few labels a strategy needs to reach fixed effectiveness levels. Classifier
  decisions use the minimum-error-rate loss (both error types equally costly).

## Code framework

A pool-based active-learning harness already exists and is fixed: it owns the labeled/unlabeled
split, retrains the model after each round, and exposes the trained classifier's class-posterior
outputs. The only open slot is the **batch acquisition rule** — given the current trained
classifier and a labeling budget `n`, which `n` currently-unlabeled pool examples should be
sent to the oracle this round. The rule can ask the base `Strategy` class for softmax posterior
probabilities on any pool slice, and it returns indices into the pool.

```python
import numpy as np


class Strategy:
    """Fixed pool-based active-learning base class. Owns the pool (self.X features,
    self.Y labels, self.idxs_lb boolean labeled mask, self.n_pool size), retrains the
    model, and exposes the trained classifier's posterior outputs. The acquisition rule is the
    only thing to design."""

    def predict_prob(self, X, Y):
        """Softmax posterior probabilities, shape [len(X), n_classes]."""
        ...

    def query(self, n):
        # Return n indices into self.X of currently-unlabeled examples to send to the oracle.
        pass


class CustomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # TODO: the acquisition rule we will design.
        #       Given the current trained classifier's outputs on the unlabeled pool,
        #       score the examples and return the n indices worth labeling this round.
        pass
```

The outer loop calls `query(n)` each round, labels the returned examples, adds them to the
labeled set, and retrains. The single empty slot is the scoring-and-selection rule.
