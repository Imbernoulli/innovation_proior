The hidden function is a DNF, and I get to choose the learner, so the first move I want to make is the
one that puts the answer's *shape* into the model itself. The target is literally an OR of ANDs of
literals; if I build a network whose hypothesis class is relaxed DNF — soft conjunctions OR'd together —
then I am not asking gradient descent to discover that Boolean functions exist, I am only asking it to
find *which* conjunctions. That inductive-bias argument is strong enough that it should be the floor I
measure everything else against: if encoding the exact target structure does not win, that itself is the
finding. So step 1 is the differentiable-DNF learner, and the question I have to answer before I can
even fill `build_model` is how to make "OR of ANDs of literals" differentiable without a single bias
threshold to tune.

First I need the base rate each family sits at, because they are not equally hard. A width-`w`
conjunction over uniform Boolean inputs is satisfied with probability `2^{-w} = 1/16` for `w = 4`, so if
the `s` terms fired independently the target is positive with probability `1 − (1 − 1/16)^s`. For random
(`s = 10`) that is `1 − 0.9375^{10} = 0.476` — a balanced concept, majority prediction earns only ~0.52.
The sparse family is the same `s = 10, w = 4` concept inside a 12-variable junta, so the same balanced
~0.476; the 48 irrelevant variables do not change the positive rate, they only bury the signal. Monotone
(`s = 20`) gives `1 − 0.9375^{20} = 0.725`, a positively-biased concept — but that independent estimate
overcounts, because 20 width-4 terms pack 80 literal slots into 40 coordinates, so terms share variables,
their firings are positively correlated, and the true positive rate sits somewhat below 0.72. So monotone
hands a learner a lot of free accuracy from the constant "yes"; a high monotone number can hide whether
the *terms* were recovered, while on random and sparse every point above 0.5 is earned structure. The
honest test of the DNF bias is the balanced families.

Start from the connectives, because the whole architecture follows from how I relax them. Extend truth
values from `{0,1}` to `[0,1]` and pick smooth surrogates that agree on the corners. The product family
does exactly that: `NOT x = 1 − x`, `x AND y = x·y`, and by De Morgan `x OR y = 1 − (1−x)(1−y)`. The AND
is the load-bearing choice. An additive perceptron has to *count* how many inputs are on and compare to
a bias threshold — and that bias is precisely the part of a trained net you cannot read back as logic,
and the part the optimizer chases when the input statistics shift. Multiplication sidesteps it
entirely: `x_1·x_2·x_3` is 1 exactly when every factor is 1 and 0 the instant any factor is 0, with no
threshold and no bias anywhere. So a conjunction neuron is a product, an OR neuron is the De Morgan
dual (a noisy-OR), and the model is `DISJ(CONJ(x))` — a bank of soft conjunctions, OR'd.

Now the real design problem: a conjunction should AND over *some subset* of variables, and I do not
know which subset, so I have to learn the selection differentiably. A softmax over the `n` inputs is
wrong twice over. It puts almost all its mass on the single largest logit, so it selects essentially
*one* variable; to select `w = 4` literals I would need four separate softmaxes per clause and would
have to commit to the clause width `w` up front, baking a number into the architecture that I am
supposed to be learning. And a softmax forced to concentrate over a long input vector converges
painfully: with `n = 60` the gradient that sharpens one logit relative to 59 others is diluted across
all of them, so the selection crawls toward a decision. The right primitive is a per-variable,
*independent* include/exclude decision, not a competition among variables. So attach to each input `i`
of clause `j` a membership and fold it into the product through a factor that is the identity (= 1) when
the variable is excluded and equals `x_i` when it is included: `F = 1 − m_i(1 − x_i)` — `m_i = 0` gives 1
for either `x_i`, `m_i = 1` gives `x_i`. So `O_conj = Π_i (1 − m_i(1 − x_i))`, and the
clause can be read straight off the memberships. Here, because the target may use *negative* literals
too (the random and monotone families differ exactly on this), I want a three-way categorical per
variable — include-positive, include-negative, skip — via a softmax over three logits, so a term can use
either polarity but can never select `x_i` and `¬x_i` at once. The conjunction becomes a product over
`pos·x + neg·(1−x) + skip`, with the three probabilities summing to 1. Note this is a softmax over
exactly three options per variable, not over `n` — the slow-concentration objection I just raised does
not touch it, because three logits sharpen fast.

Three implementation facts are not optional; they decide whether this trains at all, and they go
straight into the forward pass. First, **log-domain products.** A clause is a product
of `n` factors each `≤ 1`. If most memberships were on so that factors averaged around 0.5, the product
over `n = 30` would be `0.5^{30} ≈ 9.3×10^{-10}`, and over `n = 60` it is `0.5^{60} ≈ 8.7×10^{-19}` —
still representable in float32 but with a gradient so small it is numerical noise, and the chain rule
multiplies that vanishing product back through every factor, so learning stalls before it starts. So I
compute `exp(Σ log(factor))` with the factors clamped to `[1e-6, 1]` — valid because every factor lies
in `[0,1]` — which turns a product of `n` small numbers into a sum of `n` moderate logs and keeps the
gradient alive. Second, **sparse initialization.** I set the skip logit to ≈ 4 and the two include logits to ≈ −4, so the softmax puts skip
probability `e^4 / (e^4 + 2e^{-4}) = 54.60 / 54.63 = 0.9993` and each include probability at `0.0183 /
54.63 ≈ 0.00034`. Then every literal factor is `pos·x + neg·(1−x) + skip ≈ 0.9993` regardless of `x`,
and a whole clause is `0.9993^n`: for the monotone family `0.9993^{40} = e^{−0.0136} = 0.9865`, for the
sparse family `0.9993^{60} = e^{−0.0204} = 0.9798`. So clauses start as the near-constant-1 function
rather than crushed toward 0 — exactly the regime where a gradient can lift a *few* memberships without
first having to climb out of an underflowed product. I also initialize the per-term gate logits low
(≈ −3), so with `σ(−3) = 0.047` each term is mostly switched off and the noisy-OR starts quiet rather
than saturated. Third, **per-term gates and sparsity penalties.** I over-provision terms
(`n_terms = max(4·num_terms, 32)`), put a `σ(g_j)` gate on each so the noisy-OR can switch a term off,
and add a one-sided literal-width penalty `(usage − w)_+^2` plus a mean-gate penalty, pushing toward
short clauses and few active terms — a crisp DNF rather than a mush of half-on memberships.

The over-provisioning factor — 40 clauses to fit 10 real terms, 80 to fit 20 — gives the optimizer four
redundant slots per true term, slack to find *a* covering assignment rather than the exact minimal one.
The membership tensor is only `n_terms × n_features × 3` (a few thousand logits per family), and each
one *means* something — is-this-literal-in-this-clause — so the model is a legible formula, not a
distributed code.

There is one more piece this task's substrate lets me add that a generic differentiable-DNF learner
would not: a **data-driven warm start.** Pure gradient descent on a noisy-OR of products is slow to
discover *which* width-`w` conjunctions matter, and I have 20000 labelled examples sitting right there.
So before training I mine candidate terms directly from the data: score each variable by the gap between
its positive-class and negative-class means (`|E[x_i | y=1] − E[x_i | y=0]|`), keep the top variables
(more for the monotone family, a tight window around the junta size for the sparse family), enumerate
width-`w` combinations and polarity patterns over them, and score each candidate conjunction by its
precision plus a recall term on the training labels, discarding any that fire on fewer than a handful of
examples. I should size this enumeration per family, because it is where the mining either stays cheap
or blows up. On monotone I keep the top 30 variables and, since all literals are positive, enumerate
only `C(30,4) = 27405` positive-pattern candidates — under the 36000 cap, so nothing is dropped. On
sparse I keep `max(sparse_subset + 4, 16) = 16` variables and enumerate `C(16,4) × 2^4 = 1820 × 16 =
29120` — also under the cap. On random I keep the top 18 variables but must consider both polarities, so
`C(18,4) × 2^4 = 3060 × 16 = 48960`, which *exceeds* the 36000 cap and gets subsampled down to 36000,
keeping only about 73% of the candidate conjunctions. That asymmetry is already a warning: the one
family where the polarity is genuinely ambiguous is also the one where I cannot afford to enumerate the
full candidate set, so the warm start there is working from a randomly thinned pool. The top-scoring
distinct terms initialize the memberships at saturated logits, so the network *starts* near a plausible
DNF and gradient descent only has to refine it. This warm start is probably the single biggest reason a
soft-DNF learner could beat a from-scratch one — and if the mining picks the *wrong* terms, the network
inherits the mistake.

The scoring rule inside the mining is what decides whether a candidate conjunction is worth a saturated
init or is noise. I score each candidate by `precision + 0.25·recall`
and blank out any that fire on fewer than 4 training examples. On the random
family: a *true* width-4 term fires on `1/16` of uniform inputs, so out of 20000 examples it fires on
about 1250, and every one of those is a positive, so its precision is ≈ 1.0 and its recall against the
~9500 positives (from the ~0.476 positive rate) is `1250/9500 ≈ 0.13`, giving a score ≈ `1.0 + 0.25·0.13
= 1.03`. Now a *spurious* width-4 conjunction that happens to correlate: say it fires on 1250 inputs of
which 900 are positive; its precision is `0.72` and recall `900/9500 = 0.095`, score `0.72 + 0.024 =
0.74`. The precision term does the separating work — real terms sit near 1.0, accidental ones fall away
— and the small `0.25·recall` weight is deliberately small so it only breaks ties among high-precision
candidates toward the ones that *cover* more positives, rather than letting a high-recall-but-impure
clause outrank a pure one. The `covered < 4` blank-out kills candidates supported by a handful of points
whose apparent precision is a small-sample artifact — with only 3 supporting examples a precision of 1.0
is one coin-flip from 0.67 and means nothing. That safeguard matters because a single mistakenly-installed
impure term contributes its own slice of union false-positive mass through the noisy-OR — and on random
the mining is fighting from a candidate pool already thinned to 73%.

The mechanism reads out cleanly on positives — a test point satisfying an installed term drives that
clause's product to ≈ 1 and the noisy-OR output to ≈ 1, so a positive is reported exactly when *some*
installed term fires. The exposure is entirely on the negatives, where each clause's small residual
`term_prob` leaks into the union.

I considered two variants and rejected both. Hard-sampling the selection each step (straight-through /
Gumbel) would remove the residual softness, but it injects variance into every gradient — the sampled
mask changes which literals are even present — and on the balanced random family that variance would
fight the already-weak mean-gap gradient; the width penalty `(usage − w)_+^2` buys most of the crispness
without the tax. Replacing the noisy-OR with a soft-max-over-terms would dodge the accumulation blowup
but breaks disjunction semantics — a point satisfying two terms should read as strongly positive, not be
capped at one clause. So I keep the noisy-OR and accept its accumulation of precision errors as the
exposure on the random family.

So why do I believe the convergence story at all? Differentiate one membership in the simple two-way
form: `∂O_conj/∂m_i = −(1 − x_i)·Π_{k≠i}(1 − m_k(1 − x_k))`, and the trailing product is non-negative,
so the sign is set by `−(1 − x_i)`. On a point where `x_i = 1` the derivative is exactly zero — an
on-example says nothing about whether `x_i` should be required. Only a point where `x_i = 0` yet the
clause is supposed to fire produces a gradient, and it *raises* `m_i` toward requiring `x_i`. The
memberships are driven by counterexamples: to install a literal I need points where dropping it would
let the clause fire wrongly. With 20000 uniform examples each width-4 conjunction's relevant
counterexamples are present in abundance — a given variable is off in half the examples — so a single
layer should converge if the warm start put it in the right neighborhood. I train with AdamW + BCE on
the logits the noisy-OR emits, 30 epochs, batch 512, the two sparsity penalties at small weight (1e-4),
and threshold at logit 0 (probability 0.5) for the 0/1 prediction. The full scaffold module —
`NeuralDNF` with the three-way literal, the noisy-OR, the term mining, and the penalised training loop —
is in the answer.

Where should this floor be strong and where weak? **Monotone** is friendliest: all literals positive, so
the three-way categorical mostly chooses between include-positive and skip, the mean gap is a clean
signal, and mining enumerates the full `C(30,4)` candidates unthinned — a high raw number, though partly
the constant-yes bias rather than recovered terms. **Sparse** is favorable too: 12 of 60 variables
relevant, the mean gap exposes them, and the 16-variable window makes the width-4 enumeration cheap — but
its mining matches over `n = 60`, so its fit time should be the longest of the three. **Random** is where
I am worried: non-monotone, so a real three-way decision per variable, a weaker mean-gap signal, and the
family whose candidate pool overflowed the cap and was thinned to 73%. If the warm start there mines
imprecise terms, the noisy-OR compounds the error — on a true negative with per-term residual `p` the
output is `1 − (1 − p)^{40}`, which is 0.33 for `p = 0.01` and 0.55 for `p = 0.02`. A one-to-two-percent
per-clause leakage, invisible term by term, unions into a third-to-a-half of negatives flipped positive;
the noisy-OR accumulates precision errors, it does not average them.

So a-priori: strong monotone and sparse, weak random, the geometric mean dragged by the mixed-polarity
family. If random lands well below the other two, the culprit is the noisy-OR unioning imprecise mined
terms — and if even the favorable families fall short of what one conjunction per root-to-leaf path buys
a plain decision tree for free, then hand-shaping the class into relaxed DNF has not paid, which is what
the next steps test.
