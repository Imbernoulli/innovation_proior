The hidden function is a DNF, and I get to choose the learner, so the first move I want to make is the
one that puts the answer's *shape* into the model itself. The target is literally an OR of ANDs of
literals; if I build a network whose hypothesis class is relaxed DNF — soft conjunctions OR'd together —
then I am not asking gradient descent to discover that Boolean functions exist, I am only asking it to
find *which* conjunctions. That inductive-bias argument is strong enough that it should be the floor I
measure everything else against: if encoding the exact target structure does not win, that itself is the
finding. So step 1 is the differentiable-DNF learner, and the question I have to answer before I can
even fill `build_model` is how to make "OR of ANDs of literals" differentiable without a single bias
threshold to tune.

Before I commit, I should know what "winning" even looks like numerically, because the three families
are not equally hard and I want a paper estimate of the base rate each one sits at. A width-`w`
conjunction of literals over uniform Boolean inputs is satisfied with probability `2^{-w}`; for `w = 4`
that is `1/16 = 0.0625`. If the `s` terms fired independently, the target would be positive with
probability `1 − (1 − 1/16)^s`. For the random family (`s = 10`) that is `1 − 0.9375^{10} = 1 − 0.5245 =
0.4755` — essentially a balanced concept, so majority-class prediction earns only about 0.52 and there
is a full 0.48 of accuracy that actually has to be *learned*. The sparse family is a 12-variable junta
with the same `s = 10, w = 4`, so within its relevant coordinates it is the same balanced ~0.476
concept — the 48 irrelevant variables do not change the positive rate, they only bury the signal. The
monotone family is different: `s = 20` gives `1 − 0.9375^{20} = 1 − 0.2751 = 0.7249`, so it is a
*positively-biased* concept where the majority class alone already scores around 0.72. That independent
estimate is an overcount, though — 20 width-4 terms over only 40 variables pack 80 literal slots into 40
coordinates, so terms share variables heavily, their firing events are positively correlated, and the
true positive rate sits somewhat below 0.72. The lesson I carry from this arithmetic is that monotone
gives a learner a lot of free accuracy from the constant "yes," so a high monotone number will look
impressive while hiding whether the *terms* were actually recovered, whereas on random and sparse every
point of accuracy above 0.5 is earned structure. That reframes what I should watch: the honest test of
whether my DNF bias works is the balanced random and sparse families, not the lopsided monotone one.

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
know which subset, so I have to learn the selection differentiably. The instinct to put a softmax over
the `n` inputs is wrong twice over, and I want to make the "twice" concrete rather than wave at it. A
softmax over `n` logits puts almost all its mass on the single largest logit, so it selects essentially
*one* variable; to select `w = 4` literals I would need four separate softmaxes per clause and I would
have to commit to the clause width `w` up front, baking a number into the architecture that I am
supposed to be learning. And a softmax forced to concentrate over a long input vector converges
painfully: with `n = 60` the gradient that sharpens one logit relative to 59 others is diluted across
all of them, so the selection crawls toward a decision. The right primitive is a per-variable,
*independent* include/exclude decision, not a competition among variables. So attach to each input `i`
of clause `j` a membership and fold it into the product through a factor that is the identity (= 1) when
the variable is excluded and equals `x_i` when it is included: `F = 1 − m_i(1 − x_i)`. Check the table —
`m_i = 0` gives 1 for either `x_i`; `m_i = 1` gives `x_i`. So `O_conj = Π_i (1 − m_i(1 − x_i))`, and the
clause can be read straight off the memberships. Here, because the target may use *negative* literals
too (the random and monotone families differ exactly on this), I want a three-way categorical per
variable — include-positive, include-negative, skip — via a softmax over three logits, so a term can use
either polarity but can never select `x_i` and `¬x_i` at once. The conjunction becomes a product over
`pos·x + neg·(1−x) + skip`, with the three probabilities summing to 1. Note this is a softmax over
exactly three options per variable, not over `n` — the slow-concentration objection I just raised does
not touch it, because three logits sharpen fast.

Three implementation facts are not optional, they decide whether this trains at all, and they are
exactly what I encode in the scaffold's `forward`. First, **log-domain products.** A clause is a product
of `n` factors each `≤ 1`. If most memberships were on so that factors averaged around 0.5, the product
over `n = 30` would be `0.5^{30} ≈ 9.3×10^{-10}`, and over `n = 60` it is `0.5^{60} ≈ 8.7×10^{-19}` —
still representable in float32 but with a gradient so small it is numerical noise, and the chain rule
multiplies that vanishing product back through every factor, so learning stalls before it starts. So I
compute `exp(Σ log(factor))` with the factors clamped to `[1e-6, 1]` — valid because every factor lies
in `[0,1]` — which turns a product of `n` small numbers into a sum of `n` moderate logs and keeps the
gradient alive. Second, **sparse initialization**, and I can check it does what I want with the actual
init logits. I set the skip logit to ≈ 4 and the two include logits to ≈ −4, so the softmax puts skip
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

It is worth pricing out this over-provisioning, because it tells me how much capacity gradient descent
has to sort through per family. The membership tensor is `n_terms × n_features × 3`. On the random
family `n_terms = max(40, 32) = 40`, so `40 × 30 × 3 = 3600` literal logits plus 40 term gates plus one
output bias — 3641 parameters. On monotone `n_terms = max(80, 32) = 80`, giving `80 × 40 × 3 = 9600`
plus 80 gates — 9681. On sparse `n_terms = 40`, `40 × 60 × 3 = 7200` plus 40 — 7241. These are tiny by
neural standards, which is the point: with a few thousand parameters that each *mean* something
(is-this-literal-in-this-clause), the model is a legible formula, not a distributed code. The
over-provisioning factor — 40 clauses to fit 10 real terms, 80 to fit 20 — gives the optimizer four
redundant slots per true term, which is slack to find *a* covering assignment rather than needing to hit
the exact minimal one.

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
DNF and gradient descent only has to refine it. This is the harness adapting the architecture to its own
research question — it is the single biggest reason this baseline could beat a from-scratch soft-DNF. I
should keep that in mind, because if the warm start mines the *wrong* terms, the network inherits the
mistake.

The scoring rule inside the mining deserves a moment, because it is what decides whether a candidate
conjunction is worth a saturated init or is noise. I score each candidate by `precision + 0.25·recall`
and blank out any that fire on fewer than 4 training examples. Walk a concrete case on the random
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
is one coin-flip from 0.67 and means nothing. This is exactly the safeguard the noisy-OR needs, because
by the accumulation arithmetic above a single mistakenly-installed impure term contributes its own slice
of union false-positive mass; the mining is the front line against that, and on random it is fighting
with a candidate pool already thinned to 73%.

I also want to trace the forward pass once on a true-positive point, to be sure the noisy-OR reports
"yes" for the right reason. Suppose after warm start clause `j` has installed exactly the four literals
of a real term at saturated logits, and take a test input that satisfies that term. Each of the four
selected factors evaluates to ≈ 1 (the literal is satisfied), each of the other `n − 4` factors is skip
≈ 1, so `O_conj ≈ 1`; its gate `σ(g_j)` after training on an active term is near 1, so `term_prob_j ≈
1`, and the noisy-OR `1 − Π_k(1 − term_prob_k)` has one factor near zero and therefore outputs ≈ 1. The
logit `log(p) − log(1−p)` is large and positive, thresholds to 1 — correct. Now the same input under a
clause whose term it does *not* satisfy: at least one required factor is ≈ 0, `O_conj ≈ 0`, that clause
contributes nothing to the union. So a positive is reported exactly when *some* installed term is
satisfied — the disjunction reads out correctly — and the whole failure mode I am worried about is not
here on the positives, it is on the negatives, where each clause's small residual `term_prob` leaks into
the union. The trace confirms the mechanism is sound on the half of the data it handles cleanly and
fragile on the other half in precisely the way the accumulation formula predicts.

I briefly considered discretizing the selection during the forward pass — a straight-through or
Gumbel-softmax that hard-samples include/skip so the clause is a genuine `{0,1}` conjunction each step —
because that would remove the "continuous relaxation never quite snaps to the formula" residual softness
entirely. I reject it here: hard sampling injects variance into every gradient (the sampled mask changes
which literals are even present from step to step), and on the balanced random family where the signal
is already muddy, that extra variance would fight the weak mean-gap gradient rather than help it. The
soft product with a width penalty gets most of the crispness — the penalty `(usage − w)_+^2` pushes the
softmax mass off the skip class only as far as it needs — without paying the sampling-variance tax. I
also considered replacing the noisy-OR union with a soft-max-over-terms (report the single most-active
clause instead of the union), which would be immune to the accumulation blowup, but that breaks the
disjunction semantics: a true DNF *is* a union, and a point satisfying two terms should read as strongly
positive, not be capped at one term's activation. So I keep the noisy-OR and accept that its accumulation
of precision errors is the exposure I will be watching on the random family.

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

Now reason about what this floor must do on each family, because that is the entire point of running it
as the first rung. The **monotone** family is the friendliest to the mechanism: all literals are
positive, so the three-way categorical mostly only has to choose between include-positive and skip, the
variable-mean gap is a clean signal (a relevant variable is on more often among positives), and the
warm-start mining enumerates its full `C(30,4)` positive candidates with nothing thinned. I expect the
differentiable DNF to do *well* here in raw-accuracy terms — though I have to remember my base-rate
arithmetic said monotone hands out ~0.72 for free, so "well" here is partly the constant-yes bias and
only partly recovered terms. The **sparse** family is also favorable in principle: only 12 of 60
variables are relevant, the mean-gap score should expose exactly those, and once the candidate mining
restricts to a 16-variable window the width-4 enumeration is cheap and accurate; I expect solid
accuracy, and here — the same balanced ~0.476 concept as random, so the same ~0.52 majority baseline —
every point above that ~0.52 is genuinely learned structure. I do note the sparse mining
runs over `n = 60`, so its per-example matching is the most expensive of the three and its fit time
should be the longest. The **random** family is where I am genuinely worried. It is non-monotone (every
literal can be either polarity), so the three-way softmax has a real three-way decision per variable,
the mean-gap signal is weaker because a mixed-polarity term correlates less cleanly with any single
variable's marginal, and the candidate enumeration over both polarities was the one that overflowed the
cap and got thinned to 73%. If the warm start mines imprecise terms there, the noisy-OR aggregation
compounds the error, and I can quantify how badly. On a true-negative test point, suppose each active
soft term still fires with some small residual probability `p`; the noisy-OR output is `1 − (1 − p)^{40}`
over the ~40 provisioned terms. At `p = 0.01` that union false-positive is `1 − 0.99^{40} = 0.33`; at
`p = 0.02` it is `1 − 0.98^{40} = 0.55`. So a mere one-to-two-percent per-term leakage, invisible clause
by clause, unions into a third-to-a-half of negatives flipped to positive. A noisy-OR is unforgiving in
exactly this way — it takes the union, so precision errors do not average out, they accumulate.

That gives me a sharp, falsifiable expectation to carry into the feedback. I expect this baseline to be
*strong on monotone and sparse and weak on random* — the geometric mean across the three will be dragged
down by whichever family the warm start serves worst, and I am betting that is the mixed-polarity random
family, both because its mean-gap signal is muddiest and because its candidate pool was the one thinned
below full enumeration. If the random number comes in well below the monotone and sparse numbers, the
diagnosis is the one above: the noisy-OR plus imperfect term mining over both polarities unioning
precision errors, not a failure of the gradient-descent refinement. And if even monotone and sparse fall
short of what a generic tabular learner would get — remembering that monotone's ~0.72 base rate means a
number in the high 0.80s is only ~0.15 of *earned* accuracy — then the inductive-bias bet has failed
outright, and the lesson is that a hand-shaped DNF hypothesis class, trained from uniform random
examples, does not actually beat a model that simply *splits on variables*, which is what a decision tree
does for free, one conjunction per root-to-leaf path. That is the comparison the next rungs exist to
make. For now the bar is just to see whether the exact-structure model clears the line, and where it
cracks.
