The hidden function is a DNF, and I get to choose the learner, so the first move I want to make is the
one that puts the answer's *shape* into the model itself. The target is literally an OR of ANDs of
literals; if I build a network whose hypothesis class is relaxed DNF — soft conjunctions OR'd together —
then I am not asking gradient descent to discover that Boolean functions exist, I am only asking it to
find *which* conjunctions. That inductive-bias argument is strong enough that it should be the floor I
measure everything else against: if encoding the exact target structure does not win, that itself is the
finding. So step 1 is the differentiable-DNF learner, and the question I have to answer before I can
even fill `build_model` is how to make "OR of ANDs of literals" differentiable without a single bias
threshold to tune.

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
the `n` inputs is wrong twice over — a softmax picks essentially one thing, so selecting `w` literals
needs `w` softmaxes and a commitment to clause width up front, and a softmax forced to concentrate over
a long input vector converges painfully when `n` is large. The right primitive is a per-variable,
*independent* include/exclude decision, not a competition among variables. So attach to each input `i`
of clause `j` a membership and fold it into the product through a factor that is the identity (= 1) when
the variable is excluded and equals `x_i` when it is included: `F = 1 − m_i(1 − x_i)`. Check the table —
`m_i = 0` gives 1 for either `x_i`; `m_i = 1` gives `x_i`. So `O_conj = Π_i (1 − m_i(1 − x_i))`, and the
clause can be read straight off the memberships. Here, because the target may use *negative* literals
too (the random and monotone families differ exactly on this), I want a three-way categorical per
variable — include-positive, include-negative, skip — via a softmax over three logits, so a term can use
either polarity but can never select `x_i` and `¬x_i` at once. The conjunction becomes a product over
`pos·x + neg·(1−x) + skip`, with the three probabilities summing to 1.

Three implementation facts are not optional, they decide whether this trains at all, and they are
exactly what I encode in the scaffold's `forward`. First, **log-domain products.** A width-`w` clause is
a product of `n` factors each ≤ 1; over `n = 60` those products underflow and the gradient through them
vanishes. So I compute `exp(Σ log(factor))` with the factors clamped to `[1e-6, 1]` — valid because
every factor lies in `[0,1]`. Second, **sparse initialization.** If most memberships start near 1, every
clause is a product of many sub-1 terms, the output is crushed toward 0, and the gradient dies before
training begins. So I initialize the skip logit high (≈ 4) and the include logits low (≈ −4), so clauses
start near the constant-1 function and only a few literals switch on as learning proceeds; I also
initialize the per-term gate logits low (≈ −3) so terms start mostly off. Third, **per-term gates and
sparsity penalties.** I over-provision terms (`n_terms = max(4·num_terms, 32)`), put a `σ(g_j)` gate on
each so the noisy-OR can switch a term off, and add a one-sided literal-width penalty `(usage − w)_+^2`
plus a mean-gate penalty, pushing toward short clauses and few active terms — a crisp DNF rather than a
mush of half-on memberships.

There is one more piece this task's substrate lets me add that a generic differentiable-DNF learner
would not: a **data-driven warm start.** Pure gradient descent on a noisy-OR of products is slow to
discover *which* width-`w` conjunctions matter, and I have 20000 labelled examples sitting right there.
So before training I mine candidate terms directly from the data: score each variable by the gap between
its positive-class and negative-class means (`|E[x_i | y=1] − E[x_i | y=0]|`), keep the top variables
(more for the monotone family, a tight window around the junta size for the sparse family), enumerate
width-`w` combinations and polarity patterns over them, and score each candidate conjunction by its
precision plus a recall term on the training labels, discarding any that fire on fewer than a handful of
examples. The top-scoring distinct terms initialize the memberships at saturated logits, so the network
*starts* near a plausible DNF and gradient descent only has to refine it. This is the harness adapting
the architecture to its own research question — it is not in the canonical differentiable-DNF method,
and it is the single biggest reason this baseline could beat a from-scratch soft-DNF. I should keep that
in mind, because if the warm start mines the *wrong* terms, the network inherits the mistake.

So why do I believe the convergence story at all? Differentiate one membership: `∂O_conj/∂m_i ∝
(x_i − 1)`, since the product over the other factors is non-negative. To *raise* a membership toward 1 I
need a counterexample — a point where `x_i = 0` yet the clause is supposed to fire — telling me `x_i`
should not be a required literal; to keep it I need the clause to hold whenever the relevant variables
are on. The memberships are driven by counterexamples, and with 20000 uniform examples each width-4
conjunction's relevant counterexamples are present in abundance, so a single layer should converge if
the warm start put it in the right neighborhood. I train with AdamW + BCE on the logits the noisy-OR
emits, 30 epochs, batch 512, the two sparsity penalties at small weight, and threshold at logit 0 for
the 0/1 prediction. The full scaffold module — `NeuralDNF` with the three-way literal, the noisy-OR, the
term mining, and the penalised training loop — is in the answer.

Now reason about what this floor must do on each family, because that is the entire point of running it
as the first rung. The **monotone** family is the friendliest: all literals are positive, so the
three-way categorical mostly only has to choose between include-positive and skip, the variable-mean
gap is a clean signal (a relevant variable is on more often among positives), and the warm-start mining
should recover most of the 20 terms. I expect the differentiable DNF to do *well* here — this is the
regime its inductive bias was built for. The **sparse** family is also favorable in principle: only 12
of 60 variables are relevant, the mean-gap score should expose exactly those, and once the candidate
mining restricts to a small variable window the width-4 enumeration is cheap and accurate; I expect
solid accuracy. The **random** family is where I am genuinely worried. It is non-monotone (every literal
can be either polarity), so the three-way softmax has a real three-way decision per variable, the
mean-gap signal is weaker because a mixed-polarity term correlates less cleanly with any single
variable's marginal, and the candidate enumeration over both polarities is far larger. If the warm start
mines imprecise terms there, the noisy-OR aggregation compounds the error: each slightly-wrong soft
conjunction contributes a little false-positive mass, and OR-ing 32+ of them together drives the
prediction toward 1 on inputs that should be 0. A noisy-OR is unforgiving in exactly this way — it
takes the union, so precision errors accumulate.

That gives me a sharp, falsifiable expectation to carry into the feedback. I expect this baseline to be
*strong on monotone and sparse and weak on random* — the geometric mean across the three will be dragged
down by whichever family the warm start serves worst, and I am betting that is the mixed-polarity random
family. If the random number comes in well below the monotone and sparse numbers, the diagnosis is the
one above: the noisy-OR plus imperfect term mining over both polarities, not a failure of the
gradient-descent refinement. And if even monotone and sparse fall short of what a generic tabular
learner would get, then the inductive-bias bet has failed outright and the lesson is that a hand-shaped
DNF hypothesis class, trained from uniform random examples, does not actually beat a model that simply
*splits on variables* — which is what a decision tree does for free, one conjunction per root-to-leaf
path. That is the comparison the next rungs exist to make. For now the bar is just to see whether the
exact-structure model clears the line, and where it cracks.
