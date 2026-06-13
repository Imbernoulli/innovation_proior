# Context: the ground a theory of learning would stand on

## Research question

Computability theory exists because Turing supplied a *precise model* of "mechanical
calculation." Once you have that model, "what can be computed?" becomes a sharp,
falsifiable question — you can prove a function uncomputable, prove a problem
NP-complete, prove a separation. Learning has no such model. We say a child "learns"
to recognize a table, a doctor "learns" to diagnose, a system "learns" a concept — but
none of this is pinned to anything you could prove a theorem about. The phenomenon we
mean is *knowledge acquisition in the absence of explicit programming*: a program for
recognizing some concept gets installed in the learner by some route other than someone
writing the program out.

The question is whether that phenomenon admits a precise computational model the way
calculation did — a model that (i) lets us prove that *whole, characterizable classes*
of concepts can be acquired this way, (ii) covers concepts rich enough to be interesting
for general knowledge, and (iii) bounds the acquisition to a *feasible* (polynomial)
number of steps. The deeper hope is symmetry with computability: just as computability
theory marks the limits of what can be computed, an analogous theory should mark the
limits of what can be learned. Why it matters: most human skill — recognizing
objects, classifying inputs — is acquired without anyone being able to articulate the
algorithm, so if machines are to acquire such skills, and if we are to understand the
boundary between what must be programmed and what can be learned, we need the model.

## Background

**The propositional-knowledge tradition.** The successful knowledge-based systems of the
era — expert systems such as DENDRAL and MYCIN — represent essentially all their
knowledge in the propositional calculus; they use no logical notation beyond it. That is
how general knowledge has actually been captured in working systems, and it makes Boolean
functions of propositional variables the natural object to try to learn: a concept is a
predicate that is true or false of given data, and a recognizer for it is a Boolean
expression or circuit.

**Inductive inference / identification in the limit (Gold 1967).** The most rigorous
prior formal model of learning. A learner is fed an ever-growing presentation of a target
language — either *text* (an enumeration of the strings in the language, positive data
only) or an *informant* (strings labeled in/out) — and after each new datum outputs a
guessed grammar. The learner *identifies the language in the limit* if it makes only
finitely many wrong guesses and thereafter sticks with a correct grammar. Two structural
features dominate: convergence is *exact* (it must settle on a correct grammar, not an
approximation), and the learner *need not know when* it has converged — a counterexample
may arrive arbitrarily later. Gold's central negative result: any language family
containing an infinite ascending chain L₁ ⊂ L₂ ⊂ ⋯ with limit also in the family is not
identifiable from text; so the regular languages, the context-free languages, indeed any
"superfinite" class, are unlearnable from positive data alone. The model deals with
recursive functions and formal grammars, not Boolean functions.

**Statistical pattern recognition (Duda–Hart 1973).** A large body of work classifies
inputs using statistical tools, and it does have the ingredients of *distribution* and
*error rate*. But it does not address general knowledge representation as a question of
which *concept classes* admit a feasibly-deducible recognizer; there is no notion of
deducing a *program* in a polynomial number of steps, and no characterization of the
class of learnable concepts.

**AI "concept learning" (Michalski–Carbonell–Mitchell 1983; Barr–Feigenbaum handbook).**
Learning studied as a branch of AI: learning by example, by analogy, by being told, with
a great emphasis on the diversity of human learning methods and on heuristic "induction"
of a general rule from data. Rich and suggestive, but without provable guarantees and
without a characterization of what is learnable.

**The probabilistic toolkit.** The combinatorics needed to control random samples is in
hand. For a process repeated in independent Bernoulli trials, the multiplicative Chernoff
lower-tail bound bounds the chance of seeing far fewer successes than the mean: in m
trials each with success probability ≥ p, the probability of at most k < mp successes is
at most e^{-mp+k}(mp/k)^k (Erdős–Spencer, *Probabilistic Methods in Combinatorics*). The
elementary inequalities (1 + 1/x)^x < e, (1 − 1/x)^x < e^{-1}, and (1 − x) ≤ e^{-x} for
x > 0 are standard. These are the levers for turning "draw enough examples" into a hard
probability bound.

**Complexity as the arbiter of feasibility.** The Cobham–Edmonds identification of
"feasible" with "polynomial time," and Cook's theory of NP-completeness, are the backdrop
that lets a learning question be posed as a complexity question. One consequence that
will bite: deciding whether a partial (nontotal) assignment forces an unrestricted DNF
formula to be true is the tautology problem, which is co-NP-hard (Cook 1971) — so some
of the most natural learning targets carry an intractable membership test inside them.

**Cryptographic hardness as a source of negative results.** The construction of
pseudorandom functions from one-way functions (Goldreich–Goldwasser–Micali 1984) makes
it possible to argue that some easy-to-compute functions cannot be learned at all: a
cipher E_k that is secure against chosen-plaintext attack is, almost by definition, a
function whose recognizer cannot be deduced from polynomially many input/output pairs.

## Baselines

A new model would be measured against the prior formal accounts of learning, by how well
each meets the three desiderata (provable class-learnability / nontrivial concepts /
polynomial feasibility).

- **Gold's identification in the limit (1967).** *Idea:* converge in the limit to an
  exactly correct grammar from a growing presentation. *Gap:* it has no time bound at all
  (convergence is asymptotic, and unannounceable); it demands exact identification, and
  against a worst-case adversarial ordering of the data. Under those demands even simple
  infinite classes are unlearnable from positive data. It answers "identifiable
  eventually?" — not anything with a clock on it.

- **Statistical pattern classification (Duda–Hart 1973).** *Idea:* fit a classifier to
  data, measure error under a distribution. *Gap:* no account of which concept *classes*
  are learnable as a matter of computational complexity, no polynomial-time deduction of
  a symbolic program, no characterization theorem.

- **Heuristic AI concept learning (Michalski et al. 1983).** *Idea:* induce a general
  rule by example/analogy/instruction. *Gap:* informal; no guarantee that the induced
  rule is even approximately correct, and no boundary on what such methods can and cannot
  acquire.

- **Plain memorization of positive examples.** *Idea:* the most naive baseline — store
  the positive examples seen and answer by table lookup / their disjunction. *Gap:* an
  arbitrary set of polynomially many positive examples cannot be relied upon to pin down
  even a single monomial: the unseen mass is unconstrained, so this generalizes to
  nothing, for any worst-case target.

## Evaluation settings

The natural objects on which a learnability claim would be tested are *classes of Boolean
recognizers* over t propositional variables: conjunctions (single AND-terms / 1-CNF),
conjunctive normal form with a bounded number of literals per clause (k-CNF), disjunctive
normal form (DNF) and its monotone restriction, and read-once formulas where each
variable occurs once. The data model is a *learning protocol*: a routine EXAMPLES that
returns a vector v with F(v)=1, where v is drawn by a fixed but arbitrary distribution D
over the positive vectors (D is "nature's" distribution and is unknown to the learner);
and a membership ORACLE that, given a vector, reports whether it positively exemplifies
the concept. Vectors may be partial (entries in {0,1,*}), with a concept extended to a
partial vector by "true on every total completion." The yardstick for a deduction
procedure has three axes: the *number of EXAMPLES/ORACLE calls*, the *running time*, and
the *quality of the output hypothesis* — how good a recognizer it is, by whatever standard
makes a learnability claim both achievable and meaningful. The target metric is whether all
of these can be held polynomial in the size of the concept, the number of (relevant)
variables, and the relevant accuracy parameters.

## Code framework

The basic protocol has two environment-supplied routines and an empty deduction procedure.
The examples and oracle provide the only access to the hidden concept; the deduction
procedure is the part to be filled in.

```python
import random

# ---- the concept being learned lives behind these routines; the learner sees only them ----

def EXAMPLES():
    """Return a vector v (a partial assignment in {0,1,*}^t) with F(v)=1,
    drawn from nature's fixed but ARBITRARY, UNKNOWN distribution D over positive vectors."""
    raise NotImplementedError  # supplied by the environment

def ORACLE(v):
    """Membership query: 1 iff v positively exemplifies the target concept, else 0."""
    raise NotImplementedError  # supplied by the environment

# ---- deduction procedure ----

def deduce(t):
    """Run the protocol (calling EXAMPLES and, if useful, ORACLE) and return a hypothesis g,
    a Boolean expression over the t variables. What standard g must meet, how many calls and
    how much time the procedure may spend, and how that standard is even stated, are open.
    """
    # TODO: deduction procedure.
    pass
```
