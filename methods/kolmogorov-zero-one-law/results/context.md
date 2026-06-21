# Context: why certain limiting probabilities refuse to land strictly between 0 and 1

## Research question

Several important probabilities, each computed by its own argument in the first decades of measure-theoretic probability, all came out pinned to the extremes — exactly `0` or exactly `1`, never anything in the open interval. The recurring examples:

- A real number drawn uniformly from `[0,1]` has, in its binary expansion, a long-run frequency of `1`'s equal to `1/2` — with probability `1`. Equivalently, in an unending sequence of independent fair coin tosses, the fraction of heads converges to `1/2` almost surely. The set of "normal" numbers has Lebesgue measure `1`; its complement has measure `0`.
- For a series `∑ₙ Xₙ` of independent random terms, the probability that the series converges is `0` or `1` — never, say, `0.3`.
- The event behind the strong law of large numbers, `{Sₙ/n → c}`, has probability `0` or `1`.

Each was settled by a special argument. The problem: **is there a single structural reason these probabilities can never land strictly inside `(0,1)`?** What feature do all these events share, and can it be turned into one theorem that subsumes the case-by-case results? A solution must identify the common feature in purely measure-theoretic terms and prove the `0`-or-`1` dichotomy from it, for *any* event with that feature.

Why it matters: such a theorem would convert a recurring nuisance ("is this limiting probability `0` or `1`, and which?") into a free structural fact ("it is automatically `0` or `1`; only *which* remains"). It would also say something striking: certain asymptotic properties of an independent stream are not random at all.

## Background

**Borel's denumerable probability (Borel 1909, *Les probabilités dénombrables et leurs applications arithmétiques*).** Borel was first to treat probability over an *infinite* sequence of trials as a subject in its own right and connect it to Lebesgue measure on `[0,1]`. His strong law of large numbers: the binary digits of a uniform point, viewed as independent fair bits, have empirical frequency converging to `1/2` almost surely; in arithmetic terms, almost every real number is normal. This is the prototype of a probability forced to `1`. Crucially Borel was *reluctant to assume countable additivity* — he saw no logical absurdity in countably many zero-probabilities summing to a positive probability, and preferred arguments that exhibit probabilities as limits of finite-trial probabilities. He produced the first `0`-or-`1` phenomenon, but his avoidance of countable additivity left the infinitary picture murky and supplied no general mechanism.

**The Borel–Cantelli lemmas (Borel 1909, Cantelli 1916/17).** For events `A₁, A₂, …`: if `∑ₙ P(Aₙ) < ∞` then `P(Aₙ i.o.) = 0` (no independence needed); and if `∑ₙ P(Aₙ) = ∞` *and the `Aₙ` are independent*, then `P(Aₙ i.o.) = 1`. So under independence the probability that infinitely many `Aₙ` occur is already forced to `0` or `1` according to whether a series converges or diverges.

**Khinchin–Kolmogorov on convergence of series (1925); ad hoc limit definitions (Kolmogorov 1928).** By the mid-1920s the convergence of a sum of independent random variables was being studied directly: a series of discrete independent terms converges with probability `1` when the series of means and the series of variances both converge. In the 1928 treatment, when passing to the limit, the probability that `∑ Xₙ` converges was *defined* by an explicit iterated-limit formula built from finite-trial probabilities, rather than read off a single measure — a sign that the infinitary step had no settled footing.

**P. Lévy's `0`-`1` cases (Lévy 1931, *Sur un théorème de M. Khintchine*, Theorem II).** Lévy isolated further situations — among them the convergence of sums of independent variables — in which a probability can only be `0` or `1`, again case by case. By 1931 the phenomenon was clearly recurrent and clearly tied to independence and to *asymptotic* events, but it remained a list of special results.

**The measure-theoretic foundation now in place.** Probability is a non-negative, **completely (countably) additive** set function `P` with `P(E) = 1` on a field of events, satisfying a continuity axiom: for a decreasing sequence `A₁ ⊇ A₂ ⊇ ⋯` with empty intersection, `P(Aₙ) → 0`. From this `P` is completely additive. This is the very assumption Borel resisted; adopting it is what lets a probability be transported from finitely-described events to limiting events.

**Conditional probability and independence as set-function statements.** For `P(A) > 0` the conditional probability is `P_A(B) = P(A∩B)/P(A)`, and with `A` fixed `P_A(·)` is itself a probability measure (same axioms; `P_A(A) = 1`). The multiplication theorem reads `P(A∩B) = P(A)·P_A(B) = P(B)·P_B(A)`, giving the Bayes form `P_B(A) = P(A)P_A(B)/P(B)`. Two events are independent when `P(A∩B) = P(A)P(B)`, equivalently `P_A(B) = P(B)`. Mutual independence of a family means the product rule for every finite subcollection. (Pairwise independence is strictly weaker than mutual independence.) For a sequence of variables, the natural independence statement is that the conditional probability of an event given the first `n` of them equals its unconditional probability, for every `n`. Dependence is the operative contrast: in a Markov chain the conditional given the whole past collapses to the conditional given only the present, `P_{𝔄⁽¹⁾…𝔄⁽ⁿ⁻¹⁾}(A⁽ⁿ⁾) = P_{𝔄⁽ⁿ⁻¹⁾}(A⁽ⁿ⁾)`, so successive trials are *not* independent.

**Fields, Borel fields, and the extension theorem.** A *field* of sets is closed under finite union, intersection, complement. A *Borel field* (σ-field) is also closed under countable unions. Every field `𝔉` sits inside a smallest Borel field `B𝔉`. The **extension theorem**: a non-negative completely additive set function on a field `𝔉` extends to all of `B𝔉` — preserving non-negativity and complete additivity — and **in exactly one way**, the uniqueness following from the minimal property of `B𝔉`. (Existence is via Carathéodory outer measure: cover `A` by countably many sets of `𝔉`, take the infimum of `∑ P(Aₖ)`.) The uniqueness is the load-bearing half: a completely additive set function is pinned down on the whole Borel field by its values on the generating field.

**Cylinder events for a sequence of variables.** When the elementary outcome is an infinite sequence of coordinates `(x₁, x₂, …)`, the events "describable through finitely many coordinates" — sets `{(x_{k₁},…,x_{kₙ}) ∈ A'}` for a Borel set `A'` — form a field, the *Borel cylinder field*. Its smallest enclosing Borel field is where every event built from the whole sequence lives. A function of the whole sequence is measurable with respect to this Borel field precisely when it is a **Baire function** of the coordinates — a function obtainable from polynomials by iterated passages to the limit.

## Baselines

- **Borel's strong law / normal numbers (Borel 1909).** Core idea: the binary digits of a uniform point are independent fair bits; their running average converges to `1/2` a.s. Math: `{ (1/n)∑_{k≤n} d_k → 1/2 }` has measure `1`.

- **Borel–Cantelli lemmas (Borel 1909, Cantelli 1916/17).** Core idea: control `P(Aₙ i.o.)` by `∑ P(Aₙ)`. Math: `∑ P(Aₙ) < ∞ ⇒ P(limsup Aₙ) = 0`; `∑ P(Aₙ) = ∞` with independence `⇒ P(limsup Aₙ) = 1`.

- **Lévy's special `0`-`1` results (Lévy 1931).** Core idea: in several settings tied to sums of independent variables, the relevant probability can only be `0` or `1`.

- **The finitely-additive viewpoint (Borel's reluctance).** Core idea: exhibit probability as a limit of finite-trial probabilities without committing to countable additivity.

## Evaluation settings

This is a theorem; the yardstick is logical, and the test cases are the phenomena it must subsume.

- **Canonical events to cover.** Convergence of `∑ₙ Xₙ` for independent `Xₙ`; the digit-frequency / normal-number event for i.i.d. digits; the strong-law event `{Sₙ/n → c}` and the value `limsup Sₙ/n`; the radius of convergence of `∑ₙ Xₙ zⁿ`; `{Aₙ i.o.}` for independent `Aₙ`. A correct general theorem returns "`0` or `1`" for every one of these as an instance, with no separate argument per case.
- **Negative controls the statement must respect.** The dichotomy must *not* hold without independence: for a Markov chain or other dependent sequence, an asymptotic event (e.g. eventual absorption in one state rather than another) can have probability strictly inside `(0,1)`. It must also not constrain an event that genuinely depends on a particular early coordinate (e.g. `{X₁ = 1}` for a fair bit has probability `1/2`). And it must fail without countable additivity, since the transport step relies on it.
- **Standard of success.** A single hypothesis on an event, satisfied by all canonical examples; a complete proof from the axioms (non-negativity, normalization, countable additivity, the extension theorem, the definition of independence) that any event meeting the hypothesis has probability in `{0,1}`; checkable by hand on small cases (a single independent bit, a finite product) and against the negative controls.

## Code framework

This is a pure-structure theorem; there is no algorithm to run. The "scaffold" is the logical skeleton the proof will fill — the objects already available, and the empty slots the contribution occupies. In pseudocode-as-mathematics:

```
# Already available (pre-method primitives):
#   P : completely-additive probability set function on a Borel field, P(E)=1
#   P_A(B) = P(A∩B)/P(A)            # conditional probability, defined when P(A)>0
#   independent(B, C):  P(B∩C) == P(B)*P(C)
#   field 𝔉  ⊆  Borel-field B𝔉      # B𝔉 = smallest σ-field containing 𝔉
#   extension theorem: a completely-additive set function on 𝔉
#       extends to B𝔉 UNIQUELY      # values on 𝔉 pin it down on all of B𝔉
#   coordinates x₁, x₂, …  ;  finite-dimensional (cylinder) events form a field 𝔉
#   Baire functions of (x₁,x₂,…) = the B𝔉-measurable functions

# The class of events the theorem concerns — TO BE DEFINED:
def the_event_class(A):
    # TODO: the property of an event we will single out
    pass

# The conclusion to be established for events in that class — TO BE DISCOVERED:
def main_claim(A):
    assert the_event_class(A)
    # TODO: the argument we will give
    return  # P(A) ∈ ?

# A consequence for measurable quantities built from that class:
def consequence_for_quantities(Y):
    # TODO: what follows for a random variable measurable w.r.t. that class
    pass
```

The three `# TODO` slots are exactly what the proof supplies: the property singled out, the conclusion forced for events with it, and what that conclusion implies for random variables built from such events.
