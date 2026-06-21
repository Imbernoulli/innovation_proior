# Context: deciding provability, and what "effective computation" even means

## Research question

Hilbert and Ackermann, in their 1928 *Grundzüge der theoretischen Logik*, posed what they regarded as the central open problem of logic — the **Entscheidungsproblem** (decision problem): is there a *definite general method* that, applied to an arbitrary formula of the first-order (restricted) functional calculus, decides in a finite number of steps whether that formula is provable? By Gödel's 1930 completeness theorem, "provable" coincides with "valid in all interpretations," so the question is equivalently: is validity in first-order logic decidable by a uniform mechanical procedure?

The stakes were large. A positive answer would be the keystone of Hilbert's program: with such a method, every well-posed mathematical assertion expressible in the calculus could in principle be settled by turning a crank. Hilbert's conviction was that there is *no ignorabimus* in mathematics — every definite question has a definite answer reachable by definite means. The Entscheidungsproblem asked whether that conviction could be cashed out as an actual algorithm.

Before the question can be attacked, one difficulty must be addressed. The words "definite general method," "finite number of steps," "mechanical procedure," "calculable by finite means" had, at the time, **no mathematical definition**. As long as "method" is informal, one can perhaps *exhibit* a method, but one can never *prove that none exists* — a proof of impossibility quantifies over all possible methods, and you cannot quantify over an undefined class. So a solution would have to achieve two things at once: pin down, rigorously and convincingly, what a "method" / "effective computation" *is*, and then settle whether a method of that kind can decide provability.

## Background

**Hilbert's program and the decision problem.** By the late 1920s Hilbert had organized foundations around formalization: mathematics is a formal system; a proof is a finite manipulation of strings by fixed rules; the meta-questions are consistency (no contradiction derivable), completeness (every truth derivable), and decidability (a method to tell, of any formula, whether it is derivable). The Entscheidungsproblem is the decidability question for first-order logic. Particular sub-cases were known to be decidable (certain restricted quantifier prefixes), which fed the optimism that the general case would yield too.

**Gödel's completeness theorem (1930).** First-order validity equals first-order provability: a formula is true in every interpretation iff it has a formal proof. This makes the Entscheidungsproblem well-posed and gives a one-sided handle — the provable formulas are *effectively enumerable*. One can build a process that mechanically generates all proofs one after another (proofs are finite strings; enumerate them in order, check each is a valid proof). So if a formula is provable, this generate-and-check process will eventually find it. If the formula is *not* provable, the process runs forever. Enumerability of the provable formulas is therefore a semi-decision procedure; a full decision procedure would also give a definite "no" in finite time.

**Gödel's incompleteness theorems (1931).** Working inside *Principia Mathematica*-style arithmetic, Gödel arithmetized syntax — **Gödel-numbering**: assign each symbol, formula, and proof a natural number so that syntactic relations ("x is a proof of y") become arithmetic relations among their numbers. The system can then express statements about its own formulas. A fixed-point/diagonal construction yields a sentence that, under the numbering, asserts its own unprovability ("I am not provable"). The **first incompleteness theorem**: any consistent, effectively axiomatized system strong enough for elementary arithmetic is incomplete — there is a sentence A such that neither A nor ¬A is provable. The **second**: such a system cannot prove its own consistency. Two ingredients here are load-bearing: (i) *self-reference by coding* — a system that manipulates finite strings can be made to manipulate descriptions of its own objects, because descriptions are themselves finite strings/numbers; (ii) the *diagonal* move that builds an object designed to disagree with a supposed exhaustive list at the place that names it.

**Cantor's diagonal argument (1891).** The oldest and sharpest form of (ii). Suppose the infinite binary sequences were enumerable, s₁, s₂, s₃, …. Define a new sequence s whose n-th digit is the *complement* of the n-th digit of sₙ (flip 0↔1 down the diagonal). Then s disagrees with s₁ at position 1, with s₂ at position 2, and in general with sₙ at position n — so s differs from *every* listed sequence and cannot be on the list, contradicting the assumption. Hence the binary sequences (equivalently the reals) are uncountable. The technique is general: from any purported exhaustive enumeration of objects-that-act-on-indices, build the object that contradicts the list at its own index.

**Church's λ-calculus and Herbrand–Gödel recursion (1934–1936).** Two attempts to *define* "effectively calculable" were already on the table. Church's λ-calculus represents functions as formal terms built by abstraction and application, with a conversion rule; a function is "λ-definable" if its values are computed by reduction of such terms. The Herbrand–Gödel "general recursive functions" define computability by closure of basic functions under composition, primitive recursion, and a minimization operator. Church had advanced the thesis that "effectively calculable" = λ-definable = general recursive, and on that basis had a candidate undecidability result for λ-conversion. Both definitions are mathematically precise closed formal classes.

**The intuitive notion of an algorithm.** Underneath all of this is the everyday practice it must formalize: a human "computer" working a definite procedure with pencil and paper — writing and reading symbols on a page, following fixed rules, keeping a bounded amount in mind, never doing anything that requires unbounded insight at a single step. Any acceptable definition of "effective method" must (a) include everything such a human can do by rote, and (b) include nothing that smuggles in non-mechanical power.

## Baselines

These are the prior approaches and tools against which the work is measured and which it must improve upon or reuse.

- **Hilbert–Ackermann's framing of the Entscheidungsproblem (1928).** Sets the target precisely — decide provability/validity in the first-order functional calculus — and supplies the consistency of that calculus (Hilbert & Ackermann establish ¬A unprovable when A provable, so the calculus is consistent).

- **Gödel's completeness theorem (1930) and the proof-enumeration procedure.** Reduces decidability to a question about a single effectively-enumerable set (the provable formulas). Enumeration generates all proofs one by one; each provable formula is eventually reached.

- **Gödel's incompleteness method (1931): arithmetization + diagonal self-reference.** The decisive technical template: encode a syntactic system's own objects as numbers/strings it can manipulate, then diagonalize to produce an object that defeats a global property. It is a statement about *unprovability of particular sentences* inside one fixed formal system.

- **Church's λ-calculus and general recursive functions (1934–36) as definitions of "effectively calculable."** Provide rigorous, closed mathematical classes and a candidate undecidability result for λ-conversion. Church's thesis identifies these classes with the intuitive notion of effective computation.

- **Cantor's diagonalization (1891) as a tool.** A ready-made engine for refuting any claimed exhaustive enumeration of index-acting objects.

## Evaluation settings

The natural yardsticks for a claimed solution are not empirical benchmarks but the standards by which a foundational result of this kind is judged adequate.

- **The target problem.** Provability/validity in the first-order (restricted) functional calculus of Hilbert–Ackermann — the formulas over which the decision method must range. A sharper sub-target useful for stratifying the result: classes defined by their quantifier prefix (prenex formulas with a fixed pattern of ∀/∃), since decidability was already known for some restricted prefixes and a negative result is stronger if it pins down a small prefix class that is already undecidable.

- **The class of "effective methods" to be characterized.** Judged by two adequacy conditions: *completeness* with respect to intuitive computation (it must capture everything a human computer can do by rote) and *robustness* (agreement with the independently proposed formalisms, λ-definability and general recursiveness, as corroboration that the right class was caught).

- **Standard of proof.** A constructive impossibility argument: assume a deciding method exists and derive a contradiction by an explicit construction, with every step a finite, checkable manipulation — the same rigor demanded of Cantor's and Gödel's diagonal arguments. The argument must also keep the new undecidability result clearly distinct from Gödel's incompleteness (a different claim about a different object).

- **Reductions.** The currency of a result of this kind is the *reduction*: showing that a solution to the decision problem would yield a solution to a problem already shown unsolvable, each link a uniform effective transformation.

## Code framework

The scaffold collects what is in hand: the informal practice of a human computer, the existing logical calculus, the enumeration idea, and the diagonal tool, with an empty slot for the work to be done.

```
# ----- Available ingredients -----

# The informal practice we must capture: a human "computer" follows a finite
# procedure on paper — reads/writes symbols, keeps a bounded amount in mind,
# acts by fixed rules. No mathematical object yet stands for this.
class EffectiveProcedure:
    """The intuitive notion of an algorithm. Currently undefined as math.
    The only concrete example to analyze is the human computer working
    by rote."""
    pass

# The diagonal engine already in hand (Cantor 1891): from a supposed
# exhaustive list of index-acting objects, build the object that disagrees
# with the list at its own index. As Cantor used it, it refutes a claimed
# exhaustive enumeration (proves uncountability).
def diagonal(list_of_objects):
    # nth output := something that differs from list_of_objects[n] at index n
    pass

# The existing logical calculus (Hilbert–Ackermann first-order logic) and the
# proof-enumerator: generates all provable formulas one by one. Gödel
# completeness makes this an enumeration of valid formulas too.
def enumerate_provable_formulas():
    pass

# ----- The work to be done -----

# TODO: from the available ingredients, settle whether provability in the
#       first-order functional calculus is decidable by an effective method.
def contribution():
    pass
```
