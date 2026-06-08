# Context: deciding provability, and what "effective computation" even means

## Research question

Hilbert and Ackermann, in their 1928 *Grundzüge der theoretischen Logik*, posed what they regarded as the central open problem of logic — the **Entscheidungsproblem** (decision problem): is there a *definite general method* that, applied to an arbitrary formula of the first-order (restricted) functional calculus, decides in a finite number of steps whether that formula is provable? By Gödel's 1930 completeness theorem, "provable" coincides with "valid in all interpretations," so the question is equivalently: is validity in first-order logic decidable by a uniform mechanical procedure?

The stakes were large. A positive answer would be the keystone of Hilbert's program: with such a method, every well-posed mathematical assertion expressible in the calculus could in principle be settled by turning a crank. Hilbert's conviction was that there is *no ignorabimus* in mathematics — every definite question has a definite answer reachable by definite means. The Entscheidungsproblem asked whether that conviction could be cashed out as an actual algorithm.

But the problem hides a prior difficulty that must be solved before it can even be attacked in the negative. The words "definite general method," "finite number of steps," "mechanical procedure," "calculable by finite means" had, at the time, **no mathematical definition**. As long as "method" is informal, one can perhaps *exhibit* a method, but one can never *prove that none exists* — a proof of impossibility quantifies over all possible methods, and you cannot quantify over an undefined class. So a solution would have to achieve two things at once: pin down, rigorously and convincingly, what a "method" / "effective computation" *is*, and then settle whether a method of that kind can decide provability.

## Background

**Hilbert's program and the decision problem.** By the late 1920s Hilbert had organized foundations around formalization: mathematics is a formal system; a proof is a finite manipulation of strings by fixed rules; the meta-questions are consistency (no contradiction derivable), completeness (every truth derivable), and decidability (a method to tell, of any formula, whether it is derivable). The Entscheidungsproblem is the decidability question for first-order logic. Particular sub-cases were known to be decidable (certain restricted quantifier prefixes), which fed the optimism that the general case would yield too.

**Gödel's completeness theorem (1930).** First-order validity equals first-order provability: a formula is true in every interpretation iff it has a formal proof. This makes the Entscheidungsproblem well-posed and gives a one-sided handle — the provable formulas are *effectively enumerable*. One can build a process that mechanically generates all proofs one after another (proofs are finite strings; enumerate them in order, check each is a valid proof). So if a formula is provable, this generate-and-check process will eventually find it. The asymmetry: if the formula is *not* provable, the process runs forever and never tells you so. Enumerability of the provable formulas is therefore not yet a decision procedure; a decision procedure would also have to give a definite "no" in finite time.

**Gödel's incompleteness theorems (1931).** Working inside *Principia Mathematica*-style arithmetic, Gödel arithmetized syntax — **Gödel-numbering**: assign each symbol, formula, and proof a natural number so that syntactic relations ("x is a proof of y") become arithmetic relations among their numbers. The system can then express statements about its own formulas. A fixed-point/diagonal construction yields a sentence that, under the numbering, asserts its own unprovability ("I am not provable"). The **first incompleteness theorem**: any consistent, effectively axiomatized system strong enough for elementary arithmetic is incomplete — there is a sentence A such that neither A nor ¬A is provable. The **second**: such a system cannot prove its own consistency. This shattered the completeness leg of Hilbert's program. Two ingredients here are load-bearing for what follows: (i) *self-reference by coding* — a system that manipulates finite strings can be made to manipulate descriptions of its own objects, because descriptions are themselves finite strings/numbers; (ii) the *diagonal* move that builds an object designed to disagree with a supposed exhaustive list at the place that names it.

**Cantor's diagonal argument (1891).** The oldest and sharpest form of (ii). Suppose the infinite binary sequences were enumerable, s₁, s₂, s₃, …. Define a new sequence s whose n-th digit is the *complement* of the n-th digit of sₙ (flip 0↔1 down the diagonal). Then s disagrees with s₁ at position 1, with s₂ at position 2, and in general with sₙ at position n — so s differs from *every* listed sequence and cannot be on the list, contradicting the assumption. Hence the binary sequences (equivalently the reals) are uncountable. The technique is general: from any purported exhaustive enumeration of objects-that-act-on-indices, build the object that contradicts the list at its own index.

**Church's λ-calculus and Herbrand–Gödel recursion (1934–1936).** Two attempts to *define* "effectively calculable" were already on the table. Church's λ-calculus represents functions as formal terms built by abstraction and application, with a conversion rule; a function is "λ-definable" if its values are computed by reduction of such terms. The Herbrand–Gödel "general recursive functions" define computability by closure of basic functions under composition, primitive recursion, and a minimization operator. Church had advanced the thesis that "effectively calculable" = λ-definable = general recursive, and on that basis had a candidate undecidability result. Both definitions are mathematically precise and (it would turn out) extensionally correct. Their weakness as *foundations for an impossibility proof* is one of conviction: each looks like a particular formal calculus chosen for technical convenience, and it is not self-evident that it exhausts everything a human following a finite procedure could compute. To prove "no method whatsoever can decide provability," one wants a definition of "method" whose claim to generality is *manifest* — argued from what computation *is*, not asserted by picking a calculus.

**The intuitive notion of an algorithm.** Underneath all of this is the everyday practice it must formalize: a human "computer" working a definite procedure with pencil and paper — writing and reading symbols on a page, following fixed rules, keeping a bounded amount in mind, never doing anything that requires unbounded insight at a single step. Any acceptable definition of "effective method" must (a) include everything such a human can do by rote, and (b) include nothing that smuggles in non-mechanical power. The convincing route to the definition is to *analyze that practice* until only finitely-describable, locally-determined operations remain.

## Baselines

These are the prior approaches and tools against which the work is measured and which it must improve upon or reuse.

- **Hilbert–Ackermann's framing of the Entscheidungsproblem (1928).** Sets the target precisely — decide provability/validity in the first-order functional calculus — and supplies the consistency of that calculus (Hilbert & Ackermann establish ¬A unprovable when A provable, so the calculus is consistent). Its gap: it presumes a positive solution is to be sought and gives no apparatus for a negative answer, because it has no definition of "method" over which to quantify.

- **Gödel's completeness theorem (1930) and the proof-enumeration procedure.** Reduces decidability to a question about a single effectively-enumerable set (the provable formulas). Its gap: enumeration gives only a semi-decision (eventual "yes," never a guaranteed "no"). It cannot, by itself, decide, and it cannot be turned into a decision procedure without independent grounds — grounds that turn out not to exist.

- **Gödel's incompleteness method (1931): arithmetization + diagonal self-reference.** The decisive technical template: encode a syntactic system's own objects as numbers/strings it can manipulate, then diagonalize to produce an object that defeats a global property. Its gap for the present purpose: it is a statement about *unprovability of particular sentences* inside one fixed formal system, not about the *non-existence of a decision method* ranging over all formulas. The undecidability claim is different in kind from incompleteness and needs its own vehicle — and that vehicle needs a precise, general notion of "method/machine."

- **Church's λ-calculus and general recursive functions (1934–36) as definitions of "effectively calculable."** Provide rigorous, closed mathematical classes and a candidate undecidability result for λ-conversion. Their gap: as definitions of *the intuitive notion*, their generality is asserted rather than argued from the nature of computation; they read as one formalism among possible others, so an impossibility proof built on them is only as convincing as the identification "this calculus = all effective methods," which is exactly what is in doubt.

- **Cantor's diagonalization (1891) as a tool.** A ready-made engine for refuting any claimed exhaustive enumeration of index-acting objects. Its gap here is that it proves *uncountability*, and the class to be analyzed is *countable* (finite descriptions can be listed); naïvely applied it yields a paradox rather than a theorem, so the technique must be redeployed — pointed not at "are these objects countable?" but at "is the property that sorts the list mechanically decidable?"

## Evaluation settings

The natural yardsticks for a claimed solution are not empirical benchmarks but the standards by which a foundational result of this kind is judged adequate.

- **The target problem.** Provability/validity in the first-order (restricted) functional calculus of Hilbert–Ackermann — the formulas over which the decision method must range. A sharper sub-target useful for stratifying the result: classes defined by their quantifier prefix (prenex formulas with a fixed pattern of ∀/∃), since decidability was already known for some restricted prefixes and a negative result is stronger if it pins down a small prefix class that is already undecidable.

- **The class of "effective methods" to be characterized.** Judged by two adequacy conditions: *completeness* with respect to intuitive computation (it must capture everything a human computer can do by rote — the human-computer analysis is the test) and *robustness* (agreement with the independently proposed formalisms, λ-definability and general recursiveness, as corroboration that the right class was caught).

- **Standard of proof.** A constructive impossibility argument: assume a deciding method exists and derive a contradiction by an explicit construction, with every step a finite, checkable manipulation — the same rigor demanded of Cantor's and Gödel's diagonal arguments. The argument must also keep the new undecidability result clearly distinct from Gödel's incompleteness (a different claim about a different object).

- **Reductions.** The currency of the result is the *reduction*: showing that a solution to the decision problem would yield a solution to a problem already shown unsolvable. The protocol is to chain reductions — from a core undecidable question about machines, to a question about a machine's printing behavior, to provability of a single constructed logical formula — each link a uniform effective transformation.

## Code framework

The scaffold starts from the informal practice of a human computer, the existing logical calculus, the enumeration idea, and the diagonal tool, leaving precise slots for the finite machine, its encoding, the universal simulator, the decider to be refuted, and the logical formula encoding a machine's run.

```
# ----- Available ingredients -----

# The informal practice we must capture: a human "computer" follows a finite
# procedure on paper — reads/writes symbols, keeps a bounded amount in mind,
# acts by fixed rules. No mathematical object yet stands for this.
class EffectiveProcedure:
    """The intuitive notion of an algorithm. Currently undefined as math.
    Must be pinned down by analyzing the human computer until only
    finitely-describable, locally-determined steps remain."""
    pass  # TODO: define the abstract machine that captures exactly this

# A finite, uniform code for "a procedure," so that procedures can be
# listed and fed to other procedures (cf. Gödel-numbering of syntax).
def encode(procedure):
    # TODO: a self-delimiting finite description from a fixed finite alphabet,
    #       injecting each procedure into the integers (=> the procedures are
    #       enumerable / countable)
    pass

def enumerate_procedures():
    # TODO: list all finite descriptions in order of their codes
    pass

# Closure under "run a description": is there one fixed procedure that, given
# the code of any procedure, does what that procedure does?
class Simulator:
    # TODO: a single procedure that reads any description and carries it out
    #       step by step
    pass

# The adversary whose existence we will test: a uniform decider for whether a
# candidate finite description is one that keeps producing figures forever.
def decides_property(candidate_description_number):
    # TODO: assume this exists, then diagonalize against it to a contradiction
    pass

# The diagonal engine already in hand (Cantor 1891): from a supposed
# exhaustive list of index-acting objects, build the object that disagrees
# with the list at its own index.
def diagonal(list_of_objects):
    # nth output := something that differs from list_of_objects[n] at index n
    # TODO: point this not at "are they countable?" but at the decider above
    pass

# The existing logical calculus (Hilbert–Ackermann first-order logic) and the
# proof-enumerator: generates all provable formulas one by one. Gödel
# completeness makes this an enumeration of valid formulas too, but still only
# gives eventual "yes" to provability, never a sure "no".
def enumerate_provable_formulas():
    # TODO: from a procedure, build ONE first-order formula whose provability
    #       tracks a fixed event in the procedure's run, so that deciding
    #       provability would decide that event
    pass
```
