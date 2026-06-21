When faced with an infinite first-order theory, the natural impulse is to look for an infinite construction that satisfies every sentence at once. That impulse is usually a dead end: later requirements can undo earlier choices, and a sequence of finite steps may never converge to a single coherent structure. The same obstacle appears in every naive approach. Checking each finite subset individually only tells us that local models exist; those models may live on unrelated domains with incompatible interpretations of symbols. Syntactic consistency is reassuring but invisible until we actually produce a model. Even the completeness theorem, while closely related, does not by itself explain why semantic failure must already appear inside a finite fragment. What we need is a principle that converts the local hypothesis—every finite part is satisfiable—into a global model, without ever treating the infinite theory as an infinite checklist.

The right insight is that first-order proofs and first-order sentences are finite objects. If an infinite theory were contradictory, the contradiction would already follow from finitely many of its sentences. So finite satisfiability rules out contradiction, and the gap between syntactic consistency and a real model can be closed by a canonical construction. This is the Compactness Theorem for First-Order Logic: a set of first-order sentences has a model if and only if every finite subset of it has a model. The theorem is not merely a statement about infinite theories; it is the reason that infinite satisfiability in first-order logic is always controlled by finite fragments.

I will present the theorem and give the standard proof through maximal Henkin theories. The idea is to expand the language with fresh constants that serve as explicit witnesses for existential sentences, extend the theory to a maximal consistent set, and then build a model whose elements are equivalence classes of closed terms. The proof below also sketches the ultraproduct route, which achieves the same conclusion by combining all finite-fragment models modulo a carefully chosen ultrafilter.

The theorem applies to any first-order language and any set T of sentences in that language. If T has a model, then trivially every finite subset has the same model. The nontrivial direction assumes that every finite subset of T is satisfiable and concludes that T itself is satisfiable. The argument runs in three stages. First, finite satisfiability implies syntactic consistency, because any formal proof of a contradiction from T uses only finitely many premises and would therefore already make some finite subset inconsistent. Second, the consistent theory T is extended to a Henkin theory: for every formula of the form exists x phi(x), a fresh constant c_phi is added along with the witness sentence exists x phi(x) -> phi(c_phi). These additions preserve consistency because any contradiction would occur at a finite stage with fresh constants that can be replaced by variables, contradicting the derivability of the existential formula. Third, the Henkin theory is extended to a maximal consistent theory H in which, for every sentence sigma, exactly one of sigma and not sigma appears. The term model M_H is built from equivalence classes of closed terms under the equality theory of H, with relations and functions interpreted according to membership in H. A truth lemma, proved by induction on formulas, shows that M_H satisfies a sentence exactly when that sentence belongs to H. Since T is contained in H, M_H is a model of T.

The ultraproduct proof reaches the same goal by semantic means. Index the finite subsets of T by I, choose a model M_Delta for each Delta, and define A_sigma to be the set of finite subsets that contain sigma. The family {A_sigma} has the finite intersection property, because any finite collection of sentences is contained in some finite subset of T. Extend this family to an ultrafilter U on I and form the ultraproduct of the M_Delta modulo U. Los's theorem says that a sentence holds in the ultraproduct precisely when the set of indices at which it holds belongs to U. Since A_sigma is in U and every Delta in A_sigma satisfies sigma, each sentence of T holds in the ultraproduct. Thus the ultraproduct is another model of T. Both proofs exploit the same finitary fact: an obstruction to the whole theory would already be an obstruction to some finite part.

One of the most striking consequences is the existence of nonstandard models. Add a new constant c to first-order arithmetic and include the infinitely many sentences c > 0, c > 1, c > 2, and so on. Every finite subset is satisfiable in the standard natural numbers by interpreting c as a sufficiently large number. By compactness, the entire theory has a model, and in that model c exceeds every standard numeral. The model therefore cannot be the standard natural numbers, even though it satisfies the same first-order arithmetic axioms. This is the characteristic signature of compactness: infinite collections of finite-looking requirements can force structures that lie outside the intended picture.

The following Python script is not a theorem prover; it is a small executable demonstration of the finitary bookkeeping that underlies compactness. It takes a simple first-order theory expressed as constraints on constants, checks that every finite subset is satisfiable by bounded search, and then constructs a global valuation using the finite-intersection pattern that an ultrafilter would formalize. The example theory says that a constant c must be larger than each standard numeral up to some bound, and the script verifies finite satisfiability before producing a model that treats c as nonstandard.

```python
from itertools import combinations
from typing import Set, Tuple, Dict

def finite_subset_satisfiable(theory: Set[str], max_val: int = 100) -> bool:
    """
    Check whether every finite subset of a simple numeral theory is satisfiable.
    Each sentence is of the form 'c > n' for a non-negative integer n.
    A finite subset {c > n_1, ..., c > n_k} is satisfied by any value c > max(n_i).
    """
    if not theory:
        return True
    max_required = -1
    for sentence in theory:
        if not sentence.startswith("c > "):
            raise ValueError(f"Unsupported sentence: {sentence}")
        n = int(sentence.split(">")[1].strip())
        max_required = max(max_required, n)
    # Any value strictly larger than max_required works.
    return max_required + 1 <= max_val

def compactness_model(theory: Set[str]) -> Dict[str, int]:
    """
    Build a global model for a finitely satisfiable theory of 'c > n' sentences.
    If the theory is infinite, choose c to exceed all required numerals.
    """
    if not finite_subset_satisfiable(theory):
        raise ValueError("Theory is not finitely satisfiable")
    required = []
    for sentence in theory:
        n = int(sentence.split(">")[1].strip())
        required.append(n)
    if required:
        # A nonstandard-looking value: larger than every finite demand in the theory.
        value = max(required) + 1
    else:
        value = 0
    return {"c": value}

def check_all_finite_subsets(theory: Set[str], bound: int) -> Tuple[bool, Set[str]]:
    """
    Verify that every finite subset of size at most `bound` is satisfiable.
    Returns (True, empty_set) if all pass, otherwise (False, offending_subset).
    """
    sentences = list(theory)
    for r in range(1, min(bound, len(sentences)) + 1):
        for subset_indices in combinations(range(len(sentences)), r):
            subset = {sentences[i] for i in subset_indices}
            max_n = max(int(s.split(">")[1].strip()) for s in subset)
            if max_n + 1 > 1000:  # arbitrary large ceiling for demonstration
                return False, subset
    return True, set()

if __name__ == "__main__":
    # Example theory: c exceeds every standard numeral.
    # In a real first-order setting this is an infinite set of sentences.
    example_theory = {f"c > {n}" for n in range(20)}

    ok, offender = check_all_finite_subsets(example_theory, bound=5)
    print("All checked finite subsets satisfiable:", ok)
    if not ok:
        print("Offending subset:", offender)

    model = compactness_model(example_theory)
    print("Compactness model:", model)
    # In the model, c is max_required + 1, larger than every finite numeral
    # appearing in the checked subsets; the infinite theory forces a nonstandard value.
```
