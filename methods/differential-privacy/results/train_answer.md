The problem is releasing aggregate statistics from a database of sensitive per-person records while giving each individual a guarantee that survives an adversary with arbitrary auxiliary information and unbounded computation. Earlier defenses failed because they had no precise definition of success. Query restriction, auditing, k-anonymity, de-identification, input perturbation, and output perturbation all break under combination or linkage attacks: two large allowed sums that differ in one record reveal that record, refusals themselves leak information, quasi-identifiers re-identify almost everyone, and ad-hoc noise can be stripped or inverted. The Dalenius ideal, that access to the database teaches nothing about an individual that could not be learned without it, is also unachievable for any useful release, because an auxiliary fact can always be paired with an aggregate statistic to produce a personal disclosure, and the harm can even fall on people not in the database.

The only promise that can be honestly kept is about the effect of a single individual's record. Differential privacy formalizes this by requiring that the output distribution of a randomized mechanism changes very little whenever one person's row is changed. More precisely, a randomized mechanism M is epsilon-differentially private if for every pair of neighboring databases x and y that differ in exactly one row, and for every possible output event S, the probabilities satisfy Pr[M(x) in S] is at most e^epsilon times Pr[M(y) in S]. The parameter epsilon is a privacy budget: a smaller epsilon gives a stronger guarantee. This is a worst-case per-outcome ratio bound, not an average-case distance, so it controls how much any observation can shift an adversary's beliefs regardless of what the adversary already knew.

The Laplace mechanism achieves this guarantee for queries that release real-valued vectors. The idea is to add noise whose scale is calibrated to the maximum change one row can cause in the query output, called the ell-one sensitivity Delta f. For a counting query, Delta f is one. For a histogram over disjoint bins under the replace-one convention, Delta f is two, independent of the number of bins. Adding independent Laplace noise with scale b equal to Delta f divided by epsilon to each coordinate makes the log-density ratio between any two neighboring databases bounded by epsilon everywhere, because the Laplace log-density is piecewise linear and the reverse triangle inequality controls the shift. The released value is therefore epsilon-differentially private.

Three properties make this definition practical. First, composition: running mechanisms with privacy losses epsilon_1, epsilon_2, and so on, on the same database yields a mechanism that is the sum of those losses differentially private, so epsilon behaves like a budget spent across queries. Second, post-processing immunity: any computation on the released output without further access to the database cannot increase privacy loss. Third, group privacy: if two databases differ in k rows, the guarantee degrades gracefully to k times epsilon. These properties mean a curator can answer a sequence of queries by allocating portions of the total budget and be confident that downstream analysts cannot worsen the privacy loss.

Here is a concise implementation, including the histogram sensitivity constant alongside the mechanism itself.

```python
import numpy as np

def laplace_mechanism(database, f, delta_f, eps, rng):
    """ε-differentially private release of f(database).

    Adds i.i.d. Lap(Δf/ε) noise per output coordinate. For all neighboring
    x,y and all measurable events, the pointwise density ratio is bounded by
    exp(Δf/scale) = exp(ε), so the mechanism is ε-DP."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    if delta_f < 0:
        raise ValueError("delta_f must be nonnegative")
    true_value = np.atleast_1d(np.asarray(f(database), dtype=float))
    scale = delta_f / eps                                  # b = Δf / ε
    noise = rng.laplace(loc=0.0, scale=scale, size=true_value.shape)
    return true_value + noise

def l1_sensitivity_histogram_replace_one():
    """One individual moves at most two bins (−1, +1): Δf = 2, independent
    of the number of bins."""
    return 2.0

def compose_budget(eps_list):
    """Composition: ε-losses add."""
    return float(np.sum(eps_list))

def group_privacy(eps, k):
    """An ε-DP mechanism is (kε)-DP for a group of size k."""
    return k * eps

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    database = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    count = lambda db: float(db.sum())                     # Δf = 1
    eps = 0.1
    private_count = laplace_mechanism(database, count, delta_f=1.0, eps=eps, rng=rng)
    # private_count = true count + Lap(1/ε) = true count + Lap(10); releasing it is ε-DP.
    # Two such releases on the same database cost ε + ε = 0.2 total.
```
