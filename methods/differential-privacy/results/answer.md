# Differential privacy

## The problem it solves

Release useful aggregate statistics from a database of sensitive per-person records while giving each individual a rigorous, provable privacy guarantee that holds against an adversary with **arbitrary auxiliary information** and unbounded computation. Prior defenses (query restriction, k-anonymity, ad-hoc perturbation) had no definition behind them and fell to combination/linkage/reconstruction attacks. The aspirational definition — Dalenius's "you learn nothing about an individual you couldn't learn without the database," the database analogue of semantic security — is **provably impossible** when the release has any utility: the Terry-Gross-height argument shows an auxiliary fact can always turn a useful aggregate into a personal disclosure, and the breach can hit individuals not even in the database.

## The key idea

Make privacy a property of the **mechanism**, not of any output. Promise the one thing the individual controls — changing one person's record should leave the mechanism's output distribution almost unchanged. Then no inference an adversary can draw is materially affected by that one row, *regardless of side information*. Achieve it by adding noise **calibrated to sensitivity** — the maximum amount one person can change the query — and the definition composes (privacy losses add), survives post-processing, and degrades gracefully for groups.

## The definitions, stated cleanly

**Neighboring databases.** Under the fixed-size replace-one convention, x, y are neighbors if their row representations have Hamming distance 1: exactly one individual's record is changed. In a histogram representation, the same replacement can move one count down and another up, so ‖x − y‖₁ ≤ 2. Under the add/drop convention the histogram bound is ≤ 1 and the related constants change accordingly.

**ε-differential privacy.** A randomized mechanism M is ε-differentially private if for all neighboring databases x, y and all measurable S ⊆ Range(M),

  Pr[M(x) ∈ S] ≤ e^ε · Pr[M(y) ∈ S].

Equivalently, in discrete mass or continuous density form, the privacy loss is bounded wherever the ratio is defined: |ln(p_x(t)/p_y(t))| ≤ ε. (The general (ε,δ) form relaxes this to Pr[M(x)∈S] ≤ e^ε Pr[M(y)∈S] + δ; δ = 0 is pure ε-DP.) The ratio (not total-variation) bound is essential: total variation can be O(1/n) while a transcript leaks an individual outright (e.g. output a random (i, xᵢ)). Via Bayes, the bound means observing M's output moves any adversary's prior log-odds on x-vs-y by at most ε, whatever the adversary already knew. ε is a policy parameter; it must be non-negligible for any utility to be possible.

**ℓ₁-sensitivity.** For f : ℕ^|X| → ℝ^k,

  Δf = max over neighbors x, y of ‖f(x) − f(y)‖₁.

A property of f alone — not of the database, not chosen by policy. Counting query under replace-one: Δf = 1. Histogram over disjoint bins under replace-one: Δf = 2, independent of the number of bins.

## The Laplace mechanism and its privacy proof

**Mechanism.** Given f : ℕ^|X| → ℝ^k, release

  M(x) = f(x) + (Y₁, …, Y_k),  Yᵢ i.i.d. ∼ Lap(Δf/ε),

where Lap(b) has density (1/2b)·exp(−|y|/b).

**Theorem.** The Laplace mechanism is ε-differentially private.

*Proof.* Let x, y be neighbors, b = Δf/ε, and z ∈ ℝ^k arbitrary. Comparing densities coordinatewise,

  p_x(z)/p_y(z) = ∏ᵢ exp(−|f(x)ᵢ − zᵢ|/b) / exp(−|f(y)ᵢ − zᵢ|/b)
        = ∏ᵢ exp( (|f(y)ᵢ − zᵢ| − |f(x)ᵢ − zᵢ|)/b )
        ≤ ∏ᵢ exp( |f(x)ᵢ − f(y)ᵢ|/b )        (reverse triangle inequality)
        = exp( ‖f(x) − f(y)‖₁ / b )
        ≤ exp( Δf / b ) = exp( Δf / (Δf/ε) ) = exp(ε).

The lower bound p_x(z)/p_y(z) ≥ e^{−ε} follows by swapping x and y. The pointwise upper bound integrates over any measurable S, giving Pr[M(x)∈S] ≤ e^ε Pr[M(y)∈S]. Hence the mechanism is ε-DP. ∎

Laplace noise fits the per-outcome ratio bound exactly: its log-density −|y|/b is piecewise linear, so the log-ratio between shifted centers is *bounded* everywhere by |shift|/b; Gaussian noise (quadratic log-density) gives a log-ratio that is linear in the outcome and unbounded in the tails, so finite-variance Gaussian noise cannot satisfy pure ε-DP for nonzero sensitivity. It can be used only with a δ tail slack. The scale b = Δf/ε depends only on Δf and ε — independent of database size n and output dimension k.

## Composition (the ε's add)

**Theorem.** If M₁ is ε₁-DP and M₂ is ε₂-DP (independent randomness), the combined release M(x) = (M₁(x), M₂(x)) is (ε₁+ε₂)-DP.

*Proof.* For neighbors x, y and any (r₁, r₂), independence factorizes the joint density:

  Pr[M(x)=(r₁,r₂)] / Pr[M(y)=(r₁,r₂)]
   = (Pr[M₁(x)=r₁]/Pr[M₁(y)=r₁]) · (Pr[M₂(x)=r₂]/Pr[M₂(y)=r₂]) ≤ e^{ε₁}·e^{ε₂} = e^{ε₁+ε₂},

with the symmetric lower bound. ∎ Iterating, k mechanisms with losses ε₁,…,ε_k compose to (Σᵢ εᵢ)-DP. (The (ε,δ) generalization: the εᵢ and the δᵢ each add.) For an *adaptive* query sequence, write the transcript with the chain rule; once a prefix is fixed, the next query is fixed, and a fresh independent answer with budget ε_t contributes conditional ratio at most e^{ε_t}. Multiplying over steps gives e^{Σ_t ε_t}. Thus ε is a **privacy budget** spent across queries: to spend total ε_total, choose ε_t with Σ_t ε_t ≤ ε_total and use scale b_t = Δf_t/ε_t for each query.

## Post-processing and group privacy

**Post-processing immunity.** If M is ε-DP and g is any (randomized) function of M's output, then g∘M is ε-DP. *Proof:* for deterministic g and event S, with T = g⁻¹(S), Pr[g(M(x))∈S] = Pr[M(x)∈T] ≤ e^ε Pr[M(y)∈T] = e^ε Pr[g(M(y))∈S]; if g has independent randomness, condition on it and integrate the same inequality. ∎ No downstream computation without further data access can increase privacy loss.

**Group privacy.** An ε-DP mechanism is (kε)-DP for any group of k individuals. *Proof:* for x, y differing in k records, chain a path x = x₀, …, x_k = y of single-row changes; applying the neighbor bound k times gives Pr[M(x)∈S] ≤ e^{kε} Pr[M(y)∈S]. ∎ The guarantee degrades linearly in group size — exactly the desired behavior.

## Reference implementation

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
