# The cap set bound via the Croot–Lev–Pach polynomial method

## Problem

A *cap set* is a subset A ⊆ F_3^n with no three-term arithmetic progression —
equivalently no three distinct points x, y, z with x + y + z = 0 (no nontrivial
line). The question is the growth rate of r_3(F_3^n) = max |A|. For fifteen years
the best upper bound (Meshulam; Bateman–Katz) was 3^n / poly(n) — only a
polynomial saving off the trivial 3^n, by Fourier analysis. The breakthrough is
an *exponential-rate* saving with an explicit c < 3, obtained by adapting the
Croot–Lev–Pach polynomial method (originally for (Z/4Z)^n) to F_3^n. No harmonic
analysis: a rank certificate in the fixed-field, growing-dimension regime.

## Key idea

The AP condition over F_3 is a − 2b + c = 0, i.e. a + b + c = 0 — three nonzero
coefficients summing to 0 over an honest field (no ring/coset bookkeeping, unlike
the 2 in the Z/4Z case). The engine is the CLP rank lemma, generalized to handle
three nonzero coefficients. On F_q^n only *reduced* monomials matter (each
variable degree ≤ q−1); write m_d for the number of reduced monomials of total
degree ≤ d. Point-indicators are reduced polynomials, so evaluation
S_n^d → functions is the natural pairing.

**Rank lemma (Proposition).** Let α, β, γ ∈ F_q with α + β + γ = 0, A ⊆ F_q^n, and
P a reduced polynomial of degree ≤ d with P(αa + βb) = 0 for all distinct
a, b ∈ A. Then #{a ∈ A : P(−γa) ≠ 0} ≤ 2·m_{d/2}.

*Proof.* Expand P(αx + βy) = Σ c_{m,m'} m(x) m'(y) over reduced monomial pairs with
deg(mm') ≤ d. In each term one of m, m' has degree ≤ d/2, so

    P(αx + βy) = Σ_{deg m ≤ d/2} m(x)F_m(y) + Σ_{deg m ≤ d/2} m(y)G_m(x).

The matrix B_{ab} = P(αa + βb) is therefore a sum of 2·m_{d/2} rank-one matrices,
so rank(B) ≤ 2·m_{d/2}. The hypothesis makes B *diagonal*; a diagonal matrix's
rank is its number of nonzero diagonal entries, and B_{aa} = P((α+β)a) = P(−γa).
∎  (CLP's original lemma is the case (α, β, γ) = (1, −1, 0), where P(−γa) = P(0).)

**From lemma to bound (Theorem).** For caps take α = β = γ = 1 (1+1+1 = 0 in F_3).
S(A) = {αa_1 + βa_2 : a_1 ≠ a_2} is disjoint from −γA (else a nontrivial AP). Let V
be the reduced polynomials of degree ≤ d vanishing off −γA; dim V ≥ m_d − (q^n − |A|).
Each P ∈ V vanishes on S(A), so by the lemma is nonzero at ≤ 2·m_{d/2} of the −γa.
A P ∈ V of maximal support Σ has |Σ| ≥ dim V (else a Q ∈ V vanishing on Σ enlarges
the support) and Σ ⊆ −γA, so |Σ| ≤ 2·m_{d/2}. Hence

    m_d − (q^n − |A|) ≤ 2·m_{d/2},  i.e.  |A| ≤ 2·m_{d/2} + (q^n − m_d).

Since q^n − m_d ≤ m_{(q−1)n − d} (exponent complement e ↦ q−1−e) and the degree
distribution is symmetric about (q−1)n/2, balancing at d = 2(q−1)n/3 gives both
thresholds equal to (q−1)n/3:

Therefore **|A| ≤ 3·m_{(q−1)n/3}**, and for q = 3, **|A| ≤ 3·m_{2n/3}**.

## The explicit constant

m_d / 3^n = P[X_1 + … + X_n ≤ d] for X_i i.i.d. uniform on {0,1,2} (mean 1). The
threshold 2n/3 is below the mean, so this is a lower-tail large-deviation
probability. By Cramér's theorem, m_{2n/3} = (3·e^{−I(2/3)})^n up to sub-exponential
factors, with rate function I(τ) = sup_θ [θτ − log((1 + e^θ + e^{2θ})/3)]. The
optimum for τ = 2/3 is at e^θ = (√33 − 1)/8, giving

    c = 3·e^{−I(2/3)} = ((5589 + 891√33)/512)^{1/3} = 2.755104613… < 3.

The finite bound remains **|A| ≤ 3·m_{2n/3}**; its nth-root limit is c.
Equivalently, for every ε > 0 and all large n, |A| ≤ (c+ε)^n, so
|A| = o(2.756^n).

## Generalization

The lemma and theorem run over any finite field for equations
αa_1 + βa_2 + γa_3 = 0 with α, β, γ nonzero and α + β + γ = 0, assuming the only
solutions in A^3 are diagonal. Balancing at d = 2(q−1)n/3 gives
|A| ≤ 3·m_{(q−1)n/3} with exponential rate c_q < q for each fixed q. For ordinary
3-term progressions in odd characteristic, take (α, β, γ) = (1, −2, 1); for
q = 3 this is (1, 1, 1).

Tao's later slice-rank language repackages the same diagonal rank phenomenon, but
the derivation here is the CLP matrix-rank lemma plus the dimension count.

## Lower bound (principled constructions, for contrast)

Caps are closed under direct product, so a cap of size M in dimension k gives
r_3(F_3^n) ≥ M^{n/k}, i.e. base M^{1/k}. The known small maxima 2,4,9,20,45,112
(dims 1–6) give 20^{1/4} ≈ 2.1147, 112^{1/6} ≈ 2.1955; Calderbank–Fishburn (1994)
refined to ≈ 2.2104; Edel (2004), a union of compatible caps, to ≈ 2.217389 — the
best principled lower bound for more than a decade. Behrend's sphere construction (for
the integers) does not transfer to F_3^n. Principled bracket in exponential-rate
terms: roughly 2.2174 from below and 2.7551 from above.

## Code

```python
import math

def degree_counts(n):
    # coeffs[k] = [x^k] (1+x+x^2)^n = # reduced F_3-monomials of degree exactly k
    poly = [1]
    for _ in range(n):
        new = [0]*(len(poly)+2)
        for i, c in enumerate(poly):
            new[i] += c; new[i+1] += c; new[i+2] += c
        poly = new
    return poly

def m_d(n, d):
    c = degree_counts(n)
    d = int(math.floor(d))
    return sum(c[k] for k in range(min(d, len(c)-1)+1))

def is_cap(points):
    S = set(points); pts = list(points)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            x, y = pts[i], pts[j]
            z = tuple((-(a+b)) % 3 for a, b in zip(x, y))
            if z != x and z != y and z in S:
                return False
    return True

def product_lower_bound(cap_size, k):
    return cap_size ** (1.0/k)        # cap of size M in dim k -> base M^(1/k)

def upper_bound(n):
    return 3 * m_d(n, 2*n/3)          # |A| <= 3 * m_{2n/3}

def upper_bound_constant():
    u = (math.sqrt(33) - 1) / 8       # e^theta at the optimum
    cramer = (1 + u + u**2) * u**(-2/3) # = 3 * exp(-I(2/3))
    closed = ((5589 + 891*math.sqrt(33)) / 512) ** (1/3)
    assert abs(cramer - closed) < 1e-12
    return cramer

if __name__ == "__main__":
    for n in [9, 18, 30, 60, 120]:
        b = upper_bound(n)
        print(n, b, "base", round(b**(1/n), 6))
    print("c =", upper_bound_constant())   # 2.755104613...
    print(round(product_lower_bound(20, 4), 4), round(product_lower_bound(112, 6), 4))
```
