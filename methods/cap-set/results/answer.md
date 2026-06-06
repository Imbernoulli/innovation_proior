# The cap set bound via the polynomial / slice-rank method

## Problem

A *cap set* is a subset A ⊆ F_3^n with no three-term arithmetic progression —
equivalently no three distinct points x, y, z with x + y + z = 0 (no nontrivial
line). The question is the growth rate of r_3(F_3^n) = max |A|. For fifteen years
the best upper bound (Meshulam; Bateman–Katz) was 3^n / poly(n) — only a
polynomial saving off the trivial 3^n, by Fourier analysis. The breakthrough is
an *exponential-rate* saving with an explicit c < 3, obtained by a rank
(polynomial-method) certificate with no Fourier analysis.

## Key idea

Over F_3, 1 − t^2 is the indicator of t = 0 (Fermat: t^2 ∈ {0,1}). So
δ_0(w) = ∏_{i=1}^n (1 − w_i^2) is a polynomial of total degree 2n equal to the
Kronecker delta at 0. Set w = x + y + z. On A^3, the cap property forces
x + y + z = 0 to mean x = y = z, so

    δ_0(x + y + z) = Σ_{a ∈ A} δ_a(x) δ_a(y) δ_a(z)   on A × A × A,

a **diagonal tensor** with |A| nonzero entries. Two opposing bounds on its
**slice rank** (the min number of terms h(x_i)·g(rest)) squeeze |A|:

- **Lower:** a diagonal tensor Σ_{a} c_a δ_a(x_1)…δ_a(x_k) with c_a ≠ 0 has slice
  rank exactly |{a : c_a ≠ 0}| (induction on k; base case = matrix rank of a
  diagonal matrix; inductive step contracts against a vector orthogonal to one
  coordinate's p slices and chosen with at least |A|−p nonzero coordinates, then
  drops to k−1). Hence slice-rank = |A|.
- **Upper:** δ_0(x+y+z) has total degree 2n, so in each monomial one of the three
  variable-blocks x, y, z has degree ≤ 2n/3 (pigeonhole). Grouping by which block
  is low gives 3 families of slices, so slice-rank ≤ 3·m_{2n/3}, where m_d is the
  number of reduced F_3-monomials (exponents in {0,1,2}) of degree ≤ d.

Therefore **|A| ≤ 3·m_{2n/3}**.

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

## Asymmetric form (generalizes to all odd q)

The asymmetric matrix-rank lemma: for nonzero α, β, γ ∈ F_q with
α + β + γ = 0, and a reduced P of degree
≤ d that vanishes at αa + βb for all distinct a, b ∈ A, the matrix
B_{ab} = P(αa + βb) is diagonal; writing P(αx+βy) = Σ_{deg m ≤ d/2} m(x)F_m(y) +
Σ_{deg m ≤ d/2} m(y)G_m(x) expresses B as 2·m_{d/2} rank-one matrices, so

    #{a : P(−γa) ≠ 0} = rank(B) ≤ 2·m_{d/2}.

A dimension count (P of maximal support in the space vanishing off −γA, which has
dim ≥ m_d − (q^n − |A|), and S(A) = {αa+βb : a≠b} is disjoint from −γA under the
no-nontrivial-solution hypothesis)
yields |A| ≤ 2·m_{d/2} + (q^n − m_d). Since q^n − m_d ≤ m_{(q−1)n − d}
(complement exponents e ↦ q−1−e), and the degree distribution is symmetric about
(q−1)n/2, balancing at d = 2(q−1)n/3 gives **|A| ≤ 3·m_{(q−1)n/3}** — exponential
rate c < q for every odd prime power q (q = 3 is the cap set case).

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
