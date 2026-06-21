The finite-field Kakeya problem asks for the smallest possible size of a set K ⊆ F_q^n that contains a full affine line in every vector direction. For fixed dimension n, the goal is a lower bound of the form |K| ≥ c_n q^n where c_n does not depend on the field size. Older approaches exploit only partial geometric or additive-combinatorial structure. Wolff’s incidence argument gives |K| ≥ C_n q^{(n+2)/2}, and additive-combinatorial methods reach roughly C_n q^{4n/7}. Both lose a fixed fraction of the exponent because they translate directionality through overlap restrictions and sum-product growth rather than using the defining feature of the problem: every Kakeya line contributes exactly q field points. A univariate polynomial of degree less than q that vanishes at all q field elements must be the zero polynomial, and that algebraic fact is the key the earlier geometric methods miss.

Homogeneous polynomial counting moves much closer. If one restricts to homogeneous forms of degree d and forces them to vanish on K, homogeneity extends the zeros to the cone through K, and a line with many intersection points can be rescaled to produce many zeros on a line through the corresponding direction. For a full Kakeya set this gives |K| ≥ C(q+n−3, n−1) ≈ q^{n−1}/(n−1)!. Taking Cartesian products improves this to |K| ≥ C_{n,ε} q^{n−ε} for any fixed ε > 0, but the product trick never reaches the clean q^n scale. The obstruction is that a single homogeneous degree slice has only about q^{n−1} coefficients when d is close to q. Recovering the missing factor of q requires using the full polynomial space of total degree at most q−1.

The new method is the polynomial method for finite-field Kakeya. It works with all polynomials of total degree at most q−1, whose coefficient space has dimension C(n+q−1, n) ≈ q^n/n!. If a Kakeya set K were smaller than that, linear algebra would force a nonzero polynomial P of degree at most q−1 to vanish on all of K. Write P = P_0 + P_1 + ... + P_t with P_t the highest nonzero homogeneous part. For any Kakeya line {b + a y : a ∈ F_q} contained in K, the univariate restriction p(a) = P(b + a y) has degree t < q and vanishes at all q field elements, so p is identically zero. The coefficient of a^t in this restriction cannot come from any lower homogeneous piece; it comes entirely from P_t(b + a y) and equals P_t(y). Since the line direction y is arbitrary, P_t vanishes at every nonzero vector. If t > 0, homogeneity also gives P_t(0) = 0, so P_t vanishes on all of F_q^n. But a nonzero polynomial of degree t ≤ q−1 can have at most t q^{n−1} < q^n zeros by Schwartz-Zippel, so P_t must be the zero polynomial, contradicting its definition. If t = 0, then P is a nonzero constant and cannot vanish on the nonempty set K. Therefore no such vanishing polynomial exists, and |K| ≥ C(q+n−1, n) ≈ q^n/n!.

A stronger form of the argument uses multiplicity. Requiring a polynomial to vanish to multiplicity m at each point of K imposes C(m+n−1, n) linear conditions per point, while the space of polynomials of degree at most d has dimension C(d+n, n). Interpolation succeeds when C(m+n−1, n) |K| < C(d+n, n). Choosing d = ℓq − 1 and m = 2ℓ − ℓ/q for a large multiple ℓ of q, and propagating the multiplicity along Kakeya lines via Hasse derivatives, forces the top homogeneous part to vanish to multiplicity ℓ at every point of F_q^n. The multiplicity version of Schwartz-Zippel then rules this out because ℓ q^n > d q^{n−1}. In the limit this yields |K| ≥ (q/(2 − 1/q))^n ≥ q^n/2^n, which matches the natural scale of known product constructions.

```python
from math import comb

def factorial(n):
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res

def kakeya_polynomial_bound(q, n, delta=1.0, gamma=1.0):
    """
    Lower bound on the size of a (delta, gamma)-Kakeya set K ⊆ F_q^n.
    Returns the homogeneous bound and the full degree-(q-1) polynomial-method bound.
    """
    d = int(q * min(delta, gamma)) - 2
    dim_hom = comb(d + n - 1, n - 1) if d >= 0 else 0
    dim_full = comb(n + q - 1, n)
    return {
        "homogeneous_bound": dim_hom,       # ≈ q^{n-1}/(n-1)! for full Kakeya
        "degree_q_minus_1_bound": dim_full, # ≈ q^n / n!
        "asymptotic_qn_scale": (q ** n) / factorial(n)
    }

def multiplicity_bound(q, n):
    """
    Sharper multiplicity-based lower bound |K| ≥ (q/(2 - 1/q))^n.
    """
    return (q / (2.0 - 1.0 / q)) ** n

# Example
if __name__ == "__main__":
    q, n = 5, 3
    b = kakeya_polynomial_bound(q, n)
    print(f"q={q}, n={n}: |K| ≥ {b['degree_q_minus_1_bound']} "
          f"(≈ {b['asymptotic_qn_scale']:.3f} q^n)")
    print(f"Multiplicity sharpening: |K| ≥ {multiplicity_bound(q, n):.3f}")
```
