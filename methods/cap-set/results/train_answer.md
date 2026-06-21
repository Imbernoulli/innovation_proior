The cap set problem asks for the largest size of a subset A of the vector space F_3^n that contains no three-term arithmetic progression. Over F_3 the condition is especially clean: three distinct points form a progression exactly when x + y + z = 0, so a cap is simply a set with no nontrivial line. The total space has 3^n points, and for a long time the central open question was whether the maximum size r_3(F_3^n) is essentially 3^n up to sub-exponential factors, or whether there is an honest exponential saving r_3(F_3^n) ≤ c^n with c < 3.

The existing upper bounds all came from Fourier analysis. Meshulam adapted Roth's argument to F_3^n and obtained r_3(F_3^n) ≤ 2·3^n / n, and the much deeper work of Bateman and Katz improved this to O(3^n / n^{1+ε}) for some absolute ε > 0. These results are substantial, but they share a fundamental limitation: the saving is only polynomial in n, not exponential. The reason is that the Fourier method always converts the absence of progressions into a large Fourier coefficient, restricts to a subspace, and repeats, losing at best one polynomial factor along the way. The breakthrough came from Croot, Lev, and Pach, who solved the analogous problem over (Z/4Z)^n using the polynomial method rather than harmonic analysis, obtaining the first exponential-rate saving for a problem of this shape. Their argument is built around a low-degree rank lemma that works in the fixed-field, growing-dimension regime where polynomial methods had not previously been effective.

The method to use is the Croot–Lev–Pach polynomial method. Its engine is a rank lemma for low-degree polynomials. Let F_q be a finite field and let m_d denote the number of reduced monomials over F_q^n of total degree at most d. For coefficients α, β, γ ∈ F_q with α + β + γ = 0, suppose P is a reduced polynomial of degree at most d such that P(αa + βb) = 0 for every pair of distinct points a, b in a set A. Then the number of points a ∈ A for which P(−γa) ≠ 0 is at most 2·m_{d/2}. The proof expands P(αx + βy) into monomials in x and y; in each term at least one side has degree at most d/2, so the matrix with entries P(αa + βb) is a sum of 2·m_{d/2} rank-one matrices. The hypothesis makes this matrix diagonal, and the rank of a diagonal matrix equals its number of nonzero diagonal entries, which are exactly the values P(−γa).

For the cap problem we take q = 3 and α = β = γ = 1, since 1 + 1 + 1 = 0 in F_3. The no-progression condition means that the set of off-diagonal sums S(A) = {a + b : a ≠ b} is disjoint from −A. We consider the space V of reduced polynomials of degree at most d that vanish on the complement of −A. This space has dimension at least m_d − (3^n − |A|). Every polynomial in V vanishes on S(A), so the rank lemma implies it can be nonzero on at most 2·m_{d/2} points of −A. A dimension argument then shows |A| ≤ 2·m_{d/2} + (3^n − m_d). Using the symmetry of reduced monomial exponents and balancing the two competing terms at d = 2n/3 gives the clean bound |A| ≤ 3·m_{2n/3}.

This bound is an exponential improvement because m_{2n/3} counts monomials whose degree is below the mean of the degree distribution. By Cramér's theorem, m_{2n/3} grows like c^n with c strictly less than 3. The explicit constant is obtained by optimizing the rate function for a uniform distribution on {0,1,2}; the optimum occurs at e^θ = (√33 − 1)/8, yielding c = ((5589 + 891√33)/512)^{1/3} ≈ 2.755104613. Thus for every ε > 0 and all sufficiently large n, any cap set satisfies |A| ≤ (c + ε)^n. The same framework applies over any finite field to equations αa + βb + γc = 0 with nonzero coefficients summing to zero, giving an exponential saving in each fixed characteristic.

```python
import math

def degree_counts(n):
    # coeffs[k] = [x^k] (1+x+x^2)^n = # reduced F_3-monomials of degree exactly k
    poly = [1]
    for _ in range(n):
        new = [0] * (len(poly) + 2)
        for i, c in enumerate(poly):
            new[i] += c
            new[i + 1] += c
            new[i + 2] += c
        poly = new
    return poly

def m_d(n, d):
    c = degree_counts(n)
    d = int(math.floor(d))
    return sum(c[k] for k in range(min(d, len(c) - 1) + 1))

def is_cap(points):
    S = set(points)
    pts = list(points)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x, y = pts[i], pts[j]
            z = tuple((-(a + b)) % 3 for a, b in zip(x, y))
            if z != x and z != y and z in S:
                return False
    return True

def product_lower_bound(cap_size, k):
    # a cap of size M in dimension k gives per-dimension lower bound M^(1/k)
    return cap_size ** (1.0 / k)

def upper_bound(n):
    # the certificate: |A| <= 3 * m_{2n/3}
    return 3 * m_d(n, 2 * n / 3)

def upper_bound_constant():
    # c = 3 e^{-I(2/3)}; optimum at u = e^theta = (sqrt(33)-1)/8
    u = (math.sqrt(33) - 1) / 8
    cramer = (1 + u + u ** 2) * u ** (-2 / 3)
    closed = ((5589 + 891 * math.sqrt(33)) / 512) ** (1 / 3)
    assert abs(cramer - closed) < 1e-12
    return cramer

if __name__ == "__main__":
    for n in [9, 18, 30, 60, 120]:
        b = upper_bound(n)
        print(n, b, "base", round(b ** (1 / n), 6))
    print("c =", upper_bound_constant())
    print(round(product_lower_bound(20, 4), 4), round(product_lower_bound(112, 6), 4))
```
