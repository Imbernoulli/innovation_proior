Let me start from the thing that's actually bugging me. I want the largest subset A of F_3^n with no three-term arithmetic progression — no distinct x, y, z with x + z = 2y. Over F_3, 2 = −1, so 2y = −y and the condition x + z = −y rearranges to x + y + z = 0. So a cap is just a set with no three distinct points summing to zero, equivalently no nontrivial line {b, b+r, b+2r}. The whole space has 3^n points. How big can A be?

Everything I know about this problem is Fourier. You count three-term progressions in A as a sum over characters, Σ_ξ â(ξ)^3 essentially, and a 3-AP-free set forces some non-principal character to be large, and a large character means A concentrates on a hyperplane, and you recurse down a dimension. Meshulam ran exactly this and got |A| ≤ 2·3^n / n. Beautiful, clean, and it saves one factor of n. Bateman and Katz fought for years and squeezed it to 3^n / n^{1+ε} by understanding the structure of the large spectrum. But stare at that: it's still 3^n divided by a *polynomial* in n. After fifteen years of the sharpest Fourier analysis available, nobody has saved an exponential factor. Nobody even agrees whether c^n with c < 3 is *true*. There's a real possibility the answer is (3−o(1))^n and all this is the best you can do.

So let me distrust the whole Fourier frame for a moment and ask: what does the constraint x + y + z = 0 with x, y, z ∈ A actually let me write down? Not as a character sum — as an *algebraic identity*. I want a certificate of smallness that isn't a Fourier coefficient.

The field F_3 gives me a tiny exact indicator. By Fermat, every element t ∈ F_3 satisfies t^2 ∈ {0, 1}: t^2 = 0 iff t = 0, and t^2 = 1 iff t ≠ 0. So 1 − t^2 is the indicator of t = 0. For a vector w ∈ F_3^n, then,
  ∏_{i=1}^n (1 − w_i^2)
equals 1 if w = 0 and 0 otherwise. It's a polynomial, of degree exactly 2 in each variable, total degree 2n, that *is* the Kronecker delta δ_0(w). That feels important: I have a low-degree polynomial expression for "w = 0," and my constraint is exactly "x + y + z = 0."

Let me set w = x + y + z and look at the function
  F(x, y, z) = δ_0(x + y + z) = ∏_{i=1}^n (1 − (x_i + y_i + z_i)^2)
restricted to A × A × A. When is F(x,y,z) = 1? Exactly when x + y + z = 0 with x, y, z ∈ A. But A is a cap: the only way three of its points sum to zero is the trivial one x = y = z. So on A^3,
  F(x, y, z) = 1 if x = y = z, and 0 otherwise.
That is: F restricted to A^3 is the "diagonal" function — it's δ_{x=y=z}. Or written out,
  F(x, y, z) = Σ_{a ∈ A} δ_a(x) δ_a(y) δ_a(z)   on A^3,
because the right-hand side is 1 exactly when x = y = z = a for some a ∈ A, which is the same diagonal. The cap property has turned my low-degree polynomial into a diagonal tensor. Now I need a notion of "rank" that (a) is forced to be large for a diagonal with |A| nonzero entries, and (b) is forced to be small because F has low degree. If those two pressures meet, |A| gets squeezed between them.

What's the right rank for a 3-tensor? Ordinary tensor rank is the wrong tool — it's NP-hard and doesn't behave. Let me invent the rank that the low-degree structure actually produces. Look again at δ_0(x+y+z) = ∏_i (1 − (x_i+y_i+z_i)^2). If I expand this out, it's a sum of monomials in the 3n variables (x_1..x_n, y_1..y_n, z_1..z_n), each monomial being (a monomial in x)·(a monomial in y)·(a monomial in z). The total degree is at most 2n. So in any single monomial, the three blocks — the x-degree, the y-degree, the z-degree — sum to at most 2n. By pigeonhole, *at least one* of the three blocks has degree at most 2n/3.

That pigeonhole is the lever. Group the monomials by which block is the small one. A monomial where the x-block has degree ≤ 2n/3 can be written as (monomial in x of degree ≤ 2n/3) times (some function of y, z). Summing all such monomials, the whole x-low part of F is
  Σ_{m : deg ≤ 2n/3} m(x) · g_m(y, z),
a sum, over the low-degree x-monomials m, of m(x) times a leftover function of the other two blocks. Each term m(x)·g_m(y,z) is "rank one in a slice sense": it's a function of one variable times a function of the rest. Do the same for the y-low and z-low parts. So F is a sum of three families, each family being Σ_m (one-variable monomial)·(function of the other two), and the number of terms in each family is at most the number of monomials in n variables (each exponent 0,1,2 because over F_3 we reduce x^3 = x) of total degree ≤ 2n/3.

So I call a function on A^3 a *slice* if it has the form h(x_i)·(function of the other two coordinates) for some choice i ∈ {1,2,3}. Define the *slice rank* of F as the minimum number of slices summing to F. The expansion gives the upper bound
  slice-rank(F) ≤ 3 · m_{2n/3},
where m_{2n/3} = #{reduced monomials in n F_3-variables of total degree ≤ 2n/3}. The factor 3 is the three blocks; the m_{2n/3} is the pigeonhole threshold.

Now the lower bound, the other jaw of the vise. F on A^3 is the diagonal Σ_{a∈A} δ_a(x)δ_a(y)δ_a(z), all coefficients 1. I claim its slice rank is exactly |A|. The ≤ direction is trivial: each term δ_a(x)·[δ_a(y)δ_a(z)] is already a slice, so slice rank ≤ |A|. The ≥ direction is the content. Let me prove: for a diagonal tensor Σ_{a∈A} c_a δ_a(x_1)…δ_a(x_k) with c_a ≠ 0, the slice rank equals |A|.

Induct on k. Base k = 2: a diagonal *matrix* with |A| nonzero entries has ordinary matrix rank |A| — that's just linear algebra, the diagonal entries are pivots. Now suppose it's true for k−1. Take a slice decomposition of the diagonal k-tensor with R slices total; I want R ≥ |A|. Sort the slices by which coordinate they single out; say the slices that single out the last coordinate x_k are f_1(x_k)g_1(rest), …, f_p(x_k)g_p(rest), and there are R − p other slices (singling out coordinates 1..k−1). Consider the f_1, …, f_p as p functions A → F. Their span has dimension ≤ p, so the space W of functions h : A → F orthogonal to all of them (under the standard pairing Σ_{a} h(a)f_j(a)) has dimension r ≥ |A| − p. I need h in W with many nonzero coordinates. Put a basis of W in row-echelon form and look at its r pivot coordinates; projection to those coordinates is an isomorphism from W onto F^r. Choose the pivot values all equal to 1, and the resulting h ∈ W is nonzero on at least those r coordinates. Let A' = {a : h(a) ≠ 0}; then |A'| ≥ r ≥ |A| − p.

Now contract the whole identity against h in the last slot: multiply by h(x_k) and sum over x_k ∈ A. The left side: Σ_{x_k} h(x_k) Σ_a c_a δ_a(x_1)…δ_a(x_k) = Σ_a c_a h(a) δ_a(x_1)…δ_a(x_{k−1}), a diagonal (k−1)-tensor with nonzero coefficients exactly on A' (where c_a h(a) ≠ 0). By induction its slice rank is |A'| ≥ |A| − p. The right side: the p slices singling out x_k contract to Σ_x_k h(x_k) f_j(x_k) · g_j = 0 for each j, because h ⊥ f_j — they vanish! Only the R − p other slices survive, and each contracts to a slice of the (k−1)-tensor. So the contracted left side has slice rank ≤ R − p. Therefore |A| − p ≤ |A'| ≤ R − p, i.e. R ≥ |A|. The induction closes. So slice-rank(F) = |A| exactly.

The two jaws meet:
  |A| = slice-rank(F) ≤ 3 · m_{2n/3}.
A cap has size at most three times the number of low-degree monomials. Now I need to know how big m_{2n/3} is, because if it's (3−ε)^n I've gained nothing.

Let me count m_d for general threshold d. A reduced monomial in n variables over F_3 is a choice of exponent in {0,1,2} for each variable; its degree is the sum of the exponents. So the number of reduced monomials of degree exactly k is the coefficient of x^k in (1 + x + x^2)^n, and m_d is the partial sum of those coefficients for k ≤ d. Equivalently, m_d / 3^n is the probability that X_1 + … + X_n ≤ d when the X_i are i.i.d. uniform on {0, 1, 2}. The mean of each X_i is 1, so the mean of the sum is n. My threshold is d = 2n/3, which is *below* the mean n. So m_{2n/3} / 3^n is a *lower-tail* large-deviation probability — and that's exponentially small. This is exactly why it works: I'm counting monomials below the typical degree, and there are exponentially few of them relative to 3^n.

Quantify it with Cramér's theorem. For X uniform on {0,1,2},
  (1/n) log( P[ (1/n)ΣX_i ≤ 2/3 ] ) → − I(2/3),
where I is the rate function I(τ) = sup_θ [ θτ − log( (1 + e^θ + e^{2θ}) / 3 ) ] (the Legendre transform of the log-moment-generating function; the /3 is because uniform on three points). So m_{2n/3} = (3 e^{−I(2/3)})^n up to sub-exponential factors, and the base is
  c = 3 e^{−I(2/3)}.
Let me actually find the optimum. With τ = 2/3, set the derivative to zero: τ = (e^θ + 2e^{2θ})/(1 + e^θ + e^{2θ}). Let u = e^θ. Then (2/3)(1 + u + u^2) = u + 2u^2, i.e. 2 + 2u + 2u^2 = 3u + 6u^2, so 4u^2 + u − 2 = 0, u = (−1 + √33)/8 = (√33 − 1)/8. Good, u > 0. Plug back:
  I(2/3) = θ·(2/3) − log((1 + u + u^2)/3),  θ = log u,
  c = 3 e^{−I(2/3)} = 3 · (1 + u + u^2)/3 · u^{−2/3} = (1 + u + u^2) · u^{−2/3}.
Numerically u = 0.59307…, and c = 2.7551046…. The clean finite statement is still |A| ≤ 3 m_{2n/3}, but (3 m_{2n/3})^{1/n} tends to c; equivalently, for every ε > 0 and all large n, |A| ≤ (c+ε)^n. There's even a closed form for the rate: c^3 = (5589 + 891√33)/512, and c = ((5589 + 891√33)/512)^{1/3} = 2.755104613…, matching the large-deviation value to machine precision. The fifteen-year ceiling of 3^n / poly(n) is gone, replaced by an honest exponential saving, and the certificate had no Fourier analysis in it at all — it was a rank bound.

Let me make sure I haven't fooled myself, because the slice-rank framing is so clean it's suspicious. The two non-obvious choices were the degree threshold 2n/3 and the field-specific δ_0(t) = 1 − t^2. The threshold: I want the smallest universal cutoff that the pigeonhole argument can certify. F has total degree 2n; splitting across three blocks, the smallest block is ≤ 2n/3, and that 2n/3 is forced — if I picked a smaller threshold the pigeonhole could fail, while any larger threshold would only count more monomials. The 1 − t^2 is the *minimal-degree* point indicator over F_3: a nonzero polynomial that is 1 at 0 and 0 at the two nonzero field elements must have degree at least 2. Minimal degree keeps the total at 2n and the threshold at 2n/3. Both choices are pinned, not free.

Now I'm slightly uneasy about one thing: I built the slice-rank story symmetrically over x, y, z, which is elegant, but there is an asymmetric version — pin one point, look at the other two — and it's worth re-deriving it that way, because the asymmetric version is what makes the *generalization* to any odd prime field obvious and it's the cleaner thing to actually compute with.

So redo it asymmetrically. Let q be an odd prime power, A ⊆ F_q^n, and let α, β, γ ∈ F_q^* with α + β + γ = 0. Suppose A has no nontrivial solution to αa_1 + βa_2 + γa_3 = 0 (for caps over F_3 take α = β = γ = 1, since 1+1+1 = 0). Take any polynomial P of total degree ≤ d in n variables, reduced (each variable degree ≤ q−1). Look at P(αx + βy) as a function of two points. Expand it in monomials: P(αx + βy) = Σ_{m, m'} c_{m,m'} m(x) m'(y), where the sum is over pairs of reduced monomials with deg(m·m') ≤ d. In every such term, since deg(m) + deg(m') ≤ d, *at least one* of m, m' has degree ≤ d/2. So I can split
  P(αx + βy) = Σ_{m : deg m ≤ d/2} m(x) F_m(y) + Σ_{m : deg m ≤ d/2} m(y) G_m(x).
Now form the |A|×|A| matrix B with B_{ab} = P(αa + βb). From the split,
  B_{ab} = Σ_{deg m ≤ d/2} m(a) F_m(b) + Σ_{deg m ≤ d/2} G_m(a) m(b),
a sum of 2·m_{d/2} rank-one matrices (each m(a)F_m(b) and each G_m(a)m(b) is an outer product). So rank(B) ≤ 2 m_{d/2}.

Now I need the off-diagonal entries to vanish. Suppose I've also arranged that P(αa + βb) = 0 for every *distinct* pair a ≠ b in A. Then B is *diagonal*. A diagonal matrix's rank is its number of nonzero diagonal entries, and the diagonal entry is B_{aa} = P(αa + βa) = P((α+β)a) = P(−γa). So
  #{ a ∈ A : P(−γa) ≠ 0 } = rank(B) ≤ 2 m_{d/2}.
This is the sharp asymmetric rank lemma: any reduced P of degree ≤ d that vanishes on { αa + βb : a ≠ b in A } can be nonzero at −γa for at most 2 m_{d/2} points a of A. (The fully symmetric δ_0(x+y+z) slice-rank computation above is the special case bundled together; this asymmetric version is what generalizes and what I'll count with.)

Why is "vanishes on αa + βb for a ≠ b" the right hypothesis, and how do I turn it into a bound on |A| itself rather than on a support? I need a P that is *forced* to be nonzero on a large chunk of A, so that the support bound 2 m_{d/2} becomes a bound on |A|. Build it by dimension counting. The set S(A) = { αa_1 + βa_2 : a_1 ≠ a_2 ∈ A } is disjoint from −γA, because a point in the intersection would be αa_1 + βa_2 = −γa_3 with a_3 ∈ A, i.e. αa_1 + βa_2 + γa_3 = 0 — a nontrivial solution, forbidden by the hypothesis. So −γA and S(A) don't meet.

Consider the space V of reduced polynomials of degree ≤ d that vanish on the complement of −γA (a set of q^n − |A| points). Vanishing on a prescribed set of points is a system of that many linear conditions on the m_d-dimensional space of reduced polynomials of degree ≤ d, so
  dim V ≥ m_d − (q^n − |A|).
Any P ∈ V vanishes off −γA, in particular it vanishes on S(A) (which is off −γA). By the lemma, such a P is nonzero at −γa for at most 2 m_{d/2} points a. Now take P ∈ V of *maximal support* Σ = { points where P ≠ 0 }. I claim |Σ| ≥ dim V: if not, the |Σ| conditions "vanish on Σ" don't exhaust V, so there's a nonzero Q ∈ V vanishing on all of Σ, and then P + Q has support strictly containing Σ — contradicting maximality. And Σ ⊆ −γA (P vanishes off −γA), with the nonzero values at −γa, so |Σ| ≤ 2 m_{d/2}. Chaining:
  m_d − (q^n − |A|) ≤ dim V ≤ |Σ| ≤ 2 m_{d/2},
  |A| ≤ 2 m_{d/2} + (q^n − m_d).
Now the second term. q^n − m_d is the number of reduced monomials of degree *greater* than d. Reduced monomials over F_q have each exponent in {0,…,q−1}; the map e_i ↦ (q−1) − e_i is a bijection sending degree D to degree (q−1)n − D, so "degree > d" monomials biject with "degree < (q−1)n − d" monomials, of which there are ≤ m_{(q−1)n − d}. Thus
  |A| ≤ 2 m_{d/2} + m_{(q−1)n − d}.
Two competing lower-tail monomial counts, and I get to pick d. The first wants d small (fewer low-degree monomials), the second wants d large (so (q−1)n − d is small). Below the mean (q−1)n/2 these counts grow exponentially with the threshold, so the balanced choice is to make the two thresholds equal: d/2 = (q−1)n − d, giving d = 2(q−1)n/3. Then both thresholds equal (q−1)n/3, and
  |A| ≤ 2 m_{(q−1)n/3} + m_{(q−1)n/3} = 3 m_{(q−1)n/3}.
For q = 3 that's 3 m_{2n/3}, the same bound the symmetric slice-rank computation gave. The choice d = 2(q−1)n/3 wasn't pulled from a hat: it's the unique balance point of the two monomial counts, dictated by the symmetry of the degree distribution.

And the count m_{(q−1)n/3} is, again, a lower-tail probability: it's 3^n (for q=3) times P[ mean of n uniform {0,1,2} digits ≤ 2/3 ], below the mean 1, exponentially small with rate I(2/3) computed above, base c = 2.7551… < 3. For general odd q the same Cramér computation runs — X uniform on {0,…,q−1}, threshold (q−1)/3 below the mean (q−1)/2, the rate is positive because (q−1)/3 < (q−1)/2 strictly — giving r_3(F_q^n)^{1/n} bounded away from q for every odd q. For the cap problem, q = 3.

I should pin where this sits against the constructions, because the upper bound only means something with the lower bound bracketing it. Caps are closed under direct product: a line in A × B projects to a line or a point in each factor, and avoiding lines in both factors avoids them in the product, so if A ⊆ F_3^k is a cap of size M then A^t ⊆ F_3^{kt} is a cap of size M^t — per-dimension base M^{1/k}. The exact small maxima 2, 4, 9, 20, 45, 112 (dims 1..6) give 20^{1/4} ≈ 2.1147 and 112^{1/6} ≈ 2.1955; Calderbank–Fishburn's refined gluing pushes to ≈ 2.2104, and Edel's union-of-compatible-caps to ≈ 2.217389. So the principled exponential-rate bracket is roughly 2.2174 from below and 2.7551 from above. The integer trick of putting points on a sphere (Behrend) doesn't port — F_3 has no convex sphere avoiding lines — so the constructive side genuinely sits near 2.2, not near 3, and the gap to 2.7551 is real.

Let me write the certificate as runnable code, because the bound is an exact finite computation: count reduced monomials by degree via the (1+x+x^2)^n generating function, sum to the 2n/3 threshold, multiply by 3, and watch the per-n base approach 2.7551; alongside, brute-check the small caps and the product lower bound for contrast.

```python
import math

def degree_counts(n):
    # coeffs[k] = [x^k] (1+x+x^2)^n = # reduced F_3-monomials of degree exactly k
    poly = [1]
    for _ in range(n):
        new = [0]*(len(poly)+2)
        for i, c in enumerate(poly):
            new[i] += c        # multiply by 1
            new[i+1] += c      # by x
            new[i+2] += c      # by x^2
        poly = new
    return poly

def m_d(n, d):
    c = degree_counts(n)
    d = int(math.floor(d))
    return sum(c[k] for k in range(min(d, len(c)-1)+1))

def is_cap(points):
    # no distinct x,y,z in the set with x+y+z = 0
    S = set(points); pts = list(points)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            x, y = pts[i], pts[j]
            z = tuple((-(a+b)) % 3 for a, b in zip(x, y))   # the third collinear point
            if z != x and z != y and z in S:
                return False
    return True

def product_lower_bound(cap_size, k):
    return cap_size ** (1.0/k)   # cap of size M in dim k -> base M^(1/k)

def upper_bound(n):
    # the certificate:  |A| <= 3 * m_{2n/3}  (= 3 * slice-rank upper bound)
    return 3 * m_d(n, 2*n/3)

def upper_bound_constant():
    # c = 3 e^{-I(2/3)};  optimum at u = e^theta = (sqrt(33)-1)/8
    u = (math.sqrt(33) - 1) / 8
    cramer = (1 + u + u**2) * u**(-2/3)   # = 3 * exp(-I(2/3))
    closed = ((5589 + 891*math.sqrt(33)) / 512) ** (1/3)
    assert abs(cramer - closed) < 1e-12
    return cramer

if __name__ == "__main__":
    for n in [9, 18, 30, 60, 120]:
        b = upper_bound(n)
        print(n, b, "base", round(b**(1/n), 6))   # base -> 2.7551..., and 3^n/b -> infinity
    print("c =", upper_bound_constant())   # 2.755104613...
    for (M, k) in [(20, 4), (112, 6)]:
        print(M, k, round(product_lower_bound(M, k), 4))   # 2.1147, 2.1955; Edel pushes to 2.2174
```

The causal chain, start to end: I refused to read x + y + z = 0 as a character sum and read it as the algebraic identity δ_0(w) = ∏(1 − w_i^2), a degree-2n polynomial; the cap property collapsed δ_0(x+y+z) on A^3 into the diagonal Σ_a δ_a δ_a δ_a; a diagonal of |A| ones has slice rank exactly |A| (induction, contracting against a vector orthogonal to one block of slices); but low total degree forces, by pigeonhole across the three variable-blocks, a slice rank of at most 3 m_{2n/3}; so |A| ≤ 3 m_{2n/3}; and m_{2n/3} is a below-the-mean monomial count with exponential rate 2.7551… by Cramér; giving |A| = o(3^n) with an explicit base 2.7551 < 3 — the first exponential saving on the cap problem, certified by rank, not by Fourier, and the asymmetric matrix-rank version of the same lemma generalizes it to every odd prime power q.
