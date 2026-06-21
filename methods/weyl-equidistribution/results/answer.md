# Weyl's equidistribution theorem

**Problem.** A sequence (αₙ) of reals is *uniformly distributed (equidistributed)
mod 1* if, writing {x} for the fractional part, every subinterval gets its fair
share of the points: for all 0 ≤ b < c ≤ 1,

  lim_{N→∞} #{n ≤ N : {αₙ} ∈ [b, c)} / N = c − b.

This quantifies over all intervals at once; deciding it directly for a concrete
sequence (nξ, n²ξ, a polynomial p(n)) is intractable.

**Key idea.** Test the sequence not against interval indicators but against the
characters e(x) := e^{2πix}, which diagonalize translation mod 1 and are
Fourier-complete. Uniform distribution becomes a single analytic condition — the
decay of exponential sums — which is then provable by closed-form geometric
series in the linear case and by *Weyl differencing* (square the sum, replace a
phase by its differences to drop the polynomial degree, iterate with
Cauchy–Schwarz) in the polynomial case.

Throughout, e(x) = e^{2πix}; note ∫₀¹ e(mx)dx = 1 if m = 0 and 0 if m ≠ 0, and
|e^{2πiη} − 1| = 2|sin πη|.

---

## Weyl's criterion

**Theorem (criterion).** For a sequence (αₙ) of reals the following are
equivalent:
1. (αₙ) is uniformly distributed mod 1;
2. for every Riemann-integrable 1-periodic f,  (1/N) Σ_{n=1}^N f(αₙ) → ∫₀¹ f(x)dx;
3. for every integer m ≠ 0,  (1/N) Σ_{n=1}^N e(m αₙ) → 0.

*Proof.* **(1 ⇒ 2).** Statement (1) is (2) for f = χ_{[b,c)}, hence for every
step function (finite combination of interval indicators). A Riemann-integrable
f is trapped, for any ε, between step functions f₁ ≤ f ≤ f₂ with ∫f₂ − ∫f₁ < ε;
averaging is monotone, so lim (1/N)Σf(αₙ) lies between ∫f₁ and ∫f₂, within ε of
∫f. Let ε → 0.

**(2 ⇒ 3).** Apply (2) to f = e(m·), whose integral is 0 for m ≠ 0.

**(3 ⇒ 2).** For a trigonometric polynomial f = Σ_{|j|≤m} c_j e(jx), linearity
gives (1/N)Σf(αₙ) = Σ_j c_j (1/N)Σe(jαₙ) → c₀ = ∫f, by (3) for j ≠ 0 and
trivially for j = 0. A continuous periodic f is, by Weierstrass/Fejér, uniformly
ε-approximable by a trig polynomial f_ε; sandwiching f_ε − ε ≤ f ≤ f_ε + ε gives
(2) for continuous f up to 2ε, so for all continuous f. A Riemann-integrable f
is squeezed between two continuous functions with integrals ε-close (replace
jumps by steep ramps); squeeze again. ∎

**(2 ⇒ 1)** is the case f = χ_{[b,c)}, so all three are equivalent.

The criterion converts an interval-counting statement into the decay of a single
exponential sum S_N(m) = Σ_{n=1}^N e(mαₙ).

---

## Linear case (and the torus)

**Theorem.** ξ irrational ⇒ (nξ) is uniformly distributed mod 1.

*Proof.* Fix m ≠ 0, set η = mξ ∉ ℤ. The sum is geometric:
  Σ_{n=1}^N e(nη) = [e((N+1)η) − e(η)] / [e(η) − 1],
so |Σ_{n=1}^N e(nη)| ≤ 2/|e(η) − 1| = 1/|sin πη|, a constant independent of N —
hence o(N). Criterion (3) holds for every m ≠ 0. ∎ (If ξ = a/b is rational, take
m = b: η ∈ ℤ, the sum is N, and (nξ) cycles through finitely many points — not
equidistributed.)

**Multidimensional criterion.** On the torus ℝ^p/ℤ^p the characters are
e(m·α) = e(m₁α₁ + … + m_pα_p), m ∈ ℤ^p. The identical proof gives: α(n) is
uniformly distributed iff for every nonzero m ∈ ℤ^p,
Σ_{n=1}^N e(m·α(n)) = o(N).

**Theorem (sharpening of Kronecker).** If ξ₁,…,ξ_p admit no nontrivial integer
linear relation l₁ξ₁ + … + l_pξ_p ∈ ℤ (l_i ∈ ℤ not all 0), then (nξ₁,…,nξ_p) is
uniformly distributed on the torus. *Proof:* for m ≠ 0, m·α(n) = nη with
η = Σ m_iξ_i ∉ ℤ; geometric series ⇒ O(1) ⇒ o(N). ∎ (Equidistribution ⇒ density,
recovering Kronecker's approximation theorem, with the frequency of visits in
addition.)

---

## Polynomial case via Weyl differencing

**Theorem.** Let φ(z) = α_q z^q + … + α_1 z + α_0 have at least one of
α_q,…,α_1 irrational. Then σ_N := Σ_{h=0}^N e(φ(h)) = o(N); equivalently
(applied to mφ for each m ≠ 0) φ(0), φ(1), φ(2), … is uniformly distributed
mod 1.

**Lemma (lattice-point count).** Fix ξ irrational. In the octahedron
|r| = |r₁| + … + |r_q| ≤ n there are n_q ~ (2n)^q/q! integer points, and
  Σ_{|r|≤n} e(r₁ r₂ ⋯ r_q ξ) = o(n_q).
In particular #{r′ in the (q−1)-dim octahedron |r′| ≤ n : r₁⋯r_{q−1}ξ mod 1 ∈
(−ε, ε)} < 3ε · n_{q−1} for n large.

*Proof of Lemma (induction on q).* Base q = 1: r₁ξ equidistributes, so the
proportion of r₁ with r₁ξ ε-near 0 → 2ε < 3ε. Step: sum the last variable first,
R = r₁⋯r_{q−1},
  |Σ_{r_q} e(r_q Rξ)| ≤ min( 1/|sin π(Rξ)|, 2n+1 ).
Split outer r′ by whether Rξ is ε-near ℤ: "good" r′ (≤ n_{q−1} of them) give
≤ 1/|sin πε|; "bad" r′ — count < 3ε·n_{q−1} by the induction hypothesis — give
the crude ≤ 2n+1. Hence
  |Σ_{|r|≤n} e(r₁⋯r_q ξ)| ≤ n_{q−1}{ 3ε(2n+1) + 1/|sin πε| }.
Since n_{q−1}(2n+1)/n_q → q, dividing by n_q gives < ε(3q+1) for n large; ε
arbitrary ⇒ o(n_q). Reading off the bad count gives the stated corollary. ∎

*Proof of Theorem (irrational leading coefficient α_q).*
**Square and difference.** |σ_N|² = Σ_{h,k} e(φ(h) − φ(k)); put h = k + r:
φ(k+r) − φ(k) = r·φ(r,k), with φ(r,k) of degree q−1 in k, leading term
q α_q k^{q−1} (α_0 cancels). So |σ_N|² = Σ_r Σ_k e(r φ(r,k)).
**Iterate with Cauchy–Schwarz.** Bounding Σ_r 1·(inner) by Cauchy–Schwarz,
  |σ_N|⁴ ≤ n₁ Σ_r |Σ_k e(rφ(r,k))|² = n₁ Σ_{r,s} Σ_l e(rs φ(r,s,l)),
where k = l + s and φ(r,s,l) has degree q−2 in l, leading term q(q−1)α_q l^{q−2}.
Each round doubles the exponent on |σ_N| (2,4,8,…) and lowers the inner degree by
1. After q−1 rounds the inner phase in the running variable h is linear:
  φ(r₁,…,r_{q−1}, h) = q! α_q · h + (β₀ + β₁r₁ + … + β_{q−1}r_{q−1}).
With Q = 2^{q−1}, ξ = q! α_q (irrational since α_q is), R = r₁⋯r_{q−1},
ϱ = R(β₀ + β₁r₁ + … + β_{q−1}r_{q−1}), and
N = (n₁)^{2^{q−2}}(n₂)^{2^{q−3}}⋯n_{q−3}n_{q−2} (N = 1 for q = 2),
the accumulated inequality is
  |σ_N|^Q ≤ N Σ_{r′ : |r′|≤n} | e(ϱ) Σ_h e(Rξ h) |,
r′ over the octahedron, h over n+1−|r′| consecutive integers; one checks
N ~ κ n^{Q−q} with κ = κ(q).
**Apply the Lemma.** |Σ_h e(Rξh)| ≤ 1/|sin π(Rξ)|. For good r′ (Rξ ε-far from
ℤ) this is ≤ 1/|sin πε|, at most n_{q−1} of them; for bad r′ (count <
3ε·n_{q−1}) use the crude ≤ n+1. Then
  |σ_N|^Q ≤ N·n_{q−1}{ 3ε(n+1) + 1/|sin πε| }.
With N ~ κ n^{Q−q}, n_{q−1} ~ (2n)^{q−1}/(q−1)!, the leading term is 3ε(n+1):
  |σ_N / N|^Q ≤ 3ε ( κ·2^{q−1}/(q−1)! + 1 )  for N large.
ε arbitrary ⇒ σ_N = o(N). ∎

*Reduction when the leading coefficient is rational.* If α_q,…,α_{l+1} are
rational but α_l irrational, let G be a common denominator of α_q,…,α_{l+1}.
Split the index into residue classes mod G:
  Σ_{index} e(φ(index)) = Σ_{r=0}^{G−1} Σ_h e(φ(Gh + r)).
For each r, φ(Gh + r) ≡ ψ_r(h) (mod 1), where ψ_r has degree l and leading
coefficient α_l G^l — irrational. By the proven case each inner sum is o(n), so
the (finite) total is o(n). ∎

**Corollaries.** ξ irrational ⇒ (n²ξ), (n^kξ) uniformly distributed. A polynomial
whose non-constant coefficients are not all rational gives a uniformly
distributed sequence of values mod 1. Several polynomials φ₁,…,φ_p are jointly
uniformly distributed on the torus iff no nontrivial integer combination
Σ m_iφ_i ≡ const (mod 1). By Abel summation, for positive decreasing a_h with
divergent sum, Σ a_h e(φ(h)) = o(Σ a_h); e.g. Σ_{h≤n} e(φ(h))/h = o(log n).

Uniformity addendum: holding the highest-index irrational coefficient fixed, the
o(N) bound is uniform over the remaining coefficients (the bound depends only on
that one irrational number).

---

## Numerical illustration (illustrative, not the proof)

The proofs are the artifact; this only exhibits the analytic facts they
establish — linear sums stay bounded by 1/|sin πmξ|, polynomial sums decay.

```python
import cmath, math

def e(x):
    return cmath.exp(2j * math.pi * x)

def char_partial_sum(seq, m, N):
    "S_N(m) = sum_{n=1}^N e(m * seq(n))"
    return sum(e(m * seq(n)) for n in range(1, N + 1))

def star_discrepancy(seq, N, grid=1000):
    pts = sorted((seq(n) % 1.0) for n in range(1, N + 1))
    d, j = 0.0, 0
    for i in range(grid + 1):
        c = i / grid
        while j < len(pts) and pts[j] < c:
            j += 1
        d = max(d, abs(j / N - c))
    return d

alpha = math.sqrt(2)

# Linear: |S_N(m)| stays <= 1/|sin(pi m alpha)|, uniformly in N (stronger than o(N)).
for m in (1, 2, 3):
    bound = 1.0 / abs(math.sin(math.pi * m * alpha))
    sup_N = max(abs(char_partial_sum(lambda n: n * alpha, m, N)) for N in range(1, 4000))
    print(f"linear m={m}: sup_N |S_N| = {sup_N:.3f},  1/|sin(pi m a)| = {bound:.3f}")

# Quadratic n^2 alpha: no geometric series, but |S_N|/N -> 0 and discrepancy -> 0.
for N in (1000, 4000, 16000):
    s = char_partial_sum(lambda n: (n * n) * alpha, 1, N)
    print(f"quadratic N={N}: |S_N|/N = {abs(s)/N:.5f},  D*_N = "
          f"{star_discrepancy(lambda n: (n*n)*alpha, N):.5f}")
```

Running it: the linear suprema match 1/|sin πmα| exactly (the sums never grow),
while for n²α both |S_N|/N and the star-discrepancy shrink toward 0 as N grows —
the quantitative shadow of equidistribution proved above.
