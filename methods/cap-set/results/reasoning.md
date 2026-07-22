I keep coming back to the affine cap problem because it sits at the cleanest extreme of everything I care about: how big can a subset A of F_3^n be with no three-term arithmetic progression? Over F_3 a progression {a, a+r, a+2r} sums to 3a + 3r = 0, so the AP condition on three points is just x + y + z = 0 with the three distinct. Equivalently, A contains no nontrivial line. The whole space is 3^n points; the trivial bound is 3^n, and the question that has no consensus answer is whether you can do better by an honest exponential factor — is r_3(F_3^n) at most c^n for some fixed c < 3, or is it (3 − o(1))^n and the trivial bound is essentially the truth?

What I actually know here is all Fourier. Meshulam ran Roth's argument in this group and got 2·3^n / n: count three-term progressions as a character sum, a progression-free set forces some non-principal Fourier coefficient to be large, a large coefficient pins the set onto a hyperplane, and you recurse down a dimension. Clean, and over F_p^n it doesn't even need a density increment because the relevant Bohr set is literally a subspace. Bateman and Katz fought for years over the *structure* of the large spectrum — whether the big Fourier coefficients are spread out or additively structured, winning in either case — and pushed it to O(3^n / n^{1+ε}). But stare at that exponent. After fifteen years of the sharpest harmonic analysis anyone has, the record is still 3^n divided by a *polynomial* in n. Every single improvement since Meshulam has come from Fourier and the density increment, and they all hit the same ceiling: a polynomial saving, never an exponential one. I don't even know which way to bet on the truth.

And then yesterday I read the Croot–Lev–Pach paper, and I cannot stop thinking about it. They prove the analogous statement over Z/4Z — a subset of (Z/4Z)^n with no three-term AP has size at most c^n with c *strictly less than 4*. An honest exponential saving, the first of its kind for a problem of this shape. And the part that has me genuinely charmed: there is no harmonic analysis anywhere in it. It's the polynomial method.

That is surprising for two reasons that are precisely my reasons. First, the polynomial method is a pain over rings like Z/4Z that aren't fields — vanishing arguments lean on a polynomial having few roots, and that's a field fact. Second, the polynomial method over finite fields almost always lives in the "fixed dimension, large field" regime, whereas the cap problem is the opposite, a *fixed* base ring with the *dimension* running off to infinity. That regime has been essentially untouched by these methods. So CLP doing it over Z/4Z with growing n is exactly the combination I'd have bet against.

Let me reconstruct what they actually do, because the mechanism is the whole point. Z/4Z looks like a problem mod 4 but it's really a problem over F_2, because the AP condition a − 2b + c = 0 has that 2 in it: the two outer terms a, c must lie in the same coset of the image of doubling, and the middle term is only pinned down modulo that same subgroup. So CLP recast it. Let G = (Z/4Z)^n and let V = 2G, an n-dimensional F_2-vector space; slice the progression-free set S by cosets of V, translate each slice into V, and the no-AP condition becomes a clean combinatorial statement purely inside the F_2-vector space — certain difference sets stay disjoint from a certain index set. And then the polynomial method finishes it, and the whole engine is one lemma.

Here is that lemma, which is where I actually feel the method working in this fixed-field, growing-dimension regime. You have a low-degree polynomial P over a field, and a set A with P(a − b) = 0 for every pair of *distinct* a, b in A. The claim is that then P(0) = 0 too, provided A is bigger than roughly the number of low-degree monomials. Why would forcing P to vanish on all the differences force it to vanish at the origin? Expand P(x − y) in monomials in x and y jointly. Each monomial is (a monomial in x) times (a monomial in y), and since deg P ≤ d, in every such product the x-part and the y-part have degrees summing to at most d — so *at least one* of the two has degree ≤ d/2. Split the sum accordingly: P(x − y) = Σ_{deg m ≤ d/2} m(x)·(stuff in y) + Σ_{deg m ≤ d/2} m(y)·(stuff in x). Now read this as a bilinear pairing. Map each point a to the vector of its low-degree monomial values, u(a), and pack the "stuff" into a partner vector v(b); then P(a − b) = ⟨u(a), v(b)⟩, a scalar product of two vectors living in a space of dimension 2m, where m is the number of monomials of degree ≤ d/2. The hypothesis P(a − b) = 0 for a ≠ b says ⟨u(a), v(b)⟩ = 0 off the diagonal; if also P(0) ≠ 0, the diagonal pairings ⟨u(a), v(a)⟩ = P(0) are all nonzero. So the u(a) are |A| vectors that are mutually "orthogonal" against the v(b) with nonvanishing self-pairing — that forces them linearly independent, hence |A| ≤ 2m. If |A| exceeds 2m, the only escape is P(0) = 0. Less than a page, and it's the polynomial method working in exactly the regime everyone thought it couldn't.

Now the thought that I cannot put down. The 2 in a − 2b + c = 0 is what dragged CLP into a ring and forced the cosets-of-2A bookkeeping. But over F_3 the AP condition is a − 2b + c = 0 with 2 = −1, i.e. a + b + c = 0 — and F_3 is *already a field*. There is no ring, no 2A, no coset slicing. The obstruction that made CLP's setup intricate simply isn't there in my problem. That at least suggests the lemma should transfer; whether the transfer actually buys an exponential saving over F_3 is a separate question I have no right to assume yet — the Z/4Z bound is below 4, but the mechanics and the resulting constant are tied up in monomial counts I haven't done. So let me port it carefully and let the count decide, rather than declaring victory off the analogy.

First I need the right notion of "low-degree polynomial" over F_q. On F_q^n only reduced monomials matter — each variable to a power ≤ q − 1, because x^q = x collapses everything else — so let M_n be the reduced monomials (each individual degree ≤ q − 1) and S_n their span. The evaluation map P ↦ (P(a))_{a ∈ F_q^n} from S_n to functions on F_q^n is a linear isomorphism: both sides have dimension q^n, and it's onto because the indicator of a single point a is ∏_i (1 − (x_i − a_i)^{q−1}), a genuine reduced polynomial. I want to make sure that per-point indicator is as cheap as I'll claim, since the whole degree budget hangs off it. Over F_3 take the single-variable factor 1 − x^2 and evaluate it at the three field elements: at x = 0 it is 1, at x = 1 it is 1 − 1 = 0, at x = 2 it is 1 − 4 ≡ 1 − 1 = 0 mod 3. So 1 − x^2 is exactly the indicator of {0} in one variable, degree 2, and no degree-1 polynomial can be 1 at one element and 0 at the other two (a degree-1 line through a field has at most one root, can't kill two points). So 2 per variable is forced minimal, and the full point-indicator has total degree 2n. Let M_n^d be the reduced monomials of total degree ≤ d, S_n^d their span, m_d = dim S_n^d.

Now the lemma itself. CLP wrote it for differences a − b — the pair (1, −1, 0) of coefficients in front of (a, b, and "the origin"). But the cap condition is symmetric: a + b + c = 0 with three *nonzero* coefficients, not two-and-a-zero. Let me carry general coefficients and see what's forced. Take α, β, γ ∈ F_q with α + β + γ = 0, a set A, and a reduced P of degree ≤ d with P(αa + βb) = 0 for every pair of distinct a, b in A. Form the |A| × |A| matrix B_{ab} = P(αa + βb). Expand P(αx + βy) = Σ_{m,m'} c_{m,m'} m(x) m'(y) over pairs with deg(m·m') ≤ d; in each term at least one of m, m' has degree ≤ d/2, so

  P(αx + βy) = Σ_{deg m ≤ d/2} m(x) F_m(y) + Σ_{deg m ≤ d/2} m(y) G_m(x).

Substituting a, b ∈ A, B_{ab} = Σ_{deg m ≤ d/2} m(a) F_m(b) + Σ_{deg m ≤ d/2} G_m(a) m(b) — a sum of 2 m_{d/2} matrices, each an outer product (a column depending only on a times a row depending only on b), so each rank one. Hence rank(B) ≤ 2 m_{d/2}.

That's one jaw. The other is the hypothesis. P(αa + βb) = 0 for distinct a, b means B is *diagonal*. The rank of a diagonal matrix is exactly its number of nonzero diagonal entries, and the diagonal entry is B_{aa} = P(αa + βa) = P((α + β)a) = P(−γa) using α + β = −γ. So

  #{ a ∈ A : P(−γa) ≠ 0 } = rank(B) ≤ 2 m_{d/2}.

This is the field version, and the (α, β, γ) = (1, −1, 0) case is exactly CLP's lemma: there P(−γa) = P(0), and the statement collapses to "P(0) ≠ 0 ⇒ |A| ≤ 2 m_{d/2}", i.e. big A forces P(0) = 0. The extra room — letting γ ≠ 0 so the relevant evaluation point is −γa rather than the single point 0 — is the flexibility the cap application needs: I don't want to force P to vanish at one point; I want it to vanish on a whole *set* of places and stay alive on A.

Before I get excited I should check the one step where a hasty generalization usually dies — that the rank-one decomposition really has 2 m_{d/2} pieces and not secretly more, and that the diagonal claim only uses what I'm entitled to. The decomposition is honest: every monomial of degree ≤ d in the product splits with one factor of degree ≤ d/2, and I'm only ever indexing by those low-degree monomials m, so the count 2 m_{d/2} is right. And the diagonal claim uses P(αa + βb) = 0 only for a ≠ b, which the no-AP hypothesis will arrange. So the field-vs-ring worry I started with doesn't bite: the port goes through, and over F_3 there's no coset bookkeeping to carry.

Now turn the support bound into a bound on |A| itself. For the cap problem α = β = γ = 1 (since 1 + 1 + 1 = 0 in F_3), and the no-AP hypothesis is: αa_1 + βa_2 + γa_3 = 0 has no solution in A^3 except a_1 = a_2 = a_3. I want a P that the lemma forces to be nonzero on a *large* chunk of A, so that "nonzero at ≤ 2 m_{d/2} points" becomes a ceiling on |A|.

Here's the geometry that makes it go. Let S(A) = { αa_1 + βa_2 : a_1 ≠ a_2 in A } be the off-diagonal sums. If a point of S(A) landed in −γA, we'd have αa_1 + βa_2 = −γa_3 with a_3 ∈ A, i.e. αa_1 + βa_2 + γa_3 = 0 nontrivially — forbidden. So S(A) should be *disjoint* from −γA. This is the one place "no AP" enters the whole proof, so I want to be sure I have the logic the right way round and not off by a translation. Let me just compute it on the smallest nontrivial cap. In F_3^2 the four points {(0,0),(0,1),(1,0),(1,1)} are a cap (brute check: no three sum to 0). Their off-diagonal sums S(A) come out to {(0,1),(1,0),(1,1),(1,2),(2,1)}, and −A = {(0,0),(0,2),(2,0),(2,2)}; the two sets share nothing, disjoint, exactly as claimed. And the test has teeth: adjoin (0,2), which makes (0,0)+(0,1)+(0,2) = (0,0) a genuine line, and now S(A) and −A do intersect. So disjointness of S(A) from −γA is precisely the cap property, not a one-directional accident — good, the leverage is real. I can ask for a polynomial that vanishes off −γA (hence in particular on all of S(A)), and the lemma will then cap how often it's nonzero on −γA itself.

Build it by counting dimensions. Let V be the space of reduced P of degree ≤ d that vanish on the complement of −γA — that's a set of q^n − |A| points, so vanishing there is q^n − |A| linear conditions on the m_d-dimensional space S_n^d:

  dim V ≥ m_d − (q^n − |A|).

Every P ∈ V vanishes off −γA, so it vanishes on S(A) (which sits off −γA), so by the lemma P(−γa) ≠ 0 for at most 2 m_{d/2} values of a. Now take P ∈ V with *maximal support* Σ = { points where P ≠ 0 }. I claim |Σ| ≥ dim V: if not, "vanish on Σ" is fewer than dim V conditions and doesn't exhaust V, so some nonzero Q ∈ V vanishes on all of Σ — but then P + Q ∈ V has support strictly larger than Σ, contradicting maximality. And Σ ⊆ −γA (P dies off −γA), with the nonzero values occurring at the −γa, so |Σ| ≤ 2 m_{d/2}. Chain it:

  m_d − (q^n − |A|) ≤ dim V ≤ |Σ| ≤ 2 m_{d/2},
  |A| ≤ 2 m_{d/2} + (q^n − m_d).

Two competing monomial counts. The first term wants d small, the second wants d large. I need to understand q^n − m_d to balance them. It's the number of reduced monomials of degree *greater* than d. Reduced monomials over F_q ought to have a symmetry: the map e_i ↦ (q − 1) − e_i on exponents is a bijection sending total degree D to (q − 1)n − D, so "degree > d" monomials should biject with "degree < (q − 1)n − d" ones, giving q^n − m_d = m_{(q−1)n − d − 1}. Let me confirm that's not wishful, by counting in F_3^6 (q^n = 729). The reduced-monomial degree counts are the coefficients of (1 + x + x^2)^6 = [1, 6, 21, 50, 90, 126, 141, 126, 90, 50, 21, 6, 1] — and they read the same forwards and backwards, palindromic about degree (q−1)n/2 = 6, as the exponent-flip symmetry predicts. And the identity holds numerically: 729 − m_3 = 651 = m_8, 729 − m_4 = 561 = m_7, 729 − m_5 = 435 = m_6. The complement is exact. So

  |A| ≤ 2 m_{d/2} + m_{(q−1)n − d}.

Now both terms are lower-tail monomial counts, controlled by their thresholds d/2 and (q − 1)n − d. Since the degree distribution is symmetric about its mean (q − 1)n/2 (that's the palindrome I just checked), the natural balance is to set the two thresholds equal: d/2 = (q − 1)n − d, which gives d = 2(q − 1)n/3, and then both thresholds become (q − 1)n/3. So

  |A| ≤ 2 m_{(q−1)n/3} + m_{(q−1)n/3} = 3 m_{(q−1)n/3}.

For q = 3 that's |A| ≤ 3 m_{2n/3}. The threshold 2(q−1)n/3 wasn't pulled from anywhere — it's the point where the two competing counts have equal threshold, dictated by the symmetry of the degree distribution about its mean; pushing d either way only loosens one side faster than it tightens the other.

Before trusting it asymptotically I should sanity-check the finite bound against the one ground truth I have: the exact maximum cap sizes 2, 4, 9, 20, 45, 112 in dimensions 1–6. Any honest upper bound must be ≥ these. Computing 3·m_{⌊2n/3⌋} for n = 1..6 gives 3, 9, 30, 45, 153, 504, against true maxima 2, 4, 9, 20, 45, 112 — every one respected, comfortably (the floor on 2n/3 makes it loose at tiny n, but it never dips below a real cap, which is the only thing that would be fatal). So the bound is at least not contradicted by the known data.

Everything asymptotic now rides on how big m_{(q−1)n/3} actually is. If it's (q − o(1))^n I've gained nothing and this was a clever way to recover Meshulam. So count it. A reduced monomial over F_q is a choice of exponent in {0, …, q − 1} for each of n variables; its degree is the sum. So m_d / q^n is exactly P[ X_1 + … + X_n ≤ d ] where the X_i are i.i.d. uniform on {0, …, q − 1}. Each X_i has mean (q − 1)/2, so the sum has mean (q − 1)n/2. My threshold is (q − 1)n/3 — strictly *below* the mean. So m_{(q−1)n/3} / q^n is a *lower-tail large-deviation* probability, and those are exponentially small. That is the entire reason the bound beats q^n: I'm counting monomials below the typical degree, and there are exponentially few of them. It works precisely because (q − 1)/3 < (q − 1)/2 strictly, for every q ≥ 2.

Make it quantitative with Cramér's theorem. For the empirical mean of n i.i.d. copies of X,

  (1/n) log P[ (1/n) Σ X_i ≤ (q−1)/3 ] → − I((q−1)/3),

where I is the rate function I(x) = sup_θ [ θx − log( (1 + e^θ + … + e^{(q−1)θ}) / q ) ], the Legendre transform of the log-moment-generating function (the /q because uniform on q points). At θ = 0 the bracket is 0 and its derivative is x − (q−1)/2, which is nonzero whenever x ≠ (q−1)/2 — and my x = (q−1)/3 is below the mean — so the supremum is strictly positive. Therefore m_{(q−1)n/3} = (q · e^{−I((q−1)/3)})^n up to sub-exponential factors, with base q · e^{−I((q−1)/3)} < q. So the same argument should give an exponential saving for any fixed finite field once I have nonzero coefficients α, β, γ with α + β + γ = 0 and only diagonal solutions in A^3. For ordinary 3-term progressions in odd characteristic I'd take (α, β, γ) = (1, −2, 1); in characteristic 3 this is exactly (1, 1, 1), the cap problem. I haven't worked out how the rate behaves as a function of q, but the base is below q for each fixed q because (q−1)/3 always sits below the mean.

Now nail down the constant for the cap problem, q = 3, x = 2/3. The optimum θ solves x = (e^θ + 2e^{2θ}) / (1 + e^θ + e^{2θ}). Let u = e^θ: (2/3)(1 + u + u^2) = u + 2u^2, i.e. 2 + 2u + 2u^2 = 3u + 6u^2, so 4u^2 + u − 2 = 0 and u = (√33 − 1)/8 ≈ 0.59307 (taking the positive root). Then

  c = 3 e^{−I(2/3)} = 3 · (1 + u + u^2)/3 · u^{−2/3} = (1 + u + u^2) · u^{−2/3} ≈ 2.755104613.

I don't want to trust that I solved the quadratic and the Legendre transform without slips, so let me cross it against two independent computations. First, plugging u = (√33 − 1)/8 back into 4u^2 + u − 2 returns 0 to machine precision, so the root is right. Second, there ought to be a closed form for the cube, c^3 = (5589 + 891√33)/512, coming from clearing the u^{−2/3}; evaluating ((5589 + 891√33)/512)^{1/3} gives 2.755104613023633, agreeing with the large-deviation value (1 + u + u^2)·u^{−2/3} = 2.755104613023633 to every digit. Two routes to the same constant — I believe it. And I should make sure the finite count actually trends to this c and isn't an asymptotic mirage: 3·m_{⌊2n/3⌋} has nth root 2.7132 at n = 30, 2.7204 at n = 60, 2.7304 at n = 120, 2.7451 at n = 480, 2.7516 at n = 1920 — climbing steadily toward 2.7551 from below (the floor and the polynomial prefactor account for the gap), exactly the Cramér prediction. Meanwhile 3^n / (3 m_{2n/3}) runs off to infinity — 20.4 at n = 30, 354 at n = 60, ~8×10^4 at n = 120 — so the saving over the trivial bound is genuinely exponential, not a constant factor. The clean finite statement stays |A| ≤ 3 m_{2n/3}, an exact integer count, whose nth root tends to c; equivalently, for every ε > 0 and all large n, |A| ≤ (c + ε)^n, so |A| = o(2.756^n). The fifteen-year Fourier ceiling of 3^n / poly(n) is replaced by an honest exponential saving — and the certificate has no harmonic analysis in it at all. It's a rank bound, ported from CLP's Z/4Z argument to F_3, made cleaner by the field.

A last pass over the two pinned choices, since it came together fast. The threshold 2(q−1)n/3 is the balance point of the two monomial counts, and the palindrome I checked is exactly why that balance is symmetric; smaller or larger d only weakens one side. The field-specific indicator — that a point-indicator ∏(1 − (x_i − a_i)^2) has degree 2 per variable — I already evaluated to be the genuine indicator and argued is minimal, and that minimum is what keeps the total degrees where the counting wants them. And the rank bound's "diagonal ⇒ rank = #nonzero diagonal" step is nothing but linear algebra: a diagonal matrix's rank is its number of nonzero diagonal entries, and the off-diagonal zeros are exactly the no-AP hypothesis, which I traced through the S(A) ∩ −γA disjointness on a concrete cap above. The "no three points sum to zero" assumption enters in precisely that one place, and that single use is what propagates all the way to the exponential saving.

I do not need a symmetric tensor formalism to make the bound work. The proof I can check line by line is the CLP-style matrix rank bound, followed by the support-maximizing dimension count; that is already enough to force the exponential saving.

Let me write the certificate as runnable code, since the bound is an exact finite computation: count reduced monomials by degree via the (1 + x + x^2)^n generating function, sum to the 2n/3 threshold, multiply by 3, and watch the per-n base approach 2.7551; alongside, brute-check the small caps and the product lower bound for contrast.

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
    # the certificate: |A| <= 3 * m_{2n/3}, after balancing the two monomial counts
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
