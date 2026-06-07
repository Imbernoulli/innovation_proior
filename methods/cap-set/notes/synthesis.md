# Synthesis — Cap Set Problem and the Polynomial Method

## The problem
- A *cap set* A ⊆ F_3^n: no three-term AP, i.e. no distinct x,y,z with x+y+z=0 (equivalently x,y,z collinear / a "line"). Over F_3 a 3-AP {a, a+r, a+2r} sums to 3a+3r=0, so the AP condition is exactly x+y+z=0. r_3(F_3^n) = max cap size.
- Want: is r_3(F_3^n) ≥ (3-ε)^n for all ε (close to trivial 3^n) or is it c^n with c bounded away from 3? Consensus was unclear pre-2016.

## Prior art (upper bounds) — the wall
- Roth/Fourier + density increment: Meshulam 1995 (the F_p^n analogue of Roth, *elementary* via Fourier, no density increment needed in this group): r_3(F_3^n) ≤ 2·3^n/n. Only polynomial savings.
- Bateman–Katz 2012: r_3(F_3^n) = O(3^n / n^{1+ε}). Deep, hard, still only polynomial-in-n savings off 3^n. After ~15 years that was the record.
- Sanders for Z_4^n analog improved log factors.
- KEY pain: every approach was Fourier / density-increment and stuck at 3^n/poly(n). No exponential improvement c<3 in sight.

## Lower bounds (constructions) — the other side
- Behrend 1946: for {1..N} ⊆ Z, sets of size N·exp(-c√log N) via points on a sphere (convex => no 3 collinear). This is the integer story; it does NOT transfer to F_3^n because F_3 has no notion of "sphere/convexity" that beats trivial, and spheres in F_3^n are small. So the cap set lower bound is genuinely different and weaker than naive Behrend intuition.
- Product/tensor construction: if A ⊆ F_3^k is a cap (size m), then A^t ⊆ F_3^{kt} is a cap of size m^t. So a single cap of size m in dimension k gives asymptotic lower bound m^{1/k} per dimension.
  - largest caps: a_1=2, a_2=4, a_3=9, a_4=20, a_5=45, a_6=112, a_7..(?).  a_4=20 => 20^{1/4}=2.1147; a_6=112 => 112^{1/6}=2.1956.
- Calderbank–Fishburn 1994: refined product (not just Cartesian power) => c ≈ 2.2104.
- Edel 2004: more elaborate product construction (union-of-caps under conditions) => c ≈ 2.217389. Best *principled* lower bound for ~two decades.
- So the principled brackets: 2.2174^n (lower) ... 2.756^n (upper). Gap remains, but both ends are exponential and away from trivial. (FunSearch/AlphaEvolve later nudged small-dim caps; those are search outputs and the ANTI-PATTERN here — principled is product constructions + Edel.)

## THE METHODOLOGICAL LEAP — Croot–Lev–Pach polynomial method
Two equivalent expositions: (A) CLP/EG "polynomial vanishing + rank-1 matrix decomposition"; (B) Tao's slice rank.

### CLP/EG core lemma (the engine)
Let S_n^d = polynomials over F_q in n vars, each variable degree ≤ q-1 (so "reduced"), total degree ≤ d. m_d = dim = # such monomials.
Evaluation map S_n -> functions on F_q^n is an isomorphism (dim q^n both sides; indicator of point a is ∏(1-(x_i-a_i)^{q-1})).

**Proposition (EG Prop 2 / CLP Lemma 1).** Let α+β+γ=0 in F_q. Suppose P ∈ S_n^d with P(αa+βb)=0 for all distinct a,b ∈ A. Then #{a ∈ A : P(-γa) ≠ 0} ≤ 2 m_{d/2}.

*Proof.* P(αx+βy) = Σ_{m,m', deg(mm')≤d} c_{m,m'} m(x)m'(y). In each term at least one of m,m' has degree ≤ d/2 (since deg(mm')≤d). So
  P(αx+βy) = Σ_{m∈M^{d/2}} m(x) F_m(y) + Σ_{m∈M^{d/2}} m(y) G_m(x).
Form matrix B_{ab}=P(αa+βb) over a,b∈A. Then B = Σ_{m∈M^{d/2}} m(a)F_m(b) + Σ_m G_m(a) m(b): a sum of 2 m_{d/2} rank-1 matrices => rank ≤ 2 m_{d/2}. But by hypothesis B is *diagonal* (off-diagonal entries P(αa+βb)=0 for a≠b). A diagonal matrix's rank = # nonzero diagonal entries = #{a: P(-γa)≠0} (since αa+βa=(α+β)a=-γa). Hence ≤ 2 m_{d/2}. ∎

### From lemma to bound (EG Theorem 4)
α=β=γ=1 (1+1+1=0 in F_3), A a cap (no nontrivial a1+a2+a3=0). The function we vanish: build P that vanishes on the *complement* of -γA = -A but we want it to detect A. The slick dimension-count:
- V = polynomials in S_n^d vanishing on complement of (−γA). dim V ≥ m_d − (q^n − |A|) = m_d − q^n + |A|.
- Such P vanish on S(A) (sums αa1+βa2 over distinct a1,a2) which is disjoint from −γA for a cap. By Prop, P(−γa)≠0 for ≤ 2 m_{d/2} points a. Taking P∈V of maximal support Σ: |Σ| ≥ dim V (else a nonzero Q∈V vanishes on Σ, P+Q has bigger support — contradiction). And |Σ| ≤ 2 m_{d/2}. So dim V ≤ 2 m_{d/2}:
  m_d − q^n + |A| ≤ 2 m_{d/2}  =>  |A| ≤ 2 m_{d/2} + (q^n − m_d).
- q^n − m_d = # reduced monomials of degree > d = (by complementation e_i -> q-1-e_i) # monomials of degree < (q-1)n − d ≤ m_{(q-1)n−d}.
- Choose d = 2(q-1)n/3. Then m_{d/2}=m_{(q-1)n/3}, and (q-1)n−d=(q-1)n/3 so q^n−m_d ≤ m_{(q-1)n/3}. Get |A| ≤ 3 m_{(q-1)n/3}.

### The constant c (EG Cor 5 via large deviations / Cramér)
m_{(q-1)n/3}/q^n = P[ X_1+...+X_n ≤ (q-1)n/3 ] where X_i uniform on {0,...,q-1}. (Reduced monomial degrees = sum of n iid uniform digits; bound on degree ↔ mean ≤ (q-1)/3, below the mean (q-1)/2, so a large-deviation lower tail.)
Cramér: lim (1/n) log(m_{(q-1)n/3}/q^n) = −I((q-1)/3), I(x)=sup_θ [θx − log((1+e^θ+...+e^{(q-1)θ})/q)].
For q=3, x=2/3: optimum at e^θ=(√33−1)/8. Plug in => m_{2n/3} = (3 e^{−I(2/3)})^n with 3 e^{−I(2/3)} < 2.756. So |A| ≤ 3 m_{2n/3} = O(2.756^n) = o(3^n). DONE. (Equivalently the entropy form: maximize h(α,β,γ) s.t. α+β+γ=1, β+2γ≤2/3, giving X≈1.013455 and c=3^X≈2.756.)

Exact c = ((5589 + 891√33)/8)^{1/3} ≈ 2.75510461... (Zeilberger/Tao rendition closed form). EG state "< 2.756".

### Tao's symmetric slice rank reformulation
- slice rank of F: A^k -> F = min # of rank-1 *slices* f(x_i)·g(rest). 
- **Diagonal lemma:** slice-rank of Σ_a c_a δ_a(x_1)...δ_a(x_k) = #{a: c_a≠0}. (Induction on k; the orthogonal-complement extraction reduces to k-1; base k=2 is matrix rank = # nonzero diagonal.)
- **Capset identity:** for a cap A, on A^3: δ_{0}(x+y+z) = Σ_{a∈A} δ_a(x)δ_a(y)δ_a(z), because x+y+z=0 with x,y,z∈A forces x=y=z (no nontrivial line). LHS is diagonal-tensor with all c_a=1 => slice rank = |A|.
- **CLP upper bound on slice rank of δ_0(x+y+z):** expand δ_0(w)=∏_i(1−w_i^2) over F_3, w=x+y+z, total degree 2n. Each monomial in (x,y,z) has total degree ≤ 2n, so by pigeonhole one of the three blocks of variables (x or y or z) carries degree ≤ 2n/3; group monomials by which block is "low-degree", giving 3 families each a slice => slice rank ≤ 3·#{monomials in n F_3-vars of degree ≤ 2n/3} = 3 m_{2n/3}.
- Combine: |A| = slicerank ≤ 3 m_{2n/3} = O(2.756^n). Same constant.

## Code (verification, real)
A faithful, runnable realization computes m_d exactly (DP over the generating function ∏(1+x+x^2)^n coefficients = # reduced monomials by degree), then the EG bound 3·m_{2n/3}, and the constant via the optimization. Also a brute-force cap check + product construction lower bound in small dims as the contrast. This is the "rank computation / construction" code the skill asks for. Grounded in the generating-function fact m_d = Σ_{deg≤d} coeff of ∏(1+x+x^2) (each F_3 variable contributes 1+x+x^2).

## Design decisions / why
- Why polynomial method not Fourier: Fourier/density-increment provably stuck at poly savings (Meshulam, Bateman–Katz); the rank/dimension counting is a *different* certificate that yields exponential savings.
- Why degree threshold d = 2(q-1)n/3 (i.e. 2n/3 over F_3): balances the two competing monomial counts (2 m_{d/2} and q^n − m_d ≤ m_{(q-1)n − d}); equality of the two exponents at d=2(q-1)n/3 by symmetry of the degree distribution around its mean (q-1)n/2.
- Why δ_0(x)=1−x^2 over F_3: Fermat, x^2=1 for x≠0, =0 for x=0, so 1−x^2 is the point indicator; degree 2 per variable is the minimum, controlling total degree 2n.
- Why "at least one of m,m' deg ≤ d/2": pigeonhole on deg(mm')≤d — the crux that makes the matrix low rank.
- Why diagonal => rank = support: a cap kills off-diagonal; diagonal rank is elementary. This is where "no 3-AP" enters.
- Why pigeonhole 2n/3 in slice form: total degree 2n split into 3 blocks, min block ≤ 2n/3.
- Lower bound: product construction is the principled constructive engine; m^{1/k} per dim; Edel 2.2174 best for two decades; Behrend's sphere idea doesn't port to F_3^n.

## Uncertainty flags
- Exact closed form c = ((5589+891√33)/8)^{1/3} from Zeilberger's "Motivated Rendition" title and consistent with EG "<2.756"; I report ≈2.75510461 and the EG-stated bound 3e^{−I(2/3)}<2.756.
- a_7 largest-cap value not needed; lower-bound numbers 2.1147 (a_4), 2.1956 (a_6=112), CF 2.2104, Edel 2.2174 cross-checked across two sources.
