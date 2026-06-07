# Synthesis — Large Sum-Free Sets in Sets of Integers

## The two problems
1. **Largest sum-free subset.** Every set A of n nonzero (WLOG positive) integers contains a sum-free subset (no x+y=z) of size s(A). Show s(A) > n/3, with the best constant.
2. **Counting.** Bound the number of sum-free subsets of [N] = {1,...,N}: it is O(2^{N/2}) (Cameron–Erdős conjecture).

## Problem 1: the lower bound

### Erdős's argument (random dilation + middle-third projection)
- Key geometric fact: the arc B = (1/3, 2/3) in T = R/Z is **sum-free**: if {θa},{θb} ∈ (1/3,2/3) then {θa}+{θb} ∈ (2/3, 4/3), which mod 1 is (2/3,1)∪(0,1/3), disjoint from (1/3,2/3). So if a+b=c in A and θa,θb both land in the arc, θc = θa+θb cannot. Hence A_θ := {a ∈ A : {θa} ∈ (1/3,2/3)} is **sum-free for every θ**.
- Randomize θ ∈ T uniformly. For each a (nonzero integer), {θa} is uniform on T (since a≠0), so P({θa} ∈ (1/3,2/3)) = 1/3. Linearity of expectation: E|A_θ| = n/3.
- Therefore there exists θ with |A_θ| ≥ n/3. Since |A_θ| is an integer, |A_θ| ≥ ⌈n/3⌉, giving s(A) ≥ n/3 (and > n/3 when 3∤n).

### Counting-function / Fourier formulation (Bourgain's frame, used by Shakan)
- Let f = 1_{[1/3,2/3)} − 1/3 (mean-zero on T). Let f_A(x) = Σ_{a∈A} f(ax). Then |A_x| = n/3 + f_A(x), and m_A := max_x f_A(x) satisfies s(A) ≥ n/3 + m_A.
- ∫ f_A = 0 ⇒ m_A ≥ 0, and since |A_θ| ∈ Z, m_A ∈ Z + (something)... precisely m_A ∈ −n/3 + Z i.e. n/3+m_A ∈ Z. So m_A > 0 strictly (as f_A is non-constant), giving the **Alon–Kleitman** improvement s(A) ≥ (n+1)/3.
- Fourier series of f: f(x) = Σ_{m≥1} (√3/π)(χ(m)/m) cos(2πmx) where χ is the nonprincipal character mod 3 (χ(1)=1, χ(2)=−1, χ(0)=0). [Coefficient: \hat f via 1_{[1/3,2/3)}; the √3/π and χ(m)/m fall out.] So f_A(x) = (√3/π) Σ_{a∈A} Σ_{m≥1} (χ(m)/m) cos(2πmax).

### Bourgain's (n+2)/3 and the L1 / Littlewood connection
- Bourgain proved s(A) ≥ n/3 + C·‖Σ_{a∈A} cos(2πa·)‖_{L^1(T)}. The L1 norm of the exponential sum (the **Littlewood norm**) measures how "spread out"/random A is. Large L1 ⇒ large surplus over n/3. Small L1 (structured, AP-like A) is the hard case.
- Concrete result (Bourgain, reproved by Shakan): for coprime A of size n, either A={1,2} or s(A) ≥ (n+2)/3. m_A ∈ n/3 + Z; m_A ≥ 1 suffices.

### Shakan's reduction (clean reconstruction of the dual/test-function method)
- Test-function (dual) method: to show m_A ≥ S/3 it suffices to find φ ≥ 0 with ∫φ = 1 and ∫ φ f_A > (S−1)/3... precisely m_A ≥ ∫ φ f_A for any prob. density φ. Build φ as products (1−c(ux))(...) with c(x)=cos(2πx); nonnegativity by construction; ∫ φ f_A computed via Plancherel using \hat f(m) ∝ χ(m)/m.
- Partition A = A_0 ∪ A_1: A_0 = elements coprime to 3, A_1 = multiples of 3.
  - **Lemma 1 (pigeonhole on residues mod 3):** m_A ≥ |A_0|/6 − |A_1|/3. (At least half of A_0 is ≡1 or ≡2 mod 3, so f_{A_0}(1/3) or f_{A_0}(2/3) ≥ |A_0|/6; A_1 contributes −|A_1|/3 at x=1/3 since 3a multiples send {ax}→0, f=−1/3.)
  - **Lemma 2 (inductive descent):** m_A ≥ m_{A_1}. Because for a∈A_1 (3|a), f_{A_1}(x)=f_{A_1}(x+1/3)=f_{A_1}(x+2/3); and for a∈A_0, f(ax)+f(a(x+1/3))+f(a(x+2/3))=0 (averaging f over the three shifts of an order-3-coprime point gives the mean 0). Summing: f_A(x)+f_A(x+1/3)+f_A(x+2/3)=3m_{A_1}, so the max ≥ m_{A_1}.
  - Induction: if |A_0| large, Lemma 1 wins; else |A_1| ≥ (n−2S)/3 is itself large, recurse via Lemma 2. Reduces Conj (m_A ≥ S/3 for all large n) to finitely many base cases N_S < n ≤ 3N_S+2S.
- Bourgain's lemma (1∉A ⇒ m_A>1/3): build φ(x)=(1−c(ux))(1−c(vx)) with u=min A, v=smallest element with u∤v; ∫φ f_A = (√3/2π)(2 − ½·1_A(u+v)) > 1/3.

## Problem 2: counting — Green's container/granularization method

Goal: |SF(N)| = O(2^{N/2}); in fact ~ c(N) 2^{N/2}, c(N) depending on parity of N.

Lower bound (Cameron–Erdős, pre-method context):
- Odd numbers in [N]: a sum-free family of size N/2 → 2^{N/2} subsets.
- Subsets of {⌈N/2⌉,...,N} (top half): sum-free, ~N/2 elements → 2^{N/2}.
- Cameron–Erdős counted sum-free subsets of {⌈(N+1)/3⌉,...,N} and got asymptotically c(N)2^{N/2}.

Green's two-part strategy:
- **Part I (containers).** Build a family F of "containers" with: (i) every A∈F is almost sum-free (o(N^2) additive triples); (ii) every sum-free A ⊆ some A'∈F; (iii) |F| = 2^{o(N)}.
- **Part II (structure + count).** For A with container F': if |F'| ≤ (1/2 − 1/120)N, those A number o(2^{N/2}). If |F'| ≥ (1/2−1/120)N, structural theorem forces A to be (essentially) all-odd or inside an interval, both counted at ~2^{N/2}.

Container construction (granularization, from Green–Ruzsa):
- Work in Z/pZ, p∈[2N,4N] prime (makes A sum-free in Z/pZ since sums < p). Fourier transform \hat A(r)=Σ A(x)e(rx/p).
- Partition Z/pZ into M APs I_i of common difference d, length ≈ L=⌈p/M⌉. Keep blocks where A has density ≥ ε_1: T={i : |A∩I_i| ≥ ε_1|I_i|}. Granularization A' = ∪_{i∈T} I_i. Then |A'\A| ≤ ε_1 p.
- **Good length d:** ‖dr/p‖ ≤ (1/4L)(δp/|\hat A(r)|)^{1/2} for all large Fourier coefficients r (|\hat A(r)|≥δp), δ = (1/16)ε_1^2 ε_2 ε_3^{1/2} α^{−1/2}. Pigeonhole (Dirichlet) + Parseval (Σ_{r∈R}|\hat A(r)|^2 ≤ α p^2, AM-GM) guarantees a good d exists once p > (4L)^{O(...)}.
- **Key prop:** if d good, smoothing g(x)=(1/(2L−1))Σ_{|j|<L} e(jdx/p) satisfies |\hat A(x)||1−g(x)^2| ≤ δp (uses 1−cos2πt ≤ 2π^2‖t‖^2). Then via Parseval, Σ_n |(A*A)(n) − (a_1*a_1)(n)|^2 ≤ α δ^2 p^3 where a_1 = A smoothed by the AP P. Consequence: A+A contains all x with A'*A'(x) ≥ ε_2 p except ≤ ε_3 p. So sum-free A ⇒ A' almost sum-free (≤ ε p^2 triples). [Prop almost-sum-free: ε_1=ε, ε_2=ε^2/144, ε_3=ε^2/80.]
- Choose ε=(log N)^{−1/11}, M=⌊N exp(−(log N)^{1/12})⌋. F = unions of APs with ≤ εp^2 triples, plus ≤ εp added elements. |F| ≤ p·2^M·(subsets of size ≤εN) = 2^{o(N)}.

Structure theorem (Part II core), Prop 15:
- If A⊆[N] has ≤ εN^2 triples and |A| = (1/2−η)N, η ≤ 1/50, then either (i) A is within an interval of length (1/2+3η+O(ε^{1/8}))N up to O(ε^{1/8}N) exceptions, or (ii) ≤ O(ε^{1/8}N) elements of A are even.
- Proof tools: popular-difference set D(A,K), the bound ½|D(A,√ε N)| + |A| ≤ N(1+2√ε) (else too many triples); Lev–Łuczak–Schoen graph lemma; Lev–Smeliansky projection π: Z→Z/tZ with t=M−m>N/4; Kneser's theorem giving an index-2 subgroup ⇒ single parity (odd).
- Corollary 4: with o(2^{N/2}) exceptions every sum-free A is all-odd or ⊆ {⌈(N+1)/3⌉,...,N}.
- Finish: count those two families (Cameron–Erdős for the interval) → c(N)2^{N/2}.

Sapozhenko independently: more combinatorial container approach, coined "containers."

## Design-decision → why
- Why interval (1/3,2/3): the unique-length-1/3 arc symmetric about 1/2 whose sumset avoids itself; 1/3 measure ⇒ n/3. Alternatives (any arc of length L) are sum-free only if L ≤ 1/2 and positioned to avoid self-sums; (1/3,2/3) maximizes density among symmetric self-sum-avoiding arcs (length up to 1/3 around 1/2... actually (a,1−a) is sum-free iff a≥1/3). Max density = 1/3.
- Why random θ (dilation) not a fixed map: a single dilation could align A badly; averaging guarantees existence without constructing it.
- Why integrality gives +1: E=n/3 over a continuum but the realized value is an integer; max ≥ ⌈⌉ and strictly > n/3 since f_A nonconstant.
- Why L1/Littlewood norm: f_A's surplus over n/3 is governed by ‖Σ cos(2πa·)‖_{L1}; structured (AP) sets have small L1 — the genuine obstruction, hence Bourgain handles them by hand / small cases.
- Why test functions φ≥0, ∫φ=1: dual certificate — ∫φ f_A ≤ m_A lower-bounds the max; nonnegative trig polynomials (squares/products of (1−cos)) give computable certificates via \hat f(m) ∝ χ(m)/m.
- Why partition mod 3: f and the arc are 3-periodic in structure; multiples of 3 (A_1) are "invisible" to the 3 shifts (constant), coprime-to-3 (A_0) average to 0 over shifts — splits the problem into a pigeonhole gain (A_0) and a self-similar sub-instance (A_1).
- Why move to Z/pZ, p∈[2N,4N]: embeds [N] in a group where Fourier analysis is clean and A stays sum-free (sums of two elements < 2N ≤ p, no wraparound).
- Why granularize into APs: replaces an arbitrary set by a union of structured blocks (a "container") computable from O(M) bits, so the family is small (2^{o(N)}); good length ensures the block-structure preserves the additive (sum-free) property approximately.
- Why almost-sum-free not sum-free: exact sum-free is too rigid to be "containerized" into few sets; relaxing to o(N^2) triples lets a small family cover all sum-free sets while still forcing structure.
- Why the (1/2−1/120)N threshold: below it the container is small enough that the number of subsets is o(2^{N/2}); above it the structure theorem kicks in.
- Why Kneser → odd: index-2 subgroup of Z/tZ is the evens; single-parity ⇒ odd (evens have too many triples).

## Code framework
This is a math task; the "code" is a faithful, runnable implementation of the algorithmic content:
- Erdős/Alon-Kleitman: brute/grid search over θ of the middle-third projection (verifies s(A) ≥ ⌈(n+1)/3⌉), plus the exact dual-certificate evaluation ∫ φ f_A via the χ(m)/m Fourier coefficients.
- Counting: enumerate/verify the container picture on small N — count sum-free subsets of [N] exactly and compare to 2^{N/2}; verify the all-odd and top-interval families.
Grounded in the explicit formulas in Shakan (f, f_A, χ(m)/m, the φ certificates) and Green (granularization, good length).

## Sources
- Shakan, "On the largest sum-free subset problem in the integers", arXiv:2207.14210 (LaTeX source read in full).
- Green, "The Cameron–Erdős Conjecture", arXiv:math/0304058 (LaTeX source read in full).
- Wikipedia Sum-free set / Cameron–Erdős conjecture; Quanta (Bedert 2025); johndcook blog; Alon–Kleitman note arXiv:1606.07823.
