# Synthesis — Subcubic matrix multiplication & the tensor-rank viewpoint

## Pain point / research question
Multiplying two n×n matrices by the definition c_ij = sum_k a_ik b_kj costs n^3 scalar
multiplications and n^2(n-1) additions — Θ(n^3) total. Universally assumed optimal (it
"obviously" must touch every product a_ik b_kj). Question: can it be done in o(n^3)?

## Key facts (all grounded this run)
- Strassen 1969, "Gaussian elimination is not optimal", Numer. Math. 13(4):354-356. Claims
  < 4.7 n^(log2 7) operations vs ~2n^3.
- Naive block recursion: split each n×n into four n/2 blocks; AB needs 8 block products +
  4 block additions ⇒ T(n)=8T(n/2)+O(n^2) ⇒ T(n)=Θ(n^3). No gain — 8 = 2^3.
- The barrier is the multiplication count 8. log2(8)=3. To beat cubic you must cut the
  number of *multiplications* (recursive subproblems), because additions are the cheap
  O(n^2) base level; the exponent is log2(#multiplications).
- Karatsuba analogy: (a0+a1X)(b0+b1X): naive 4 products, but 3 suffice via
  a0b0, a1b1, (a0+a1)(b0+b1)-a0b0-a1b1. Trade 1 mult for a few adds; recursion gives
  n^log2(3). Precedent that #multiplications, not #additions, drives the exponent.
- Strassen's 7 products (canonical / Wikipedia form), with A=[[a11,a12],[a21,a22]],
  B likewise:
  M1=(a11+a22)(b11+b22)
  M2=(a21+a22)b11
  M3=a11(b12-b22)
  M4=a22(b21-b11)
  M5=(a11+a12)b22
  M6=(a21-a11)(b11+b12)
  M7=(a12-a22)(b21+b22)
  C11=M1+M4-M5+M7 ; C12=M3+M5 ; C21=M2+M4 ; C22=M1-M2+M3+M6.
  VERIFIED symbolically (sympy) against c_ij. 7 mults, 18 add/sub.
- Crucially the 7 products use NO commutativity of the entries — each M_k is (linear form in
  A entries)·(linear form in B entries). So it lifts to BLOCK matrices (entries are
  matrices, which don't commute) ⇒ recursion is valid.
- Recurrence T(n)=7T(n/2)+O(n^2) ⇒ T(n)=Θ(n^(log2 7)), log2 7 ≈ 2.807.
- The canonical code (hardianlawi) uses the Strassen–Winograd ordering:
  p1=a(f-h), p2=(a+b)h, p3=(c+d)e, p4=d(g-e), p5=(a+d)(e+h), p6=(b-d)(g+h), p7=(a-c)(e+f);
  C11=p5+p4-p2+p6, C12=p1+p2, C21=p3+p4, C22=p1+p5-p3-p7. With block names
  A=[[a,b],[c,d]], B=[[e,f],[g,h]]. VERIFIED runs == numpy matmul for n=1..31 (odd via padding).
  Both orderings are equivalent decompositions (the 7-mult algorithm is unique up to
  symmetry — de Groote 1978).

## Bilinear / tensor framing (Bläser survey, Theory of Computing Grad. Surveys 5 (2013))
- Any bilinear algorithm = decomposition XY = sum_{k=1}^r u_k(X) v_k(Y) W_k, u_k,v_k linear
  forms, W_k fixed matrices. r = number of multiplications = the cost driver.
- The bilinear map <n,n,n>: K^{n×n}×K^{n×n}->K^{n×n} has a TENSOR t in K^{n^2 × n^2 × n^2}:
  entries t_{(i,j),(k,l),(p,q)} = δ_jk δ_lp δ_qi (each x_{ik} y_{kj} appears in f_{ij}).
  The 2×2 tensor lives in K^{4×4×4}.
- A "triad" / rank-1 tensor is u⊗v⊗w. Tensor RANK R(t) = min #triads summing to t.
  Theorem: R(t) = bilinear complexity = min #multiplications. So the question
  "min #mults for matrix mult" IS "rank of the matrix-mult tensor."
- Exponent ω = inf{β : R(<n,n,n>) ≤ O(n^β)}. ω = ω̃ (counting all ops). Strassen ⇒ ω ≤ log2 7.
- Tensor product of matrix tensors: <k,m,n> ⊗ <k',m',n'> = <kk',mm',nn'>. So
  R(<n,n,n>) ≤ R(<2,2,2>)^{log2 n} = 7^{log2 n} = n^{log2 7}. Recursion IS the tensor power.
- Submultiplicativity R(t⊗t') ≤ R(t)R(t') gives ω ≤ log_n R(<n,n,n>) for any single n. So
  any rank bound on a fixed-size tensor yields an exponent.

## Border rank (Bini et al. 1979) — Bläser §6
- Tensor rank is NOT semicontinuous: a tensor can be a limit of lower-rank tensors.
  Example: a0b0, a1b0+a0b1 has rank 3 but border rank 2:
  t(ε)=(1,ε)⊗(1,ε)⊗(0,1/ε)+(1,0)⊗(1,0)⊗(1,-1/ε) → t as ε→0, rank 2 ∀ε>0.
- Border rank R̲(t): min r with sum of r triads (over K[ε]) = ε^h t + O(ε^{h+1}).
- Bini's partial <2,2,2>-like (computing z11,z12,z21) has rank 6 but border rank 5
  (5 products p1..p5 with εz = combos + O(ε^2)). Two copies ⇒ approximate <2,2,3> with 10.
- Bini's theorem: border rank gives exponent too: R̲(<n,n,n>) ≤ r ⇒ ω ≤ log_n r.
  Cost of "undoing" the ε-approximation is only polynomial (log factors), absorbed in inf.
  Corollary 6.7: ω ≤ 2.78 from the <2,2,3> border-rank-10 construction.

## Schönhage τ-theorem + laser (high-level only) — Bläser §7-9
- Schönhage's asymptotic-sum / τ-theorem: if border rank of a DIRECT SUM
  ⊕_i <k_i,m_i,n_i> ≤ r with r > p (number of summands), then ω ≤ 3τ where
  Σ_i (k_i m_i n_i)^τ = r. Lets you combine several disjoint matrix products cheaply.
  Corollary 7.8: ω ≤ 2.55.
- Strassen's laser method (1986): start from a tensor that is "almost" a direct sum of
  matrix tensors; take a high tensor power; via monomial/combinatorial degeneration extract
  a large independent "diagonal" of disjoint matrix products; apply the τ-theorem.
  "Laser" = make many incoherent terms cohere (Shokrollahi anecdote, Bläser §8).
- Coppersmith–Winograd tensor + this machinery ⇒ ω < 2.376 (1990); modern refinements
  ω < 2.3719 (2024). These are the SEARCHED/optimized layer — gesture only; the *principled*
  leap is rank/border-rank + degeneration of the matrix-mult tensor.

## Design decisions → why
- Why cut multiplications not additions? Exponent = log2(#recursive subproblems); additions
  are O(n^2) glue at each level, dominated. log2 7 < log2 8 = 3.
- Why must the 7 identities avoid commuting entries? So they apply when entries are blocks
  (matrices don't commute) ⇒ recursion. Strassen's forms are bilinear (one A-form × one
  B-form each), so they do.
- Why does this stop at 7 (not 6)? R(<2,2,2>)=7 exactly (Winograd lower bound, Bläser
  Thm 3.1 / §4); 6 is impossible for exact rank, but border rank reaches lower for larger
  formats — motivates border rank.
- Why tensor framing? It makes "minimum #mults" a basis-free invariant (rank), explains the
  recursion as a tensor power, and unlocks border rank + degeneration as ways to lower the
  exponent that are invisible in the explicit-formula view.
- Practical: recurse only to a crossover size (leaf), then call optimized cubic kernel —
  the 18 adds + recursion overhead beat naive only for large n (~hundreds-thousands).
  Pad odd dimensions to even (or to a power of two). Numerical stability slightly worse
  than classical (more additions, cancellation), but acceptable.

## Canonical code (final form)
Recursive Strassen with crossover + power-of-two padding, grounded in hardianlawi structure,
float dtype, numpy for leaf. Splits into a,b,c,d / e,f,g,h; 7 products p1..p7; 4 quadrants.

## Source URLs read this run
- arXiv 1708.08083 (Ikenmeyer & Lysikov, conceptual/coordinate-free proof) — refs/strassen-conceptual.txt
- Bläser, Fast Matrix Multiplication, ToC Grad Surveys 5 (2013) theoryofcomputing.org/articles/gs005 — refs/blaser-survey.txt
- en.wikipedia.org/wiki/Strassen_algorithm (M1..M7, recovery, counts)
- github hardianlawi/algorithm-implementation strassenAlgorithm.py — code/
- Quanta 2024 + Wikipedia computational-complexity-of-matmul (laser/CW history, ω<2.3719)
