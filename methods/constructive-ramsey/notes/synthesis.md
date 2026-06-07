# Synthesis — Explicit Ramsey graphs via Frankl–Wilson

## The task (research question)
Build EXPLICITLY (deterministic, no coin flips) a graph on N vertices with
ω(G), α(G) ≤ polylog/quasipolynomial in N. Trivial Ramsey gives only
ω or α ≥ (1/2) log₂ N for every graph; the probabilistic method (Erdős 1947)
proves graphs with ω, α ≤ 2 log₂ N EXIST but gives no construction. Goal: a
hands-on rule for adjacency that provably kills both large cliques and large
independent sets, approaching the existence bound.

## Pain points / why hard
- Random graph WORKS but is non-explicit; derandomizing is the whole difficulty.
- Naive explicit attempts (Cayley/algebraic graphs, Paley graphs) are conjectured
  good but UNPROVEN; no proof bounds their clique/indep simultaneously.
- The obstruction is symmetry: to bound BOTH ω and α you need a structural
  certificate that works for the graph AND its complement.

## The key sources (verified this run)
- Frankl & Wilson, "Intersection theorems with geometric consequences",
  Combinatorica 1 (1981) 357–368. The modular intersection theorem + Ramsey app.
- Gopalan, "Constructing Ramsey Graphs from Boolean Function Representations"
  (refs/gopalan_ramsey.txt): states FW construction p prime, m=p³, vertices =
  (p²-1)-subsets of [m], adjacent iff |A∩B| ≡ -1 (mod p); bound via extremal
  set theory; degree p-1 polynomials.
- "Ramsey properties of algebraic graphs" arXiv:2103.05618 (refs/algebraic-graphs.txt,
  lines 66-78): AUTHORITATIVE clean statement —
  vertices = (p²-1)-subsets of [n], edge iff |A∩B| ≡ -1 (mod p),
  N = binom(n, p²-1), no clique or independent set larger than binom(n, p-1)
  = O_p(N^{1/(p+1)}); choosing n = p³ gives clique/indep size 2^{O(√(log N log log N))}.
- Gil Kalai blog (verified): the polynomial/dimension proof (geometric {-1,1}^n
  version): P_y(x) = ∏_{k=1}^{p-1}(⟨x,y⟩ - k) over F_p; multilinearize; the P̄_y are
  linearly independent in the space of multilinear polys of degree ≤ p-1, dim =
  Σ_{i=0}^{p-1} binom(n,i). Standard triangular/independence argument.
- BRSW (refs/brsw.txt): Barak–Rao–Shaltiel–Wigderson, Annals 2012, "2-source
  dispersers for n^{o(1)} entropy and Ramsey graphs beating the Frankl–Wilson
  construction" — extractor/disperser machinery, gets 2^{n^{o(1)}} = far below
  FW's 2^{Õ(√n)}. CONTEXT ONLY (later work; do not put in reasoning as posterior).
- Barak (refs/grolmusz.txt = arXiv math/0601651): simple Abbott-product construction
  matching FW bound, "only explicit not very explicit". Context.

## The construction (final landing)
Fix prime p. Ground set [n]. Vertices V = all (p²-1)-element subsets of [n].
Edge {A,B} iff |A∩B| ≡ -1 (mod p) (equivalently ≡ p-1). N = binom(n, p²-1).

WHY p²-1 as the set size: we want set size k ≡ -1 (mod p) so that the
"diagonal" |A∩A| = k ≡ -1, and we want one extra factor of p of "room"
so the set size is p·(p-1)+... Actually k = p²-1 = (p-1)(p+1); k ≡ -1 mod p.
Need k large enough that residue classes mod p genuinely constrain
intersections AND that the family is huge. p²-1 is the smallest natural size
≡ -1 mod p that is Θ(p²); with n=p³ ground set you get N ≈ exp around p² log p.

## The two-sided bound — the heart
Let G be the graph.

CLIQUE: a clique is a family F of (p²-1)-sets with EVERY pairwise
|A∩B| ≡ -1 (mod p). So all pairwise intersections lie in the single residue
class L = {-1} mod p, while the set size k = p²-1 ≡ -1 ∉ {0,...,p-2}.
Hmm: the relevant theorem for CLIQUE (all intersections ≡ -1, same as k mod p):
use that |A∩B| ≡ -1 mod p with |A∩B| ∈ {0,...,p²-2} (proper subsets), so the
ACTUAL integer values are in {p-1, 2p-1, ..., }. By Ray–Chaudhuri–Wilson
(non-modular L-intersecting, |L| distinct sizes) → bound binom(n,|L|). The set of
attainable integer intersection sizes ≡ -1 mod p, below p²-1, is
{p-1, 2p-1, ..., p²-p-1} which has p-1 values → binom(n, p-1).
So clique ≤ binom(n, p-1). ✓ (|L| = p-1.)

INDEPENDENT SET: family with NO edge, i.e. all pairwise |A∩B| ≢ -1 (mod p),
i.e. residues in {0,1,...,p-2} (that's p-1 forbidden... allowed residues), and
k = p²-1 ≡ -1 ∉ {0,...,p-2}. This is exactly the MODULAR Frankl–Wilson:
k-uniform family, s = p-1 residues μ_1,...,μ_{p-1} all ≢ k (mod p), pairwise
intersections hit one of them → |F| ≤ binom(n, s) = binom(n, p-1). ✓

Both sides → binom(n, p-1). THE SAME number p-1 appears on both sides — that's
the magic. Modulus p gives p-1 nonzero/forbidden residues either way.

## Polynomial/dimension proof (modular FW, the indep-set side; clique side
   is RCW, structurally identical with integer evaluation)
For each set A ∈ F (independent set) with char vector v_A ∈ {0,1}^n, define
  P_A(x) = ∏_{μ ∈ {0,...,p-2}} (⟨x, v_A⟩ - μ)  over F_p,  x ∈ {0,1}^n.
Degree p-1. For B ≠ A in F: ⟨v_B, v_A⟩ = |A∩B| ≡ some μ ∈ {0,...,p-2} → P_A(v_B)=0.
For B = A: ⟨v_A,v_A⟩ = p²-1 ≡ -1 (mod p), and -1 ∉ {0,...,p-2}, so
  P_A(v_A) = ∏(-1 - μ) = ∏_{μ=0}^{p-2}(-(1+μ)) = (-1)^{p-1}(p-1)! ≠ 0 (mod p).
Multilinearize (x_i² → x_i since x∈{0,1}) → P̄_A, still degree ≤ p-1, same values
on {0,1}^n. The P̄_A are linearly independent: Σ λ_A P̄_A = 0, evaluate at v_B →
λ_B P̄_B(v_B) = 0 → λ_B = 0. They live in the space of multilinear polynomials of
degree ≤ p-1 in n vars, dimension Σ_{i=0}^{p-1} binom(n,i) ≤ (about) binom(n,p-1)·const.
Hence |F| ≤ Σ_{i=0}^{p-1} binom(n,i). The leading term binom(n,p-1) dominates;
quoted bound binom(n,p-1).

## Why the SAME proof handles complement (the deep point)
Random graph: bounding ω is easy (count), bounding α needs the SAME structure on
complement — a random graph has it by symmetry but you can't *certify* it. FW's
certificate is the linear-algebra dimension bound, and it is INSENSITIVE to which
residue class you forbid: cliques forbid all-but-{-1}, indep sets forbid {-1}, and
mod p both give exactly p-1 constraint values → both get binom(n,p-1). The
algebra is the derandomization.

## Parameter calculation (n = p³)
N = binom(p³, p²-1).  log₂ N ≈ (p²-1) log₂(p³/(p²-1)) ≈ p² · log₂ p (roughly p² log p).
Bound binom(n,p-1) = binom(p³, p-1) ≈ (p³)^{p-1}/(p-1)! ; log₂ ≈ (p-1)·3 log₂ p
≈ 3 p log₂ p.
Let M = bound = 2^{Θ(p log p)}, and log₂ N = Θ(p² log p).
So p ≈ √(log N / log log N) roughly, and log₂ M = Θ(p log p)
= Θ(√(log N · log log N)).  ⇒ ω,α ≤ 2^{O(√(log N log log N))}.
This is quasipolynomial-beating-trivial but still exp(√) above the existence
2 log₂ N. Equivalently R(t) ≥ t^{Ω(log t / log log t)} as a Ramsey lower bound.

## Code (grounded; build the graph)
Real, runnable: enumerate (p²-1)-subsets of [n] (for tiny p, e.g. p=2,3 it's
feasible; p=2 → set size 3, n=8; p=3 → set size 8, n=27 is already huge so use
small n for demo), build adjacency by intersection mod p, then brute-force verify
ω, α ≤ binom(n,p-1). Use itertools.combinations. This mirrors how the construction
is checked computationally. For p=2: vertices = 3-subsets, edge iff |A∩B| ≡ 1 (mod 2),
i.e. |A∩B| odd.

## Uncertainty flags
- Exact constant in 2^{O(√(log N log log N))} not pinned; sourced as O(·) form
  from arXiv:2103.05618. Fine.
- Clique side: I derive |L|=p-1 (integer intersection sizes ≡ -1 mod p below p²-1).
  Authoritative source states both ω,α ≤ binom(n,p-1); consistent.
- FW 1981 original uses set size = p²-1 with m = p³ per Gopalan; the general-n
  statement (binom(n,p-1), N=binom(n,p²-1)) is from arXiv:2103.05618.
