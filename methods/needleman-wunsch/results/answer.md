# The Needleman–Wunsch algorithm: global pairwise sequence alignment by dynamic programming

## Problem

Given two sequences (amino-acid or nucleotide), find the highest-scoring global alignment — a way of pairing residues that preserves order and may insert gaps in either sequence — and report both the optimal score (the "maximum match") and the alignment that attains it. Order-preserving gapped alignments are exponentially many, so they cannot be enumerated.

## Key idea

Lay the two sequences on the axes of a grid; any gapped alignment is a monotone path of diagonal (pair), down (gap in one sequence) and right (gap in the other) steps. The optimal path restricted to any prefix is optimal for that prefix (Bellman optimal substructure), and the last column of an alignment of two prefixes is exactly one of three shapes — match/mismatch, gap-in-A, gap-in-B. This yields a recurrence that fills an (n+1)×(m+1) table in O(nm) time and O(nm) space, after which a traceback recovers the alignment.

## Algorithm

Let `s(a,b)` score a residue pair (e.g. identity 1/0, or a codon-/substitution-matrix value) and `d` be the per-gap penalty.

Recurrence (linear gap):

  F(i,j) = max( F(i−1,j−1) + s(aᵢ,bⱼ),  F(i−1,j) − d,  F(i,j−1) − d )

Boundary (global alignment charges leading gaps):

  F(0,0) = 0,  F(i,0) = −i·d,  F(0,j) = −j·d

The maximum match is F(n,m). Record, per cell, which term won; traceback from (n,m) to (0,0) emits the alignment columns (diagonal → paired, down → gap opposite aᵢ, right → gap opposite bⱼ), reversed.

Variants on the same lattice:
- **Affine gaps (Gotoh):** a length-g gap costs `open + (g−1)·extend`, so one long indel beats many short ones. Keep three tables — M (ends in match), Iₓ (ends in gap consuming A), I_y (ends in gap consuming B):

  M(i,j)  = s(aᵢ,bⱼ) + max( M(i−1,j−1), Iₓ(i−1,j−1), I_y(i−1,j−1) )
  Iₓ(i,j) = max( M(i−1,j) − open,  Iₓ(i−1,j) − extend )
  I_y(i,j) = max( M(i,j−1) − open,  I_y(i,j−1) − extend )

  answer = max(M,Iₓ,I_y) at (n,m); still O(nm).
- **Edit distance (Levenshtein):** the minimization dual — score match 0, mismatch/gap 1, minimize; `F(i,0)=i`, `F(0,j)=j`.
- **Local alignment (Smith–Waterman):** add a `0` option, `H(i,j)=max(0, …)`, init first row/col to 0, traceback from the global-max cell to the first 0.

## Code

A clean global aligner, structured like the canonical dynamic-programming aligner (score matrix + traceback matrix, boundary init, three-term fill, backtrack):

```python
import numpy as np

GAP = 1.0  # per-gap penalty d

def score_pair(a, b, match=1.0, mismatch=0.0):
    """Per-pair score s(a,b). Swap in a codon-/Dayhoff-weighted matrix freely;
    the DP never inspects this function."""
    return match if a == b else mismatch

DIAG, UP, LEFT = (1, 1), (1, 0), (0, 1)

def build_table(seqA, seqB, gap=GAP, score=score_pair):
    n, m = len(seqA), len(seqB)
    F = np.zeros((n + 1, m + 1))                 # F[i][j]: best score for A[:i], B[:j]
    ptr = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):                    # leading gaps cost one penalty each
        F[i][0] = -i * gap; ptr[i][0] = UP
    for j in range(1, m + 1):
        F[0][j] = -j * gap; ptr[0][j] = LEFT
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = F[i-1][j-1] + score(seqA[i-1], seqB[j-1])
            up   = F[i-1][j]   - gap             # gap in B (consume A[i-1])
            left = F[i][j-1]   - gap             # gap in A (consume B[j-1])
            best = max(diag, up, left)
            F[i][j] = best
            ptr[i][j] = DIAG if best == diag else (UP if best == up else LEFT)
    return F, ptr

def traceback(seqA, seqB, ptr):
    i, j = len(seqA), len(seqB)
    a_out, b_out = [], []
    while i > 0 or j > 0:
        di, dj = ptr[i][j]
        if (di, dj) == DIAG:
            a_out.append(seqA[i-1]); b_out.append(seqB[j-1])
        elif (di, dj) == UP:
            a_out.append(seqA[i-1]); b_out.append('-')
        else:
            a_out.append('-');       b_out.append(seqB[j-1])
        i, j = i - di, j - dj
    return ''.join(reversed(a_out)), ''.join(reversed(b_out))

def align(seqA, seqB, gap=GAP, score=score_pair):
    F, ptr = build_table(seqA, seqB, gap, score)
    a_aln, b_aln = traceback(seqA, seqB, ptr)
    return F[len(seqA)][len(seqB)], a_aln, b_aln   # maximum match + alignment

# >>> align("GATTACA", "GCATGCU")
# crude identity scoring; for real use, pass a substitution-matrix `score`.
```

Affine-gap variant (Gotoh), three tables, one O(nm) pass:

```python
NEG = float('-inf')

def align_affine(seqA, seqB, gap_open=2.0, gap_extend=0.5, score=score_pair):
    n, m = len(seqA), len(seqB)
    M  = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in match/mismatch
    Ix = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in gap consuming A
    Iy = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in gap consuming B
    M[0][0] = 0.0
    for i in range(1, n + 1):
        Ix[i][0] = -gap_open - (i - 1) * gap_extend
    for j in range(1, m + 1):
        Iy[0][j] = -gap_open - (j - 1) * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = score(seqA[i-1], seqB[j-1])
            M[i][j]  = s + max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1])
            Ix[i][j] = max(M[i-1][j] - gap_open, Ix[i-1][j] - gap_extend)
            Iy[i][j] = max(M[i][j-1] - gap_open, Iy[i][j-1] - gap_extend)
    return max(M[n][m], Ix[n][m], Iy[n][m])
```

The widely used library form is `Bio.pairwise2` / `Bio.Align.PairwiseAligner` in Biopython, whose generic global routine implements exactly this score-plus-traceback table fill (with general gap functions and optional non-penalized end gaps); the code above mirrors that structure.

## Complexity and properties

- Time and space O(nm); the optimum is exact (no heuristic).
- The optimization is independent of the scoring policy: identity, codon-correspondence, or empirical substitution matrices (the PAM/BLOSUM lineage) all drop into `s(·,·)`.
- The gap penalty `d` is a barrier: `d→∞` forbids gaps (pure diagonal comparison), `d=0` makes gaps free; affine `open/extend` models indels as single multi-residue events.
- Significance of a maximum match is assessed against scores of randomly shuffled sequences (composition preserved), summarized as a z-score.
