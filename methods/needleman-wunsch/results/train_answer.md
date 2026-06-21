We want a machine to tell us, objectively, whether two protein (or nucleotide) sequences are related, and to quantify *how well* one can be matched against the other once we allow residues to be inserted or deleted. The way it is done now is to write the two sequences out and stare at them: tedious, and the verdict ("these look homologous") is a feeling rather than a number. Two difficulties compound. A meaningful comparison must allow **gaps**, because real proteins diverge by insertions and deletions and not only by point substitutions; but the moment gaps are allowed, the number of ways to lay one sequence against the other explodes combinatorially. And even with a fixed layout we still need a single scalar score whose value we can test against chance. The existing tools each cover only part of this. Direct gap-free comparison — slide one sequence past the other and count residues that fall in register — is fast but breaks completely under a single indel, which throws the rest of both chains out of phase exactly when a homology test is most needed. Braunitzer-style gapped enumeration admits indels but, with no principled penalty and no pruning, "multiplies the number of comparisons" into junk: partial, unnecessary layouts, both combinatorially explosive and prone to over-fitting gaps to manufacture a match. And the statistical-comparison line (Fitch 1966; Needleman & Blair 1969) framed the *scoring* and *significance* of a comparison — residue-pair correspondence arrays, codon- or chemistry-weighted pair values, null distributions from shuffling — but left the *alignment* problem untouched: it never said how to choose, among exponentially many gapped layouts, the single best one and report it. That gap is what we close.

The first move is to make the object concrete. Put sequence $A$ down the rows and sequence $B$ across the columns of a grid; lattice point $(i,j)$ means we have consumed $A[1..i]$ and $B[1..j]$. Biology forces two constraints on any pairing of residues: each residue is one physical thing, so it is used in at most one pair, and the chains have a direction, so the pairing must preserve order. Those two constraints are exactly the statement that an alignment is a *monotone* staircase path across the grid — a diagonal step from $(i-1,j-1)$ to $(i,j)$ pairs $A_i$ with $B_j$, a down step advances in $A$ while $B$ waits (so $A_i$ sits opposite a gap in $B$), a right step advances in $B$ (so $B_j$ sits opposite a gap in $A$), and the path never goes back up or left. Every gapped alignment is one such path, and the "maximum match" is the highest-scoring path, where a diagonal step earns the value of its pair and a gap step costs a penalty. There are exponentially many monotone paths for sequences of length $\sim 150$, so enumeration is dead; the method must find the best path without listing paths. I propose the **Needleman–Wunsch algorithm**: solve this maximization by dynamic programming over the grid of sequence prefixes.

What makes it work is optimal substructure. Take the optimal full path and any lattice point $(i,j)$ it passes through; it splits into a head from the origin to $(i,j)$ and a tail from $(i,j)$ to the corner. The head must itself be the best possible path to $(i,j)$ — if some other head reached $(i,j)$ with a higher score we could splice it onto the same tail and beat the supposed optimum, a contradiction, since the tail only sees the score handed to it at $(i,j)$, not which head delivered it. So the optimum restricted to any prefix region is optimal for that prefix, and the natural subproblem is "best score of aligning a prefix of $A$ with a prefix of $B$." Let $F(i,j)$ be the best score of any alignment of the first $i$ residues of $A$ with the first $j$ residues of $B$. The key observation is that the very last column of such a prefix-alignment is exactly one of three shapes, mutually exclusive and exhaustive: it pairs $A_i$ with $B_j$ (preceded by the best alignment of $A[1..i-1]$ with $B[1..j-1]$, worth $F(i-1,j-1)$, plus the pair value $s(A_i,B_j)$); or it is a gap in $B$ with $A_i$ opposite it (preceded by $F(i-1,j)$, minus a gap penalty); or a gap in $A$ with $B_j$ opposite it (preceded by $F(i,j-1)$, minus a gap penalty). There is no fourth shape a column can take, so the best prefix-alignment takes whichever of the three is largest:

$$F(i,j) = \max\!\big(\, F(i-1,j-1) + s(A_i,B_j),\;\; F(i-1,j) - d,\;\; F(i,j-1) - d \,\big),$$

with $d$ the per-gap penalty. Each cell is computed from three smaller cells, turning the exponential path search into an $(n+1)\times(m+1)$ table filled in $O(nm)$ time with constant work per cell. The boundaries encode that leading gaps cost: $F(0,0)=0$ (two empty prefixes score nothing), and $F(i,0) = -i\,d$, $F(0,j) = -j\,d$, because aligning a prefix against an empty sequence forces every residue opposite a gap. Sweeping the interior in any order that has the three predecessors ready — row by row, left to right — leaves $F(n,m)$ at the far corner as the maximum match for the whole sequences. (If leading or trailing gaps should not be charged, because one sequence is a fragment, the corresponding borders are set to $0$; the strict global version charges them.)

Two design choices in that recurrence are load-bearing. First, the gap penalty must be *subtracted before the max is taken*, cell by cell, not tallied at the end: it has to actually deter the optimizer as the path is being chosen, since the penalty is a barrier — a gap is worth taking only when the matches it unlocks are worth more than $d$. With $d=0$ gaps are free and the optimizer slides every coincidental match into register, giving unrelated sequences a spurious score; with $d\to\infty$ gaps are forbidden and the method collapses to pure diagonal frame-shift comparison; the interesting regime is in between, and $d$ is a knob to sweep. Second, this three-cell form is the *efficient* form of an equivalent but slower idea. If instead one scans, for each cell, every later cell in the next row or column to find the best gap-predecessor, the cost is $O(nm(n+m))$. But when the gap penalty is linear — $d$ per skipped residue — a long gap is just a run of one-residue gaps and its cost is additive, so the best cell two or more steps away in a row is already folded into the immediate neighbor's value recursively. We never need to look past one cell, and the row/column scan collapses exactly back to the three-term recurrence. The forward three-cell version and the backward scan compute the identical optimum under linear gaps; we keep the $O(nm)$ form.

The score is only half of what we want; we also need the alignment itself. This falls out for free: while filling $F(i,j)$, remember which of the three terms won — diagonal, up, or left — as a pointer to the predecessor the optimum came through. After the table is full, start at $(n,m)$ and follow pointers back to $(0,0)$, emitting a paired column for each diagonal pointer, $A_i$ over a gap for each up, a gap over $B_j$ for each left; reverse the emitted columns and we have the two aligned strings whose score is exactly $F(n,m)$. Ties among the three terms mean co-optimal alignments, and one can follow either branch or enumerate them all. A second separation is just as clean: the recurrence never inspects what $s$ is, only that it returns a number per pair. The crude version scores $1$ for identical residues and $0$ otherwise, so $F$ literally counts matched residues; but identity throws away that some mismatches are near-misses. Two amino acids whose codons differ in only one of three bases are a single point mutation apart, while ones sharing no codon bases are far apart, so $s$ can be graded by codon-base correspondence (Marshall, Caskey & Nirenberg) or read off an empirical interchange table like Dayhoff's *Atlas* counts. All of that biology lives inside $s(\cdot,\cdot)$; the dynamic-programming machinery is untouched — scoring policy and optimization are decoupled.

A flat per-residue gap cost is biologically crude, because a deletion of five residues is *one* evolutionary event, not five, yet a linear penalty charges one five-residue gap and five scattered one-residue gaps the same total. What we want is for a gap of length $g$ to cost $\text{open} + (g-1)\cdot\text{extend}$: a steep one-time charge to *open* a gap and a gentle charge to *extend* it, so one long indel beats many short ones. The single number $F(i,j)$ cannot support this, because it has no memory of whether we are currently inside a gap, and that memory is exactly what decides open-versus-extend. So we split the state into three tables: $M(i,j)$ for the best prefix-alignment ending in a match/mismatch column, $X(i,j)$ for one ending with $A_i$ opposite a gap in $B$, and $Y(i,j)$ for one ending with $B_j$ opposite a gap in $A$. Now the penalty depends on which table we came from:

$$M(i,j) = s(A_i,B_j) + \max\!\big( M(i-1,j-1),\, X(i-1,j-1),\, Y(i-1,j-1) \big),$$
$$X(i,j) = \max\!\big( M(i-1,j) - \text{open},\; Y(i-1,j) - \text{open},\; X(i-1,j) - \text{extend} \big),$$
$$Y(i,j) = \max\!\big( M(i,j-1) - \text{open},\; X(i,j-1) - \text{open},\; Y(i,j-1) - \text{extend} \big).$$

Entering $X$ or $Y$ from a non-gap state means opening a new gap and paying $\text{open}$; staying within the same table means extending and paying the cheaper $\text{extend}$. The best prefix score is $\max(M,X,Y)$ at $(i,j)$, and the answer is that max at $(n,m)$ — three tables instead of one, but still constant work per cell, so still $O(nm)$, with biologically saner gaps. The same lattice read as a *minimization* — match $0$, mismatch and gap $1$, with $F(i,0)=i$ and $F(0,j)=j$ — is exactly Levenshtein edit distance, confirming that the grid, not the choice of objective, is the right object. And adding a fourth option, $H(i,j)=\max(0,\dots)$, lets any cell reset to "start fresh here" so the alignment can begin and end anywhere — local alignment, with the answer the largest cell anywhere in the table and traceback running from that peak back to the first $0$. The question we started from — are these two whole proteins related — is global, so the global linear-gap version is the one to write first.

Here is the core, the global linear-gap aligner: one score table, one traceback table, the three-term fill with leading-gap boundaries, then a walk back from the corner.

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

And the affine-gap variant, three tables, one $O(nm)$ pass, when one long indel should be cheaper than many short ones:

```python
NEG = float('-inf')

def align_affine(seqA, seqB, gap_open=2.0, gap_extend=0.5, score=score_pair):
    n, m = len(seqA), len(seqB)
    M  = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in match/mismatch
    Ix = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends with A aligned to a gap
    Iy = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends with B aligned to a gap
    M[0][0] = 0.0
    for i in range(1, n + 1):
        Ix[i][0] = -gap_open - (i - 1) * gap_extend
    for j in range(1, m + 1):
        Iy[0][j] = -gap_open - (j - 1) * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = score(seqA[i-1], seqB[j-1])
            M[i][j]  = s + max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1])
            Ix[i][j] = max(M[i-1][j] - gap_open,
                           Iy[i-1][j] - gap_open,
                           Ix[i-1][j] - gap_extend)
            Iy[i][j] = max(M[i][j-1] - gap_open,
                           Ix[i][j-1] - gap_open,
                           Iy[i][j-1] - gap_extend)
    return max(M[n][m], Ix[n][m], Iy[n][m])
```
