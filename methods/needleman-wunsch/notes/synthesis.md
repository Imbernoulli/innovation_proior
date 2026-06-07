# Needleman–Wunsch — synthesis (pre-Phase-2)

## Primary source (read in full, refs/needleman-wunsch-1970.pdf)
Needleman & Wunsch 1970, J. Mol. Biol. 48:443–453. Read all 12 pages.
- Problem: are two proteins homologous? Visual comparison tedious + subjective; need a *computer-adaptable*, *objective* "maximum match" = largest number of amino acids of one protein matchable to the other allowing for all possible interruptions (gaps) in either sequence.
- Pain: allowing gaps (Braunitzer 1965) multiplies the number of comparisons enormously and "introduces unnecessary and partial comparisons." Number of gapped alignments is huge.
- Representation: 2-D array MAT, rows = B residues (i), cols = A residues (j); cell MATij = pair (Aj,Bi). Every comparison = a *pathway* through the array. A residue can occur only once → if MATmn is on a pathway through MATij then indices satisfy m>i,n>j OR m<i,n<j (monotone). "Necessary pathway" starts at first row/col, each step either i or j (or both) increases, one of them by exactly 1. Complete diagonals = no gaps. i−m ≠ j−n ⇒ a gap.
- Cell value in simplest method: 1 if Aj==Bi (identical), 0 otherwise. Sophistication: weight by genetic-code base correspondence (type 3/2/1/0 pairs; type3 identical=1.0, type0=0).
- Gap penalty: a "penalty factor" subtracted for each gap, can be a function of size and/or direction; assessed *before* the max-match is formed; acts as a barrier — gap allowed only if benefit exceeds barrier. If penalty=0, all gaps allowed → "simple frame-shift method" (sum the diagonals).
- THE RECURRENCE (their words, p.444–445), worked BACKWARD from terminals (i=y,j=z) toward origins: for each cell, add to its own value "the maximum value from among all the cells which lie on a pathway to it." A cell at (i,j) reaches forward to (i+1,j+1) [the diagonal, no gap] and to every cell further along row i+1 / column j+1 [a gap]. Repeat decrementing i,j until whole matrix processed. Then each border cell holds the maximum match for a pathway originating there. Max-match = largest number in first row or first column. Traceback: record the origin of the number added to each cell.
- Worked example Fig 1 & 2: sequences (col) A B C N J R O C L C R P M, (row) A J C J N R C K C R B P; max match = 8.
- Sec 3 significance (random shuffles), Sec 4 weighting, Sec 5–6 experiments + Tables 1,2 + Fig 3 → PROPOSED-METHOD RESULTS, out of scope.

Note: the 1970 formulation, scanning the whole next row+column per cell, is O(n^3)-ish. The modern equivalent (and what canonical code does) replaces "max over the whole row/column" with the local 3-neighbour recurrence F(i,j)=max(diag+s, up−d, left−d) — exact equivalence for linear (constant) gap penalty, because the best gapped predecessor is reachable one step at a time. reasoning.md should derive the 1970 backward formulation, then realize the linear-gap case collapses to the 3-cell forward recurrence; affine gaps (Gotoh) recover the need to look back but with 3 matrices in O(nm).

## Background / ancestors (context.md Background + Baselines)
- **Bellman dynamic programming (1950s)**: principle of optimality — an optimal policy's tail is optimal for the tail subproblem. Optimal substructure + overlapping subproblems → tabulate. The general hammer NW is an instance of.
- **Levenshtein 1966 (Soviet Physics Doklady 10:707–710)** "Binary codes capable of correcting deletions, insertions and reversals": defines edit distance (min # of single-char insert/delete/substitute), proven a metric. DP recurrence (verified, Wikipedia): lev(i,0)=i, lev(0,j)=j, lev(i,j)=min(lev(i−1,j)+1, lev(i,j−1)+1, lev(i−1,j−1)+[a_i≠b_j]). O(nm) table. This is the MINIMIZATION dual of NW's MAXIMIZATION; same lattice. NW (1970) predates the wide CS awareness of this; they discovered the alignment DP independently in the biology setting.
- **Fitch 1966 (J. Mol. Biol. 16:9)** + **Needleman & Blair 1969**: earlier computer-based statistical comparison of sequences; Fitch's amino-acid correspondence matrix is reused by NW. Limitation: didn't optimally handle gaps/alignment.
- **Braunitzer 1965**: allowing gaps in comparison → blows up the comparison count, "unnecessary and partial comparisons." Motivates needing an *efficient* method that excludes useless pathways.
- **Substitution scoring**: Eck & Dayhoff 1966 Atlas (later PAM); Marshall/Caskey/Nirenberg 1967 codon table → NW's codon-correspondence weighting. (BLOSUM is later; mention PAM lineage only as the scoring-matrix idea.)
- **Margoliash/Zuckerkandl & Pauling**: molecular-clock / evolutionary-distance context.

## Descendants for context only? NO — Smith-Waterman 1981 & Gotoh 1982 are POSTERIOR to NW; they belong in reasoning.md as the narrator's own forward extensions (affine gaps anticipated by NW's "penalty as a function of size"), NOT in context.md. Keep context strictly pre-1970-method.

## Verified recurrences (all from retrieved sources this run)
- Linear NW (Biopython _make_score_matrix_generic + UCSC notes): F(i,j)=max(F(i−1,j−1)+s(a_i,b_j), F(i−1,j)−d, F(i,j−1)−d); F(0,0)=0, F(i,0)=−i·d, F(0,j)=−j·d. O(nm). Traceback from (n,m) to (0,0).
- Smith–Waterman local (Wikipedia, verified): H(i,j)=max(0, H(i−1,j−1)+s, H(i−1,j)−d, H(i,j−1)−d); first row/col = 0; traceback from global max cell, stop at 0.
- Gotoh affine (UCSC bme205 notes, verified): three states. M(i,j)=s(a_i,b_j)+max(M,Ix,Iy at (i−1,j−1)). Ix(i,j)=max(M(i−1,j)−open, Ix(i−1,j)−extend [, Iy(i−1,j)−double]). Iy(i,j)=max(M(i,j−1)−open, Iy(i,j−1)−extend [, Ix(i,j−1)−double]). γ(g)=−open−(g−1)·extend. O(nm). Distinguishes opening vs extending a gap, so a single long gap is cheaper than many short ones.

## Canonical code (code/biopython_pairwise2.py, fetched from biopython master)
`_make_score_matrix_generic`: builds score_matrix (lenA+1)×(lenB+1); init first row/col with gap scores; for each (row,col): nogap=diag+match_fn; row/col gap = max over the row/column with gap_fn (general gap function, slow, NW-faithful); cell=max of all; trace_matrix encodes which moves were optimal (binary). `_recover_alignments` does traceback. For our deliverable, write a clean linear-gap NW (and an affine variant) mirroring this structure: score+traceback matrices, init, fill, backtrack, plus a global-alignment driver. Faithful, not invented.

## Design-decision → why
- Score = max over THREE predecessors: a column in the alignment is exactly one of {match/mismatch (diag), gap in B = consume a_i only (up), gap in A = consume b_j only (left)}; these are mutually exclusive and exhaustive ways the last column can look → max of 3.
- Boundary F(i,0)=−i·d: aligning prefix of A against empty prefix of B forces i leading gaps.
- Why DP works = optimal substructure: best alignment of (a_1..i, b_1..j) ends in one of the 3 column types; chop the last column → remaining is the best alignment of the shorter prefixes (else swap it for a better one and beat the optimum → contradiction). Overlapping subproblems → tabulate, O(nm) not exponential.
- Linear vs affine: constant per-residue penalty makes one long gap cost the same as scattered gaps; biology says a single multi-residue indel is one event → affine charges a big open + small extend. Needs Ix/Iy memory of "am I already in a gap," hence 3 matrices (Gotoh), still O(nm).
- Maximization (similarity) vs minimization (edit distance): same lattice, dual objective; NW maximizes match score, Levenshtein minimizes edits.
