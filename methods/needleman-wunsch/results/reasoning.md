OK, let me think this through from scratch. I have two protein sequences and I want a machine to tell me, objectively, whether they are related — and how well one can be matched to the other. Right now the way this is done is by writing the two sequences out and staring at them. That is slow, and worse, the verdict is a feeling. I want a number.

What does "matched" even mean? The smallest honest unit is a pair: one residue from sequence A lined up against one residue from sequence B. So a comparison of the whole sequences is a way of pairing up residues — some of A's residues sit opposite some of B's. Two constraints are forced on me by biology. A residue is one physical thing, so it can be used in at most one pair. And the chains have a direction (N to C terminus); a pairing that reorders residues is meaningless, so the pairing must preserve order — if Aⱼ pairs with Bᵢ and a later Aⱼ′ (j′>j) pairs with Bᵢ′, then I need i′>i too. Pairs that aren't forced together get skipped on one side or the other; that skip is a gap.

Let me draw this. Put sequence A along the columns and sequence B along the rows of a grid; cell (i,j) stands for the pair (Bᵢ, Aⱼ). An alignment is then a path of cells through this grid: a diagonal step from (i,j) to (i+1,j+1) means "pair Bᵢ₊₁ with Aⱼ₊₁," a step straight down means "advance in B but skip in A" (a gap in A), a step straight right means "advance in A, skip in B" (a gap in B). Order-preservation is exactly the statement that the path is monotone — it never goes back up or left. So: every gapped alignment of the two sequences ↔ one monotone staircase path across the grid. Good, that's clean.

Now I want the *best* path. Score a path by what its steps are worth: a diagonal step through (i,j) earns the value of that pair — call it 1 if Bᵢ and Aⱼ are the same residue and 0 if not, or later something more graded — and a gap step costs me a penalty. The "maximum match" is the highest-scoring path. So the whole problem is: maximize a sum over monotone paths.

Here's where it looks hopeless. How many monotone paths are there? At each cell I can in principle go diagonal, or skip any number of residues on either side before resuming. The count of order-preserving gapped alignments of two sequences of length ~150 is astronomical — it grows exponentially. I cannot enumerate paths and score each. Braunitzer's gapped comparisons already ran into this: the moment you allow gaps freely you "multiply the number of comparisons" and most of them are junk — partial, unnecessary layouts that no sensible alignment would use. So brute force is dead. I need structure.

Let me stare at a single best path and ask what it's made of. Suppose I knew the optimal path from the start corner all the way to the far corner. Pick any cell it passes through, say (i,j). The path splits there into a head (start → (i,j)) and a tail ((i,j) → end). Now — is the head necessarily the *best* possible path from start to (i,j)? It must be. If some other head reached (i,j) with a higher score, I could splice it onto the same tail and get a better full path, contradicting that I started with the optimum. The tail can't tell which head delivered it to (i,j); it only sees the score handed over. That's the whole trick: the optimal path's restriction to any prefix region is itself optimal for that prefix. This is exactly the kind of structure Bellman exploits — optimal substructure. The best solution decomposes into best solutions of subproblems, and the subproblems are "best score of comparing some prefix of A to some prefix of B," which is a small grid of cells. They overlap massively (the prefix (i,j) feeds into (i+1,j+1), (i+2,j+1), …), so instead of recomputing I should tabulate.

So let me define a table. Let F(i,j) be the best score of any path that compares the first i residues of B with the first j residues of A — the best alignment of the two prefixes. If I can write F(i,j) in terms of *smaller* F's, I fill the whole table once and I'm done, no enumeration.

Think about the very last thing the best prefix-alignment does — the final column of the alignment of B[1..i] with A[1..j]. There are only three things that last column can be. Either it pairs Bᵢ with Aⱼ (a diagonal step) — then before it I had the best alignment of B[1..i−1] with A[1..j−1], worth F(i−1,j−1), and I add the pair's value s(Bᵢ,Aⱼ). Or the last column is a gap in A: Bᵢ is opposite a gap, Aⱼ wasn't consumed yet — then before it I had B[1..i−1] vs A[1..j], worth F(i−1,j), and I pay a gap penalty. Or symmetrically a gap in B: Aⱼ opposite a gap, worth F(i,j−1) minus a gap penalty. Those three are mutually exclusive (the last column is exactly one shape) and exhaustive (there's no fourth shape a column can take). The best prefix-alignment must use whichever of the three is largest:

F(i,j) = max( F(i−1,j−1) + s(Bᵢ,Aⱼ),  F(i−1,j) − d,  F(i,j−1) − d )

with d the per-gap penalty. That's it — each cell from three smaller cells. I've turned an exponential search into a grid I fill in O(n·m): n·m cells, constant work each.

I need the edges. F(0,0) = 0: comparing two empty prefixes scores nothing. F(i,0): comparing the first i residues of B against an empty A — every residue of B has to sit opposite a gap, i gaps, so F(i,0) = −i·d. Likewise F(0,j) = −j·d. These boundary rows encode "leading gaps cost." (If I decide leading/trailing gaps shouldn't be charged — say one sequence is a fragment — I'd set those borders to 0 instead, but the honest global version charges them.) Then I sweep the interior in any order that has the three predecessors ready — row by row, left to right works — and F(n,m) at the far corner is the maximum match for the whole sequences.

Let me sanity-check this against how I'd compute it by hand the other way, because that's actually how I first reasoned it and I want to be sure they agree. Start at the terminal corner instead and walk *backward* toward the origin. Sit on a cell and ask: what's the best score of a path that *starts* here and runs to the end? A path leaving (i,j) either steps diagonally to (i+1,j+1) and continues optimally from there, or it takes a gap — and a gap can be long, skipping several residues before it resumes, so in full generality the next "real" cell could be any cell further along the row i+1 or the column j+1. So I'd write: value(i,j) = s(Bᵢ,Aⱼ) + max over all cells reachable forward — the single diagonal neighbour, or any cell along the next row or next column with a gap penalty subtracted. Fill the array from the far corner back; the maximum match is the largest number appearing in the first row or first column, since a path can originate at any N-terminal offset. This backward "add to each cell the maximum value among all cells on a pathway to it" is the same optimization — just oriented from the terminals — and it's exactly what I'd code first.

But wait — staring at "any cell along the next row or column" bothers me. For each cell I'm scanning a whole row and a whole column to find the best gap predecessor. That's O(n) extra work per cell, so O(n²·m) overall. Can I avoid re-scanning? Here's the thing: if the gap penalty is simply d per skipped residue — a flat, constant cost — then a long gap is just a sequence of one-residue gaps, and its cost is additive. The best way to "skip three residues then resume" scores the same whether I treat it as one jump of three or as three consecutive single steps. So I never need to look further than one cell. The "best cell two or more away in the row" is already folded into the immediate neighbour's value, recursively. That collapses the row/column scan down to a single left-neighbour and a single up-neighbour — and I'm right back at the three-term recurrence F(i,j) = max(diag+s, up−d, left−d). The forward three-cell version and the backward scan-the-row version compute the identical optimum; the three-cell one is just the efficient form, valid precisely because a constant gap penalty is additive over gap length. Good — they agree, and I'll keep the O(nm) three-cell form.

Let me make sure I haven't fooled myself on the gap penalty's role. Why penalize at all? If d = 0, gaps are free, and the max will happily insert gaps everywhere to slide every coincidental match into register — two unrelated sequences get a spuriously high score. The penalty is a barrier: a gap is taken only when the matches it unlocks are worth more than d. Set d enormous and you forbid gaps entirely, recovering pure diagonal (frame-shift) comparison; set d to zero and you get the lax free-gap comparison; the interesting regime is in between, and d is a knob I'll sweep. And the penalty has to be subtracted *before* the max is taken — it has to actually deter the optimizer cell by cell, not be tallied afterward, or it changes which path wins.

Now the score is only half of what I want — I also need the *alignment itself*, not just its number. The fix falls right out of the recurrence: while filling F(i,j), remember which of the three terms won — diagonal, up, or left. That's the pointer back to the predecessor cell the optimum came through. After the table is full, start at the far corner (n,m) and follow pointers back to (0,0). Each diagonal pointer emits a paired column (Bᵢ over Aⱼ); each "up" emits Bᵢ over a gap symbol; each "left" emits a gap over Aⱼ. Reverse the emitted columns and I have the two aligned strings whose score is exactly F(n,m). (If two terms tie, there are co-optimal alignments; I can follow either, or branch to list them all.) This is the "record the origin of the number added to each cell" step, made concrete as a traceback.

One more thing about the cell score s. The crude version is 1 for identical, 0 otherwise — then F is literally counting matched residues. But identity throws away that some mismatches are near-misses. Two amino acids whose codons differ in only one of three bases are a single point mutation apart; ones with no shared codon bases are far apart. So I can grade s: weight a pair by how many codon bases the two residues share (the codon tables of Marshall, Caskey & Nirenberg give this), or more generally read s off an empirical interchange table like Dayhoff's Atlas counts. The recurrence doesn't care what s is — it just needs a number per pair — so all this sophistication lives entirely inside s(·,·) and the DP machinery is untouched. That's a nice separation: scoring policy vs. optimization.

Now let me push on the gap penalty, because the flat-per-residue cost is biologically wrong and I should see how far the same machinery stretches. A real insertion or deletion of, say, five residues is *one* evolutionary event, not five independent ones. A flat penalty 5d charges it as five events and therefore prefers five scattered single gaps over one clean five-long gap — backwards. What I actually want is a gap of length g to cost open + (g−1)·extend: a steep one-time charge to *open* a gap and a gentle charge to *extend* it, so one long gap is cheap relative to many short ones. This is a penalty "as a function of the size of the gap," which the framework already invited me to allow.

But now the cheap O(nm) three-cell recurrence breaks. Its whole validity rested on the gap cost being additive over length, so that I never had to know whether the cell above me was *already* in the middle of a gap. With an affine cost I do need to know: extending an open gap costs `extend`, but starting a fresh gap costs the big `open`. The single number F(i,j) can't remember "am I currently inside a gap in A, a gap in B, or neither," and that memory is exactly what decides open-vs-extend. So I split the state. Keep three tables: M(i,j) for the best alignment of the prefixes that *ends in a match/mismatch column*; Iₓ(i,j) for the best that *ends in a gap in one sequence*; I_y(i,j) for the best that *ends in a gap in the other*. Now I can charge open vs. extend correctly by looking at which table I'm coming from:

  M(i,j)  = s(Bᵢ,Aⱼ) + max( M(i−1,j−1), Iₓ(i−1,j−1), I_y(i−1,j−1) )
  Iₓ(i,j) = max( M(i−1,j) − open,  Iₓ(i−1,j) − extend )
  I_y(i,j) = max( M(i,j−1) − open,  I_y(i,j−1) − extend )

— coming into a gap from M means I'm *opening* (pay open); staying within the same gap table means I'm *extending* (pay the cheaper extend). The best prefix score is max(M,Iₓ,I_y) at (i,j), and the answer is that max at (n,m). Three tables instead of one, but still constant work per cell, so still O(nm) — I keep the efficiency and get biologically sane gaps. (One can also allow a gap in A to switch straight to a gap in B; including that transition is a refinement, but the open/extend split is the load-bearing idea.) The traceback now also remembers *which table* it's in at each step, but it's the same walk back to the origin.

Let me also notice what happens if I flip the objective. Instead of maximizing matched residues, suppose I score a match 0, a mismatch 1, and a gap 1, and *minimize*. Then F(i,j) = min( F(i−1,j−1)+[Bᵢ≠Aⱼ], F(i−1,j)+1, F(i,j−1)+1 ), with F(i,0)=i, F(0,j)=j. That's counting the minimum insertions, deletions and substitutions to turn one string into the other — the edit distance Levenshtein defined. Same grid, same three-neighbour structure, same O(nm) fill; max-similarity and min-distance are two readings of one dynamic program. Reassuring — it tells me the lattice is the right object, not an artifact of how I chose to score.

And a local twist, for when only a *region* of the two proteins is homologous and the rest is unrelated: I don't want leading/trailing junk to drag the score negative. Add a fourth option, 0, to the recurrence — H(i,j) = max(0, diag+s, up−d, left−d) — so any cell can reset to "start fresh here," and let the alignment begin and end wherever it likes. Then the best local match is the largest cell anywhere in the table, and traceback runs from that peak back to the first 0. That's a small mutation of the same machinery; the global version (start corner to start corner, full traceback) is what I'll write first, since the question I started from — are these two whole proteins related — is global.

Let me put the core, the global linear-gap version, into code. I'll keep one score table and one traceback table, fill with the three-term recurrence and the leading-gap boundaries, then walk back from the corner.

```python
import numpy as np

GAP = 1.0  # penalty charged per gapped position; the barrier d

def score_pair(a, b, match=1.0, mismatch=0.0):
    # crude identity scoring; swap in a codon-/Dayhoff-weighted s(a,b) freely —
    # the DP below never looks inside this function
    return match if a == b else mismatch

# diagonal / up / left moves, used by both the fill and the traceback
DIAG, UP, LEFT = (1, 1), (1, 0), (0, 1)

def build_table(seqA, seqB, gap=GAP, score=score_pair):
    n, m = len(seqA), len(seqB)
    F = np.zeros((n + 1, m + 1))          # F[i][j] = best score aligning A[:i], B[:j]
    ptr = [[None] * (m + 1) for _ in range(n + 1)]
    # boundaries: leading gaps cost one penalty per skipped residue
    for i in range(1, n + 1):
        F[i][0] = -i * gap
        ptr[i][0] = UP
    for j in range(1, m + 1):
        F[0][j] = -j * gap
        ptr[0][j] = LEFT
    # the three-term recurrence: each cell from its diagonal, up, left predecessor
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = F[i-1][j-1] + score(seqA[i-1], seqB[j-1])  # match/mismatch column
            up   = F[i-1][j]   - gap                          # gap in B (consume A[i-1])
            left = F[i][j-1]   - gap                          # gap in A (consume B[j-1])
            best = max(diag, up, left)
            F[i][j] = best
            ptr[i][j] = DIAG if best == diag else (UP if best == up else LEFT)
    return F, ptr

def traceback(seqA, seqB, ptr):
    i, j = len(seqA), len(seqB)
    a_out, b_out = [], []
    while i > 0 or j > 0:               # walk from the far corner back to the origin
        di, dj = ptr[i][j]
        if (di, dj) == DIAG:
            a_out.append(seqA[i-1]); b_out.append(seqB[j-1])   # paired column
        elif (di, dj) == UP:
            a_out.append(seqA[i-1]); b_out.append('-')         # gap opposite A[i-1]
        else:
            a_out.append('-');       b_out.append(seqB[j-1])   # gap opposite B[j-1]
        i, j = i - di, j - dj
    return ''.join(reversed(a_out)), ''.join(reversed(b_out))

def align(seqA, seqB, gap=GAP, score=score_pair):
    F, ptr = build_table(seqA, seqB, gap, score)
    a_aln, b_aln = traceback(seqA, seqB, ptr)
    return F[len(seqA)][len(seqB)], a_aln, b_aln   # maximum match + the alignment
```

And the affine-gap version, when I want one long indel to be cheaper than many short ones — three tables, open vs. extend, still one pass:

```python
NEG = float('-inf')

def align_affine(seqA, seqB, gap_open=2.0, gap_extend=0.5, score=score_pair):
    n, m = len(seqA), len(seqB)
    M  = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in match/mismatch
    Ix = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in a gap consuming A
    Iy = [[NEG] * (m + 1) for _ in range(n + 1)]   # ends in a gap consuming B
    M[0][0] = 0.0
    for i in range(1, n + 1):                       # leading gap in A: open then extend
        Ix[i][0] = -gap_open - (i - 1) * gap_extend
    for j in range(1, m + 1):
        Iy[0][j] = -gap_open - (j - 1) * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = score(seqA[i-1], seqB[j-1])
            M[i][j]  = s + max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1])
            Ix[i][j] = max(M[i-1][j] - gap_open,   Ix[i-1][j] - gap_extend)  # open vs extend
            Iy[i][j] = max(M[i][j-1] - gap_open,   Iy[i][j-1] - gap_extend)
    return max(M[n][m], Ix[n][m], Iy[n][m])
```

So the causal chain: I wanted an objective number for how well two ordered sequences match under gaps; pairings-that-respect-order are monotone paths on a grid, and there are exponentially many, so enumeration is out; but the best path restricted to any prefix is optimal for that prefix (optimal substructure), and the last column of a prefix-alignment is one of exactly three shapes, which gives F(i,j) = max(diag+s, up−d, left−d) with leading-gap boundaries — a table I fill in O(nm); remembering which term won lets me trace the actual alignment back from the corner; grading s carries all the biology of substitution scoring without touching the DP; and when a constant gap cost is too crude, splitting the state into match/gap-in-A/gap-in-B tables charges gap-open against gap-extend (affine) while staying O(nm). The same lattice read as a minimization is edit distance; with a max(0,·) reset it becomes local alignment.
