# Context: objective global comparison of two protein sequences

## Research question

Given two protein (or nucleotide) sequences, can we decide *objectively and by machine* whether they are related — and quantify *how well* one can be matched against the other when we allow residues to be inserted or deleted? Around 1970 the routine way to compare two amino-acid sequences was to write them out and look. Visual comparison is tedious, and the verdict ("these look homologous") is left to intuition. Two difficulties stack on top of each other. First, a meaningful comparison has to allow **gaps** — stretches where one sequence has residues the other lacks — because real proteins diverge by insertions and deletions, not only point substitutions; but once gaps are allowed, the number of ways to line up two sequences explodes. Second, even with a fixed alignment we need a *number* — a score whose value we can test for significance against chance.

What a solution must achieve: define a single quantity, a "maximum match," that is the best attainable score of lining up the two sequences subject to gaps being penalized; compute it without enumerating the astronomically many gapped alignments; recover the actual alignment that attains it; and do all of this on a computer, deterministically.

## Background

**The combinatorial wall.** A residue from one sequence can be paired with at most one residue of the other, and the pairing must respect order (it cannot cross itself). Lay the two sequences on the two axes of a grid; any gapped alignment is a monotone staircase path from one corner toward the other, where a diagonal step pairs two residues and a horizontal/vertical step skips a residue (a gap). The number of such paths grows exponentially in sequence length — for two sequences of length ~150 (the size of a small globin) it is hopeless to score them one by one. Allowing gaps (as Braunitzer's 1965 comparisons did) "greatly multiplies the number of comparisons that can be made but introduces unnecessary and partial comparisons." The art is to evaluate only the paths that can possibly contribute to the best score.

**Scoring a single paired position.** The smallest unit of comparison is a pair of residues, one from each sequence. The crudest score assigns 1 to an identical pair and 0 to a non-identical one, so a path's score is just the count of matched residues. More refined scores weight a pair by how chemically/genetically close the two residues are: Eck & Dayhoff's 1966 *Atlas of Protein Sequence and Structure* tabulated observed amino-acid interchanges (the lineage that becomes the PAM substitution matrices), and codon tables (Marshall, Caskey & Nirenberg 1967) let one weight a pair by how many of the three codon bases the two amino acids share — non-identical pairs whose codons differ at a single position are "cheaper" mutations than pairs sharing no bases. Fitch (1966) had already built such a 20×20 amino-acid correspondence array for statistical sequence comparison.

**Gaps cost something.** A gap is an evolutionary event (an insertion or deletion) and should be charged for; otherwise free gaps let any two sequences be slid into spurious agreement. The natural device is a penalty subtracted from the score for each gap, possibly depending on the gap's length or direction, acting as a barrier: a gap is worth taking only if the extra matches it makes possible outweigh the barrier. With the penalty set to zero, every possible gap is allowed; with a penalty as large as the theoretical maximum match, gaps are effectively forbidden and the comparison collapses to summing along unbroken diagonals (the simple frame-shift comparison).

**A hammer in the air: dynamic programming.** Bellman's principle of optimality (1950s) is the general method for problems with overlapping subproblems: when the same subproblem recurs, one tabulates its answer instead of recomputing it. The same circle of ideas had surfaced for strings: Levenshtein (1966, *Soviet Physics Doklady* 10:707–710) defined the **edit distance** between two strings — the minimum number of single-character insertions, deletions and substitutions to turn one into the other — and proved it a metric. Edit distance is a *minimization* over strings, in some sense a mirror of the maximization-of-matched-residues problem here; whether the two are close enough to share machinery is not obvious in advance.

**Why it matters biologically.** Primary structure largely determines tertiary structure, so sequence similarity is evidence of common ancestry. A reliable maximum-match score, together with a way to judge it against the score of randomly shuffled sequences, would let one accept or reject the hypothesis that two proteins are related by chance — and, when the scoring is tied to codon differences, even estimate the minimum number of mutations separating them (the evolutionary-distance idea of Margoliash, Needleman & Stewart 1963 and Zuckerkandl & Pauling 1965).

## Baselines

**Direct (gap-free) comparison.** Slide one sequence against the other and count identical residues in register, or sum along the main diagonals of the comparison grid. Core idea: matches in fixed register. Gap: it cannot accommodate insertions/deletions, so a single indel throws the rest of the two sequences out of phase and destroys the apparent similarity — exactly the case where a homology test is most needed.

**Gapped enumeration (Braunitzer-style).** Allow interruptions in either sequence and compare across them. Core idea: admit indels so distant homologies line up. Gap: with no principled penalty and no pruning, it "multiplies the number of comparisons" and admits unnecessary, partial pathways — it is both combinatorially explosive and prone to over-fitting gaps to manufacture a match.

**Statistical sequence comparison (Fitch 1966; Needleman & Blair 1969).** Build a residue-pair correspondence array and apply statistical tests to a comparison. Core idea: replace eyeballing with a computed, testable quantity, and weight residue pairs by codon/chemical relatedness. Gap: these framed the *scoring* and *significance* of a comparison but did not solve the *alignment* problem — how to choose, among exponentially many gapped layouts, the single best one and report it.

## Evaluation settings

The natural test material is pairs of real proteins of known sequence: small, well-characterized chains such as the globins (whale myoglobin, human β-hemoglobin, ~150 residues) for a positive control where homology is expected, and an unrelated pair (e.g. bovine ribonuclease vs. hen egg-white lysozyme) as a negative control. Residue-pair scores are drawn from identity (1/0) or from codon-base correspondence between amino acids; gap penalties are swept across several values; the score of a pair is compared against a null distribution generated by randomly shuffling one sequence's residues (preserving composition) and recomputing the score many times, summarized by a mean and standard deviation. Metrics of interest: the maximum-match score, its distance from the random mean in standard-deviation units, and the number of gaps in the best layout.

## Code framework

The primitives that already exist: two sequences as strings, a per-pair scoring function, and a scalar gap penalty. What is missing is the procedure that turns these into a maximum-match score and reports the layout that attains it.

```python
def score_pair(a, b, match=1.0, mismatch=0.0):
    """Score for aligning residue a against residue b (e.g. 1 if identical else 0,
    or a codon-/substitution-weighted value)."""
    return match if a == b else mismatch

GAP = 1.0  # penalty charged per gapped position; tunable barrier

def align(seqA, seqB, gap=GAP, score=score_pair):
    """Return the maximum-match score of comparing seqA against seqB under the
    order-preserving / gap-penalized rules, together with the layout (with gap
    symbols) that attains it, computed without enumerating gapped alignments."""
    # TODO
    pass
```
