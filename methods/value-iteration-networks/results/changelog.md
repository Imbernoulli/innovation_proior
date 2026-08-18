# Value Iteration Networks changelog

## 2026-08-17 — source-value recheck
`refs/` was empty for this unit; it now holds the author self-accounts listed in `notes/synthesis.md`
(four talk transcripts, all arXiv versions with their LaTeX comments, reviews, supplemental, repo
issue threads, Berkeley 2016 deck). Two changes to `results/reasoning.md`:
- Tied weights: the trace argued the case for tying purely on parameter-counting. It now carries the
  measurement that settles it — untying the `K` recurrent layers and changing nothing else drops the
  held-out success rate from 99.3% to 95.6% at full data, 99.4% to 95.2% at half, and 98.2% to 91.9%
  at a fifth, the gap widening as data shrinks, which is the signature of a parameter-count problem
  rather than a capacity one.
- Generalization beyond the grid: sharpened to the form the author states himself — one value-iteration
  sweep is a linear operation followed by a pointwise max, which is exactly what a convolutional
  network already computes and, crucially, already knows how to backpropagate through, so the
  correspondence is not a fact about grids. Added the structural counterpart the authors wrote in their
  own (commented-out) outline: convolutions won in vision by exploiting sparsity and weight sharing,
  and the analogous structure here is that transition distributions are almost never dense.
- No factual errors found; the 5×5 numeric equivalence check (max abs diff ~7e-8), the channel-max
  distinction, the architecture, the landing and the code are unchanged.
