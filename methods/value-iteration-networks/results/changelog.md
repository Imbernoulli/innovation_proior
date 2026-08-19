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

## 2026-08-18 — epistemic correction
The 2026-08-17 recheck above put the untying ablation's *outcome* in the narrator's mouth: at this
point in the frame the method's own experiments have not run yet, so "the success rate drops from
99.3% to 95.6%..." etc. was an observation the proposal isn't entitled to. Removed the reported
numbers from `results/reasoning.md`; kept the ablation's design (untie the `K` recurrent layers,
change nothing else, compare held-out success across training-set sizes), the prediction (drops
hardest when data is scarce), the discriminating signature (a widening gap says parameter-count
problem, a flat gap says capacity problem), and the existing honesty line ("I won't claim the size
of the effect until I've run it"). The generalization rewrite (linear-op-then-max; sparse dynamics)
was not touched — it states no observed result, so it wasn't in scope.
