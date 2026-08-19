# Changelog

- 2026-08-18 `results/reasoning.md` (move-selection paragraph, "Now the real
  question: how do I move through permutation space?...") — epistemic
  correction. The prior svfix pass (commit `ae988d818`) had the narrator
  claim to fork the tuck operator into the actual greedy loop and report
  outcomes from running it: "for a long stretch this is genuinely my
  best-scoring variant, ahead of plain insertion," a description of three
  named move functions patching each other's gaps discovered by building and
  watching the code run, and "I trust that trade because I watched it, live,
  refuse to converge to a single move." That is a self-supplied observation
  of the method's own (pre-)experimental behavior inside a proposal-voice
  document, which the unit's frame does not permit — at this point in the
  narrative the method has no results yet.
  Fix: removed the claimed build-and-observe narrative. Kept the tuck
  operator's prior-work characterization (accuracy champion on dense graphs,
  DFS over covered/singular/general tucks) and, as a genuine improvement
  worth preserving, the caching-property design argument the svfix pass
  contributed — a move's cost must depend only on which predecessors sit in
  the prefix to be cacheable, tuck's cost instead depends on the induced
  edge structure (covered vs. singular vs. general), so it cannot collapse
  to that shape without the same depth-knob machinery being set aside — as
  pure on-page reasoning rather than an observed fact. The paragraph again
  ends on an open question ("whether there's a move that leaps as far... but
  whose cost is a clean function of the prefix"), answered by the next
  paragraph's best-position proposal exactly as before the svfix pass, so no
  landing is left unjustified and no trajectory-conversion flag is needed.
  `answer.md`/`train_answer.md` were not touched by the flagged svfix commit
  (diff-clean), so nothing to fix there.
