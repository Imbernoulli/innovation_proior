# Changelog

## 2026-08-18 — obs-fix(V): experiments show depth-4 not worth it
- `reasoning.md`, `answer.md`, `train_answer.md`: removed the narrator-run comparison "the
  experiments show depth four is not worth extra cost over depth three" / "depth 4 was
  empirically similar" — at output time this is a proposal, so the narrator cannot have already
  run the depth-3-vs-depth-4 comparison. Kept the mechanism (throttle exponent sharpens with
  `N`, extra factor costs an extra matmul per step) and replaced the reported outcome with the
  discriminating test design (depth 3 vs depth 4, matched optimizer/init-scale/step-budget,
  same ground truth), each side's prediction, and the decision rule (depth 3 ships as default
  unless depth 4 shows a measurably tighter kept/discarded singular-value separation at matched
  budget). No numbers were removed since none were reported beyond "empirically similar"; no
  provenance leaks introduced.
