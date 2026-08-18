# Changelog

## 2026-08-18 — svfix(D_candidate)
Fixed a factual error in the decisive step (choice of sign and magnitude for the
leading-coefficient perturbation `ε₁` in `L_Poly-1 = -log(P_t) + ε₁(1-P_t)`).
The trace previously justified the sign via an invented "balanced classification
vs. over-served imbalanced majority" heuristic. This is inconsistent with the
primary source: both benchmarks the paper actually tunes `ε₁` on (ImageNet-21K
pretraining, COCO/Mask R-CNN detection) are explicitly imbalanced, yet they take
opposite-sign `ε₁` (+2 vs. -1). Re-grounded the sign/magnitude reasoning in
`results/reasoning.md`, `results/answer.md`, and `results/train_answer.md` in
the documented criterion instead: the sign is read off each task's own logged
mean-`P_t` training trajectory (chronically under-confident -> `ε₁ > 0`;
chronically saturated/over-confident -> `ε₁ < 0`), with the magnitude set by a
constrained 1-D grid search per task (`ε₁ ∈ {0,...,7}` on a held-out minival
for the classification head; `ε₁ ≥ -1` boundary for the detection head),
landing on `ε₁ = 2` and `ε₁ = -1` respectively. Sources: arXiv primary
(Table 4, Table 5, Figure 5/6 text) and two ICLR 2022 OpenReview author-rebuttal
notes (forum `gSdSJoenupI`). See `notes/sources.md`. Code and the landing
formula (`L = -log(P_t) + ε₁(1-P_t)`) are unchanged — only the sign/magnitude
justification was wrong, not the mechanism.
