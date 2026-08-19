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

## 2026-08-18 — svfix(epistemic)
The previous svfix pass fixed the *justification* (imbalance alone doesn't
decide the sign) but, in doing so, had the narrator claim to have already run
the discriminating experiment: logging mean-`P_t` trajectories, watching
accuracy/AP/AR move as `ε₁` swept, and landing on `ε₁ = 2` / `ε₁ = -1` as
observed sweep outcomes. This unit is a single-turn proposal — the method's
own experiments have not happened yet inside the frame — so narrating them as
already run and reporting their results (real numbers, sourced from the
primary's own Table 4/5, or not) is out of voice for `reasoning.md` /
`answer.md` / `train_answer.md`; the method's own results belong only in a
trajectory observation turn. Removed the claimed observations (the shape of
each logged curve, accuracy/AP/AR following it, the sweep "landing on" `+2`
and `-1`) from all three files. Kept: the imbalance counter-argument (both
tasks are imbalanced, so that alone can't be the criterion), the diagnostic
DESIGN (log mean `P_t` per task; grid search `ε₁ ∈ {0,...,7}` on a held-out
minival for the classification head; `ε₁ ≥ -1` boundary for the detection
head), each hypothesis's PREDICTION (which task should be chronically low vs.
chronically saturated, and why), and the decision rule (read the sign/size off
whichever way each task's own logged curve leans, confirmed before either
sign is committed — not assumed in advance). The landing method/formula is
unchanged. Because the concrete sign/magnitude values are no longer
established inside this unit, it needs a trajectory-conversion pass to supply
the actual observation turn.
