# Source-value fix — simsiam (track B_selfaccount)

Self-account already on file: `refs/self_accounts/chen-2021-simsiam-and-beyond-dlct-slides.txt` (Xinlei Chen, "SimSiam and Beyond," DLCT talk slides, 2021). Already catalogued in `notes/source_matrix.md`. This entry records the specific load-bearing numbers used in this svfix pass.

## Quote/data used ("Proof-of-Concept 1: Multi-step alternation")

> "Multi-step alternation: Update θ multiple times (with SGD) before updating η again.
> top-1: 1-step 68.1, 10-step 68.7, 100-step 68.9, 1-epoch 67.0.
> Has a 'momentum encoder' effect that uses predictions from previous weights. Suggest alternating optimization is a valid formulation."
> (`refs/self_accounts/chen-2021-simsiam-and-beyond-dlct-slides.txt`, lines ~237-247.)

This is the authors' own empirical test of the (θ,η) alternating-optimization hypothesis: if solving the θ subproblem more fully before refreshing η is a real alternation and not just an algebraic coincidence, doing so should act like a momentum encoder (a lagging, more faithful target) and should move accuracy in a specific, testable way. The result (68.1→68.7→68.9 as the alternation is solved more faithfully, then a drop to 67.0 once η goes stale over a full epoch) matches the prediction. This was previously unused in `results/reasoning.md`, which already used "Proof-of-Concept 2" (moving-average target, no predictor, 55.0 top-1) but not this one.

## Where used
- `results/reasoning.md`, new paragraph inserted immediately after "...with the stop-gradient explained rather than assumed" and before the predictor paragraph — turns the alternating-optimization reformulation from an algebraic rewrite into a hypothesis with a falsifiable, tested prediction, matching the self-account's own framing of it as a testable "hypothesis" (the slides explicitly label this section "The Role of Stop-Grad — Hypothesis").
