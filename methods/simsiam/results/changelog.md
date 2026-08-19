# Changelog

## 2026-08-18 — svfix(epistemic)
- The 2026-08-17 svfix(B_selfaccount) pass ("Xinlei Chen's multi-step-alternation
  ablation ... now grounds the (theta,eta) alternating-optimization hypothesis as a
  tested prediction") crossed the line: it wrote the sourced ablation into
  reasoning.md in first-person present tense as the narrator's OWN executed
  experiment ("Run it: 1, 10, 100 SGD steps ... Accuracy moves 68.1->68.7->68.9,
  then drops to 67.0 ... That is the momentum-encoder-shaped signature the
  alternation view predicts") -- a single-turn proposal has no results yet, sourced
  or not.
- Rewrote the passage to keep the hypothesis (solving the theta subproblem more
  faithfully before refreshing eta should behave like a momentum encoder's lagging
  target), the discriminating-experiment DESIGN (sweep 1/10/100 SGD steps on theta
  between eta refreshes, holding optimizer steps/architecture/every other
  hyperparameter fixed), the PREDICTION (a non-monotonic, momentum-encoder-shaped
  accuracy curve: rising as the inner solve gets more faithful, then falling once
  eta goes stale), and the decision rule (that shape backs the reformulation itself;
  a monotonic or flat curve says it doesn't). Removed the claimed observation and
  the 68.1/68.7/68.9/67.0 numbers.
- The landing (stripped shared-encoder Siamese recipe with stop-gradient) does not
  depend on this ablation for its justification and is unchanged. This unit needs
  conversion to a trajectory observation turn to supply the actual sourced result.
