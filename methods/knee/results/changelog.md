# changelog — knee

## 2026-08-18 (epistemic correction pass)
Prior svfix commit `1dbe0f4ef` ("svfix(D_candidate): knee — ... Wide-minima-density arXiv
2003.03977 control experiment (10,000-epoch low-LR run, 93.9% vs >95%) + 50-repeat/30-epoch
outlier grounds the decisive step") added a strong discriminating-experiment design to the
density-vs-narrow-minima step but wrote it in the wrong voice for a single-turn PROPOSAL:
`reasoning.md` had the narrator *run* both controls in-line — a 10,000-epoch low-LR run
converging with test accuracy landing at 93.9% "well under the >95" a long-hot-phase run
reaches, and fifty repeats of a 30/200-epoch hot phase averaging 94.81% with one spiking to
95.24% above the "long-explore average of 95.1" — i.e. reporting the method's own experimental
outcomes as already observed, which this frame does not allow (the method's own results belong
only in a separate trajectory-observation turn, not the proposal).

Fixed by rewriting the passage to keep: the patience-vs-heat hypothesis pair, the matched-budget
control design (same net, 10,000 epochs at fixed low LR = 0.001, vs. a long-hot-phase reference
run), each hypothesis's prediction (patience predicts the cold run's accuracy climbs to match
the reference given enough wall-clock; heat predicts it falls short regardless of duration
because it's the escape mechanism, not distance traveled, that matters), the second
rarity-specific design (cut the hot phase to 30/200 epochs, repeat fifty times at that matched
budget) with its own predictions (a smooth account predicts tight clustering; a rarity/density
account predicts mostly-mediocre repeats with an occasional high-spiking outlier — a rare-hit
tail), and the decision rule (whichever pattern the two controls actually show — patience,
smooth, or density — is the one trusted; density is taken as the working hypothesis to design
from, with both controls as what could overturn it). Removed the claimed observations and every
result number (93.9, >95, 94.81, 95.24, 95.1). The batch-size-arithmetic consistency argument
(on-page computation from earlier in the same reasoning, not a new observation) was kept.

This unit now needs a trajectory-observation turn to actually run the extended-low-rate control
and the repeated-short-hot-phase spread and supply their outcomes. No changes were needed to
answer.md or train_answer.md — the svfix diff for this method (`svfix-baseline-2026-08-17..HEAD`)
touched only `reasoning.md`.
