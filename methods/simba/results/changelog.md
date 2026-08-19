# changelog — simba

## 2026-08-18 (obs-fix: self-supplied observations)
`reasoning.md` and `train_answer.md` each stated the three-ingredient ablation as an
accomplished result ("the component ablations show that removing any of the three
ingredients hurts performance" / "the ablations confirm that removing any one of the
three hurts"). This is a single-turn PROPOSAL — the method's own ablations have not been
run yet at output time — so the claimed outcome does not belong here.

Fixed by rewriting both passages into hypothesis -> discriminating test design -> decision
rule, with the numbers removed:
- `reasoning.md`: the "useful claim" paragraph now states scaling-should-become-beneficial
  and each-ingredient-should-be-load-bearing as things to test, spells out the controlled
  scaling sweep against the plain-MLP baseline plus the matched-budget leave-one-out
  ablation over {running-stat normalization, pre-LN residual branch, final LayerNorm}, the
  per-hypothesis prediction (upward separation from the plain MLP; return loss at both a
  small and a large scale for every single-ingredient removal), and the decision rule (an
  ingredient that costs nothing at either scale comes out of the design).
- `train_answer.md`: the paragraph on the three ingredients' distinct roles now ends with
  the same leave-one-out ablation as the thing that decides load-bearing-ness, and the
  decision rule (a component removable without cost at both scales is not doing the work
  its role claims), instead of an already-confirmed result.

Kept all mechanism content (what each of the three ingredients does and why) and the
constants table untouched. No changes needed to `answer.md` (no violation found) or
`context.md` (no ablation numbers there to reconcile). This unit now needs a
trajectory-observation turn to supply the actual ablation result.

## 2026-08-18 (obs-fix repair pass: adjacent overclaim missed by the scanner)

The scanner only matches `abl_shows`/`run_num` patterns, so it missed a second, adjacent
sentence in `train_answer.md` that had the same defect as the one already fixed above: "the
stronger, more modest claim is that running the same RL algorithm with this normalized
residual encoder makes scaling the critic beneficial precisely over the range where a plain
MLP degrades" states the critic-scaling outcome in flat present tense, as an accomplished
causal fact, for the exact claim `reasoning.md`'s twin sentence was already hedged for in
the prior pass ("becomes beneficial" -> "should become beneficial").

Fixed by the same should/predict treatment: "makes ... beneficial" -> "should make ...
beneficial", and named what decides the claim (the scaling sweep and leave-one-out ablation
already described earlier in this file) instead of leaving it resting on the simplicity
diagnostic alone. No numbers, mechanism content, or other passages touched.
